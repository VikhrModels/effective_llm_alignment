import multiprocessing
import os
import random
import uuid
import warnings
from dataclasses import dataclass
from typing import List

import torch
from accelerate import PartialState
from accelerate.logging import get_logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed
from trl import ModelConfig, get_peft_config

from src.callbacks.training_parameters_callback import ParameterStatsCallback
from src.configs.classificaion_config import ClassificationConfig
from src.configs.common_script_args import CommonScriptArguments
from src.trainers.classification_trainer import ClassificationTrainer
from src.utils.datasets import load_datasets
from src.utils.logger import setup_logging
from src.utils.model_preparation import setup_model_and_tokenizer, unfreeze_modules_by_patterns
from src.utils.yaml_args_parser import H4ArgumentParser

logger = get_logger(__name__)

LOGGING_TASK_NAME = str(uuid.uuid4())

os.environ['WANDB_RUN_ID'] = str(random.randint(100000, 999999))
os.environ['WANDB_NAME'] = LOGGING_TASK_NAME
os.environ['CLEARML_TASK'] = LOGGING_TASK_NAME

DATASET_PROCESSING_THREADS = multiprocessing.cpu_count() // 2


@dataclass
class CLFScriptArguments(CommonScriptArguments):
    def __post_init__(self):
        self.project_name = "classification" if self.project_name == "default-project" else self.project_name


def get_label_list(raw_dataset, split="train") -> List[str]:
    """Get the list of labels from a multi-label dataset"""

    if isinstance(raw_dataset[split]["label"][0], list):
        label_list = [label for sample in raw_dataset[split]["label"] for label in sample]
        label_list = list(set(label_list))
    else:
        label_list = raw_dataset[split].unique("label")
    # we will treat the label list as a list of string instead of int, consistent with model.config.label2id
    label_list = [str(label) for label in label_list]
    return label_list


def main():
    parser = H4ArgumentParser((CLFScriptArguments, ClassificationConfig, ModelConfig))
    args, classification_config, model_config = parser.parse()

    setup_logging(logger, classification_config)
    set_seed(classification_config.seed)  # in case of new tokens added without initialize...

    os.environ["WANDB_PROJECT"] = args.project_name
    os.environ['CLEARML_PROJECT'] = args.project_name

    ################
    # Model & Tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name_or_path,
        torch_dtype=torch.bfloat16 if classification_config.bf16 else torch.float16,
        attn_implementation=model_config.attn_implementation,
        num_labels=classification_config.num_labels
    )

    setup_model_and_tokenizer(args, model, tokenizer, classification_config.max_length)

    if model_config.use_peft:
        for n, p in model.named_parameters():
            p.requires_grad = False
        if model_config.lora_task_type != "SEQ_CLS":
            warnings.warn(
                "You are using a `task_type` that is different than `SEQ_CLS` for PEFT. This will lead to silent bugs"
                " Make sure to pass --lora_task_type SEQ_CLS when using this script."
            )
        if args.unfreeze_layers_patterns:
            warnings.warn(
                "You can't use non-empty unfreeze_layers_patterns and peft together at this time, only peft config will be used"
            )
        peft_config = get_peft_config(model_config)
    else:
        if args.unfreeze_layers_patterns:
            unfreeze_modules_by_patterns(model, args.unfreeze_layers_patterns)
        peft_config = None

    if PartialState().is_main_process:
        logger.info(f'Tokenizer: {tokenizer}')
        logger.info(f'Model config: {model.config}')
        logger.info(f'Model: {model}')

    ################
    # Dataset
    ################
    ds = load_datasets(args.dataset, args.test_size, args.dataset_ratio)

    is_multi_label = False
    if ds["train"].features["label"].dtype == "list":  # multi-label classification
        is_multi_label = True
        logger.info("Label type is list, doing multi-label classification")

    elif is_multi_label:
        model.config.problem_type = "multi_label_classification"
        logger.info("setting problem type to multi label classification")
    else:
        model.config.problem_type = "single_label_classification"
        logger.info("setting problem type to single label classification")

    # Trying to find the number of labels in a multi-label classification task
    # We have to deal with common cases that labels appear in the training set but not in the validation/test set.
    # So we build the label list from the union of labels in train/val/test.
    label_list = get_label_list(ds, split="train")
    for split in ["validation", "test"]:
        if split in ds:
            val_or_test_labels = get_label_list(ds, split=split)
            diff = set(val_or_test_labels).difference(set(label_list))
            if len(diff) > 0:
                # add the labels that appear in val/test but not in train, throw a warning
                logger.warning(
                    f"Labels {diff} in {split} set but not in training set, adding them to the label list"
                )
                label_list += list(diff)
    # if label is -1, we throw a warning and remove it from the label list
    for label in label_list:
        if label == -1:
            logger.warning("Label -1 found in label list, removing it.")
            label_list.remove(label)

    label_list.sort()
    num_labels = len(label_list)
    if num_labels <= 1:
        raise ValueError("You need more than one label to do classification.")

    label_to_id = {v: i for i, v in enumerate(label_list)}
    # update config with label infos
    if model.config.label2id != label_to_id:
        logger.warning(
            "The label2id key in the model config.json is not equal to the label2id key of this "
            "run. You can ignore this if you are doing finetuning."
        )
    model.config.label2id = label_to_id
    model.config.id2label = {id: label for label, id in label_to_id.items()}

    def multi_labels_to_ids(labels: List[str]) -> List[float]:
        ids = [0.0] * len(label_to_id)  # BCELoss requires float as target type
        for label in labels:
            ids[label_to_id[label]] = 1.0
        return ids

    def preprocess_function(example):
        text = tokenizer.apply_chat_template(example["prompt"], tokenize=False, add_generation_prompt=False)
        tokenized = tokenizer(text=text, truncation=True, max_length=classification_config.max_length)

        if label_to_id is not None and "label" in example:
            if is_multi_label:
                tokenized["label"] = multi_labels_to_ids(example["label"])
            else:
                tokenized["label"] = label_to_id[str(example["label"])] if example["label"] != -1 else -1

        return tokenized

    # Preprocess the dataset and filter out examples that are longer than args.max_length
    with PartialState().local_main_process_first():
        ds = ds.map(
            preprocess_function,
            batched=False,
            num_proc=DATASET_PROCESSING_THREADS,
            keep_in_memory=True,
            load_from_cache_file=True
        )
    train_dataset = ds["train"]
    eval_dataset = ds["test"]

    if PartialState().is_main_process:
        logger.info('Example from train dataset:')
        logger.info(train_dataset[0])
        logger.info('Example from test dataset:')
        logger.info(eval_dataset[0])

    PartialState().wait_for_everyone()

    ################
    # Training
    ################
    trainer = ClassificationTrainer(
        model=model,
        tokenizer=tokenizer,
        args=classification_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        callbacks=[ParameterStatsCallback],
        is_binary=len(label_to_id) == 2
    )

    # train and save the model
    trainer.train()

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    trainer.save_model(classification_config.output_dir)


if __name__ == '__main__':
    main()
