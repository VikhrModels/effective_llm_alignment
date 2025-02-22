import os
import random
import uuid
import warnings
from dataclasses import dataclass

import torch
from accelerate import PartialState
from accelerate.logging import get_logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardTrainer, RewardConfig, ModelConfig, get_peft_config

from src.utils.logger import setup_logging
from src.configs.common_script_args import CommonScriptArguments
from src.callbacks.training_parameters_callback import ParameterStatsCallback
from src.utils.datasets import load_datasets
from src.utils.model_preparation import setup_model_and_tokenizer, unfreeze_modules_by_patterns
from src.utils.yaml_args_parser import H4ArgumentParser

logger = get_logger(__name__)

LOGGING_TASK_NAME = str(uuid.uuid4())

os.environ['WANDB_RUN_ID'] = str(random.randint(100000, 999999))
os.environ['WANDB_NAME'] = LOGGING_TASK_NAME
os.environ['CLEARML_TASK'] = LOGGING_TASK_NAME


@dataclass
class RMScriptArguments(CommonScriptArguments):
    def __post_init__(self):
        self.project_name = "reward-modeling" if self.project_name == "default-project" else self.project_name


def main():
    parser = H4ArgumentParser((RMScriptArguments, RewardConfig, ModelConfig))
    args, reward_config, model_config = parser.parse()

    setup_logging(logger, reward_config)

    os.environ["WANDB_PROJECT"] = args.project_name
    os.environ['CLEARML_PROJECT'] = args.project_name

    ################
    # Model & Tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name_or_path,
        # device_map="auto",
        torch_dtype=torch.bfloat16 if reward_config.bf16 else torch.float16,
        # max_position_embeddings=sft_config.max_seq_length,
        attn_implementation=model_config.attn_implementation,
        num_labels=1
    )

    setup_model_and_tokenizer(args, model, tokenizer, reward_config.max_length)

    if model_config.use_peft:
        for n, p in model.named_parameters():
            p.requires_grad = False
        if model_config.lora_task_type != "SEQ_CLS":
            warnings.warn(
                "You are using a `task_type` that is different than `SEQ_CLS` for PEFT. This will lead to silent bugs"
                " Make sure to pass --lora_task_type SEQ_CLS when using this script."
            )
        peft_config = get_peft_config(model_config)
    else:
        unfreeze_modules_by_patterns(model, ['score', '*.layers.*'])
        peft_config = None

    if PartialState().is_main_process:
        logger.info(f'Tokenizer: {tokenizer}')
        logger.info(f'Model config: {model.config}')
        logger.info(f'Model: {model}')

    ################
    # Dataset
    ################
    ds = load_datasets(args.dataset, args.test_size, args.dataset_ratio)

    def preprocess_function(examples):
        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }
        for prompt, chosen, rejected in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
            chosen = tokenizer.apply_chat_template(prompt + chosen, tokenize=False, add_generation_prompt=False)
            rejected = tokenizer.apply_chat_template(prompt + rejected, tokenize=False, add_generation_prompt=False)

            tokenized_chosen = tokenizer(text=chosen, truncation=True, max_length=reward_config.max_length)
            tokenized_rejected = tokenizer(text=rejected, truncation=True, max_length=reward_config.max_length)

            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

        return new_examples

    # Preprocess the dataset and filter out examples that are longer than args.max_length
    ds = ds.map(
        preprocess_function,
        batched=True,
        num_proc=8,
        keep_in_memory=True,
        load_from_cache_file=False
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
    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=reward_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        callbacks=[ParameterStatsCallback]
    )

    # train and save the model
    trainer.train()
    trainer.save_model(reward_config.output_dir)


if __name__ == '__main__':
    main()
