import multiprocessing
import os
import random
import sys
import uuid

import torch
from accelerate import PartialState
from accelerate.logging import get_logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed, HfArgumentParser
from trl import RewardConfig, ModelConfig

from src.callbacks.training_parameters_callback import ParameterStatsCallback
from src.configs.additional.reward_args import RMScriptArguments
from src.configs.prompts_optimization_comfig import PromptsOptimizationConfig
from src.trainers.prompts_optimization.prompts_reward_trainer import PromptsRewardTrainer
from src.utils.datasets import load_datasets
from src.utils.logger import setup_logging
from src.utils.model_preparation import setup_model_and_tokenizer


logger = get_logger(__name__)

LOGGING_TASK_NAME = str(uuid.uuid4())

os.environ['WANDB_RUN_ID'] = str(random.randint(100000, 999999))
os.environ['WANDB_NAME'] = LOGGING_TASK_NAME
os.environ['CLEARML_TASK'] = LOGGING_TASK_NAME

DATASET_PROCESSING_THREADS = min(multiprocessing.cpu_count() // 2, 16)


def main():
    parser = HfArgumentParser((RMScriptArguments, RewardConfig, ModelConfig, PromptsOptimizationConfig))
    args, reward_config, model_config, prompts_config = parser.parse_yaml_file(os.path.abspath(sys.argv[1]))

    setup_logging(logger, reward_config)
    set_seed(reward_config.seed)  # in case of new tokens added without initialize...

    os.environ["WANDB_PROJECT"] = args.project_name
    os.environ['CLEARML_PROJECT'] = args.project_name

    ################
    # Model & Tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name_or_path,
        torch_dtype=torch.bfloat16 if reward_config.bf16 else torch.float16,
        attn_implementation=model_config.attn_implementation,
        num_labels=1
    )

    setup_model_and_tokenizer(args, model, tokenizer, reward_config.max_length)

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
    with PartialState().local_main_process_first():
        ds = ds.map(
            preprocess_function,
            batched=True,
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
    trainer = PromptsRewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=reward_config,
        prompt_args=prompts_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=None,
        callbacks=[ParameterStatsCallback]
    )

    # train and save the model
    trainer.train()

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    trainer.save_model(reward_config.output_dir)


if __name__ == '__main__':
    assert len(sys.argv) >= 2 and sys.argv[1].endswith(".yaml"), "You must provide .yaml file with training config as argument"
    main()
