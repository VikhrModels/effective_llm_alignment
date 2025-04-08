import multiprocessing
import os
import random
import uuid
import warnings
from functools import partial

import torch
from accelerate import PartialState
from accelerate.logging import get_logger
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers.integrations import is_deepspeed_zero3_enabled
from trl import ModelConfig, get_peft_config, ORPOConfig, ORPOTrainer

from src.callbacks.generate_examples import GenerateExamplesCallback
from src.configs.additional.orpo_args import ORPOScriptArguments
from src.utils.datasets import load_datasets, prepare_generative_row
from src.utils.logger import setup_logging
from src.utils.model_preparation import setup_model_and_tokenizer
from src.utils.yaml_args_parser import H4ArgumentParser

logger = get_logger(__name__)

LOGGING_TASK_NAME = str(uuid.uuid4())

os.environ['WANDB_RUN_ID'] = str(random.randint(100000, 999999))
os.environ['WANDB_NAME'] = LOGGING_TASK_NAME
os.environ['CLEARML_TASK'] = LOGGING_TASK_NAME


def main():
    parser = H4ArgumentParser((ORPOScriptArguments, ORPOConfig, ModelConfig))
    args, orpo_config, model_config = parser.parse()

    setup_logging(logger, orpo_config)
    set_seed(orpo_config.seed)  # in case of new tokens added without initialize...

    os.environ["WANDB_PROJECT"] = args.project_name
    os.environ['CLEARML_PROJECT'] = args.project_name

    ################
    # Model & Tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        torch_dtype=torch.bfloat16 if orpo_config.bf16 else torch.float16,
        # max_position_embeddings=sft_config.max_seq_length,
        attn_implementation=model_config.attn_implementation
    )

    for n, p in model.named_parameters():
        p.requires_grad = not model_config.use_peft

    peft_config = get_peft_config(model_config)

    if model_config.lora_task_type != "CAUSAL_LM":
        warnings.warn(
            "You are using a `task_type` that is different than `CAUSAL_LM` for PEFT. This will lead to silent bugs"
            " Make sure to pass --lora_task_type CAUSAL_LM when using this script."
        )

    setup_model_and_tokenizer(args, model, tokenizer)

    if PartialState().is_main_process:
        logger.info(f'Tokenizer: {tokenizer}')
        logger.info(f'Model config: {model.config}')

    ################
    # Dataset
    ################
    ds = load_datasets(args.dataset, args.test_size)
    generate_dataset = ds['test']

    def apply_chat_templates(row):
        row["prompt"] = tokenizer.apply_chat_template(row["prompt"], tokenize=False)
        row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
        row["rejected"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
        return row

    with PartialState().main_process_first():
        ds = ds.map(
            apply_chat_templates,
            num_proc=multiprocessing.cpu_count(),
            load_from_cache_file=True,
        )
        generate_dataset = generate_dataset.map(
            partial(prepare_generative_row, tokenizer=tokenizer, max_length=orpo_config.max_prompt_length),
            num_proc=multiprocessing.cpu_count(),
            load_from_cache_file=True
        )

    train_dataset = ds["train"]
    eval_dataset = ds["test"]

    if PartialState().is_main_process:
        logger.info('Example from train dataset:')
        logger.info(train_dataset[0])
        logger.info('Example from test dataset:')
        logger.info(eval_dataset[0])
        logger.info('Example from gen dataset:')
        logger.info(generate_dataset[0])

    generate_callback = GenerateExamplesCallback(
        preprocessed_dataset=generate_dataset,
        tokenizer=tokenizer,
        num_examples=args.num_gen_examples,
        is_deepspeed_zero3=is_deepspeed_zero3_enabled(),
        logger_backend=orpo_config.report_to[0]
    )

    PartialState().wait_for_everyone()

    ################
    # Training
    ################
    trainer = ORPOTrainer(
        model,
        args=orpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[generate_callback] if args.generate_eval_examples else []
    )

    # train and save the model
    trainer.train()

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    trainer.save_model(orpo_config.output_dir)


if __name__ == '__main__':
    main()