import multiprocessing
import os
import random
import uuid
import warnings

import torch
from accelerate import PartialState
from accelerate.logging import get_logger
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers.integrations import is_deepspeed_zero3_enabled
from trl import SFTTrainer, SFTConfig, ModelConfig, get_peft_config

from src.callbacks.generate_examples import GenerateExamplesCallback
from src.collators.completions_only import DataCollatorForCompletionOnlyLM
from src.configs.additional.sft_args import SFTScriptArguments
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


def main():
    parser = H4ArgumentParser((SFTScriptArguments, SFTConfig, ModelConfig))
    args, sft_config, model_config = parser.parse()

    setup_logging(logger, sft_config)
    set_seed(sft_config.seed)  # in case of new tokens added without initialize...

    os.environ["WANDB_PROJECT"] = args.project_name
    os.environ['CLEARML_PROJECT'] = args.project_name

    ################
    # Model & Tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        torch_dtype=torch.bfloat16 if sft_config.bf16 else torch.float16,
        # max_position_embeddings=sft_config.max_seq_length,
        attn_implementation=model_config.attn_implementation
    )
    if sft_config.use_liger:
        from liger_kernel.transformers import apply_liger_kernel_to_llama, apply_liger_kernel_to_mistral, apply_liger_kernel_to_qwen2
        apply_liger_kernel_to_llama(
            rope=False,
            swiglu=True,
            cross_entropy=False,
            fused_linear_cross_entropy=True,
            rms_norm=True
        )
        apply_liger_kernel_to_mistral(
            rope=False,
            swiglu=True,
            cross_entropy=False,
            fused_linear_cross_entropy=True,
            rms_norm=True
        )
        apply_liger_kernel_to_qwen2(
            rope=False,
            swiglu=True,
            cross_entropy=False,
            fused_linear_cross_entropy=True,
            rms_norm=True
        )

    setup_model_and_tokenizer(args, model, tokenizer, sft_config.max_seq_length)

    if model_config.use_peft:
        for n, p in model.named_parameters():
            p.requires_grad = False
        if model_config.lora_task_type != "CAUSAL_LM":
            warnings.warn(
                "You are using a `task_type` that is different than `CAUSAL_LM` for PEFT. This will lead to silent bugs"
                " Make sure to pass --lora_task_type CAUSAL_LM when using this script."
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
    def process_row(row, add_gen_prompt=False):
        system_message = [{'role': 'system', 'content': args.system_prompt}] if args.system_prompt else []
        history = row[args.conversation_field] if not add_gen_prompt else row[args.conversation_field][:-1]
        if not args.model_support_system_role and history[0]["role"] == "system":
            if len(history) > 1 and history[1]["role"] == "user":
                # add sys prompt to first user message
                history[1]["content"] = history[0]["content"] + "\n" + history[1]["content"]
                history = history[1:]
            else:
                history[0]["role"] = "user"
        
        constructed_prompt = tokenizer.apply_chat_template(
            system_message + history,
            tokenize=False,
            add_generation_prompt=add_gen_prompt
        )
        if tokenizer.bos_token is not None:
            if constructed_prompt.startswith(tokenizer.bos_token):  # Remove extra bos token
                constructed_prompt = constructed_prompt[len(tokenizer.bos_token):]
        return tokenizer(constructed_prompt, truncation=True, padding=True, max_length=sft_config.max_seq_length)

    ds = load_datasets(args.dataset, args.test_size, args.dataset_ratio)
    generate_dataset = ds['test']

    signature_columns = ["input_ids", "labels", "attention_mask"]
    extra_columns = list(set(ds['train'].column_names) - set(signature_columns))

    with PartialState().local_main_process_first():
        ds = ds.map(
            process_row,
            num_proc=DATASET_PROCESSING_THREADS,
            keep_in_memory=True,
            load_from_cache_file=True,
            remove_columns=extra_columns
        )
        generate_dataset = generate_dataset.map(
            lambda row: process_row(row, add_gen_prompt=True),
            num_proc=DATASET_PROCESSING_THREADS,
            keep_in_memory=True,
            load_from_cache_file=True,
            remove_columns=extra_columns
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

    collator = DataCollatorForCompletionOnlyLM(
        response_prompt_template=args.assistant_message_template,
        tokenizer=tokenizer
    ) if args.train_only_on_completions else None

    generate_callback = GenerateExamplesCallback(
        preprocessed_dataset=generate_dataset,
        tokenizer=tokenizer,
        num_examples=args.num_gen_examples,
        is_deepspeed_zero3=is_deepspeed_zero3_enabled(),
        logger_backend=sft_config.report_to[0]
    )

    PartialState().wait_for_everyone()

    sft_config.dataset_kwargs = {
        "skip_prepare_dataset": True
    }

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        data_collator=collator,
        callbacks=[generate_callback] if args.generate_eval_examples else []
    )

    # train and save the model
    trainer.train()

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    trainer.save_model(sft_config.output_dir)


if __name__ == '__main__':
    main()
