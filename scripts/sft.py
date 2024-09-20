import multiprocessing
import os
import random
import uuid
import warnings
from dataclasses import dataclass, field

import torch
from accelerate import PartialState
from accelerate.logging import get_logger
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers.integrations import is_deepspeed_zero3_enabled
from trl import SFTTrainer, SFTConfig, ModelConfig, get_peft_config

from src.callbacks.generate_examples import GenerateExamplesCallback
from src.collators.completions_only import DataCollatorForCompletionOnlyLM
from src.configs.common_script_args import CommonScriptArguments
from src.utils.datasets import load_datasets
from src.utils.logger import setup_logging
from src.utils.model_preparation import setup_model_and_tokenizer
from src.utils.yaml_args_parser import H4ArgumentParser

logger = get_logger(__name__)

LOGGING_TASK_NAME = str(uuid.uuid4())

os.environ['WANDB_RUN_ID'] = str(random.randint(100000, 999999))
os.environ['WANDB_NAME'] = LOGGING_TASK_NAME
os.environ['CLEARML_TASK'] = LOGGING_TASK_NAME


@dataclass
class SFTScriptArguments(CommonScriptArguments):
    conversation_field: str | None = field(
        default="prompt",
        metadata={"help": "Field in dataset with conversations (in list of dicts format)"}
    )
    system_prompt: str | None = field(
        default=None,
        metadata={"help": "Will use system prompt if there is no one in dialogue, set to None to disable"}
    )
    train_only_on_completions: bool | None = field(
        default=True,
        metadata={"help": "Do train only on completions or not"}
    )
    generate_eval_examples: bool | None = field(
        default=True,
        metadata={"help": "Do generate examples on eval"}
    )
    assistant_message_template: str | None = field(
        default="<|start_header_id|>assistant<|end_header_id|>\n\n",
        metadata={"help": "Assistant message template for the training only on completions"}
    )
    num_gen_examples: int | None = field(
        default=50,
        metadata={"help": "Number of examples to generate on eval phase"}
    )
    model_support_system_role: bool | None = field(
        default=True,
        metadata={"help": "Flag that indicates if model have support for system prompt. If not, will use user for setting system prompt"}
    )

    def __post_init__(self):
        self.project_name = "sft-tuning" if self.project_name == "default-project" else self.project_name


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
        from liger_kernel.transformers import apply_liger_kernel_to_llama, apply_liger_kernel_to_mistral
        apply_liger_kernel_to_llama(
            rope=False,
            swiglu=True,
            cross_entropy=False,
            fused_linear_cross_entropy=True,
            rms_norm=False
        )
        apply_liger_kernel_to_mistral(
            rope=False,
            swiglu=True,
            cross_entropy=False,
            fused_linear_cross_entropy=True,
            rms_norm=False
        )

    for n, p in model.named_parameters():
        p.requires_grad = not model_config.use_peft

    # TODO: Remake get_peft_config() to allow PromptLearning, Vera, etc...
    peft_config = get_peft_config(model_config)

    if model_config.lora_task_type != "CAUSAL_LM":
        warnings.warn(
            "You are using a `task_type` that is different than `CAUSAL_LM` for PEFT. This will lead to silent bugs"
            " Make sure to pass --lora_task_type CAUSAL_LM when using this script."
        )

    setup_model_and_tokenizer(args, model, tokenizer, sft_config.max_seq_length)

    if PartialState().is_main_process:
        print(f'Tokenizer: {tokenizer}')
        print(f'Model config: {model.config}')

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
            num_proc=multiprocessing.cpu_count(),
            load_from_cache_file=True,
            remove_columns=extra_columns
        )
        generate_dataset = generate_dataset.map(
            lambda row: process_row(row, add_gen_prompt=True),
            num_proc=multiprocessing.cpu_count(),
            load_from_cache_file=True,
            remove_columns=extra_columns
        )

    train_dataset = ds["train"]
    eval_dataset = ds["test"]

    if PartialState().is_main_process:
        print('Example from train dataset:')
        print(train_dataset[0])
        print('Example from test dataset:')
        print(eval_dataset[0])
        print('Example from gen dataset:')
        print(generate_dataset[0])

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
    trainer.save_model(sft_config.output_dir)


if __name__ == '__main__':
    main()
