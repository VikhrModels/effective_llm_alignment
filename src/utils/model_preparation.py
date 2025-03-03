from copy import deepcopy

try:
    import deepspeed
except Exception as e:
    print(e)

import torch
from accelerate import Accelerator
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.configs.arguments.common_script_args import CommonScriptArguments


def setup_model_and_tokenizer(
    args: CommonScriptArguments,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    max_seq_len: int = None,
):
    if max_seq_len is not None:
        tokenizer.model_max_length = max_seq_len
    if tokenizer.eos_token != args.eos_token:
        tokenizer.eos_token = args.eos_token
        model.config.eos_token_id = tokenizer.eos_token_id
        if model.generation_config:
            model.generation_config.eos_token_id = tokenizer.eos_token_id
    if (
        tokenizer.bos_token is None or args.bos_token is not None
    ) and tokenizer.bos_token != args.bos_token:
        tokenizer.bos_token = args.bos_token
        model.config.bos_token_id = tokenizer.bos_token_id
        if model.generation_config:
            model.generation_config.bos_token_id = tokenizer.bos_token_id
    if tokenizer.pad_token != args.pad_token:
        tokenizer.pad_token = args.pad_token
        model.config.pad_token_id = tokenizer.pad_token_id
        if model.generation_config:
            model.generation_config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.chat_template is None or (
        args.chat_template is not None and args.force_chat_template
    ):
        tokenizer.chat_template = args.chat_template
    if args.added_special_tokens is not None:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": args.added_special_tokens}
        )
        model.resize_token_embeddings(len(tokenizer))


def unfreeze_modules_by_patterns(model, patterns):
    """
    Замораживает все параметры модели, затем размораживает те модули,
    полное имя которых соответствует хотя бы одному паттерну из списка.

    Аргументы:
      model: torch.nn.Module – модель (например, экземпляр Qwen2ForSequenceClassification).
      patterns: список строк – шаблоны для имен модулей (поддерживаются подстановочные знаки * и ?).

    Пример паттернов:
      ["*.mlp.up_proj", "score", "model.layers.0.self_attn.*"]
    """
    import fnmatch

    # Сначала замораживаем все параметры
    for param in model.parameters():
        param.requires_grad = False

    # Проходим по всем модулям модели. model.named_modules() возвращает пары (имя_модуля, модуль).
    for module_name, module in model.named_modules():
        # Для каждого модуля проверяем, удовлетворяет ли его имя какому-нибудь паттерну
        for pattern in patterns:
            if fnmatch.fnmatch(module_name, pattern):
                # Если совпадение найдено – размораживаем все параметры этого модуля
                for param in module.parameters():
                    param.requires_grad = True
                # Если наш модуль уже разморожен – переходим к следующему
                break


def prepare_ref_model_for_deepspeed(
    model: PreTrainedModel | nn.Module, accelerator: Accelerator
) -> PreTrainedModel | nn.Module:
    deepspeed_plugin = accelerator.state.deepspeed_plugin
    config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)
    if model is not None:
        if hasattr(model, "config"):
            hidden_size: int | None = (  # type: ignore
                max(model.config.hidden_sizes)  # type: ignore
                if getattr(model.config, "hidden_sizes", None)  # type: ignore
                else getattr(model.config, "hidden_size", None)  # type: ignore
            )

            if (
                hidden_size is not None
                and config_kwargs["zero_optimization"]["stage"] == 3
            ):
                config_kwargs.update(
                    {
                        "zero_optimization.reduce_bucket_size": hidden_size
                        * hidden_size,
                        "zero_optimization.stage3_param_persistence_threshold": 10
                        * hidden_size,
                        "zero_optimization.stage3_prefetch_bucket_size": 0.9
                        * hidden_size
                        * hidden_size,
                    }
                )

    if config_kwargs["zero_optimization"]["stage"] != 3:
        config_kwargs["zero_optimization"]["stage"] = 0
    # if not "offload_optimizer" in config_kwargs['zero_optimization']:
    #     config_kwargs['zero_optimization']['offload_optimizer'] = {
    #         "device": "cpu",
    #         "pin_memory": True
    #     }
    if "offload_param" in config_kwargs["zero_optimization"]:
        del config_kwargs["zero_optimization"]["offload_param"]

    config_kwargs["optimizer"] = {"type": None}

    model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
    model.eval()
    return model


def peft_module_casting_to_bf16(model):
    from peft.tuners.tuners_utils import BaseTunerLayer

    for name, module in model.named_modules():
        if isinstance(module, BaseTunerLayer):
            module = module.to(torch.bfloat16)
        elif isinstance(module, torch.nn.LayerNorm) or "norm" in name:
            module = module.to(torch.float32)
        elif any(x in name for x in ["lm_head", "embed_tokens", "wte", "wpe"]):
            if hasattr(module, "weight"):
                if module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
