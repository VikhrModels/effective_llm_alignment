import re

from accelerate import PartialState
from accelerate.utils import gather_object
from transformers import (
    TrainerCallback,
)
from transformers.utils import logging

logger = logging.get_logger(__name__)


def count_model_parameters(model):
    """
    Returns a tuple (total_params, trainable_params),
    where total_params is the total number of parameters,
    and trainable_params is the number of parameters with requires_grad=True.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def normalize_module_name(module_name):
    """
    Converts the module name into a normalized (grouping) form,
    replacing numeric indices with the "X" character. Example:
      "model.layers.0.self_attn" -> "model.layers.X.self_attn"
    """
    if not module_name:
        return "<root>"
    normalized = re.sub(r"\b\d+\b", "X", module_name)
    return normalized


def compute_module_trainable_stats(model):
    stats = {}
    for param_name, param in model.named_parameters():
        total = param.numel()
        trainable = param.numel() if param.requires_grad else 0
        norm_name = normalize_module_name(param_name)
        if norm_name in stats:
            prev_total, prev_trainable = stats[norm_name]
            stats[norm_name] = (prev_total + total, prev_trainable + trainable)
        else:
            stats[norm_name] = (total, trainable)
    return stats


class ParameterStatsCallback(TrainerCallback):
    """
    Callback for Trainer that logs the following before training begins:
      - Total number of parameters
      - Number of trainable parameters and the percentage of trainable parameters
      - List of "deduplicated" modules with the percentage of trainable parameters per group
    """

    def on_train_begin(self, args, state, control, **kwargs):
        model = kwargs.get("model", None)
        if model is None:
            trainer = kwargs.get("trainer", None)
            if trainer is not None:
                model = trainer.model

        total_params, trainable_params = count_model_parameters(model)
        percent = 100 * trainable_params / total_params if total_params > 0 else 0

        print(
            f"\n===== Model view inside Trainer (process: {PartialState().process_index}) ====="
        )
        print(model)

        if PartialState().is_main_process:
            print(f"Total number of parameters      : {total_params:,}")
            print(f"Number of trainable parameters  : {trainable_params:,}")
            print(f"Percentage of trainable parameters: {percent:.2f}%\n")

        print(f"\n===== Model parameter statistics (process: {PartialState().process_index}) =====")

        module_stats = compute_module_trainable_stats(model)
        module_stats_percent = []
        for mod_name, (mod_total, mod_trainable) in module_stats.items():
            mod_percent = 100 * mod_trainable / mod_total if mod_total > 0 else 0
            module_stats_percent.append(
                (mod_name, mod_total, mod_trainable, mod_percent)
            )

        module_stats_percent.sort(key=lambda x: x[3], reverse=True)

        print(
            f"List of module groups with normalized names:"
        )
        for mod_name, mod_total, mod_trainable, mod_percent in module_stats_percent:
            print(
                f"  {mod_name:30s} - trainable: {mod_trainable:,} / {mod_total:,} ({mod_percent:.2f}%)"
            )
        print("========================================\n")
