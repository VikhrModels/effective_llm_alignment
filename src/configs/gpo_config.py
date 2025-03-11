from dataclasses import dataclass
from typing import Dict, Literal, Optional
from transformers import TrainingArguments


@dataclass
class GroupedPOConfig(TrainingArguments):
    r"""
    GroupedPOConfig collects all training arguments related to the [`GroupedPOTrainer`] class.

    Using [`HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        max_prompt_length (`int`, defaults to `None`):
            The maximum length of the prompt. This argument is required if you want to use the default data collator.
        lower_clip_percentile (`Optional[float]`, defaults to 0.02):
            Lower percentile of token log probs value allowed for PO loss calculation for rejected completions. Works like winsorizing. Recommended range [0.01, 0.05]
        min_log_prob (`Optional[float]`, defaults to -2.3):
            Lowest possible token log prob value allowed in rejected completions. Will clip all log probs, works after percentile winsorizing.
        upper_clip_percentile (`Optional[float]`, defaults to `None`):
            Upper percentile of token log probs value allowed for PO loss calculation for chosen completions. Works like winsorizing. Recommended range [0.95, 0.99]
        label_pad_token_id (`int`, defaults to `-100`):
            The label pad token id. This argument is required if you want to use the default data collator.
        padding_value (`int`, defaults to `None`):
            The padding value if it is different to the tokenizer's pad_token_id.
        disable_dropout (`bool`, defaults to `True`):
            Whether or not to disable dropouts in `model`.
        model_init_kwargs (`Optional[Dict]`, *optional*):
            Dict of Optional kwargs to pass when instantiating the model from a string
        dataset_num_proc (`Optional[int]`, *optional*):
            The number of workers to use to tokenize the data. Defaults to None.
    """

    max_prompt_length: Optional[int] = 512
    max_completion_length: Optional[int] = 1024

    kl_beta: float = 0.0
    # lower_clip_percentile: Optional[float] = 0.02
    # upper_clip_percentile: Optional[float] = None
    # min_log_prob: Optional[float] = -2.3

    disable_dropout: bool = True
    label_pad_token_id: int = -100
    padding_value: int = None

    model_init_kwargs: Optional[Dict] = None

    dataset_num_proc: Optional[int] = None
