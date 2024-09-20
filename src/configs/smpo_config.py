from dataclasses import dataclass
from typing import Dict, Literal, Optional
from transformers import TrainingArguments


@dataclass
class SimpleMarginPOConfig(TrainingArguments):
    r"""
    SimpleMarginPOConfig collects all training arguments related to the [`MarginPOTrainer`] class.

    Using [`HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        max_length (`int`, defaults to `None`):
            The maximum length of the sequences in the batch. This argument is required if you want to use the default data collator.
        max_prompt_length (`int`, defaults to `None`):
            The maximum length of the prompt. This argument is required if you want to use the default data collator.
        max_target_length (`int`, defaults to `None`):
            The maximum length of the target. This argument is required if you want to use the default data collator and your model is an encoder-decoder.
        beta (`float`, defaults to 0.8):
            The beta factor in SimpleMarginPO loss.
        margin_min (`float`, defaults to 0.4):
            The minimal target reward margin in SimpleMarginPO loss. Can be zero for sigmoid and hinge losses.
        margin_delta (`float`, defaults to 0.2):
            The delta of minmal and maximal target reward margin in SimpleMarginPO loss. Only applicable to `smooth_double_bound` loss.
        chosen_sft_ratio (`float`, defaults to 0.75):
            SFT loss balance weight between chosen and rejected, used in the SimpleMarginPO loss (1.0 will use maximum of chosen loss and zero of rejected loss).
        loss_type (`str`, defaults to `smooth_lower_bound`):
            The type of loss to use. This argument is required if you want to use the default data collator.
        label_pad_token_id (`int`, defaults to `-100`):
            The label pad token id. This argument is required if you want to use the default data collator.
        padding_value (`int`, defaults to `None`):
            The padding value if it is different to the tokenizer's pad_token_id.
        truncation_mode (`str`, defaults to `keep_end`):
            The truncation mode to use, either `keep_end` or `keep_start`. This argument is required if you want to use the default data collator.
        generate_during_eval (`bool`, defaults to `False`):
            Whether to sample and log generations during evaluation step.
        is_encoder_decoder (`Optional[bool]`, `optional`, defaults to `None`):
            If no model is provided, we need to know if the model_init returns an encoder-decoder.
        disable_dropout (`bool`, defaults to `True`):
            Whether or not to disable dropouts in `model`.
        model_init_kwargs (`Optional[Dict]`, *optional*):
            Dict of Optional kwargs to pass when instantiating the model from a string
        dataset_num_proc (`Optional[int]`, *optional*):
            The number of workers to use to tokenize the data. Defaults to None.
    """
    
    max_length: Optional[int] = None
    max_prompt_length: Optional[int] = None
    max_completion_length: Optional[int] = None
    max_target_length: Optional[int] = None

    beta: float = 0.8
    margin_min: float = 0.4
    margin_delta: float = 0.2
    chosen_sft_ratio: float = 0.75
    
    loss_type: Literal['sigmoid', 'hinge', 'ipo', 'smooth_lower_bound', 'smooth_double_bound'] = "smooth_lower_bound"
    disable_dropout: bool = True

    label_pad_token_id: int = -100
    padding_value: int = None
    truncation_mode: str = "keep_end"
    generate_during_eval: bool = False
    is_encoder_decoder: Optional[bool] = None

    model_init_kwargs: Optional[Dict] = None

    dataset_num_proc: Optional[int] = None

