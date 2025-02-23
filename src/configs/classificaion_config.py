from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments

@dataclass
class ClassificationConfig(TrainingArguments):
    r"""
    Configuration class for the [`ClassificationTrainer`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        max_length (`int` or `None`, *optional*, defaults to `1024`):
            Maximum length of the sequences in the batch, filters out entries that exceed the limit.
        disable_dropout (`bool`, *optional*, defaults to `True`):
            Whether to disable dropout in the model.
        dataset_num_proc (`int`, *optional*, defaults to `None`):
            Number of processes to use for processing the dataset.
        remove_unused_columns (`bool`, *optional*, defaults to `False`):
            Whether to remove the columns that are not used by the model's forward pass. Can be `True` only if
            the dataset is pretokenized.
    """

    max_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "Maximum length of the sequences in the batch, filters out entries that exceed the limit."
        },
    )
    num_labels: Optional[int] = field(
        default=2,
        metadata={
            "help": "Number of classes used in dataset labels"
        },
    )
    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Whether to disable dropout in the model."},
    )
    dataset_num_proc: Optional[int] = field(
        default=None,
        metadata={"help": "Number of processes to use for processing the dataset."},
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={
            "help": "Whether to remove the columns that are not used by the model's forward pass. Can be `True` only "
            "if the dataset is pretokenized."
        },
    )