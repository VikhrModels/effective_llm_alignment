from dataclasses import dataclass, field
from typing import List


@dataclass
class CommonScriptArguments:
    dataset: str | List[str] = field(
        default="/path/to/dataset",
        metadata={
            "help": "The name on HF or path to jsonl file of the dataset to use. Can be a list of paths."
        },
    )
    dataset_ratio: float | None = field(
        default=None,
        metadata={
            "help": "How much of dataset should we take. Each ratio should be between 0 and 1"
        },
    )
    test_size: float = field(
        default=None,
        metadata={
            "help": "Test set split proportion (like 0.05). If dataset already contain test split leave empty"
        },
    )
    project_name: str | None = field(
        default="default-project",
        metadata={"help": "Name of logging project (wandb or clearml)"},
    )
    pad_token: str | None = field(default=None, metadata={"help": "Special pad token"})
    bos_token: str | None = field(default=None, metadata={"help": "Special bos token"})
    eos_token: str | None = field(default=None, metadata={"help": "Special eos token"})
    chat_template: str | None = field(
        default="{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}",
        metadata={"help": "Chat template for the model"},
    )
    force_chat_template: bool = field(
        default=False,
        metadata={"help": "Force custom chat template from chat_template argument"},
    )
    added_special_tokens: List[str] | None = field(
        default=None, metadata={"help": "Additional special tokens"}
    )
    unfreeze_layers_patterns: List[str] | None = field(
        default=None,
        metadata={"help": "Patterns of layer names needed to be unfreeze for learning"},
    )
