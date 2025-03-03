from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class PromptsOptimizationConfig:
    num_prompts: Optional[int] = field(
        default=5,
        metadata={"help": "Number of optimized prompts."},
    )
    prompt_len: Optional[int] = field(
        default=128,
        metadata={"help": "Prompt length of each prompt"},
    )
    dissim_coef: Optional[float] = field(
        default=0.3,
        metadata={"help": "Used in aux loss for prompts similarity penalty"},
    )
    special_token_coef: Optional[float] = field(
        default=0.8,
        metadata={
            "help": "Used in aux loss for penalty of using forbidden (special) tokens"
        },
    )
    forbidden_token_ids: Optional[List[int]] = field(
        default=None,
        metadata={"help": "List of ids of forbidden tokens in created prompts"},
    )
    inserted_chat_role: str = field(
        default="system",
        metadata={"help": "Chat role used for templating of created prompts insertion"},
    )
