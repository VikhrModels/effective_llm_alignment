from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class PromptsOptimizationConfig:
    num_prompts: Optional[int] = field(
        default=3,
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
    gumbel_temp: Optional[float] = field(
        default=0.5,
        metadata={"help": "Temperature for gumbel softmax trick"},
    )
    gumbel_noise_scale: Optional[float] = field(
        default=0.05,
        metadata={"help": "Multiplier of added gumbel noise inside softmax"},
    )
    forbidden_token_ids: Optional[List[int]] = field(
        default=None,
        metadata={"help": "List of ids of forbidden tokens in created prompts"},
    )
    inserted_chat_role: str = field(
        default="system",
        metadata={"help": "Chat role used for templating of created prompts insertion"},
    )
    fused_forward: bool = field(
        default=True,
        metadata={
            "help": "Use full in-batch forward, instead of for loop, memory usage increase."
        },
    )
    init_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "Prompt to init optimization from"},
    )
