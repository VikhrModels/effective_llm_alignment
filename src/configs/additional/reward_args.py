from dataclasses import dataclass, field

from src.configs.additional.common_script_args import CommonScriptArguments


@dataclass
class RMScriptArguments(CommonScriptArguments):
    def __post_init__(self):
        self.project_name = (
            "reward-modeling"
            if self.project_name == "default-project"
            else self.project_name
        )
