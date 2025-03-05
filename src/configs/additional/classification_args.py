from dataclasses import dataclass

from src.configs.additional.common_script_args import CommonScriptArguments


@dataclass
class CLFScriptArguments(CommonScriptArguments):
    def __post_init__(self):
        self.project_name = (
            "classification"
            if self.project_name == "default-project"
            else self.project_name
        )
