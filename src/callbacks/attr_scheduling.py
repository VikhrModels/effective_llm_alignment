import math
from typing import Literal

from transformers import TrainerCallback


class VariableSchedulerCallback(TrainerCallback):
    """General-purpose variable scheduler callback with multiple schedule types."""

    def __init__(
        self,
        attribute_name: str,
        initial_value: float,
        final_value: float,
        schedule_type: str = "cosine",
        warmup_steps: int = 0,
        cycle: bool = False,
        cycle_scale: float = 1.0,
        target: Literal["model", "trainer"] = "model",
    ):
        """
        Args:
            attribute_name: Name of the attribute to update in the model
            initial_value: Starting value of the variable
            final_value: Target end value of the variable
            schedule_type: Type of schedule ('cosine', 'linear', 'exponential')
            warmup_steps: Number of steps to maintain initial value before starting schedule
            cycle: Whether to cycle the schedule (for cosine only)
            cycle_scale: Scale factor for cycling (number of cycles)
        """
        self.attribute_name = attribute_name
        self.initial_value = initial_value
        self.final_value = final_value
        self.schedule_type = schedule_type
        self.warmup_steps = warmup_steps
        self.cycle = cycle
        self.cycle_scale = cycle_scale
        self.total_steps = None
        self.target = target

        if self.target not in ["model", "trainer"]:
            raise ValueError(f"Invalid target '{target}'. Must be 'model' or 'trainer'.")

    def on_train_begin(self, args, state, control, **kwargs):
        self.total_steps = state.max_steps - self.warmup_steps
        if self.total_steps <= 0:
            raise ValueError("Total training steps must be greater than warmup steps")

    def on_step_begin(self, args, state, control, **kwargs):
        """Update variable at the beginning of each step"""
        current_step = state.global_step

        # Handle warmup period
        if current_step < self.warmup_steps:
            current_value = self.initial_value
        else:
            progress = (current_step - self.warmup_steps) / self.total_steps
            progress = min(progress, 1.0)  # Clamp progress to 1.0

            if self.cycle and self.schedule_type == "cosine":
                progress = progress * self.cycle_scale % 1.0

            # Calculate current value based on schedule type
            if self.schedule_type == "cosine":
                current_value = self.final_value + 0.5 * (
                    self.initial_value - self.final_value
                ) * (1 + math.cos(math.pi * progress))
            elif self.schedule_type == "linear":
                current_value = (
                    self.initial_value
                    + (self.final_value - self.initial_value) * progress
                )
            elif self.schedule_type == "exponential":
                ratio = progress
                current_value = (
                    self.initial_value
                    * (self.final_value / self.initial_value) ** ratio
                )
            else:
                raise ValueError(f"Unknown schedule type: {self.schedule_type}")

        # Get target object
        target_obj = kwargs.get(self.target)
        if not target_obj:
            raise ValueError(f"Could not find {self.target} in callback arguments")

        # Handle distributed/DataParallel models
        target_obj = target_obj.module if hasattr(target_obj, "module") else target_obj

        if hasattr(target_obj, self.attribute_name):
            setattr(target_obj, self.attribute_name, current_value)
        else:
            raise AttributeError(f"{type(target_obj)} does not have attribute {self.attribute_name}")

    def get_current_value(self, model):
        """Get current value of the scheduled variable"""
        model_obj = model.module if hasattr(model, "module") else model
        return getattr(model_obj, self.attribute_name)
