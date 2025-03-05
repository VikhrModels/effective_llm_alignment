from collections import defaultdict
from typing import Union, Any, Optional, Dict, Literal

import pandas as pd
import torch
from torch import nn as nn
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
)
from trl import RewardTrainer, RewardConfig, SFTTrainer
from trl.trainer.utils import log_table_to_comet_experiment, print_rich_table

from src.configs.prompts_optimization_comfig import PromptsOptimizationConfig
from src.trainers.prompts_optimization.vq_prompts_tuner_module import (
    PromptCodebookTuner,
)


class PromptsSFTTrainer(SFTTrainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module],
        args: RewardConfig,
        prompt_args: PromptsOptimizationConfig,
        tokenizer: PreTrainedTokenizerBase,
        **kwargs,
    ):
        self.prompt_args = prompt_args

        # Wrap the model with PromptCodebookTuner
        tuned_model = PromptCodebookTuner(
            model=model,
            tokenizer=tokenizer,
            num_prompts=prompt_args.num_prompts,
            prompt_len=prompt_args.prompt_len,
            forbidden_token_ids=prompt_args.forbidden_token_ids,
            dissim_coef=prompt_args.dissim_coef,
            special_token_coef=prompt_args.special_token_coef,
            role=prompt_args.inserted_chat_role,
            init_prompt=prompt_args.init_prompt,
            fused_forward=prompt_args.fused_forward,
            gumbel_temp=prompt_args.gumbel_temp,
        )
        # Initialize the parent RewardTrainer with the tuned_model and other parameters
        super().__init__(model=tuned_model, args=args, tokenizer=tokenizer, **kwargs)

        # Initialize stored metrics
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

    def store_metrics(
        self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train"
    ) -> None:
        """Store metrics for later logging."""
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """Log metrics, including stored ones."""
        # Determine if we're in train or eval mode
        train_eval = "train" if "loss" in logs else "eval"

        # Add stored metrics to the logs
        for key, values in self._stored_metrics[train_eval].items():
            if values:
                logs[key] = torch.stack(values).mean().item()
        # Clear stored metrics after logging
        self._stored_metrics[train_eval].clear()

        # Call the original log method
        super().log(logs, start_time)

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict[str, torch.Tensor]]]:
        # Forward passes and loss calculation as before

        outputs = model(**inputs)

        loss_per_prompt = outputs["losses"]
        total_sft_loss = loss_per_prompt.mean()

        # Combine with auxiliary loss
        aux_loss = outputs.get("aux_loss", 0.0)
        total_loss = total_sft_loss + aux_loss

        # Collect metrics
        metrics = {}
        for i in range(self.prompt_args.num_prompts):
            metrics[f"loss_prompt_{i}"] = loss_per_prompt[i].detach().cpu()
        metrics["aux_loss"] = aux_loss.detach().cpu()
        metrics["logits_scale"] = (
            self.accelerator.unwrap_model(model).logit_scale.data.clone().detach().cpu()
        )
        metrics["mean_sft_loss"] = total_sft_loss.detach().cpu()

        # Store metrics based on current phase
        train_eval = "eval" if return_outputs else "train"
        self.store_metrics(metrics, train_eval=train_eval)

        if return_outputs:
            return total_loss, {
                "logits": outputs["logits"].mean(dim=0),
                "aux_loss": aux_loss,
            }

        return total_loss

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Ensure aux_loss is ignored in logits processing
        if ignore_keys is None:
            ignore_keys = []
        ignore_keys.append("aux_loss")
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

    def evaluate(self, *args, **kwargs):
        # num_print_samples = kwargs.pop("num_print_samples", 4)
        # self.visualize_samples(num_print_samples)
        self.log_codebook_prompts()  # Log codebook prompts during evaluation
        return Trainer.evaluate(self, *args, **kwargs)

    def log_codebook_prompts(self):
        """Log current codebook prompts to console and logging services."""
        if self.accelerator.process_index != 0:
            return  # Only log from main process

        # Retrieve current codebook prompts
        prompts_info = self.model.get_codebook_tokens(return_strings=True)
        prompts = prompts_info["prompts"]
        tokens = prompts_info["tokens"]

        # Create DataFrame for logging
        df = pd.DataFrame(
            {"Prompt Index": range(len(prompts)), "Prompt": prompts, "Tokens": tokens}
        )

        # Print to console
        print("\nCurrent Codebook Prompts:")
        print_rich_table(df)

        # Log to WandB
        if "wandb" in self.args.report_to:
            import wandb

            if wandb.run is not None:
                wandb.log({"codebook_prompts": wandb.Table(dataframe=df)})

        # Log to Comet.ml
        if "comet_ml" in self.args.report_to:
            log_table_to_comet_experiment(
                name="codebook_prompts.csv",
                table=df,
            )
