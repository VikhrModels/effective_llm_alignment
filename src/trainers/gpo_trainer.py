import inspect
import random
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Union, Optional, Dict, Callable, List, Tuple, Type, Any

import datasets
import numpy as np
import torch
import transformers
from accelerate import PartialState
from accelerate.utils import is_peft_available, is_peft_model
from packaging import version
from torch import nn
from torch.utils.data import Sampler, Dataset, IterableDataset, DataLoader
from transformers import (
    Trainer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    BaseImageProcessor,
    FeatureExtractionMixin,
    ProcessorMixin,
    EvalPrediction,
    TrainerCallback,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.models.auto.modeling_auto import MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES
from transformers.trainer_utils import seed_worker
from trl import create_reference_model
from trl.trainer.utils import (
    selective_log_softmax,
    prepare_deepspeed,
    disable_dropout_in_model,
    peft_module_casting_to_bf16,
    pad,
)

from src.configs.gpo_config import GroupedPOConfig

if is_peft_available():
    from peft import (
        PeftModel,
        get_peft_model,
        prepare_model_for_kbit_training,
        PeftConfig,
    )


class GroupBatchSampler(Sampler):
    r"""
    Batch sampler which returns per-device batches with elements only from single group.
    """

    def __init__(self, group_ids, max_device_batch_size, gpus_count, shuffle=True):
        self.group_ids = group_ids
        self.max_device_batch_size = max_device_batch_size
        self.gpus_count = gpus_count
        self.shuffle = shuffle

        # Группируем индексы по group_id
        self.groups = defaultdict(list)
        for idx, gid in enumerate(group_ids):
            self.groups[gid].append(idx)

        # Проверяем условия для каждой группы
        for gid, indices in self.groups.items():
            group_size = len(indices)
            assert group_size % gpus_count == 0, (
                f"Group {gid} size {group_size} is not divisible by {gpus_count}."
            )
            per_gpu_batch = group_size // gpus_count
            assert per_gpu_batch <= max_device_batch_size, (
                f"Per-GPU batch {per_gpu_batch} for group {gid} exceeds max {max_device_batch_size}."
            )

        # Подготовка эффективных батчей
        self.batches = []
        for gid, indices in self.groups.items():
            self.batches.append(indices)

        if self.shuffle:
            random.shuffle(self.batches)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


@dataclass
class GPODataCollatorWithPadding:
    r"""
    GPO DataCollator class that pads the tokenized inputs to the maximum length of the batch.

    Args:
        pad_token_id (`int` defaults to 0):
            The tokenizer's pad_token_id.
        label_pad_token_id (`int`, defaults to -100):
            The label used for masking.
        is_encoder_decoder (`bool` or `None`, `optional`, defaults to `None`):
            Whether you model has an encoder_decoder architecture.
    """

    pad_token_id: int = 0
    label_pad_token_id: int = -100
    is_encoder_decoder: Optional[bool] = False

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        prompt_input_ids = [
            torch.tensor(example["prompt_input_ids"]) for example in features
        ]
        prompt_attention_mask = [
            torch.ones_like(input_ids) for input_ids in prompt_input_ids
        ]
        completion_input_ids = [
            torch.tensor(example["completion_input_ids"]) for example in features
        ]
        completion_attention_mask = [
            torch.ones_like(input_ids) for input_ids in completion_input_ids
        ]

        pad_value = (
            self.pad_token_id if self.is_encoder_decoder else self.label_pad_token_id
        )

        output = {
            "prompt_input_ids": pad(
                prompt_input_ids, padding_value=pad_value, padding_side="left"
            ),
            "prompt_attention_mask": pad(
                prompt_attention_mask, padding_value=0, padding_side="left"
            ),
            "completion_input_ids": pad(completion_input_ids, padding_value=pad_value),
            "completion_attention_mask": pad(
                completion_attention_mask, padding_value=0
            ),
            "group_id": torch.tensor([ex["group_id"] for ex in features]),
            "group_size": torch.tensor([ex["group_size"] for ex in features]),
            "advantage": torch.tensor([ex["advantage"] for ex in features]),
        }

        return output


def _enable_gradient_checkpointing(
    model: PreTrainedModel, args: GroupedPOConfig
) -> PreTrainedModel:
    """Enables gradient checkpointing for the model."""
    # Ensure use_cache is disabled
    model.config.use_cache = False

    # Enable gradient checkpointing on the base model for PEFT
    if is_peft_model(model):
        model.base_model.gradient_checkpointing_enable()
    # Enable gradient checkpointing for non-PEFT models
    else:
        model.gradient_checkpointing_enable()

    gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
    use_reentrant = (
        "use_reentrant" not in gradient_checkpointing_kwargs
        or gradient_checkpointing_kwargs["use_reentrant"]
    )

    if use_reentrant:
        model.enable_input_require_grads()

    return model


class GroupedPOTrainer(Trainer):
    _tag_names = ["trl", "gpo"]

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: GroupedPOConfig = None,
        train_dataset: Optional[
            Union[Dataset, IterableDataset, "datasets.Dataset"]
        ] = None,
        eval_dataset: Optional[
            Union[Dataset, Dict[str, Dataset], "datasets.Dataset"]
        ] = None,
        processing_class: Optional[
            Union[
                PreTrainedTokenizerBase,
                BaseImageProcessor,
                FeatureExtractionMixin,
                ProcessorMixin,
            ]
        ] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_loss_func: Optional[Callable] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[
            Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]
        ] = (None, None),
        optimizer_cls_and_kwargs: Optional[
            Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]
        ] = None,
        preprocess_logits_for_metrics: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        peft_config: Optional["PeftConfig"] = None,
    ):
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}

        self.label_pad_token_id = args.label_pad_token_id
        self.padding_value = (
            args.padding_value
            if args.padding_value is not None
            else self.processing_class.pad_token_id
        )
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length

        # Logp Regularization (not used for now)
        # self.lower_clip_percentile = args.lower_clip_percentile
        # self.upper_clip_percentile = args.upper_clip_percentile
        # self.min_log_prob = args.min_log_prob
        # self.special_token_id = self.tokenizer.eos_token_id

        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if (
                isinstance(torch_dtype, torch.dtype)
                or torch_dtype == "auto"
                or torch_dtype is None
            ):
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False
                if args.gradient_checkpointing
                else model_init_kwargs.get("use_cache")
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        # PEFT setup
        if peft_config is not None:
            if not is_peft_available():
                raise ImportError(
                    "PEFT is required to use `peft_config`. Run `pip install peft`."
                )
            # if model is a peft model and we have a peft_config, we merge and unload it first
            if isinstance(model, PeftModel):
                model = model.merge_and_unload()
            model = get_peft_model(model, peft_config)

            if getattr(model, "is_loaded_in_8bit", False) or getattr(
                model, "is_loaded_in_4bit", False
            ):
                _support_gc_kwargs = hasattr(
                    args, "gradient_checkpointing_kwargs"
                ) and "gradient_checkpointing_kwargs" in list(
                    inspect.signature(prepare_model_for_kbit_training).parameters
                )

                prepare_model_kwargs = {
                    "use_gradient_checkpointing": args.gradient_checkpointing
                }

                if _support_gc_kwargs:
                    prepare_model_kwargs["gradient_checkpointing_kwargs"] = (
                        args.gradient_checkpointing_kwargs
                    )

                model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)

            if args.bf16 and getattr(model, "is_loaded_in_4bit", False):
                peft_module_casting_to_bf16(model)
                # If args.bf16 we need to explicitly call `generate` with torch amp autocast context manager
                self._peft_has_been_casted_to_bf16 = True

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model = _enable_gradient_checkpointing(model, args)

        self.is_encoder_decoder = model.config.is_encoder_decoder
        self.is_vision_model = (
            model.config.model_type in MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES.keys()
        )
        self.is_peft_model = is_peft_available() and isinstance(model, PeftModel)

        if self.is_vision_model:
            warnings.warn(
                "Vision models are not fully supported in GPOTrainer (no pixel_values)",
                UserWarning,
            )

        # Reference model
        self.beta = args.kl_beta
        if self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        elif is_deepspeed_zero3_enabled():
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                model_id, **model_init_kwargs
            )
        elif is_peft_model(model):
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None
        else:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)

        # Disable dropout in the model and reference model
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(
                model_id, padding_side="right"
            )

        # Preparing special data collator
        data_collator = GPODataCollatorWithPadding(
            pad_token_id=self.padding_value,
            label_pad_token_id=self.label_pad_token_id,
            is_encoder_decoder=self.is_encoder_decoder,
        )

        if args.remove_unused_columns:
            args.remove_unused_columns = False
            # warn users
            warnings.warn(
                "When using GPODataCollatorWithPadding, you should set `remove_unused_columns=False` in your TrainingArguments"
                " we have set it for you, but you should do it yourself in the future.",
                UserWarning,
            )

        with PartialState().local_main_process_first():
            # tokenize the dataset
            train_dataset = train_dataset.map(
                self.tokenize_row,
                batched=True,
                batch_size=1,
                num_proc=args.dataset_num_proc,
                with_indices=True,
                keep_in_memory=True,
            )
            if eval_dataset is not None:
                eval_dataset = eval_dataset.map(
                    self.tokenize_row,
                    batched=True,
                    batch_size=1,
                    num_proc=args.dataset_num_proc,
                    with_indices=True,
                    keep_in_memory=True,
                )

        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            processing_class,
            model_init,
            compute_loss_func,
            compute_metrics,
            callbacks,
            optimizers,
            optimizer_cls_and_kwargs,
            preprocess_logits_for_metrics,
        )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(
                    self.ref_model, evaluation_mode=True
                )

    def _tokenize_single(self, prompt: str, completion: str):
        batch = {}

        bos_token_id = self.processing_class.bos_token_id
        eos_token_id = self.processing_class.eos_token_id

        prompt_input_ids = self.processing_class(
            prompt,
            truncation=True,
            max_length=self.max_prompt_length,
            add_special_tokens=False,
        )["input_ids"]
        completion_input_ids = self.processing_class(
            completion,
            truncation=True,
            max_length=self.max_completion_length,
            add_special_tokens=False,
        )["input_ids"]

        if bos_token_id is not None and bos_token_id != prompt_input_ids[0]:
            prompt_input_ids = [bos_token_id] + prompt_input_ids
        if eos_token_id is not None and eos_token_id != prompt_input_ids[-1]:
            prompt_input_ids = prompt_input_ids + [eos_token_id]
        if self.is_encoder_decoder:
            completion_input_ids = [bos_token_id] + completion_input_ids
        if eos_token_id is not None and eos_token_id != completion_input_ids[-1]:
            completion_input_ids = completion_input_ids + [eos_token_id]

        batch["prompt_input_ids"] = prompt_input_ids
        batch["completion_input_ids"] = completion_input_ids

        return batch

    def tokenize_row(
        self,
        features,  # Input is a batch (even if batch_size=1)
        indices: List[int],  # List of indices since we're using batched processing
    ) -> Dict[str, List]:
        # For batch_size=1, unpack the single feature
        prompt = features["prompt"][0]
        completions_list = features["completions"][0]
        rewards_list = features["rewards"][0]

        reward_mean = np.mean(rewards_list)
        reward_std = np.std(rewards_list)

        tokenized_examples = defaultdict(list)

        idx = indices[0]  # Since batch_size=1, indices is a single-element list
        for completion, reward in zip(completions_list, rewards_list):
            batch = self._tokenize_single(prompt, completion)

            # Append each tokenized example to the batch
            for key in tokenized_examples:
                tokenized_examples[key].append(batch[key])

            tokenized_examples["group_id"].append(idx)
            tokenized_examples["group_size"].append(len(completions_list))

            advantage = (reward - reward_mean) / (reward_std + 1e-4)
            tokenized_examples["advantage"].append(advantage)

        return tokenized_examples

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        group_ids = [ex["group_id"] for ex in self.train_dataset]
        data_collator = self.data_collator

        dataloader_params = {
            # "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            # dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["batch_sampler"] = GroupBatchSampler(
                group_ids=group_ids,
                max_device_batch_size=self.args.per_device_train_batch_size,
                gpus_count=self.accelerator.num_processes,
            )
            # dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def get_eval_dataloader(
        self, eval_dataset: Optional[Union[str, Dataset]] = None
    ) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`str` or `torch.utils.data.Dataset`, *optional*):
                If a `str`, will use `self.eval_dataset[eval_dataset]` as the evaluation dataset. If a `Dataset`, will override `self.eval_dataset` and must implement `__len__`. If it is a [`~datasets.Dataset`], columns not accepted by the `model.forward()` method are automatically removed.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        dataloader_key = eval_dataset if isinstance(eval_dataset, str) else "eval"
        if (
            hasattr(self, "_eval_dataloaders")
            and dataloader_key in self._eval_dataloaders
            and self.args.dataloader_persistent_workers
        ):
            return self.accelerator.prepare(self._eval_dataloaders[dataloader_key])

        eval_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset
            if eval_dataset is not None
            else self.eval_dataset
        )
        group_ids = [ex["group_id"] for ex in eval_dataset]
        data_collator = self.data_collator

        dataloader_params = {
            # "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            # dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["batch_sampler"] = GroupBatchSampler(
                group_ids=group_ids,
                max_device_batch_size=self.args.per_device_eval_batch_size,
                gpus_count=self.accelerator.num_processes,
            )
            # dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version
        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
        if self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = eval_dataloader
            else:
                self._eval_dataloaders = {dataloader_key: eval_dataloader}

        return self.accelerator.prepare(eval_dataloader)

    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=logits_to_keep + 1,
        ).logits
        logits = logits[
            :, :-1, :
        ]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

        input_ids = input_ids[:, -logits_to_keep:]
        # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
        # See https://github.com/huggingface/trl/issues/2770
        logits = logits[:, -logits_to_keep:]
        return selective_log_softmax(
            logits, input_ids
        )  #  compute logprobs for the input tokens

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = (
            inputs["prompt_input_ids"],
            inputs["prompt_attention_mask"],
        )
        completion_ids, completion_mask = (
            inputs["completion_input_ids"],
            inputs["completion_attention_mask"],
        )
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(
            1
        )  # we only need to compute the logits for the completion tokens

        per_token_logps = self._get_per_token_logps(
            model, input_ids, attention_mask, logits_to_keep
        )

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            with torch.no_grad():
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model,
                        input_ids,
                        attention_mask,
                        logits_to_keep,
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(
                            self.model,
                            input_ids,
                            attention_mask,
                            logits_to_keep,
                        )
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )

        # Compute the loss
        advantages = inputs["advantage"].unsqueeze(1)  # [B, 1]
        group_sizes = inputs["group_size"]  # [B]

        per_toke_probs = torch.exp(per_token_logps)
        per_token_rewards = per_token_loss = -(per_toke_probs * advantages)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        loss = (per_token_loss * completion_mask).sum(1) / completion_mask.sum(
            1
        )  # average over tokens
        loss = (loss / group_sizes).sum()  # per group loss

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(
                self.accelerator.gather_for_metrics(mean_kl).mean().item()
            )

        mean_logpts = (per_toke_probs * completion_mask).sum() / completion_mask.sum()
        mean_rewards = (
            -per_token_rewards * completion_mask
        ).sum() / completion_mask.sum()
        self._metrics[mode]["mean_logps"].append(
            self.accelerator.gather_for_metrics(mean_logpts).mean().item()
        )
        self._metrics[mode]["mean_rewards"].append(
            self.accelerator.gather_for_metrics(mean_rewards).mean().item()
        )

        return loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys: Optional[list[str]] = None,
    ):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "eval" if self.control.should_evaluate else "train"
        metrics = {
            key: sum(val) / len(val) for key, val in self._metrics[mode].items()
        }  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics[mode].clear()
