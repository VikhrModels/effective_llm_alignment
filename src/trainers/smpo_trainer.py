import inspect
import warnings
from collections import defaultdict
from contextlib import nullcontext
from functools import wraps
from typing import Any, Callable, List, Tuple, Union
from typing import Dict, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    is_wandb_available,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput
from transformers.utils import is_peft_available
from trl.trainer.utils import (
    DPODataCollatorWithPadding,
    disable_dropout_in_model,
    pad_to_length,
    peft_module_casting_to_bf16,
    trl_sanitze_kwargs_for_tagging,
)

from src.callbacks.attr_scheduling import VariableSchedulerCallback
from src.configs.smpo_config import SimpleMarginPOConfig

if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training

if is_wandb_available():
    pass


class SimpleMarginPOTrainer(Trainer):
    r"""
    Initialize SimpleMarginPOTrainer.

    Args:
        model (`transformers.PreTrainedModel`):
            The model to train, preferably an `AutoModelForSequenceClassification`.
        args (`SimpleMarginPOConfig`):
            The SimpleMarginPOConfig config arguments to use for training.
        data_collator (`transformers.DataCollator`):
            The data collator to use for training. If None is specified, the default data collator (`DPODataCollatorWithPadding`) will be used
            which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
        train_dataset (`datasets.Dataset`):
            The dataset to use for training.
        eval_dataset (`datasets.Dataset`):
            The dataset to use for evaluation.
        tokenizer (`transformers.PreTrainedTokenizerBase`):
            The tokenizer to use for training. This argument is required if you want to use the default data collator.
        model_init (`Callable[[], transformers.PreTrainedModel]`):
            The model initializer to use for training. If None is specified, the default model initializer will be used.
        callbacks (`List[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
        peft_config (`Dict`, defaults to `None`):
            The PEFT configuration to use for training. If you pass a PEFT configuration, the model will be wrapped in a PEFT model.
        compute_metrics (`Callable[[EvalPrediction], Dict]`, *optional*):
            The function to use to compute the metrics. Must take a `EvalPrediction` and return
            a dictionary string to metric values.
    """

    _tag_names = ["trl", "SimpleMarginPO"]

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        args: Optional[SimpleMarginPOConfig] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        peft_config: Optional[Dict] = None,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
    ):
        if args.model_init_kwargs is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError(
                "You passed model_kwargs to the SimpleMarginPOTrainer. But your model is already instantiated."
            )
        else:
            model_init_kwargs = args.model_init_kwargs
            model_init_kwargs["torch_dtype"] = (
                model_init_kwargs["torch_dtype"]
                if model_init_kwargs["torch_dtype"] in ["auto", None]
                else getattr(torch, model_init_kwargs["torch_dtype"])
            )

        if isinstance(model, str):
            warnings.warn(
                "You passed a model_id to the SimpleMarginPOTrainer. This will automatically create an "
                "`AutoModelForCausalLM` or a `PeftModel` (if you passed a `peft_config`) for you."
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

        # Initialize this variable to False. This helps tracking the case when `peft_module_casting_to_bf16`
        # has been called in order to properly call autocast if needed.
        self._peft_has_been_casted_to_bf16 = False

        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            # if model is a peft model and we have a peft_config, we merge and unload it first
            if isinstance(model, PeftModel):
                model = model.merge_and_unload()

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
            elif getattr(args, "gradient_checkpointing", False):
                # For backward compatibility with older versions of transformers
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                else:

                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)

                    model.get_input_embeddings().register_forward_hook(
                        make_inputs_require_grad
                    )

            # get peft model with the given config
            model = get_peft_model(model, peft_config)
            if args.bf16 and getattr(model, "is_loaded_in_4bit", False):
                peft_module_casting_to_bf16(model)
                # If args.bf16 we need to explicitly call `generate` with torch amp autocast context manager
                self._peft_has_been_casted_to_bf16 = True

        # For models that use gradient_checkpointing, we need to attach a hook that enables input
        # to explicitly have `requires_grad=True`, otherwise training will either silently
        # fail or completely fail.
        elif getattr(args, "gradient_checkpointing", False):
            # For backward compatibility with older versions of transformers
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(
                    make_inputs_require_grad
                )

        if model is not None:
            self.is_encoder_decoder = model.config.is_encoder_decoder
        elif args.is_encoder_decoder is None:
            raise ValueError(
                "When no model is provided, you need to pass the parameter is_encoder_decoder."
            )
        else:
            self.is_encoder_decoder = args.is_encoder_decoder

        if self.is_encoder_decoder:
            self.decoder_start_token_id = model.config.decoder_start_token_id
            self.pad_token_id = model.config.pad_token_id

        if tokenizer is None:
            raise ValueError(
                "tokenizer must be specified to tokenize a SimpleMarginPO dataset."
            )
        if args.max_length is None:
            warnings.warn(
                "`max_length` is not set in the SimpleMarginPOConfig's init"
                " it will default to `512` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_length = 512
        else:
            max_length = args.max_length
        if args.max_prompt_length is None:
            warnings.warn(
                "`max_prompt_length` is not set in the SimpleMarginPOConfig's init"
                " it will default to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_prompt_length = 128
        else:
            max_prompt_length = args.max_prompt_length

        if args.max_target_length is None and self.is_encoder_decoder:
            warnings.warn(
                "When using an encoder decoder architecture, you should set `max_target_length` in the SimpleMarginPOConfig's init"
                " it will default to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_target_length = 128
        else:
            max_target_length = args.max_target_length

        if data_collator is None:
            data_collator = DPODataCollatorWithPadding(
                pad_token_id=tokenizer.pad_token_id,
                label_pad_token_id=args.label_pad_token_id,
                is_encoder_decoder=self.is_encoder_decoder,
            )

            if args.remove_unused_columns:
                args.remove_unused_columns = False
                # warn users
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `remove_unused_columns=False` in your TrainingArguments"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_dpo_data_collator = True
        else:
            self.use_dpo_data_collator = False

        if args.disable_dropout:
            disable_dropout_in_model(model)

        self.max_length = max_length
        self.label_pad_token_id = args.label_pad_token_id
        self.padding_value = (
            args.padding_value
            if args.padding_value is not None
            else tokenizer.pad_token_id
        )
        self.max_prompt_length = max_prompt_length
        self.truncation_mode = args.truncation_mode
        self.max_target_length = max_target_length
        self.tokenizer = tokenizer

        self.beta = args.beta
        self.target_margin = args.target_margin
        self.chosen_sft_ratio = args.chosen_sft_ratio
        self.loss_type = args.loss_type
        self.lower_clip_percentile = args.lower_clip_percentile
        self.upper_clip_percentile = args.upper_clip_percentile
        self.min_log_prob = args.min_log_prob
        self.special_token_id = self.tokenizer.eos_token_id

        assert args.target_margin >= 0, "target_margin must be greater or equal to 0"
        if args.lower_clip_percentile is not None:
            assert (
                args.lower_clip_percentile > 0 and args.lower_clip_percentile <= 0.5
            ), "lower_trim_percentile must > 0 and <= 0.5"
        if args.upper_clip_percentile is not None:
            assert (
                args.upper_clip_percentile < 1 and args.upper_clip_percentile >= 0.5
            ), "lower_trim_percentile must < 1 and >= 0.5"
        if args.min_log_prob is not None:
            assert args.min_log_prob < 0, (
                "min_log_prob must be below zero, recommended value: -2.3"
            )

        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        # Compute that only on the main process for faster data processing.
        # see: https://github.com/huggingface/trl/pull/1255
        with PartialState().local_main_process_first():
            # tokenize the dataset
            train_dataset = train_dataset.map(
                self.tokenize_row, num_proc=args.dataset_num_proc, keep_in_memory=True
            )
            if eval_dataset is not None:
                eval_dataset = eval_dataset.map(
                    self.tokenize_row,
                    num_proc=args.dataset_num_proc,
                    keep_in_memory=True,
                )

        if args.use_margin_schedule:
            callbacks.append(
                VariableSchedulerCallback(
                    attribute_name="target_margin",
                    initial_value=0.01,
                    final_value=self.target_margin,
                    schedule_type="linear",
                    target="trainer",
                )
            )

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        if not hasattr(self, "accelerator"):
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )

    def build_tokenized_answer(self, prompt, answer):
        """
        Llama tokenizer does satisfy `enc(a + b) = enc(a) + enc(b)`.
        It does ensure `enc(a + b) = enc(a) + enc(a + b)[len(enc(a)):]`.
        Reference:
            https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
        """

        full_tokenized = self.processing_class(
            prompt + answer, add_special_tokens=False
        )
        prompt_input_ids = self.processing_class(prompt, add_special_tokens=False)[
            "input_ids"
        ]

        answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids) :]
        answer_attention_mask = full_tokenized["attention_mask"][
            len(prompt_input_ids) :
        ]

        # Concat tokens to form `enc(a) + enc(a + b)[len(enc(a)):]`
        full_concat_input_ids = np.concatenate([prompt_input_ids, answer_input_ids])

        # Prepare input tokens for token by token comparison
        full_input_ids = np.array(full_tokenized["input_ids"])

        if len(full_input_ids) != len(full_concat_input_ids):
            raise ValueError(
                "Prompt input ids and answer input ids should have the same length."
            )

        # On some tokenizers, like Llama-2 tokenizer, there are occasions where tokens
        # can be merged together when tokenizing prompt+answer. This could result
        # on the last token from the prompt being different when tokenized on its own
        # vs when done as prompt+answer.
        response_token_ids_start_idx = len(prompt_input_ids)

        # If tokenized prompt is different than both prompt+answer, then it means the
        # last token has changed due to merging.
        if (
            prompt_input_ids
            != full_tokenized["input_ids"][:response_token_ids_start_idx]
        ):
            response_token_ids_start_idx -= 1

        prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
        prompt_attention_mask = full_tokenized["attention_mask"][
            :response_token_ids_start_idx
        ]

        if len(prompt_input_ids) != len(prompt_attention_mask):
            raise ValueError(
                "Prompt input ids and attention mask should have the same length."
            )

        answer_input_ids = full_tokenized["input_ids"][response_token_ids_start_idx:]
        answer_attention_mask = full_tokenized["attention_mask"][
            response_token_ids_start_idx:
        ]

        return dict(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            input_ids=answer_input_ids,
            attention_mask=answer_attention_mask,
        )

    def tokenize_row(
        self, feature, model: Optional[Union[PreTrainedModel, nn.Module]] = None
    ) -> Dict:
        """Tokenize a single row from a SimpleMarginPO specific dataset.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
        in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """
        batch = {}
        prompt = feature["prompt"]
        chosen = feature["chosen"]
        rejected = feature["rejected"]

        if not self.is_encoder_decoder:
            # Check issues below for more details
            #  1. https://github.com/huggingface/trl/issues/907
            #  2. https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
            #  3. https://github.com/LianjiaTech/BELLE/issues/337

            if not isinstance(prompt, str):
                raise ValueError(f"prompt should be an str but got {type(prompt)}")
            prompt_tokens = self.processing_class(prompt, add_special_tokens=False)
            prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}

            if not isinstance(chosen, str):
                raise ValueError(f"chosen should be an str but got {type(chosen)}")
            chosen_tokens = self.build_tokenized_answer(prompt, chosen)

            if not isinstance(rejected, str):
                raise ValueError(f"rejected should be an str but got {type(rejected)}")
            rejected_tokens = self.build_tokenized_answer(prompt, rejected)

            # Last prompt token might get merged by tokenizer and
            # it should not be included for generation if that happens
            prompt_len_input_ids = len(prompt_tokens["prompt_input_ids"])

            chosen_prompt_len_input_ids = len(chosen_tokens["prompt_input_ids"])
            rejected_prompt_len_input_ids = len(rejected_tokens["prompt_input_ids"])
            prompt_len_input_ids = min(
                chosen_prompt_len_input_ids, rejected_prompt_len_input_ids
            )

            for k, v in prompt_tokens.items():
                prompt_tokens[k] = v[:prompt_len_input_ids]

            # Make sure prompts only have one different token at most an
            # and length only differs by 1 at most
            num_diff_tokens = sum(
                [
                    a != b
                    for a, b in zip(
                        chosen_tokens["prompt_input_ids"],
                        rejected_tokens["prompt_input_ids"],
                    )
                ]
            )
            num_diff_len = abs(
                chosen_prompt_len_input_ids - rejected_prompt_len_input_ids
            )
            if num_diff_tokens > 1 or num_diff_len > 1:
                raise ValueError(
                    "Chosen and rejected prompt_input_ids might only differ on the "
                    "last token due to tokenizer merge ops."
                )

            # add BOS token to head of prompt. Avoid adding if it's already there
            bos_token_id = self.processing_class.bos_token_id
            if (
                prompt_len_input_ids == 0
                or bos_token_id != prompt_tokens["prompt_input_ids"][0]
            ) and bos_token_id is not None:
                prompt_tokens["prompt_input_ids"] = [bos_token_id] + prompt_tokens[
                    "prompt_input_ids"
                ]
                prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens[
                    "prompt_attention_mask"
                ]
            if (
                chosen_prompt_len_input_ids == 0
                or bos_token_id != chosen_tokens["prompt_input_ids"][0]
            ) and bos_token_id is not None:
                chosen_tokens["prompt_input_ids"] = [bos_token_id] + chosen_tokens[
                    "prompt_input_ids"
                ]
                chosen_tokens["prompt_attention_mask"] = [1] + chosen_tokens[
                    "prompt_attention_mask"
                ]
            if (
                rejected_prompt_len_input_ids == 0
                or bos_token_id != rejected_tokens["prompt_input_ids"][0]
            ) and bos_token_id is not None:
                rejected_tokens["prompt_input_ids"] = [bos_token_id] + rejected_tokens[
                    "prompt_input_ids"
                ]
                rejected_tokens["prompt_attention_mask"] = [1] + rejected_tokens[
                    "prompt_attention_mask"
                ]

            # add EOS token to end of answer. Avoid adding if it's already there
            eos_token_id = self.processing_class.eos_token_id
            if (
                len(chosen_tokens["input_ids"]) == 0
                or eos_token_id != chosen_tokens["input_ids"][-1]
            ):
                chosen_tokens["input_ids"].append(eos_token_id)
                chosen_tokens["attention_mask"].append(1)
            if (
                len(rejected_tokens["input_ids"]) == 0
                or eos_token_id != rejected_tokens["input_ids"][-1]
            ):
                rejected_tokens["input_ids"].append(eos_token_id)
                rejected_tokens["attention_mask"].append(1)

            longer_response_length = max(
                len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"])
            )

            # if combined sequence is too long, truncate the prompt
            for answer_tokens in [chosen_tokens, rejected_tokens, prompt_tokens]:
                if (
                    len(answer_tokens["prompt_input_ids"]) + longer_response_length
                    > self.max_length
                ):
                    if self.truncation_mode == "keep_start":
                        for k in ["prompt_input_ids", "prompt_attention_mask"]:
                            answer_tokens[k] = answer_tokens[k][
                                : self.max_prompt_length
                            ]
                    elif self.truncation_mode == "keep_end":
                        for k in ["prompt_input_ids", "prompt_attention_mask"]:
                            answer_tokens[k] = answer_tokens[k][
                                -self.max_prompt_length :
                            ]
                    else:
                        raise ValueError(
                            f"Unknown truncation mode: {self.truncation_mode}"
                        )

            # if that's still too long, truncate the response
            for answer_tokens in [chosen_tokens, rejected_tokens]:
                if (
                    len(answer_tokens["prompt_input_ids"]) + longer_response_length
                    > self.max_length
                ):
                    for k in ["input_ids", "attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][
                            : self.max_length - self.max_prompt_length
                        ]

            # Create labels
            chosen_sequence_tokens = {
                k: chosen_tokens[f"prompt_{k}"] + chosen_tokens[k]
                for k in ["input_ids", "attention_mask"]
            }
            rejected_sequence_tokens = {
                k: rejected_tokens[f"prompt_{k}"] + rejected_tokens[k]
                for k in ["input_ids", "attention_mask"]
            }
            chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
            chosen_sequence_tokens["labels"][
                : len(chosen_tokens["prompt_input_ids"])
            ] = [self.label_pad_token_id] * len(chosen_tokens["prompt_input_ids"])
            rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][
                :
            ]
            rejected_sequence_tokens["labels"][
                : len(rejected_tokens["prompt_input_ids"])
            ] = [self.label_pad_token_id] * len(rejected_tokens["prompt_input_ids"])

            for k, toks in {
                "chosen_": chosen_sequence_tokens,
                "rejected_": rejected_sequence_tokens,
                "": prompt_tokens,
            }.items():
                for type_key, tokens in toks.items():
                    if type_key == "token_type_ids":
                        continue
                    batch[f"{k}{type_key}"] = tokens

        else:
            chosen_tokens = self.processing_class(
                chosen,
                truncation=True,
                max_length=self.max_target_length,
                add_special_tokens=True,
            )
            rejected_tokens = self.processing_class(
                rejected,
                truncation=True,
                max_length=self.max_target_length,
                add_special_tokens=True,
            )
            prompt_tokens = self.processing_class(
                prompt,
                truncation=True,
                max_length=self.max_prompt_length,
                add_special_tokens=True,
            )

            batch["chosen_labels"] = chosen_tokens["input_ids"]
            batch["rejected_labels"] = rejected_tokens["input_ids"]
            batch["prompt_input_ids"] = prompt_tokens["input_ids"]
            batch["prompt_attention_mask"] = prompt_tokens["attention_mask"]

            if model is not None and hasattr(
                model, "prepare_decoder_input_ids_from_labels"
            ):
                batch["rejected_decoder_input_ids"] = (
                    model.prepare_decoder_input_ids_from_labels(
                        labels=torch.tensor(batch["rejected_labels"])
                    )
                )
                batch["chosen_decoder_input_ids"] = (
                    model.prepare_decoder_input_ids_from_labels(
                        labels=torch.tensor(batch["chosen_labels"])
                    )
                )

        return batch

    @staticmethod
    def concatenated_inputs(
        batch: Dict[str, Union[List, torch.LongTensor]],
        is_encoder_decoder: bool = False,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
            is_encoder_decoder: Whether the model is an encoder-decoder model.
            label_pad_token_id: The label pad token id.
            padding_value: The padding value to use for the concatenated inputs_ids.
            device: The device for the concatenated inputs.

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        concatenated_batch = {}

        if is_encoder_decoder:
            max_length = max(
                batch["chosen_labels"].shape[1], batch["rejected_labels"].shape[1]
            )
        else:
            max_length = max(
                batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1]
            )

        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(
                    batch[k], max_length, pad_value=pad_value
                )
        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(device=device)

        if is_encoder_decoder:
            concatenated_batch["concatenated_input_ids"] = (
                batch["prompt_input_ids"].repeat(2, 1).to(device=device)
            )
            concatenated_batch["concatenated_attention_mask"] = (
                batch["prompt_attention_mask"].repeat(2, 1).to(device=device)
            )

        return concatenated_batch

    def smpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the SimpleMarginPO loss for a batch of policy model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the SimpleMarginPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        pi_logratios = pi_logratios.to(self.accelerator.device)

        logits_lower_bound = pi_logratios - self.target_margin

        if self.loss_type == "sigmoid":
            losses = -F.logsigmoid(self.beta * logits_lower_bound)
        elif self.loss_type == "hinge":
            losses = torch.relu(-self.beta * logits_lower_bound)
        elif self.loss_type == "ipo":
            losses = (self.beta * logits_lower_bound).pow(2)
        elif self.loss_type == "smooth_lower_bound":
            losses = torch.relu(-self.beta * logits_lower_bound).pow(2)
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'smooth_lower_bound']"
            )

        chosen_rewards = (
            self.beta * policy_chosen_logps.to(self.accelerator.device).detach()
        )
        rejected_rewards = (
            self.beta * policy_rejected_logps.to(self.accelerator.device).detach()
        )

        return losses, chosen_rewards, rejected_rewards

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[
        torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor
    ]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = (
            {
                "labels": concatenated_batch["concatenated_labels"],
                "decoder_input_ids": concatenated_batch.pop(
                    "concatenated_decoder_input_ids", None
                ),
            }
            if self.is_encoder_decoder
            else {}
        )

        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            **model_kwargs,
        ).logits

        all_logps = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            len_chosen,
            average_log_prob=True,  # SimpleMarginPO/IPO mode
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            lower_clip_percentile=self.lower_clip_percentile,
            upper_clip_percentile=self.upper_clip_percentile,
            min_log_prob=self.min_log_prob,
            special_token_id=self.special_token_id,
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        chosen_labels = concatenated_batch["concatenated_labels"][:len_chosen]
        rejected_labels = concatenated_batch["concatenated_labels"][len_chosen:]

        return (
            chosen_logps,
            rejected_logps,
            chosen_logits,
            rejected_logits,
            chosen_labels,
            rejected_labels,
        )

    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        chosen_count: int,
        lower_clip_percentile: float = None,
        upper_clip_percentile: float = None,
        min_log_prob: float = None,
        average_log_prob: bool = True,
        label_pad_token_id: int = -100,
        special_token_id: int = None,
        is_encoder_decoder: bool = False,
    ) -> torch.FloatTensor:
        if logits.shape[:-1] != labels.shape:
            raise ValueError(
                "Logits (batch and sequence length dim) and labels must have the same shape."
            )

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id

        labels[(labels == label_pad_token_id)] = 0

        # TODO: replace with selective_log_softmax from trl
        per_token_logps = torch.gather(
            logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
        ).squeeze(2)

        # Invert logp for special_token_id for rejected tokens
        if special_token_id is not None:
            special_token_mask = (labels == special_token_id) & loss_mask
            per_token_logps[special_token_mask][chosen_count:] = -per_token_logps[
                special_token_mask
            ][chosen_count:]

        # Winsorize extremal values for rejected tokens
        if lower_clip_percentile is not None:
            per_token_logps_float = (
                per_token_logps[loss_mask][chosen_count:].detach().float()
            )
            lower_bound = torch.quantile(
                per_token_logps_float, lower_clip_percentile, dim=-1
            )
            per_token_logps[loss_mask][chosen_count:] = torch.where(
                per_token_logps[loss_mask][chosen_count:] < lower_bound,
                lower_bound,
                per_token_logps[loss_mask][chosen_count:],
            )
            # loss_mask[chosen_count:] = torch.where(per_token_logps[chosen_count:] < lower_bound, False, loss_mask[chosen_count:])

        # Winsorize extremal values for chosen tokens
        if upper_clip_percentile is not None:
            per_token_logps_float = (
                per_token_logps[loss_mask][:chosen_count].detach().float()
            )
            upper_bound = torch.quantile(
                per_token_logps_float, upper_clip_percentile, dim=-1
            )
            per_token_logps[loss_mask][:chosen_count] = torch.where(
                per_token_logps[loss_mask][:chosen_count] > upper_bound,
                upper_bound,
                per_token_logps[loss_mask][:chosen_count],
            )
            # loss_mask[:chosen_count] = torch.where(per_token_logps[:chosen_count] > upper_bound, False, loss_mask[:chosen_count])

        # Clip minimum logprob for rejected tokens
        if min_log_prob is not None:
            per_token_logps[loss_mask][chosen_count:] = torch.where(
                per_token_logps[loss_mask][chosen_count:] < min_log_prob,
                min_log_prob,
                per_token_logps[loss_mask][chosen_count:],
            )

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the SimpleMarginPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}
        prefix = "eval_" if train_eval == "eval" else ""

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            chosen_labels,
            rejected_labels,
        ) = self.concatenated_forward(model, batch)

        losses, chosen_rewards, rejected_rewards = self.smpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
        )

        loss = losses.mean()

        # Double SFT part
        loss_func = nn.CrossEntropyLoss()

        if not self.is_encoder_decoder:
            policy_chosen_logits = policy_chosen_logits[..., :-1, :].contiguous()
            chosen_labels = chosen_labels[..., 1:].clone()
        chosen_sft_loss = loss_func(
            policy_chosen_logits.view(-1, policy_chosen_logits.shape[-1]),
            chosen_labels.view(-1),
        )
        metrics[f"{prefix}chosen_sft_loss"] = chosen_sft_loss.detach().cpu()

        if not self.is_encoder_decoder:
            policy_rejected_logits = policy_rejected_logits[..., :-1, :].contiguous()
            rejected_labels = rejected_labels[..., 1:].clone()
        rejeced_sft_loss = loss_func(
            policy_rejected_logits.view(-1, policy_rejected_logits.shape[-1]),
            rejected_labels.view(-1),
        )
        metrics[f"{prefix}rejected_sft_loss"] = rejeced_sft_loss.detach().cpu()

        metrics[f"{prefix}weighted_sft_loss"] = (
            chosen_sft_loss.detach().cpu()
            + rejeced_sft_loss.detach().cpu() * (1 - self.chosen_sft_ratio)
        )

        combined_sft_loss = (
            chosen_sft_loss * self.chosen_sft_ratio
            + rejeced_sft_loss * (1 - self.chosen_sft_ratio)
        )
        loss = combined_sft_loss + loss

        # Caclulating metrics

        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (
            (chosen_rewards - rejected_rewards).mean().cpu()
        )
        if self.args.use_margin_schedule:
            metrics[f"{prefix}rewards/target_margin"] = self.target_margin

        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logits/rejected"] = (
            policy_rejected_logits.detach().mean().cpu()
        )
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean().cpu()

        return loss, metrics

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
        num_items_in_batch=None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.use_dpo_data_collator:
            warnings.warn(
                "compute_loss is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )

        compute_loss_context_manager = (
            torch.cuda.amp.autocast
            if self._peft_has_been_casted_to_bf16
            else nullcontext
        )

        with compute_loss_context_manager():
            loss, metrics = self.get_batch_loss_metrics(
                model, inputs, train_eval="train"
            )

        # force log the metrics
        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        if not self.use_dpo_data_collator:
            warnings.warn(
                "prediction_step is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        prediction_context_manager = (
            torch.cuda.amp.autocast
            if self._peft_has_been_casted_to_bf16
            else nullcontext
        )

        with torch.no_grad(), prediction_context_manager():
            loss, metrics = self.get_batch_loss_metrics(
                model, inputs, train_eval="eval"
            )

        # force log the metrics
        self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)

        # logits for the chosen and rejected samples from model
        logits_dict = {
            "eval_logits/chosen": metrics["eval_logits/chosen"],
            "eval_logits/rejected": metrics["eval_logits/rejected"],
        }
        logits = tuple(
            v.unsqueeze(dim=0) for k, v in logits_dict.items() if k not in ignore_keys
        )
        logits = torch.stack(logits).mean(axis=1).to(self.accelerator.device)
        labels = torch.zeros(logits.shape[0], device=self.accelerator.device)

        return (loss.detach(), logits, labels)

    def store_metrics(
        self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train"
    ) -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs, start_time)

    @wraps(Trainer.push_to_hub)
    def push_to_hub(
        self,
        commit_message: Optional[str] = "End of training",
        blocking: bool = True,
        **kwargs,
    ) -> str:
        """
        Overwrite the `push_to_hub` method in order to force-add the tag "smpo" when pushing the
        model on the Hub. Please refer to `~transformers.Trainer.push_to_hub` for more details.
        """
        kwargs = trl_sanitze_kwargs_for_tagging(
            model=self.model, tag_names=self._tag_names, kwargs=kwargs
        )

        return super().push_to_hub(
            commit_message=commit_message, blocking=blocking, **kwargs
        )
