import warnings
from typing import Union, List, Any, Dict

import torch
from transformers import DataCollatorForLanguageModeling

from src.utils.array_utils import filter_indices


class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    """
    Data collator used for completion tasks. It ensures that all the tokens of the labels are set to an 'ignore_index'
    when they do not come from the assistant. This ensure that the loss is only
    calculated on the completion made by the assistant.

    Args:
        response_template (`Union[str, List[int]]`): the template form that indicates the start of the response, typically something like
            '### Response:\n'. It can also be passed as tokenized ids, which can be useful when using a tokenizer that encodes the response
            differently if it does not have proper context.
        mlm (`bool`, *optional*, defaults to `False`): Whether or not to use masked language modeling in the underlying
            `DataCollatorForLanguageModeling` class. Note that this option currently has no effect but is present
             for flexibility and backwards-compatibility.
        ignore_index (`int`, *optional*, defaults to `-100`):
            The index to use to ignore the initial tokens with
    """

    def __init__(
        self,
        response_prompt_template: Union[str, List[int]],
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)

        self.response_prompt_template = response_prompt_template

        if isinstance(response_prompt_template, str):
            self.response_token_ids = self.tokenizer.encode(
                self.response_prompt_template, add_special_tokens=False
            )
        else:
            self.response_token_ids = self.response_prompt_template

        self.eos_token_id = self.tokenizer.eos_token_id

        if not self.mlm and self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            warnings.warn(
                "The pad_token_id and eos_token_id values of this tokenizer are identical. "
                "If you are planning for multi-turn training, "
                "it can result in the model continuously generating questions and answers without eos token. "
                "To avoid this, set the pad_token_id to a different value."
            )

        self.ignore_index = ignore_index

    def torch_call(
        self, examples: List[Union[List[int], Any, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        for i in range(len(examples)):
            response_token_ids_start_indexes = []
            eos_token_indexes = []

            for idx in torch.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                if (
                    self.response_token_ids
                    == batch["labels"][i][
                        idx : idx + len(self.response_token_ids)
                    ].tolist()
                ):
                    response_token_ids_start_indexes.append(idx.item())

            for idx in torch.where(batch["labels"][i] == self.eos_token_id)[0]:
                eos_token_indexes.append(idx.item())

            eos_token_indexes = filter_indices(
                response_token_ids_start_indexes, eos_token_indexes
            )

            if not response_token_ids_start_indexes or not eos_token_indexes:
                warnings.warn(
                    f"Could not find response key `{self.response_prompt_template}` in the "
                    f"following instance: {self.tokenizer.decode(batch['input_ids'][i])} "
                    f"This instance will be ignored in loss calculation. "
                    f"Note, if this happens often, consider increasing the `max_seq_length`."
                )
                batch["labels"][i, :] = self.ignore_index
            else:
                new_labels = torch.full_like(batch["labels"][i], self.ignore_index).to(
                    device=batch["labels"][i].device
                )

                for start, end in zip(
                    response_token_ids_start_indexes, eos_token_indexes
                ):
                    new_labels[start : end + 1] = batch["labels"][i, start : end + 1]

                batch["labels"][i] = new_labels

        return batch
