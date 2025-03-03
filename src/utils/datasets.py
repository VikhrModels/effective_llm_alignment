import numpy as np
import warnings
from accelerate import PartialState
from datasets import load_dataset, DatasetDict, Dataset, concatenate_datasets


def _load_dataset_from_path(path: str, test_size: float | None) -> DatasetDict:
    if path.endswith("jsonl"):
        dataset = load_dataset("json", data_files=path)
    else:
        dataset = load_dataset(path)
    if test_size is not None:
        dataset = dataset["train"].train_test_split(
            test_size, seed=42, load_from_cache_file=True
        )
    return dataset


def _get_subset_from_dataset(dataset: Dataset, dataset_ratio: float | None) -> Dataset:
    indices = np.random.choice(
        range(len(dataset)), int(dataset_ratio * len(dataset)), replace=False
    )
    dataset = dataset.select(indices)
    return dataset


def _get_subset_from_dataset_dict(
    dataset: DatasetDict, dataset_ratio: float | None
) -> DatasetDict:
    dataset["train"] = _get_subset_from_dataset(dataset["train"], dataset_ratio)
    dataset["test"] = _get_subset_from_dataset(dataset["test"], dataset_ratio)
    return dataset


def load_datasets(
    path: str | list, test_size: float | None, dataset_ratio: float | list | None
):
    with PartialState().local_main_process_first():
        if dataset_ratio is None:
            warnings.warn(
                "You haven't set dataset ratio for your datasets. Assuming that it's 1 for all datasets."
            )
            dataset_ratio = [1] * len(path) if isinstance(path, list) else 1
        if isinstance(path, list) and not isinstance(dataset_ratio, list):
            raise ValueError("You shold pass dataset ratio for all of your datasets.")
        if not isinstance(path, list) and isinstance(dataset_ratio, list):
            raise ValueError("You shold pass datasets for all of your dataset ratios.")
        if (
            isinstance(path, list)
            and isinstance(dataset_ratio, list)
            and len(path) != len(dataset_ratio)
        ):
            raise ValueError(
                f"You have set {len(path)} datasets and {len(dataset_ratio)} dataset ratios, but it should be equal."
            )
        if isinstance(path, list):
            all_datasets = [_load_dataset_from_path(d, test_size) for d in path]
            truncated_datasets = [
                _get_subset_from_dataset_dict(d, ratio)
                for d, ratio in zip(all_datasets, dataset_ratio)
            ]
            ds = DatasetDict()
            ds["train"] = concatenate_datasets([d["train"] for d in truncated_datasets])
            ds["test"] = concatenate_datasets([d["test"] for d in truncated_datasets])
        else:
            ds = _load_dataset_from_path(path, test_size)
            ds = _get_subset_from_dataset_dict(ds, dataset_ratio)
    return ds


def prepare_generative_row(row, tokenizer, max_length):
    constructed_prompt = tokenizer.apply_chat_template(
        row["prompt"], tokenize=False, add_generation_prompt=True
    )
    return tokenizer(
        constructed_prompt,
        truncation=True,
        padding=True,
        max_length=max_length,
        add_special_tokens=False,
    )
