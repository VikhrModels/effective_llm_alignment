import glob
import os
import random
import shutil
import tempfile

import pandas as pd
import torch
import wandb
from accelerate import PartialState
from datasets import Dataset
from tabulate import tabulate
from tqdm import tqdm
from transformers import TrainerCallback, TrainerState, TrainerControl, PreTrainedTokenizer

try:
    from clearml import Task
except ImportError:
    pass


def pretty_print_dataframe(df, max_prompt_length=200):
    formatted_df = df.copy()
    formatted_df['Prompt'] = formatted_df['Prompt'].apply(
        lambda x: ('...' + x[-max_prompt_length:]) if len(x) > max_prompt_length else x
    )
    print(tabulate(formatted_df, headers='keys', tablefmt='simple', showindex=False))


def save_dataframe(df, process_index, temp_dir_path):
    if not os.path.exists(temp_dir_path):
        os.makedirs(temp_dir_path)
    filename = os.path.join(temp_dir_path, f'df_{process_index}.csv')
    df.to_csv(filename, index=False)


class GenerateExamplesCallback(TrainerCallback):
    def __init__(self,
                 preprocessed_dataset: Dataset,
                 tokenizer: PreTrainedTokenizer,
                 num_examples=5,
                 max_new_tokens=256,
                 is_deepspeed_zero3=False,
                 logger_backend='wandb'):
        """
        :param preprocessed_dataset: Preprocessed datasets.Dataset after applying tokenizer
        :param tokenizer: Tokenizer for decoding
        :param num_examples: Number of examples to generate for each evaluation
        :param max_new_tokens: Max new tokens length of generated examples
        :param logger_backend: 'clearml' or 'wandb' for choosing the logging tool
        """
        self.dataset = preprocessed_dataset
        self.tokenizer = tokenizer
        self.num_examples = min(num_examples, len(self.dataset))
        self.max_new_tokens = max_new_tokens
        self.logger_backend = logger_backend
        self.is_deepspeed_zero3 = is_deepspeed_zero3

        temp_dir_base = tempfile.gettempdir()
        self.temp_dir_path = os.path.join(temp_dir_base, "callback_generate_examples_dir")

        sample_indices = random.choices(range(len(self.dataset)), k=self.num_examples)
        self.samples = self.dataset.select(sample_indices)

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):

        model = kwargs['model']
        model.eval()

        records = []

        PartialState().wait_for_everyone()

        with PartialState().split_between_processes(self.samples) as samples:
            for example in tqdm(samples, desc='Generating examples'):
                input_ids = torch.tensor(example["input_ids"]).unsqueeze(0).to(model.device)
                attention_mask = torch.tensor(example["attention_mask"]).unsqueeze(0).to(model.device)

                with torch.no_grad():
                    outputs = model.generate(input_ids,
                                             attention_mask=attention_mask,
                                             use_cache=not self.is_deepspeed_zero3,
                                             synced_gpus=self.is_deepspeed_zero3,
                                             max_new_tokens=self.max_new_tokens)

                prompt_text = self.tokenizer.decode(input_ids.squeeze(), skip_special_tokens=False)
                completion_text = self.tokenizer.decode(outputs.squeeze(), skip_special_tokens=False)[len(prompt_text):]

                pred_dict = {"Prompt": prompt_text, "Completion": completion_text}
                if 'chosen' in example.keys():
                    pred_dict['Chosen'] = example['chosen'][0]['content']
                if 'rejected' in example.keys():
                    pred_dict['Rejected'] = example['rejected'][0]['content']
                records.append(pred_dict)

        # Сохраняем на каждом процессе свою версию
        save_dataframe(pd.DataFrame(records), PartialState().process_index, self.temp_dir_path)

        PartialState().wait_for_everyone()

        # печатаем и логируем только на основном потоке
        if PartialState().is_main_process:

            # Читаем все файлы DataFrame из временной папки и объединяем их
            all_files = glob.glob(os.path.join(self.temp_dir_path, 'df_*.csv'))
            df_list = [pd.read_csv(file) for file in all_files]
            combined_df = pd.concat(df_list, ignore_index=True)

            # После объединения можно удалять временную папку
            shutil.rmtree(self.temp_dir_path)

            if self.logger_backend == 'clearml':
                task = Task.current_task()
                if task:
                    logger = task.get_logger()
                    logger.report_table("Generated Text Samples", "DataFrame", iteration=state.global_step, table_plot=combined_df)
            elif self.logger_backend == 'wandb':
                wandb.log({f"eval/generated_text_{state.global_step}": wandb.Table(dataframe=combined_df)})

            pretty_print_dataframe(combined_df.sample(3))
