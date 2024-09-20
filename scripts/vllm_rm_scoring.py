import argparse
import json
import os

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from tqdm import tqdm
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset


def print_scores_table(results, llm_model_path, rm_model_path):
    """
    Prints a pretty compact table of RM scores statistics.

    Parameters:
    - results: List of dictionaries containing 'rm_score'.
    - model_path: String representing the model path.
    """
    all_scores = np.array([r['rm_score'] for r in results])
    all_lens = np.array([r['completion_len'] for r in results])

    # Calculate statistics
    
    average_len = all_lens.mean()
    l_q95 = np.quantile(all_lens, 0.95)
    
    average_score = all_scores.mean()
    median_score = np.median(all_scores)
    q25 = np.quantile(all_scores, 0.25)
    q75 = np.quantile(all_scores, 0.75)
    min_score = np.min(all_scores)
    max_score = np.max(all_scores)

    console = Console()

    table = Table(title='RM Answers Stats', show_header=True, show_lines=True, header_style="bold magenta")
    table.add_column("Metric", style="dim", width=25)
    table.add_column("Value", justify="left")

    table.add_row("LLM Model path", str(llm_model_path))
    table.add_row("RM Model path", str(rm_model_path))
    table.add_row("", "")
    
    table.add_row("95th percentile length", f"{l_q95:.2f}")
    table.add_row("Average answer length", f"{average_len:.2f}")
    table.add_row("", "")

    table.add_row("Minimum score", f"{min_score:.2f}")
    table.add_row("25th percentile", f"{q25:.2f}")
    table.add_row("Median (50th percentile)", f"{median_score:.2f}")
    table.add_row("75th percentile", f"{q75:.2f}")
    table.add_row("Maximum score", f"{max_score:.2f}")
    table.add_row("Average score", f"{average_score:.2f}")

    # Print the table
    console.print(table)


def main(args):
    # Проверка доступности устройства
    device = torch.device(f"cuda:{args.rm_gpu_index}" if torch.cuda.is_available() and args.rm_gpu_index else "cpu")

    # Загрузка LLM модели и токенизатора
    llm = LLM(model=args.llm_model_path, dtype='half', max_model_len=args.max_seq_len, tensor_parallel_size=args.llm_tp)
    llm_tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path)
    llm_tokenizer.model_max_length = args.max_seq_len

    # Загрузка RM модели и токенизатора с использованием SDPA
    rm_model = AutoModelForSequenceClassification.from_pretrained(args.rm_model_path,
                                                                  attn_implementation=args.rm_model_atten_impl)
    rm_tokenizer = AutoTokenizer.from_pretrained(args.rm_model_path)
    rm_tokenizer.model_max_length = args.max_seq_len
    rm_model.to(dtype=torch.float16, device=device).eval()
    # rm_model = torch.compile(rm_model)

    # Загрузка входного датасета с помощью datasets
    dataset = load_dataset(args.input_dataset_path, split='train')
    print(dataset)

    # Проверка существования файла с результатами
    if args.results_save_path is not None and os.path.exists(args.results_save_path):
        existing_df = pd.read_json(args.results_save_path, orient='records', lines=True)
        start_idx = len(existing_df)
        results = existing_df.to_dict(orient='records')
        print(f"Continuing from index {start_idx}")
    else:
        start_idx = 0
        results = []

    for idx, row in tqdm(enumerate(dataset), total=len(dataset)):
        if idx < start_idx:
            continue

        conversation = row[args.conversation_column]

        # Удаление последнего ответа ассистента
        if conversation[-1]['role'] == 'assistant':
            conversation = conversation[:-1]

        # Генерация промптов для LLM
        prompt = llm_tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

        # Параметры семплирования
        sampling_params = SamplingParams(temperature=args.sampling_temperature, max_tokens=2048)

        # Генерация ответов
        outputs = llm.generate([prompt], sampling_params, use_tqdm=False)[0]

        generated_text = outputs.outputs[0].text.strip()
        generated_conversation = conversation + [{'role': 'assistant', 'content': generated_text}]

        # Оценка исходного диалога и сгенерированных диалогов с помощью RM модели
        templated_conversation = rm_tokenizer.apply_chat_template(generated_conversation,
                                                                  tokenize=False,
                                                                  add_generation_prompt=False)
        input_tokens = rm_tokenizer(templated_conversation,
                                    return_tensors="pt", 
                                    truncation=True, 
                                    add_special_tokens=True).to(device)

        with torch.inference_mode():
            outputs = rm_model(**input_tokens)
            score = outputs.logits.item()

        result = {
            'conversation': generated_conversation,
            'rm_score': score,
            'completion_len': len(generated_text)
        }
        results.append(result)

        if args.results_save_path is not None:
            with open(args.results_save_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

    print_scores_table(results, args.llm_model_path, args.rm_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scoring of LLM generations using RM")
    parser.add_argument("--llm_model_path", type=str, required=True, help="Path to the LLM model")
    parser.add_argument("--llm_tp", type=int, default=1, help="vLLM tensor parallel")
    parser.add_argument("--rm_model_path", type=str, required=True, help="Path to the RM model")
    parser.add_argument("--rm_model_atten_impl", type=str, default='eager', help="attn_implementation param")
    parser.add_argument("--rm_gpu_index", type=int, default=None, help="Index of RM GPU device, leave None to use CPU.")
    parser.add_argument("--max_seq_len", type=int, default=16000, help="VLLM max_model_len param")
    parser.add_argument("--input_dataset_path", type=str, required=True,
                        help="Path to the input dataset (jsonl format)")
    parser.add_argument("--conversation_column", type=str, default="conversation",
                        help="Column name for conversations in the dataset")
    parser.add_argument("--sampling_temperature", type=float, default=0.0, help="Temperature for sampling")
    parser.add_argument("--results_save_path", type=str, required=False,
                        help="Path to save the output dataset (jsonl format)")

    args = parser.parse_args()

    main(args)
