import argparse
import json
import os
import pandas as pd
from tqdm import tqdm
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset


def main(args):
    # Проверка доступности устройства
    device = torch.device(f"cuda:{args.rm_gpu_index}" if torch.cuda.is_available() and args.rm_gpu_index else "cpu")

    # Загрузка LLM модели и токенизатора
    llm = LLM(model=args.llm_model_path, dtype='half', max_model_len=args.max_seq_len, tensor_parallel_size=args.llm_tp)
    llm_tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path)
    llm_tokenizer.model_max_length = args.max_seq_len

    # Загрузка RM модели и токенизатора с использованием SDPA
    rm_model = AutoModelForSequenceClassification.from_pretrained(args.rm_model_path, attn_implementation=args.rm_model_atten_impl)
    rm_tokenizer = AutoTokenizer.from_pretrained(args.rm_model_path)
    rm_tokenizer.model_max_length = args.max_seq_len
    rm_model.to(dtype=torch.float16, device=device).eval()

    # Загрузка входного датасета с помощью datasets
    dataset = load_dataset(args.input_dataset_path, split='train')
    print(dataset)

    # Проверка существования файла с результатами
    if os.path.exists(args.output_dataset_path):
        existing_df = pd.read_json(args.output_dataset_path, orient='records', lines=True)
        start_idx = len(existing_df) // args.save_n_worst
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
        conversation_without_last = conversation[:-1]

        # Генерация промптов для LLM
        prompt = llm_tokenizer.apply_chat_template(conversation_without_last, tokenize=False, add_generation_prompt=True)
        prompts = [prompt] * args.n_sampling_count

        # Параметры семплирования
        sampling_params = SamplingParams(temperature=args.sampling_temperature, max_tokens=2048)

        # Генерация ответов
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

        generated_conversations = []
        for output in outputs:
            generated_text = output.outputs[0].text.strip()
            new_conversation = conversation_without_last + [{'role': 'assistant', 'content': generated_text}]
            generated_conversations.append(new_conversation)

        # Оценка исходного диалога и сгенерированных диалогов с помощью RM модели
        original_conversation_text = rm_tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
        batch_texts = [original_conversation_text] + [
            rm_tokenizer.apply_chat_template(conv, tokenize=False) for conv in generated_conversations
        ]
        batch = rm_tokenizer.batch_encode_plus(batch_texts, return_tensors='pt', padding=True, truncation=True)
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.inference_mode():
            scores = rm_model(**batch).logits.detach().cpu().view(-1).numpy()

        scores = scores.astype(float)

        # Оценка исходного диалога
        original_rm_score = scores[0]

        # Сортировка сгенерированных диалогов по оценкам
        generated_scores = list(zip(generated_conversations, scores[1:]))
        generated_scores.sort(key=lambda x: x[1])

        # Сохранение результатов для save_n_worst худших генераций
        for i in range(args.save_n_worst):
            rejected_conversation, rejected_score = generated_scores[i]

            result = {
                'prompt': conversation_without_last,
                'chosen': [conversation[-1]],
                'chosen_score': original_rm_score,
                'rejected': [rejected_conversation[-1]],
                'rejected_score': rejected_score,
                'chosen_gen': [generated_scores[-1][0][-1]],
                'chosen_gen_score': generated_scores[-1][1]
            }
            results.append(result)

            # Сохранение результата в файл после обработки каждого диалога
            with open(args.output_dataset_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Sampling and RM Scoring Script")
    parser.add_argument("--llm_model_path", type=str, required=True, help="Path to the LLM model")
    parser.add_argument("--llm_tp", type=int, default=1, help="vLLM tensor parallel")
    parser.add_argument("--rm_model_path", type=str, required=True, help="Path to the RM model")
    parser.add_argument("--rm_model_atten_impl", type=str, default='eager', help="attn_implementation param")
    parser.add_argument("--rm_gpu_index", type=int, default=None, help="Index of RM GPU device, leave None to use CPU.")
    parser.add_argument("--max_seq_len", type=int, default=16000, help="VLLM max_model_len param")
    parser.add_argument("--input_dataset_path", type=str, required=True, help="Path to the input dataset (jsonl format)")
    parser.add_argument("--n_sampling_count", type=int, default=7, help="Number of samples to generate for each prompt")
    parser.add_argument("--conversation_column", type=str, default="conversation", help="Column name for conversations in the dataset")
    parser.add_argument("--sampling_temperature", type=float, default=0.8, help="Temperature for sampling")
    parser.add_argument("--output_dataset_path", type=str, required=True, help="Path to save the output dataset (jsonl format)")
    parser.add_argument("--save_n_worst", type=int, required=True, help="Number of worst generations to save")

    args = parser.parse_args()

    # Проверка, что save_n_worst меньше n_sampling_count
    if args.save_n_worst >= args.n_sampling_count:
        raise ValueError("save_n_worst must be less than n_sampling_count")

    main(args)
