import os
import asyncio
import argparse
import json
import pandas as pd
import numpy as np
import torch
from openai import AsyncOpenAI, AsyncAzureOpenAI
from openai.types.chat import ChatCompletion
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

parser = argparse.ArgumentParser(description='Асинхронная генерация ответов через OpenAI API с Rejection Sampling через RM.')
parser.add_argument('--openai_api_key', type=str, default=os.environ.get("OPENAI_API_KEY"), help='API ключ')
parser.add_argument('--api_type', type=str, default="openai", help='Тип API - openai или azure')
parser.add_argument('--output_folder', type=str, default="data", help='Название папки для сохранения результатов')
parser.add_argument('--openai_base_url', type=str, default="http://localhost:8000/v1", help='Базовый URL для API')
parser.add_argument('--model_name', type=str, required=True, help='Название модели для API запросов')
parser.add_argument('--global_system_prompt', type=str, required=False, help='Глобальный системный промпт для генерации')
parser.add_argument('--local_system_prompt_field', type=str, default="system_prompt", help='Название поля с локальным системным промптом')
parser.add_argument('--id_field', type=str, default="id", help='Название поля содержащее id промпта')
parser.add_argument('--prompt_field', type=str, default="prompt", help='Название поля содержащее промпт в виде [{...}]')
parser.add_argument('--follow_up_prompt_field', type=str, default="follow_up_prompt", help='Название поля содержащее follow-up промпт')
parser.add_argument('--correct_answer_field', type=str, default="correct_answer", help='Название поля содержащее правильный ответ (в любой форме)')
parser.add_argument('--n_parallel', type=int, default=4, help='Количество параллельных запросов')
parser.add_argument('--temperature', type=float, default=0.8, help='Температура для генерации')
parser.add_argument('--max_gen_tokens', type=int, default=3072, help='Максимальное количество токенов для генерации')
parser.add_argument('--prompts_source', type=str, required=True, help='Путь к JSONL файлу с запросами для модели в формате OpenAI')
parser.add_argument('--n_hypos', type=int, default=5, help='Количество генераций на один промпт')
parser.add_argument('--rm_model_path', type=str, required=True, help='Путь к Reward Model')
parser.add_argument('--rm_model_atten_impl', type=str, default='sdpa', help='Имплементация attention для RM')
parser.add_argument('--rm_max_seq_len', type=int, default=16000, help='Максимальная длина последовательности для RM')
parser.add_argument('--rm_max_batch_size', type=int, default=8, help='Максимальный batch size для RM')
parser.add_argument('--rm_device', type=str, default='cuda:0', help='Устройство для RM (cuda/cpu)')

args = parser.parse_args()

# Инициализируем клиента с указанными параметрами
if args.api_type == 'openai':
    client = AsyncOpenAI(
        api_key=args.openai_api_key,
        base_url=args.openai_base_url
    )
elif args.api_type == 'azure':
    client = AsyncAzureOpenAI(
        api_key=args.openai_api_key,
        azure_endpoint=args.openai_base_url,
        api_version="2024-08-01-preview"
    )

# Загрузка Reward модели
print('Loading Reward Model...')
rm_tokenizer = AutoTokenizer.from_pretrained(args.rm_model_path)
rm_tokenizer.model_max_length = args.rm_max_seq_len

rm_device = torch.device(args.rm_device)
rm_model = AutoModelForSequenceClassification.from_pretrained(
    args.rm_model_path, 
    attn_implementation=args.rm_model_atten_impl
)
rm_model.to(dtype=torch.float16, device=rm_device).eval()

# Очереди для асинхронной обработки
scoring_queue = asyncio.Queue()
result_queue = asyncio.Queue()

# Контроль паралеллизма
tp_executor = ThreadPoolExecutor(max_workers=1)
semaphore = asyncio.Semaphore(args.n_parallel)
write_lock = asyncio.Lock()

# Файл для сохранения результатов
output_file = f'{args.output_folder}/{args.prompts_source.split("/")[-1].split(".")[-2]}_rs.jsonl'
if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)


def score_generations(generated_conversations, correct_answer):
    scores = []
    
    # Преобразуем все разговоры в текст
    def transform_conv(conv):
        if correct_answer is None:
            return conv
        else:
            conv = [{'role': 'system', 'content': f'The correct final answer must be: {correct_answer}'}] + conv
            return conv
    
    all_texts = [
        rm_tokenizer.apply_chat_template(transform_conv(conv), tokenize=False) 
        for conv in generated_conversations
    ]
    
    # Обрабатываем батчами
    for i in range(0, len(all_texts), args.rm_max_batch_size):
        batch_texts = all_texts[i:i + args.rm_max_batch_size]
        batch = rm_tokenizer.batch_encode_plus(
            batch_texts,
            return_tensors='pt',
            padding=True,
            truncation=True
        )
        batch = {k: v.to(rm_device) for k, v in batch.items()}

        with torch.inference_mode():
            batch_scores = rm_model(**batch).logits.detach().cpu().view(-1).numpy()
            scores.extend(batch_scores)

    return np.array(scores).astype(float)


async def generate_hypotheses(row: pd.Series):
    async with semaphore:
        try:
            # Определяем системный промпт
            system_prompt = None
            if args.local_system_prompt_field in row and pd.notna(row[args.local_system_prompt_field]):
                system_prompt = row[args.local_system_prompt_field]
            elif args.global_system_prompt:
                system_prompt = args.global_system_prompt

            # Формируем базовый промпт
            if system_prompt:
                base_prompt = [{'role': 'system', 'content': system_prompt}] + row[args.prompt_field]
            else:
                base_prompt = row[args.prompt_field]

            response_format = {'type': 'text'} if 'response_format' not in row else row['response_format']

            # Если есть follow-up промпт
            if args.follow_up_prompt_field in row and pd.notna(row[args.follow_up_prompt_field]):
                completion = await client.chat.completions.create(
                    messages=base_prompt,
                    model=args.model_name,
                    temperature=args.temperature,
                    response_format=response_format,
                    max_tokens=args.max_gen_tokens
                )

                answer = completion.choices[0].message.model_dump(
                    exclude={"function_call", "tool_calls", "refusal", "audio"}
                )
                base_prompt = base_prompt + [answer] + row[args.follow_up_prompt_field]

            # Генерируем n_hypos вариантов
            generated_conversations = []
            for _ in range(args.n_hypos):
                completion = await client.chat.completions.create(
                    messages=base_prompt,
                    model=args.model_name,
                    temperature=args.temperature,
                    response_format=response_format,
                    max_tokens=args.max_gen_tokens
                )
                if args.api_type == 'eliza':
                    completion = ChatCompletion(**completion.response)

                answer = completion.choices[0].message.model_dump(
                    exclude={"function_call", "tool_calls", "refusal", "audio"}
                )
                current_conversation = base_prompt + [answer]

                generated_conversations.append(current_conversation)

            # Отправляем на оценку
            await scoring_queue.put((row, base_prompt, generated_conversations))

        except Exception as e:
            print(f'Generation error: {e}')


async def score_and_select():
    while True:
        try:
            row, base_prompt, generated_conversations = await scoring_queue.get()
            
            # Запускаем scoring в отдельном потоке через ThreadPoolExecutor
            scores = await asyncio.get_event_loop().run_in_executor(
                tp_executor,
                score_generations,
                generated_conversations,
                row[args.correct_answer_field] if args.correct_answer_field in row else None
            )

            # Находим лучшую и худшую генерации
            best_idx = scores.argmax()
            worst_idx = scores.argmin()

            result = {
                args.id_field: row[args.id_field],
                'prompt': base_prompt,
                'chosen': [generated_conversations[best_idx][-1]],
                'chosen_score': float(scores[best_idx]),
                'rejected': [generated_conversations[worst_idx][-1]],
                'rejected_score': float(scores[worst_idx]),
                'all_generations': [conv[-1] for conv in generated_conversations],
                'all_scores': [float(score) for score in scores],
                'gen_model': args.model_name,
                'rm_model': args.rm_model_path
            }
            if args.correct_answer_field in row:
                result['target_answer'] = row[args.correct_answer_field ]

            await result_queue.put(result)
        except Exception as e:
            print(f'Scoring error: {e}')
        finally:
            scoring_queue.task_done()


async def save_results():
    while True:
        try:
            result = await result_queue.get()
            async with write_lock:
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f'Saving error: {e}')
        finally:
            result_queue.task_done()


async def main():
    try:
        # Чтение файла с запросами
        print('Loading Prompts Source file...')
        
        df = pd.read_json(args.prompts_source, lines=True)
        if args.prompt_field not in df or args.id_field not in df:
            raise ValueError(f'Файл с запросами должен содержать колонки {args.prompt_field} и {args.id_field}!')
    
        # Проверка уже обработанных промптов
        if os.path.exists(output_file):
            df_existing_responses = pd.read_json(output_file, lines=True)
            processed_prompts_ids = set(df_existing_responses[args.id_field])
            print(f"Skipping {len(processed_prompts_ids)} already completed prompts...")
        else:
            processed_prompts_ids = set()
    
        filtered_prompts = df[~df[args.id_field].isin(processed_prompts_ids)]

        print('Starting generation...')
    
        # Запускаем воркеры
        scoring_worker = asyncio.create_task(score_and_select())
        saving_worker = asyncio.create_task(save_results())
    
        # Генерируем ответы
        generation_tasks = [generate_hypotheses(row) for _, row in filtered_prompts.iterrows()]
        for f in tqdm(asyncio.as_completed(generation_tasks), total=len(generation_tasks)):
            await f

        # for _, row in tqdm(filtered_prompts.iterrows(), total=len(filtered_prompts)):
        #     await generate_hypotheses(row)
    
        # Ждем завершения всех задач
        await scoring_queue.join()
        await result_queue.join()
    
        # Останавливаем воркеры
        scoring_worker.cancel()
        saving_worker.cancel()
    finally:
        tp_executor.shutdown()


if __name__ == '__main__':
    asyncio.run(main())