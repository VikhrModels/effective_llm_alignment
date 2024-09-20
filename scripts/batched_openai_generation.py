import os
import asyncio
import argparse
import json
import pandas as pd
from openai import AsyncOpenAI, AsyncAzureOpenAI

from tqdm import tqdm


# Настраиваем парсер аргументов командной строки
parser = argparse.ArgumentParser(description='Асинхронная генерация ответов от модели OpenAI.')
parser.add_argument('--openai_api_key', type=str, default=os.environ.get("OPENAI_API_KEY"), help='API ключ для OpenAI')
parser.add_argument('--api_type', type=str, default="openai", help='Тип API - openai или azure')
parser.add_argument('--output_folder', type=str, default="data", help='Название папки для сохранения результатов')
parser.add_argument('--openai_base_url', type=str, default="http://localhost:8000/v1", help='Базовый URL для OpenAI API')
parser.add_argument('--model_name', type=str, required=True, help='Название модели для API запросов')
parser.add_argument('--system_prompt', type=str, required=False, help='Системный промпт для генерации')
parser.add_argument('--id_field', type=str, default="id", help='Название поля содержащее id промпта')
parser.add_argument('--prompt_field', type=str, default="prompt", help='Название поля содержащее промпт в виде [{...}]')
parser.add_argument('--n_parallel', type=int, default=4, help='Количество параллельных запросов')
parser.add_argument('--temperature', type=float, default=0.0, help='Температура для генерации')
parser.add_argument('--prompts_source', type=str, required=True, help='Путь к JSONL файлу с запросами для модели в формате OpenAI')
args = parser.parse_args()

# python batched_openai_generation.py --openai_api_key token-abc123 --model_name NousResearch/Meta-Llama-3-8B-Instruct --prompts_source lmsys-human-prompts.jsonl --n_parallel 6

# Инициализируем клиента с указанными параметрами
if args.api_type == 'openai':
    client = AsyncOpenAI(
        api_key=args.openai_api_key,
        base_url=args.openai_base_url
    )
else:
    client = AsyncAzureOpenAI(
        api_key=args.openai_api_key,
        azure_endpoint=args.openai_base_url,
        api_version="2024-02-15-preview"
    )

semaphore = asyncio.Semaphore(args.n_parallel)
write_lock = asyncio.Lock()

output_file = f'{args.output_folder}/{args.prompts_source.split("/")[-1].split(".")[-2]}_{args.model_name.split("/")[-1].lower()}_responses.jsonl'

if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)


async def write_to_file(record):
    async with write_lock:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')


async def generate_answer(row: pd.Series):
    async with semaphore:
        if args.system_prompt:
            generation_prompt = [{'role': 'system', 'content': args.system_prompt}] + row[args.prompt_field]
        else:
            generation_prompt = row[args.prompt_field]
        try:
            completion = await client.chat.completions.create(
                messages=generation_prompt,
                model=args.model_name,
                temperature=args.temperature
            )
            model_answer = [completion.choices[0].message.dict(exclude={"function_call", "tool_calls"})]
            
            record = {
                **row.to_dict(),
                "conversation": generation_prompt + model_answer
            }
            
            await write_to_file(record)
        except Exception as e:
            print(f'Error orccured: {e}')
            record = "$ERROR$"
        return record


async def main() -> None:
    # Чтение файла с запросами с помощью pandas
    df = pd.read_json(args.prompts_source, lines=True)
    if args.prompt_field not in df or args.id_field not in df:
        raise ValueError(f'Файл с запросами должен содержать колонки {args.prompt_field} и {args.id_field}!')

    if os.path.exists(output_file):
        df_existing_responses = pd.read_json(output_file, lines=True)
        processed_prompts_ids = set(df_existing_responses[args.id_field])
        print(f"Skipping {len(processed_prompts_ids)} already completed prompts...")
    else:
        processed_prompts_ids = set()

    filtered_prompts = df[~df[args.id_field].isin(processed_prompts_ids)]

    tasks = [generate_answer(row) for _, row in filtered_prompts.iterrows()]
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        await f


if __name__ == '__main__':
    asyncio.run(main())
  
