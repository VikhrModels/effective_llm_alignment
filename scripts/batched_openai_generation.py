# python batched_openai_generation.py --openai_api_key token-abc123 --model_name NousResearch/Meta-Llama-3-8B-Instruct --prompts_source lmsys-human-prompts.jsonl --n_parallel 6

# Формат файла с запросами (args.prompts_source):
#
# Тип: jsonl
# Обязательные колонки:
#   * args.id_field - колонка с уникальными идентификаторами, желательно uuid
#   * args.prompt_field - колонка с диалогом в формате [{'role': ..., 'content': ...}], последняя роль должна быть не assistant
# Необзяательные колонки:
#   * args.local_system_prompt_field - колонка с системным промптом в формате обычной строки
#   * args.follow_up_prompt_field - колонка с диалогом для продолжения генерации после первого ответа, [{'role': ..., 'content': ...}], последняя роль должна быть не assistant
#
# Формат jsonl файла с результатами:
# Колонки:
#   * все колонки из args.prompts_source, кроме prompt_field, local_system_prompt_field и follow_up_prompt_field
#   * conversation - полный диалог, включая примененный системный промпт, реплики юзера (или другие из промптов) и сгенерированные ответы модели, формат [{'role': ..., 'content': ...}]
#   * generated_message_indices - индексы сгенерированных ответов в диалоге conversation, в формате list (одно значение в списке если нет follow_up_prompt_field)
#   * finish_reason - причина заверешения генерации ответа, в формате list (одно значение в списке если нет follow_up_prompt_field)
#   * prompt_tokens - количество токенов затраченное на промпты при генерации
#   * completion_tokens - количество токеноа затраченное на ответы при генерации

import os
import asyncio
import argparse
import json
import pandas as pd
from openai import AsyncOpenAI, AsyncAzureOpenAI
from openai.types.chat import ChatCompletion

from tqdm import tqdm


# Настраиваем парсер аргументов командной строки
parser = argparse.ArgumentParser(description='Асинхронная генерация ответов через OpenAI API')
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
parser.add_argument('--n_parallel', type=int, default=4, help='Количество параллельных запросов')
parser.add_argument('--temperature', type=float, default=0.0, help='Температура для генерации')
parser.add_argument('--max_gen_tokens', type=int, default=3072, help='Максимальное количество токенов для генерации')
parser.add_argument('--prompts_source', type=str, required=True, help='Путь к JSONL файлу с запросами для модели в формате OpenAI')
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

semaphore = asyncio.Semaphore(args.n_parallel)
write_lock = asyncio.Lock()

output_file = f'{args.output_folder}/{args.prompts_source.split("/")[-1].split(".")[-2]}_gen_responses.jsonl'

if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)


async def write_to_file(record):
    async with write_lock:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


async def generate_answer(row: pd.Series):
    async with semaphore:
        try:
            # Определяем системный промпт
            system_prompt = None
            if args.local_system_prompt_field in row and pd.notna(row[args.local_system_prompt_field]):
                system_prompt = row[args.local_system_prompt_field]
            elif args.global_system_prompt:
                system_prompt = args.global_system_prompt

            # Формируем промпт с учетом системного промпта
            if system_prompt:
                generation_prompt = [{'role': 'system', 'content': system_prompt}] + row[args.prompt_field]
            else:
                generation_prompt = row[args.prompt_field]

            response_format = {'type': 'text'} if 'response_format' not in row else row['response_format']
            completion = await client.chat.completions.create(
                messages=generation_prompt,
                model=args.model_name,
                temperature=args.temperature,
                response_format=response_format,
                max_tokens=args.max_gen_tokens
            )

            first_answer = completion.choices[0].message.model_dump(exclude={"function_call", "tool_calls", "refusal", "audio"})
            conversation = generation_prompt + [first_answer]
            generated_message_indices = [len(conversation) - 1]  # Индекс первого ответа модели
            
            # Создаем копию row и удаляем ненужные поля
            record_dict = row.to_dict()
            fields_to_remove = [args.prompt_field, args.local_system_prompt_field, args.follow_up_prompt_field]
            for field in fields_to_remove:
                record_dict.pop(field, None)

            record = {
                **record_dict,
                "model": args.model_name,
                "conversation": conversation,
                "generated_message_indices": generated_message_indices,
                "finish_reason": [completion.choices[0].finish_reason],
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens
            }

            # Если есть follow-up промпт, генерируем второй ответ
            if args.follow_up_prompt_field in row and pd.notna(row[args.follow_up_prompt_field]):
                follow_up_prompt = conversation + row[args.follow_up_prompt_field]

                follow_up_completion = await client.chat.completions.create(
                    messages=follow_up_prompt,
                    model=args.model_name,
                    temperature=args.temperature,
                    response_format=response_format,
                    max_tokens=args.max_gen_tokens
                )

                if args.api_type == 'eliza':
                    follow_up_completion = ChatCompletion(**follow_up_completion.response)

                follow_up_answer = follow_up_completion.choices[0].message.model_dump(
                    exclude={"function_call", "tool_calls", "refusal", "audio"}
                )

                # Обновляем conversation и индексы сгенерированных сообщений
                conversation = follow_up_prompt + [follow_up_answer]
                generated_message_indices = [
                    len(generation_prompt),  # Индекс первого ответа
                    len(conversation) - 1    # Индекс второго ответа
                ]

                # Обновляем запись
                record.update({
                    "conversation": conversation,
                    "generated_message_indices": generated_message_indices,
                    "finish_reason": [completion.choices[0].finish_reason, follow_up_completion.choices[0].finish_reason],
                    "prompt_tokens": completion.usage.prompt_tokens + follow_up_completion.usage.prompt_tokens,
                    "completion_tokens": completion.usage.completion_tokens + follow_up_completion.usage.completion_tokens
                })

            await write_to_file(record)
            
        except Exception as e:
            print(f'Error occurred: {e}')
            record = "$ERROR$"
        return record


async def main() -> None:
    # Чтение файла с запросами с помощью pandas
    print('Loading Prompts Source file...')
    
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

    print('Starting generation...')

    tasks = [generate_answer(row) for _, row in filtered_prompts.iterrows()]
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        await f


if __name__ == '__main__':
    asyncio.run(main())
