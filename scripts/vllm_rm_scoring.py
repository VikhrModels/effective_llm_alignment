import os
import asyncio
import argparse
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
import torch
import json
from openai import AsyncOpenAI, AsyncAzureOpenAI
from openai.types.chat import ChatCompletion
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

parser = argparse.ArgumentParser(description='Async generation and reward evaluation')
parser.add_argument('--openai_api_key', type=str, default=os.environ.get("OPENAI_API_KEY"))
parser.add_argument('--api_type', type=str, default="openai")
parser.add_argument('--output_folder', type=str, default="data")
parser.add_argument('--openai_base_url', type=str, default="http://localhost:8000/v1")
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--global_system_prompt', type=str, required=False)
parser.add_argument('--local_system_prompt_field', type=str, default="system_prompt")
parser.add_argument('--id_field', type=str, default="id")
parser.add_argument('--prompt_field', type=str, default="prompt")
parser.add_argument('--follow_up_prompt_field', type=str, default="follow_up_prompt")
parser.add_argument('--correct_answer_field', type=str, default="correct_answer")
parser.add_argument('--n_parallel', type=int, default=4)
parser.add_argument('--temperature', type=float, default=0.0)
parser.add_argument('--max_gen_tokens', type=int, default=3072)
parser.add_argument('--prompts_source', type=str, required=True)
parser.add_argument('--rm_model_path', type=str, required=True)
parser.add_argument('--rm_model_atten_impl', type=str, default='sdpa')
parser.add_argument('--rm_max_seq_len', type=int, default=16000)
parser.add_argument('--rm_device', type=str, default='cuda:0')

args = parser.parse_args()

# Initialize client
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

# Load Reward Model
print('Loading Reward Model...')
rm_tokenizer = AutoTokenizer.from_pretrained(args.rm_model_path)
rm_tokenizer.model_max_length = args.rm_max_seq_len

rm_device = torch.device(args.rm_device)
rm_model = AutoModelForSequenceClassification.from_pretrained(
    args.rm_model_path,
    attn_implementation=args.rm_model_atten_impl
)
rm_model.to(dtype=torch.float16, device=rm_device).eval()

semaphore = asyncio.Semaphore(args.n_parallel)
write_lock = asyncio.Lock()

tp_executor = ThreadPoolExecutor(max_workers=1)

_printable_model_name = args.model_name.replace('/', '__')
_printable_source_name = args.prompts_source.split("/")[-1].split(".")[-2]
output_file = f'{args.output_folder}/{_printable_source_name}_{_printable_model_name}_rm_scoring.jsonl'

if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)


async def write_to_file(record):
    async with write_lock:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            

def calculate_reward(conversation, correct_answer=None):
    if correct_answer is not None:
        conversation = [{'role': 'system', 'content': f'The correct final answer must be: {correct_answer}'}] + conversation
    
    text = rm_tokenizer.apply_chat_template(conversation, tokenize=False)
    inputs = rm_tokenizer(text, return_tensors='pt', truncation=True).to(rm_device)
    
    with torch.inference_mode():
        score = rm_model(**inputs).logits.detach().cpu().numpy()[0][0]
    
    return float(score)

    
async def generate_and_evaluate(row: pd.Series, responses, rewards):
    async with semaphore:
        try:
            # Determine system prompt
            system_prompt = None
            if args.local_system_prompt_field in row and pd.notna(row[args.local_system_prompt_field]):
                system_prompt = row[args.local_system_prompt_field]
            elif args.global_system_prompt:
                system_prompt = args.global_system_prompt

            # Form base prompt
            if system_prompt:
                base_prompt = [{'role': 'system', 'content': system_prompt}] + row[args.prompt_field]
            else:
                base_prompt = row[args.prompt_field]

            response_format = {'type': 'text'} if 'response_format' not in row else row['response_format']

            # Handle follow-up prompt if exists
            if args.follow_up_prompt_field in row and pd.notna(row[args.follow_up_prompt_field]):
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
                base_prompt = base_prompt + [answer] + row[args.follow_up_prompt_field]

            # Generate final response
            completion = await client.chat.completions.create(
                messages=base_prompt,
                model=args.model_name,
                temperature=args.temperature,
                response_format=response_format,
                max_tokens=args.max_gen_tokens
            )

            response = completion.choices[0].message.model_dump(
                exclude={"function_call", "tool_calls", "refusal", "audio"}
            )
            full_conversation = base_prompt + [response]
            
            # Calculate reward
            correct_answer = row.get(args.correct_answer_field)
            reward = await asyncio.get_event_loop().run_in_executor(
                tp_executor,
                calculate_reward,
                full_conversation,
                correct_answer
            )

            # Prepare record for saving
            record = {
                args.id_field: row[args.id_field],
                'rm_model': args.rm_model_path,
                'temperature': args.temperature,
                'response': response['content'],
                'reward': reward,
                'full_conversation': full_conversation
            }
            if args.correct_answer_field in row:
                record['target_answer'] = row[args.correct_answer_field]
            
            # Write to file
            await write_to_file(record)

            responses.append(response['content'])
            rewards.append(reward)

        except Exception as e:
            print(f'Generation error: {e}')


async def main():
    try:
        # Read prompts file
        print('Loading Prompts Source file...')
        df = pd.read_json(args.prompts_source, lines=True)
        
        if args.prompt_field not in df or args.id_field not in df:
            raise ValueError(f'Input file must contain {args.prompt_field} and {args.id_field} columns!')

        # Initialize responses and rewards lists
        responses = []
        rewards = []
        
        # Check for existing progress
        if os.path.exists(output_file):
            df_existing_responses = pd.read_json(output_file, lines=True)
            processed_prompts_ids = set(df_existing_responses[args.id_field])
            print(f"Skipping {len(processed_prompts_ids)} already completed prompts...")
            
            # Load existing responses and rewards
            responses.extend(df_existing_responses['response'].tolist())
            rewards.extend(df_existing_responses['reward'].tolist())
        else:
            processed_prompts_ids = set()

        filtered_prompts = df[~df[args.id_field].isin(processed_prompts_ids)]
        
        if len(filtered_prompts) > 0:
            print(f'Starting generation and evaluation for {len(filtered_prompts)} prompts...')
    
            # Generate and evaluate responses
            tasks = [generate_and_evaluate(row, responses, rewards) 
                    for _, row in filtered_prompts.iterrows()]
            for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
                await f

        # Initialize Rich Console
        console = Console()
        
        # Calculate statistics only if we have data
        if responses and rewards:
            response_lengths = [len(response) for response in responses]
            
            stats = {
                'Response Length': {
                    'Mean': np.mean(response_lengths),
                    'Std': np.std(response_lengths),
                    'Min': np.min(response_lengths),
                    'Max': np.max(response_lengths),
                    '3%': np.percentile(response_lengths, 3),
                    '10%': np.percentile(response_lengths, 10),
                    '25%': np.percentile(response_lengths, 25),
                    'Median': np.median(response_lengths),
                    '75%': np.percentile(response_lengths, 75),
                    '90%': np.percentile(response_lengths, 90),
                    '97%': np.percentile(response_lengths, 97)
                },
                'Reward': {
                    'Mean': np.mean(rewards),
                    'Std': np.std(rewards),
                    'Min': np.min(rewards),
                    'Max': np.max(rewards),
                    '3%': np.percentile(rewards, 3),
                    '10%': np.percentile(rewards, 10),
                    '25%': np.percentile(rewards, 25),
                    'Median': np.median(rewards),
                    '75%': np.percentile(rewards, 75),
                    '90%': np.percentile(rewards, 90),
                    '97%': np.percentile(rewards, 97)
                }
            }
        
            # Create a rich table
            table = Table(title=f"Generation Statistics: {args.model_name}", show_lines=True)
        
            # Add columns to the table
            table.add_column("Metric", justify="left", style="cyan", no_wrap=True)
            table.add_column("Response Length", justify="right", style="green")
            table.add_column("Reward", justify="right", style="magenta")
        
            # Add rows for each metric
            metrics = ['Mean', 'Std', 'Min', 'Max', '3%', '10%', '25%', 'Median', '75%', '90%', '97%']
            for metric in metrics:
                table.add_row(
                    metric,
                    f"{stats['Response Length'][metric]:.4f}",
                    f"{stats['Reward'][metric]:.4f}"
                )
        
            # Print the table using rich
            console.print(table)
        
        else:
            console.print("\n[bold red]No data to calculate statistics.[/bold red]")

    finally:
        tp_executor.shutdown()


if __name__ == '__main__':
    asyncio.run(main())
