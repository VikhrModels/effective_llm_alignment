#
# Python script to start Gradio chat application for using served models using OpenAI API
# VLLM: python -m vllm.entrypoints.openai.api_server --model NousResearch/Meta-Llama-3-8B-Instruct --served-model-name custom_model -tp 1 --max-model-len 8000 --dtype auto --api-key secret_token_228 --host localhost
#

import argparse

import gradio as gr
from openai import OpenAI

# Argument parser setup
parser = argparse.ArgumentParser(
    description='Chatbot Interface with Customizable Parameters')
parser.add_argument('--model-url',
                    type=str,
                    default='http://localhost:8000/v1',
                    help='Model URL')
parser.add_argument('--api-key',
                    type=str,
                    default='secret_token_228',
                    help='api key')
parser.add_argument('-m',
                    '--model',
                    type=str,
                    default='custom_model',
                    help='Model name for the chatbot')
parser.add_argument('--stop-token-ids',
                    type=str,
                    default='',
                    help='Comma-separated stop token IDs')
parser.add_argument("--host", type=str, default=None)
parser.add_argument("--port", type=int, default=8730)
parser.add_argument("--share", type=bool, default=True)

# Parse the arguments
args = parser.parse_args()

# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = args.api_key
openai_api_base = args.model_url

# Create an OpenAI client to interact with the API server
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)


def predict(message: str, history: list, system_prompt: str, temp: float, top_p: float, top_k: int):
    # Convert chat history to OpenAI format
    history_openai_format = []
    if system_prompt:
        history_openai_format.append({"role": "system", "content": system_prompt})
    for human, assistant in history:
        history_openai_format.append({"role": "user", "content": human})
        history_openai_format.append({
            "role": "assistant",
            "content": assistant
        })
    history_openai_format.append({"role": "user", "content": message})

    # Create a chat completion request and send it to the API server
    stream = client.chat.completions.create(
        model=args.model,  # Model name to use
        messages=history_openai_format,  # Chat history
        temperature=temp,  # Temperature for text generation
        top_p=top_p,
        stream=True,  # Stream response
        extra_body={
            'top_k': top_k,
            'repetition_penalty': 1,
            'stop_token_ids': [
                int(id.strip()) for id in args.stop_token_ids.split(',') if id.strip()
            ] if args.stop_token_ids else []
        })

    # Read and return generated text from response stream
    partial_message = ""
    for chunk in stream:
        partial_message += (chunk.choices[0].delta.content or "")
        yield partial_message


# Create and launch a chat interface with Gradio
with gr.Blocks() as demo:
    gr.ChatInterface(
        predict,
        fill_height=False,
        additional_inputs=[
            gr.Textbox(label="System prompt", max_lines=2, render=False),
            gr.Slider(0, 1, step=0.1, label="Temperature", value=1.0, render=False),
            gr.Slider(0.3, 1, step=0.1, label="Top P", value=1.0, render=False),
            gr.Slider(10, 100, step=10, label="Top K", value=50, render=False)
        ],
        title='Custom model vLLM test'
    )

demo.queue().launch(
    server_name=args.host,
    server_port=args.port,
    share=args.share
)
