# CUDA_VISIBLE_DEVICES=2 accelerate launch --main_process_port 29501 --config_file accelerate/stage2_config.yaml scripts/vlm.py --model_id "Qwen/Qwen2-VL-2B-Instruct" --data_file "llava_instruct_150k_russian.json" --image_dir "/mnt/storage/lvl/images/" --output_dir "/mnt/storage/lvl/vikhr-Qwen-VL-2b-25.10.24" --epochs 3 --batch_size 4 --learning_rate 2e-4 --fp16

import os
import torch
from datasets import load_dataset
from transformers import AutoModelForVision2Seq, AutoProcessor, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from accelerate import Accelerator
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Distributed Training with Accelerate")
    
    # Paths and model configs
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="Hugging Face model ID")
    parser.add_argument("--data_file", type=str, default="llava_instruct_150k_russian.json", help="Path to JSON dataset")
    parser.add_argument("--image_dir", type=str, default="/mnt/storage/lvl/images/", help="Directory containing images")
    parser.add_argument("--output_dir", type=str, default="/mnt/storage/lvl/vikhr-Qwen-VL-2b-25.10.24", help="Output directory for the model")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--grad_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--bf16", action="store_true", help="Use bf16 precision")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 precision")

    return parser.parse_args()

def format_data(sample, image_dir):
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}],
        }
    ]
    
    for msg in sample["conversations"]:
        content = [{"type": "text", "text": msg["value"].replace("<image>\n", "")}]
        if "<image>" in msg["value"]:
            content.append({"type": "image", "image": os.path.join(image_dir, sample['image'])})
        
        role = "user" if msg["from"] == "human" else "assistant"
        messages.append({"role": role, "content": content})

    return {"messages": messages}

def collate_fn(examples, processor):
    texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
    image_inputs = [process_vision_info(example["messages"])[0] for example in examples]
    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    
    image_tokens = [151652, 151653, 151655] if isinstance(processor, Qwen2VLProcessor) else [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100
    batch["labels"] = labels

    return batch

def main():
    args = parse_args()
    accelerator = Accelerator(mixed_precision="bf16" if args.bf16 else "fp16" if args.fp16 else "no")

    dataset = load_dataset("json", data_files=args.data_file)
    dataset = [format_data(sample, args.image_dir) for sample in dataset['train']]
    
    model = AutoModelForVision2Seq.from_pretrained(args.model_id, torch_dtype=torch.float16)
    processor = AutoProcessor.from_pretrained(args.model_id)
    
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=64,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accumulation_steps,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        logging_steps=5,
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True}
    )
    sft_config.remove_unused_columns = False

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        data_collator=lambda examples: collate_fn(examples, processor),
        dataset_text_field="",
        peft_config=peft_config,
        tokenizer=processor.tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    
    del model
    del trainer
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
