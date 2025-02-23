# Run this script with accelerate:
# accelerate launch --config_file accelerate/default_config.yaml scripts/distill.py --dataset_train_path from_dvc/ru_openhermes_translated_jsonl/dataset/train.jsonl --dataset_test_path from_dvc/ru_openhermes_translated_jsonl/dataset/test.jsonl --teacher_model from_s3/llama3_70b_instruct/ --student_model from_s3/llama3_8b_instruct/ --max_length 2048 --distill_loss kl_divergence --apply_hard_labels_coef --alpha 2.0 --run_name run_kl_div_hlc_a2.0_lora_16_all_max2048 --grad_ch

import argparse
import os
import random
import uuid
from collections import defaultdict
from typing import Literal, Dict

import torch
from accelerate.logging import get_logger
from accelerate.state import PartialState
from datasets import load_dataset, DatasetDict
from peft import LoraConfig, get_peft_model
from torch.nn.functional import kl_div, softmax, log_softmax, mse_loss, cosine_similarity, one_hot
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, \
    DataCollatorForLanguageModeling

from src.utils.model_preparation import prepare_ref_model_for_deepspeed

logger = get_logger(__name__)

LOGGING_TASK_NAME = str(uuid.uuid4())

os.environ['WANDB_RUN_ID'] = str(random.randint(100000, 999999))
os.environ['WANDB_NAME'] = LOGGING_TASK_NAME
os.environ['CLEARML_TASK'] = LOGGING_TASK_NAME
os.environ["WANDB_PROJECT"] = "llm_distillation"
os.environ['CLEARML_PROJECT'] = "llm_distillation"


def parse_args():
    parser = argparse.ArgumentParser(description='KL Distillation with LoRA and Accelerate')
    parser.add_argument('--dataset_train_path', required=True, type=str, help='Path to the train JSONL dataset')
    parser.add_argument('--dataset_test_path', required=True, type=str, help='Path to the test JSONL dataset')
    parser.add_argument('--teacher_model', required=True, type=str, help='Name or path of the teacher model')
    parser.add_argument('--student_model', required=True, type=str, help='Name or path of the student model')
    parser.add_argument('--run_name', type=str, default=os.environ['WANDB_NAME'], help='Name of current run')
    parser.add_argument('--distill_loss', type=str, default='kl_divergence', help='Name or path of the student model')
    parser.add_argument('--output_dir', type=str, default='train_output',
                        help='Directory to save the trained model and logs')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for KL divergence loss')
    parser.add_argument('--alpha', type=float, default=1.0, help='Weight for KL divergence loss')
    parser.add_argument('--lora_r', type=int, default=16, help='LoRA rank')
    parser.add_argument('--max_length', type=int, default=2048, help='Max length of text in tokens')
    parser.add_argument('--grad_ch', action='store_true', help='Use or not grad. checkpointing')
    parser.add_argument('--apply_hard_labels_coef', action='store_true', help='Apply hard labels coefficient to loss')
    parser.add_argument('--dont_learn_clm', action='store_true', help='Do not add aux. CLM (SFT) loss')
    return parser.parse_args()


def apply_attention_mask(loss, attention_mask):
    mask = attention_mask.float()
    loss = loss.sum(-1) * mask  # Get loss sum by attending tokens -> [BS, Seqlen]
    loss = loss.sum(-1) / torch.clamp(mask.sum(-1), min=1e-9)  # Normalize by the number of non-masked tokens
    return loss.mean()  # mean by batch


def apply_hard_labels_mask(loss, hard_labels):
    """
    Применяет labels mask где лейбл = -100 к лоссу или чему угодно с шейпом [batch_size, sequence_length, dim]
    """
    mask = hard_labels != -100
    loss = loss.sum(-1) * mask  # Get loss sum by attending tokens -> [BS, Seqlen]
    loss = loss.sum(-1) / torch.clamp(mask.sum(-1), min=1e-9)  # Normalize by the number of non-masked tokens
    return loss.mean()  # mean by batch


def hard_labels_coefficient(student_logits, teacher_logits, hard_labels):
    # Calculate the probabilities using softmax
    student_probs = torch.softmax(student_logits, dim=-1)
    teacher_probs = torch.softmax(teacher_logits, dim=-1)

    # Calculate the coefficients
    student_coef = 1 - student_probs.gather(-1, hard_labels.unsqueeze(-1))
    teacher_coef = teacher_probs.gather(-1, hard_labels.unsqueeze(-1))
    coef = student_coef * teacher_coef

    return coef


def kl_divergence_loss(student_logits, teacher_logits, temperature):
    student_logprobs = log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = softmax(teacher_logits / temperature, dim=-1)
    return kl_div(student_logprobs, teacher_probs, reduction='none', log_target=False) * (temperature ** 2)


def mse_loss_fn(student_logits, teacher_logits):
    return mse_loss(student_logits, teacher_logits, reduction='none')


def soft_target_cross_entropy_loss(student_logits, teacher_logits, temperature):
    teacher_probs = softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = log_softmax(student_logits / temperature, dim=-1)
    return -(teacher_probs * student_log_probs)


def slim_loss(student_logits, teacher_logits, temperature, hard_labels):
    student_probs, teacher_probs = softmax(student_logits, dim=-1), softmax(teacher_logits, dim=-1)
    kd_loss = soft_target_cross_entropy_loss(student_logits, teacher_logits, temperature)
    hard_labels = one_hot(hard_labels, num_classes=student_logits.size(-1)).to(device=student_logits.device,
                                                                               dtype=student_logits.dtype)
    filtered_student_probs = hard_labels * student_probs
    filtered_teacher_probs = hard_labels * teacher_probs
    diff = filtered_teacher_probs / torch.clamp(filtered_student_probs, min=1e-9)
    coef = 1 - torch.exp(-diff)
    return coef * kd_loss


def cosine_similarity_loss(student_logits, teacher_logits):
    return 1 - cosine_similarity(student_logits, teacher_logits, dim=-1)


def jensen_shannon_divergence(student_logits, teacher_logits, temperature):
    teacher_probs = softmax(teacher_logits / temperature, dim=-1)
    student_probs = softmax(student_logits / temperature, dim=-1)
    m = 0.5 * (teacher_probs + student_probs)
    jsd = 0.5 * (kl_div(torch.log(student_probs), m, reduction='none') + kl_div(torch.log(teacher_probs), m,
                                                                                reduction='none'))
    return jsd


def earth_mover_distance(student_logits, teacher_logits):
    student_probs = softmax(student_logits, dim=-1)
    teacher_probs = softmax(teacher_logits, dim=-1)
    return torch.cdist(student_probs, teacher_probs, p=1)


def alpha_beta_divergence_loss(student_logits, teacher_logits, alpha=1.0, beta=2.0):
    student_probs = softmax(student_logits, dim=-1)
    teacher_probs = softmax(teacher_logits, dim=-1)

    if alpha == beta:
        divergence = (1 / alpha) * (torch.sum(teacher_probs ** alpha) - 1)
    else:
        divergence = (1 / (alpha * (beta - alpha))) * (
                    torch.sum(teacher_probs ** alpha) - torch.sum(teacher_probs ** beta))

    return divergence


def main():
    args = parse_args()

    # Load the dataset using datasets
    train_dataset = load_dataset('json', data_files=args.dataset_train_path)['train']
    test_dataset = load_dataset('json', data_files=args.dataset_test_path)['train']
    dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

    # Load the tokenizer and teacher model
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    tokenizer.pad_token = "<|reserved_special_token_0|>"

    # Tokenize the dataset
    def tokenize_function(example):
        formatted_example = tokenizer.apply_chat_template(example['messages'], tokenize=False)
        return tokenizer(formatted_example, truncation=True, padding=True, max_length=args.max_length,
                         add_special_tokens=False)

    with PartialState().main_process_first():
        tokenized_dataset = dataset.map(tokenize_function,
                                        batched=False,
                                        remove_columns=dataset.column_names['train'],
                                        num_proc=10)

    class DistillationTrainer(Trainer):
        def __init__(self, teacher_model, distill_loss, temperature, alpha, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.teacher_model = prepare_ref_model_for_deepspeed(teacher_model, self.accelerator)
            self.temperature = temperature
            self.alpha = alpha
            self.distillation_loss_fn = self.select_distillation_loss(distill_loss)
            self._stored_metrics = defaultdict(lambda: defaultdict(list))

        def select_distillation_loss(self, loss_type):
            distillation_losses = {
                "kl_divergence": kl_divergence_loss,
                "mse": mse_loss_fn,
                "soft_cross_entropy": soft_target_cross_entropy_loss,
                "cosine_similarity": cosine_similarity_loss,
                "jensen_shannon": jensen_shannon_divergence,
                "earth_mover_distance": earth_mover_distance,
                "alpha_beta_divergence": alpha_beta_divergence_loss,
                "slim": slim_loss
            }
            if loss_type not in distillation_losses:
                raise ValueError(f"Unsupported distillation loss type: {loss_type}")
            return distillation_losses[loss_type]

        def compute_loss(self, model, inputs, return_outputs=False):
            attention_mask = inputs['attention_mask'][..., 1:].contiguous()
            hard_labels = inputs['labels'][..., 1:].contiguous()
            kd_coef = None

            student_outputs = model(**inputs)
            student_logits = student_outputs.logits[..., :-1, :].contiguous()

            with torch.no_grad():
                teacher_outputs = self.teacher_model(**inputs)
                teacher_logits = teacher_outputs.logits[..., :-1, :].contiguous()

            # Calculate the distillation loss
            if self.distillation_loss_fn in [kl_divergence_loss,
                                             soft_target_cross_entropy_loss,
                                             jensen_shannon_divergence]:
                distillation_loss = self.distillation_loss_fn(student_logits, teacher_logits, self.temperature)
            elif self.distillation_loss_fn in [slim_loss]:
                distillation_loss = self.distillation_loss_fn(student_logits, teacher_logits, self.temperature,
                                                              hard_labels)
            else:
                distillation_loss = self.distillation_loss_fn(student_logits, teacher_logits)

            # Apply hard labels coefficient to loss if needed
            if args.apply_hard_labels_coef and not self.distillation_loss_fn in [slim_loss]:
                kd_coef = hard_labels_coefficient(student_logits, teacher_logits, self.temperature, hard_labels)
                distillation_loss = kd_coef * distillation_loss

            # We need to calculate distill losses only on non -100 labels
            distillation_loss = apply_hard_labels_mask(distillation_loss, hard_labels)

            # We need to calculate distill losses only on attending tokens
            distillation_loss = apply_attention_mask(distillation_loss, attention_mask)

            # CLM loss - already computed by model
            sft_loss = student_outputs.loss

            # Total loss computation
            # loss = self.alpha * distillation_loss + (1 - self.alpha) * sft_loss
            if args.dont_learn_clm:
                loss = self.alpha * distillation_loss
            else:
                loss = self.alpha * distillation_loss + sft_loss

            metrics = {
                "distillation_loss": distillation_loss.cpu(),
                "sft_loss": sft_loss.cpu()
            }
            if kd_coef is not None:
                kd_coef = apply_attention_mask(kd_coef, attention_mask)  # for logging
                metrics['kd_coef'] = kd_coef.cpu()

            self.store_metrics(metrics, train_eval='train' if self.model.training else 'eval')

            if return_outputs:
                return loss, student_outputs
            return loss

        def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
            for key, value in metrics.items():
                self._stored_metrics[train_eval][key].append(value)

        def log(self, logs: Dict[str, float]) -> None:
            """
            Log `logs` on the various objects watching training, including stored metrics.

            Args:
                logs (`Dict[str, float]`):
                    The values to log.
            """
            # logs either has 'loss' or 'eval_loss'
            train_eval = "train" if "loss" in logs else "eval"
            # Add averaged stored metrics to logs
            for key, metrics in self._stored_metrics[train_eval].items():
                logs[key] = torch.tensor(metrics).mean().item()
            del self._stored_metrics[train_eval]
            return super().log(logs)

        # def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        #     self.model = self.model.eval()
        #     eval_dataloader = self.get_eval_dataloader(eval_dataset)
        #     generation_results = []

        #     for step, inputs in enumerate(eval_dataloader):
        #         inputs = self._prepare_inputs(inputs)
        #         print(inputs['input_ids'].shape)

        #         # Sample a few inputs to generate text
        #         if step < 5:  # Example: limit to first 5 steps for demonstration
        #             input_ids = inputs['input_ids'][:1]  # Take one example
        #             generated_ids = self.model.generate(input_ids=input_ids, do_sample=False, max_new_tokens=128)
        #             generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        #             generation_results.append(generated_text)

        #             # Log generated text to wandb
        #             self.log({f"generated_text_step_{step}": generated_text})

        #     metrics = super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

        #     return metrics

    # Set up the data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Set up the training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, args.run_name),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        gradient_checkpointing=args.grad_ch,
        learning_rate=8e-5,
        weight_decay=0.01,
        save_steps=500,
        save_strategy='steps',
        evaluation_strategy='steps',
        eval_steps=2000,
        run_name=args.run_name,
        report_to='clearml',
        logging_steps=1,
        logging_first_step=True,
        warmup_steps=10,
        bf16=True,
        dataloader_num_workers=2
    )

    # Load the student model
    student_model = AutoModelForCausalLM.from_pretrained(args.student_model,
                                                         torch_dtype=torch.bfloat16,
                                                         attn_implementation='sdpa')

    # Prepare the student model for LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r,
        target_modules=['embed_tokens',
                        'q_proj', 'v_proj', 'k_proj', 'o_proj', 'down_proj', 'up_proj', 'gate_proj',
                        'lm_head'],
        # modules_to_save=['embed_tokens', 'lm_head'],
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM'
    )
    student_model = get_peft_model(student_model, lora_config)
    student_model.print_trainable_parameters()

    # Load the teacher model
    teacher_model = AutoModelForCausalLM.from_pretrained(args.teacher_model,
                                                         torch_dtype=torch.bfloat16,
                                                         attn_implementation='sdpa')

    PartialState().wait_for_everyone()

    # Create the trainer
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        distill_loss=args.distill_loss,
        temperature=args.temperature,
        alpha=args.alpha,
        model=student_model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    # Start training
    trainer.train()

    # Save the fine-tuned model
    trainer.save_model(args.output_dir)


if __name__ == '__main__':
    main()
