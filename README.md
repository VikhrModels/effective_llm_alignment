# 🎉 Toolkit for Effective LLM Alignment

## ✨ Description and Features

This is a **super customizable**, **concise**, **user-friendly**, and **efficient toolkit** for training and aligning LLMs. To get started, simply define a YAML configuration with parameters from HF TrainingArguments and some specific parameters for each method.

### 🛠️ Toolkit Foundations

**Key Libraries:**
- **Core:** PyTorch, Transformers, TRL
- **Distributed Training:** Accelerate, FSDP, DeepSpeed (Zero 2/3)
- **Acceleration:** vLLM, Flash Attention, SDPA, Liger Kernel (for fused CrossEntropy in SFT)
- **Build and Installation:** Poetry
- **Result Logging:** Choose between wandb or clearml

## 📚 Supported Methods and Features

### LLM Alignment
- **SFT:** With the possibility to disable loss on unwanted message roles. ([code](scripts/model_training/sft.py))
- **DPO and ORPO:** All TRL options (IPO, SLic-HF, RPO, etc) ([code](scripts/model_training/dpo.py))
- **CPO and SimPO:** All TRL options ([code](scripts/model_training/cpo.py))
- **SMPO:** Our own stable alignment method (details below) ([code](scripts/model_training/smpo.py))
- **GPO:** Unique, our own implementation of offline GRPO (details below) ([code](scripts/model_training/gpo.py))
- **Distillation**: Custom script with many distill losses support ([code](scripts/model_training/distill.py))
- **Gradient-based prompts training**: Our method of training prompts using real tokens from tokenizer via Gumbel-Softmax trick ([code](scripts/prompts_training/sft.py))

### Reward modeling and Classification
- **Bradley-Terry Reward Training:** With margins and rewards centring support from TRL ([code](scripts/model_training/rewards.py))
- **Rejection Sampling:** Effective async preference dataset generation using vLLM and RM ([code](scripts/inference/rm_rejection_sampling.py))
- **LLM scoring using RM:** Use RM model and your dataset to caluclate RM scores statistics to compare models. ([code](scripts/inference/rm_scoring.py))
- **Classification:** Support for multiclass/multilabel/binary classification. ([code](scripts/model_training/classification.py))

### Additional
- The ability to **mix any number of datasets** for training, provided they use the same column names for replicas.
- All datasets follow the **JSON lines format** and conform to Hugging Face standards (storing messages in the format `[{'role': ..., 'content': ...}]`).
- **Special prompts basket format for generation:** Our prompts basket format (vi-basket) allows to generate dialogs with follow-ups and system-prompts.
- **Batched generation using OpenAI client**: Async batched generation, with support of follow-ups and system prompts. ([code](scripts/inference/batched_generation.py))
- **Effective FAISS Map-Reduce Deduplication:** We have tools for Map-Reduce based deduplication for dense embeddings. It can deduplicate VERY large datasets in parallel. ([code](src/utils/embeddings_utils.py))
- Support for freezing specific model modules for training without LoRA (see `unfreeze_layers_patterns` in [common args](src/configs/additional/common_script_args.py))

### SMPO - Simple Margin Preference Optimization

Our own alignment method designed for PO stability. The method is inspired by such methods as IPO, SimPO, C-RLFT, as well as introducing its own loss function of separating chosen and rejected pairs.

The main idea of the method is the desire to smoothly achieve the desired margin level without forcing the model to retrain by adding a balancing SFT loss on chosen and rejected at the same time.

This method is also distinguished by the presence of a separate scheduler for margin (`use_margin_schedule`in config), which provides stability in the early stages of training and prevents hacking of rewards.

The implementation of the method is [here](src/trainers/smpo_trainer.py), and the config is [here](src/configs/smpo_config.py).

### GPO - Offline GRPO

Ou own implementation of offline version of GRPO. This version does not use vLLM during training. 

This method is similar to KTO in that it does not require pairwise comparisons, but unlike KTO, formulas from GRPO are used, as well as rewards can be any float and the number of completions (group size) per prompt is unlimited.

It requires prepared dataset with columns [`prompt`, `completions`, `rewards`], where `prompt` is a `List[Dict]` (OpenAI format)  `completions` is a `List[List[Dict]]` of size G and `rewards` is a `List[float]` with precompute rewards.

The implementation of the method is [here](src/trainers/gpo_trainer.py), and the config is [here](src/configs/gpo_config.py).

### Prompts Optimization

Our own method of training system (any role) prompts (using real tokens from tokenizer) using gradient-based method via Gumbel-Softmax trick.

The implementation of the method is [here](scripts/prompts_training/sft.py), and the config is [here](src/configs/prompts_optimization_comfig.py).

## 🚀 How to Use

### 📦 Installation

#### 🛠️ Project Installation

Run the following commands inside the project folder:

1. Install Poetry:

   ```bash
   pip install poetry
   ```

2. Install project dependencies:

   ```bash
   poetry install
   ```

   Verify with:

   ```bash
   poetry show
   ```

3. (Optional) Set the environment variable `HF_HOME` to your desired folder:

   ```bash
   export HF_HOME=/mnt/hf/
   ```

4. (Optional) Log in to Hugging Face CLI:

   ```bash
   poetry run huggingface-cli login
   ```

5. (Optional) Log in to Weights & Biases:

   ```bash
   poetry run wandb login
   ```

6. Check the configuration settings in the `accelerate/` folder (number of GPUs, etc.).

#### ⚙️ Prerequisites and Troubleshooting

First, make sure you have all the necessary developer Linux libraries installed, including GCC and G++ version 8 or higher. You can check this by running:

```bash
gcc --version
```

It is recommened to do this steps before any installation:

```bash
apt update
apt install build-essential zlib1g-dev libffi-dev libssl-dev libbz2-dev libreadline-dev libsqlite3-dev liblzma-dev libncurses-dev tk-dev
```

Next, ensure that CUDA is version 11.8 or higher (preferably 12.1) and that all your GPUs are detected. Use the command:

```bash
nvidia-smi
```

After completing the first step of installation, check that you have Poetry version 1.8+ installed, and it’s best to use Python **3.10.16**. If not, update Poetry with:

```bash
poetry self update
```

and run:

```bash
poetry env use 3.10.16
```

You can install 3.10.16 version of python via [PyEnv](https://github.com/pyenv/pyenv).

After the second installation step, make sure that running `poetry run ds_report` returns meaningful text. Additionally, verify the version of Torch and the presence of NVIDIA packages.

If you encounter an error related to DeepSpeed and `fused_adam` during training, you need to remove DeepSpeed from your environment and install it with:

```bash
DS_BUILD_FUSED_ADAM=1 poetry run pip install deepspeed==0.14.5
```

Sometimes, deepspeed errors depends on Python version, on 3.10.16 everythong was tested in different environments.

### 🏃‍♂️ Example of Running a Script from Config

You need to select a DeepSpeed config + a training config + the script itself. Here’s an example command to start SFT training using YAML config:

```bash

PYTHONPATH="${PYTHONPATH}:src/" poetry run accelerate launch --config_file accelerate/fsdp_gradop_config.yaml scripts/sft.py training_configs/sft/sft-phi4-lora-GrandmasterRAG-v4.yaml
```

### 📝 YAML config examples

<details>
  <summary>SFT Training</summary>
  
  Config for training SFT Llama 3.1, using Liger kernel, only assistant answers, modified chat template, LoRA, generating examples on eval.
  
  ```yaml
  model_name_or_path: "unsloth/Meta-Llama-3.1-8B-Instruct"
  dataset:
    - "Vikhrmodels/GrandMaster-PRO-MAX"
    - "Vikhrmodels/Grounded-RAG-RU-v2"
  train_only_on_completions: True
  per_device_train_batch_size: 1
  per_device_eval_batch_size: 1
  num_train_epochs: 1
  save_strategy: "steps"
  save_steps: 400
  save_total_limit: 6
  learning_rate: 0.00004
  gradient_accumulation_steps: 8
  gradient_checkpointing: True
  logging_steps: 1
  remove_unused_columns: False
  dataloader_num_workers: 2
  save_only_model: True
  generate_eval_examples: True
  use_liger: True
  max_seq_length: 16000
  evaluation_strategy: "steps"
  eval_steps: 400
  run_name: "sft-grndmrag-llama-3.1-unsloth-lora-256-qkvogudlm-v1"
  output_dir: "/home/models/sft-grndmrag-llama-3.1-unsloth-lora-256-qkvogudlm-v1"
  warmup_steps: 20
  report_to: "wandb"
  conversation_field: "conversation"
  bf16: True
  seed: 42
  logging_first_step: True
  use_peft: True
  lora_target_modules:
    - "k_proj"
    - "v_proj"
    - "q_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
    - "lm_head"
  lora_r: 256
  lora_alpha: 256
  assistant_message_template: "<|start_header_id|>assistant<|end_header_id|>\n\n"
  pad_token: "<|reserved_special_token_0|>"
  eos_token: "<|eot_id|>"
  chat_template: "{{ bos_token }}{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
  force_chat_template: True
  ```
  
</details>
