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

### 📚 Supported Methods
- **SFT:** With the possibility to disable loss on unwanted message roles
- **Distillation:** Options include KL Div, JS Div, SLIM, Earth Mover, MSE, Soft CE, Cosine, Alpha-Beta Div
- **DPO:** All TRL options (IPO, SLic-HF, RPO, etc)
- **ORPO:** All TRL options
- **CPO and SimPO:** All TRL options
- **SMPO:** Our own most stable alignment method (details below)
- **Non-pair Reward Modeling:** With margins and centring support from TRL
- **Rejection Sampling:** Preference dataset generation using vLLM and RM
- **LLM scoring using RM:** Use RM model and your dataset to caluclate RM scores statistics to compare models.
- **NER, CLIP, Classification, STS:** Not native to this toolkit but tested (Work in Progress)

### 📌 Additional Features
- All datasets follow the **JSON lines format** and conform to Hugging Face standards (storing messages in the format ` [{'role': ..., 'content': ...}]`).
- The ability to **mix any number of datasets** for training, provided they use the same column names for replicas.
- **Generation and logging** in wandb/clearml of test replicas during evaluation runs for SFT and Preference training (using `generate_eval_examples` and `num_gen_examples` options in configs).
- **vLLM batched generation** of answers for some datasets using an OpenAI-like server.

### SMPO - Simple Margin Preference Optimization

Our own alignment method designed for PO stability. The method is inspired by such methods as IPO, SimPO, C-RLFT, as well as introducing its own loss function of separating chosen and rejected pairs.

The main idea of the method is the desire to smoothly achieve the desired margin level without forcing the model to retrain by adding a balancing SFT loss on chosen and rejected at the same time.

The implementation of the method is [here](src/trainers/smpo_trainer.py), and the config is [here](src/configs/smpo_config.py).

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

Next, ensure that CUDA is version 11.8 or higher (preferably 12.1) and that all your GPUs are detected. Use the command:

```bash
nvidia-smi
```

After completing the first step of installation, check that you have Poetry version 1.8+ installed, and it’s best to use Python 3.10. If not, update Poetry with:

```bash
poetry self update
```

and run:

```bash
poetry env use 3.10
```

After the second installation step, make sure that running `poetry run ds_report` returns meaningful text. Additionally, verify the version of Torch and the presence of NVIDIA packages.

If you encounter an error related to DeepSpeed and `fused_adam` during training, you need to remove DeepSpeed from your environment and install it with:

```bash
DS_BUILD_FUSED_ADAM=1 poetry run pip install deepspeed==0.14.5
```

### 🏃‍♂️ Example of Running a Script from Config

You need to select a DeepSpeed config + a training config + the script itself. Here’s an example command to start SFT training using YAML config:

```bash

PYTHONPATH="${PYTHONPATH}:src/" poetry run accelerate launch --config_file accelerate/stage2_config.yaml scripts/sft.py training_configs/sft-llama-3.1-8b-it-lora-GrandmasterRAG-v1.yaml
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

---

Feel free to modify any part of this translation! 😊
