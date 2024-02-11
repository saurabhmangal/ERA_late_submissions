#!/usr/bin/env python3
# coding: utf-8

# Import Required Libraries
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, AutoModel
from peft import LoraConfig
from trl import SFTTrainer
from datasets import load_dataset
import pandas as pd
import datasets

# Install dependencies
# !pip install -q -U trl transformers accelerate peft einops datasets bitsandbytes scipy

# Define Constants
MODEL_NAME = "microsoft/phi-2"
DATASET_NAME = "OpenAssistant/oasst1"
OUTPUT_DIR = "./results"

# Initialize BitsAndBytes Config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Load Model and Tokenizer
# model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb_config, trust_remote_code=True)
model = AutoModel.from_pretrained('/home/saurabh/era_saurabh/late_submissions/s27/phi-2')
model.config.use_cache = False
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Print Model for Layer Identification
print(model)

# LORA Configuration
lora_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["Wqkv", "out_proj", "fc1", "fc2"]
)

# Training Arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    save_steps=100,
    logging_steps=10,
    learning_rate=2e-4,
    fp16=True,
    max_grad_norm=0.3,
    max_steps=700,
    warmup_ratio=0.05,
    lr_scheduler_type="constant"
)

# Load and Prepare Dataset
dataset = load_dataset(DATASET_NAME, split="train").to_pandas()
assistant_responses = dataset.query('role == "assistant" and rank == 0.0')
prompters = dataset.query('role == "prompter"').set_index("message_id")

# Combine Prompts and Responses
assistant_responses['prompt_response'] = assistant_responses.apply(
    lambda row: "### Human: " + prompters.loc[row.parent_id, 'text'] + "### Assistant: " + row['text'], axis=1
)

# Create HuggingFace Dataset
hf_dataset = datasets.Dataset.from_pandas(assistant_responses)

# Initialize Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=hf_dataset.map(lambda x: {"prompt_response": x['prompt_response']}),
    peft_config=lora_config,
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_args,
    dataset_text_field='prompt_response'
    
)

# Adjust Normalization Layers
for _, module in trainer.model.named_modules():
    if isinstance(module, torch.nn.LayerNorm):
        module.float()

# Train the Model
trainer.train()

# Test the Model
def generate_text(prompt):
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=200)
    result = pipe(f"<s>[INST] {prompt} [/INST]")
    return result[0]['generated_text']

# Example Usage
print(generate_text("What is large language model?"))
print(generate_text("What is QLora that stands for Quantization and Low-Rank Adapters"))
