
### this is for running in local ###
import os
try:
    os.environ['HTTP_PROXY']='http://185.46.212.90:80'
    os.environ['HTTPS_PROXY']='http://185.46.212.90:80'
    print ("proxy_exported")
except:
    None

# !pip install -q -U trl transformers accelerate git+https://github.com/huggingface/peft.git
# !pip install -q datasets bitsandbytes einops


from datasets import load_dataset
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer

dataset_name = "timdettmers/openassistant-guanaco"
dataset = load_dataset(dataset_name, split="train")

model_name = "microsoft/phi-2"
bnb_config = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.float16,)
model = AutoModelForCausalLM.from_pretrained(model_name,quantization_config=bnb_config,trust_remote_code=True)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

from peft import LoraConfig

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["Wqkv","out_proj","fc1","fc2"])

from transformers import TrainingArguments

new_model = "phi-2-fine-tuned"
output_dir = "./results"

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",
    save_steps=500,
    logging_steps=10,
    learning_rate=2e-4,
    fp16=True,
    max_grad_norm=0.3,
    max_steps=500,
    warmup_ratio=0.05,
    group_by_length=True,
    lr_scheduler_type="constant",
    gradient_checkpointing=False)

from trl import SFTTrainer

max_seq_length = 300

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments)

for name, module in trainer.model.named_modules():
    if "ln" in name:
        module = module.to(torch.float16)


trainer.train()
trainer.model.save_pretrained(new_model)
