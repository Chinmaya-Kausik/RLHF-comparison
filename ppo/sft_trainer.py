from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_from_disk
#from accelerate import Accelerator
from trl import SFTTrainer
from peft import LoraConfig
import torch
import wandb

#accelerator = Accelerator()
#tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
#tokenizer.add_special_tokens({"pad_token": "[PAD]"})
dataset = load_from_disk("./datasets/sft_tldr_dataset")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model_id = "openai-community/gpt2-medium"
model_kwargs = {
    "quantization_config": bnb_config
}

training_args = TrainingArguments(
    output_dir="models/tldr-gpt2-sft",
    max_steps=1,
    report_to="wandb"
)

sft_trainer = SFTTrainer(
    model=model_id,
    packing=True,
    train_dataset=dataset,
    args=training_args,
    peft_config = peft_config,
    model_init_kwargs=model_kwargs
)

sft_trainer.train()
