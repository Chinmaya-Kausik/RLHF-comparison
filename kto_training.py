from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, concatenate_datasets
from numpy import random
from trl import KTOConfig, KTOTrainer
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

kto_train_dataset = load_from_disk("/home/ckausik/kto_train_dataset")
print("dataset loaded")

tldr_sft = AutoModelForCausalLM.from_pretrained("gpt2-medium", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
tokenizer.pad_token = tokenizer.eos_token

print("model and tokenizer initialized")

# tldr_sft = AutoModelForCausalLM.from_pretrained("CarperAI/openai_summarize_tldr_sft", device_map="auto")
# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

training_args = KTOConfig(output_dir = "/home/ckausik/RLHF-comparison",
    beta=0.1,
    desirable_weight=1.0,
    undesirable_weight=1.0
)

kto_trainer = KTOTrainer(
    tldr_sft,
    args=training_args,
    train_dataset=kto_train_dataset,
    tokenizer=tokenizer,
    peft_config = lora_config
)

print("trainer initialised")

print(kto_trainer.train_dataset[0])


