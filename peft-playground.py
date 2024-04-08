from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig

dataset = load_dataset("imdb", split="train")

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

trainer = SFTTrainer(
    "EleutherAI/gpt-neo-125m",
    train_dataset=dataset,
    dataset_text_field="text",
    peft_config=peft_config
)

trainer.train()
