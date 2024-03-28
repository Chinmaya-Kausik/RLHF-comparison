from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from accelerate import Accelerator
from data import tldr_sft_generator, ProcessedData
from trl import SFTTrainer
import torch

accelerator = Accelerator()

tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side="right")
tokenizer.add_special_tokens({"pad_token": "[PAD]"})

dataset = ProcessedData(tokenizer, tldr_sft_generator)
dataset = dataset.with_format("torch")


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model_id = "openai-community/gpt2-large"
model_kwargs = {
#    "quantization_config": bnb_config
}
training_args = TrainingArguments(
    output_dir="ppo/models/tldr-gpt2-sft",
    max_steps=1,
)

sft_trainer = SFTTrainer(
    model=model_id,
    packing=True,
    train_dataset=dataset,
    args=training_args,
    model_init_kwargs=model_kwargs
)

sft_trainer.train()