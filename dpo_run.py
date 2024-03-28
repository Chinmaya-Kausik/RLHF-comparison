from datasets import load_dataset, IterableDataset
#from trl_edit.trl.trainer import SFTTrainer, DPOTrainer
from trl import DPOTrainer
from peft import LoraConfig
from transformers import GPT2Model, GPT2Tokenizer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM


# Processing dataset
dataset = load_dataset('openai/summarize_from_feedback',
                       'comparisons', 
                       split='train',
                       streaming=False
).select_columns(['info', 'summaries', 'choice'])

train_dataset = dataset.map(lambda example: {"prompt": example['info']['post'], 
                                             "chosen": example['summaries'][example['choice']]['text'],
                                             "rejected": example['summaries'][example['choice']-1]['text']})

# def my_generator():
#     for example in dataset:
#         yield {"prompt": example['info']['post'], 
#                "chosen": example['summaries'][example['choice']]['text'],
#                "rejected": example['summaries'][example['choice']-1]['text']}

# train_dataset = IterableDataset.from_generator(my_generator)   

# Load base model
model = AutoModelForCausalLM.from_pretrained("gpt2-medium")
tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
tokenizer.pad_token = tokenizer.eos_token

# Training
training_args = TrainingArguments(output_dir="./output", )

# PEFT configurations
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# DPO trainer
dpo_trainer = DPOTrainer(
    model,
    args=training_args,
    beta=0.1,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    peft_config=lora_config,
)

dpo_trainer.train()