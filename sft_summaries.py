from datasets import load_dataset, concatenate_datasets
#from trl_edit.trl.trainer import SFTTrainer, DPOTrainer
from trl import DPOTrainer, SFTTrainer
from peft import LoraConfig
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM
import torch

# Processing dataset
dataset = load_dataset('openai/summarize_from_feedback',
                       'comparisons', 
                       split='train',
                       streaming=False
).select_columns(['info', 'summaries', 'choice'])


def dataset_process(example, index):
    return {"prompt": "SUBREDDIT: r/" + example['info']['subreddit'] + "\n\nTITLE: " + example['info']['title'] + "\n\nPOST: " + example['info']['post'] + "\n\nTL;DR:", 
            "completion": example['summaries'][index]['text'],
    }

train_dataset0 = dataset.map(lambda example: dataset_process(example, index=0))
train_dataset1 = dataset.map(lambda example: dataset_process(example, index=1))
train_dataset = concatenate_datasets([train_dataset0, train_dataset1])


# train_dataset = dataset.map(lambda example: {"prompt": "SUBREDDIT: r/" + example['info']['subreddit'] + "\n\nTITLE: " + example['info']['title'] + "\n\nPOST: " + example['info']['post'] + "\n\nTL;DR:", 
#                                              "completion": example['summaries'][example['choice']]['text'],
#                                              # "rejected": example['summaries'][example['choice']-1]['text']
# })

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len( example )):
        text = f"### Prompt: {example['prompt']}\n ### Completion: {example['completion']}"
        output_texts.append(text)
    return output_texts


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
training_args = TrainingArguments(output_dir="./sft_output", 
                                  report_to="wandb",
                                  optim="adamw_torch",
                                  lr_scheduler_type="cosine",
                                  warmup_ratio=0.05,    
)

# PEFT configurations
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# DPO trainer
sft_trainer = SFTTrainer(
    model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    peft_config=lora_config,
    formatting_func=formatting_prompts_func,
    # optimizers=(torch.optim.Adam(), torch.optim.lr_scheduler.CosineAnnealingLR),
)

sft_trainer.train()