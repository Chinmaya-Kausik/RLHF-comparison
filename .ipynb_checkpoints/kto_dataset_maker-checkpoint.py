from datasets import load_dataset, concatenate_datasets, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from numpy import random

train_dataset = load_dataset('openai/summarize_from_feedback', 'comparisons', split='train')

def dataset_process(example, index=0, random=False):
    if not random:
        return {"prompt": example["info"]["post"], "completion": example["summaries"][index]["text"], "label": True if index==example["choice"] else False}
    else:
        index = random.randint(0,2)
        return {"prompt": example["info"]["post"], "completion": example["summaries"][index]["text"], "label": True if index==example["choice"] else False}
    
train_dataset_0 = train_dataset.map(lambda example: dataset_process(example, index=0), num_proc=4)
train_dataset_1 = train_dataset.map(lambda example: dataset_process(example, index=1), num_proc=4)
kto_train_dataset = concatenate_datasets([train_dataset_0, train_dataset_1])
kto_train_dataset.save_to_disk("kto_train_dataset")

    