from typing import Dict
from datasets import load_dataset, IterableDataset
from transformers import AutoTokenizer


def build_ppo_tldr_dataset(tokenizer):
    name = "openai/summarize_from_feedback"
    dataset = load_dataset(name, "comparisons", split="train")
    old_columns = dataset.column_names

    def preprocess(sample):
        new_sample = {}
        post = sample["info"]["post"]
        query = "Question: " + post
        tokenized_query = tokenizer(query)

        new_sample["input_ids"] = tokenized_query["input_ids"]
        new_sample["attention_mask"] = tokenized_query["attention_mask"]
        new_sample["query"] = query

        return new_sample

    dataset = dataset.map(preprocess, remove_columns=old_columns)
    dataset = dataset.with_format("torch")
    return dataset


def build_sft_tldr_dataset(tokenizer):
    name = "openai/summarize_from_feedback"
    dataset = load_dataset(name, "comparisons", split="train")
    old_columns = dataset.column_names

    def preprocess(sample):
        new_sample = {}
        choice = sample["choice"]
        question = "Question: " + sample["info"]["post"]

        chosen_response = sample["summaries"][choice]["text"]
        chosen_query = question + "\n\nAnswer:" + chosen_response

        new_sample["prompt"] = question
        new_sample["completion"] = chosen_query

        return new_sample

    dataset = dataset.map(preprocess, remove_columns=old_columns)
    dataset = dataset.with_format("torch")
    return dataset



def build_reward_tldr_dataset(tokenizer):
    name = "openai/summarize_from_feedback"
    dataset = load_dataset(name, "comparisons", split="train")
    old_columns = dataset.column_names

    def preprocess(sample):
        new_sample = {}
        choice = sample["choice"]
        question = sample["info"]["post"]

        chosen_response = sample["summaries"][choice]["text"]
        chosen_query = "Question: " + question + "\n\nAnswer:" + chosen_response

        rejected_response = sample["summaries"][1 - choice]["text"]
        rejected_query = "Question: " + question + "\n\nAnswer:" + rejected_response

        tokenized_chosen = tokenizer(chosen_query)
        new_sample["input_ids_chosen"] = tokenized_chosen["input_ids"]
        new_sample["attention_mask_chosen"] = tokenized_chosen["attention_mask"]

        tokenized_rejected = tokenizer(rejected_query)
        new_sample["input_ids_rejected"] = tokenized_rejected["input_ids"]
        new_sample["attention_mask_rejected"] = tokenized_rejected["attention_mask"]

        return new_sample

    dataset = dataset.map(preprocess, remove_columns=old_columns)
    dataset = dataset.with_format("torch")
    return dataset


class ProcessedData(IterableDataset):
    def __init__(self, tokenizer, generator):
        super().__init__(self)
        self.__tokenizer = tokenizer
        self.__generator = generator

    def __iter__(self):
        return self.__generator(self.__tokenizer)


tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")


ppo_tldr_dataset = build_ppo_tldr_dataset(tokenizer)
sft_tldr_dataset = build_sft_tldr_dataset(tokenizer)
reward_tldr_dataset = build_reward_tldr_dataset(tokenizer)

ppo_tldr_dataset.save_to_disk("./datasets/ppo_tldr_dataset")
print("PPO saved")
sft_tldr_dataset.save_to_disk("./datasets/sft_tldr_dataset")
print("SFT saved")
reward_tldr_dataset.save_to_disk("./datasets/rw_tldr_dataset")
print("Reward saved")
