from typing import Dict
from datasets import load_dataset, IterableDataset


def tldr_ppo_generator(
        tokenizer,
):
    name="openai/summarize_from_feedback"
    dataset = load_dataset(name, "comparisons", split="train", streaming=True)
    original_columns = dataset.column_names

    for _, sample in enumerate(dataset):
        new_sample = {}
        choice = sample["choice"]
        post = sample["info"]["post"]
        query = "Question: " + post
        tokenized_query = tokenizer(query)

        new_sample["input_ids"] = tokenized_query["input_ids"]
        new_sample["attention_mask"] = tokenized_query["attention_mask"]
        new_sample["query"] = query
        
        yield new_sample



def tldr_sft_generator(
        tokenizer,
):
    name="openai/summarize_from_feedback"
    dataset = load_dataset(name, "comparisons", split="train", streaming=True)

    for _, sample in enumerate(dataset):
        new_sample = {}
        choice = sample["choice"]
        question = "Question: " + sample["info"]["post"]

        chosen_response = sample["summaries"][choice]["text"]
        chosen_query = question + "\n\nAnswer:" + chosen_response

        new_sample["prompt"] = question
        new_sample["completion"] = chosen_query
        
        yield new_sample



def tldr_reward_generator(
        tokenizer,
):
    name="openai/summarize_from_feedback"
    dataset = load_dataset(name, "comparisons", split="train", streaming=True)
    original_columns = dataset.column_names

    for _, sample in enumerate(dataset):
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

        yield new_sample


class ProcessedData(IterableDataset):
    def __init__(self, tokenizer, generator):
        super().__init__(self)
        self.__tokenizer = tokenizer
        self.__generator = generator

    def __iter__(self):
        return self.__generator(self.__tokenizer)