from typing import Dict
from datasets import load_dataset, IterableDataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader


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
    dataset = load_dataset(name, "comparisons", split="train[:1]")
    old_columns = dataset.column_names

    def preprocess(samples):
        new_samples = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }
        for info, summaries, choice in zip(
                samples["info"],
                samples["summaries"],
                samples["choice"]
                ):

            subreddit = f"Subreddit: {info["subreddit"]}"
            title = f"\n\nTitle: {info["title"]}"
            post = f"\n\nPost: {info["post"]}"
            prompt = subreddit + title + post

            chosen_response = f"\n\nSummary: {summaries[choice]["text"]}"
            chosen_query = prompt + chosen_response

            rejected_response = f"\n\nSummary: {summaries[1 - choice]["text"]}"
            rejected_query = prompt + rejected_response

            prompt = tokenizer(prompt, truncation=True)
            chosen_query = tokenizer(chosen_query, truncation=True)
            rejected_query = tokenizer(rejected_query, truncation=True)

            new_samples["labels"].append(prompt["input_ids"])
            new_samples["input_ids"].append(chosen_query["input_ids"])
            new_samples["attention_mask"].append(chosen_query["attention_mask"])

            new_samples["labels"].append(prompt["input_ids"])
            new_samples["input_ids"].append(rejected_query["input_ids"])
            new_samples["attention_mask"].append(rejected_query["attention_mask"])

        return new_samples

    dataset = dataset.map(preprocess, remove_columns=old_columns, batched=True)
    dataset = dataset.with_format("torch")
    return dataset


def build_reward_tldr_dataset(tokenizer):
    name = "openai/summarize_from_feedback"
    dataset = load_dataset(name, "comparisons", split="train")
    old_columns = dataset.column_names

    def preprocess(samples):
        new_samples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": []
        }

        for info, summaries, choice in zip(
                samples["info"],
                samples["summaries"],
                samples["choice"]
                ):
            post = "Post: " + info["post"]
            chosen_response = summaries[choice]["text"]
            chosen_query = post + "\n\nAnswer:" + chosen_response

            rejected_response = summaries[1 - choice]["text"]
            rejected_query = post + "\n\nAnswer:" + rejected_response

            tokenized_chosen = tokenizer(chosen_query, truncation=True)
            new_samples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            new_samples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])

            tokenized_rejected = tokenizer(rejected_query, truncation=True)
            new_samples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            new_samples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

        return new_samples

    dataset = dataset.map(preprocess, remove_columns=old_columns, batched=True)
    dataset = dataset.with_format("torch")
    return dataset


class ProcessedData(IterableDataset):
    def __init__(self, tokenizer, generator):
        super().__init__(self)
        self.__tokenizer = tokenizer
        self.__generator = generator

    def __iter__(self):
        return self.__generator(self.__tokenizer)


tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = "[PAD]"

name = "openai/summarize_from_feedback"

#ppo_tldr_dataset = build_ppo_tldr_dataset(tokenizer)
dataset = build_sft_tldr_dataset(tokenizer)
# for sample in dataset:
#     print(sample)
loader = DataLoader(dataset, collate_fn=DataCollatorWithPadding(tokenizer), batch_size=2)
for batch in loader:
    print(tokenizer.batch_decode(batch["labels"]))
#reward_tldr_dataset = build_reward_tldr_dataset(tokenizer)

#ppo_tldr_dataset.save_to_disk("./datasets/ppo_tldr_dataset")
#print("PPO saved")
#sft_tldr_dataset.save_to_disk("./datasets/sft_tldr_dataset")
# print("SFT saved")
# reward_tldr_dataset.save_to_disk("./datasets/rw_tldr_dataset")
# print("Reward saved")
