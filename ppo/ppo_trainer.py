from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, AutoModelForCausalLM
from datasets import load_dataset
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from data import tldr_ppo_generator, ProcessedData
from tqdm import tqdm
from accelerate import Accelerator
import torch

accelerator = Accelerator()

tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side="right")
tokenizer.add_special_tokens({"pad_token": "[PAD]"})

dataset = ProcessedData(tokenizer, tldr_ppo_generator)

model = AutoModelForCausalLM.from_pretrained("ppo/models/tldr-gpt2-sft")
model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
reward_model = AutoModelForSequenceClassification.from_pretrained("ppo/models/tldr-gpt2-reward")

config = PPOConfig(
    learning_rate=1.51e-4,
    # log_with="wandb"
)

ppo_trainer = PPOTrainer(
    model=model,
    config=config,
    dataset=dataset,
)

generation_kwargs = {
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "eos_token_id": tokenizer.eos_token_id,
    "return_prompt": True
}

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]

    #### Generate response
    response_tensors = ppo_trainer.generate(
        query_tensors,
    )

    reward_outputs = reward_model(response_tensors)
    reward_model.call_count = 0
    rewards = [torch.tensor(output) for output in reward_outputs]

    #### Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

accelerator.save_model(model, "ppo/models/imdb-gpt2-ppo")
