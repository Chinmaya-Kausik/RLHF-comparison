from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import RewardConfig, RewardTrainer
from peft import LoraConfig, TaskType
# from accelerate import Accelerator
from data import tldr_reward_generator, ProcessedData

# accelerator = Accelerator()

# tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side="right")
# tokenizer.add_special_tokens({"pad_token": "[PAD]"})

# dataset = ProcessedData(tokenizer, tldr_reward_generator)
dataset = load_from_disk("./datasets/rw_tldr_dataset")

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

config = RewardConfig(
    output_dir="ppo/models/tldr-gpt2-reward",
    max_steps=128,
)

trainer = RewardTrainer(
    model="gpt2",
    args=config,
    tokenizer=tokenizer,
    train_dataset=dataset,
    peft_config=peft_config,
)

trainer.train()
