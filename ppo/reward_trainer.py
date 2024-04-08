from transformers import (
        AutoTokenizer,
        AutoModel,
        AutoModelForSequenceClassification,
        Trainer
)
from transformers.tokenization_utils_base import (
        PreTrainedTokenizerBase,
        PaddingStrategy
)
from typing import List, Dict, Any, Union, Optional
from trl import RewardConfig
from peft import LoraConfig, TaskType
from accelerate import Accelerator
from datasets import load_from_disk
from dataclasses import dataclass
import torch.nn as nn


class RewardModel(nn.Module):
    def __init__(self, model_id='gpt2-medium', dropout=0.5):
        super(RewardModel, self).__init__()

        self.base_model = AutoModel.from_pretrained(model_id)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(1024, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        # You write you new head here
        outputs = self.dropout(outputs[0])
        outputs = self.linear(outputs)

        return outputs


@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_j = []
        features_k = []
        for feature in features:
            features_j.append(
                {
                    "input_ids": feature["input_ids_j"],
                    "attention_mask": feature["attention_mask_j"],
                }
            )
            features_k.append(
                {
                    "input_ids": feature["input_ids_k"],
                    "attention_mask": feature["attention_mask_k"],
                }
            )
        batch_j = self.tokenizer.pad(
            features_j,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_k = self.tokenizer.pad(
            features_k,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids_j": batch_j["input_ids"],
            "attention_mask_j": batch_j["attention_mask"],
            "input_ids_k": batch_k["input_ids"],
            "attention_mask_k": batch_k["attention_mask"],
            "return_loss": True,
        }
        return batch


class RewardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        rw_chosen = model(
                input_ids=inputs["input_ids_chosen"],
                attention_mask=inputs["attention_mask_chosen"])[0]
        rw_rejected = model(
                input_ids=inputs["input_ids_rejected"],
                attention_mask=inputs["attention_mask_rejected"])[0]
        loss = -nn.functional.logsigmoid(rw_chosen - rw_rejected).mean()
        if return_outputs:
            return loss, {"rw_chosen": rw_chosen, "rw_rejected": rw_rejected}
        return loss


accelerator = Accelerator()

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({"pad_token": "[PAD]"})

dataset = load_from_disk("./datasets/rw_tldr_dataset")

#model = RewardModel()
model = AutoModelForSequenceClassification.from_pretrained("gpt2", num_labels=1)

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

config = RewardConfig(
    output_dir="ppo/models/tldr-gpt2-reward",
)

trainer = RewardTrainer(
    model=model,
    args=config,
    train_dataset=dataset,
    tokenizer=tokenizer,
    #peft_config=peft_config,
)

trainer.train()
