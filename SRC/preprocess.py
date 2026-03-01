# src/preprocess.py

import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

MODEL_NAME = "sshleifer/tiny-gpt2"
DATA_PATH = "../data/newdata_valid.json"
MAX_LEN = 256


def split_solution_and_answer(solution_text):
    """
    Splits solution string into:
    - reasoning
    - final answer
    If 'Answer:' not found, returns full solution and empty answer.
    """

    if "Answer:" in solution_text:
        parts = solution_text.split("Answer:")
        reasoning = parts[0].strip()
        answer = parts[1].strip()
        return reasoning, answer
    else:
        return solution_text.strip(), ""


class JEEDataset(Dataset):
    def __init__(self):

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        with open(DATA_PATH, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self.samples = []

        for item in raw_data:

            question = item.get("question", "")
            solution_raw = item.get("solution", "")

            reasoning, answer = split_solution_and_answer(solution_raw)

            text = f"""### Problem:
{question}

### Solution:
{reasoning}

### Final Answer:
{answer}
"""

            tokens = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=MAX_LEN,
                return_tensors="pt"
            )

            self.samples.append({
                "input_ids": tokens["input_ids"].squeeze(0),
                "attention_mask": tokens["attention_mask"].squeeze(0),
                "labels": tokens["input_ids"].squeeze(0)
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
