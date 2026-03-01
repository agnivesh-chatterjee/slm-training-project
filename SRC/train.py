## train.py
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM
from preprocess import JEEDataset

MODEL_NAME = "sshleifer/tiny-gpt2"
DEVICE = "cpu"
EPOCHS = 5
LR = 5e-5

def main():

    dataset = JEEDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0

        for batch in dataloader:

            # ✅ DO NOT unsqueeze — DataLoader already batches
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")

    model.save_pretrained("../outputs/final_model")
if __name__ == "__main__":
    main()