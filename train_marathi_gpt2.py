import torch
from torch.utils.data import Dataset, DataLoader
from marathi_gpt2 import GPT2Config, GPT2Model
import os

# Parameters
DATA_PATH = "data/mrwiki_text_tokenized.txt"
BATCH_SIZE = 1  # Reduced for low memory
SEQ_LEN = 64    # Reduced for low memory
EPOCHS = 1  # Increase for real training
LR = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu":
    print("Warning: CUDA GPU not available. Training will run on CPU and may be very slow.")

class TokenizedTextDataset(Dataset):
    def __init__(self, file_path, seq_len):
        self.seq_len = seq_len
        self.data = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                ids = [int(tok) for tok in line.strip().split()]
                # Split long lines into multiple sequences
                for i in range(0, len(ids) - seq_len, seq_len):
                    self.data.append(ids[i:i+seq_len])
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.long)
        y = torch.tensor(self.data[idx][1:] + [0], dtype=torch.long)  # next token prediction
        return x, y

def train():
    dataset = TokenizedTextDataset(DATA_PATH, SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    config = GPT2Config()
    model = GPT2Model(config).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for i, (x, y) in enumerate(loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = loss_fn(logits.view(-1, config.vocab_size), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (i+1) % 100 == 0:
                print(f"Epoch {epoch+1} Step {i+1} Loss {loss.item():.4f}")
            if (i+1) % 100000 == 0:
                ckpt_path = f"marathi_gpt2_step_{epoch+1}_{i+1}.pt"
                torch.save(model.state_dict(), ckpt_path)
                print(f"Checkpoint saved: {ckpt_path}")
        print(f"Epoch {epoch+1} avg loss: {total_loss/len(loader):.4f}")
    torch.save(model.state_dict(), "marathi_gpt2.pt")
    print("Model saved as marathi_gpt2.pt")

if __name__ == "__main__":
    train()
