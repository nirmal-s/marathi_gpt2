import torch
from torch.utils.data import Dataset, DataLoader
from marathi_gpt2 import GPT2Config, GPT2Model
import os

# Parameters
DATA_PATH = "data/mrwiki_text_tokenized.txt"
BATCH_SIZE = 1  # Keep at 1 for low memory
SEQ_LEN = 64    # Keep at 64 for low memory
EPOCHS = 3  # Increase for real training
LR = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu":
    print("Warning: CUDA GPU not available. Training will run on CPU and may be very slow.")

# Use a smaller model config for low memory GPUs
class SmallGPT2Config(GPT2Config):
    def __init__(self, vocab_size=32000, n_positions=1024, n_embd=384, n_layer=6, n_head=6, dropout=0.1):
        super().__init__(vocab_size, n_positions, n_embd, n_layer, n_head, dropout)

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

def find_latest_checkpoint():
    ckpts = [fn for fn in os.listdir('.') if fn.startswith('marathi_gpt2_step_') and fn.endswith('.pt')]
    if not ckpts:
        return None, 0, 0
    # Extract epoch and step from filename
    def parse_ckpt(fn):
        parts = fn.replace('.pt','').split('_')
        try:
            epoch = int(parts[3])
            step = int(parts[4])
            return (epoch, step, fn)
        except Exception:
            return (0, 0, fn)
    ckpts_parsed = [parse_ckpt(fn) for fn in ckpts]
    latest = max(ckpts_parsed, key=lambda x: (x[0], x[1]))
    return latest[2], latest[0], latest[1]

def train():
    # Clear CUDA memory if using GPU
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    dataset = TokenizedTextDataset(DATA_PATH, SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    config = GPT2Config()
    model = GPT2Model(config).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Resume from latest checkpoint if available
    latest_ckpt, start_epoch, start_step = find_latest_checkpoint()
    if latest_ckpt:
        print(f"Resuming from checkpoint: {latest_ckpt}")
        state_dict = torch.load(latest_ckpt, map_location=DEVICE)
        model.load_state_dict(state_dict, strict=False)
        # Optionally, load optimizer state if you saved it
    else:
        print("No checkpoint found. Starting from scratch.")
        start_epoch, start_step = 0, 0

    model.train()
    for epoch in range(start_epoch, EPOCHS):
        total_loss = 0
        for i, (x, y) in enumerate(loader):
            # If resuming, skip steps already completed in this epoch
            if epoch == start_epoch and i < start_step:
                continue
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = loss_fn(logits.view(-1, config.vocab_size), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
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
