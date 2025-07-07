import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from tqdm import tqdm
from TinyPeLLMModel import TinyPeLLM
from TinyPeLLMTokenizer import TinyPeLLMTokenizer

class TinyTextDataset(Dataset):
    def __init__(self, filepath, tokenizer, block_size=128):
        with open(filepath, 'r', encoding='utf-8') as f:
            self.lines = [line.strip() for line in f if line.strip()]
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        encoded = self.tokenizer.encode(self.lines[idx])
        if len(encoded) > self.block_size:
            encoded = encoded[:self.block_size]
        return encoded

def collate_fn(batch):
    inputs = pad_sequence(batch, batch_first=True, padding_value=0)
    labels = inputs.clone()
    return inputs, labels

def train(model, dataloader, optimizer, device, epochs=3):
    model.train()
    loss_fn = CrossEntropyLoss(ignore_index=0)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            inputs, labels = [x.to(device) for x in batch]
            outputs = model(inputs)
            shift_logits = outputs[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {epoch_loss / len(dataloader):.4f}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = TinyPeLLMTokenizer("tinypellm.model")

    dataset = TinyTextDataset("tinypellm_corpus.txt", tokenizer)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    vocab_size = tokenizer.vocab_size()
    model = TinyPeLLM(vocab_size).to(device)

    optimizer = AdamW(model.parameters(), lr=2e-4)

    train(model, dataloader, optimizer, device, epochs=5)

    torch.save(model.state_dict(), "tinypellm_model.pt")
