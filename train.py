import torch
import torch.nn as nn
import torch.optim as optim
import sentencepiece as spm
import numpy as np
import random
import string
from selfattention import GPTClassifier
from rlbasedmodel import MaskingDQNAgent

device = "cuda" if torch.cuda.is_available() else "cpu"

sp = spm.SentencePieceProcessor(model_file="peptide_tokenizer.model")
vocab_size = sp.GetPieceSize()

valid_aas = set("ACDEFGHIKLMNPQRSTVWY")

def is_valid_peptide(seq):
    return all(residue in valid_aas for residue in seq.upper())

def read_fasta(filepath):
    sequences = []
    with open(filepath, 'r') as f:
        seq = ''
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq:
                    sequences.append(seq)
                    seq = ''
            else:
                seq += line
        if seq:
            sequences.append(seq)
    return sequences

def random_invalid_sequence(length):
    letters = string.ascii_uppercase
    invalid = [l for l in letters if l not in valid_aas]
    return ''.join(random.choices(invalid, k=length))

amp_sequences = (
    read_fasta(r"C:\Users\Nikhal Kumar\Downloads\amp-scanner-v2-main\original-dataset\AMP.tr.fa") +
    read_fasta(r"C:\Users\Nikhal Kumar\Downloads\amp-scanner-v2-main\original-dataset\AMP.eval.fa") +
    read_fasta(r"C:\Users\Nikhal Kumar\Downloads\amp-scanner-v2-main\original-dataset\AMP.te.fa")
)
decoy_sequences = (
    read_fasta(r"C:\Users\Nikhal Kumar\Downloads\amp-scanner-v2-main\original-dataset\DECOY.tr.fa") +
    read_fasta(r"C:\Users\Nikhal Kumar\Downloads\amp-scanner-v2-main\original-dataset\DECOY.eval.fa") +
    read_fasta(r"C:\Users\Nikhal Kumar\Downloads\amp-scanner-v2-main\original-dataset\DECOY.te.fa")
)

synthetic_negatives = [(random_invalid_sequence(10), 0) for _ in range(1000)]

dataset = [(seq, 1) for seq in amp_sequences] + \
          [(seq, 0) for seq in decoy_sequences] + \
          synthetic_negatives
random.shuffle(dataset)

transformer = GPTClassifier(vocab_size).to(device)
optimizer = optim.Adam(transformer.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()
agent = MaskingDQNAgent(state_size=312, action_size=312)

batch_size = 32
checkpoint_interval = 100
pretrain_epochs = 10
masked_epochs = 3

print("🔹 Pretraining Transformer without masking...")

for epoch in range(pretrain_epochs):
    total_loss = 0
    correct = 0

    for i, (seq, label) in enumerate(dataset):
        tokens = sp.Encode(seq)[:312]
        tokens += [0] * (312 - len(tokens))
        input_tensor = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        label_tensor = torch.tensor([label], dtype=torch.long, device=device)

        transformer.train()
        optimizer.zero_grad()
        logits = transformer(input_tensor)
        loss = criterion(logits, label_tensor)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred_label = torch.argmax(logits, dim=-1).item()
        if pred_label == label:
            correct += 1

        if (i + 1) % checkpoint_interval == 0:
            print(f"Pretrain checkpoint steps | Step {i+1}")

    torch.save(transformer.state_dict(), f"transformer_pretrain_checkpoint_{i+1}.pt")
    acc = correct / len(dataset)
    print(f"Pretrain Epoch {epoch+1}/{pretrain_epochs} | Loss: {total_loss / len(dataset):.4f} | Accuracy: {acc:.4f}")

print("✅ Pretraining complete. Starting masked training...")

for epoch in range(masked_epochs):
    total_loss = 0
    correct = 0

    for i, (seq, label) in enumerate(dataset):
        tokens = sp.Encode(seq)[:312]
        tokens += [0] * (312 - len(tokens))
        state = np.array(tokens)

        raw_mask = (agent.act(state) > 0.5).astype(int)
        min_mask_tokens = int(0.1 * len(state))
        max_mask_tokens = int(0.5 * len(state))

        if raw_mask.sum() < min_mask_tokens:
            unmasked_indices = np.where(raw_mask == 0)[0]
            additional = np.random.choice(unmasked_indices, min_mask_tokens - raw_mask.sum(), replace=False)
            raw_mask[additional] = 1
        elif raw_mask.sum() > max_mask_tokens:
            mask_indices = np.where(raw_mask == 1)[0]
            selected_indices = np.random.choice(mask_indices, max_mask_tokens, replace=False)
            raw_mask = np.zeros_like(raw_mask)
            raw_mask[selected_indices] = 1

        mask = raw_mask

        if not is_valid_peptide(seq):
            mask = np.ones(312)
            label = 0

        masked_input = [t if m == 0 else 0 for t, m in zip(state, mask)]
        input_tensor = torch.tensor(masked_input, dtype=torch.long, device=device).unsqueeze(0)
        label_tensor = torch.tensor([label], dtype=torch.long, device=device)

        transformer.train()
        optimizer.zero_grad()
        logits = transformer(input_tensor)
        loss = criterion(logits, label_tensor)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred_label = torch.argmax(logits, dim=-1).item()
        if pred_label == label:
            correct += 1

        pred_prob = torch.softmax(logits, dim=-1)[0, label].item()
        reward = pred_prob - 0.2 * (mask.sum() / len(mask))  

        if (i + 1) % 500 == 0:
            print(f"Step {i+1} | Logits: {logits.detach().cpu().numpy()} | Probs: {torch.softmax(logits, dim=-1).detach().cpu().numpy()} | Masked Tokens: {mask.sum()}")

        next_state = state.copy()
        done = True
        agent.remember(state, mask, reward, next_state, done)
        agent.replay(batch_size)

        if (i + 1) % checkpoint_interval == 0:
            agent.save(f"masking_agent_checkpoint_{i+1}.pt") 
            torch.save(transformer.state_dict(), f"transformer_masked_checkpoint_{i+1}.pt")
            print(f"Step {i+1} | Last Reward: {reward:.4f} | Masked Tokens: {mask.sum()} / {len(mask)}")

    acc = correct / len(dataset)
    print(f"Masked Epoch {epoch + 1}/{masked_epochs} | Loss: {total_loss / len(dataset):.4f} | Accuracy: {acc:.4f}")


agent.save("masking_agent_final.pt")
torch.save(transformer.state_dict(), "transformer_final.pt")
print("✅ Joint training complete. Transformer and DQN agent saved.")


