import torch
import torch.nn as nn
import numpy as np
import sentencepiece as spm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

from selfattention import GPTClassifier
from rlbasedmodel import MaskingDQNAgent

device = "cuda" if torch.cuda.is_available() else "cpu"
sp = spm.SentencePieceProcessor(model_file="peptide_tokenizer.model")

valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
def is_valid_peptide(seq):
    return all(residue in valid_aas for residue in seq.upper())


transformer = GPTClassifier(sp.GetPieceSize()).to(device)
transformer.load_state_dict(torch.load("transformer_masked_checkpoint_1400.pt", map_location=device))
transformer.eval()

agent = MaskingDQNAgent(state_size=312, action_size=312)
agent.load("masking_agent_checkpoint_1400.h5")


def encode_sequence(seq):
    tokens = sp.Encode(seq)
    tokens = tokens[:312]
    tokens += [0] * (312 - len(tokens))
    return np.array(tokens)


def apply_mask(state):
    mask = agent.act(state)
    masked = [t if m == 0 else 0 for t, m in zip(state, mask)]
    return np.array(masked), mask



def evaluate_model(test_dataset):
    y_true = []
    y_pred = []
    y_prob = []

    criterion = nn.CrossEntropyLoss()
    total_loss = 0

    for seq, label in test_dataset:
        y_true.append(label)

        if not is_valid_peptide(seq):
            y_pred.append(0)
            y_prob.append(1.0 if label == 0 else 0.0)
            continue

        tokens = encode_sequence(seq)
        masked_tokens, mask = apply_mask(tokens)

        input_tensor = torch.tensor(masked_tokens, dtype=torch.long, device=device).unsqueeze(0)

        with torch.no_grad():
            logits = transformer(input_tensor)
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            pred_label = np.argmax(probabilities)

            loss = criterion(logits, torch.tensor([label], device=device))
            total_loss += loss.item()

        y_pred.append(pred_label)
        y_prob.append(probabilities[1]) 


    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    try:
        auroc = roc_auc_score(y_true, y_prob)
    except:
        auroc = 0.0

    try:
        auprc = average_precision_score(y_true, y_prob)
    except:
        auprc = 0.0

    avg_loss = total_loss / len(test_dataset)

    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "AUROC": auroc,
        "AUPRC": auprc,
        "Loss": avg_loss,
        "Confusion Matrix": cm.tolist()
    }



if __name__ == "__main__":
    def read_fasta(filepath):
        seqs = []
        with open(filepath, "r") as f:
            seq = ""
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if seq:
                        seqs.append(seq)
                        seq = ""
                else:
                    seq += line
            if seq:
                seqs.append(seq)
        return seqs

    amp_eval = read_fasta("AMP.eval.fa")
    decoy_eval = read_fasta("DECOY.eval.fa")

    test_dataset = [(seq, 1) for seq in amp_eval] + [(seq, 0) for seq in decoy_eval]

    results = evaluate_model(test_dataset)
    print("\n===== MODEL EVALUATION =====")
    for k, v in results.items():
        print(f"{k}: {v}")
