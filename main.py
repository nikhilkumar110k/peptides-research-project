import torch
import sentencepiece as spm
import numpy as np
from selfattention import GPTClassifier
from rlbasedmodel import MaskingDQNAgent

device = "cuda" if torch.cuda.is_available() else "cpu"

sp = spm.SentencePieceProcessor(model_file="peptide_tokenizer.model")
vocab_size = sp.GetPieceSize()


valid_aas = set("ACDEFGHIKLMNPQRSTVWY")

def is_valid_peptide(seq):
    return all(residue in valid_aas for residue in seq.upper())
transformer = GPTClassifier(vocab_size).to(device)
transformer.load_state_dict(torch.load("transformer_masked_checkpoint_1400.pt", map_location=device))
transformer.eval()

agent = MaskingDQNAgent(state_size=312, action_size=312)
agent.load("masking_agent_checkpoint_1400.pt")


def predict_user_sequence(sequence: str):
    
    tokens = sp.Encode(sequence)
    tokens = tokens[:312]  
    tokens += [0] * (312 - len(tokens))
    state = np.array(tokens)

    mask = agent.act(state)
    masked_input = [t if m == 0 else 0 for t, m in zip(state, mask)]

    input_tensor = torch.tensor(masked_input, dtype=torch.long, device=device).unsqueeze(0)
    logits = transformer(input_tensor)
    pred_label = torch.argmax(logits, dim=-1).item()

    return {
        "mask": mask,
        "masked_input_tokens": masked_input,
        "predicted_label": pred_label
    }

if __name__ == "__main__":
    seq = input("Enter peptide sequence (or 'exit' to quit): ")
    if not is_valid_peptide(seq):
        print("Predicted Label: 0 (invalid peptide)")
        exit()
    tokens = sp.Encode(seq)[:312]
    tokens += [0] * (312 - len(tokens))
    state = np.array(tokens)

    mask = agent.act(state)
    masked_input = [t if m == 0 else 0 for t, m in zip(state, mask)]
    input_tensor = torch.tensor(masked_input, dtype=torch.long, device=device).unsqueeze(0)

    transformer.eval()
    with torch.no_grad():
        pred_label = torch.argmax(transformer(input_tensor), dim=-1).item()

    unmasked_input = torch.tensor(state, dtype=torch.long, device=device).unsqueeze(0)
    logits = transformer(unmasked_input)
    pred_label = torch.argmax(logits, dim=-1).item()
    print(f"Unmasked Prediction: {pred_label}")

    print(f"Predicted Label: {pred_label}")
    print(f"Mask applied by agent: {mask}")
    print(f"Masked Input Tokens: {masked_input}")
