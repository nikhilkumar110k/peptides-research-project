# 🧬 Peptide Binary Classification Transformer with RL Masking

A custom-built **Transformer model (from scratch in PyTorch)** for **peptide binary classification**.  
The system integrates a **self-attention model with relative positional embeddings** and a **DQN-based reinforcement learning agent** that dynamically masks irrelevant amino acids to improve classification performance.

---

## 📌 Overview

- **Transformer Model**: Built from scratch with PyTorch (custom QKV attention, feedforward, and relative positional bucket-based embeddings like R5).  
- **Tokenizer**: SentencePiece tokenizer for amino acid sequences → generates subword tokens and vocab for embeddings.  
- **DQN Agent**: Reinforcement learning agent that learns to mask non-important amino acids.  
- **Training Pipeline**:  
  1. **Stage 1** → Pre-train Transformer on unmasked dataset.  
  2. **Stage 2** → Train with DQN agent, where the agent masks sequences (1 = masked) and the Transformer learns classification.  
- **Exploration Strategy**: Epsilon-greedy method for balancing exploration vs. exploitation in RL.  

---

## 🛠 Tech Stack & Components

### 🔹 Core
- **PyTorch** → Transformer model and DQN agent implementation.
- **SentencePiece** → Tokenization and vocabulary generation for amino acids.
- **NumPy / Pandas** → Data preprocessing.

### 🔹 Transformer Model
- Custom **QKV (Query-Key-Value) Attention** implementation.
- **Relative Positional Embeddings (bucket-based)** for order awareness.
- **Binary classification head** on top of encoder.

### 🔹 DQN Agent
- Learns to mask irrelevant tokens (amino acids).
- Uses **Q-learning** with replay buffer.
- Implements **epsilon-greedy strategy** for exploration/exploitation.
- Provides **masked input** to Transformer during fine-tuning.

