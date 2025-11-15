# rlbasedmodel.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

device = "cuda" if torch.cuda.is_available() else "cpu"

class MaskingDQNNet(nn.Module):
    def __init__(self, state_size, action_size, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_size)   # outputs raw logits for each action
        )

    def forward(self, x):
        return self.net(x)  # raw logits; we'll apply sigmoid where needed


class MaskingDQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        memory_size=2000,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        lr=1e-3
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.model = MaskingDQNNet(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.BCEWithLogitsLoss()

    def remember(self, state, action, reward, next_state, done):
        
        self.memory.append((state.astype(np.float32), action.astype(np.float32), float(reward), next_state.astype(np.float32), bool(done)))

    def act(self, state):
        
        if np.random.rand() <= self.epsilon:
            return np.random.randint(0, 2, size=self.action_size).astype(np.int32)

        self.model.eval()
        with torch.no_grad():
            s = torch.from_numpy(state.astype(np.float32)).to(device).unsqueeze(0)  # (1, state_size)
            logits = self.model(s).squeeze(0)  # (action_size,)
            probs = torch.sigmoid(logits).cpu().numpy()
            mask = (probs > 0.5).astype(np.int32)
        return mask

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        
        states = np.stack([m[0] for m in minibatch])        # (B, S)
        actions = np.stack([m[1] for m in minibatch])       # (B, A)
        rewards = np.array([m[2] for m in minibatch], dtype=np.float32)   # (B,)
        next_states = np.stack([m[3] for m in minibatch])  # (B, S)
        dones = np.array([m[4] for m in minibatch], dtype=np.bool_)      # (B,)

        states_t = torch.from_numpy(states).to(device)            # (B, S)
        next_states_t = torch.from_numpy(next_states).to(device)  # (B, S)
        actions_t = torch.from_numpy(actions).to(device)          # (B, A)

        # Current preds (logits) -> convert to probabilities with sigmoid when interpreting, but BCEWithLogitsLoss accepts logits.
        preds_logits = self.model(states_t)  # (B, A)

        # Compute target per-action vector:
        # The original TF implementation used target = reward (if done) else reward + gamma * next_q
        # Here we approximate next_q as sigmoid(next_logits). Use next_q per action.
        with torch.no_grad():
            next_logits = self.model(next_states_t)  # (B, A)
            next_q = torch.sigmoid(next_logits)     # (B, A)

        # Broadcast rewards to shape (B, A)
        rewards_t = torch.from_numpy(rewards).to(device).unsqueeze(1).expand(-1, self.action_size)  # (B, A)

        # Targets: if done -> all entries = reward, else reward + gamma * next_q (per-action)
        targets = rewards_t.clone()
        not_done_mask = (~torch.from_numpy(dones)).to(device)
        if not_done_mask.any():
            nd_idx = not_done_mask.nonzero(as_tuple=False).squeeze(1)
            targets[nd_idx] = rewards_t[nd_idx] + self.gamma * next_q[nd_idx]

        # Train with BCEWithLogitsLoss: target values should be in [0,1] (rewards may not be in that range).
        # To keep similar behavior to the TF version (which fit raw target values with binary_crossentropy),
        # we'll clamp targets to [0,1] which works as pseudo-probabilities for actions.
        targets = torch.clamp(targets, 0.0, 1.0)

        self.model.train()
        self.optimizer.zero_grad()
        loss = self.criterion(preds_logits, targets)
        loss.backward()
        self.optimizer.step()

        # decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path):
        """
        Save model weights and optimizer + epsilon state
        """
        state = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epsilon": self.epsilon
        }
        torch.save(state, path)

    def load(self, path):
        """
        Load saved checkpoint. Accepts a PyTorch .pt / .pth file saved with save().
        """
        ckpt = torch.load(path, map_location=device)
        # If file was Keras previously, user must re-save in PyTorch format. We assume PyTorch format here.
        if "model_state" in ckpt:
            self.model.load_state_dict(ckpt["model_state"])
            if "optimizer_state" in ckpt:
                try:
                    self.optimizer.load_state_dict(ckpt["optimizer_state"])
                except Exception:
                    # Optimizer state may be incompatible across torch versions/devices; ignore if so.
                    pass
            if "epsilon" in ckpt:
                self.epsilon = ckpt["epsilon"]
        else:
            # If a raw state_dict was provided
            self.model.load_state_dict(ckpt)
