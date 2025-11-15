import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


def relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
    ret = 0
    n = -relative_position 

    max_exact = num_buckets // 2
    is_small = n < max_exact

    val_if_large = max_exact + (
        torch.log(n.float() / max_exact)
        / torch.log(torch.tensor(max_distance / max_exact))
        * (num_buckets - max_exact)
    ).long()

    val_if_large = torch.clamp(val_if_large, max=num_buckets - 1)

    ret = torch.where(is_small, n, val_if_large)
    ret = torch.clamp(ret, min=0)

    return ret



class T5Attention(nn.Module):
    def __init__(self, embed_size, num_heads, num_buckets=32, max_distance=128):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        self.q = nn.Linear(embed_size, embed_size)
        self.k = nn.Linear(embed_size, embed_size)
        self.v = nn.Linear(embed_size, embed_size)
        self.o = nn.Linear(embed_size, embed_size)

        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

        self.num_buckets = num_buckets
        self.max_distance = max_distance

    def forward(self, x):
        B, L, C = x.size()

        q = self.q(x).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        context = torch.arange(L, device=x.device)
        relative = context[None, :] - context[:, None] 
        bucket_ids = relative_position_bucket(
            relative, self.num_buckets, self.max_distance
        )  

        bias = self.relative_attention_bias(bucket_ids)  
        bias = bias.permute(2, 0, 1)                   
        attn_scores = attn_scores + bias.unsqueeze(0)    

        attn_weights = torch.softmax(attn_scores, dim=-1)

        out = attn_weights @ v

        out = out.transpose(1, 2).reshape(B, L, C)
        return self.o(out)



class TransformerBlock(nn.Module):
    def __init__(self, embed, heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed)
        self.attn = T5Attention(embed, heads)
        self.ln2 = nn.LayerNorm(embed)
        self.fc1 = nn.Linear(embed, embed * 4)
        self.fc2 = nn.Linear(embed * 4, embed)

    def forward(self, x):
        h = self.ln1(x)
        x = x + self.attn(h)

        h = self.ln2(x)
        h = F.gelu(self.fc1(h))
        h = self.fc2(h)

        x = x + h
        return x


class GPTClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size=400, context=312, n_heads=8, n_layers=8):
        super().__init__()

        self.embeddings = nn.Embedding(vocab_size, embed_size)

        self.positions = nn.Embedding(context, embed_size)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_size, n_heads) for _ in range(n_layers)
        ])

        self.ln = nn.LayerNorm(embed_size)
        self.classifier = nn.Linear(embed_size, 2)

    def forward(self, inp):
        B, L = inp.size()
        tok = self.embeddings(inp)

        pos = self.positions(torch.arange(L, device=inp.device))
        pos = pos.unsqueeze(0).expand(B, L, -1)

        x = tok + pos

        for block in self.blocks:
            x = block(x)

        x = self.ln(x)
        logits = self.classifier(x[:, -1, :])  
        return logits
