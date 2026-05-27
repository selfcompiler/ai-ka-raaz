"""
=============================================================================
 MINI-GPT — Everything Together: A Working Transformer Language Model
=============================================================================

 This combines EVERY concept you've learned:
   1. Tokenization        → character-level (from bigram_lm.py)
   2. Embeddings          → token + positional (from embeddings_deep_dive.py)
   3. Self-Attention       → Q, K, V with causal mask (from self_attention_deep_dive.py)
   4. Multi-Head Attention → multiple parallel attention heads
   5. Feed-Forward Network → per-token computation
   6. Transformer Blocks   → attention + FFN + residual + layernorm
   7. Stacking Blocks      → 6 layers deep
   8. Training Loop        → predict next character, backprop, repeat
   9. Text Generation      → autoregressive sampling

 Architecture (GPT-2 style, just smaller):
   - 6 transformer blocks
   - 6 attention heads
   - 384-dim embeddings
   - 256 character context window
   - ~10M parameters

=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import time

# ============================================================================
# HYPERPARAMETERS
# ============================================================================

# Model
n_embd = 384        # embedding dimension
n_head = 6          # number of attention heads
n_layer = 6         # number of transformer blocks
block_size = 256    # max context length (how far back the model can look)
dropout = 0.2       # dropout rate (regularization)

# Training
batch_size = 64     # sequences per batch
learning_rate = 3e-4
max_steps = 5000
eval_interval = 500
eval_iters = 200

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# ============================================================================
# STAGE 1: DATA — Load and Tokenize Shakespeare
# ============================================================================

print("\n" + "=" * 60)
print("STAGE 1: Loading Data + Tokenization")
print("=" * 60)

data_path = os.path.join(os.path.dirname(__file__), "shakespeare.txt")
with open(data_path, "r") as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [char_to_idx[c] for c in s]
decode = lambda l: "".join(idx_to_char[i] for i in l)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"  Dataset: {len(text):,} characters")
print(f"  Vocabulary: {vocab_size} unique characters")
print(f"  Train: {len(train_data):,}  |  Val: {len(val_data):,}")


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# ============================================================================
# STAGE 2: MODEL — The Complete Mini-GPT
# ============================================================================

print("\n" + "=" * 60)
print("STAGE 2: Building the Mini-GPT")
print("=" * 60)

print("""
  Architecture:
  ┌─────────────────────────────────────────────┐
  │  Token Embedding    [65 × 384]              │ ← learned from tokenization_deep_dive
  │  + Position Embedding [256 × 384]           │ ← learned from embeddings_deep_dive
  │                                             │
  │  ┌─ Transformer Block ×6 ──────────────┐   │
  │  │  LayerNorm                          │   │
  │  │  Multi-Head Attention (6 heads)     │   │ ← learned from self_attention_deep_dive
  │  │  + Residual Connection              │   │
  │  │  LayerNorm                          │   │
  │  │  Feed-Forward Network (384→1536→384)│   │
  │  │  + Residual Connection              │   │
  │  └────────────────────────────────────┘   │
  │                                             │
  │  Final LayerNorm                            │
  │  Linear → logits [384 → 65]                │
  └─────────────────────────────────────────────┘
""")


class Head(nn.Module):
    """Single head of self-attention."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)     # (B, T, head_size)
        q = self.query(x)   # (B, T, head_size)

        # Q × Kᵀ / √d_k  — the attention scores
        scores = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)

        # Causal mask — can't look at future tokens
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        # Softmax — convert to probabilities
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # × V — gather information
        v = self.value(x)
        return weights @ v


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """Per-token computation: expand → activate → contract."""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),    # expand 384 → 1536
            nn.GELU(),                          # activation
            nn.Linear(4 * n_embd, n_embd),     # contract 1536 → 384
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """One transformer block: attention + feed-forward + residuals."""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))     # attention + residual
        x = x + self.ffwd(self.ln2(x))   # feed-forward + residual
        return x


class MiniGPT(nn.Module):
    """The complete mini-GPT language model."""

    def __init__(self):
        super().__init__()
        # Embeddings (from embeddings_deep_dive.py)
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)

        # Transformer blocks (stacked)
        self.blocks = nn.Sequential(*[
            TransformerBlock(n_embd, n_head) for _ in range(n_layer)
        ])

        # Final layer norm + output projection
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Stage 1: Embeddings
        tok_emb = self.token_embedding(idx)             # (B, T, n_embd)
        pos_emb = self.position_embedding(torch.arange(T, device=device))  # (T, n_embd)
        x = tok_emb + pos_emb                           # (B, T, n_embd)

        # Stage 2: Transformer blocks (attention + FFN)
        x = self.blocks(x)                              # (B, T, n_embd)

        # Stage 3: Final projection to vocabulary
        x = self.ln_f(x)                                # (B, T, n_embd)
        logits = self.lm_head(x)                        # (B, T, vocab_size)

        if targets is None:
            return logits, None

        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressive generation: predict one token at a time."""
        for _ in range(max_new_tokens):
            # Crop context to block_size
            idx_cond = idx[:, -block_size:]
            # Forward pass
            logits, _ = self(idx_cond)
            # Take last token's predictions
            logits = logits[:, -1, :] / temperature

            # Optional top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Sample from probability distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


model = MiniGPT().to(device)
total_params = sum(p.numel() for p in model.parameters())

print(f"  Model built!")
print(f"  Total parameters: {total_params:,}")
print(f"\n  Parameter breakdown:")
tok_params = vocab_size * n_embd
pos_params = block_size * n_embd
print(f"    Token embeddings:      {vocab_size:>5} × {n_embd} = {tok_params:>10,}")
print(f"    Position embeddings:   {block_size:>5} × {n_embd} = {pos_params:>10,}")
block_params = sum(p.numel() for p in model.blocks[0].parameters())
print(f"    Per transformer block:               {block_params:>10,}")
print(f"    × {n_layer} blocks:                         {block_params * n_layer:>10,}")
lm_head_params = n_embd * vocab_size
print(f"    Output projection:     {n_embd:>5} × {vocab_size:<3} = {lm_head_params:>10,}")
print(f"    {'─' * 42}")
print(f"    TOTAL:                              {total_params:>10,}")


# ============================================================================
# STAGE 3: GENERATE BEFORE TRAINING (random output)
# ============================================================================

print("\n" + "=" * 60)
print("STAGE 3: Text BEFORE Training (random garbage)")
print("=" * 60)

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=200)[0].tolist()))


# ============================================================================
# STAGE 4: TRAINING
# ============================================================================

print("\n" + "=" * 60)
print("STAGE 4: Training the Mini-GPT")
print("=" * 60)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


print(f"  Training for {max_steps:,} steps...")
print(f"  Batch size: {batch_size}, Context length: {block_size}")
print(f"  Learning rate: {learning_rate}")
print(f"\n  {'Step':>6} │ {'Train Loss':>10} │ {'Val Loss':>10} │ {'Time':>8}")
print(f"  {'─'*6} │ {'─'*10} │ {'─'*10} │ {'─'*8}")

start_time = time.time()
for step in range(max_steps):
    if step % eval_interval == 0 or step == max_steps - 1:
        losses = estimate_loss()
        elapsed = time.time() - start_time
        print(f"  {step:>6} │ {losses['train']:>10.4f} │ {losses['val']:>10.4f} │ {elapsed:>7.1f}s")

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

total_time = time.time() - start_time
print(f"\n  Training complete! Total time: {total_time:.1f}s")


# ============================================================================
# STAGE 5: GENERATE AFTER TRAINING
# ============================================================================

print("\n" + "=" * 60)
print("STAGE 5: Text AFTER Training — Shakespeare-style!")
print("=" * 60)

print("\n--- Sample 1: Free generation (temperature=0.8) ---\n")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=500, temperature=0.8, top_k=40)
print(decode(generated[0].tolist()))

print("\n--- Sample 2: Starting with a prompt ---\n")
prompt = "ROMEO:\nO, "
prompt_encoded = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
generated = model.generate(prompt_encoded, max_new_tokens=300, temperature=0.8, top_k=40)
print(decode(generated[0].tolist()))

print("\n--- Sample 3: Lower temperature (more conservative) ---\n")
prompt = "KING HENRY:\n"
prompt_encoded = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
generated = model.generate(prompt_encoded, max_new_tokens=300, temperature=0.5, top_k=20)
print(decode(generated[0].tolist()))


# ============================================================================
# STAGE 6: WHAT YOU BUILT
# ============================================================================

print("\n" + "=" * 60)
print("WHAT YOU BUILT — The Complete Picture")
print("=" * 60)

print(f"""
  You built a {total_params:,}-parameter GPT language model!

  ┌─────────────────────────────────────────────────────────────┐
  │  "ROMEO:\\nO, "                                              │
  │      ↓                                                      │
  │  TOKENIZATION (char-level)                                  │
  │      ['R','O','M','E','O',':','\\n','O',',',' ']            │
  │      → [44, 37, 35, 17, 37, 12, 0, 37, 6, 1]              │
  │      ↓                                                      │
  │  TOKEN EMBEDDING    [65 vocab × {n_embd} dims]               │
  │      Each token ID → {n_embd}-dim vector                      │
  │      ↓                                                      │
  │  + POSITION EMBEDDING  [{block_size} positions × {n_embd} dims]       │
  │      Each position → {n_embd}-dim vector                      │
  │      ↓                                                      │
  │  TRANSFORMER BLOCK ×{n_layer}                                    │
  │  ┌──────────────────────────────────────────────────────┐   │
  │  │  LayerNorm                                          │   │
  │  │  Multi-Head Self-Attention ({n_head} heads × {n_embd//n_head} dims)     │   │
  │  │    Q × Kᵀ / √d_k → softmax → × V                  │   │
  │  │    + Causal mask (can't see future)                 │   │
  │  │  + Residual                                         │   │
  │  │  LayerNorm                                          │   │
  │  │  Feed-Forward ({n_embd} → {4*n_embd} → {n_embd})                │   │
  │  │  + Residual                                         │   │
  │  └──────────────────────────────────────────────────────┘   │
  │      ↓                                                      │
  │  FINAL LAYERNORM                                            │
  │      ↓                                                      │
  │  LINEAR PROJECTION  [{n_embd} → {vocab_size} vocab]                 │
  │      → logits (score for each possible next character)      │
  │      ↓                                                      │
  │  SOFTMAX → probabilities                                    │
  │      ↓                                                      │
  │  SAMPLE next token                                          │
  │      ↓                                                      │
  │  APPEND to sequence, REPEAT                                 │
  │      ↓                                                      │
  │  "ROMEO:\\nO, what light through yonder..."                  │
  └─────────────────────────────────────────────────────────────┘

  COMPARISON TO REAL MODELS:

  ┌─────────────────┬──────────────┬───────────────┬───────────────┐
  │                 │  Your Model  │   GPT-2 Small │    GPT-3      │
  ├─────────────────┼──────────────┼───────────────┼───────────────┤
  │ Parameters      │  {total_params/1e6:>5.1f}M      │      117M     │     175B      │
  │ Layers          │      {n_layer}       │       12      │      96       │
  │ Heads           │      {n_head}       │       12      │      96       │
  │ Embed dim       │    {n_embd}       │      768      │   12,288      │
  │ Context window  │    {block_size}       │    1,024      │    2,048      │
  │ Training data   │    ~1MB      │     40GB      │   570GB       │
  │ Training time   │   ~{total_time:.0f}s       │    ~days      │   ~months     │
  └─────────────────┴──────────────┴───────────────┴───────────────┘

  YOUR LEARNING JOURNEY (complete!):

    bigram_lm.py                → simplest possible language model
    tokenization_deep_dive.py   → text → numbers (BPE)
    embeddings_deep_dive.py     → numbers → meaningful vectors
    cosine_similarity_explained.py → measuring vector similarity
    cosine_problems.py          → limitations and real-world issues
    self_attention_deep_dive.py → the core of transformers (Q,K,V)
    attention_math_explained.py → the formula step by step
    mini_gpt.py                 → EVERYTHING TOGETHER ← YOU ARE HERE

  You now understand how large language models work from first principles!
""")
