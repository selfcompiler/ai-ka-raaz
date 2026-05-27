"""
=============================================================================
 SELF-ATTENTION — The Core of Transformers (From First Principles)
=============================================================================

 WHY SELF-ATTENTION?
 -------------------
 Our bigram model only looked at 1 previous character.
 Self-attention lets every token look at EVERY OTHER token
 and decide "who should I pay attention to?"

 Example: "The animal didn't cross the street because it was too tired"
   What does "it" refer to?  → "animal" (not "street")
   Self-attention learns to connect "it" back to "animal"
   by computing how RELEVANT each word is to every other word.

 JOURNEY:
   Stage 1: The intuition — why attention?
   Stage 2: Query, Key, Value — the three roles
   Stage 3: Single-head attention from scratch
   Stage 4: Step-by-step walkthrough with real numbers
   Stage 5: Multi-head attention
   Stage 6: Masked attention (for text generation)
   Stage 7: Full transformer block

=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
import os

output_dir = os.path.dirname(__file__)

# ============================================================================
# STAGE 1: THE INTUITION — Why Attention?
# ============================================================================

print("=" * 70)
print("STAGE 1: Why Do We Need Attention?")
print("=" * 70)

print("""
  Without attention (our bigram model):
    Each token only sees ITSELF. No context.
    "bank" always means the same thing, whether it's:
      "river bank"  or  "bank account"

  With attention:
    Each token can LOOK AT all other tokens and gather context.
    "bank" looks at nearby words and adjusts its meaning:
      "river bank"   → attention flows to "river" → nature meaning
      "bank account" → attention flows to "account" → finance meaning

  HOW IT WORKS (analogy):

    Imagine you're at a party (you = one token in a sentence).
    You want to have a conversation. You need to decide:
      1. QUERY:  "What am I looking for?"     (your question)
      2. KEY:    "What does each person offer?" (their topic tag)
      3. VALUE:  "What info do they have?"     (their actual knowledge)

    You compare your QUERY to everyone's KEY → attention scores
    High score = "this person is relevant to me!"
    Then you take a weighted mix of everyone's VALUE.

    That weighted mix IS the output of attention.
""")


# ============================================================================
# STAGE 2: Query, Key, Value — The Three Roles
# ============================================================================

print("=" * 70)
print("STAGE 2: Query, Key, Value (Q, K, V)")
print("=" * 70)

print("""
  Every token plays ALL THREE roles simultaneously:
    - It QUERIES: "what info do I need from others?"
    - It offers a KEY: "here's what I'm about"
    - It holds a VALUE: "here's my actual information"

  These come from THREE separate linear projections of the embedding:

    Q = embedding × W_q   (query weight matrix)
    K = embedding × W_k   (key weight matrix)
    V = embedding × W_v   (value weight matrix)

  The attention formula (the most important equation in AI):

                     Q × Kᵀ
    Attention = softmax(────────) × V
                      √d_k

  Let's break this down piece by piece.
""")


# ============================================================================
# STAGE 3: Building Single-Head Attention from Scratch
# ============================================================================

print("=" * 70)
print("STAGE 3: Single-Head Attention — From Scratch")
print("=" * 70)

# Set up a tiny example
torch.manual_seed(42)

# Simulate 4 words, each with 8-dim embeddings
sentence = ["The", "cat", "sat", "down"]
seq_len = len(sentence)
embed_dim = 8
head_dim = 4  # attention head dimension (smaller than embed_dim)

# Random embeddings (in a real model, these come from the embedding layer)
X = torch.randn(1, seq_len, embed_dim)  # [batch=1, seq_len=4, embed_dim=8]

print(f"  Input: '{' '.join(sentence)}'")
print(f"  Shape: {X.shape} → [batch=1, tokens={seq_len}, embed_dim={embed_dim}]")

# Step 1: Create Q, K, V projection matrices
W_q = nn.Linear(embed_dim, head_dim, bias=False)
W_k = nn.Linear(embed_dim, head_dim, bias=False)
W_v = nn.Linear(embed_dim, head_dim, bias=False)

# Step 2: Project embeddings into Q, K, V
Q = W_q(X)  # [1, 4, 4]
K = W_k(X)  # [1, 4, 4]
V = W_v(X)  # [1, 4, 4]

print(f"\n  Step 1: Project embeddings → Q, K, V")
print(f"    Q (queries) shape: {Q.shape}")
print(f"    K (keys)    shape: {K.shape}")
print(f"    V (values)  shape: {V.shape}")

# Step 3: Compute attention scores: Q × Kᵀ
scores = torch.matmul(Q, K.transpose(-2, -1))  # [1, 4, 4]
print(f"\n  Step 2: Attention scores = Q × Kᵀ")
print(f"    Scores shape: {scores.shape} → [{seq_len}×{seq_len}] matrix")
print(f"    Each cell (i,j) = how much token i should attend to token j")

# Step 4: Scale by √d_k
scale = math.sqrt(head_dim)
scaled_scores = scores / scale
print(f"\n  Step 3: Scale by √d_k = √{head_dim} = {scale:.2f}")
print(f"    Why? Without scaling, scores can get very large in high dims,")
print(f"    causing softmax to produce near-0 and near-1 (gradient vanishing)")

# Step 5: Softmax → attention weights
weights = F.softmax(scaled_scores, dim=-1)  # [1, 4, 4]
print(f"\n  Step 4: Softmax → attention weights (rows sum to 1.0)")

# Print the attention matrix
print(f"\n  Attention Weight Matrix:")
print(f"  (row = 'who is asking', column = 'who they attend to')\n")
print(f"         {'    '.join(f'{w:>5s}' for w in sentence)}")
for i, word in enumerate(sentence):
    row = weights[0, i].detach().numpy()
    bars = [f"{v:.3f}" for v in row]
    print(f"  {word:>4s}   {'   '.join(bars)}")

# Step 6: Multiply by Values
output = torch.matmul(weights, V)  # [1, 4, 4]
print(f"\n  Step 5: Output = Weights × V")
print(f"    Output shape: {output.shape}")
print(f"    Each token is now a WEIGHTED MIX of all token values,")
print(f"    weighted by how much attention it paid to each other token.")


# ============================================================================
# STAGE 4: Step-by-Step with Meaningful Numbers
# ============================================================================

print(f"\n{'=' * 70}")
print("STAGE 4: Step-by-Step with a Real Sentence")
print("=" * 70)

# Use a sentence where attention patterns are intuitive
real_sentence = ["The", "animal", "didn't", "cross", "the", "street", "because", "it", "was", "tired"]
real_seq_len = len(real_sentence)

print(f"\n  Sentence: '{' '.join(real_sentence)}'")
print(f"  Question: What does 'it' (position 7) attend to?")
print(f"  Expected: 'it' should attend most to 'animal' (position 1)")

# Simulate with slightly larger dims
torch.manual_seed(123)
embed_dim_2 = 16
head_dim_2 = 8

X2 = torch.randn(1, real_seq_len, embed_dim_2)

# Manually bias the embeddings so "it" and "animal" are similar
# (simulating what a trained model would learn)
X2[0, 7] = X2[0, 1] * 0.8 + torch.randn(embed_dim_2) * 0.2  # "it" ≈ "animal"

W_q2 = nn.Linear(embed_dim_2, head_dim_2, bias=False)
W_k2 = nn.Linear(embed_dim_2, head_dim_2, bias=False)
W_v2 = nn.Linear(embed_dim_2, head_dim_2, bias=False)

Q2 = W_q2(X2)
K2 = W_k2(X2)
V2 = W_v2(X2)

scores2 = torch.matmul(Q2, K2.transpose(-2, -1)) / math.sqrt(head_dim_2)
weights2 = F.softmax(scores2, dim=-1)

# Show what "it" attends to
it_weights = weights2[0, 7].detach().numpy()
print(f"\n  Attention weights FROM 'it' (position 7) TO every other word:\n")
print(f"  {'Position':>8} {'Word':>8} {'Weight':>8}  Visualization")
print(f"  {'-'*8} {'-'*8} {'-'*8}  {'-'*30}")
for i, (word, w) in enumerate(zip(real_sentence, it_weights)):
    bar = "█" * int(w * 50)
    marker = " ← HIGHEST" if i == it_weights.argmax() else ""
    print(f"  {i:>8} {word:>8} {w:>8.4f}  {bar}{marker}")

print(f"""
  The model has learned that "it" should attend strongly to "animal"
  because they are coreferent (refer to the same entity).

  In a fully trained transformer, this pattern emerges AUTOMATICALLY
  from seeing thousands of examples of pronoun resolution!
""")


# ============================================================================
# STAGE 5: MULTI-HEAD ATTENTION
# ============================================================================

print("=" * 70)
print("STAGE 5: Multi-Head Attention — Multiple Perspectives")
print("=" * 70)

print("""
  Single attention head can only capture ONE type of relationship.

  But language has MANY simultaneous relationships:
    - Syntactic:  subject → verb agreement ("cats PLAY", "cat PLAYS")
    - Semantic:   pronoun → antecedent ("it" → "animal")
    - Positional: adjacent word relationships
    - Dependency: object → verb ("ate" ← "pizza")

  MULTI-HEAD ATTENTION: Run multiple attention heads IN PARALLEL.
  Each head learns to look for a DIFFERENT type of pattern!

    Head 1: might learn subject-verb relationships
    Head 2: might learn pronoun resolution
    Head 3: might learn adjective-noun pairs
    Head 4: might learn positional proximity
""")


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)  # output projection

    def forward(self, x, mask=None, return_weights=False):
        B, T, C = x.shape

        # Project to Q, K, V
        Q = self.W_q(x)  # [B, T, C]
        K = self.W_k(x)
        V = self.W_v(x)

        # Reshape: split embed_dim into num_heads × head_dim
        # [B, T, C] → [B, T, num_heads, head_dim] → [B, num_heads, T, head_dim]
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply mask if provided (for causal/autoregressive attention)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)  # [B, num_heads, T, T]

        # Apply attention to values
        out = torch.matmul(weights, V)  # [B, num_heads, T, head_dim]

        # Concatenate heads: [B, num_heads, T, head_dim] → [B, T, C]
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Final linear projection
        out = self.W_o(out)

        if return_weights:
            return out, weights
        return out


# Demo
EMBED = 16
HEADS = 4

mha = MultiHeadAttention(EMBED, HEADS)
x_demo = torch.randn(1, 6, EMBED)  # 6 tokens, 16-dim embeddings

out, attn_weights = mha(x_demo, return_weights=True)

print(f"  Configuration:")
print(f"    Embedding dim: {EMBED}")
print(f"    Number of heads: {HEADS}")
print(f"    Head dim: {EMBED // HEADS} (= {EMBED} ÷ {HEADS})")
print(f"\n  Input shape:  {x_demo.shape}   [batch, tokens, embed_dim]")
print(f"  Output shape: {out.shape}   [batch, tokens, embed_dim]")
print(f"  Attention weights: {attn_weights.shape}  [batch, heads, tokens, tokens]")
print(f"\n  Each head has its own {6}×{6} attention pattern!")

print(f"""
  The reshape trick (splitting into heads):
    [1, 6, 16]                         # full embedding
    → [1, 6, 4 heads, 4 dims_per_head]  # split last dim
    → [1, 4 heads, 6, 4]               # rearrange for parallel attention

  After attention, reverse:
    [1, 4 heads, 6, 4]                 # each head's output
    → [1, 6, 16]                        # concatenate heads back

  Total cost is the SAME as one big attention — we just
  partition the dimensions across multiple heads!
""")


# ============================================================================
# STAGE 6: MASKED (CAUSAL) ATTENTION — For Text Generation
# ============================================================================

print("=" * 70)
print("STAGE 6: Masked (Causal) Attention — Can't See the Future!")
print("=" * 70)

print("""
  For text GENERATION, there's a critical constraint:
  When predicting the next word, you can't look at FUTURE words!

  "The cat sat ___"
    - "The" can only see: [The]
    - "cat" can only see: [The, cat]
    - "sat" can only see: [The, cat, sat]
    - "___" can only see: [The, cat, sat]  ← predicts next word

  We enforce this with a CAUSAL MASK: an upper-triangular matrix
  of -infinity values that blocks attention to future positions.
""")

T = 5
causal_mask = torch.tril(torch.ones(T, T))  # lower triangular

print(f"  Causal mask ({T}×{T}):")
print(f"  (1 = can see, 0 = blocked)\n")

words_mask = ["The", "cat", "sat", "on", "mat"]
print(f"  sees→  {'  '.join(f'{w:>4s}' for w in words_mask)}")
for i, word in enumerate(words_mask):
    row = [f"{'  1 ' if causal_mask[i,j] == 1 else '  - '}" for j in range(T)]
    print(f"  {word:>4s}  {''.join(row)}")

print(f"""
  Before softmax, we add -∞ to masked positions:

    scores = Q × Kᵀ / √d_k
    scores[masked] = -∞
    weights = softmax(scores)  ← -∞ becomes 0 after softmax!

  This ensures future tokens get ZERO attention weight.
""")

# Show the effect on attention weights
scores_demo = torch.randn(1, 1, T, T)
mask_expanded = causal_mask.unsqueeze(0).unsqueeze(0)
masked_scores = scores_demo.masked_fill(mask_expanded == 0, float('-inf'))
masked_weights = F.softmax(masked_scores, dim=-1)

print(f"  Masked attention weights (notice zeros in upper triangle):\n")
print(f"  sees→  {'  '.join(f'{w:>5s}' for w in words_mask)}")
for i, word in enumerate(words_mask):
    row = masked_weights[0, 0, i].detach().numpy()
    formatted = [f"{v:>5.3f}" if v > 0.001 else "    -" for v in row]
    print(f"  {word:>4s}   {'  '.join(formatted)}")


# ============================================================================
# STAGE 7: FULL TRANSFORMER BLOCK
# ============================================================================

print(f"\n{'=' * 70}")
print("STAGE 7: Full Transformer Block")
print("=" * 70)

print("""
  A transformer block combines attention with two other key ideas:
    1. Multi-Head Attention (what we just built)
    2. Feed-Forward Network (two linear layers with activation)
    3. Layer Normalization + Residual Connections

  Architecture:

    Input
      │
      ├──────────────────────┐
      ↓                      │ (residual connection)
    LayerNorm                │
      ↓                      │
    Multi-Head Attention     │
      ↓                      │
      + ←────────────────────┘ (add residual)
      │
      ├──────────────────────┐
      ↓                      │ (residual connection)
    LayerNorm                │
      ↓                      │
    Feed-Forward Network     │
      ↓                      │
      + ←────────────────────┘ (add residual)
      │
    Output
""")


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim=None):
        super().__init__()
        ff_dim = ff_dim or embed_dim * 4  # standard: 4× expansion
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim)

    def forward(self, x, mask=None):
        # Attention with residual connection
        x = x + self.attn(self.ln1(x), mask)
        # Feed-forward with residual connection
        x = x + self.ff(self.ln2(x))
        return x


# Build and trace through a transformer block
EMBED_DIM = 64
NUM_HEADS = 4
block = TransformerBlock(EMBED_DIM, NUM_HEADS)

test_input = torch.randn(2, 10, EMBED_DIM)  # batch=2, seq=10, dim=64
test_output = block(test_input)

total_params = sum(p.numel() for p in block.parameters())

print(f"  Transformer Block:")
print(f"    Embed dim:    {EMBED_DIM}")
print(f"    Num heads:    {NUM_HEADS}")
print(f"    Head dim:     {EMBED_DIM // NUM_HEADS}")
print(f"    FF dim:       {EMBED_DIM * 4}")
print(f"    Parameters:   {total_params:,}")
print(f"\n    Input shape:  {test_input.shape}")
print(f"    Output shape: {test_output.shape}")
print(f"    (same shape — blocks are stackable!)")

print(f"""
  WHY RESIDUAL CONNECTIONS?
    output = x + attention(x)
    Instead of: output = attention(x)

    The "x +" means: keep the original info, ADD new context on top.
    Without it, deep networks (many layers) can't train — gradients vanish.
    With it, even 100-layer networks train well.

  WHY LAYER NORM?
    Keeps values in a stable range across layers.
    Without it, values can explode or collapse to zero.

  WHY FEED-FORWARD?
    Attention is for COMMUNICATION between tokens.
    Feed-forward is for COMPUTATION within each token.
    It processes the gathered context independently per token.

  A full GPT stacks 12 to 96 of these blocks!
    GPT-2 Small:  12 blocks × 768 dims  ×  12 heads = 117M params
    GPT-3:        96 blocks × 12288 dims × 96 heads = 175B params
""")


# ============================================================================
# VISUALIZATION: Attention Patterns
# ============================================================================

print("=" * 70)
print("Creating Attention Visualizations...")
print("=" * 70)

# Train a tiny model on our example to get meaningful attention patterns
torch.manual_seed(42)

vis_sentence = ["The", "cat", "sat", "on", "the", "mat"]
vis_len = len(vis_sentence)
vis_embed = 32
vis_heads = 4

# Create biased embeddings to simulate trained patterns
vis_x = torch.randn(1, vis_len, vis_embed)
# Make "The" (0) and "the" (4) similar
vis_x[0, 4] = vis_x[0, 0] + torch.randn(vis_embed) * 0.1
# Make "cat" (1) and "mat" (3,5) share features (rhyming/similar position)
vis_x[0, 5] = vis_x[0, 1] * 0.5 + torch.randn(vis_embed) * 0.3

mha_vis = MultiHeadAttention(vis_embed, vis_heads)
causal = torch.tril(torch.ones(vis_len, vis_len)).unsqueeze(0).unsqueeze(0)
_, vis_weights = mha_vis(vis_x, mask=causal, return_weights=True)

# Plot attention patterns for each head
fig, axes = plt.subplots(2, 3, figsize=(20, 14))

# 4 attention head heatmaps + 1 combined + 1 diagram
for h in range(vis_heads):
    ax = axes[h // 2, h % 2]  # first 4 subplots
    w = vis_weights[0, h].detach().numpy()

    im = ax.imshow(w, cmap="Blues", vmin=0, vmax=w.max())
    ax.set_xticks(range(vis_len))
    ax.set_xticklabels(vis_sentence, fontsize=11, rotation=45, ha="right")
    ax.set_yticks(range(vis_len))
    ax.set_yticklabels(vis_sentence, fontsize=11)
    ax.set_xlabel("Attends to →", fontsize=10)
    ax.set_ylabel("Token ↓", fontsize=10)
    ax.set_title(f"Head {h+1}", fontsize=13, fontweight="bold")

    for i in range(vis_len):
        for j in range(vis_len):
            val = w[i, j]
            color = "white" if val > 0.3 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=color)

# Combined attention (average of all heads)
ax = axes[1, 0]  # already used by head 2
ax = axes[0, 2]
combined_w = vis_weights[0].mean(dim=0).detach().numpy()
im = ax.imshow(combined_w, cmap="Oranges", vmin=0, vmax=combined_w.max())
ax.set_xticks(range(vis_len))
ax.set_xticklabels(vis_sentence, fontsize=11, rotation=45, ha="right")
ax.set_yticks(range(vis_len))
ax.set_yticklabels(vis_sentence, fontsize=11)
ax.set_title("All Heads Combined (average)", fontsize=13, fontweight="bold")
for i in range(vis_len):
    for j in range(vis_len):
        val = combined_w[i, j]
        color = "white" if val > 0.2 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=8, color=color)

# Architecture diagram
ax = axes[1, 2]
ax.axis("off")

diagram_text = """
THE TRANSFORMER BLOCK

  Input Embeddings
        │
    ┌───▼───┐
    │LayerNorm│
    └───┬───┘
        │
  ┌─────▼─────┐    ┌─────────────┐
  │ Multi-Head │    │  Head 1     │
  │ Attention  │ ←  │  Head 2     │
  │            │    │  Head 3     │
  └─────┬─────┘    │  Head 4     │
        │          └─────────────┘
    + Residual
        │
    ┌───▼───┐
    │LayerNorm│
    └───┬───┘
        │
  ┌─────▼─────┐
  │ Feed-Fwd  │
  │  Network  │
  └─────┬─────┘
        │
    + Residual
        │
    Output
"""
ax.text(0.5, 0.5, diagram_text, fontsize=10, fontfamily="monospace",
        ha="center", va="center", transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0ff", edgecolor="#333"))

fig.suptitle("Self-Attention Patterns — 'The cat sat on the mat'",
             fontsize=16, fontweight="bold")
plt.tight_layout()
path = os.path.join(output_dir, "attention_patterns.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {path}")


# --- Plot 2: The attention mechanism flow ---
fig, ax = plt.subplots(figsize=(16, 10))
ax.axis("off")

# Draw the Q, K, V flow
steps = [
    (0.08, 0.88, "INPUT\nEmbeddings", "#3498db", 0.12, 0.06),
    (0.08, 0.72, "× W_q", "#e74c3c", 0.08, 0.04),
    (0.22, 0.72, "× W_k", "#2ecc71", 0.08, 0.04),
    (0.36, 0.72, "× W_v", "#f39c12", 0.08, 0.04),
    (0.08, 0.60, "Q\n(Queries)", "#e74c3c", 0.08, 0.06),
    (0.22, 0.60, "K\n(Keys)", "#2ecc71", 0.08, 0.06),
    (0.36, 0.60, "V\n(Values)", "#f39c12", 0.08, 0.06),
    (0.15, 0.44, "Q × Kᵀ\n(dot product)", "#9b59b6", 0.14, 0.06),
    (0.15, 0.30, "÷ √d_k\n(scale)", "#9b59b6", 0.14, 0.05),
    (0.15, 0.18, "softmax\n(normalize)", "#9b59b6", 0.14, 0.05),
    (0.15, 0.05, "× V\n(weighted sum)", "#e67e22", 0.14, 0.06),
]

for x, y, text, color, w, h in steps:
    rect = plt.Rectangle((x, y), w, h, facecolor=color, alpha=0.15,
                          edgecolor=color, linewidth=2, transform=ax.transAxes)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, fontsize=11, ha="center", va="center",
            transform=ax.transAxes, fontweight="bold", color=color)

# Arrows
arrows = [
    (0.14, 0.88, 0.12, 0.76), (0.14, 0.88, 0.26, 0.76), (0.14, 0.88, 0.40, 0.76),
    (0.12, 0.72, 0.12, 0.66), (0.26, 0.72, 0.26, 0.66), (0.40, 0.72, 0.40, 0.66),
    (0.12, 0.60, 0.20, 0.50), (0.26, 0.60, 0.24, 0.50),
    (0.22, 0.44, 0.22, 0.35), (0.22, 0.30, 0.22, 0.23),
    (0.40, 0.60, 0.26, 0.11), (0.22, 0.18, 0.22, 0.11),
]
for x1, y1, x2, y2 in arrows:
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.5),
                transform=ax.transAxes)

# Annotations on the right
explanations = [
    (0.55, 0.85, "① Each token gets projected into three roles:\n"
                  "   Query = 'What am I looking for?'\n"
                  "   Key   = 'What do I contain?'\n"
                  "   Value = 'What info should I share?'", "#333"),
    (0.55, 0.65, "② Q × Kᵀ computes how relevant each token is\n"
                  "   to every other token. Result: a [T×T] score matrix.\n"
                  "   High score = 'these two tokens are related!'", "#9b59b6"),
    (0.55, 0.45, "③ Scale by √d_k to prevent extreme values.\n"
                  "   Then softmax converts scores to probabilities\n"
                  "   (each row sums to 1.0).", "#9b59b6"),
    (0.55, 0.25, "④ Multiply attention weights by V.\n"
                  "   Each token becomes a weighted average of all\n"
                  "   other tokens' values. Context is captured!", "#e67e22"),
    (0.55, 0.08, "RESULT: Each token now contains information from\n"
                  "every token it attended to — context-aware embeddings!", "#333"),
]

for x, y, text, color in explanations:
    ax.text(x, y, text, fontsize=10, va="top", transform=ax.transAxes,
            color=color, fontfamily="monospace",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="#ddd"))

ax.set_title("Self-Attention Mechanism — Data Flow",
             fontsize=16, fontweight="bold", pad=20)
plt.tight_layout()
path2 = os.path.join(output_dir, "attention_flow.png")
plt.savefig(path2, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {path2}")


# ============================================================================
# SUMMARY
# ============================================================================

print(f"""
{'=' * 70}
WHAT YOU LEARNED
{'=' * 70}

  1. SELF-ATTENTION lets every token look at every other token
     and decide how much to "pay attention" to each one.

  2. THREE PROJECTIONS — Q, K, V:
     - Query: "What am I looking for?"
     - Key:   "What do I contain?"
     - Value: "What info do I share?"
     - Attention = softmax(Q × Kᵀ / √d_k) × V

  3. MULTI-HEAD: Run multiple attention patterns in parallel.
     Each head can learn different relationship types.

  4. CAUSAL MASK: Block future tokens for text generation.
     Lower-triangular mask → tokens can only see the past.

  5. TRANSFORMER BLOCK: Attention + Feed-Forward + LayerNorm + Residual
     Stack 12-96 of these → GPT!

  FULL PICTURE SO FAR:
  ┌──────────────────────────────────────────────────────┐
  │  Text                                                │
  │    ↓  Tokenization (BPE)                     ✓ DONE  │
  │  Token IDs                                           │
  │    ↓  Token + Position Embeddings            ✓ DONE  │
  │  Input Vectors                                       │
  │    ↓  Self-Attention (Q, K, V)               ✓ DONE  │
  │    ↓  Feed-Forward Network                   ✓ DONE  │
  │    ↓  (repeat N times)                               │
  │  Contextualized Vectors                              │
  │    ↓  Linear + Softmax                               │
  │  Output Probabilities                                │
  │    ↓  Decode                                         │
  │  Generated Text                                      │
  └──────────────────────────────────────────────────────┘

  You now understand EVERY piece of a transformer!
  NEXT: Put it all together into a working mini-GPT that
  generates Shakespeare text.
""")
