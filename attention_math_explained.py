"""
=============================================================================
 THE ATTENTION FORMULA — Every Single Step With Real Numbers
=============================================================================

  Attention(Q, K, V) = softmax( Q × Kᵀ / √d_k ) × V

  We'll break this into 5 pieces:
    Piece 1: Q, K, V — where they come from
    Piece 2: Q × Kᵀ — why multiply queries and keys?
    Piece 3: / √d_k — why divide by square root?
    Piece 4: softmax — turning scores into probabilities
    Piece 5: × V — the final weighted mix

=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import math
import os

output_dir = os.path.dirname(__file__)

# ============================================================================
# SETUP: A tiny sentence we can trace by hand
# ============================================================================

sentence = ["I", "love", "cats"]
seq_len = 3
embed_dim = 4  # tiny so we can see every number

print("=" * 70)
print(f"SENTENCE: '{' '.join(sentence)}'")
print(f"Tokens: {seq_len}, Embedding dim: {embed_dim}")
print("=" * 70)

# Manually set embeddings (pretend these came from the embedding layer)
# These are chosen to make the math easy to follow
X = torch.tensor([
    [1.0,  0.0,  1.0,  0.0],   # "I"    — person-like
    [0.0,  1.0,  0.0,  1.0],   # "love" — emotion-like
    [1.0,  1.0,  0.0,  0.0],   # "cats" — animal-like
], dtype=torch.float).unsqueeze(0)  # add batch dim: [1, 3, 4]

print(f"\nInput embeddings (X):")
for i, word in enumerate(sentence):
    print(f"  {word:>6s} = {X[0, i].tolist()}")


# ============================================================================
# PIECE 1: Q, K, V — Where They Come From
# ============================================================================

print(f"\n{'=' * 70}")
print("PIECE 1: Creating Q, K, V from embeddings")
print("=" * 70)

print("""
  Each token's embedding gets multiplied by THREE weight matrices:

    Q = X × W_q    (what am I looking for?)
    K = X × W_k    (what do I contain?)
    V = X × W_v    (what info do I carry?)

  These weight matrices are LEARNED during training.
  For this demo, we'll set them manually so the numbers are clear.
""")

d_k = 3  # dimension of Q, K, V (often smaller than embed_dim)

# Set weights manually for clarity
W_q = torch.tensor([
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
    [0, 1, 1],
], dtype=torch.float)  # [embed_dim=4, d_k=3]

W_k = torch.tensor([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 1],
    [1, 0, 0],
], dtype=torch.float)

W_v = torch.tensor([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 0],
], dtype=torch.float)

# Compute Q, K, V
Q = X @ W_q  # [1, 3, 3]
K = X @ W_k  # [1, 3, 3]
V = X @ W_v  # [1, 3, 3]

print(f"  W_q shape: {W_q.shape} (embed_dim × d_k)")
print(f"  W_k shape: {W_k.shape}")
print(f"  W_v shape: {W_v.shape}")

print(f"\n  Computing Q = X × W_q:")
print(f"  {'Token':>6s}   {'Embedding':>20s}   ×   W_q   →   {'Query':>15s}")
print(f"  {'-'*6}   {'-'*20}       {'-'*3}       {'-'*15}")
for i, word in enumerate(sentence):
    emb = X[0, i].tolist()
    q = Q[0, i].tolist()
    print(f"  {word:>6s}   {str(emb):>20s}   ×   W_q   →   {str([round(v,1) for v in q]):>15s}")

print(f"\n  All Q (queries):  what each token is LOOKING FOR")
for i, word in enumerate(sentence):
    print(f"    {word:>6s}: {Q[0,i].tolist()}")

print(f"\n  All K (keys):  what each token ADVERTISES about itself")
for i, word in enumerate(sentence):
    print(f"    {word:>6s}: {K[0,i].tolist()}")

print(f"\n  All V (values):  what each token's actual INFORMATION is")
for i, word in enumerate(sentence):
    print(f"    {word:>6s}: {V[0,i].tolist()}")


# ============================================================================
# PIECE 2: Q × Kᵀ — The Dot Product (Matching Queries to Keys)
# ============================================================================

print(f"\n{'=' * 70}")
print("PIECE 2: Q × Kᵀ — How Much Does Each Token Care About Each Other?")
print("=" * 70)

print(f"""
  This is the HEART of attention.

  For each pair of tokens (i, j):
    score(i, j) = Query_i · Key_j   (dot product)

  High dot product → token i's query MATCHES token j's key
                   → "these two are relevant to each other!"

  Low dot product  → poor match
                   → "these two aren't related"

  Result: a [{seq_len} × {seq_len}] matrix of scores.
""")

# Compute step by step
scores = Q @ K.transpose(-2, -1)  # [1, 3, 3]

print(f"  Computing Q × Kᵀ (each cell = dot product of query_i and key_j):\n")

# Show the detailed dot product for each pair
for i, word_i in enumerate(sentence):
    for j, word_j in enumerate(sentence):
        qi = Q[0, i].tolist()
        kj = K[0, j].tolist()
        products = [round(a * b, 1) for a, b in zip(qi, kj)]
        total = sum(products)
        detail = " + ".join(f"{a}×{b}" for a, b in zip(
            [round(v, 1) for v in qi], [round(v, 1) for v in kj]))
        print(f"    score({word_i:>4s}, {word_j:<4s}) = Q_{word_i} · K_{word_j} = {detail} = {total:.1f}")
    print()

print(f"  Score matrix (Q × Kᵀ):")
print(f"  {'':>6s}  {'  '.join(f'{w:>6s}' for w in sentence)}")
for i, word in enumerate(sentence):
    row = scores[0, i].tolist()
    print(f"  {word:>6s}  {'  '.join(f'{v:>6.1f}' for v in row)}")

print(f"""
  Reading row by row:
    "I"    → highest score with ??? → that's who "I" attends to most
    "love" → highest score with ??? → that's who "love" attends to most
    "cats" → highest score with ??? → that's who "cats" attends to most
""")


# ============================================================================
# PIECE 3: / √d_k — Why Divide by Square Root?
# ============================================================================

print(f"{'=' * 70}")
print("PIECE 3: ÷ √d_k — The Scaling Factor")
print("=" * 70)

print(f"""
  Before: scores can be VERY large numbers (especially in high dims).
  Problem: large numbers → softmax gives near-0 and near-1 values.

  Example WITHOUT scaling (imagine d_k = 512):
""")

# Show the problem
torch.manual_seed(0)
big_q = torch.randn(1, 1, 512)
big_k = torch.randn(1, 1, 512)
big_score = (big_q @ big_k.transpose(-1, -2)).item()
print(f"    Random dot product with d_k=512: {big_score:.1f}")
print(f"    These scores can range from -40 to +40!")

# Show what happens to softmax with large vs small inputs
large_scores = torch.tensor([[15.0, 1.0, 2.0]])
small_scores = torch.tensor([[3.0,  0.2, 0.4]])

soft_large = F.softmax(large_scores, dim=-1)
soft_small = F.softmax(small_scores, dim=-1)

print(f"""
    softmax([15.0, 1.0, 2.0]) = {[round(v,6) for v in soft_large[0].tolist()]}
    → Almost ALL weight on first token! Others get ~0.

    softmax([3.0, 0.2, 0.4])  = {[round(v,4) for v in soft_small[0].tolist()]}
    → Weight is spread out — model can attend to multiple tokens.

  The fix: divide by √d_k to keep scores in a reasonable range.
""")

scale = math.sqrt(d_k)
scaled_scores = scores / scale

print(f"  d_k = {d_k}")
print(f"  √d_k = √{d_k} = {scale:.4f}")
print(f"\n  Scaled scores = scores / {scale:.4f}:")
print(f"  {'':>6s}  {'  '.join(f'{w:>6s}' for w in sentence)}")
for i, word in enumerate(sentence):
    row = scaled_scores[0, i].tolist()
    print(f"  {word:>6s}  {'  '.join(f'{v:>6.2f}' for v in row)}")

print(f"""
  WHY √d_k specifically?

  Math reason: if Q and K have random values with variance 1,
  their dot product has variance = d_k (each of d_k multiplications
  adds variance). Dividing by √d_k brings variance back to 1.

  Intuition: it's like grading on a curve. More dimensions = bigger
  raw scores, so we normalize to keep scores comparable regardless
  of dimension size.
""")


# ============================================================================
# PIECE 4: softmax — Turning Scores into Probabilities
# ============================================================================

print(f"{'=' * 70}")
print("PIECE 4: softmax — From Raw Scores to Attention Weights")
print("=" * 70)

print(f"""
  softmax converts any list of numbers into PROBABILITIES (0 to 1, sum to 1).

  Formula:  softmax(x_i) = e^(x_i) / Σ e^(x_j)

  Three things softmax does:
    1. Makes all values POSITIVE (e^x is always > 0)
    2. Makes them SUM TO 1.0 (it's a probability distribution)
    3. AMPLIFIES differences (big scores get much bigger share)
""")

# Step-by-step softmax for one row
row_idx = 0
row_word = sentence[row_idx]
row_scores = scaled_scores[0, row_idx]

print(f"  Step-by-step softmax for '{row_word}'s attention:")
print(f"\n  Scaled scores: {[round(v, 4) for v in row_scores.tolist()]}")

# Step 1: e^x for each
exp_values = torch.exp(row_scores)
print(f"\n  Step 1: Compute e^(score) for each:")
for j, word in enumerate(sentence):
    val = row_scores[j].item()
    exp_val = exp_values[j].item()
    print(f"    e^({val:.4f}) = {exp_val:.4f}   ← '{word}'")

# Step 2: Sum
exp_sum = exp_values.sum().item()
print(f"\n  Step 2: Sum all e^values: {' + '.join(f'{v:.4f}' for v in exp_values.tolist())} = {exp_sum:.4f}")

# Step 3: Divide
weights_manual = exp_values / exp_values.sum()
print(f"\n  Step 3: Divide each by sum:")
for j, word in enumerate(sentence):
    exp_val = exp_values[j].item()
    weight = weights_manual[j].item()
    print(f"    {exp_val:.4f} / {exp_sum:.4f} = {weight:.4f}   ← '{word}' gets {weight*100:.1f}% attention")

# Verify with PyTorch
weights = F.softmax(scaled_scores, dim=-1)
print(f"\n  PyTorch softmax verification: {[round(v, 4) for v in weights[0, 0].tolist()]} ✓")

# Full attention weight matrix
print(f"\n  FULL ATTENTION WEIGHT MATRIX (after softmax):")
print(f"  (each row sums to 1.0 — it's a probability distribution)\n")
print(f"  {'':>6s}  {'  '.join(f'{w:>6s}' for w in sentence)}  │ sum")
print(f"  {'─'*6}  {'  '.join('──────' for _ in sentence)}  │ ───")
for i, word in enumerate(sentence):
    row = weights[0, i].tolist()
    row_sum = sum(row)
    bars = ""
    for v in row:
        bars_count = int(v * 20)
        bars += f"{'█' * bars_count:>6s}  "
    print(f"  {word:>6s}  {'  '.join(f'{v:>6.4f}' for v in row)}  │ {row_sum:.2f}")

print(f"\n  Visual (█ = attention weight):")
for i, word in enumerate(sentence):
    row = weights[0, i].tolist()
    print(f"  {word:>6s}:", end="")
    for j, (w, target) in enumerate(zip(row, sentence)):
        bar = "█" * int(w * 30)
        print(f"  {target}={'█' * int(w * 30):<10s}({w:.2f})", end="")
    print()


# ============================================================================
# PIECE 5: × V — The Final Weighted Mix
# ============================================================================

print(f"\n{'=' * 70}")
print("PIECE 5: × V — Gathering Information Based on Attention")
print("=" * 70)

print(f"""
  Now we know HOW MUCH each token attends to every other token.
  The final step: use these weights to create a WEIGHTED MIX of values.

  For each token i:
    output_i = Σ (attention_weight_ij × Value_j)  for all j

  This means: each token becomes a BLEND of other tokens' information,
  weighted by how much it paid attention to each one.
""")

output = weights @ V  # [1, 3, 3]

# Show detailed calculation for "I"
print(f"  Detailed calculation for '{sentence[0]}':\n")
print(f"  '{sentence[0]}' attends to:")
for j, word in enumerate(sentence):
    w = weights[0, 0, j].item()
    v = V[0, j].tolist()
    product = [round(w * val, 4) for val in v]
    print(f"    {w:.4f} × V_{word:>4s} {str([round(x,1) for x in v]):>20s}  =  {product}")

weighted_sum = output[0, 0].tolist()
print(f"    {'─' * 55}")
print(f"    SUM {'':>41s}  =  {[round(v, 4) for v in weighted_sum]}")
print(f"\n  BEFORE attention: '{sentence[0]}' = {X[0, 0].tolist()} (just its own embedding)")
print(f"  AFTER  attention: '{sentence[0]}' = {[round(v, 4) for v in weighted_sum]} (blend of all tokens!)")

print(f"\n  All outputs (context-aware embeddings):")
for i, word in enumerate(sentence):
    old = X[0, i].tolist()
    new = [round(v, 4) for v in output[0, i].tolist()]
    print(f"    {word:>6s}: {str(old):>25s}  →  {new}")

print(f"""
  WHAT CHANGED:
  Each token is no longer just about ITSELF.
  It now contains a weighted blend of information from tokens
  it found relevant. The embedding is now CONTEXT-AWARE.

  "I" isn't just a generic pronoun anymore —
  it now carries info from "love" and "cats" because
  attention connected them.
""")


# ============================================================================
# THE COMPLETE FORMULA — All Together
# ============================================================================

print(f"{'=' * 70}")
print("THE COMPLETE FORMULA — All 5 Pieces Together")
print("=" * 70)

print(f"""
  Attention(Q, K, V) = softmax( Q × Kᵀ / √d_k ) × V

  Piece by piece:

  ┌─────────────────────────────────────────────────────────────────┐
  │                                                                 │
  │   Q × Kᵀ         "How relevant is each token to every other?"  │
  │   ──────          Dot product between all query-key pairs.      │
  │                   Result: [{seq_len}×{seq_len}] score matrix                   │
  │       ↓                                                         │
  │                                                                 │
  │   / √d_k          "Normalize the scores"                       │
  │                   Prevents extreme values in high dimensions.   │
  │                   √{d_k} = {scale:.2f} in our case                        │
  │       ↓                                                         │
  │                                                                 │
  │   softmax         "Convert to probabilities"                    │
  │                   Each row sums to 1.0.                         │
  │                   Big scores → big weights, small → near zero.  │
  │       ↓                                                         │
  │                                                                 │
  │   × V             "Gather information"                          │
  │                   Weighted sum of value vectors.                │
  │                   Output = context-aware embeddings.            │
  │                                                                 │
  └─────────────────────────────────────────────────────────────────┘

  DATA SHAPES through the formula:

  Q:  [{seq_len} × {d_k}]    K:  [{seq_len} × {d_k}]    V:  [{seq_len} × {d_k}]
         │              │               │
         └──── × ───────┘               │
               ↓                        │
       Q × Kᵀ: [{seq_len} × {seq_len}]                │
               ↓ (÷ √d_k)              │
       scaled: [{seq_len} × {seq_len}]                │
               ↓ (softmax)             │
       weights: [{seq_len} × {seq_len}]                │
               │                        │
               └──────── × ────────────┘
                         ↓
                  output: [{seq_len} × {d_k}]
                  (same shape as input!)
""")


# ============================================================================
# VISUALIZATION
# ============================================================================

print("Creating visualization...")

fig = plt.figure(figsize=(22, 16))
gs = gridspec.GridSpec(3, 4, hspace=0.4, wspace=0.35)

# --- Panel 1: Input Embeddings ---
ax = fig.add_subplot(gs[0, 0])
data = X[0].detach().numpy()
im = ax.imshow(data, cmap="RdBu_r", aspect="auto", vmin=-1.5, vmax=1.5)
ax.set_yticks(range(seq_len))
ax.set_yticklabels(sentence, fontsize=12, fontweight="bold")
ax.set_xlabel("Embedding dim", fontsize=10)
ax.set_title("① Input X\n(token embeddings)", fontsize=12, fontweight="bold", color="#3498db")
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        ax.text(j, i, f"{data[i,j]:.1f}", ha="center", va="center", fontsize=10)

# --- Panel 2: Q ---
ax = fig.add_subplot(gs[0, 1])
data = Q[0].detach().numpy()
im = ax.imshow(data, cmap="Reds", aspect="auto")
ax.set_yticks(range(seq_len))
ax.set_yticklabels(sentence, fontsize=12, fontweight="bold")
ax.set_xlabel("d_k dim", fontsize=10)
ax.set_title("② Q = X × W_q\n(queries)", fontsize=12, fontweight="bold", color="#e74c3c")
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        ax.text(j, i, f"{data[i,j]:.1f}", ha="center", va="center", fontsize=10)

# --- Panel 3: K ---
ax = fig.add_subplot(gs[0, 2])
data = K[0].detach().numpy()
im = ax.imshow(data, cmap="Greens", aspect="auto")
ax.set_yticks(range(seq_len))
ax.set_yticklabels(sentence, fontsize=12, fontweight="bold")
ax.set_xlabel("d_k dim", fontsize=10)
ax.set_title("③ K = X × W_k\n(keys)", fontsize=12, fontweight="bold", color="#2ecc71")
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        ax.text(j, i, f"{data[i,j]:.1f}", ha="center", va="center", fontsize=10)

# --- Panel 4: V ---
ax = fig.add_subplot(gs[0, 3])
data = V[0].detach().numpy()
im = ax.imshow(data, cmap="Oranges", aspect="auto")
ax.set_yticks(range(seq_len))
ax.set_yticklabels(sentence, fontsize=12, fontweight="bold")
ax.set_xlabel("d_k dim", fontsize=10)
ax.set_title("④ V = X × W_v\n(values)", fontsize=12, fontweight="bold", color="#e67e22")
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        ax.text(j, i, f"{data[i,j]:.1f}", ha="center", va="center", fontsize=10)

# --- Panel 5: Q × Kᵀ (raw scores) ---
ax = fig.add_subplot(gs[1, 0])
data = scores[0].detach().numpy()
im = ax.imshow(data, cmap="PuRd", aspect="auto")
ax.set_xticks(range(seq_len))
ax.set_xticklabels(sentence, fontsize=11)
ax.set_yticks(range(seq_len))
ax.set_yticklabels(sentence, fontsize=11, fontweight="bold")
ax.set_ylabel("Query (from) →", fontsize=10)
ax.set_xlabel("Key (to) →", fontsize=10)
ax.set_title("⑤ Q × Kᵀ\n(raw scores)", fontsize=12, fontweight="bold", color="#9b59b6")
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        ax.text(j, i, f"{data[i,j]:.1f}", ha="center", va="center", fontsize=12, fontweight="bold")

# --- Panel 6: Scaled scores ---
ax = fig.add_subplot(gs[1, 1])
data = scaled_scores[0].detach().numpy()
im = ax.imshow(data, cmap="PuRd", aspect="auto")
ax.set_xticks(range(seq_len))
ax.set_xticklabels(sentence, fontsize=11)
ax.set_yticks(range(seq_len))
ax.set_yticklabels(sentence, fontsize=11, fontweight="bold")
ax.set_title(f"⑥ ÷ √d_k (÷ {scale:.2f})\n(scaled scores)", fontsize=12, fontweight="bold", color="#9b59b6")
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center", fontsize=11)

# --- Panel 7: Attention weights (softmax) ---
ax = fig.add_subplot(gs[1, 2])
data = weights[0].detach().numpy()
im = ax.imshow(data, cmap="Blues", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(seq_len))
ax.set_xticklabels(sentence, fontsize=11)
ax.set_yticks(range(seq_len))
ax.set_yticklabels(sentence, fontsize=11, fontweight="bold")
ax.set_title("⑦ softmax\n(attention weights)", fontsize=12, fontweight="bold", color="#2980b9")
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        color = "white" if data[i,j] > 0.45 else "black"
        ax.text(j, i, f"{data[i,j]:.3f}", ha="center", va="center", fontsize=11,
                fontweight="bold", color=color)

# --- Panel 8: Output ---
ax = fig.add_subplot(gs[1, 3])
data = output[0].detach().numpy()
im = ax.imshow(data, cmap="YlOrRd", aspect="auto")
ax.set_yticks(range(seq_len))
ax.set_yticklabels(sentence, fontsize=12, fontweight="bold")
ax.set_xlabel("d_k dim", fontsize=10)
ax.set_title("⑧ Weights × V\n(context-aware output)", fontsize=12, fontweight="bold", color="#e67e22")
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center", fontsize=11, fontweight="bold")

# --- Bottom panel: The full flow ---
ax = fig.add_subplot(gs[2, :])
ax.axis("off")

flow = (
    "THE COMPLETE ATTENTION FORMULA:   Attention(Q, K, V) = softmax( Q × Kᵀ / √d_k ) × V\n\n"
    "  ①②③④                    ⑤                 ⑥              ⑦                ⑧\n"
    " X → Q,K,V          Q × Kᵀ          ÷ √d_k          softmax            × V\n"
    " (project)      (match queries     (prevent        (convert to       (weighted sum\n"
    "                  against keys)     extremes)      probabilities)     of values)\n\n"
    " [3×4] → [3×3]      [3×3]           [3×3]           [3×3]           [3×3]\n"
    " embeddings       score matrix    scaled scores   attn weights    context-aware\n"
    "                   'who matches?'  'normalized'    'how much?'     'blended info'\n"
)

ax.text(0.5, 0.5, flow, fontsize=13, fontfamily="monospace",
        ha="center", va="center", transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#f0f8ff", edgecolor="#2980b9", linewidth=2))

fig.suptitle("Self-Attention Math: \"I love cats\" — Every Step Visualized",
             fontsize=18, fontweight="bold", y=0.98)

path = os.path.join(output_dir, "attention_math_visual.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {path}")


# ============================================================================
# FINAL SUMMARY
# ============================================================================

print(f"""
{'=' * 70}
SUMMARY: The Attention Formula Decoded
{'=' * 70}

  Attention(Q, K, V) = softmax( Q × Kᵀ / √d_k ) × V
  ──────────────────   ───────  ──────  ──────    ───
        │                 │       │       │        │
        │                 │       │       │        └─ Gather: weighted
        │                 │       │       │           mix of values
        │                 │       │       │
        │                 │       │       └─ Scale: prevent
        │                 │       │          extreme values
        │                 │       │
        │                 │       └─ Match: how relevant
        │                 │          is each token pair?
        │                 │
        │                 └─ Normalize: convert
        │                    scores → probabilities
        │
        └─ Final output: context-aware embeddings
           (each token now "knows about" other tokens)

  WHY EACH PIECE EXISTS:
    Q × Kᵀ    → without this, tokens can't compare with each other
    / √d_k    → without this, softmax saturates (all-or-nothing attention)
    softmax   → without this, weights don't sum to 1 (not a valid mix)
    × V       → without this, we know WHO to attend to but don't
                 actually GATHER their information
""")
