"""
=============================================================================
 EMBEDDINGS — Deep Dive (From First Principles)
=============================================================================

 THE PROBLEM
 -----------
 After tokenization, we have numbers:  "the cat sat" → [5, 12, 8]
 But these numbers are ARBITRARY. Token 12 isn't "bigger" than token 5.
 "cat" (12) isn't "closer" to "dog" (7) than to "quantum" (31).

 The neural network needs a MEANINGFUL numerical representation where:
   - Similar words are CLOSE together
   - Different words are FAR apart
   - Relationships are captured (king - man + woman ≈ queen)

 That's what embeddings do.

 JOURNEY:
   Stage 1: The problem with raw numbers (one-hot encoding)
   Stage 2: What embeddings actually are
   Stage 3: Build and train embeddings from scratch
   Stage 4: Watch embeddings learn meaning
   Stage 5: Positional embeddings (where in the sentence?)
   Stage 6: How real LLMs use embeddings

=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================================================================
# STAGE 1: THE PROBLEM — Why Can't We Just Use Token IDs?
# ============================================================================

print("=" * 70)
print("STAGE 1: Why Raw Numbers Don't Work")
print("=" * 70)

print("""
After tokenization: "the cat sat on the mat" → [0, 1, 2, 3, 0, 4]

Problem: if we feed these raw numbers to a neural network:
  - "cat" (1) seems closer to "the" (0) than to "mat" (4)
  - But semantically, "cat" and "mat" rhyme and appear in similar contexts!
  - The number assignments are ARBITRARY

First attempt to fix this: ONE-HOT ENCODING
""")

# One-hot encoding demo
vocab = ["the", "cat", "sat", "on", "mat"]
vocab_size = len(vocab)

print("One-hot encoding for our vocabulary:\n")
print(f"  {'Word':<6} {'One-Hot Vector':<30} {'Length'}")
print(f"  {'-'*6} {'-'*30} {'-'*6}")
for i, word in enumerate(vocab):
    one_hot = [0] * vocab_size
    one_hot[i] = 1
    print(f"  {word:<6} {str(one_hot):<30} {vocab_size}")

print(f"""
┌──────────────────────────────────────────────────────────────────┐
│  ONE-HOT ENCODING PROBLEMS:                                    │
│                                                                  │
│  1. HUGE vectors: vocab of 50,000 → each word is a 50,000-dim  │
│     vector with a single 1 and 49,999 zeros. Wasteful!         │
│                                                                  │
│  2. NO SIMILARITY: every word is equally distant from every     │
│     other word. "cat" is as far from "dog" as from "quantum".  │
│                                                                  │
│     Distance("cat", "dog")     = √2 ≈ 1.414                    │
│     Distance("cat", "quantum") = √2 ≈ 1.414                    │
│     All the same! The representation captures NO meaning.       │
│                                                                  │
│  3. NO LEARNING: these vectors are fixed, not learnable.        │
│     The network can't adjust them to capture patterns.          │
└──────────────────────────────────────────────────────────────────┘
""")

# Prove that all one-hot vectors are equidistant
cat_vec = torch.tensor([0, 1, 0, 0, 0], dtype=torch.float)
dog_vec = torch.tensor([0, 0, 1, 0, 0], dtype=torch.float)  # pretend "sat" is "dog"
the_vec = torch.tensor([1, 0, 0, 0, 0], dtype=torch.float)

print("Euclidean distances between one-hot vectors:")
print(f"  cat ↔ dog:  {torch.dist(cat_vec, dog_vec):.4f}")
print(f"  cat ↔ the:  {torch.dist(cat_vec, the_vec):.4f}")
print(f"  dog ↔ the:  {torch.dist(dog_vec, the_vec):.4f}")
print(f"  All identical! One-hot captures ZERO semantic information.\n")


# ============================================================================
# STAGE 2: WHAT EMBEDDINGS ACTUALLY ARE
# ============================================================================

print("=" * 70)
print("STAGE 2: What Embeddings Actually Are")
print("=" * 70)

print("""
An embedding is a DENSE, LOW-DIMENSIONAL, LEARNABLE vector for each token.

Instead of one-hot (sparse, high-dim, fixed):
  "cat" → [0, 1, 0, 0, 0]  (5 dims, mostly zeros)

We use embeddings (dense, low-dim, learned):
  "cat" → [0.23, -0.45, 0.89, 0.12]  (4 dims, all meaningful)

Key properties:
  1. DENSE:     Every dimension carries information (no wasted zeros)
  2. COMPACT:   Typically 64 to 4096 dimensions (not 50,000)
  3. LEARNABLE: The network adjusts these during training
  4. SEMANTIC:  Similar words end up with similar vectors
""")

# Demonstrate nn.Embedding
print("--- How nn.Embedding works (the lookup table) ---\n")

torch.manual_seed(42)
embedding_dim = 4
embed = nn.Embedding(vocab_size, embedding_dim)

print(f"Embedding table shape: [{vocab_size} × {embedding_dim}]")
print(f"  {vocab_size} rows (one per token in vocab)")
print(f"  {embedding_dim} columns (the embedding dimension)\n")

print("The table (randomly initialized — not yet trained):\n")
print(f"  {'ID':<4} {'Word':<6} {'Embedding Vector'}")
print(f"  {'-'*4} {'-'*6} {'-'*40}")
for i, word in enumerate(vocab):
    token_id = torch.tensor(i)
    vec = embed(token_id)
    formatted = [f"{v:+.3f}" for v in vec.tolist()]
    print(f"  {i:<4} {word:<6} [{', '.join(formatted)}]")

print(f"""
HOW LOOKUP WORKS:

  Input token IDs:    [0, 1, 2]   ("the cat sat")
                       ↓  ↓  ↓
  Lookup row 0:  → [{', '.join(f'{v:+.3f}' for v in embed(torch.tensor(0)).tolist())}]   ← "the"
  Lookup row 1:  → [{', '.join(f'{v:+.3f}' for v in embed(torch.tensor(1)).tolist())}]   ← "cat"
  Lookup row 2:  → [{', '.join(f'{v:+.3f}' for v in embed(torch.tensor(2)).tolist())}]   ← "sat"

  Output shape:  [3 tokens × {embedding_dim} dims]

  It's just a TABLE LOOKUP — fast, simple, differentiable.
  During training, backpropagation adjusts the values in this table
  so that similar words get similar vectors.
""")


# ============================================================================
# STAGE 3: TRAINING EMBEDDINGS — Watch Them Learn!
# ============================================================================

print("=" * 70)
print("STAGE 3: Training Embeddings — Watch Them Learn Meaning!")
print("=" * 70)

print("""
We'll train embeddings on sentences where words appear in similar contexts.
The key insight (from Word2Vec):

  "You shall know a word by the company it keeps." — J.R. Firth, 1957

Words that appear in similar contexts should have similar embeddings.
  - "cat" and "dog" both appear near "pet", "cute", "runs" → similar embeddings
  - "cat" and "algebra" appear in very different contexts → different embeddings
""")

# Training corpus — carefully designed so context reveals meaning
training_sentences = [
    "the cat sat on the mat",
    "the dog sat on the rug",
    "the cat chased the mouse",
    "the dog chased the cat",
    "a king ruled the land",
    "a queen ruled the kingdom",
    "the king wore a crown",
    "the queen wore a crown",
    "the prince is the son of the king",
    "the princess is the daughter of the queen",
    "the cat is a small animal",
    "the dog is a loyal animal",
    "the mouse is a tiny animal",
    "a man became the king",
    "a woman became the queen",
]

# Build vocabulary
all_words = set()
for s in training_sentences:
    all_words.update(s.split())
word_list = sorted(all_words)
w2i = {w: i for i, w in enumerate(word_list)}
i2w = {i: w for w, i in w2i.items()}
V = len(word_list)
print(f"Vocabulary: {V} words")
print(f"Words: {word_list}\n")


# Create training data: predict center word from context (CBOW-style)
# For each word, use surrounding words as context
def create_training_data(sentences, w2i, window=2):
    data = []
    for sentence in sentences:
        words = sentence.split()
        for i, word in enumerate(words):
            target = w2i[word]
            # Context = words within 'window' positions
            for j in range(max(0, i - window), min(len(words), i + window + 1)):
                if j != i:
                    context = w2i[words[j]]
                    data.append((context, target))
    return data


pairs = create_training_data(training_sentences, w2i)
print(f"Training pairs (context → target): {len(pairs)} pairs")
print(f"Examples:")
for ctx, tgt in pairs[:8]:
    print(f"  '{i2w[ctx]}' → '{i2w[tgt]}'")
print(f"  ...")


# Simple embedding model: predict target word from context word
class WordEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, context_ids):
        embeds = self.embeddings(context_ids)  # [batch, embed_dim]
        logits = self.linear(embeds)           # [batch, vocab_size]
        return logits


# Train!
EMBED_DIM = 16
model = WordEmbeddingModel(V, EMBED_DIM)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

context_ids = torch.tensor([p[0] for p in pairs])
target_ids = torch.tensor([p[1] for p in pairs])

print(f"\nTraining embeddings ({EMBED_DIM}-dimensional)...")
print(f"{'Epoch':>6} {'Loss':>10}")
print("-" * 18)

for epoch in range(301):
    logits = model(context_ids)
    loss = F.cross_entropy(logits, target_ids)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"{epoch:6d} {loss.item():10.4f}")

print("Training complete!\n")


# ============================================================================
# STAGE 4: EXPLORING LEARNED EMBEDDINGS
# ============================================================================

print("=" * 70)
print("STAGE 4: Exploring What the Embeddings Learned")
print("=" * 70)

# Extract learned embeddings
learned_embeds = model.embeddings.weight.detach()

# Cosine similarity: measures how similar two vectors are (-1 to +1)
def cosine_sim(w1, w2):
    v1 = learned_embeds[w2i[w1]]
    v2 = learned_embeds[w2i[w2]]
    return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()


print("\n--- Cosine Similarity Between Word Pairs ---")
print("(1.0 = identical, 0.0 = unrelated, -1.0 = opposite)\n")

similarity_pairs = [
    ("cat", "dog",      "Both animals, similar contexts"),
    ("cat", "mouse",    "Both animals"),
    ("king", "queen",   "Both royalty"),
    ("king", "prince",  "Both male royalty"),
    ("queen", "princess", "Both female royalty"),
    ("man", "woman",    "Both people"),
    ("king", "cat",     "Unrelated concepts"),
    ("crown", "animal", "Unrelated concepts"),
    ("sat", "chased",   "Both verbs"),
    ("mat", "rug",      "Both floor coverings"),
]

print(f"  {'Word 1':<10} {'Word 2':<10} {'Similarity':>10}  Reason")
print(f"  {'-'*10} {'-'*10} {'-'*10}  {'-'*30}")
for w1, w2, reason in similarity_pairs:
    sim = cosine_sim(w1, w2)
    bar = "█" * int(max(0, sim) * 20)
    print(f"  {w1:<10} {w2:<10} {sim:>+10.4f}  {bar} {reason}")

# Word analogy: king - man + woman ≈ queen?
print(f"\n--- Word Analogies (Vector Arithmetic) ---")
print("""
The famous result from Word2Vec:
  king - man + woman ≈ queen

This works because embeddings capture RELATIONSHIPS as directions:
  "male royalty" - "male" + "female" ≈ "female royalty"
""")


def analogy(a, b, c, top_n=5):
    """a is to b as c is to ?: compute b - a + c"""
    va = learned_embeds[w2i[a]]
    vb = learned_embeds[w2i[b]]
    vc = learned_embeds[w2i[c]]

    result_vec = vb - va + vc  # the magic operation

    # Find closest words to result
    similarities = F.cosine_similarity(result_vec.unsqueeze(0), learned_embeds)
    # Exclude input words
    exclude = {w2i[a], w2i[b], w2i[c]}
    results = []
    for idx in similarities.argsort(descending=True):
        if idx.item() not in exclude:
            results.append((i2w[idx.item()], similarities[idx].item()))
        if len(results) >= top_n:
            break
    return results


analogies = [
    ("man", "king", "woman",    "man→king as woman→???"),
    ("king", "prince", "queen", "king→prince as queen→???"),
    ("cat", "sat", "dog",       "cat→sat as dog→???"),
]

for a, b, c, desc in analogies:
    results = analogy(a, b, c)
    print(f"  {desc}")
    print(f"    {b} - {a} + {c} = ?")
    for word, sim in results[:3]:
        print(f"      → {word:<12} (similarity: {sim:+.4f})")
    print()

print("""NOTE: With only 15 short training sentences, analogies won't be perfect.
Real word embeddings (Word2Vec, GloVe) train on billions of words and
produce much more reliable analogies. But the PRINCIPLE is the same!
""")


# ============================================================================
# STAGE 5: VISUALIZING EMBEDDINGS IN 2D
# ============================================================================

print("=" * 70)
print("STAGE 5: Visualizing Embeddings (Projected to 2D)")
print("=" * 70)

print("""
Our embeddings are 16-dimensional (humans can't visualize that).
We'll use PCA to project them down to 2D for visualization.
PCA finds the two directions with the most variation.
""")

# Simple PCA implementation (no sklearn needed)
def pca_2d(embeddings):
    """Project embeddings to 2D using PCA."""
    # Center the data
    mean = embeddings.mean(dim=0)
    centered = embeddings - mean
    # Compute covariance matrix
    cov = centered.T @ centered / (centered.shape[0] - 1)
    # Eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    # Take top 2 components (largest eigenvalues are last)
    top2 = eigenvectors[:, -2:]
    # Project
    projected = centered @ top2
    return projected


coords = pca_2d(learned_embeds)

# ASCII scatter plot
def ascii_scatter(coords, labels, width=60, height=25):
    """Draw an ASCII scatter plot."""
    xs = coords[:, 0].numpy()
    ys = coords[:, 1].numpy()

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    x_range = x_max - x_min or 1
    y_range = y_max - y_min or 1

    # Create grid
    grid = [[" " for _ in range(width)] for _ in range(height)]

    # Place words
    placed = []
    for i, label in enumerate(labels):
        col = int((xs[i] - x_min) / x_range * (width - len(label) - 1))
        row = int((1 - (ys[i] - y_min) / y_range) * (height - 1))
        col = max(0, min(col, width - len(label) - 1))
        row = max(0, min(row, height - 1))

        # Check for overlap
        can_place = True
        for r, c, l in placed:
            if abs(r - row) <= 0 and abs(c - col) < len(l) + 1:
                row = min(row + 1, height - 1)

        for j, ch in enumerate(label):
            if col + j < width:
                grid[row][col + j] = ch
        placed.append((row, col, label))

    # Draw
    border = "+" + "-" * width + "+"
    print(border)
    for row in grid:
        print("|" + "".join(row) + "|")
    print(border)


# Select interesting words to plot
plot_words = ["king", "queen", "prince", "princess", "man", "woman",
              "cat", "dog", "mouse", "animal",
              "sat", "chased", "ruled",
              "mat", "rug", "crown"]

plot_indices = [w2i[w] for w in plot_words if w in w2i]
plot_labels = [i2w[i] for i in plot_indices]
plot_coords = coords[plot_indices]

print("2D Projection of Learned Embeddings:")
print("(Words that appear in similar contexts cluster together)\n")
ascii_scatter(plot_coords, plot_labels)

print("""
Look for clusters:
  - Royalty words (king, queen, prince, princess) should be near each other
  - Animal words (cat, dog, mouse) should cluster together
  - Action words (sat, chased, ruled) should group
  - The spatial DIRECTION between king→queen might parallel man→woman
""")


# ============================================================================
# STAGE 6: POSITIONAL EMBEDDINGS
# ============================================================================

print("=" * 70)
print("STAGE 6: Positional Embeddings — Where Am I in the Sentence?")
print("=" * 70)

print("""
TOKEN embeddings tell the model WHAT each word is.
But transformers process all positions in PARALLEL (not left-to-right).
So the model doesn't know word ORDER without help.

  "dog bites man" vs "man bites dog" — same tokens, different meaning!

POSITIONAL EMBEDDINGS tell the model WHERE each token is in the sequence.

The final input to a transformer is:
  input = token_embedding + positional_embedding

Two approaches:
  1. LEARNED positional embeddings (GPT-2): one learnable vector per position
  2. SINUSOIDAL positional embeddings (original Transformer): computed with sin/cos
""")

# Approach 1: Learned positional embeddings
print("--- Approach 1: Learned Positional Embeddings (GPT-2 style) ---\n")

max_seq_len = 8
pos_embed_learned = nn.Embedding(max_seq_len, EMBED_DIM)

print(f"  Position embedding table: [{max_seq_len} positions × {EMBED_DIM} dims]")
print(f"  Each position (0, 1, 2, ..., {max_seq_len-1}) gets its own learnable vector")
print(f"  These are trained along with the rest of the model\n")

# Show how token + position embeddings combine
sentence = "the cat sat on"
words = sentence.split()
token_ids = torch.tensor([w2i[w] for w in words])
position_ids = torch.arange(len(words))

tok_embeds = model.embeddings(token_ids)  # [4, 16] - what each word IS
pos_embeds = pos_embed_learned(position_ids)  # [4, 16] - where each word IS

combined = tok_embeds + pos_embeds  # [4, 16] - what + where

print(f"  Sentence: '{sentence}'")
print(f"  Token IDs:    {token_ids.tolist()}")
print(f"  Position IDs: {position_ids.tolist()}\n")
print(f"  Token embedding shape:    {tok_embeds.shape}")
print(f"  Position embedding shape: {pos_embeds.shape}")
print(f"  Combined (tok + pos):     {combined.shape}")

print(f"\n  For word '{words[0]}' at position 0:")
t = tok_embeds[0][:4].tolist()
p = pos_embeds[0][:4].tolist()
c = combined[0][:4].tolist()
print(f"    Token embed (first 4 dims): [{', '.join(f'{v:+.3f}' for v in t)}]")
print(f"  + Pos embed   (first 4 dims): [{', '.join(f'{v:+.3f}' for v in p)}]")
print(f"  = Combined    (first 4 dims): [{', '.join(f'{v:+.3f}' for v in c)}]")

# Approach 2: Sinusoidal positional embeddings
print(f"\n--- Approach 2: Sinusoidal Positional Embeddings (Original Transformer) ---\n")

print("""  Instead of learning positions, COMPUTE them with sin and cos waves:

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

  Why sin/cos?
  - Each position gets a unique pattern
  - The model can learn to attend to relative positions
  - Generalizes to sequence lengths not seen during training
""")

def sinusoidal_position_encoding(max_len, d_model):
    """Compute sinusoidal position encodings."""
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

    pe[:, 0::2] = torch.sin(position * div_term)  # even dimensions
    pe[:, 1::2] = torch.cos(position * div_term)  # odd dimensions
    return pe


pe = sinusoidal_position_encoding(max_seq_len, EMBED_DIM)

# Visualize the sinusoidal patterns as ASCII heatmap
print("  Sinusoidal Position Encoding Heatmap:")
print(f"  (rows = positions 0-{max_seq_len-1}, columns = embedding dimensions 0-{EMBED_DIM-1})\n")

def value_to_char(v):
    """Map a value from [-1, 1] to an ASCII character."""
    chars = "░░▒▒▓▓██"
    idx = int((v + 1) / 2 * (len(chars) - 1))
    idx = max(0, min(idx, len(chars) - 1))
    return chars[idx]


print(f"       dim→ {' '.join(f'{d:2d}' for d in range(EMBED_DIM))}")
print(f"  pos↓     {'-' * (EMBED_DIM * 3)}")
for pos in range(max_seq_len):
    chars = [value_to_char(pe[pos, d].item()) for d in range(EMBED_DIM)]
    vals = " ".join(f"{c} " for c in chars)
    print(f"   {pos}       {vals}")

print(f"\n  Legend: ░ = -1.0 (low)  ▒ = 0.0 (mid)  █ = +1.0 (high)")
print(f"\n  Notice: low-frequency dims (right) change slowly across positions")
print(f"          high-frequency dims (left) change rapidly")
print(f"          → like a binary clock! Each position has a unique pattern")

# Show similarity between positions
print(f"\n  Position similarity (cosine):")
print(f"         {''.join(f'  pos{j}' for j in range(max_seq_len))}")
for i in range(max_seq_len):
    sims = []
    for j in range(max_seq_len):
        sim = F.cosine_similarity(pe[i].unsqueeze(0), pe[j].unsqueeze(0)).item()
        sims.append(f"{sim:+.2f}")
    print(f"  pos{i}  {'  '.join(sims)}")

print("""
  Notice: nearby positions have HIGH similarity, distant ones have LOW.
  This is by design — it lets the model reason about relative distance!
""")


# ============================================================================
# STAGE 7: PUTTING IT ALL TOGETHER — Full Embedding Pipeline
# ============================================================================

print("=" * 70)
print("STAGE 7: Full Embedding Pipeline (as in a real Transformer)")
print("=" * 70)


class TransformerEmbedding(nn.Module):
    """
    Complete embedding layer as used in GPT-style models.
    Combines token embeddings + positional embeddings + dropout.
    """
    def __init__(self, vocab_size, embed_dim, max_seq_len, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = embed_dim

    def forward(self, token_ids):
        B, T = token_ids.shape
        tok_emb = self.token_embedding(token_ids)          # [B, T, embed_dim]
        pos_ids = torch.arange(T, device=token_ids.device) # [T]
        pos_emb = self.position_embedding(pos_ids)         # [T, embed_dim]
        x = tok_emb + pos_emb                              # [B, T, embed_dim]
        x = self.dropout(x)
        return x


# Demo with real dimensions
REAL_EMBED_DIM = 64
MAX_SEQ = 128
full_embed = TransformerEmbedding(V, REAL_EMBED_DIM, MAX_SEQ)

# Simulate a batch of 2 sentences
sentence1 = "the king wore a crown"
sentence2 = "the cat sat on mat"
s1_ids = torch.tensor([[w2i[w] for w in sentence1.split()]])
s2_ids = torch.tensor([[w2i[w] for w in sentence2.split()]])
batch = torch.cat([s1_ids, s2_ids], dim=0)  # [2, 5]

output = full_embed(batch)

print(f"""
  Input:  2 sentences, each 5 tokens
  Batch shape:  {batch.shape}   (batch_size=2, seq_len=5)
  Output shape: {output.shape}  (batch_size=2, seq_len=5, embed_dim={REAL_EMBED_DIM})

  What happened:
    1. Each token ID looked up its {REAL_EMBED_DIM}-dim token embedding
    2. Each position (0-4) looked up its {REAL_EMBED_DIM}-dim position embedding
    3. Token + Position embeddings were ADDED together
    4. Dropout randomly zeroed some values (regularization)
    5. Result: each token is now a rich {REAL_EMBED_DIM}-dimensional vector
       encoding BOTH what the token is AND where it appears

  This output goes directly into the SELF-ATTENTION layers next!
""")

# Show the parameter count
tok_params = V * REAL_EMBED_DIM
pos_params = MAX_SEQ * REAL_EMBED_DIM
total_params = sum(p.numel() for p in full_embed.parameters())
print(f"  Parameter count:")
print(f"    Token embeddings:    {V:>5} × {REAL_EMBED_DIM} = {tok_params:>8,} parameters")
print(f"    Position embeddings: {MAX_SEQ:>5} × {REAL_EMBED_DIM} = {pos_params:>8,} parameters")
print(f"    Total:                         {total_params:>8,} parameters")

print(f"""
  In GPT-3 (175B parameters):
    Token embeddings:    50,257 × 12,288 = 617,558,016 (~617M parameters!)
    Position embeddings:  2,048 × 12,288 =  25,165,824 (~25M parameters)
    That's 642M params JUST for embeddings — before any attention or FFN!
""")


# ============================================================================
# SUMMARY
# ============================================================================

print("=" * 70)
print("WHAT YOU LEARNED")
print("=" * 70)
print("""
1. RAW TOKEN IDS are meaningless numbers — the network needs rich vectors

2. ONE-HOT ENCODING fails: too sparse, too large, no similarity info

3. EMBEDDINGS are dense, learnable vectors that capture meaning:
   - Similar words → similar vectors (cat ≈ dog)
   - Relationships as directions (king - man + woman ≈ queen)
   - Learned automatically from data through backpropagation

4. POSITIONAL EMBEDDINGS tell the model WHERE each token sits:
   - Learned (GPT-2): one trainable vector per position
   - Sinusoidal (original Transformer): computed with sin/cos
   - Combined: input = token_embedding + position_embedding

5. REAL-WORLD SCALE:
   - GPT-3: 50K vocab × 12,288 dims = 617M embedding parameters
   - Embedding dimensions: 768 (GPT-2) to 12,288 (GPT-3)
   - These are the first and last layers the data touches

6. THE FULL PICTURE SO FAR:
   ┌──────────────────────────────────────────────────────┐
   │  Text                                                │
   │    ↓  Tokenization (BPE)                             │
   │  Token IDs                                           │
   │    ↓  Token Embedding (lookup table)                 │
   │  Token Vectors                                       │
   │    +  Position Embedding                             │
   │  Input Vectors                                       │
   │    ↓  ← ← ← YOU ARE HERE                            │
   │  Self-Attention  ← NEXT TOPIC!                       │
   │    ↓                                                 │
   │  Feed-Forward Network                                │
   │    ↓                                                 │
   │  Output Probabilities                                │
   │    ↓  Decode (lookup table in reverse)               │
   │  Generated Text                                      │
   └──────────────────────────────────────────────────────┘

NEXT STEP: Self-Attention — the mechanism that lets the model look at
ALL other tokens to understand context. This is the core of transformers!
""")
