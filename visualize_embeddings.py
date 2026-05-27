"""
=============================================================================
 EMBEDDING VISUALIZATION — See How Words Cluster by Meaning
=============================================================================
 Trains word embeddings, then creates 4 visualizations:
   1. 2D scatter plot with color-coded semantic groups
   2. Cosine similarity heatmap
   3. Analogy arrows (king - man + woman ≈ queen)
   4. Positional embedding patterns
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import math
import os

# ── Training corpus ──────────────────────────────────────────────────────────
# Carefully designed so context reveals semantic relationships.

training_sentences = [
    # Animals
    "the cat sat on the mat",
    "the dog sat on the rug",
    "the cat chased the mouse",
    "the dog chased the cat",
    "the cat is a small animal",
    "the dog is a loyal animal",
    "the mouse is a tiny animal",
    "a cat and a dog are both animals",
    "the dog ran across the rug",
    "the cat slept on the mat",
    # Royalty
    "a king ruled the land",
    "a queen ruled the kingdom",
    "the king wore a crown",
    "the queen wore a crown",
    "the prince is the son of the king",
    "the princess is the daughter of the queen",
    "the king sat on the throne",
    "the queen sat on the throne",
    "a great king ruled a great kingdom",
    "a wise queen ruled the land",
    # Gender
    "a man became the king",
    "a woman became the queen",
    "the man wore a crown",
    "the woman wore a crown",
    "a boy is a young man",
    "a girl is a young woman",
    # Descriptors
    "a great king and a loyal dog",
    "a wise queen and a small cat",
    "the tiny mouse ran across the land",
    "the young prince chased the small mouse",
]

# ── Build vocabulary ─────────────────────────────────────────────────────────
all_words = set()
for s in training_sentences:
    all_words.update(s.split())
word_list = sorted(all_words)
w2i = {w: i for i, w in enumerate(word_list)}
i2w = {i: w for w, i in w2i.items()}
V = len(word_list)

# ── Create training pairs (Skip-gram style: context → target) ───────────────
def create_training_data(sentences, w2i, window=2):
    data = []
    for sentence in sentences:
        words = sentence.split()
        for i, word in enumerate(words):
            target = w2i[word]
            for j in range(max(0, i - window), min(len(words), i + window + 1)):
                if j != i:
                    context = w2i[words[j]]
                    data.append((context, target))
    return data

pairs = create_training_data(training_sentences, w2i)

# ── Model ────────────────────────────────────────────────────────────────────
class WordEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, context_ids):
        embeds = self.embeddings(context_ids)
        logits = self.linear(embeds)
        return logits

EMBED_DIM = 32
torch.manual_seed(42)
model = WordEmbeddingModel(V, EMBED_DIM)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

context_ids = torch.tensor([p[0] for p in pairs])
target_ids = torch.tensor([p[1] for p in pairs])

# ── Train ────────────────────────────────────────────────────────────────────
print("Training embeddings...")
for epoch in range(501):
    logits = model(context_ids)
    loss = F.cross_entropy(logits, target_ids)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"  Epoch {epoch:4d}  Loss: {loss.item():.4f}")

print("Training complete!\n")

learned = model.embeddings.weight.detach()

# ── PCA projection to 2D ────────────────────────────────────────────────────
def pca_2d(embeddings):
    mean = embeddings.mean(dim=0)
    centered = embeddings - mean
    cov = centered.T @ centered / (centered.shape[0] - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    top2 = eigenvectors[:, -2:]
    return (centered @ top2).numpy()

coords = pca_2d(learned)

# ── Semantic groups for coloring ─────────────────────────────────────────────
groups = {
    "Animals":  ["cat", "dog", "mouse", "animal", "animals"],
    "Royalty":  ["king", "queen", "prince", "princess", "crown", "throne", "kingdom"],
    "People":   ["man", "woman", "boy", "girl", "son", "daughter"],
    "Actions":  ["sat", "chased", "ruled", "wore", "became", "ran", "slept"],
    "Places":   ["mat", "rug", "land"],
    "Adjectives": ["small", "tiny", "loyal", "great", "wise", "young"],
}

group_colors = {
    "Animals": "#e74c3c",
    "Royalty": "#9b59b6",
    "People": "#3498db",
    "Actions": "#2ecc71",
    "Places": "#e67e22",
    "Adjectives": "#1abc9c",
}

def get_group(word):
    for group, words in groups.items():
        if word in words:
            return group
    return None

output_dir = os.path.dirname(__file__)

# ==========================================================================
# PLOT 1: 2D Scatter — Words colored by semantic group
# ==========================================================================
print("Creating Plot 1: 2D Embedding Scatter...")

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_facecolor("#1a1a2e")
fig.patch.set_facecolor("#16213e")

plot_words = []
for group_words in groups.values():
    plot_words.extend(group_words)
plot_words = [w for w in plot_words if w in w2i]

for word in plot_words:
    idx = w2i[word]
    group = get_group(word)
    color = group_colors.get(group, "#95a5a6")
    x, y = coords[idx]
    ax.scatter(x, y, c=color, s=120, zorder=5, edgecolors="white", linewidth=0.5)
    ax.annotate(word, (x, y), fontsize=11, fontweight="bold", color="white",
                textcoords="offset points", xytext=(8, 6),
                bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.7, edgecolor="none"))

# Draw lines between related pairs
related = [("cat", "dog"), ("king", "queen"), ("man", "woman"),
           ("prince", "princess"), ("boy", "girl"), ("mat", "rug"),
           ("son", "daughter")]
for w1, w2 in related:
    if w1 in w2i and w2 in w2i:
        x1, y1 = coords[w2i[w1]]
        x2, y2 = coords[w2i[w2]]
        ax.plot([x1, x2], [y1, y2], color="white", alpha=0.2, linewidth=1, linestyle="--")

handles = [mpatches.Patch(color=c, label=g) for g, c in group_colors.items()]
ax.legend(handles=handles, loc="upper left", fontsize=10, facecolor="#16213e",
          edgecolor="white", labelcolor="white", framealpha=0.8)

ax.set_title("Word Embeddings in 2D (PCA Projection)\nWords with similar meaning cluster together",
             fontsize=16, color="white", pad=15)
ax.set_xlabel("Principal Component 1", color="white", fontsize=12)
ax.set_ylabel("Principal Component 2", color="white", fontsize=12)
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_color("white")
    spine.set_alpha(0.3)

plt.tight_layout()
path1 = os.path.join(output_dir, "plot1_embedding_scatter.png")
plt.savefig(path1, dpi=150, facecolor=fig.get_facecolor())
plt.close()
print(f"  Saved: {path1}")


# ==========================================================================
# PLOT 2: Cosine Similarity Heatmap
# ==========================================================================
print("Creating Plot 2: Similarity Heatmap...")

heatmap_words = ["cat", "dog", "mouse", "animal", "king", "queen", "prince",
                 "princess", "man", "woman", "boy", "girl", "crown", "throne",
                 "mat", "rug", "sat", "chased", "ruled"]
heatmap_words = [w for w in heatmap_words if w in w2i]
n = len(heatmap_words)

sim_matrix = np.zeros((n, n))
for i, w1 in enumerate(heatmap_words):
    for j, w2 in enumerate(heatmap_words):
        v1 = learned[w2i[w1]].unsqueeze(0)
        v2 = learned[w2i[w2]].unsqueeze(0)
        sim_matrix[i, j] = F.cosine_similarity(v1, v2).item()

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(sim_matrix, cmap="RdYlGn", vmin=-0.5, vmax=1.0, aspect="auto")

ax.set_xticks(range(n))
ax.set_xticklabels(heatmap_words, rotation=45, ha="right", fontsize=11)
ax.set_yticks(range(n))
ax.set_yticklabels(heatmap_words, fontsize=11)

for i in range(n):
    for j in range(n):
        val = sim_matrix[i, j]
        color = "white" if val < 0.3 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("Cosine Similarity", fontsize=12)

# Draw group boundaries
boundaries = [4, 8, 12, 14, 16]  # after animal, royalty, people, place, place, verbs
for b in boundaries:
    if b < n:
        ax.axhline(y=b - 0.5, color="white", linewidth=2)
        ax.axvline(x=b - 0.5, color="white", linewidth=2)

ax.set_title("Cosine Similarity Between Word Embeddings\n"
             "Green = similar meaning, Red = different meaning",
             fontsize=14, pad=15)
plt.tight_layout()
path2 = os.path.join(output_dir, "plot2_similarity_heatmap.png")
plt.savefig(path2, dpi=150)
plt.close()
print(f"  Saved: {path2}")


# ==========================================================================
# PLOT 3: Analogy Arrows — Showing vector relationships
# ==========================================================================
print("Creating Plot 3: Analogy Arrows...")

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_facecolor("#f8f9fa")

for word in plot_words:
    idx = w2i[word]
    group = get_group(word)
    color = group_colors.get(group, "#95a5a6")
    x, y = coords[idx]
    ax.scatter(x, y, c=color, s=100, zorder=5, edgecolors="black", linewidth=0.5)
    ax.annotate(word, (x, y), fontsize=10, fontweight="bold",
                textcoords="offset points", xytext=(6, 6))

# Draw analogy arrows
analogy_sets = [
    {
        "pairs": [("man", "king"), ("woman", "queen")],
        "color": "#e74c3c",
        "label": "person → royalty"
    },
    {
        "pairs": [("king", "prince"), ("queen", "princess")],
        "color": "#3498db",
        "label": "ruler → child"
    },
    {
        "pairs": [("man", "boy"), ("woman", "girl")],
        "color": "#2ecc71",
        "label": "adult → young"
    },
]

arrow_handles = []
for aset in analogy_sets:
    for w1, w2 in aset["pairs"]:
        if w1 in w2i and w2 in w2i:
            x1, y1 = coords[w2i[w1]]
            x2, y2 = coords[w2i[w2]]
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle="-|>", color=aset["color"],
                                       lw=2.5, connectionstyle="arc3,rad=0.1"))
    arrow_handles.append(mpatches.Patch(color=aset["color"], label=aset["label"]))

ax.legend(handles=arrow_handles, loc="upper left", fontsize=11,
          title="Analogy Directions", title_fontsize=12)

ax.set_title("Word Analogies as Parallel Arrows\n"
             "Parallel arrows = the model learned consistent relationships\n"
             "(e.g., man→king is parallel to woman→queen)",
             fontsize=14, pad=15)
ax.set_xlabel("Principal Component 1", fontsize=12)
ax.set_ylabel("Principal Component 2", fontsize=12)
ax.grid(True, alpha=0.2)

plt.tight_layout()
path3 = os.path.join(output_dir, "plot3_analogy_arrows.png")
plt.savefig(path3, dpi=150)
plt.close()
print(f"  Saved: {path3}")


# ==========================================================================
# PLOT 4: Positional Embeddings — Sinusoidal patterns
# ==========================================================================
print("Creating Plot 4: Positional Encoding Patterns...")

def sinusoidal_pe(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

max_len = 64
d_model = 64
pe = sinusoidal_pe(max_len, d_model)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# (a) Full heatmap
ax = axes[0, 0]
im = ax.imshow(pe.numpy(), cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
ax.set_xlabel("Embedding Dimension", fontsize=11)
ax.set_ylabel("Position in Sequence", fontsize=11)
ax.set_title("Sinusoidal Position Encodings\n(each row = unique position fingerprint)", fontsize=12)
plt.colorbar(im, ax=ax, shrink=0.8)

# (b) Individual dimension waves
ax = axes[0, 1]
for dim in [0, 1, 4, 8, 16, 32]:
    ax.plot(pe[:, dim].numpy(), label=f"dim {dim}", linewidth=1.5)
ax.set_xlabel("Position", fontsize=11)
ax.set_ylabel("Value", fontsize=11)
ax.set_title("Individual Dimensions as Waves\n(low dims = fast waves, high dims = slow waves)", fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (c) Position similarity matrix
ax = axes[1, 0]
pos_sim = np.zeros((max_len, max_len))
for i in range(max_len):
    for j in range(max_len):
        pos_sim[i, j] = F.cosine_similarity(pe[i].unsqueeze(0), pe[j].unsqueeze(0)).item()
im = ax.imshow(pos_sim, cmap="viridis", aspect="auto")
ax.set_xlabel("Position", fontsize=11)
ax.set_ylabel("Position", fontsize=11)
ax.set_title("Position Similarity (Cosine)\n(nearby positions are more similar)", fontsize=12)
plt.colorbar(im, ax=ax, shrink=0.8)

# (d) Relative distance vs similarity
ax = axes[1, 1]
distances = []
similarities = []
for i in range(max_len):
    for j in range(i + 1, max_len):
        distances.append(abs(i - j))
        similarities.append(pos_sim[i, j])
ax.scatter(distances, similarities, alpha=0.15, s=8, color="#3498db")
# Running average
from collections import defaultdict
dist_sims = defaultdict(list)
for d, s in zip(distances, similarities):
    dist_sims[d].append(s)
avg_dists = sorted(dist_sims.keys())
avg_sims = [np.mean(dist_sims[d]) for d in avg_dists]
ax.plot(avg_dists, avg_sims, color="#e74c3c", linewidth=2.5, label="Average")
ax.set_xlabel("Distance Between Positions", fontsize=11)
ax.set_ylabel("Cosine Similarity", fontsize=11)
ax.set_title("Similarity Decreases with Distance\n(model can infer relative position)", fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.suptitle("Positional Embeddings — How the Model Knows Word Order",
             fontsize=16, fontweight="bold", y=1.01)
plt.tight_layout()
path4 = os.path.join(output_dir, "plot4_positional_embeddings.png")
plt.savefig(path4, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {path4}")


# ==========================================================================
# Summary
# ==========================================================================
print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  4 visualizations saved:                                       ║
║                                                                ║
║  1. plot1_embedding_scatter.png                                ║
║     → Words colored by category, clustered by meaning          ║
║                                                                ║
║  2. plot2_similarity_heatmap.png                               ║
║     → Pairwise similarity matrix (green=similar, red=different)║
║                                                                ║
║  3. plot3_analogy_arrows.png                                   ║
║     → Vector arithmetic: man→king parallel to woman→queen      ║
║                                                                ║
║  4. plot4_positional_embeddings.png                             ║
║     → How sinusoidal encodings give each position a fingerprint║
╚══════════════════════════════════════════════════════════════════╝
""")
