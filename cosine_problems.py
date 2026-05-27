"""
=============================================================================
 PROBLEMS WITH COSINE SIMILARITY — What Can Go Wrong
=============================================================================
 Cosine similarity is the standard for comparing embeddings, but it has
 real flaws that affect how LLMs behave. Understanding these helps you
 understand why models sometimes fail in surprising ways.
=============================================================================
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import math
import os

output_dir = os.path.dirname(__file__)


# ============================================================================
# PROBLEM 1: MAGNITUDE CARRIES MEANING — But Cosine Throws It Away
# ============================================================================

print("=" * 70)
print("PROBLEM 1: Cosine Throws Away Magnitude (Sometimes That's Bad!)")
print("=" * 70)

print("""
  Cosine IGNORES vector length. Usually that's good.
  But sometimes magnitude carries important information!

  Example: Word frequency affects embedding magnitude.
  Words seen MORE often during training get LARGER embeddings.

    "the"     → high magnitude (seen millions of times)
    "quantum" → low magnitude  (seen rarely)

  If two words have the same direction but different magnitudes,
  cosine says they're identical — but they might not be!
""")

# Demo
common_word = torch.tensor([4.0, 3.0])     # "the" — large magnitude
rare_word   = torch.tensor([0.4, 0.3])     # "thee" — same direction, small magnitude
different   = torch.tensor([3.8, 3.2])     # "a" — slightly different direction, similar magnitude

cos_common_rare = F.cosine_similarity(common_word.unsqueeze(0), rare_word.unsqueeze(0)).item()
cos_common_diff = F.cosine_similarity(common_word.unsqueeze(0), different.unsqueeze(0)).item()
euc_common_rare = torch.dist(common_word, rare_word).item()
euc_common_diff = torch.dist(common_word, different).item()

print(f"  'the'  = {common_word.tolist()}   (common word, large vector)")
print(f"  'thee' = {rare_word.tolist()}  (rare word, same direction, small vector)")
print(f"  'a'    = {different.tolist()}   (common word, slightly different direction)")

print(f"""
  Cosine('the', 'thee') = {cos_common_rare:.4f}  ← Cosine says IDENTICAL!
  Cosine('the', 'a')    = {cos_common_diff:.4f}  ← Cosine says slightly different

  But 'the' and 'a' are both common determiners used similarly.
  'thee' is archaic Shakespeare English — quite different in usage!

  Euclidean('the', 'thee') = {euc_common_rare:.4f}  ← Euclidean catches the difference
  Euclidean('the', 'a')    = {euc_common_diff:.4f}  ← Euclidean says these are close

  LESSON: Cosine can't distinguish "same meaning, different confidence"
          from "truly identical". Magnitude = model's confidence/frequency.
""")


# ============================================================================
# PROBLEM 2: THE CURSE OF DIMENSIONALITY
# ============================================================================

print("=" * 70)
print("PROBLEM 2: Curse of Dimensionality — Everything Looks the Same")
print("=" * 70)

print("""
  In HIGH dimensions (real embeddings are 768 to 12,288 dims),
  random vectors become nearly ORTHOGONAL (cosine ≈ 0).

  This means ALL similarities cluster around 0, making it
  hard to tell what's truly similar vs. truly different.
""")

dims_to_test = [2, 10, 50, 100, 768, 4096]
print(f"  {'Dimensions':>12}  {'Mean Cosine':>12}  {'Std Dev':>10}  Distribution")
print(f"  {'-'*12}  {'-'*12}  {'-'*10}  {'-'*30}")

dim_results = []
for dim in dims_to_test:
    # Generate 1000 random vector pairs
    a = torch.randn(1000, dim)
    b = torch.randn(1000, dim)
    sims = F.cosine_similarity(a, b).numpy()
    mean_sim = sims.mean()
    std_sim = sims.std()
    dim_results.append((dim, sims))

    bar_width = int(std_sim * 100)
    bar = "█" * bar_width + "░" * (20 - bar_width)
    print(f"  {dim:>12}  {mean_sim:>+12.4f}  {std_sim:>10.4f}  {bar}")

print(f"""
  As dimensions increase:
    - Mean cosine → 0 (everything becomes perpendicular)
    - Std deviation → 0 (all similarities become the SAME)

  In 4096 dimensions, a random pair has cosine ≈ 0.00 ± 0.01
  So a "meaningful" similarity of 0.05 is actually significant!
  But it LOOKS tiny compared to the 0.96 we saw in 2D examples.

  REAL IMPACT: In high-dim embeddings, the difference between
  "related" and "unrelated" might be just 0.1 to 0.3 — not the
  dramatic 0.9 vs 0.1 we see in tutorials with low dimensions.
""")


# ============================================================================
# PROBLEM 3: THE HUBNESS PROBLEM
# ============================================================================

print("=" * 70)
print("PROBLEM 3: Hubness — Some Words Are 'Nearest Neighbor' to Everything")
print("=" * 70)

print("""
  In high dimensions, some vectors become "HUBS" — they appear as
  the nearest neighbor to MANY other vectors, even unrelated ones.

  Imagine searching for "most similar word to X":
    - For X = "cat"    → nearest = "hub_word"
    - For X = "rocket" → nearest = "hub_word"
    - For X = "piano"  → nearest = "hub_word"

  The hub word isn't truly similar to all of these — it just sits
  in a region of the space where it's close to many vectors.
""")

# Demonstrate hubness
torch.manual_seed(42)
dim = 100
n_vectors = 200

# Create vectors with one "hub" that's close to many
vectors = torch.randn(n_vectors, dim) * 0.5
hub_vector = torch.zeros(dim)  # origin-like vector is a natural hub

# Count how often each vector is someone else's nearest neighbor
nearest_counts = torch.zeros(n_vectors)
for i in range(n_vectors):
    sims = F.cosine_similarity(vectors[i].unsqueeze(0), vectors)
    sims[i] = -2  # exclude self
    nearest = sims.argmax().item()
    nearest_counts[nearest] += 1

max_hub = nearest_counts.max().item()
hub_idx = nearest_counts.argmax().item()
never_nearest = (nearest_counts == 0).sum().item()

print(f"  With {n_vectors} random vectors in {dim} dimensions:")
print(f"    - Most popular 'hub': vector #{hub_idx} (nearest neighbor to {int(max_hub)} others!)")
print(f"    - Vectors that are NEVER anyone's nearest: {never_nearest} ({never_nearest/n_vectors*100:.0f}%)")
print(f"""
  So {never_nearest/n_vectors*100:.0f}% of words would NEVER be retrieved in a
  similarity search — even if they're the correct answer!

  REAL IMPACT: When you search for similar documents/words,
  hubs dominate the results. You keep getting the same few
  "popular" matches instead of the truly relevant ones.

  MITIGATION: Use CSLS (Cross-domain Similarity Local Scaling)
  or normalize by average similarity to reduce hub dominance.
""")


# ============================================================================
# PROBLEM 4: ANISOTROPY — Embeddings Cluster in a Narrow Cone
# ============================================================================

print("=" * 70)
print("PROBLEM 4: Anisotropy — All Embeddings Point the Same Way")
print("=" * 70)

print("""
  In theory, embeddings should spread out uniformly in all directions.
  In practice, trained embeddings cluster in a NARROW CONE.

  This means:
    - ALL cosine similarities are high (0.5 to 0.9)
    - Even UNRELATED words have high similarity
    - The useful range shrinks dramatically

  This is called ANISOTROPY (non-uniform distribution).
""")

# Simulate anisotropic embeddings (like real LLM embeddings)
torch.manual_seed(42)
n_words = 300
dim = 64

# Isotropic (ideal): random directions
isotropic = torch.randn(n_words, dim)

# Anisotropic (realistic): all shifted toward one direction
bias_direction = torch.randn(dim)
bias_direction = bias_direction / bias_direction.norm()
anisotropic = torch.randn(n_words, dim) * 0.3 + bias_direction * 2.0

# Compare similarity distributions
iso_sims = []
aniso_sims = []
for i in range(min(500, n_words)):
    for j in range(i + 1, min(500, n_words)):
        iso_sims.append(F.cosine_similarity(
            isotropic[i].unsqueeze(0), isotropic[j].unsqueeze(0)).item())
        aniso_sims.append(F.cosine_similarity(
            anisotropic[i].unsqueeze(0), anisotropic[j].unsqueeze(0)).item())

iso_sims = np.array(iso_sims)
aniso_sims = np.array(aniso_sims)

print(f"  Isotropic (ideal) embeddings:")
print(f"    Mean similarity: {iso_sims.mean():+.4f}")
print(f"    Range: [{iso_sims.min():+.4f}, {iso_sims.max():+.4f}]")
print(f"    Useful range: {iso_sims.max() - iso_sims.min():.4f}")

print(f"\n  Anisotropic (real LLM) embeddings:")
print(f"    Mean similarity: {aniso_sims.mean():+.4f}")
print(f"    Range: [{aniso_sims.min():+.4f}, {aniso_sims.max():+.4f}]")
print(f"    Useful range: {aniso_sims.max() - aniso_sims.min():.4f}")

print(f"""
  With anisotropic embeddings:
    - "cat" vs "dog"     might have cosine = 0.85
    - "cat" vs "quantum" might have cosine = 0.78
    - Difference is only 0.07 — hard to set a meaningful threshold!

  With isotropic embeddings:
    - "cat" vs "dog"     might have cosine = 0.65
    - "cat" vs "quantum" might have cosine = 0.02
    - Difference is 0.63 — much easier to separate!

  MITIGATION: Subtract the mean embedding before comparing.
  This "centers" the embeddings and restores useful spread.
""")


# ============================================================================
# PROBLEM 5: NOT A TRUE DISTANCE METRIC
# ============================================================================

print("=" * 70)
print("PROBLEM 5: Triangle Inequality Violation — Broken Reasoning")
print("=" * 70)

print("""
  A true distance metric must satisfy the TRIANGLE INEQUALITY:
    distance(A, C) ≤ distance(A, B) + distance(B, C)

  In plain English: the direct path is never longer than a detour.

  Cosine similarity is NOT a distance metric. This means:
    - A is similar to B ✓
    - B is similar to C ✓
    - A is similar to C ✗  ← NOT guaranteed!
""")

# Construct a concrete counterexample
A = torch.tensor([1.0, 0.0])
B = torch.tensor([0.707, 0.707])  # 45° from A
C = torch.tensor([0.0, 1.0])      # 90° from A, 45° from B

sim_AB = F.cosine_similarity(A.unsqueeze(0), B.unsqueeze(0)).item()
sim_BC = F.cosine_similarity(B.unsqueeze(0), C.unsqueeze(0)).item()
sim_AC = F.cosine_similarity(A.unsqueeze(0), C.unsqueeze(0)).item()

print(f"  A = {A.tolist()}")
print(f"  B = {B.tolist()}")
print(f"  C = {C.tolist()}")
print(f"""
  cosine(A, B) = {sim_AB:.4f}  ← A and B are similar (45° apart)
  cosine(B, C) = {sim_BC:.4f}  ← B and C are similar (45° apart)
  cosine(A, C) = {sim_AC:.4f}  ← A and C are NOT similar (90° apart)!

  If similarity were transitive:
    "cat ≈ pet" and "pet ≈ rock" should mean "cat ≈ rock"
    But it doesn't! "pet rock" breaks the chain.

  REAL IMPACT:
    - RAG retrieval: document A matches query, document B matches A,
      but B might be irrelevant to the original query.
    - Clustering: nearest-neighbor chains can link unrelated items.
    - Recommendations: "similar to similar" doesn't mean "similar".
""")


# ============================================================================
# PROBLEM 6: ZERO AND NEAR-ZERO VECTORS
# ============================================================================

print("=" * 70)
print("PROBLEM 6: Zero & Near-Zero Vectors — Division by Zero")
print("=" * 70)

A = torch.tensor([0.3, 0.4])
zero = torch.tensor([0.0, 0.0])
tiny = torch.tensor([1e-10, 1e-10])

print(f"  cosine(A, zero_vector):")
try:
    result = F.cosine_similarity(A.unsqueeze(0), zero.unsqueeze(0))
    print(f"    Result: {result.item()}")
except Exception as e:
    print(f"    ERROR: {e}")

result_tiny = F.cosine_similarity(A.unsqueeze(0), tiny.unsqueeze(0)).item()
print(f"""
  cosine(A, near_zero) = {result_tiny:.4f}

  The formula divides by ‖A‖ × ‖B‖. If either is zero → undefined!
  Near-zero vectors produce numerically unstable results.

  WHEN THIS HAPPENS:
    - Padding tokens often get near-zero embeddings
    - Rare tokens with little training can have tiny embeddings
    - After certain normalization operations

  MITIGATION: Add a small epsilon: ‖A‖ + ε in the denominator.
  PyTorch does this internally (eps=1e-8 by default).
""")


# ============================================================================
# PROBLEM 7: SENSITIVE TO IRRELEVANT DIMENSIONS
# ============================================================================

print("=" * 70)
print("PROBLEM 7: Noise in Extra Dimensions Degrades Similarity")
print("=" * 70)

# Two clearly similar vectors in 2D
A_2d = torch.tensor([3.0, 4.0])
B_2d = torch.tensor([4.0, 3.0])
sim_2d = F.cosine_similarity(A_2d.unsqueeze(0), B_2d.unsqueeze(0)).item()
print(f"  2D vectors (pure signal):")
print(f"    A = {A_2d.tolist()}, B = {B_2d.tolist()}")
print(f"    Cosine = {sim_2d:.4f}")

# Add noisy dimensions
print(f"\n  Same vectors + random noise dimensions:")
print(f"  {'Extra dims':>12} {'Cosine':>10}  Change")
print(f"  {'-'*12} {'-'*10}  {'-'*20}")

torch.manual_seed(42)
for extra_dims in [0, 2, 10, 50, 200, 1000]:
    noise_a = torch.randn(extra_dims) * 0.5
    noise_b = torch.randn(extra_dims) * 0.5
    A_noisy = torch.cat([A_2d, noise_a])
    B_noisy = torch.cat([B_2d, noise_b])
    sim = F.cosine_similarity(A_noisy.unsqueeze(0), B_noisy.unsqueeze(0)).item()
    change = sim - sim_2d
    bar = "▓" * max(0, int(sim * 20))
    print(f"  {extra_dims:>12} {sim:>+10.4f}  {change:>+.4f} {bar}")

print(f"""
  The meaningful signal was in the first 2 dimensions.
  But adding noisy dimensions DROWNS OUT the real similarity!

  REAL IMPACT: If only 50 of 768 embedding dimensions carry
  meaningful information, the other 718 add noise that
  dilutes the cosine similarity signal.

  MITIGATION: Dimensionality reduction (PCA) or learned
  projections to remove noisy dimensions before comparing.
""")


# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("=" * 70)
print("Creating Visualizations...")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(20, 13))
fig.suptitle("Problems with Cosine Similarity", fontsize=18, fontweight="bold", y=1.02)

# --- (1) Magnitude loss ---
ax = axes[0, 0]
words = {"the": [4, 3], "thee": [0.4, 0.3], "a": [3.8, 3.2]}
colors_w = {"the": "#e74c3c", "thee": "#3498db", "a": "#2ecc71"}
for word, pos in words.items():
    ax.annotate("", xy=(pos[0], pos[1]), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=colors_w[word], lw=3))
    ax.text(pos[0] + 0.1, pos[1] + 0.15, f'"{word}"', fontsize=12,
            fontweight="bold", color=colors_w[word])

ax.set_xlim(-0.5, 5)
ax.set_ylim(-0.5, 4)
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color="black", linewidth=0.5)
ax.axvline(x=0, color="black", linewidth=0.5)
ax.set_title('P1: Magnitude Loss\n"the" and "thee" look identical\n(cos=1.0) despite different usage',
             fontsize=11, color="#e74c3c")

# --- (2) Curse of dimensionality ---
ax = axes[0, 1]
for dim, sims in dim_results:
    ax.hist(sims, bins=50, alpha=0.5, label=f"d={dim}", density=True)
ax.set_xlabel("Cosine Similarity", fontsize=10)
ax.set_ylabel("Density", fontsize=10)
ax.legend(fontsize=8)
ax.set_title("P2: Curse of Dimensionality\nHigher dims → all similarities\ncluster around 0",
             fontsize=11, color="#e74c3c")

# --- (3) Hubness ---
ax = axes[0, 2]
# Show distribution of "nearest neighbor count"
counts, bins, _ = ax.hist(nearest_counts.numpy(), bins=range(int(nearest_counts.max()) + 2),
                           color="#3498db", edgecolor="black", alpha=0.7)
ax.axvline(x=nearest_counts.mean(), color="#e74c3c", linestyle="--", linewidth=2,
           label=f"Mean = {nearest_counts.mean():.1f}")
ax.set_xlabel("Times a vector is nearest neighbor", fontsize=10)
ax.set_ylabel("Count of vectors", fontsize=10)
ax.legend(fontsize=9)
ax.set_title(f"P3: Hubness Problem\nMost vectors are nearest to 0-1 others\n"
             f"but 'hubs' dominate ({int(max_hub)} times!)",
             fontsize=11, color="#e74c3c")

# --- (4) Anisotropy ---
ax = axes[1, 0]
ax.hist(iso_sims, bins=50, alpha=0.6, color="#2ecc71", label="Isotropic (ideal)", density=True)
ax.hist(aniso_sims, bins=50, alpha=0.6, color="#e74c3c", label="Anisotropic (real)", density=True)
ax.set_xlabel("Cosine Similarity", fontsize=10)
ax.set_ylabel("Density", fontsize=10)
ax.legend(fontsize=9)
ax.set_title("P4: Anisotropy\nReal embeddings cluster in narrow cone\n→ all similarities are high",
             fontsize=11, color="#e74c3c")

# --- (5) Triangle inequality ---
ax = axes[1, 1]
ax.set_xlim(-0.5, 2)
ax.set_ylim(-0.5, 1.5)
ax.set_aspect("equal")

points_tri = {"A\n[1, 0]": (1, 0), "B\n[.71, .71]": (0.707, 0.707), "C\n[0, 1]": (0, 1)}
colors_tri = ["#e74c3c", "#f39c12", "#3498db"]
for (label, pos), color in zip(points_tri.items(), colors_tri):
    ax.annotate("", xy=pos, xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=3))
    ax.text(pos[0] + 0.08, pos[1] + 0.08, label, fontsize=10, fontweight="bold", color=color)

# Annotate similarities
ax.annotate(f"cos=0.71", xy=(0.85, 0.35), fontsize=10, color="#2ecc71",
            fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="#2ecc71"))
ax.annotate(f"cos=0.71", xy=(0.2, 0.9), fontsize=10, color="#2ecc71",
            fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="#2ecc71"))
ax.annotate(f"cos=0.00!", xy=(0.35, -0.15), fontsize=11, color="#e74c3c",
            fontweight="bold",
            bbox=dict(facecolor="#ffe0e0", alpha=0.9, edgecolor="#e74c3c"))

ax.grid(True, alpha=0.3)
ax.axhline(y=0, color="black", linewidth=0.5)
ax.axvline(x=0, color="black", linewidth=0.5)
ax.set_title("P5: Triangle Inequality Violation\nA≈B and B≈C but A≠C\n(similarity isn't transitive)",
             fontsize=11, color="#e74c3c")

# --- (6) Noise dimensions ---
ax = axes[1, 2]
extra_dims_list = [0, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
sims_by_noise = []
torch.manual_seed(42)
for extra_dims in extra_dims_list:
    noise_a = torch.randn(extra_dims) * 0.5
    noise_b = torch.randn(extra_dims) * 0.5
    A_n = torch.cat([A_2d, noise_a])
    B_n = torch.cat([B_2d, noise_b])
    sim_val = F.cosine_similarity(A_n.unsqueeze(0), B_n.unsqueeze(0)).item()
    sims_by_noise.append(sim_val)

ax.plot(extra_dims_list, sims_by_noise, "o-", color="#e74c3c", linewidth=2.5, markersize=8)
ax.axhline(y=sim_2d, color="#2ecc71", linestyle="--", linewidth=1.5,
           label=f"True similarity ({sim_2d:.2f})")
ax.fill_between(extra_dims_list, sims_by_noise, sim_2d, alpha=0.15, color="#e74c3c")
ax.set_xlabel("Number of Noisy Dimensions Added", fontsize=10)
ax.set_ylabel("Cosine Similarity", fontsize=10)
ax.legend(fontsize=9)
ax.set_title("P6: Noise Dimensions Dilute Signal\nAdding irrelevant dims destroys\nthe real similarity signal",
             fontsize=11, color="#e74c3c")
ax.grid(True, alpha=0.3)

plt.tight_layout()
path = os.path.join(output_dir, "cosine_problems.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {path}")


# ============================================================================
# SUMMARY: MITIGATIONS
# ============================================================================

print(f"""
{'=' * 70}
SUMMARY: Problems & What Real Systems Do About Them
{'=' * 70}

  ┌─────────────────────┬────────────────────────────────────────────┐
  │ Problem             │ Mitigation                                │
  ├─────────────────────┼────────────────────────────────────────────┤
  │ 1. Magnitude loss   │ Use BOTH cosine AND magnitude when needed │
  │                     │ Or use dot product (keeps magnitude info)  │
  ├─────────────────────┼────────────────────────────────────────────┤
  │ 2. Curse of dims    │ Reduce dimensions with PCA first          │
  │                     │ Use learned projections for comparison     │
  ├─────────────────────┼────────────────────────────────────────────┤
  │ 3. Hubness          │ CSLS (Cross-domain Similarity Scaling)    │
  │                     │ Inverted softmax normalization             │
  ├─────────────────────┼────────────────────────────────────────────┤
  │ 4. Anisotropy       │ Subtract mean embedding before comparing  │
  │                     │ "Whitening" transforms (ZCA whitening)    │
  ├─────────────────────┼────────────────────────────────────────────┤
  │ 5. No transitivity  │ Don't chain similarity! A≈B, B≈C ≠ A≈C  │
  │                     │ Use graph-based methods for multi-hop      │
  ├─────────────────────┼────────────────────────────────────────────┤
  │ 6. Noise dimensions │ PCA / SVD to keep top components          │
  │                     │ Feature selection before comparing         │
  ├─────────────────────┼────────────────────────────────────────────┤
  │ 7. Zero vectors     │ Add epsilon (PyTorch default: 1e-8)       │
  │                     │ Filter out zero/padding embeddings first  │
  └─────────────────────┴────────────────────────────────────────────┘

  BOTTOM LINE:
  Cosine similarity is a useful DEFAULT, not a perfect solution.
  Modern systems (RAG, search, recommendations) combine it with:
    - Learned similarity functions (trained to match human judgment)
    - Re-ranking models (cross-encoders that compare pairs directly)
    - Hybrid retrieval (cosine + keyword matching like BM25)
""")
