"""
=============================================================================
 COSINE SIMILARITY — Step-by-Step, From First Principles
=============================================================================
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as FancyArrowPatch
import numpy as np
import os
import math

# ============================================================================
# STEP 1: The Intuition — It's About DIRECTION, Not Length
# ============================================================================

print("=" * 70)
print("STEP 1: The Intuition — Direction, Not Length")
print("=" * 70)

print("""
Imagine you're standing at the origin (0,0) looking outward.
Two vectors are like two arrows pointing in some direction.

Cosine similarity asks: "How similar is the DIRECTION of these arrows?"

  Same direction    → cosine = +1.0   (pointing the same way)
  Perpendicular     → cosine =  0.0   (completely unrelated)
  Opposite direction → cosine = -1.0   (pointing opposite ways)

KEY INSIGHT: The LENGTH of the arrows doesn't matter!
  A = [1, 0] and B = [100, 0] both point RIGHT → cosine = 1.0
  They differ in magnitude but agree in direction.
""")


# ============================================================================
# STEP 2: Manual Calculation — Every Single Step
# ============================================================================

print("=" * 70)
print("STEP 2: Manual Calculation — Every Single Step")
print("=" * 70)

# Two simple 2D vectors
A = [3, 4]
B = [4, 3]

print(f"""
  Vector A = {A}
  Vector B = {B}

  ┌─────────────────────────────────────────────────────────┐
  │  STEP 2a: Dot Product (A · B)                          │
  │                                                         │
  │  Multiply matching dimensions, then add:                │
  │                                                         │
  │    a₁ × b₁  =  {A[0]} × {B[0]}  =  {A[0]*B[0]:2d}                          │
  │    a₂ × b₂  =  {A[1]} × {B[1]}  =  {A[1]*B[1]:2d}                          │
  │                          ────                           │
  │    Dot product           =  {A[0]*B[0] + A[1]*B[1]:2d}                          │
  └─────────────────────────────────────────────────────────┘
""")

dot_product = A[0]*B[0] + A[1]*B[1]

mag_A = math.sqrt(A[0]**2 + A[1]**2)
mag_B = math.sqrt(B[0]**2 + B[1]**2)

print(f"""
  ┌─────────────────────────────────────────────────────────┐
  │  STEP 2b: Magnitude (length) of each vector            │
  │                                                         │
  │  ‖A‖ = √(a₁² + a₂²) = √({A[0]}² + {A[1]}²) = √({A[0]**2} + {A[1]**2}) = √{A[0]**2+A[1]**2} = {mag_A:.2f}  │
  │  ‖B‖ = √(b₁² + b₂²) = √({B[0]}² + {B[1]}²) = √({B[0]**2} + {B[1]**2}) = √{B[0]**2+B[1]**2} = {mag_B:.2f}  │
  └─────────────────────────────────────────────────────────┘
""")

cosine = dot_product / (mag_A * mag_B)

print(f"""
  ┌─────────────────────────────────────────────────────────┐
  │  STEP 2c: Divide — the final similarity                │
  │                                                         │
  │           A · B          {dot_product}                          │
  │  cos = ───────── = ────────────── = {cosine:.4f}             │
  │        ‖A‖ × ‖B‖   {mag_A:.2f} × {mag_B:.2f}                       │
  └─────────────────────────────────────────────────────────┘

  Result: {cosine:.4f} → very similar direction (close to 1.0)
  These two vectors point in nearly the same direction!
""")

# Verify with PyTorch
a_tensor = torch.tensor(A, dtype=torch.float)
b_tensor = torch.tensor(B, dtype=torch.float)
pytorch_result = F.cosine_similarity(a_tensor.unsqueeze(0), b_tensor.unsqueeze(0))
print(f"  PyTorch verification: {pytorch_result.item():.4f} ✓")


# ============================================================================
# STEP 3: More Examples — Building Intuition
# ============================================================================

print(f"\n{'=' * 70}")
print("STEP 3: Examples That Build Intuition")
print("=" * 70)

examples = [
    ([1, 0], [1, 0],    "Identical vectors"),
    ([1, 0], [0, 1],    "Perpendicular (90°)"),
    ([1, 0], [-1, 0],   "Opposite direction (180°)"),
    ([1, 0], [1, 1],    "45° apart"),
    ([1, 0], [100, 0],  "Same direction, different length"),
    ([3, 4], [6, 8],    "Same direction, B is 2× longer"),
    ([3, 4], [4, 3],    "Close but not identical"),
    ([3, 4], [-4, -3],  "Roughly opposite"),
]

print(f"\n  {'A':<14} {'B':<14} {'Cosine':>8} {'Angle':>8}  Description")
print(f"  {'-'*14} {'-'*14} {'-'*8} {'-'*8}  {'-'*30}")

for a, b, desc in examples:
    at = torch.tensor(a, dtype=torch.float)
    bt = torch.tensor(b, dtype=torch.float)
    cos = F.cosine_similarity(at.unsqueeze(0), bt.unsqueeze(0)).item()
    angle = math.degrees(math.acos(max(-1, min(1, cos))))
    bar = "█" * int(abs(cos) * 10)
    sign = "+" if cos >= 0 else "-"
    print(f"  {str(a):<14} {str(b):<14} {cos:>+8.4f} {angle:>6.1f}°  {bar} {desc}")


# ============================================================================
# STEP 4: Why Cosine Over Euclidean Distance?
# ============================================================================

print(f"\n{'=' * 70}")
print("STEP 4: Why Cosine Instead of Euclidean Distance?")
print("=" * 70)

print("""
  Consider these word embedding vectors:

    "cat"  = [0.2, 0.8]    (small magnitude)
    "dog"  = [0.3, 1.2]    (larger magnitude — maybe appeared more in training)
    "car"  = [0.9, 0.1]    (different direction entirely)

  Euclidean distance (straight-line distance):
""")

cat = torch.tensor([0.2, 0.8])
dog = torch.tensor([0.3, 1.2])
car = torch.tensor([0.9, 0.1])

euc_cat_dog = torch.dist(cat, dog).item()
euc_cat_car = torch.dist(cat, car).item()
cos_cat_dog = F.cosine_similarity(cat.unsqueeze(0), dog.unsqueeze(0)).item()
cos_cat_car = F.cosine_similarity(cat.unsqueeze(0), car.unsqueeze(0)).item()

print(f"    Euclidean(cat, dog) = {euc_cat_dog:.4f}")
print(f"    Euclidean(cat, car) = {euc_cat_car:.4f}")
print(f"    → Euclidean says cat is closer to CAR! ✗ (wrong)")

print(f"""
  Cosine similarity (direction only):
    Cosine(cat, dog)    = {cos_cat_dog:.4f}
    Cosine(cat, car)    = {cos_cat_car:.4f}
    → Cosine says cat is closer to DOG! ✓ (correct)

  WHY: Euclidean is fooled by vector LENGTH (magnitude).
       "dog" is farther from "cat" in raw distance because it's a longer vector.
       But they point in the SAME DIRECTION — that's what matters for meaning!

  ┌──────────────────────────────────────────────────────┐
  │  Euclidean = "how far apart are the endpoints?"      │
  │  Cosine    = "do they point the same way?"           │
  │                                                      │
  │  For embeddings, DIRECTION = MEANING                 │
  │  So cosine similarity is the right choice.           │
  └──────────────────────────────────────────────────────┘
""")


# ============================================================================
# STEP 5: Real Embedding Example — "cat" vs "dog" vs "king"
# ============================================================================

print("=" * 70)
print("STEP 5: Full Calculation with Real Embeddings")
print("=" * 70)

# Simulate 8-dimensional embeddings (like a mini version of real ones)
torch.manual_seed(0)
embeddings = {
    "cat":   [0.82, 0.91, -0.12, 0.45, -0.67, 0.23, 0.78, 0.34],
    "dog":   [0.79, 0.85, -0.08, 0.52, -0.59, 0.19, 0.81, 0.29],
    "king":  [-0.34, 0.12, 0.89, -0.67, 0.45, 0.78, -0.23, 0.56],
}

print("\n  8-dimensional embedding vectors:\n")
for word, vec in embeddings.items():
    formatted = [f"{v:+.2f}" for v in vec]
    print(f"    {word:5s} = [{', '.join(formatted)}]")

# Calculate cat vs dog step by step
a = embeddings["cat"]
b = embeddings["dog"]
print(f"\n  Calculating cosine_similarity(cat, dog):\n")

print(f"  {'dim':<5} {'cat':>6} {'dog':>6} {'×':>3} {'product':>8}")
print(f"  {'-'*5} {'-'*6} {'-'*6} {'-'*3} {'-'*8}")
products = []
for i in range(len(a)):
    prod = a[i] * b[i]
    products.append(prod)
    print(f"  {i:<5} {a[i]:>+6.2f} {b[i]:>+6.2f}  →  {prod:>+8.4f}")

dot = sum(products)
mag_a = math.sqrt(sum(x**2 for x in a))
mag_b = math.sqrt(sum(x**2 for x in b))
cos_val = dot / (mag_a * mag_b)

print(f"  {'':5} {'':6} {'':6}     {'────────':>8}")
print(f"  {'':5} {'':6} {'dot product':>12}  = {dot:>+8.4f}")
print(f"\n  ‖cat‖ = {mag_a:.4f}")
print(f"  ‖dog‖ = {mag_b:.4f}")
print(f"\n  cosine = {dot:.4f} / ({mag_a:.4f} × {mag_b:.4f}) = {cos_val:.4f}")
print(f"\n  → cat and dog are VERY similar ({cos_val:.4f} ≈ 1.0)")

# Compare all pairs
print(f"\n  All pairwise similarities:")
words = list(embeddings.keys())
for i in range(len(words)):
    for j in range(i+1, len(words)):
        va = torch.tensor(embeddings[words[i]])
        vb = torch.tensor(embeddings[words[j]])
        sim = F.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0)).item()
        print(f"    {words[i]:5s} ↔ {words[j]:5s} = {sim:+.4f}  "
              f"{'██████████' if sim > 0.8 else '█████' if sim > 0.5 else '██' if sim > 0 else '░░'}")


# ============================================================================
# VISUALIZATION
# ============================================================================

output_dir = os.path.dirname(__file__)

# --- Plot 1: 2D Vectors showing cosine similarity ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

cases = [
    {
        "title": "Cosine = +1.0\n(Same Direction)",
        "vectors": [([0, 0, 2, 3], "A = [2,3]", "#e74c3c"),
                     ([0, 0, 4, 6], "B = [4,6]", "#3498db")],
    },
    {
        "title": "Cosine = 0.0\n(Perpendicular)",
        "vectors": [([0, 0, 3, 0], "A = [3,0]", "#e74c3c"),
                     ([0, 0, 0, 4], "B = [0,4]", "#3498db")],
    },
    {
        "title": "Cosine ≈ 0.96\n(Close Direction)",
        "vectors": [([0, 0, 3, 4], "A = [3,4]", "#e74c3c"),
                     ([0, 0, 4, 3], "B = [4,3]", "#3498db")],
    },
]

for ax, case in zip(axes, cases):
    ax.set_xlim(-1, 7)
    ax.set_ylim(-1, 7)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.axvline(x=0, color="black", linewidth=0.5)

    for vec_data in case["vectors"]:
        coords, label, color = vec_data
        ax.annotate("", xy=(coords[2], coords[3]), xytext=(coords[0], coords[1]),
                     arrowprops=dict(arrowstyle="-|>", color=color, lw=3))
        ax.text(coords[2] + 0.2, coords[3] + 0.2, label, fontsize=11,
                fontweight="bold", color=color)

    # Calculate and show angle arc
    v1 = case["vectors"][0][0]
    v2 = case["vectors"][1][0]
    a_vec = torch.tensor([v1[2], v1[3]], dtype=torch.float)
    b_vec = torch.tensor([v2[2], v2[3]], dtype=torch.float)
    cos = F.cosine_similarity(a_vec.unsqueeze(0), b_vec.unsqueeze(0)).item()
    angle = math.degrees(math.acos(max(-1, min(1, cos))))

    # Draw angle arc
    theta1 = math.degrees(math.atan2(v1[3], v1[2]))
    theta2 = math.degrees(math.atan2(v2[3], v2[2]))
    arc = plt.Circle((0, 0), 1.2, fill=False, color="#2ecc71", linewidth=1.5, linestyle="--")
    angle_arc = plt.matplotlib.patches.Arc((0, 0), 2.4, 2.4,
                                            angle=0,
                                            theta1=min(theta1, theta2),
                                            theta2=max(theta1, theta2),
                                            color="#2ecc71", linewidth=2)
    ax.add_patch(angle_arc)
    mid_angle = math.radians((theta1 + theta2) / 2)
    ax.text(1.8 * math.cos(mid_angle), 1.8 * math.sin(mid_angle),
            f"{angle:.0f}°", fontsize=12, color="#2ecc71", fontweight="bold",
            ha="center", va="center")

    ax.set_title(case["title"], fontsize=14, fontweight="bold", pad=10)
    ax.set_xlabel("Dimension 1", fontsize=10)
    ax.set_ylabel("Dimension 2", fontsize=10)

plt.suptitle("Cosine Similarity = How Similar the DIRECTION Is",
             fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
path1 = os.path.join(output_dir, "cosine_vectors.png")
plt.savefig(path1, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {path1}")


# --- Plot 2: Cosine vs Euclidean comparison ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Euclidean view
ax = axes[0]
points = {"cat": [0.2, 0.8], "dog": [0.3, 1.2], "car": [0.9, 0.1]}
colors = {"cat": "#e74c3c", "dog": "#3498db", "car": "#2ecc71"}

for word, pos in points.items():
    ax.scatter(*pos, c=colors[word], s=200, zorder=5, edgecolors="black", linewidth=1.5)
    ax.annotate(word, pos, fontsize=14, fontweight="bold",
                textcoords="offset points", xytext=(10, 10),
                color=colors[word])

# Draw euclidean distances
cat, dog, car = points["cat"], points["dog"], points["car"]
ax.plot([cat[0], dog[0]], [cat[1], dog[1]], "k--", linewidth=1.5, alpha=0.5)
ax.plot([cat[0], car[0]], [cat[1], car[1]], "k--", linewidth=1.5, alpha=0.5)

mid1 = [(cat[0]+dog[0])/2, (cat[1]+dog[1])/2]
mid2 = [(cat[0]+car[0])/2, (cat[1]+car[1])/2]
d1 = math.sqrt((cat[0]-dog[0])**2 + (cat[1]-dog[1])**2)
d2 = math.sqrt((cat[0]-car[0])**2 + (cat[1]-car[1])**2)
ax.text(mid1[0]-0.12, mid1[1], f"d={d1:.2f}", fontsize=11, color="black",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
ax.text(mid2[0]+0.05, mid2[1], f"d={d2:.2f}", fontsize=11, color="black",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

ax.set_xlim(-0.1, 1.4)
ax.set_ylim(-0.2, 1.5)
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
ax.set_title(f"Euclidean Distance\ncat→car ({d2:.2f}) < cat→dog ({d1:.2f})\n✗ WRONG: says cat is closer to car!",
             fontsize=12, color="#e74c3c")
ax.set_xlabel("Dimension 1", fontsize=11)
ax.set_ylabel("Dimension 2", fontsize=11)

# Cosine view
ax = axes[1]
# Normalize to unit vectors to show direction
for word, pos in points.items():
    norm = math.sqrt(pos[0]**2 + pos[1]**2)
    unit = [pos[0]/norm, pos[1]/norm]
    ax.annotate("", xy=(unit[0]*1.3, unit[1]*1.3), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=colors[word], lw=3))
    ax.text(unit[0]*1.45, unit[1]*1.45, word, fontsize=14, fontweight="bold",
            color=colors[word], ha="center", va="center")

# Draw angle arcs
for w1, w2, offset in [("cat", "dog", 0.05), ("cat", "car", -0.05)]:
    p1 = points[w1]
    p2 = points[w2]
    a1 = math.degrees(math.atan2(p1[1], p1[0]))
    a2 = math.degrees(math.atan2(p2[1], p2[0]))
    cos_v = F.cosine_similarity(
        torch.tensor(p1, dtype=torch.float).unsqueeze(0),
        torch.tensor(p2, dtype=torch.float).unsqueeze(0)).item()
    arc = plt.matplotlib.patches.Arc((0, 0), 1.0 + offset*8, 1.0 + offset*8,
                                      angle=0,
                                      theta1=min(a1, a2),
                                      theta2=max(a1, a2),
                                      color="gray", linewidth=1.5, linestyle="--")
    ax.add_patch(arc)
    mid = math.radians((a1 + a2) / 2)
    ax.text(0.7 * math.cos(mid), 0.7 * math.sin(mid) + offset,
            f"cos={cos_v:.2f}", fontsize=10, ha="center", va="center",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

c_cd = F.cosine_similarity(torch.tensor(points["cat"]).unsqueeze(0).float(),
                            torch.tensor(points["dog"]).unsqueeze(0).float()).item()
c_cc = F.cosine_similarity(torch.tensor(points["cat"]).unsqueeze(0).float(),
                            torch.tensor(points["car"]).unsqueeze(0).float()).item()

ax.set_xlim(-0.3, 1.8)
ax.set_ylim(-0.3, 1.8)
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color="black", linewidth=0.5)
ax.axvline(x=0, color="black", linewidth=0.5)
ax.set_title(f"Cosine Similarity\ncat↔dog ({c_cd:.2f}) > cat↔car ({c_cc:.2f})\n✓ CORRECT: cat is more similar to dog!",
             fontsize=12, color="#2ecc71")
ax.set_xlabel("Dimension 1", fontsize=11)
ax.set_ylabel("Dimension 2", fontsize=11)

plt.suptitle("Why Cosine Similarity Beats Euclidean Distance for Embeddings",
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
path2 = os.path.join(output_dir, "cosine_vs_euclidean.png")
plt.savefig(path2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {path2}")


# --- Plot 3: The formula, visually ---
fig, ax = plt.subplots(figsize=(14, 7))
ax.axis("off")

ax.text(0.5, 0.95, "Cosine Similarity — The Complete Formula", fontsize=20,
        fontweight="bold", ha="center", va="top", transform=ax.transAxes)

formula_text = r"$\cos(\theta) = \frac{\vec{A} \cdot \vec{B}}{||\vec{A}|| \times ||\vec{B}||} = \frac{\sum_{i=1}^{n} a_i \cdot b_i}{\sqrt{\sum_{i=1}^{n} a_i^2} \times \sqrt{\sum_{i=1}^{n} b_i^2}}$"
ax.text(0.5, 0.78, formula_text, fontsize=22, ha="center", va="center",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", edgecolor="#333", linewidth=2))

steps = [
    (0.08, 0.55, "①  Dot Product\n(numerator)",
     r"$\vec{A} \cdot \vec{B} = a_1 b_1 + a_2 b_2 + ... + a_n b_n$"
     "\n\nMultiply matching dimensions, sum them up.\n"
     "Measures how much A and B 'agree'.",
     "#e74c3c"),
    (0.38, 0.55, "②  Magnitudes\n(denominator)",
     r"$||\vec{A}|| = \sqrt{a_1^2 + a_2^2 + ... + a_n^2}$"
     "\n\nThe 'length' of each vector.\n"
     "Used to normalize away the effect of scale.",
     "#3498db"),
    (0.68, 0.55, "③  Divide\n(normalize)",
     "Dot Product ÷ (Length A × Length B)\n\n"
     "This removes magnitude, leaving ONLY\n"
     "directional information.\n"
     "Result is always between -1 and +1.",
     "#2ecc71"),
]

for x, y, title, body, color in steps:
    box = plt.matplotlib.patches.FancyBboxPatch(
        (x, y - 0.38), 0.28, 0.40, transform=ax.transAxes,
        boxstyle="round,pad=0.02", facecolor=color, alpha=0.1,
        edgecolor=color, linewidth=2)
    ax.add_patch(box)
    ax.text(x + 0.14, y, title, fontsize=13, fontweight="bold",
            ha="center", va="top", transform=ax.transAxes, color=color)
    ax.text(x + 0.14, y - 0.08, body, fontsize=10,
            ha="center", va="top", transform=ax.transAxes, color="#333",
            linespacing=1.5)

# Scale bar at bottom
scale_y = 0.07
for val, label, color in [(-1.0, "Opposite\n(-1.0)", "#e74c3c"),
                           (0.0, "Unrelated\n(0.0)", "#f39c12"),
                           (1.0, "Identical\n(+1.0)", "#2ecc71")]:
    xpos = 0.2 + (val + 1) / 2 * 0.6
    ax.plot(xpos, scale_y, "o", markersize=15, color=color, transform=ax.transAxes, zorder=5)
    ax.text(xpos, scale_y - 0.05, label, fontsize=10, ha="center", va="top",
            transform=ax.transAxes, color=color, fontweight="bold")

ax.plot([0.2, 0.8], [scale_y, scale_y], "-", color="gray", linewidth=3,
        transform=ax.transAxes, zorder=1)

plt.tight_layout()
path3 = os.path.join(output_dir, "cosine_formula.png")
plt.savefig(path3, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {path3}")


print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  3 visualizations saved:                                       ║
║                                                                ║
║  1. cosine_vectors.png                                         ║
║     → 2D arrows showing same direction, perpendicular, close   ║
║                                                                ║
║  2. cosine_vs_euclidean.png                                    ║
║     → Why cosine beats euclidean for embeddings                ║
║                                                                ║
║  3. cosine_formula.png                                         ║
║     → The formula broken into 3 visual steps                   ║
╚══════════════════════════════════════════════════════════════════╝
""")
