"""
Step-by-Step Visual: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

Generates a series of annotated heatmaps showing exactly what happens
at every stage of scaled dot-product attention on "I love cats".
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
import math
import os

output_dir = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Setup: same tiny example from your existing script
# ---------------------------------------------------------------------------
sentence = ["I", "love", "cats"]
seq_len = 3
embed_dim = 4
d_k = 3

X = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],   # "I"
    [0.0, 1.0, 0.0, 1.0],   # "love"
    [1.0, 1.0, 0.0, 0.0],   # "cats"
], dtype=torch.float)

W_q = torch.tensor([[1,0,1],[0,1,0],[1,0,0],[0,1,1]], dtype=torch.float)
W_k = torch.tensor([[0,1,0],[1,0,1],[0,1,1],[1,0,0]], dtype=torch.float)
W_v = torch.tensor([[1,0,0],[0,1,0],[0,0,1],[1,1,0]], dtype=torch.float)

Q = X @ W_q
K = X @ W_k
V = X @ W_v
scores = Q @ K.T
scale = math.sqrt(d_k)
scaled_scores = scores / scale
weights = F.softmax(scaled_scores, dim=-1)
output = weights @ V

# ---------------------------------------------------------------------------
# Helper: draw a matrix heatmap with values annotated inside cells
# ---------------------------------------------------------------------------
def draw_matrix(ax, data, row_labels, col_labels, title, cmap, fmt=".1f",
                vmin=None, vmax=None, highlight_max_row=False, fontsize=13):
    if isinstance(data, torch.Tensor):
        data = data.detach().numpy()
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=11)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)

    for i in range(data.shape[0]):
        row_max = data[i].argmax() if highlight_max_row else -1
        for j in range(data.shape[1]):
            val = data[i, j]
            color = "white" if val > (data.max() + data.min()) / 2 else "black"
            weight = "bold" if j == row_max else "normal"
            ax.text(j, i, f"{val:{fmt}}", ha="center", va="center",
                    fontsize=fontsize, color=color, fontweight=weight)
    return im


# ---------------------------------------------------------------------------
# Figure 1: The full pipeline — 8 panels
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(28, 20))
fig.patch.set_facecolor("#fafafa")

gs = gridspec.GridSpec(4, 4, hspace=0.55, wspace=0.4,
                       height_ratios=[1, 1, 1, 0.6])

dim_labels = [f"d{i}" for i in range(d_k)]
embed_labels = [f"d{i}" for i in range(embed_dim)]

# ---- Step 1: Input Embeddings X ----
ax1 = fig.add_subplot(gs[0, 0])
draw_matrix(ax1, X, sentence, embed_labels,
            "Step 1: Input X\n(token embeddings)", "coolwarm",
            vmin=-1.5, vmax=1.5)
ax1.set_xlabel("embedding dim (4)", fontsize=10)

# ---- Step 2: Weight matrices side by side ----
ax_wq = fig.add_subplot(gs[0, 1])
draw_matrix(ax_wq, W_q, embed_labels, dim_labels,
            "Step 2a: W_Q\n(query weights)", "Reds", fmt=".0f")

ax_wk = fig.add_subplot(gs[0, 2])
draw_matrix(ax_wk, W_k, embed_labels, dim_labels,
            "Step 2b: W_K\n(key weights)", "Greens", fmt=".0f")

ax_wv = fig.add_subplot(gs[0, 3])
draw_matrix(ax_wv, W_v, embed_labels, dim_labels,
            "Step 2c: W_V\n(value weights)", "Oranges", fmt=".0f")

# ---- Step 3: Q, K, V ----
ax_q = fig.add_subplot(gs[1, 0])
draw_matrix(ax_q, Q, sentence, dim_labels,
            "Step 3a: Q = X * W_Q\n\"What am I looking for?\"", "Reds")

ax_k = fig.add_subplot(gs[1, 1])
draw_matrix(ax_k, K, sentence, dim_labels,
            "Step 3b: K = X * W_K\n\"What do I contain?\"", "Greens")

ax_v = fig.add_subplot(gs[1, 2])
draw_matrix(ax_v, V, sentence, dim_labels,
            "Step 3c: V = X * W_V\n\"What info do I carry?\"", "Oranges")

# ---- Step 4: QK^T raw scores ----
ax_scores = fig.add_subplot(gs[1, 3])
draw_matrix(ax_scores, scores, sentence, sentence,
            "Step 4: Q * K^T\n(raw attention scores)", "PuRd",
            highlight_max_row=True)
ax_scores.set_xlabel("Key (attending TO) -->", fontsize=10)
ax_scores.set_ylabel("<-- Query (FROM)", fontsize=10)

# ---- Step 5: Scaled scores ----
ax_scaled = fig.add_subplot(gs[2, 0])
draw_matrix(ax_scaled, scaled_scores, sentence, sentence,
            f"Step 5: / sqrt(d_k)\n(divide by {scale:.2f})",
            "PuRd", fmt=".2f", highlight_max_row=True)
ax_scaled.set_xlabel("Key -->", fontsize=10)

# ---- Step 6: Softmax (attention weights) ----
ax_soft = fig.add_subplot(gs[2, 1])
draw_matrix(ax_soft, weights, sentence, sentence,
            "Step 6: softmax\n(attention weights, rows sum to 1)",
            "Blues", fmt=".3f", vmin=0, vmax=1, highlight_max_row=True)
ax_soft.set_xlabel("Key -->", fontsize=10)

# ---- Step 7: Output = weights @ V ----
ax_out = fig.add_subplot(gs[2, 2])
draw_matrix(ax_out, output, sentence, dim_labels,
            "Step 7: Weights * V\n(context-aware output)",
            "YlOrRd", fmt=".2f")
ax_out.set_xlabel("output dim", fontsize=10)

# ---- Step 8: Before vs After comparison ----
ax_cmp = fig.add_subplot(gs[2, 3])
ax_cmp.axis("off")
ax_cmp.set_title("Before vs After Attention", fontsize=14, fontweight="bold", pad=10)
y_pos = 0.9
for i, word in enumerate(sentence):
    before = [f"{v:.1f}" for v in X[i].tolist()]
    after = [f"{v:.2f}" for v in output[i].tolist()]
    ax_cmp.text(0.5, y_pos, f'"{word}"', fontsize=14, fontweight="bold",
                ha="center", va="center", transform=ax_cmp.transAxes)
    y_pos -= 0.12
    ax_cmp.text(0.5, y_pos,
                f"Before: [{', '.join(before)}]\n After:  [{', '.join(after)}]",
                fontsize=11, ha="center", va="center", transform=ax_cmp.transAxes,
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#eef6ff", edgecolor="#888"))
    y_pos -= 0.22

# ---- Bottom panel: formula flow ----
ax_flow = fig.add_subplot(gs[3, :])
ax_flow.axis("off")

boxes = [
    ("X", "#dfe6e9", "Input\nEmbeddings\n[3x4]"),
    ("Q, K, V", "#ffeaa7", "Projections\nQ=XW_q  K=XW_k  V=XW_v\n[3x3] each"),
    ("QK^T", "#fab1a0", "Dot Product\nScore Matrix\n[3x3]"),
    ("/ sqrt(d_k)", "#fd79a8", f"Scale\ndivide by {scale:.2f}\n[3x3]"),
    ("softmax", "#74b9ff", "Normalize\nrows sum to 1\n[3x3]"),
    ("x V", "#ffeaa7", "Weighted Sum\nblend values\n[3x3]"),
    ("Output", "#55efc4", "Context-aware\nEmbeddings\n[3x3]"),
]

n = len(boxes)
box_w = 0.11
gap = (1.0 - n * box_w) / (n + 1)

for idx, (label, color, desc) in enumerate(boxes):
    cx = gap + idx * (box_w + gap) + box_w / 2
    cy = 0.55

    rect = mpatches.FancyBboxPatch(
        (cx - box_w/2, cy - 0.35), box_w, 0.7,
        boxstyle="round,pad=0.02", facecolor=color, edgecolor="#2d3436",
        linewidth=1.5, transform=ax_flow.transAxes, clip_on=False)
    ax_flow.add_patch(rect)

    ax_flow.text(cx, cy + 0.1, label, fontsize=12, fontweight="bold",
                 ha="center", va="center", transform=ax_flow.transAxes)
    ax_flow.text(cx, cy - 0.12, desc, fontsize=8, ha="center", va="center",
                 transform=ax_flow.transAxes, color="#555")

    if idx < n - 1:
        arrow_x = cx + box_w / 2 + 0.005
        ax_flow.annotate("", xy=(arrow_x + gap - 0.01, cy),
                         xytext=(arrow_x, cy),
                         xycoords="axes fraction", textcoords="axes fraction",
                         arrowprops=dict(arrowstyle="-|>", color="#2d3436", lw=2))

fig.suptitle('Attention Step-by-Step:  "I love cats"',
             fontsize=22, fontweight="bold", y=0.99, color="#2d3436")

path1 = os.path.join(output_dir, "attention_step_by_step.png")
plt.savefig(path1, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: {path1}")


# ---------------------------------------------------------------------------
# Figure 2: Zoomed softmax — showing the "why" of each step
# ---------------------------------------------------------------------------
fig2, axes = plt.subplots(1, 3, figsize=(20, 6))
fig2.patch.set_facecolor("#fafafa")

titles = [
    "Raw Scores (QK^T)\nBefore scaling",
    f"Scaled Scores (/ sqrt({d_k}))\nAfter dividing by {scale:.2f}",
    "Attention Weights (softmax)\nFinal probabilities"
]
data_list = [scores, scaled_scores, weights]
cmaps = ["PuRd", "PuRd", "Blues"]
fmts = [".1f", ".2f", ".3f"]

for idx, (ax, data, title, cmap, fmt) in enumerate(zip(
        axes, data_list, titles, cmaps, fmts)):
    draw_matrix(ax, data, sentence, sentence, title, cmap, fmt=fmt,
                vmin=0 if idx == 2 else None,
                vmax=1 if idx == 2 else None,
                highlight_max_row=True, fontsize=14)
    ax.set_xlabel("Key (attending TO) -->", fontsize=10)
    ax.set_ylabel("<-- Query (FROM)", fontsize=10)

    if idx < 2:
        ax.annotate("", xy=(1.12, 0.5), xytext=(1.02, 0.5),
                     xycoords="axes fraction", textcoords="axes fraction",
                     arrowprops=dict(arrowstyle="-|>", color="#e74c3c", lw=3))
        label = f"/ {scale:.2f}" if idx == 0 else "softmax"
        ax.text(1.07, 0.62, label, transform=ax.transAxes, fontsize=11,
                ha="center", fontweight="bold", color="#e74c3c")

fig2.suptitle("Zoomed In: From Raw Scores to Attention Weights",
              fontsize=18, fontweight="bold", y=1.02)

path2 = os.path.join(output_dir, "attention_softmax_zoom.png")
plt.savefig(path2, dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
plt.close()
print(f"Saved: {path2}")


# ---------------------------------------------------------------------------
# Figure 3: Attention flow per token — who attends to whom?
# ---------------------------------------------------------------------------
fig3, axes3 = plt.subplots(1, 3, figsize=(21, 7))
fig3.patch.set_facecolor("#fafafa")
colors = ["#e74c3c", "#3498db", "#2ecc71"]
w_np = weights.detach().numpy()

for idx, (ax, word) in enumerate(zip(axes3, sentence)):
    attn = w_np[idx]
    bars = ax.barh(sentence[::-1], attn[::-1], color=colors[idx], edgecolor="#333",
                   height=0.5)
    ax.set_xlim(0, 1)
    ax.set_title(f'"{word}" attends to...', fontsize=16, fontweight="bold",
                 color=colors[idx])
    ax.set_xlabel("Attention Weight", fontsize=12)

    for bar, val in zip(bars, attn[::-1]):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f"{val:.1%}", va="center", fontsize=13, fontweight="bold")

    ax.axvline(1/3, color="#999", ls="--", lw=1, label="uniform (1/3)")
    ax.legend(fontsize=10, loc="lower right")
    ax.tick_params(axis="y", labelsize=13)

fig3.suptitle("Where Does Each Token Focus Its Attention?",
              fontsize=18, fontweight="bold", y=1.02)
fig3.tight_layout()

path3 = os.path.join(output_dir, "attention_per_token.png")
plt.savefig(path3, dpi=150, bbox_inches="tight", facecolor=fig3.get_facecolor())
plt.close()
print(f"Saved: {path3}")


# ---------------------------------------------------------------------------
# Figure 4: The weighted sum — how output is built token by token
# ---------------------------------------------------------------------------
fig4, axes4 = plt.subplots(1, 3, figsize=(22, 7))
fig4.patch.set_facecolor("#fafafa")
V_np = V.detach().numpy()
out_np = output.detach().numpy()

for idx, (ax, word) in enumerate(zip(axes4, sentence)):
    attn = w_np[idx]
    bottom = np.zeros(d_k)

    for j in range(seq_len):
        contribution = attn[j] * V_np[j]
        ax.bar(dim_labels, contribution, bottom=bottom, label=f'{attn[j]:.2f} x V_{sentence[j]}',
               color=colors[j], edgecolor="#333", alpha=0.85)
        for d in range(d_k):
            if contribution[d] > 0.05:
                ax.text(d, bottom[d] + contribution[d]/2,
                        f"{contribution[d]:.2f}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color="white")
        bottom += contribution

    for d in range(d_k):
        ax.text(d, bottom[d] + 0.05, f"{out_np[idx, d]:.2f}",
                ha="center", fontsize=12, fontweight="bold", color="#2d3436")

    ax.set_ylim(0, max(bottom) * 1.2)
    ax.set_title(f'Output for "{word}"\n= weighted sum of V', fontsize=14, fontweight="bold")
    ax.set_xlabel("Output dimension", fontsize=11)
    ax.set_ylabel("Value", fontsize=11)
    ax.legend(fontsize=10, loc="upper right")

fig4.suptitle("Step 7 Breakdown: How Each Output Vector is Built from V",
              fontsize=18, fontweight="bold", y=1.02)
fig4.tight_layout()

path4 = os.path.join(output_dir, "attention_weighted_sum.png")
plt.savefig(path4, dpi=150, bbox_inches="tight", facecolor=fig4.get_facecolor())
plt.close()
print(f"Saved: {path4}")

print("\nAll 4 figures saved!")
print(f"  1. {path1}  — full pipeline (8 panels)")
print(f"  2. {path2}  — softmax zoom (3 panels)")
print(f"  3. {path3}  — per-token attention bars")
print(f"  4. {path4}  — weighted sum breakdown")
