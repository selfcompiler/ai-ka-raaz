"""
🎬 AI Ka Raaz — Episode 1: "AI Sochta Kaise Hai?"
A tiny AI that learns to write like Shakespeare — in 5 seconds!

Run: python ep01_ai_thinks.py
"""
import torch
import torch.nn.functional as F
import time, os

# --- STEP 1: Load Shakespeare ---
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shakespeare.txt')
if not os.path.exists(data_path):
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shakespeare.txt')
text = open(data_path).read()

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join(itos[i] for i in l)

data = torch.tensor(encode(text), dtype=torch.long)

# --- STEP 2: The World's Simplest AI (Bigram Model) ---
torch.manual_seed(42)

import torch.nn as nn

class TinyAI(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x):
        return self.table(x)

model = TinyAI(vocab_size)

def generate(num_chars=200):
    idx = torch.zeros(1, dtype=torch.long)
    result = []
    with torch.no_grad():
        for _ in range(num_chars):
            logits = model(idx[-1:])
            probs = F.softmax(logits[0], dim=0)
            next_idx = torch.multinomial(probs, 1)
            result.append(next_idx.item())
            idx = torch.cat([idx, next_idx])
    return decode(result)

# --- STEP 3: Show BEFORE training ---
print("=" * 60)
print("🤖 AI KA RAAZ — Episode 1")
print("=" * 60)
print()
print("📖 Shakespeare loaded:", f"{len(text):,} characters")
print(f"📝 Vocabulary: {vocab_size} unique characters")
print()

print("─" * 60)
print("❌ BEFORE TRAINING (AI ko kuch nahi aata!):")
print("─" * 60)
print(generate(200))
print()

# --- STEP 4: Train! ---
print("─" * 60)
print("🏋️ TRAINING SHURU... (sirf 5 seconds!)")
print("─" * 60)

batch_size = 256
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
num_steps = 10000
start = time.time()

for step in range(1, num_steps + 1):
    ix = torch.randint(len(data) - 1, (batch_size,))
    x = data[ix]
    y = data[ix + 1]

    logits = model(x)
    loss = F.cross_entropy(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 2000 == 0:
        elapsed = time.time() - start
        filled = step * 10 // num_steps
        bar = "█" * filled + "░" * (10 - filled)
        print(f"  Step {step:>5} | Loss: {loss.item():.2f} | [{bar}] {elapsed:.1f}s")

elapsed = time.time() - start
n_params = sum(p.numel() for p in model.parameters())
print(f"\n✅ Training complete! ({elapsed:.1f} seconds)")
print(f"   Parameters: {n_params:,} (that's it!)")

# --- STEP 5: Show AFTER training ---
print()
print("─" * 60)
print("✅ AFTER TRAINING (AI ne Shakespeare SEEKH liya!):")
print("─" * 60)
print(generate(300))
print()

# --- STEP 6: The Big Reveal ---
print("=" * 60)
print("🧠 KYA SEEKHA AI NE?")
print("=" * 60)
print()
print("AI ne sirf EK cheez seekhi: ek character ke baad")
print("kaunsa character aata hai. That's it!")
print()

top_k = 5
with torch.no_grad():
    for ch in ['T', 'h', ' ']:
        idx = stoi[ch]
        logits = model(torch.tensor([idx]))
        probs = F.softmax(logits[0], dim=0)
        top = torch.topk(probs, top_k)
        predictions = [f"'{itos[i.item()]}' ({p.item():.0%})" for p, i in zip(top.values, top.indices)]
        display = ch if ch != ' ' else '␣'
        print(f"  '{display}' ke baad → {', '.join(predictions)}")

print()
print("─" * 60)
print("💡 KEY INSIGHT:")
print("   ChatGPT BHI yahi karta hai — NEXT token predict karo.")
print("   Bas scale alag hai:")
print(f"   • Humara model:  {vocab_size * vocab_size:,} parameters")
print(f"   • GPT-4:         1,800,000,000,000 parameters")
print("   Same concept. Fark sirf SIZE ka hai! 🚀")
print("─" * 60)
