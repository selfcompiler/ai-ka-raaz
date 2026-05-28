"""
🎬 AI Ka Raaz — Episode 1: "AI Sochta Kaise Hai?"
A tiny AI that learns Bollywood dialogues — in 5 seconds!

Run: python ep01_ai_thinks.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time, os

# ============================================================
# STEP 1: Load Bollywood Dialogues
# ============================================================
print("=" * 60)
print("🤖 AI KA RAAZ — Episode 1")
print("=" * 60)

input("\n👉 Step 1: Bollywood dialogues load karte hain... [ENTER]")

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bollywood.txt')
if not os.path.exists(data_path):
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bollywood.txt')
text = open(data_path).read()

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join(itos[i] for i in l)

data = torch.tensor(encode(text), dtype=torch.long)

print(f"\n🎬 Bollywood dialogues loaded: {len(text):,} characters")
print(f"🎭 From 1,760 movies!")
print(f"📝 Vocabulary: {vocab_size} unique characters")

# ============================================================
# STEP 2: Build the AI
# ============================================================
input("\n👉 Step 2: AI model banate hain... [ENTER]")

torch.manual_seed(42)

class TinyAI(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x):
        return self.table(x)

model = TinyAI(vocab_size)
n_params = sum(p.numel() for p in model.parameters())

print(f"\n🧠 TinyAI ready!")
print(f"   Brain = ek table of {vocab_size} x {vocab_size} = {n_params:,} numbers")
print(f"   Abhi sab numbers RANDOM hain — AI ko kuch nahi aata!")

# ============================================================
# STEP 3: Generate BEFORE training (gibberish)
# ============================================================
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

input("\n👉 Step 3: Dekhte hain BINA training ke AI kya likhta hai... [ENTER]")

print("\n" + "─" * 60)
print("❌ BEFORE TRAINING:")
print("─" * 60)
print(generate(200))
print("\n😂 Gibberish! Random letters! AI ko kuch nahi aata!")

# ============================================================
# STEP 4: Train the AI
# ============================================================
input("\n👉 Step 4: Ab TRAIN karte hain... [ENTER]")

print("\n" + "─" * 60)
print("🏋️ TRAINING SHURU...")
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
print(f"\n✅ Training complete! ({elapsed:.1f} seconds)")

# ============================================================
# STEP 5: Generate AFTER training
# ============================================================
input("\n👉 Step 5: Ab dekhte hain training ke BAAD... [ENTER]")

print("\n" + "─" * 60)
print("✅ AFTER TRAINING (AI ne Bollywood SEEKH liya!):")
print("─" * 60)
print(generate(300))

# ============================================================
# STEP 6: What did AI learn?
# ============================================================
input("\n👉 Step 6: AI ne kya seekha? Dekhte hain... [ENTER]")

print("\n" + "=" * 60)
print("🧠 KYA SEEKHA AI NE?")
print("=" * 60)
print()
print("AI ne sirf EK cheez seekhi: ek character ke baad")
print("kaunsa character aata hai. That's it!")
print()

top_k = 5
with torch.no_grad():
    for ch in ['M', 'a', ' ']:
        if ch not in stoi:
            continue
        idx = stoi[ch]
        logits = model(torch.tensor([idx]))
        probs = F.softmax(logits[0], dim=0)
        top = torch.topk(probs, top_k)
        predictions = [f"'{itos[i.item()]}' ({p.item():.0%})" for p, i in zip(top.values, top.indices)]
        display = ch if ch != ' ' else '␣'
        print(f"  '{display}' ke baad → {', '.join(predictions)}")
