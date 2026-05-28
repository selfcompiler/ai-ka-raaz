"""
🎬 AI Ka Raaz — Episode 2: "Computer Ko Hindi Kaise Sikhaye?"
See how AI converts Bollywood dialogues to numbers and back!

Run: python ep02_tokenization.py
"""
import os

# ============================================================
# STEP 1: Load Bollywood Dialogues
# ============================================================
print("=" * 60)
print("🔢 AI KA RAAZ — Episode 2: Tokenization")
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

print(f"\n🎬 Bollywood dialogues loaded!")
print(f"📝 Vocabulary: {vocab_size} unique characters")
print(f"   Characters: {''.join(chars[:30])}...")

# ============================================================
# STEP 2: The Secret Code — Encode
# ============================================================
input("\n👉 Step 2: Text ko numbers mein convert karte hain... [ENTER]")

print("\n" + "─" * 60)
print("🔐 ENCODING: Text → Numbers")
print("─" * 60)

demos = [
    "Mere paas maa hai",
    "Mogambo khush hua",
    "Rishte mein toh hum tumhare baap lagte hain",
]

for sentence in demos:
    tokens = [stoi[c] for c in sentence if c in stoi]
    print(f"\n  \"{sentence}\"")
    print(f"  → {tokens}")
    mapping = [f"{c}={stoi[c]}" for c in sentence[:10] if c in stoi]
    print(f"     ({', '.join(mapping)}...)")

# ============================================================
# STEP 3: Decode — Numbers back to Text
# ============================================================
input("\n👉 Step 3: Ab numbers se wapas text banate hain... [ENTER]")

print("\n" + "─" * 60)
print("🔓 DECODING: Numbers → Text")
print("─" * 60)

secret_text = "Kitne aadmi the"
secret = [stoi[c] for c in secret_text if c in stoi]
decoded = ''.join(itos[i] for i in secret if i in itos)
print(f"\n  Secret message: {secret}")
print(f"  Decoded:        \"{decoded}\"")
print(f"\n  ✅ Numbers → Text → Numbers. Perfectly reversible!")

# ============================================================
# STEP 4: The Problem
# ============================================================
input("\n👉 Step 4: Lekin ek PROBLEM hai... [ENTER]")

print("\n" + "─" * 60)
print("⚠️ PROBLEM: Character Tokens Ka Masla")
print("─" * 60)

examples = [
    "Namaste",
    "Mogambo khush hua",
    "Rishte mein toh hum tumhare baap lagte hain",
]

print("\n  Character-level tokenization:")
for sentence in examples:
    count = len(sentence)
    bar = "█" * min(count, 50)
    print(f"  \"{sentence}\"")
    print(f"   → {count} tokens {bar}")

print(f"\n  😱 ChatGPT limit: ~4000 tokens")
print(f"     Character level pe ek chhota paragraph = limit khatam!")

# ============================================================
# STEP 5: BPE — The Solution
# ============================================================
input("\n👉 Step 5: Solution dekho — BPE! [ENTER]")

print("\n" + "─" * 60)
print("🧠 BPE (Byte Pair Encoding) — AI Ka Real Trick!")
print("─" * 60)

sample = "mera naam hai mera kaam hai mera daam hai"
print(f"\n  Text: \"{sample}\"")
print(f"\n  Step-by-step BPE merges:")

tokens = list(sample)
merge_count = 0

for _ in range(10):
    pairs = {}
    for i in range(len(tokens) - 1):
        pair = (tokens[i], tokens[i+1])
        pairs[pair] = pairs.get(pair, 0) + 1

    if not pairs:
        break

    best = max(pairs, key=pairs.get)
    count = pairs[best]
    if count < 2:
        break

    merged = best[0] + best[1]
    new_tokens = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == best[0] and tokens[i+1] == best[1]:
            new_tokens.append(merged)
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1

    merge_count += 1
    tokens = new_tokens
    left = f"'{best[0]}' + '{best[1]}'"
    print(f"    Merge {merge_count}: {left:>15} → '{merged}' (found {count}x)")
    print(f"            Tokens now: {len(tokens)}")

print(f"\n  📊 Result:")
print(f"     Before BPE: {len(sample)} characters = {len(sample)} tokens")
print(f"     After BPE:  {len(sample)} characters = {len(tokens)} tokens")
print(f"     {(1 - len(tokens)/len(sample))*100:.0f}% compression! 🎉")

# ============================================================
# STEP 6: Key Insight
# ============================================================
input("\n👉 Step 6: Summary... [ENTER]")

print("\n" + "=" * 60)
print("💡 KEY INSIGHT")
print("=" * 60)
print()
print("  Computer sirf NUMBERS samajhta hai.")
print("  Tokenization = text → numbers. Yeh FIRST step hai har AI ka!")
print()
print("  ┌──────────────────────────┬────────────┬──────────────┐")
print("  │                          │ Char Token │ BPE (GPT)    │")
print("  ├──────────────────────────┼────────────┼──────────────┤")
print("  │ 'Namaste'                │ 7 tokens   │ 1-2 tokens   │")
print("  │ 'Mogambo khush hua'      │ 18 tokens  │ ~5 tokens    │")
print("  │ 'mera naam hai'          │ 13 tokens  │ ~3 tokens    │")
print("  └──────────────────────────┴────────────┴──────────────┘")
print()
print("  🔥 Next Episode: \"20 Lines Mein AI!\"")
print("     — AI Indian names generate karega jo exist nahi karte!")
print("─" * 60)
