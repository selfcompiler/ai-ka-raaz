"""
🎬 AI Ka Raaz — Episode 2: "Computer Ko Hindi Kaise Sikhaye?"
See how AI converts text to numbers and back!

Run: python ep02_tokenization.py
"""
import os

# --- STEP 1: Build a Character Tokenizer ---
print("=" * 60)
print("🔢 AI KA RAAZ — Episode 2: Tokenization")
print("=" * 60)

# Load Shakespeare for our vocabulary
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shakespeare.txt')
if not os.path.exists(data_path):
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shakespeare.txt')
text = open(data_path).read()

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}

print(f"\n📖 Vocabulary: {vocab_size} characters from Shakespeare")
print(f"   Characters: {''.join(chars[:20])}... (first 20)")
print()

# --- STEP 2: Encode & Decode Demo ---
print("─" * 60)
print("🔐 ENCODING: Text → Numbers")
print("─" * 60)

demos = [
    "Hello",
    "The king shall not",
    "To be or not to be",
]

for sentence in demos:
    tokens = [stoi[c] for c in sentence if c in stoi]
    print(f"\n  \"{sentence}\"")
    print(f"  → {tokens}")
    mapping = [f"{c}={stoi[c]}" for c in sentence if c in stoi]
    print(f"     ({', '.join(mapping)})")

# --- STEP 3: Decode back ---
print()
print("─" * 60)
print("🔓 DECODING: Numbers → Text")
print("─" * 60)

secret = [32, 46, 43, 1, 49, 47, 52, 45, 1, 57, 46, 39, 50, 50, 1, 52, 53, 58]
decoded = ''.join(itos[i] for i in secret if i in itos)
print(f"\n  Secret message: {secret}")
print(f"  Decoded:        \"{decoded}\"")
print(f"\n  ✅ Numbers → Text → Numbers. Perfectly reversible!")

# --- STEP 4: The Problem with Character Tokens ---
print()
print("─" * 60)
print("⚠️ PROBLEM: Character Tokens Ka Masla")
print("─" * 60)

examples = [
    ("Hello", 5),
    ("Shakespeare", 11),
    ("Supercalifragilistic", 20),
    ("To be or not to be that is the question", 40),
]

print("\n  Character-level tokenization:")
for word, count in examples:
    bar = "█" * count
    print(f"  \"{word}\" → {count} tokens {bar}")

print(f"\n  😱 ChatGPT limit: ~4000 tokens")
print(f"     Character level pe ek chhota paragraph = limit khatam!")

# --- STEP 5: BPE — The Solution ---
print()
print("─" * 60)
print("🧠 BPE (Byte Pair Encoding) — AI Ka Real Trick!")
print("─" * 60)

sample = "the cat sat on the mat the cat ate the hat"
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
    print(f"            Tokens now: {len(tokens)} → {tokens[:15]}{'...' if len(tokens) > 15 else ''}")

print(f"\n  📊 Result:")
print(f"     Before BPE: {len(sample)} characters = {len(sample)} tokens")
print(f"     After BPE:  {len(sample)} characters = {len(tokens)} tokens")
print(f"     {(1 - len(tokens)/len(sample))*100:.0f}% compression! 🎉")

# --- STEP 6: Summary ---
print()
print("=" * 60)
print("💡 KEY INSIGHT")
print("=" * 60)
print()
print("  Computer sirf NUMBERS samajhta hai.")
print("  Tokenization = text → numbers.")
print()
print("  Character level: Simple, lekin bohot zyada tokens.")
print("  BPE: Smart merging, kam tokens. REAL AI yeh use karta hai!")
print()
print("  ┌──────────────────┬────────────┬──────────────────┐")
print("  │                  │ Char Token │ BPE (GPT-style)  │")
print("  ├──────────────────┼────────────┼──────────────────┤")
print("  │ 'Hello'          │ 5 tokens   │ 1 token          │")
print("  │ 'the'            │ 3 tokens   │ 1 token          │")
print("  │ 'Supercali...'   │ 20 tokens  │ ~5 tokens        │")
print("  │ Speed            │ Slow ❌    │ Fast ✅          │")
print("  └──────────────────┴────────────┴──────────────────┘")
print()
print("  🔥 Next Episode: \"20 Lines Mein AI!\"")
print("     — Names generate karega jo exist nahi karte!")
print("─" * 60)
