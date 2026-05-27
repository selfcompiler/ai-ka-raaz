"""
=============================================================================
 TOKENIZATION — Deep Dive (From First Principles)
=============================================================================

 WHY TOKENIZATION MATTERS
 -------------------------
 Neural networks only understand NUMBERS, not text.
 Tokenization is the bridge: Text → Numbers → Neural Network → Numbers → Text

 The way you tokenize FUNDAMENTALLY affects:
   - How much text the model can "see" (context window)
   - How well it handles rare/unknown words
   - How fast training is
   - What languages it can support
   - How much memory it uses

 We'll explore 4 approaches, from simplest to what real LLMs use:

   Level 1: Character-level  — "hello" → ['h','e','l','l','o']
   Level 2: Word-level       — "hello world" → ['hello', 'world']
   Level 3: BPE (from scratch!) — "hello" → ['hel', 'lo']  (learned subwords)
   Level 4: Real-world tokenizers (tiktoken / GPT)

=============================================================================
"""

import re
from collections import Counter, defaultdict

# Sample texts we'll use throughout
sample_texts = [
    "The cat sat on the mat.",
    "The cat sat on the mat, and the dog sat on the rug.",
    "Tokenization is surprisingly important for language models!",
    "GPT-4 uses byte-pair encoding (BPE) tokenization.",
    "未知の単語もトークン化できます。",  # Japanese: "Unknown words can also be tokenized."
    "Supercalifragilisticexpialidocious is a long word.",
    "I looooooove this!!! 😊😊😊",
    "The price is $99.99 for the 2024 model.",
]


# ============================================================================
# LEVEL 1: CHARACTER-LEVEL TOKENIZATION
# ============================================================================
# The simplest approach: each character is one token.
# This is what our bigram model used.

print("=" * 70)
print("LEVEL 1: CHARACTER-LEVEL TOKENIZATION")
print("=" * 70)

class CharTokenizer:
    """Tokenizes text one character at a time."""

    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}

    def train(self, text):
        chars = sorted(set(text))
        self.char_to_id = {ch: i for i, ch in enumerate(chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text):
        return [self.char_to_id[ch] for ch in text]

    def decode(self, ids):
        return "".join(self.id_to_char[i] for i in ids)

    @property
    def vocab_size(self):
        return len(self.char_to_id)


char_tok = CharTokenizer()
all_sample_text = "".join(sample_texts)
# Include both upper and lowercase plus common ASCII for robustness
char_tok.train(all_sample_text + "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz 0123456789.,!?;:'\"()-")

text = "Hello world"
encoded = char_tok.encode(text)
print(f"\nInput:    '{text}'")
print(f"Tokens:   {list(text)}")
print(f"Token IDs: {encoded}")
print(f"Decoded:  '{char_tok.decode(encoded)}'")
print(f"Vocab size: {char_tok.vocab_size}")
print(f"Tokens needed: {len(encoded)} (one per character)")

print(f"""
┌──────────────────────────────────────────────────────────────────┐
│  PROS:                                                          │
│  ✓ Tiny vocabulary (~100-300 characters covers most languages)  │
│  ✓ No "unknown" words — any text can be tokenized               │
│  ✓ Simple to implement                                          │
│                                                                  │
│  CONS:                                                          │
│  ✗ Very LONG sequences — "hello" = 5 tokens                    │
│    (wastes the model's limited context window)                  │
│  ✗ Hard to learn meaning — model must figure out that           │
│    'h','e','l','l','o' together mean "hello"                    │
│  ✗ Each token carries very little information                   │
└──────────────────────────────────────────────────────────────────┘
""")


# ============================================================================
# LEVEL 2: WORD-LEVEL TOKENIZATION
# ============================================================================
# Split text into words. Each word is one token.
# This is the opposite extreme from character-level.

print("=" * 70)
print("LEVEL 2: WORD-LEVEL TOKENIZATION")
print("=" * 70)

class WordTokenizer:
    """Tokenizes text by splitting on whitespace and punctuation."""

    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.unk_id = 0

    def train(self, texts):
        self.word_to_id = {"<UNK>": 0}  # unknown token for words we haven't seen
        for text in texts:
            words = self._split(text)
            for word in words:
                if word not in self.word_to_id:
                    self.word_to_id[word] = len(self.word_to_id)
        self.id_to_word = {i: w for w, i in self.word_to_id.items()}

    def _split(self, text):
        # Split on word boundaries, keeping punctuation as separate tokens
        return re.findall(r"\w+|[^\w\s]", text)

    def encode(self, text):
        return [self.word_to_id.get(w, self.unk_id) for w in self._split(text)]

    def decode(self, ids):
        return " ".join(self.id_to_word.get(i, "<UNK>") for i in ids)

    @property
    def vocab_size(self):
        return len(self.word_to_id)


word_tok = WordTokenizer()
word_tok.train(sample_texts)

text = "The cat sat on the mat."
encoded = word_tok.encode(text)
tokens = word_tok._split(text)
print(f"\nInput:    '{text}'")
print(f"Tokens:   {tokens}")
print(f"Token IDs: {encoded}")
print(f"Vocab size: {word_tok.vocab_size}")
print(f"Tokens needed: {len(encoded)} (one per word/punctuation)")

# Show the problem: unknown words
unknown_text = "The elephant danced on the moon."
encoded_unk = word_tok.encode(unknown_text)
tokens_unk = word_tok._split(unknown_text)
print(f"\nUnseen text: '{unknown_text}'")
print(f"Tokens: {tokens_unk}")
print(f"IDs:    {encoded_unk}")
print(f"Notice: 'elephant', 'danced', 'moon' → ID 0 (<UNK>) — LOST forever!")

print(f"""
┌──────────────────────────────────────────────────────────────────┐
│  PROS:                                                          │
│  ✓ Short sequences — each token is a meaningful word            │
│  ✓ Each token carries a lot of semantic meaning                 │
│                                                                  │
│  CONS:                                                          │
│  ✗ HUGE vocabulary needed (English has 170,000+ words)          │
│  ✗ Cannot handle unseen words → <UNK>                           │
│  ✗ "running", "runs", "ran" are all different tokens            │
│    (no shared knowledge between related forms)                  │
│  ✗ Struggles with: typos, new words, code, other languages     │
│  ✗ "low-cost" — is this 1 token, 2, or 3?                     │
└──────────────────────────────────────────────────────────────────┘
""")


# ============================================================================
# THE KEY INSIGHT: WHY SUBWORD TOKENIZATION?
# ============================================================================

print("=" * 70)
print("THE KEY INSIGHT: Subword Tokenization")
print("=" * 70)

print("""
Character-level:  Too granular — sequences are too long
Word-level:       Too coarse  — vocabulary is too large, can't handle new words

SUBWORD is the sweet spot:
  - Common words stay as single tokens:  "the" → ["the"]
  - Rare words get split into pieces:    "unhappiness" → ["un", "happi", "ness"]
  - The model can understand new words by combining known pieces!

Example: the model has never seen "tokenizers" but knows:
  "token" + "izer" + "s" → it can guess the meaning!

The most popular subword algorithm is BPE (Byte Pair Encoding).
Let's build it FROM SCRATCH to understand exactly how it works.
""")


# ============================================================================
# LEVEL 3: BPE (Byte Pair Encoding) — FROM SCRATCH
# ============================================================================
# BPE was originally a data compression algorithm (1994).
# OpenAI adapted it for NLP in 2018. GPT-2, GPT-3, GPT-4 all use BPE.
#
# The core idea is beautifully simple:
#   1. Start with individual characters as your vocabulary
#   2. Find the most frequent PAIR of adjacent tokens
#   3. Merge that pair into a new single token
#   4. Repeat until you reach your desired vocabulary size
#
# Let's see this step by step.

print("=" * 70)
print("LEVEL 3: BPE — Byte Pair Encoding (From Scratch)")
print("=" * 70)

print("\n--- Step-by-Step BPE Merges ---\n")

# Let's trace through BPE on a small example
demo_text = "low low low low low lowest lowest newer newer newer newer newer newer wider wider wider new new"

# Step 1: Split into characters, with a special end-of-word marker </w>
# The </w> marker helps the model know where words end.
# Without it, "the" inside "other" would look the same as the word "the".

def get_word_freqs(text):
    """Count how often each word appears, split into characters."""
    words = text.split()
    freq = Counter(words)
    # Represent each word as a tuple of characters + end marker
    word_freqs = {}
    for word, count in freq.items():
        chars = tuple(list(word) + ["</w>"])
        word_freqs[chars] = count
    return word_freqs


word_freqs = get_word_freqs(demo_text)
print("Starting vocabulary (each character is a token):")
print(f"Word frequencies:")
for word, freq in sorted(word_freqs.items(), key=lambda x: -x[1]):
    print(f"  {' '.join(word):30s} (appears {freq}x)")

# Get all unique tokens
all_tokens = set()
for word in word_freqs:
    all_tokens.update(word)
print(f"\nInitial vocabulary ({len(all_tokens)} tokens): {sorted(all_tokens)}")


def get_pair_counts(word_freqs):
    """Count how often each pair of adjacent tokens appears."""
    pairs = Counter()
    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += freq
    return pairs


def merge_pair(word_freqs, pair):
    """Merge all occurrences of a pair into a single new token."""
    new_word_freqs = {}
    merged = pair[0] + pair[1]
    for word, freq in word_freqs.items():
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                new_word.append(merged)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word_freqs[tuple(new_word)] = freq
    return new_word_freqs


# Now let's perform BPE merges and watch what happens!
num_merges = 10
merges = []  # record merge operations

print(f"\n--- Performing {num_merges} BPE merges ---\n")
for i in range(num_merges):
    pairs = get_pair_counts(word_freqs)
    if not pairs:
        break

    # Find the most frequent pair
    best_pair = max(pairs, key=pairs.get)
    best_count = pairs[best_pair]

    print(f"Merge #{i+1}: '{best_pair[0]}' + '{best_pair[1]}' → "
          f"'{best_pair[0]+best_pair[1]}' (appeared {best_count} times)")

    merges.append(best_pair)
    word_freqs = merge_pair(word_freqs, best_pair)

    # Show current state of words
    for word, freq in sorted(word_freqs.items(), key=lambda x: -x[1]):
        print(f"         {' '.join(word):30s} ({freq}x)")
    print()

# Show final vocabulary
final_tokens = set()
for word in word_freqs:
    final_tokens.update(word)
print(f"Final vocabulary ({len(final_tokens)} tokens):")
print(f"  {sorted(final_tokens, key=len)}")

print("""
WHAT JUST HAPPENED:
  - 'e' and 'r' appeared together a lot → merged into 'er'
  - 'n' and 'e' appeared together → merged into 'ne'
  - 'ne' and 'w' merged → 'new'
  - Common words became single tokens, rare words stay split
  - The algorithm automatically discovers meaningful pieces!
""")


# ============================================================================
# BUILDING A COMPLETE BPE TOKENIZER
# ============================================================================

print("=" * 70)
print("COMPLETE BPE TOKENIZER (from scratch)")
print("=" * 70)


class BPETokenizer:
    """
    A complete Byte Pair Encoding tokenizer built from scratch.

    How it works:
    1. TRAINING: Learn merge rules from a text corpus
       - Start with character-level tokens
       - Repeatedly merge the most frequent pair
       - Save the merge rules in order

    2. ENCODING: Apply learned merges to new text
       - Split text into characters
       - Apply merge rules in the order they were learned
       - Each remaining piece is a token

    3. DECODING: Convert token IDs back to text
    """

    def __init__(self, vocab_size=300):
        self.target_vocab_size = vocab_size
        self.merges = []          # ordered list of merge rules: (token_a, token_b)
        self.token_to_id = {}
        self.id_to_token = {}

    def train(self, text):
        """Learn BPE merge rules from training text."""
        print(f"\nTraining BPE tokenizer (target vocab size: {self.target_vocab_size})...")

        # Step 1: Count word frequencies
        word_freqs = self._count_words(text)
        print(f"  Unique words in corpus: {len(word_freqs)}")

        # Step 2: Build initial character vocabulary
        vocab = set()
        for word in word_freqs:
            vocab.update(word)
        initial_vocab_size = len(vocab)
        print(f"  Initial character vocab: {initial_vocab_size} tokens")

        # Step 3: Perform merges until we reach target vocab size
        num_merges = self.target_vocab_size - initial_vocab_size
        print(f"  Will perform {num_merges} merges...")

        for i in range(num_merges):
            pairs = get_pair_counts(word_freqs)
            if not pairs:
                print(f"  No more pairs to merge at step {i}")
                break

            best_pair = max(pairs, key=pairs.get)
            self.merges.append(best_pair)
            word_freqs = merge_pair(word_freqs, best_pair)

            if (i + 1) % 50 == 0 or i < 5:
                print(f"  Merge {i+1:3d}: '{best_pair[0]}' + '{best_pair[1]}' "
                      f"→ '{best_pair[0]+best_pair[1]}'")

        # Step 4: Build final vocabulary mapping
        # Collect all tokens that exist after merging
        final_vocab = set()
        for word in word_freqs:
            final_vocab.update(word)

        # Also include individual characters (for encoding unseen text)
        all_chars = set(text)
        final_vocab.update(all_chars)

        # Assign IDs
        sorted_vocab = sorted(final_vocab, key=lambda x: (len(x), x))
        self.token_to_id = {token: i for i, token in enumerate(sorted_vocab)}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}

        print(f"  Final vocab size: {len(self.token_to_id)} tokens")
        print(f"  Merges learned: {len(self.merges)}")

    def _count_words(self, text):
        """Split text into words and count frequencies."""
        words = re.findall(r"\w+|[^\w\s]", text.lower())
        freq = Counter(words)
        word_freqs = {}
        for word, count in freq.items():
            chars = tuple(list(word) + ["</w>"])
            word_freqs[chars] = count
        return word_freqs

    def _apply_merges(self, tokens):
        """Apply learned merge rules to a list of tokens."""
        for merge_pair in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (i < len(tokens) - 1
                        and tokens[i] == merge_pair[0]
                        and tokens[i + 1] == merge_pair[1]):
                    new_tokens.append(merge_pair[0] + merge_pair[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def encode(self, text):
        """Encode text into token IDs."""
        words = re.findall(r"\w+|[^\w\s]|\s+", text.lower())
        all_ids = []
        for word in words:
            if word.isspace():
                # Handle spaces — encode as character tokens
                for ch in word:
                    if ch in self.token_to_id:
                        all_ids.append(self.token_to_id[ch])
                continue
            # Split word into characters + end marker
            tokens = list(word) + ["</w>"]
            # Apply learned merges
            tokens = self._apply_merges(tokens)
            # Convert to IDs
            for token in tokens:
                if token in self.token_to_id:
                    all_ids.append(self.token_to_id[token])
                else:
                    # Fall back to character-level for unknown characters
                    for ch in token:
                        if ch in self.token_to_id:
                            all_ids.append(self.token_to_id[ch])
        return all_ids

    def decode(self, ids):
        """Decode token IDs back to text."""
        tokens = [self.id_to_token.get(i, "?") for i in ids]
        text = "".join(tokens)
        # Remove end-of-word markers and clean up
        text = text.replace("</w>", " ").rstrip()
        return text

    def tokenize(self, text):
        """Show the actual token strings (not just IDs)."""
        words = re.findall(r"\w+|[^\w\s]|\s+", text.lower())
        all_tokens = []
        for word in words:
            if word.isspace():
                all_tokens.append(word)
                continue
            tokens = list(word) + ["</w>"]
            tokens = self._apply_merges(tokens)
            all_tokens.extend(tokens)
        return all_tokens


# Train our BPE tokenizer on the Shakespeare data!
import os
data_path = os.path.join(os.path.dirname(__file__), "shakespeare.txt")
with open(data_path, "r") as f:
    shakespeare = f.read()

bpe_tok = BPETokenizer(vocab_size=500)
bpe_tok.train(shakespeare)

# Now let's see it in action!
print("\n--- BPE Tokenization Examples ---\n")

test_sentences = [
    "To be or not to be",
    "The quality of mercy is not strained",
    "Something wicked this way comes",
    "Tokenization is amazing",
    "Supercalifragilistic",
]

for sent in test_sentences:
    tokens = bpe_tok.tokenize(sent)
    ids = bpe_tok.encode(sent)
    decoded = bpe_tok.decode(ids)

    # Clean display of tokens
    display_tokens = [t.replace("</w>", "·") for t in tokens if not t.isspace()]
    print(f"Input:   '{sent}'")
    print(f"Tokens:  {display_tokens}")
    print(f"IDs:     {ids[:15]}{'...' if len(ids) > 15 else ''}")
    print(f"Decoded: '{decoded}'")
    print(f"Token count: {len([t for t in tokens if not t.isspace()])}")
    print()


# ============================================================================
# COMPARING ALL THREE APPROACHES
# ============================================================================

print("=" * 70)
print("COMPARISON: Character vs Word vs BPE")
print("=" * 70)

comparison_text = "The quality of mercy is not strained"

char_tok_2 = CharTokenizer()
char_tok_2.train(shakespeare)
word_tok_2 = WordTokenizer()
word_tok_2.train([shakespeare])

char_tokens = list(comparison_text)
word_tokens = word_tok_2._split(comparison_text)
bpe_tokens = [t.replace("</w>", "·") for t in bpe_tok.tokenize(comparison_text) if not t.isspace()]

print(f"\nText: '{comparison_text}'\n")
print(f"{'Method':<12} {'Vocab Size':>10} {'# Tokens':>10}  Tokens")
print("-" * 70)
print(f"{'Character':<12} {char_tok_2.vocab_size:>10} {len(char_tokens):>10}  {char_tokens}")
print(f"{'Word':<12} {word_tok_2.vocab_size:>10} {len(word_tokens):>10}  {word_tokens}")
print(f"{'BPE':<12} {len(bpe_tok.token_to_id):>10} {len(bpe_tokens):>10}  {bpe_tokens}")

print("""
                    ┌─────────────────────────────────────────┐
                    │     THE TOKENIZATION TRADEOFF            │
                    │                                         │
    Vocab Size      │  ←── Smaller          Larger ──→        │
    Sequence Length  │  ←── Longer          Shorter ──→        │
                    │                                         │
    Character ──────│──●                                      │
    BPE ────────────│────────────●                             │
    Word ───────────│──────────────────────────────────●       │
                    │                                         │
                    │  BPE hits the sweet spot:                │
                    │  - Reasonable vocab (~50K tokens)        │
                    │  - Compact sequences                    │
                    │  - Handles any text                     │
                    └─────────────────────────────────────────┘
""")


# ============================================================================
# LEVEL 4: REAL-WORLD TOKENIZERS
# ============================================================================

print("=" * 70)
print("LEVEL 4: How Real LLMs Tokenize (GPT, Claude, etc.)")
print("=" * 70)

print("""
Real LLMs use enhanced versions of BPE. Here's what they add:

1. BYTE-LEVEL BPE (used by GPT-2, GPT-3, GPT-4)
   - Instead of starting with characters, start with raw BYTES (0-255)
   - Can represent ANY text in ANY language (even binary data!)
   - No more <UNK> tokens ever

2. SENTENCEPIECE (used by LLaMA, T5)
   - Treats the input as a raw byte stream (no pre-tokenization)
   - Language-agnostic: works equally well for English, Chinese, Arabic
   - Includes a "Unigram" model as an alternative to BPE

3. SPECIAL TOKENS
   - <|endoftext|>  — marks where one document ends
   - <|im_start|>   — marks start of a message
   - <|pad|>        — padding for fixed-length batches
   - These are added on top of the learned vocabulary

TYPICAL VOCAB SIZES:
  ┌────────────────────┬──────────────┐
  │ Model              │ Vocab Size   │
  ├────────────────────┼──────────────┤
  │ GPT-2              │    50,257    │
  │ GPT-3/3.5          │    50,257    │
  │ GPT-4 / GPT-4o     │   100,256    │
  │ Claude (Anthropic)  │   ~100,000   │
  │ LLaMA 2            │    32,000    │
  │ LLaMA 3            │   128,256    │
  └────────────────────┴──────────────┘
""")

# Try to use tiktoken if available (OpenAI's fast BPE implementation)
try:
    import tiktoken
    has_tiktoken = True
except ImportError:
    has_tiktoken = False
    print("(Install 'tiktoken' to see GPT-4's actual tokenizer: pip install tiktoken)\n")

if has_tiktoken:
    print("\n--- GPT-4's Actual Tokenizer (tiktoken) ---\n")

    enc = tiktoken.encoding_for_model("gpt-4")

    for text in test_sentences:
        token_ids = enc.encode(text)
        token_strings = [enc.decode([t]) for t in token_ids]
        print(f"Input:  '{text}'")
        print(f"Tokens: {token_strings}")
        print(f"IDs:    {token_ids}")
        print(f"Count:  {len(token_ids)} tokens")
        print()


# ============================================================================
# INTERESTING TOKENIZATION EDGE CASES
# ============================================================================

print("=" * 70)
print("EDGE CASES & GOTCHAS")
print("=" * 70)

print("""
Tokenization has real consequences for model behavior:

1. ARITHMETIC IS HARD
   "380 + 245" might tokenize as ["380", " +", " 245"]
   But "1234567" might split as ["123", "456", "7"]
   → The model sees different "pieces" of numbers inconsistently
   → This is partly why LLMs struggle with math!

2. SPACING MATTERS
   " hello" (with leading space) and "hello" are DIFFERENT tokens
   GPT-4 treats " hello" as one token but "hello" as another
   → Subtle bugs in prompts if you're not careful

3. NON-ENGLISH LANGUAGES ARE EXPENSIVE
   English "hello" = 1 token
   Japanese "こんにちは" = might be 3-5 tokens
   → Same message costs more tokens in some languages
   → Less context window available for non-English text

4. CODE TOKENIZATION
   "    " (4 spaces / 1 tab indent) = could be 1 or 4 tokens
   "def" = 1 token, but "async def" might be 2 tokens
   → Indentation-heavy languages (Python) use more tokens

5. REPETITION
   "aaaa" and "aaaaa" tokenize completely differently
   → Models behave unpredictably with repeated characters
""")


# ============================================================================
# VISUALIZE: TOKEN BOUNDARIES
# ============================================================================

print("=" * 70)
print("VISUALIZING TOKEN BOUNDARIES")
print("=" * 70)

def visualize_tokens(text, tokenizer):
    """Show where our BPE tokenizer splits text."""
    tokens = tokenizer.tokenize(text)
    display_tokens = [t for t in tokens if not t.isspace()]
    result = ""
    for t in display_tokens:
        clean = t.replace("</w>", "")
        result += f"[{clean}]"
    return result, len(display_tokens)


viz_texts = [
    "the",
    "there",
    "therefore",
    "the king",
    "kingdom",
    "unkingly",
    "the unkingly king",
    "hello world",
    "shakespeare",
]

print(f"\n{'Input':<25} {'Tokenized':<40} {'# Tokens':>8}")
print("-" * 75)
for text in viz_texts:
    viz, count = visualize_tokens(text, bpe_tok)
    print(f"{text:<25} {viz:<40} {count:>8}")


# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("WHAT YOU LEARNED")
print("=" * 70)
print("""
1. TOKENIZATION converts text → numbers for neural networks

2. THREE LEVELS:
   - Character: simple but sequences are too long
   - Word: compact but can't handle new words
   - Subword (BPE): the Goldilocks solution used by real LLMs

3. BPE ALGORITHM:
   - Start with characters
   - Count pair frequencies
   - Merge the most common pair
   - Repeat until vocab size reached
   - Result: common words = 1 token, rare words = split into pieces

4. WHY IT MATTERS:
   - Token count = cost (API pricing)
   - Token count = context limit (how much the model can "see")
   - Tokenization affects model behavior (math, multilingual, code)

5. REAL-WORLD:
   - GPT-4 uses Byte-level BPE with ~100K tokens
   - Special tokens mark document boundaries and roles
   - tiktoken library lets you see exactly how GPT tokenizes

FILES CREATED:
  - bigram_lm.py         → Your first language model (from last session)
  - tokenization_deep_dive.py → This file (tokenization deep dive)

NEXT STEPS:
  → Add SELF-ATTENTION to the bigram model (the key idea behind transformers)
  → Or explore EMBEDDINGS — how tokens become meaningful vectors
""")
