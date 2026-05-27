"""
=============================================================================
 BUILDING A LANGUAGE MODEL FROM SCRATCH — Step by Step
=============================================================================

 What is a Language Model?
 -------------------------
 A language model predicts "what comes next" given some context.
 For example, given "hel", a good model might predict "l" → "hello"

 What is a Bigram Model?
 -----------------------
 The SIMPLEST possible language model. It only looks at the PREVIOUS
 character to predict the NEXT character. "Bigram" = two characters.

 Example: if it sees 'h', it learns that 'e' often follows.
          if it sees 'e', it learns that 'l' often follows.

 We'll build this in 6 stages:
   Stage 1: Load and explore the data
   Stage 2: Tokenize (convert text → numbers)
   Stage 3: Prepare training data
   Stage 4: Build the model
   Stage 5: Train the model
   Stage 6: Generate new text!

=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# STAGE 1: LOAD AND EXPLORE THE DATA
# ============================================================================
# We need text data to train on. We'll use a small Shakespeare dataset.
# The model will learn patterns in Shakespeare's writing.

print("=" * 60)
print("STAGE 1: Loading and Exploring Data")
print("=" * 60)

# Download Shakespeare text if not already present
import urllib.request
import os

data_path = os.path.join(os.path.dirname(__file__), "shakespeare.txt")
if not os.path.exists(data_path):
    print("Downloading Shakespeare dataset...")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    urllib.request.urlretrieve(url, data_path)

with open(data_path, "r") as f:
    text = f.read()

print(f"Total characters in dataset: {len(text):,}")
print(f"\nFirst 200 characters:")
print(text[:200])
print("...")

# ============================================================================
# STAGE 2: TOKENIZATION — Converting Text to Numbers
# ============================================================================
# Neural networks work with NUMBERS, not characters.
# We need to convert each character to a unique number (and back).
#
# This is called "tokenization". Real LLMs use subword tokens (like "hel" + "lo"),
# but we'll use character-level tokens for simplicity.
#
# Our "vocabulary" is just the set of unique characters in the text.

print("\n" + "=" * 60)
print("STAGE 2: Tokenization")
print("=" * 60)

chars = sorted(set(text))
vocab_size = len(chars)

print(f"Vocabulary size: {vocab_size} unique characters")
print(f"Characters: {''.join(chars)}")

# Create mappings: character → number and number → character
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# Encode: string → list of integers
def encode(s):
    return [char_to_idx[c] for c in s]

# Decode: list of integers → string
def decode(indices):
    return "".join(idx_to_char[i] for i in indices)

# Demo
sample = "Hello"
encoded = encode(sample)
decoded = decode(encoded)
print(f"\nEncoding demo:")
print(f"  '{sample}' → {encoded} → '{decoded}'")

# Convert entire text to a tensor (a big list of numbers)
data = torch.tensor(encode(text), dtype=torch.long)
print(f"\nFull dataset shape: {data.shape} (one number per character)")
print(f"First 20 tokens: {data[:20].tolist()}")

# ============================================================================
# STAGE 3: PREPARE TRAINING DATA
# ============================================================================
# We split data into training (90%) and validation (10%).
#
# For a bigram model, each training example is simple:
#   Input:  one character  (e.g., 'h' = token 46)
#   Target: next character (e.g., 'e' = token 43)
#
# We process these in "batches" for efficiency.
# A "batch" = multiple examples processed simultaneously.
#
# "block_size" = how many consecutive characters we look at.
# For bigram, we only need 1, but we'll use a small block to show
# how this generalizes to larger models.

print("\n" + "=" * 60)
print("STAGE 3: Preparing Training Data")
print("=" * 60)

train_split = int(0.9 * len(data))
train_data = data[:train_split]
val_data = data[train_split:]
print(f"Training set:   {len(train_data):,} characters")
print(f"Validation set: {len(val_data):,} characters")

block_size = 8   # context length: how many characters the model sees
batch_size = 32   # how many sequences to process in parallel

def get_batch(split):
    """
    Grab a random batch of training examples.
    Returns (inputs, targets) where:
      - inputs:  [batch_size, block_size] tensor
      - targets: [batch_size, block_size] tensor (shifted by 1)
    """
    d = train_data if split == "train" else val_data
    # Pick random starting positions
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x, y

# Demo: show what a batch looks like
xb, yb = get_batch("train")
print(f"\nBatch shape: inputs={xb.shape}, targets={yb.shape}")
print(f"  (That's {batch_size} sequences, each {block_size} characters long)")

print(f"\nExample from the batch:")
for i in range(min(3, block_size)):
    context = xb[0, :i + 1].tolist()
    target = yb[0, i].item()
    print(f"  When input is {context} → predict {target}")
    print(f"  i.e., '{decode(context)}' → '{decode([target])}'")


# ============================================================================
# STAGE 4: BUILD THE BIGRAM MODEL
# ============================================================================
# The bigram model is incredibly simple:
#
#   1. It has a "lookup table" (embedding) of size [vocab_size × vocab_size]
#   2. For each input character, it looks up a row in this table
#   3. That row contains scores ("logits") for what the next character should be
#
# For example, if 'h' is character #46, the model looks up row 46,
# which might say: 'e' has high score, 'z' has low score.
#
# The model learns these scores during training!

print("\n" + "=" * 60)
print("STAGE 4: Building the Bigram Model")
print("=" * 60)


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # This is the ENTIRE model: one embedding table.
        # token_embedding_table[i] = probability distribution over next characters
        # given that the current character is i
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx shape: [batch_size, block_size] — the input characters
        # Look up each character in the embedding table
        logits = self.token_embedding_table(idx)  # [batch, block_size, vocab_size]

        if targets is None:
            return logits, None

        # Reshape for loss calculation
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)

        # Cross-entropy loss: how wrong are our predictions?
        # Lower loss = better predictions
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        Generate new text, one character at a time.
        idx: starting context [batch_size, current_length]
        """
        for _ in range(max_new_tokens):
            # Get predictions (only need the last character for bigram)
            logits, _ = self(idx)
            logits = logits[:, -1, :]  # [batch_size, vocab_size]

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)  # [batch_size, vocab_size]

            # Sample from the probability distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # [batch_size, 1]

            # Append to the sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


model = BigramLanguageModel(vocab_size)
print(f"Model created!")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  (That's just a {vocab_size}×{vocab_size} = {vocab_size**2:,} number table!)")

# Generate text BEFORE training (should be random garbage)
print(f"\nText generated BEFORE training (random gibberish):")
context = torch.zeros((1, 1), dtype=torch.long)  # start with character 0
generated = model.generate(context, max_new_tokens=100)
print(decode(generated[0].tolist()))

# ============================================================================
# STAGE 5: TRAIN THE MODEL
# ============================================================================
# Training = adjusting the embedding table so predictions get better.
#
# The training loop:
#   1. Grab a batch of (input, target) pairs
#   2. Ask the model to predict → get loss (how wrong it is)
#   3. Compute gradients (which direction to adjust weights)
#   4. Update weights to reduce loss
#   5. Repeat!
#
# We use the "Adam" optimizer — a smart version of gradient descent.

print("\n" + "=" * 60)
print("STAGE 5: Training the Model")
print("=" * 60)

learning_rate = 1e-2  # how big each adjustment step is (smaller = more careful)
max_steps = 10000     # how many training steps
eval_interval = 1000  # how often to check progress

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

@torch.no_grad()
def estimate_loss():
    """Average loss over multiple batches (more reliable than single batch)."""
    model.eval()
    losses = {}
    for split in ["train", "val"]:
        batch_losses = torch.zeros(200)
        for k in range(200):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            batch_losses[k] = loss.item()
        losses[split] = batch_losses.mean()
    model.train()
    return losses

print(f"Training for {max_steps:,} steps...")
print(f"{'Step':>6} | {'Train Loss':>10} | {'Val Loss':>10}")
print("-" * 35)

for step in range(max_steps):
    # Evaluate periodically
    if step % eval_interval == 0 or step == max_steps - 1:
        losses = estimate_loss()
        print(f"{step:6d} | {losses['train']:10.4f} | {losses['val']:10.4f}")

    # Training step
    xb, yb = get_batch("train")          # 1. Get batch
    logits, loss = model(xb, yb)          # 2. Forward pass → loss
    optimizer.zero_grad(set_to_none=True) # 3. Clear old gradients
    loss.backward()                       # 4. Compute new gradients
    optimizer.step()                      # 5. Update weights

print("\nTraining complete!")

# ============================================================================
# STAGE 6: GENERATE TEXT!
# ============================================================================
# Now our model has learned character-by-character patterns from Shakespeare.
# Let's generate some text and see what it produces!

print("\n" + "=" * 60)
print("STAGE 6: Generating Text!")
print("=" * 60)

context = torch.zeros((1, 1), dtype=torch.long)
generated = model.generate(context, max_new_tokens=500)
print(decode(generated[0].tolist()))

print("\n" + "=" * 60)
print("WHAT YOU JUST BUILT")
print("=" * 60)
print("""
You built a character-level bigram language model! Here's what each piece does:

1. TOKENIZER: Converts characters ↔ numbers
   - 'h' → 46, 'e' → 43, etc.

2. EMBEDDING TABLE: A [vocab_size × vocab_size] matrix
   - Row i = "given character i, how likely is each next character?"
   - This is the ENTIRE "knowledge" of the model

3. TRAINING: Adjusted the table using gradient descent
   - Showed it millions of character pairs from Shakespeare
   - It learned: after 'q', 'u' is very likely
                  after 'T', 'h' is likely (as in "The")
                  after newline, uppercase is likely

4. GENERATION: Sample one character at a time
   - Look up current character → get probability distribution → sample → repeat

LIMITATIONS (and why real LLMs are more complex):
  - Only looks at 1 previous character (no long-range context)
  - Character-level (real LLMs use subword tokens)
  - Tiny dataset and model (GPT-4 has ~1.8 trillion parameters, ours has ~4,000)

NEXT STEPS to make this more powerful:
  - Add self-attention → the model can look at MORE than 1 previous character
  - Add multiple layers → the model can learn more complex patterns
  - Use subword tokenization → more efficient representation
  → That's essentially what GPT is!
""")
