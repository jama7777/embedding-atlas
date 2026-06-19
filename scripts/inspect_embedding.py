"""
Phase 1 — Inspect the embedding vector for a target token.

Run this first, look at the printed dimension values,
then tell us which 2 dims to change and by how much.

Usage:
    cd /Users/m4_pro/Desktop/jama/embedding-atlas
    python scripts/inspect_embedding.py
"""
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = "cache/local_qwen_0.5b"
TARGET_WORD = "the"     # ← change this if you want to inspect a different token
TOP_N_DIMS  = 30        # how many dimension values to print

print(f"\n{'='*60}")
print(f"  Loading Qwen2.5-0.5B from: {MODEL_DIR}")
print(f"{'='*60}\n")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model     = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True)
model.eval()

# Get the raw embedding matrix  (vocab_size × hidden_dim)
emb_matrix = model.get_input_embeddings().weight.detach()
print(f"Embedding matrix shape: {emb_matrix.shape}  "
      f"(vocab={emb_matrix.shape[0]}, dim={emb_matrix.shape[1]})\n")

# Find ALL token-ids that decode to something containing TARGET_WORD
vocab_size = emb_matrix.shape[0]
matches = []
for tid in range(vocab_size):
    decoded = tokenizer.decode([tid])
    clean   = decoded.strip().lower()
    if clean == TARGET_WORD or clean == " " + TARGET_WORD:
        matches.append((tid, repr(decoded)))

if not matches:
    # fallback: substring search
    for tid in range(vocab_size):
        decoded = tokenizer.decode([tid])
        if TARGET_WORD in decoded.lower():
            matches.append((tid, repr(decoded)))
    print(f"No exact match for '{TARGET_WORD}', showing substring matches:\n")
else:
    print(f"Found {len(matches)} token(s) that decode to '{TARGET_WORD}':\n")

for tid, rep in matches[:5]:   # show at most 5
    vec = emb_matrix[tid].float().numpy()
    print(f"  token_id={tid}  decoded={rep}")
    print(f"  Embedding dims 0–{TOP_N_DIMS-1}:")
    for i in range(TOP_N_DIMS):
        marker = " ← " if i < 2 else "    "   # highlight first 2 as examples
        print(f"    dim[{i:>3}] = {vec[i]:+.6f}{marker}")
    print(f"\n  L2-norm  = {np.linalg.norm(vec):.4f}")
    print(f"  Min val  = {vec.min():+.6f}  (dim {vec.argmin()})")
    print(f"  Max val  = {vec.max():+.6f}  (dim {vec.argmax()})\n")

print("="*60)
print("Next step:")
print("  Tell us which 2 dimension indices to perturb and by how much,")
print("  e.g.  dim[5] += 0.001  and  dim[12] -= 0.001")
print("  Then run:  python scripts/perturb_chat.py")
print("="*60)
