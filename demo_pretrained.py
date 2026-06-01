"""
BPE tokenization -> embeddings, using PRETRAINED models only.
Two layers shown:
  (A) the raw BPE tokenizer (what text becomes integer IDs)
  (B) the embedding model (what those IDs become as vectors)
"""

# ----------------------------------------------------------------------
# (A) LOW LEVEL: see the BPE tokenizer that ships with a pretrained model
# ----------------------------------------------------------------------
from transformers import AutoTokenizer

# GPT-2 uses byte-level BPE. The tokenizer is pretrained — we just load it.
tok = AutoTokenizer.from_pretrained("gpt2")

text = "BPE tokenization to embeddings"
ids = tok.encode(text)
pieces = tok.convert_ids_to_tokens(ids)

print("TEXT   :", text)
print("BPE IDs:", ids)
print("PIECES :", pieces)        # 'Ġ' marks a leading space in GPT-2 BPE
print("DECODED:", tok.decode(ids))
print("-" * 60)

# ----------------------------------------------------------------------
# (B) HIGH LEVEL: one vector per sentence, ready for search/similarity
#     sentence-transformers handles tokenize -> model -> pooling for you.
# ----------------------------------------------------------------------
from sentence_transformers import SentenceTransformer
import numpy as np

# small, fast, pretrained embedding model (384-dim vectors)
model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "How do I split text into tokens?",
    "What is actually use of split text in tokens",
    "I love eating pizza on the weekend.",
    "My favorite food is a eating pizza.",
]
emb = model.encode(sentences, normalize_embeddings=True)  # unit vectors -> dot = cosine
print("Embedding matrix shape:", emb.shape)   # [4, 384]

sim = emb @ emb.T
print("\nCosine similarity matrix (higher = more similar):")
print("        " + "  ".join(f"s{j}" for j in range(len(sentences))))
for i, row in enumerate(sim):
    print(f"  s{i}: " + "  ".join(f"{v:+.2f}" for v in row))

print("\nNearest neighbor of each sentence:")
for i in range(len(sentences)):
    scores = sim[i].copy()
    scores[i] = -2  # exclude itself
    best = int(np.argmax(scores))
    print(f"  s{i} '{sentences[i]}'")
    print(f"       -> s{best} '{sentences[best]}'  (cos={sim[i][best]:.3f})")
