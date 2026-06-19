"""
Export all-MiniLM-L6-v2's token embeddings for the TensorFlow Embedding Projector
(https://projector.tensorflow.org -> 'Load' button).

Produces:
  vectors.tsv   - one row per token, 384 tab-separated floats (NO header)
  metadata.tsv  - token text + category (WITH header, since 2 columns)

In the projector: click 'Load', upload vectors.tsv as the first file and
metadata.tsv as the second. Then pick PCA / t-SNE / UMAP and rotate in 3D.
"""
import numpy as np, string, csv
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
emb = model[0].auto_model.embeddings.word_embeddings.weight.detach().cpu().numpy()
tok = model.tokenizer.convert_ids_to_tokens(range(emb.shape[0]))

PUNCT = set(string.punctuation)
def categorize(t):
    if t.startswith("[") and t.endswith("]"): return "special"
    core = t[2:] if t.startswith("##") else t
    if t.startswith("##"):                     return "subword"
    if core.isdigit():                         return "number"
    if core and all(c in PUNCT for c in core): return "punctuation"
    if core.isalpha() and core.isascii():      return "word"
    return "other"

# OPTIONAL: drop the ~999 near-empty [unusedN] rows so the view is cleaner.
keep = [i for i, t in enumerate(tok) if not t.startswith("[unused")]
print(f"Exporting {len(keep)} of {len(tok)} tokens (dropped [unused] slots)")

# vectors.tsv  -- raw 384-dim vectors, no header
np.savetxt("vectors.tsv", emb[keep], delimiter="\t", fmt="%.5f")

# metadata.tsv -- header required when there are 2+ columns
with open("metadata.tsv", "w", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["token", "category"])
    for i in keep:
        w.writerow([tok[i], categorize(tok[i])])

print("Saved -> vectors.tsv, metadata.tsv")
print("Load both at https://projector.tensorflow.org (Load button).")
