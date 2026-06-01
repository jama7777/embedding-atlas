"""
Embed a curated word list through the NVIDIA embeddings API and cache the result.
Word list = real alphabetic words from MiniLM's vocab (so it lines up with the
existing atlases for the comparison tool). Saves cache/nvidia_emb.npy + words.
"""
import os, re, json, time, urllib.request
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL = "nvidia/nv-embedqa-e5-v5"
N_WORDS = 6000          # how many words to embed
BATCH = 64
URL = "https://integrate.api.nvidia.com/v1/embeddings"

def load_key():
    for line in open(".env"):
        if "=" in line:
            k, v = line.strip().split("=", 1); v = v.strip().strip('"').strip("'")
            if v.startswith("nvapi-"):
                return v
    raise SystemExit("no nvapi- key in .env")

KEY = load_key()

# word list from MiniLM vocab (alphabetic, deduped, lowercased)
tok = SentenceTransformer("all-MiniLM-L6-v2").tokenizer
seen, words = set(), []
for t in tok.convert_ids_to_tokens(range(tok.vocab_size)):
    w = t[2:] if t.startswith("##") else t
    w = w.lower()
    if w.isalpha() and w.isascii() and len(w) >= 3 and w not in seen:
        seen.add(w); words.append(w)
    if len(words) >= N_WORDS:
        break
print(f"embedding {len(words)} words via {MODEL} ...")

def embed_batch(batch, retries=4):
    body = json.dumps({"input": batch, "model": MODEL,
                       "input_type": "passage", "encoding_format": "float"}).encode()
    req = urllib.request.Request(URL, data=body, headers={
        "Authorization": "Bearer " + KEY, "Content-Type": "application/json",
        "Accept": "application/json"})
    for attempt in range(retries):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=60))
            return [d["embedding"] for d in sorted(r["data"], key=lambda d: d["index"])]
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries - 1:
                time.sleep(2 * (attempt + 1)); continue
            raise

vecs = []
for i in range(0, len(words), BATCH):
    vecs.extend(embed_batch(words[i:i + BATCH]))
    if (i // BATCH) % 10 == 0:
        print(f"  {len(vecs)}/{len(words)}")
emb = np.asarray(vecs, dtype=np.float32)
print("done:", emb.shape)

os.makedirs("cache", exist_ok=True)
np.save("cache/nvidia_emb.npy", emb)
json.dump(words, open("cache/nvidia_words.json", "w"))
print("Saved -> cache/nvidia_emb.npy, cache/nvidia_words.json")
