"""
Embeddings-only loader: fetch ONLY the token-embedding tensor from a model's
safetensors on the HF Hub via HTTP range requests — no full-model download.
Returns (embeddings ndarray [vocab, dim], token strings).
"""
import json, struct, urllib.request
import numpy as np
import torch
from huggingface_hub import hf_hub_url, hf_hub_download, list_repo_files
from transformers import AutoTokenizer

# candidate names for the input embedding tensor across architectures
CANDIDATES = [
    "model.embed_tokens.weight",     # Llama/Qwen/Gemma/Mistral/DeepSeek/SmolLM
    "gpt_neox.embed_in.weight",      # Pythia / GPT-NeoX
    "transformer.wte.weight",        # GPT-2 family
    "embed_tokens.weight",
    "tok_embeddings.weight",
    "model.embed_in.weight",
    "wte.weight",
]
ST_DTYPE = {"F64": torch.float64, "F32": torch.float32,
            "F16": torch.float16, "BF16": torch.bfloat16}

def _range(url, start, end, token):
    req = urllib.request.Request(url)
    req.add_header("Range", f"bytes={start}-{end-1}")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req) as r:
        return r.read()

def _read_header(url, token):
    n = struct.unpack("<Q", _range(url, 0, 8, token))[0]   # u64 header length
    hdr = json.loads(_range(url, 8, 8 + n, token))
    return hdr, 8 + n                                       # header, data start offset

def _locate(repo, token):
    """Return (filename, tensor_name) for the embedding tensor."""
    files = list_repo_files(repo, token=token)
    st_files = [f for f in files if f.endswith(".safetensors")]
    if not st_files:
        raise RuntimeError(f"{repo} has no .safetensors (only {[f for f in files if f.endswith('.bin')][:2]}…)")

    if "model.safetensors.index.json" in files:            # sharded -> use index
        idx = hf_hub_download(repo, "model.safetensors.index.json", token=token)
        wm = json.load(open(idx))["weight_map"]
        for c in CANDIDATES:
            if c in wm:
                return wm[c], c
        for k in wm:
            if "embed" in k and k.endswith("weight"):
                return wm[k], k

    for fn in st_files:                                    # scan single/each file's header
        hdr, _ = _read_header(hf_hub_url(repo, fn), token)
        for c in CANDIDATES:
            if c in hdr:
                return fn, c
        for k in hdr:
            if "embed" in k and k.endswith("weight"):
                return fn, k
    raise RuntimeError(f"no embedding tensor found in {repo}")

def get_token_embeddings(repo, token=None):
    fn, name = _locate(repo, token)
    url = hf_hub_url(repo, fn)
    hdr, data_start = _read_header(url, token)
    meta = hdr[name]
    dt, shape = ST_DTYPE[meta["dtype"]], meta["shape"]
    b0, b1 = meta["data_offsets"]
    raw = _range(url, data_start + b0, data_start + b1, token)
    emb = torch.frombuffer(bytearray(raw), dtype=dt).reshape(shape).float().numpy()

    tok = AutoTokenizer.from_pretrained(repo, token=token, trust_remote_code=True)
    toks = tok.convert_ids_to_tokens(range(emb.shape[0]))
    toks = [t if isinstance(t, str) else "<pad>" for t in toks]
    return emb, toks, name


if __name__ == "__main__":   # quick self-test on a tiny open model
    import sys
    repo = sys.argv[1] if len(sys.argv) > 1 else "EleutherAI/pythia-410m"
    emb, toks, name = get_token_embeddings(repo)
    print(f"{repo}: tensor={name} shape={emb.shape}")
    print("sample tokens:", toks[:10])
