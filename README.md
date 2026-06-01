# Embedding Atlas

Interactive visualization and comparison of **token/word embeddings** across many
models and tokenizer families — built from scratch to *see* how different models
organize meaning.

Each model's embeddings are projected to 2D with **UMAP** and rendered as an
interactive [Plotly](https://plotly.com/) map you can pan, zoom, filter, and search.

## Features

- **Whole-model atlas** — every token in a model's vocabulary plotted in 2D, colored by token type.
- **Three search modes** per atlas:
  - `semantic` — nearest neighbors in the full embedding space (true meaning, cosine k-NN)
  - `UMAP` — nearest neighbors on the 2D map (shows projection distortion)
  - `substring` — text match
- **Search history + state** that persists across model switches (localStorage).
- **Model comparison** (`compare_ui.html`) — pick any two models, see **vocabulary
  overlap %** and **semantic agreement %** live, plus agree/disagree word lists.
- **Multiple embedding sources**:
  - Sentence-transformers: MiniLM, MPNet, DistilRoBERTa, Multilingual-MiniLM
  - Decoder LLMs (token-embedding tables, fetched *embeddings-only* via safetensors range requests): Qwen2.5, DeepSeek-Coder, SmolLM2, Pythia, Gemma-2, Mistral-7B
  - API models: NVIDIA `nv-embedqa-e5-v5` (real output embeddings of a word list)

## Tokenizer families covered

| Family | Models |
|---|---|
| WordPiece | MiniLM, MPNet, Multilingual-MiniLM |
| BPE | DistilRoBERTa, Qwen, DeepSeek, SmolLM2, Pythia |
| SentencePiece | Gemma-2, Mistral-7B |

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` (gitignored) with the keys you need:

```
# for gated HF models (Gemma, Llama, Mistral) — accept each model's license first
HF_token=hf_xxxxxxxxxxxxxxxxxxxxx
# for the NVIDIA API model
NVIDIA_API=nvapi-xxxxxxxxxxxxxxxxxxxxx
```

> Gated models also require accepting the license on each model's Hugging Face page
> with the same account the token belongs to.

## Usage

```bash
# 1. Build all model atlases + comparison (downloads models on first run, caches to cache/)
python3 atlas_multi.py          # -> atlas_*.html, one per model
python3 nvidia_embed.py         # embed a word list via NVIDIA API -> cache
python3 compare_models.py       # -> compare.html, compare_ui.html, compare_matrices.png

# then open any atlas in a browser and use the model dropdown
open atlas_minilm.html
```

Other standalone demos:

```bash
python3 demo_pretrained.py        # text -> tokens -> embeddings, similarity
python3 train_bpe_compare.py      # how the training corpus changes BPE token IDs
python3 frozen_tokenizer_demo.py  # why a tokenizer is frozen once a model trains on it
python3 compare_projections.py    # PCA vs t-SNE vs UMAP side by side
python3 export_projector.py       # export TSVs for projector.tensorflow.org
```

## How it works

```
text → [tokenizer: BPE / WordPiece / SentencePiece] → token IDs
     → [embedding table: vocab × dim] → vectors
     → UMAP (dim → 2D) → interactive map
```

For decoder LLMs, [`embed_loader.py`](embed_loader.py) fetches **only the embedding
tensor** from the model's safetensors via HTTP range requests — no full-model download.

## Notes

- Large vocabularies are sampled to 40k tokens for the map; large models use
  approximate k-NN (pynndescent).
- All generated artifacts (`*.html`, `*.npy`, `*.tsv`, `cache/`) are gitignored and
  rebuilt by the scripts.

🤖 Built interactively with [Claude Code](https://claude.com/claude-code).
