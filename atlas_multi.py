"""
Multi-model embedding atlas.
Builds one self-contained interactive HTML per model (UMAP 2D map of its token
embeddings, with semantic / UMAP / substring search). A model dropdown at the top
switches between them. Per-model results are cached so re-runs are instant.
"""
import os, json, string
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import umap
import plotly.graph_objects as go
from embed_loader import get_token_embeddings   # embeddings-only HF fetch for LLMs

# loader "st" = sentence-transformers; "llm" = embeddings-only fetch (decoder LLMs)
MODELS = [
    {"key": "minilm",        "loader": "st", "name": "sentence-transformers/all-MiniLM-L6-v2",
     "label": "MiniLM-L6 · 384d · WordPiece"},
    {"key": "mpnet",         "loader": "st", "name": "sentence-transformers/all-mpnet-base-v2",
     "label": "MPNet-base · 768d · WordPiece"},
    {"key": "distilroberta", "loader": "st", "name": "sentence-transformers/all-distilroberta-v1",
     "label": "DistilRoBERTa · 768d · BPE"},
    {"key": "multi",         "loader": "st", "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
     "label": "Multilingual-MiniLM · 384d · SentencePiece"},
    # --- frontier decoder LLMs (token-embedding tables, fetched embeddings-only) ---
    {"key": "qwen",     "loader": "llm", "name": "Qwen/Qwen2.5-0.5B",
     "label": "Qwen2.5-0.5B · 896d · BPE(tiktoken) · 152k"},
    {"key": "deepseek", "loader": "llm", "name": "deepseek-ai/deepseek-coder-6.7b-base",
     "label": "DeepSeek-Coder-6.7B · 4096d · BPE · 32k"},
    {"key": "smollm",   "loader": "llm", "name": "HuggingFaceTB/SmolLM2-1.7B",
     "label": "SmolLM2-1.7B · 2048d · BPE · 49k"},
    {"key": "pythia",   "loader": "llm", "name": "EleutherAI/pythia-410m",
     "label": "Pythia-410m · 1024d · BPE(NeoX) · 50k"},
    # --- gated (need HF token in .env + accepted license) ---
    {"key": "gemma",    "loader": "llm", "name": "google/gemma-2-2b",
     "label": "Gemma-2-2B · 2304d · SentencePiece · 256k"},
    {"key": "llama",    "loader": "llm", "name": "meta-llama/Llama-3.2-1B",
     "label": "Llama-3.2-1B · 2048d · BPE(tiktoken) · 128k"},
    {"key": "mistral",  "loader": "llm", "name": "mistralai/Mistral-7B-v0.3",
     "label": "Mistral-7B-v0.3 · 4096d · SentencePiece · 33k"},
    {"key": "gemmafull", "loader": "llm", "name": "google/gemma-2-2b", "full": True,
     "label": "Gemma-2-2B FULL · 2304d · SentencePiece · 256k"},
    # --- API model: real output embeddings for a word list (see nvidia_embed.py) ---
    {"key": "nvidia",   "loader": "api", "name": "nvidia/nv-embedqa-e5-v5",
     "label": "NVIDIA nv-embedqa-e5 · 1024d · API · 6k words"},
]

def _load_token():
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if tok:
        return tok
    import re
    if os.path.exists(".env"):
        for line in open(".env"):
            m = re.match(r'(?:export\s+)?(\w+)\s*=\s*["\']?([^"\'\s]+)', line.strip())
            if m and ("HF" in m.group(1).upper() or "HUGG" in m.group(1).upper()):
                return m.group(2)
    return None

HF_TOKEN = _load_token()
MAX_TOKENS = 40000          # cap per model (sample if vocab larger)
K = 15                      # neighbors per token
CACHE = "cache"
os.makedirs(CACHE, exist_ok=True)

PUNCT = set(string.punctuation)
def categorize(t):
    """Tokenizer-agnostic: strip ##/Ġ/▁ markers, then classify the core."""
    if (t.startswith("[") and t.endswith("]")) or (t.startswith("<") and t.endswith(">")):
        return "special"
    core = t
    for mk in ("##", "Ġ", "▁", "Ċ"):
        if core.startswith(mk):
            core = core[len(mk):]
    if not core:                                   return "other"
    if core.isdigit():                             return "number"
    if all(c in PUNCT for c in core):              return "punctuation"
    if core.isalpha() and core.isascii():          return "word"
    return "other"                                 # unicode / mixed (e.g. other scripts)

CATS = ["word", "number", "punctuation", "special", "other"]
PALETTE = {"word": "#1f77b4", "number": "#2ca02c", "punctuation": "#d62728",
           "special": "#9467bd", "other": "#8c564b"}

def build_model(m):
    """Return dict(tokens, cats, xy, nn) for a model, using cache when present."""
    key = m["key"]
    fmeta = f"{CACHE}/{key}_meta.json"
    fxy   = f"{CACHE}/{key}_xy.npy"
    fnn   = f"{CACHE}/{key}_nn.npy"
    if os.path.exists(fmeta) and os.path.exists(fxy) and os.path.exists(fnn):
        meta = json.load(open(fmeta))
        print(f"[{key}] loaded from cache ({len(meta['tokens'])} tokens)")
        return {"tokens": meta["tokens"], "cats": meta["cats"],
                "xy": np.load(fxy), "nn": np.load(fnn)}

    print(f"[{key}] loading model {m['name']} ({m['loader']}) ...")
    if m["loader"] == "api":
        emb = np.load(f"{CACHE}/{key}_emb.npy")
        tok = json.load(open(f"{CACHE}/{key}_words.json"))
        print(f"[{key}] loaded {emb.shape[0]} API word-embeddings")
    elif m["loader"] == "llm":
        emb, tok, tname = get_token_embeddings(m["name"], token=HF_TOKEN)
        print(f"[{key}] fetched embeddings-only tensor '{tname}'")
    else:
        model = SentenceTransformer(m["name"])
        emb = model[0].auto_model.get_input_embeddings().weight.detach().cpu().numpy()
        tok = model.tokenizer.convert_ids_to_tokens(range(emb.shape[0]))
        tok = [t if isinstance(t, str) else "<pad>" for t in tok]
    print(f"[{key}] full vocab: {emb.shape[0]} x {emb.shape[1]}")

    # cap / sample  (skipped when the model is flagged "full")
    if not m.get("full") and emb.shape[0] > MAX_TOKENS:
        rng = np.random.default_rng(0)
        sel = np.sort(rng.choice(emb.shape[0], MAX_TOKENS, replace=False))
        emb, tok = emb[sel], [tok[i] for i in sel]
        print(f"[{key}] sampled down to {len(tok)} tokens")

    cats = [categorize(t) for t in tok]
    big = emb.shape[0] > 60000          # large vocab -> approximate methods

    print(f"[{key}] UMAP ({emb.shape[0]} pts){' [parallel]' if big else ''} ...")
    xy = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine",
                   random_state=None if big else 42).fit_transform(emb).astype(np.float32)

    if big:                              # approximate k-NN (brute is infeasible at this scale)
        print(f"[{key}] approximate {K}-NN via pynndescent ...")
        from pynndescent import NNDescent
        index = NNDescent(emb, metric="cosine", n_neighbors=K + 1)
        idx = index.neighbor_graph[0][:, 1:].astype(np.int32)
    else:
        print(f"[{key}] {K}-NN (brute) ...")
        nn = NearestNeighbors(n_neighbors=K + 1, metric="cosine", algorithm="brute").fit(emb)
        _, idx = nn.kneighbors(emb)
        idx = idx[:, 1:].astype(np.int32)

    json.dump({"tokens": tok, "cats": cats}, open(fmeta, "w"))
    np.save(fxy, xy)
    np.save(fnn, idx)
    print(f"[{key}] cached.")
    return {"tokens": tok, "cats": cats, "xy": xy, "nn": idx}


def make_figure(data, label, ntok):
    tokens, cats, xy = data["tokens"], np.array(data["cats"]), data["xy"]
    fig = go.Figure()
    for cat in CATS:
        mk = cats == cat
        if not mk.any():
            continue
        fig.add_trace(go.Scattergl(
            x=xy[mk, 0], y=xy[mk, 1], mode="markers", name=f"{cat} ({int(mk.sum())})",
            marker=dict(size=3, color=PALETTE[cat], opacity=0.6),
            text=[f"'{tokens[i]}'" for i in np.where(mk)[0]], hoverinfo="text"))
    fig.add_trace(go.Scattergl(  # neighbors
        x=[], y=[], mode="markers+text", name="◆ neighbors",
        marker=dict(size=11, color="#ff7f0e", symbol="diamond", line=dict(width=1, color="black")),
        textposition="top center", textfont=dict(size=10, color="#7a3b00"), hoverinfo="text"))
    neighbor_trace = len(fig.data) - 1
    fig.add_trace(go.Scattergl(  # query
        x=[], y=[], mode="markers+text", name="★ query",
        marker=dict(size=18, color="red", symbol="star", line=dict(width=1.5, color="yellow")),
        textposition="top center", textfont=dict(size=12, color="darkred"), hoverinfo="text"))
    query_trace = len(fig.data) - 1
    fig.update_layout(
        title=f"{label} — {ntok} tokens (UMAP 2D)", template="plotly_white",
        legend=dict(title="token type — click to filter", itemsizing="constant"),
        width=1200, height=820)
    return fig, neighbor_trace, query_trace


TEMPLATE = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>Embedding Atlas</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 0; }}
  #bar {{ position: sticky; top: 0; background: #fff; padding: 10px 14px;
          border-bottom: 1px solid #ddd; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }}
  #q {{ font-size: 15px; padding: 6px 10px; width: 240px; border: 1px solid #bbb; border-radius: 6px; }}
  #bar button, #model {{ font-size: 14px; padding: 6px 10px; border: 1px solid #bbb;
                 border-radius: 6px; background: #f3f3f3; cursor: pointer; }}
  #status {{ color: #555; font-size: 13px; }}
  .sep {{ color:#bbb; }}
</style></head><body>
<div id="bar">
  <strong>model:</strong>
  <select id="model" onchange="window.location.href=this.value">{options}</select>
  <span class="sep">|</span>
  <input id="q" list="history" placeholder="search a token" autocomplete="off" autofocus>
  <datalist id="history"></datalist>
  <button onclick="runSearch()">Search</button>
  <button onclick="clearSearch()">Clear</button>
  <label style="font-size:13px;"><input type="radio" name="mode" value="semantic" checked> semantic</label>
  <label style="font-size:13px;"><input type="radio" name="mode" value="umap"> UMAP (map)</label>
  <label style="font-size:13px;"><input type="radio" name="mode" value="substring"> substring</label>
  <span id="status">pick a model, search a token</span>
  <a href="compare_ui.html" style="margin-left:auto;font-size:13px;color:#1f77b4;text-decoration:none;">⇄ Compare models</a>
</div>
{plot_div}
<script>
const D = {data};
const gd = document.getElementById("atlas");
const status = document.getElementById("status");
const LS = window.localStorage;   // persists across model switches (page navigations)
const IDX = {{}};
for (let i = 0; i < D.tok.length; i++) {{ const t = D.tok[i].toLowerCase(); if (!(t in IDX)) IDX[t] = i; }}
function resolveToken(q) {{
  if (q in IDX) return IDX[q];
  if (("##" + q) in IDX) return IDX["##" + q];
  for (let i = 0; i < D.tok.length; i++) if (D.tok[i].toLowerCase().includes(q)) return i;
  return -1;
}}
function setTrace(trace, idxs) {{
  const xs = idxs.map(i => D.x[i]), ys = idxs.map(i => D.y[i]), tx = idxs.map(i => D.tok[i]);
  Plotly.restyle(gd, {{x: [xs], y: [ys], text: [tx]}}, [trace]); return {{xs, ys}};
}}
function zoomTo(xs, ys) {{
  if (!xs.length) return; const pad = 2;
  Plotly.relayout(gd, {{"xaxis.range": [Math.min(...xs)-pad, Math.max(...xs)+pad],
                        "yaxis.range": [Math.min(...ys)-pad, Math.max(...ys)+pad]}});
}}
function runSearch() {{
  const q = document.getElementById("q").value.trim().toLowerCase(); if (!q) return;
  const mode = document.querySelector('input[name="mode"]:checked').value;
  pushHistory(q); LS.setItem("atlas_q", q); LS.setItem("atlas_mode", mode);
  if (mode === "substring") {{
    const hits = [];
    for (let i = 0; i < D.tok.length; i++) {{ const t = D.tok[i].toLowerCase();
      if (t === q || t === "##" + q || t.includes(q)) {{ hits.push(i); if (hits.length >= 300) break; }} }}
    setTrace(D.queryTrace, []); const {{xs, ys}} = setTrace(D.neighborTrace, hits);
    if (!hits.length) {{ status.textContent = "no match for '" + q + "'"; return; }}
    zoomTo(xs, ys); status.textContent = hits.length + " substring match(es)" + (hits.length>=300?" (first 300)":"");
    return;
  }}
  const qi = resolveToken(q);
  if (qi < 0) {{ status.textContent = "'" + q + "' is not in this model's vocabulary"; return; }}
  let neigh;
  if (mode === "umap") {{
    const qx = D.x[qi], qy = D.y[qi], dist = [];
    for (let i = 0; i < D.x.length; i++) {{ if (i===qi) continue;
      const dx = D.x[i]-qx, dy = D.y[i]-qy; dist.push([dx*dx+dy*dy, i]); }}
    dist.sort((a,b)=>a[0]-b[0]); neigh = dist.slice(0,15).map(d=>d[1]);
    setTrace(D.neighborTrace, neigh); const {{xs,ys}} = setTrace(D.queryTrace, [qi]);
    zoomTo(xs.concat(neigh.map(i=>D.x[i])), ys.concat(neigh.map(i=>D.y[i])));
    status.textContent = "'" + D.tok[qi] + "' → nearest ON THE MAP: " + neigh.slice(0,10).map(i=>D.tok[i]).join(", ");
    return;
  }}
  neigh = D.nn[qi];
  setTrace(D.neighborTrace, neigh); const {{xs,ys}} = setTrace(D.queryTrace, [qi]);
  zoomTo(xs.concat(neigh.map(i=>D.x[i])), ys.concat(neigh.map(i=>D.y[i])));
  status.textContent = "'" + D.tok[qi] + "' → nearest by meaning: " + neigh.slice(0,10).map(i=>D.tok[i]).join(", ");
}}
function clearSearch() {{
  setTrace(D.neighborTrace, []); setTrace(D.queryTrace, []);
  document.getElementById("q").value = ""; LS.removeItem("atlas_q");
  Plotly.relayout(gd, {{"xaxis.autorange": true, "yaxis.autorange": true}}); status.textContent = "cleared";
}}

// ---- search history (shared across all models via localStorage) ----
function loadHistory() {{ try {{ return JSON.parse(LS.getItem("atlas_history") || "[]"); }} catch (e) {{ return []; }} }}
function renderHistory() {{
  document.getElementById("history").innerHTML =
    loadHistory().map(function (q) {{ return '<option value="' + q.replace(/"/g, "&quot;") + '">'; }}).join("");
}}
function pushHistory(q) {{
  var h = loadHistory().filter(function (x) {{ return x !== q; }});
  h.unshift(q); h = h.slice(0, 25);
  LS.setItem("atlas_history", JSON.stringify(h)); renderHistory();
}}

document.getElementById("q").addEventListener("keydown", e => {{ if (e.key === "Enter") runSearch(); }});
document.getElementById("q").addEventListener("input", e => {{ LS.setItem("atlas_q", e.target.value); }});

// ---- restore state when a new model page loads (don't empty the search) ----
(function init() {{
  renderHistory();
  const sm = LS.getItem("atlas_mode");
  if (sm) {{ const r = document.querySelector('input[name="mode"][value="' + sm + '"]'); if (r) r.checked = true; }}
  const sq = LS.getItem("atlas_q");
  if (sq) {{ document.getElementById("q").value = sq; runSearch(); }}   // re-run on the new model
}})();
</script></body></html>"""


def main():
    built = []
    for m in MODELS:
        try:
            data = build_model(m)
            built.append((m, data))
        except Exception as e:
            print(f"[{m['key']}] SKIPPED — {type(e).__name__}: {e}")

    # model dropdown options (only models that built successfully)
    built_models = [m for m, _ in built]
    def options_for(cur):
        return "".join(
            f'<option value="atlas_{m["key"]}.html"{" selected" if m["key"]==cur else ""}>{m["label"]}</option>'
            for m in built_models)

    first_plotly = True
    for m, data in built:
        fig, nt, qt = make_figure(data, m["label"], len(data["tokens"]))
        # inline plotly.js only in the first file; others load it from that file's copy via CDN fallback
        plot_div = fig.to_html(full_html=False, include_plotlyjs=True, div_id="atlas")
        DATA = {
            "x": [round(float(v), 3) for v in data["xy"][:, 0]],
            "y": [round(float(v), 3) for v in data["xy"][:, 1]],
            "tok": data["tokens"], "nn": data["nn"].tolist(),
            "neighborTrace": nt, "queryTrace": qt,
        }
        html = TEMPLATE.format(options=options_for(m["key"]),
                               plot_div=plot_div, data=json.dumps(DATA))
        out = f"atlas_{m['key']}.html"
        with open(out, "w") as f:
            f.write(html)
        print(f"wrote {out} ({os.path.getsize(out)//1_000_000} MB)")

    print("\nDone. Open atlas_minilm.html and use the model dropdown to switch.")


if __name__ == "__main__":
    main()
