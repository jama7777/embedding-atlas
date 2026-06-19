"""
Atlas-style visualization of the ENTIRE embedding model.

Takes every token vector in all-MiniLM-L6-v2's vocabulary (~30k x 384),
projects to 2D with UMAP, and renders one interactive Plotly map:
  - hover a point  -> see the token text + id
  - scroll         -> zoom,  drag -> pan
  - click legend   -> filter token categories (metadata filtering)
Output: a standalone interactive_atlas.html you can open in any browser.
"""
import numpy as np
import string
from sentence_transformers import SentenceTransformer
import umap
import plotly.graph_objects as go

# ----------------------------------------------------------------------
# 1. Pull the full token-embedding matrix straight out of the model.
#    This IS the "embedding model": one learned vector per vocab token.
# ----------------------------------------------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")
bert  = model[0].auto_model                      # underlying transformer
emb   = bert.embeddings.word_embeddings.weight   # [vocab, 384] tensor
emb   = emb.detach().cpu().numpy()
tok   = model.tokenizer
vocab_size, dim = emb.shape
print(f"Entire embedding model: {vocab_size} tokens x {dim} dims")

tokens = tok.convert_ids_to_tokens(range(vocab_size))

# ----------------------------------------------------------------------
# 2. Tag each token with a category -> gives color + legend filtering.
# ----------------------------------------------------------------------
PUNCT = set(string.punctuation)
def categorize(t: str) -> str:
    if t.startswith("[") and t.endswith("]"):     return "special"   # [CLS],[PAD]...
    core = t[2:] if t.startswith("##") else t
    if t.startswith("##"):                         return "subword"   # word piece
    if core.isdigit():                             return "number"
    if core and all(c in PUNCT for c in core):     return "punctuation"
    if core.isalpha() and core.isascii():          return "word"
    return "other"                                                    # unicode/mixed

cats = np.array([categorize(t) for t in tokens])

# ----------------------------------------------------------------------
# 3. UMAP: 384-dim -> 2-dim. Cosine metric matches how these vectors are used.
#    Cache the projection so re-runs (to tweak the plot) are instant.
# ----------------------------------------------------------------------
import os
CACHE = "umap_xy.npy"
if os.path.exists(CACHE):
    xy = np.load(CACHE)
    print("Loaded cached projection:", xy.shape)
else:
    print("Running UMAP (this takes ~30-60s the first time)...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
    xy = reducer.fit_transform(emb)              # [vocab, 2]
    np.save(CACHE, xy)
    print("Projection done & cached:", xy.shape)

# ----------------------------------------------------------------------
# 4. Interactive Plotly map (WebGL handles ~30k points smoothly).
# ----------------------------------------------------------------------
palette = {
    "word": "#1f77b4", "subword": "#ff7f0e", "number": "#2ca02c",
    "punctuation": "#d62728", "special": "#9467bd", "other": "#8c564b",
}
fig = go.Figure()
for cat in ["word", "subword", "number", "punctuation", "special", "other"]:
    m = cats == cat
    if not m.any():
        continue
    fig.add_trace(go.Scattergl(
        x=xy[m, 0], y=xy[m, 1],
        mode="markers",
        name=f"{cat} ({int(m.sum())})",
        marker=dict(size=3, color=palette[cat], opacity=0.6),
        text=[f"'{tokens[i]}'  (id {i})" for i in np.where(m)[0]],
        hoverinfo="text",
    ))

# ----------------------------------------------------------------------
# 4b. Precompute k-nearest-neighbors in the FULL 384-dim space (cosine).
#     This powers semantic search: a token's true meaning-neighbors,
#     not just substring matches. Cached so re-runs are instant.
# ----------------------------------------------------------------------
from sklearn.neighbors import NearestNeighbors
K = 15
NN_CACHE = "knn.npy"
if os.path.exists(NN_CACHE):
    nn_idx = np.load(NN_CACHE)
    print("Loaded cached k-NN:", nn_idx.shape)
else:
    print(f"Computing {K}-NN over {vocab_size} tokens (cosine, ~1-2 min)...")
    nn = NearestNeighbors(n_neighbors=K + 1, metric="cosine", algorithm="brute")
    nn.fit(emb)
    _, nn_idx = nn.kneighbors(emb)      # [vocab, K+1], first col is self
    nn_idx = nn_idx[:, 1:].astype(np.int32)
    np.save(NN_CACHE, nn_idx)
    print("k-NN done & cached:", nn_idx.shape)

# Two empty highlight traces the search box fills in:
#   neighbors (orange) + the query token itself (red).
fig.add_trace(go.Scattergl(
    x=[], y=[], mode="markers+text", name="◆ neighbors",
    marker=dict(size=11, color="#ff7f0e", symbol="diamond",
                line=dict(width=1, color="black")),
    textposition="top center", textfont=dict(size=10, color="#7a3b00"),
    hoverinfo="text",
))
NEIGHBOR_TRACE = len(fig.data) - 1
fig.add_trace(go.Scattergl(
    x=[], y=[], mode="markers+text", name="★ query",
    marker=dict(size=18, color="red", symbol="star",
                line=dict(width=1.5, color="yellow")),
    textposition="top center", textfont=dict(size=12, color="darkred"),
    hoverinfo="text",
))
QUERY_TRACE = len(fig.data) - 1

fig.update_layout(
    title=f"all-MiniLM-L6-v2 — entire embedding model ({vocab_size} tokens, UMAP 2D)",
    template="plotly_white",
    legend=dict(title="token type — click to filter", itemsizing="constant"),
    width=1200, height=850,
)

# ----------------------------------------------------------------------
# 5. Wrap the plot in custom HTML with a client-side search box.
# ----------------------------------------------------------------------
import json
plot_div = fig.to_html(full_html=False, include_plotlyjs=True, div_id="atlas")

# embed coords + token text + neighbor table for JS (round coords to shrink size)
DATA = {
    "x": [round(float(v), 3) for v in xy[:, 0]],
    "y": [round(float(v), 3) for v in xy[:, 1]],
    "tok": tokens,
    "nn": nn_idx.tolist(),               # [vocab, K] neighbor indices
    "neighborTrace": NEIGHBOR_TRACE,
    "queryTrace": QUERY_TRACE,
}

html = """<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Embedding Atlas</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 0; }}
  #bar {{ position: sticky; top: 0; background: #fff; padding: 10px 14px;
          border-bottom: 1px solid #ddd; display: flex; gap: 8px; align-items: center; }}
  #q {{ font-size: 15px; padding: 6px 10px; width: 280px;
        border: 1px solid #bbb; border-radius: 6px; }}
  #bar button {{ font-size: 14px; padding: 6px 12px; border: 1px solid #bbb;
                 border-radius: 6px; background: #f3f3f3; cursor: pointer; }}
  #status {{ color: #555; font-size: 13px; }}
</style></head><body>
<div id="bar">
  <input id="q" placeholder="search a token, e.g.  king   pizza   2024" autofocus>
  <button onclick="runSearch()">Search</button>
  <button onclick="clearSearch()">Clear</button>
  <label style="font-size:13px;"><input type="radio" name="mode" value="semantic" checked> semantic (meaning neighbors)</label>
  <label style="font-size:13px;"><input type="radio" name="mode" value="umap"> UMAP (map neighbors)</label>
  <label style="font-size:13px;"><input type="radio" name="mode" value="substring"> substring (text match)</label>
  <span id="status">type a token and press Enter</span>
</div>
{plot_div}
<script>
const D = {data};
const gd = document.getElementById("atlas");
const status = document.getElementById("status");

// build exact-lookup map: lowercased token -> first index
const IDX = {{}};
for (let i = 0; i < D.tok.length; i++) {{
  const t = D.tok[i].toLowerCase();
  if (!(t in IDX)) IDX[t] = i;
}}

function resolveToken(q) {{          // query string -> a vocab index (or -1)
  if (q in IDX) return IDX[q];
  if (("##" + q) in IDX) return IDX["##" + q];
  for (let i = 0; i < D.tok.length; i++)
    if (D.tok[i].toLowerCase().includes(q)) return i;
  return -1;
}}

function setTrace(trace, idxs) {{     // fill a highlight trace from token indices
  const xs = idxs.map(i => D.x[i]);
  const ys = idxs.map(i => D.y[i]);
  const tx = idxs.map(i => D.tok[i]);
  Plotly.restyle(gd, {{x: [xs], y: [ys], text: [tx]}}, [trace]);
  return {{xs, ys}};
}}

function zoomTo(xs, ys) {{
  if (!xs.length) return;
  const pad = 2;
  Plotly.relayout(gd, {{
    "xaxis.range": [Math.min(...xs) - pad, Math.max(...xs) + pad],
    "yaxis.range": [Math.min(...ys) - pad, Math.max(...ys) + pad],
  }});
}}

function runSearch() {{
  const q = document.getElementById("q").value.trim().toLowerCase();
  if (!q) return;
  const mode = document.querySelector('input[name="mode"]:checked').value;

  if (mode === "substring") {{
    const hits = [];
    for (let i = 0; i < D.tok.length; i++) {{
      const t = D.tok[i].toLowerCase();
      if (t === q || t === "##" + q || t.includes(q)) {{ hits.push(i); if (hits.length >= 300) break; }}
    }}
    setTrace(D.queryTrace, []);                      // clear query marker
    const {{xs, ys}} = setTrace(D.neighborTrace, hits);
    if (!hits.length) {{ status.textContent = "no match for '" + q + "'"; return; }}
    zoomTo(xs, ys);
    status.textContent = hits.length + " substring match(es)" + (hits.length >= 300 ? " (first 300)" : "");
    return;
  }}

  // resolve query to a vocab token (shared by semantic + umap modes)
  const qi = resolveToken(q);
  if (qi < 0) {{ status.textContent = "'" + q + "' is not in the vocabulary"; return; }}

  let neigh;
  if (mode === "umap") {{
    // UMAP mode: 15 tokens physically CLOSEST on the 2D map (Euclidean in x,y).
    const qx = D.x[qi], qy = D.y[qi];
    const dist = [];
    for (let i = 0; i < D.x.length; i++) {{
      if (i === qi) continue;
      const dx = D.x[i] - qx, dy = D.y[i] - qy;
      dist.push([dx * dx + dy * dy, i]);
    }}
    dist.sort((a, b) => a[0] - b[0]);
    neigh = dist.slice(0, 15).map(d => d[1]);
    setTrace(D.neighborTrace, neigh);
    const {{xs, ys}} = setTrace(D.queryTrace, [qi]);
    zoomTo(xs.concat(neigh.map(i => D.x[i])), ys.concat(neigh.map(i => D.y[i])));
    status.textContent = "'" + D.tok[qi] + "' → nearest ON THE MAP: " +
                         neigh.slice(0, 10).map(i => D.tok[i]).join(", ");
    return;
  }}

  // semantic mode: highlight its meaning-neighbors (full 384-dim cosine)
  neigh = D.nn[qi];
  setTrace(D.neighborTrace, neigh);
  const {{xs, ys}} = setTrace(D.queryTrace, [qi]);
  zoomTo(xs.concat(neigh.map(i => D.x[i])), ys.concat(neigh.map(i => D.y[i])));
  status.textContent = "'" + D.tok[qi] + "' → nearest by meaning: " +
                       neigh.slice(0, 10).map(i => D.tok[i]).join(", ");
}}

function clearSearch() {{
  setTrace(D.neighborTrace, []);
  setTrace(D.queryTrace, []);
  Plotly.relayout(gd, {{"xaxis.autorange": true, "yaxis.autorange": true}});
  status.textContent = "cleared";
}}

document.getElementById("q").addEventListener("keydown", e => {{
  if (e.key === "Enter") runSearch();
}});
</script></body></html>"""

out = "interactive_atlas.html"
with open(out, "w") as f:
    f.write(html.format(plot_div=plot_div, data=json.dumps(DATA)))
print("Saved ->", out)

# Static PNG preview (matplotlib) — quick way to eyeball the whole map.
import matplotlib.pyplot as plt
fig2, ax = plt.subplots(figsize=(12, 9))
for cat in ["word", "subword", "number", "punctuation", "special", "other"]:
    m = cats == cat
    if m.any():
        ax.scatter(xy[m, 0], xy[m, 1], s=2, c=palette[cat],
                   label=f"{cat} ({int(m.sum())})", alpha=0.5, linewidths=0)
ax.set_title(f"all-MiniLM-L6-v2 — entire embedding model ({vocab_size} tokens, UMAP 2D)")
ax.legend(markerscale=4, loc="best", fontsize=9)
ax.set_xticks([]); ax.set_yticks([])
fig2.tight_layout()
fig2.savefig("atlas_preview.png", dpi=130)
print("Saved -> atlas_preview.png")
