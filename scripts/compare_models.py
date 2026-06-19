"""
Compare embedding models pairwise on TWO axes:
  1) Vocabulary overlap  — shared vs distinct tokens (% match), full tokenizer vocab,
     normalized so ##the / Ġthe / ▁the / the all count as the same word.
  2) Semantic agreement  — for words both models know, how much their nearest-neighbor
     sets overlap (do they organize meaning the same way?).
Outputs: printed report, an interactive compare.html (two heatmaps), and compare_matrices.png.
"""
import os, json, itertools
import numpy as np
from transformers import AutoTokenizer
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

from atlas_multi import MODELS, HF_TOKEN, CACHE   # reuse the model registry

def normalize(t):
    """Strip tokenizer markers + lowercase, so the same word matches across families."""
    if not isinstance(t, str):
        return ""
    for mk in ("##", "Ġ", "▁", "Ċ"):
        if t.startswith(mk):
            t = t[len(mk):]
    return t.lower()

# only models that were actually built (have a cache)
built = [m for m in MODELS if os.path.exists(f"{CACHE}/{m['key']}_meta.json")]
keys   = [m["key"] for m in built]
labels = [m["label"].split(" · ")[0] for m in built]   # short names
print("comparing:", labels)

# --- per-model data: full normalized vocab (for overlap) + neighbor sets (for agreement) ---
full_vocab, nbr_sets = {}, {}
for m in built:
    k = m["key"]
    meta = json.load(open(f"{CACHE}/{k}_meta.json"))
    toks = meta["tokens"]
    full_vocab[k] = {normalize(t) for t in toks}
    full_vocab[k].discard("")

    nn = np.load(f"{CACHE}/{k}_nn.npy")
    norm = [normalize(t) for t in toks]
    ns = {}
    for i in range(len(toks)):
        nt = norm[i]
        if not nt:
            continue
        s = ns.setdefault(nt, set())
        for j in nn[i]:
            s.add(norm[j])
    nbr_sets[k] = ns
    print(f"[{k}] full vocab {len(full_vocab[k])} | sampled neighbor-words {len(ns)}")

n = len(built)
vocab_mat = np.zeros((n, n))      # % vocabulary overlap (Jaccard)
sem_mat   = np.zeros((n, n))      # % semantic agreement (mean neighbor Jaccard)
pair_stats = {}

pair_data = {}     # for the UI: keyed by "ka|kb" (sorted)
for a, b in itertools.combinations(range(n), 2):
    ka, kb = keys[a], keys[b]
    A, B = full_vocab[ka], full_vocab[kb]
    inter, union = len(A & B), len(A | B)
    voc = 100 * inter / union
    vocab_mat[a, b] = vocab_mat[b, a] = voc

    shared = set(nbr_sets[ka]) & set(nbr_sets[kb])
    scored = []      # (jaccard, word) for real alphabetic words
    jall = []
    for w in shared:
        sa, sb = nbr_sets[ka][w], nbr_sets[kb][w]
        u = len(sa | sb)
        if not u:
            continue
        jac = len(sa & sb) / u
        jall.append(jac)
        if w.isalpha() and len(w) >= 3:
            scored.append((jac, w))
    sem = 100 * (sum(jall) / len(jall)) if jall else 0.0
    sem_mat[a, b] = sem_mat[b, a] = sem
    scored.sort(reverse=True)

    pair_stats[(ka, kb)] = dict(shared_vocab=inter, only_a=len(A - B), only_b=len(B - A),
                                vocab_pct=voc, sem_pct=sem, sem_words=len(jall))
    pair_data["|".join(sorted([ka, kb]))] = dict(
        vocab=round(voc, 1), sem=round(sem, 1), sem_words=len(jall),
        shared=inter, distinct={ka: len(A - B), kb: len(B - A)},
        agree=[w for _, w in scored[:15]], disagree=[w for _, w in scored[-15:]])

np.fill_diagonal(vocab_mat, 100); np.fill_diagonal(sem_mat, 100)

# ---------- printed report ----------
print("\n================ PAIRWISE REPORT ================")
for (ka, kb), s in pair_stats.items():
    la = labels[keys.index(ka)]; lb = labels[keys.index(kb)]
    print(f"\n{la}  vs  {lb}")
    print(f"  vocabulary : {s['vocab_pct']:5.1f}% match  "
          f"(shared {s['shared_vocab']}, only-{la} {s['only_a']}, only-{lb} {s['only_b']})")
    print(f"  semantic   : {s['sem_pct']:5.1f}% neighbor agreement  (over {s['sem_words']} shared words)")

# ---------- detailed example pair: first two models ----------
if n >= 2:
    ka, kb = keys[0], keys[1]
    shared = set(nbr_sets[ka]) & set(nbr_sets[kb])
    scored = []
    for w in shared:
        sa, sb = nbr_sets[ka][w], nbr_sets[kb][w]
        u = len(sa | sb)
        if u >= 5:
            scored.append((len(sa & sb) / u, w))
    scored.sort(reverse=True)
    print(f"\n--- {labels[0]} vs {labels[1]}: words they MOST agree on ---")
    print("  ", ", ".join(w for _, w in scored[:12]))
    print(f"--- words they MOST disagree on ---")
    print("  ", ", ".join(w for _, w in scored[-12:]))

# ---------- interactive heatmaps ----------
def heat(mat, title):
    return go.Heatmap(z=mat, x=labels, y=labels, zmin=0, zmax=100, colorscale="Viridis",
                      text=[[f"{v:.0f}%" for v in row] for row in mat],
                      texttemplate="%{text}", hovertemplate="%{y} vs %{x}: %{z:.1f}%<extra></extra>",
                      colorbar=dict(title="%"))
fig = make_subplots(rows=1, cols=2, subplot_titles=("Vocabulary overlap %", "Semantic agreement %"))
fig.add_trace(heat(vocab_mat, "vocab"), 1, 1)
fig.add_trace(heat(sem_mat, "sem"), 1, 2)
fig.update_layout(title="Model comparison — vocabulary overlap vs semantic agreement",
                  width=1300, height=650, template="plotly_white")
os.makedirs("html", exist_ok=True)
fig.write_html("html/compare.html", include_plotlyjs=True)
print("\nSaved -> html/compare.html")

# ---------- static PNG ----------
f2, axes = plt.subplots(1, 2, figsize=(16, 7))
for ax, mat, t in zip(axes, [vocab_mat, sem_mat], ["Vocabulary overlap %", "Semantic agreement %"]):
    im = ax.imshow(mat, vmin=0, vmax=100, cmap="viridis")
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=8)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{mat[i,j]:.0f}", ha="center", va="center",
                    color="white" if mat[i,j] < 60 else "black", fontsize=7)
    ax.set_title(t); f2.colorbar(im, ax=ax, shrink=0.8)
f2.tight_layout(); f2.savefig("html/compare_matrices.png", dpi=130)
print("Saved -> html/compare_matrices.png")

# ---------- interactive two-dropdown comparison UI ----------
UI_DATA = {
    "models": [{"key": m["key"], "label": m["label"]} for m in built],
    "selfVocab": {m["key"]: len(full_vocab[m["key"]]) for m in built},
    "pairs": pair_data,
}
UI = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>Compare embedding models</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 0; background:#f7f7f8; color:#222; }}
  #bar {{ background:#fff; padding:14px 18px; border-bottom:1px solid #ddd; display:flex;
          gap:10px; align-items:center; flex-wrap:wrap; }}
  select {{ font-size:15px; padding:7px 10px; border:1px solid #bbb; border-radius:6px; }}
  a.atlas {{ margin-left:auto; font-size:13px; color:#1f77b4; text-decoration:none; }}
  .wrap {{ max-width:1000px; margin:24px auto; padding:0 18px; }}
  .cards {{ display:flex; gap:18px; flex-wrap:wrap; }}
  .card {{ flex:1; min-width:260px; background:#fff; border:1px solid #e3e3e6; border-radius:12px;
           padding:20px 22px; box-shadow:0 1px 3px rgba(0,0,0,.05); }}
  .pct {{ font-size:52px; font-weight:700; line-height:1; }}
  .pct small {{ font-size:18px; font-weight:600; color:#888; }}
  .lbl {{ font-size:14px; color:#666; margin-bottom:10px; text-transform:uppercase; letter-spacing:.04em; }}
  .sub {{ font-size:13px; color:#666; margin-top:12px; }}
  .barrow {{ display:flex; height:16px; border-radius:5px; overflow:hidden; margin:14px 0 4px; }}
  .barrow div {{ height:100%; }}
  .legend {{ font-size:12px; color:#666; display:flex; gap:14px; flex-wrap:wrap; }}
  .dot {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:4px; vertical-align:middle; }}
  .words {{ background:#fff; border:1px solid #e3e3e6; border-radius:12px; padding:18px 22px; margin-top:18px; }}
  .chips span {{ display:inline-block; background:#eef2f7; border-radius:12px; padding:3px 10px;
                 margin:3px; font-size:13px; }}
  h3 {{ margin:0 0 6px; font-size:15px; }}
</style></head><body>
<div id="bar">
  <strong>Compare:</strong>
  <select id="A" onchange="render()"></select>
  <span>vs</span>
  <select id="B" onchange="render()"></select>
  <a class="atlas" href="atlas_minilm.html">← back to atlas</a>
</div>
<div class="wrap">
  <div class="cards">
    <div class="card"><div class="lbl">Vocabulary overlap</div>
      <div class="pct" id="vpct">–<small>%</small></div>
      <div class="barrow" id="vbar"></div>
      <div class="legend" id="vleg"></div>
      <div class="sub" id="vsub"></div></div>
    <div class="card"><div class="lbl">Semantic agreement</div>
      <div class="pct" id="spct">–<small>%</small></div>
      <div class="sub" id="ssub"></div>
      <div class="sub" style="margin-top:18px;color:#999">How often the two models pick the
        same nearest-by-meaning neighbors for words they share.</div></div>
  </div>
  <div class="words" id="wordbox">
    <div class="cards">
      <div style="flex:1"><h3>✅ Most agree on</h3><div class="chips" id="agree"></div></div>
      <div style="flex:1"><h3>❌ Most disagree on</h3><div class="chips" id="disagree"></div></div>
    </div>
  </div>
</div>
<script>
const D = {data};
const A = document.getElementById("A"), B = document.getElementById("B");
D.models.forEach(function (m, i) {{
  A.add(new Option(m.label, m.key)); B.add(new Option(m.label, m.key));
}});
A.value = "glove"; B.value = "swaram-li1";
const labelOf = {{}}; D.models.forEach(m => labelOf[m.key] = m.label.split(" · ")[0]);

function chips(el, arr) {{
  el.innerHTML = (arr && arr.length) ? arr.map(w => "<span>" + w + "</span>").join("") : "<span style='color:#aaa'>—</span>";
}}
function render() {{
  const ka = A.value, kb = B.value;
  const la = labelOf[ka], lb = labelOf[kb];
  if (ka === kb) {{
    document.getElementById("vpct").innerHTML = "100<small>%</small>";
    document.getElementById("spct").innerHTML = "100<small>%</small>";
    document.getElementById("vbar").innerHTML = "";
    document.getElementById("vleg").innerHTML = "";
    document.getElementById("vsub").textContent = "Same model — " + D.selfVocab[ka].toLocaleString() + " tokens.";
    document.getElementById("ssub").textContent = "Same model.";
    chips(document.getElementById("agree"), []); chips(document.getElementById("disagree"), []);
    return;
  }}
  const p = D.pairs[[ka, kb].sort().join("|")];
  document.getElementById("vpct").innerHTML = p.vocab + "<small>%</small>";
  document.getElementById("spct").innerHTML = p.sem + "<small>%</small>";
  const da = p.distinct[ka], db = p.distinct[kb], sh = p.shared;
  const tot = da + db + sh;
  document.getElementById("vbar").innerHTML =
    '<div style="width:' + (100*da/tot) + '%;background:#d62728"></div>' +
    '<div style="width:' + (100*sh/tot) + '%;background:#2ca02c"></div>' +
    '<div style="width:' + (100*db/tot) + '%;background:#1f77b4"></div>';
  document.getElementById("vleg").innerHTML =
    '<span><i class="dot" style="background:#d62728"></i>only ' + la + ': ' + da.toLocaleString() + '</span>' +
    '<span><i class="dot" style="background:#2ca02c"></i>shared: ' + sh.toLocaleString() + '</span>' +
    '<span><i class="dot" style="background:#1f77b4"></i>only ' + lb + ': ' + db.toLocaleString() + '</span>';
  document.getElementById("vsub").textContent = "Jaccard overlap of full tokenizer vocabularies.";
  document.getElementById("ssub").textContent = "Based on " + p.sem_words.toLocaleString() + " shared words.";
  chips(document.getElementById("agree"), p.agree);
  chips(document.getElementById("disagree"), p.disagree.slice().reverse());
}}
render();
</script></body></html>"""
ui_content = UI.format(data=json.dumps(UI_DATA))
if keys:
    ui_content = ui_content.replace("atlas_minilm.html", f"atlas_{keys[0]}.html")

with open("html/compare_ui.html", "w") as f:
    f.write(ui_content)
print("Saved -> html/compare_ui.html")
