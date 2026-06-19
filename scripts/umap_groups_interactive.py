"""
Interactive 2D UMAP of labeled word groups.
Same idea as the winning UMAP panel from the comparison, but hoverable:
hover a point to read the word; colored by its known group.
"""
import numpy as np
from sentence_transformers import SentenceTransformer
import umap
import plotly.graph_objects as go

model = SentenceTransformer("all-MiniLM-L6-v2")
emb_all = model[0].auto_model.embeddings.word_embeddings.weight.detach().cpu().numpy()
tok = model.tokenizer.convert_ids_to_tokens(range(emb_all.shape[0]))
vocab = {t.lower(): i for i, t in enumerate(tok)}

groups = {
    "royalty":  ["king","queen","prince","princess","monarch","throne","royal","kingdom","duke","emperor"],
    "animals":  ["dog","cat","horse","cow","sheep","lion","tiger","wolf","rabbit","bear","elephant","mouse"],
    "numbers":  ["one","two","three","four","five","six","seven","eight","nine","ten","hundred","thousand"],
    "colors":   ["red","blue","green","yellow","purple","orange","pink","black","white","brown","gray"],
    "food":     ["pizza","pasta","bread","cheese","apple","banana","rice","soup","cake","burger","salad"],
    "emotions": ["happy","sad","angry","afraid","joyful","anxious","calm","excited","nervous","proud"],
}
words, labels = [], []
for g, ws in groups.items():
    for w in ws:
        if w in vocab:
            words.append(w); labels.append(g)
X = np.stack([emb_all[vocab[w]] for w in words])
labels = np.array(labels)
print(f"{len(words)} words across {len(groups)} groups")

xy = umap.UMAP(n_components=2, n_neighbors=10, min_dist=0.1,
               metric="cosine", random_state=0).fit_transform(X)

palette = dict(zip(groups, ["#d62728","#1f77b4","#2ca02c","#9467bd","#ff7f0e","#17becf"]))
fig = go.Figure()
for g in groups:
    m = labels == g
    fig.add_trace(go.Scatter(
        x=xy[m,0], y=xy[m,1], mode="markers+text",
        name=g, marker=dict(size=12, color=palette[g], line=dict(width=0.5, color="black")),
        text=[words[i] for i in np.where(m)[0]],
        textposition="top center", textfont=dict(size=9),
        hovertext=[f"{words[i]}  ({g})" for i in np.where(m)[0]], hoverinfo="text",
    ))
fig.update_layout(
    title="Interactive UMAP (2D) — word groups in all-MiniLM-L6-v2",
    template="plotly_white", width=1100, height=800,
    legend=dict(title="group — click to filter", itemsizing="constant"),
)
import os
os.makedirs("html", exist_ok=True)
out = "html/umap_groups.html"
fig.write_html(out, include_plotlyjs=True)
print("Saved ->", out)
