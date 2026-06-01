"""
Same tokens, three projection methods side by side: PCA vs t-SNE vs UMAP.
Uses a labeled subset of clearly different word groups so you can judge
which method separates the known clusters best.
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt

model = SentenceTransformer("all-MiniLM-L6-v2")
emb_all = model[0].auto_model.embeddings.word_embeddings.weight.detach().cpu().numpy()
tok = model.tokenizer.convert_ids_to_tokens(range(emb_all.shape[0]))
vocab = {t.lower(): i for i, t in enumerate(tok)}

# known groups (ground-truth colors) so we can SEE which method separates them
groups = {
    "royalty":  ["king","queen","prince","princess","monarch","throne","royal","kingdom"],
    "animals":  ["dog","cat","horse","cow","sheep","lion","tiger","wolf","rabbit","bear"],
    "numbers":  ["one","two","three","four","five","six","seven","eight","nine","ten"],
    "colors":   ["red","blue","green","yellow","purple","orange","pink","black","white"],
    "food":     ["pizza","pasta","bread","cheese","apple","banana","rice","soup","cake"],
    "emotions": ["happy","sad","angry","afraid","joyful","anxious","calm","excited"],
}
words, labels = [], []
for g, ws in groups.items():
    for w in ws:
        if w in vocab:
            words.append(w); labels.append(g)
X = np.stack([emb_all[vocab[w]] for w in words])
labels = np.array(labels)
print(f"{len(words)} words across {len(groups)} groups, dim={X.shape[1]}")

# three projections
pca   = PCA(n_components=2, random_state=0).fit_transform(X)
tsne  = TSNE(n_components=2, perplexity=8, random_state=0, init="pca").fit_transform(X)
ump   = umap.UMAP(n_components=2, n_neighbors=10, min_dist=0.1,
                  metric="cosine", random_state=0).fit_transform(X)

palette = {g: c for g, c in zip(groups, ["#d62728","#1f77b4","#2ca02c","#9467bd","#ff7f0e","#17becf"])}
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
for ax, (name, coords) in zip(axes, [("PCA (linear, true distances)", pca),
                                      ("t-SNE (tight local clusters)", tsne),
                                      ("UMAP (local + some global)", ump)]):
    for g in groups:
        m = labels == g
        ax.scatter(coords[m,0], coords[m,1], s=60, c=palette[g], label=g,
                   edgecolors="black", linewidths=0.4, alpha=0.85)
    for (x,y), w in zip(coords, words):
        ax.annotate(w, (x,y), fontsize=7, xytext=(3,3), textcoords="offset points")
    ax.set_title(name, fontsize=13); ax.set_xticks([]); ax.set_yticks([])
axes[0].legend(loc="best", fontsize=8)
fig.suptitle("Same 384-dim word embeddings -> three ways to flatten to 2D", fontsize=15)
fig.tight_layout()
fig.savefig("projection_comparison.png", dpi=130)
print("Saved -> projection_comparison.png")
