"""
Visualize all-MiniLM-L6-v2 sentence embeddings in 2D.
384-dim vectors -> PCA -> 2D scatter, colored by topic.
"""
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

# grouped into 3 topics so clustering is visible in 2D
data = {
    "tokenization": [
        "How do I split text into tokens?",
        "What is byte pair encoding in NLP?",
        "rest in piece",
        "Subword tokenizers break words into pieces.",
        "BPE merges frequent character pairs.",
    ],
    "food": [
        "I love eating pizza on the weekend.",
        "My favorite food is a cheesy pizza.",
        "This pasta recipe is delicious.",
        "We grilled burgers for dinner.",
    ],
    "weather": [
        "It is raining heavily outside today.",
        "The forecast predicts a sunny morning.",
        "A cold storm is coming this weekend.",
        "The sky is clear and bright blue.",
    ],
}

sentences = [s for group in data.values() for s in group]
labels    = [topic for topic, group in data.items() for _ in group]

emb = model.encode(sentences, normalize_embeddings=True)   # [12, 384]
print("Embeddings:", emb.shape)

# 384-dim -> 2-dim
coords = PCA(n_components=2, random_state=0).fit_transform(emb)

# --- plot ---
fig, ax = plt.subplots(figsize=(11, 8))
colors = {"tokenization": "#1f77b4", "food": "#d62728", "weather": "#2ca02c"}

for topic in data:
    pts = coords[[i for i, l in enumerate(labels) if l == topic]]
    ax.scatter(pts[:, 0], pts[:, 1], s=120, c=colors[topic], label=topic,
               edgecolors="black", linewidths=0.6, alpha=0.85)

# annotate each point with a short version of the sentence
for (x, y), sent in zip(coords, sentences):
    short = sent if len(sent) <= 32 else sent[:29] + "..."
    ax.annotate(short, (x, y), fontsize=8,
                xytext=(6, 4), textcoords="offset points")

ax.set_title("all-MiniLM-L6-v2 sentence embeddings (384-dim -> PCA 2D)", fontsize=13)
ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
ax.legend(title="topic", loc="best")
ax.grid(True, linestyle="--", alpha=0.3)
fig.tight_layout()

out = "embeddings_2d.png"
fig.savefig(out, dpi=150)
print("Saved ->", out)
