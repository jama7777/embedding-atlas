import os
import sys
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
import plotly.graph_objects as go

# ----------------------------------------------------------------------
# 1. DEFINE SENTENCES & CONFIG
# ----------------------------------------------------------------------
model_name = "HuggingFaceTB/SmolLM2-135M"
local_dir = "cache/local_smollm2_135m"
os.makedirs("html", exist_ok=True)

sentences = [
    {"text": "The muddy bank of the river was covered in moss.", "category": "River Bank (Nature)", "color": "#10b981"},
    {"text": "We sat on the river bank and watched the water flow.", "category": "River Bank (Nature)", "color": "#059669"},
    {"text": "I deposited cash into my checking account at the bank.", "category": "Financial Bank (Money)", "color": "#3b82f6"},
    {"text": "The local financial bank is offering low interest rates.", "category": "Financial Bank (Money)", "color": "#2563eb"}
]
target_word = "bank"

if os.path.exists(local_dir):
    print(f"Loading tokenizer and model OFFLINE from local folder: '{local_dir}'...")
    tokenizer = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(local_dir, output_hidden_states=True, local_files_only=True)
else:
    print(f"Loading tokenizer and model online/cached from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
model.eval()
print("Model loaded successfully!")

# ----------------------------------------------------------------------
# 2. RUN INFERENCE & EXTRACT LAYER-BY-LAYER ACTIVATIONS
# ----------------------------------------------------------------------
# We will collect the activation vector of the target word "bank" for each layer
# SmolLM2-135M has 30 layers, outputting 31 hidden states (Layer 0 = embedding output, Layers 1-30 = transformer outputs)
vectors = []
labels = []

for s_idx, sent in enumerate(sentences):
    text = sent["text"]
    inputs = tokenizer(text, return_tensors="pt")
    
    # Run forward pass (no grads)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # hidden_states is a tuple of 13 tensors of shape [batch, seq_len, hidden_dim]
    hidden_states = outputs.hidden_states
    
    # Find token index for target word "bank"
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    target_idx = -1
    for t_idx, token in enumerate(tokens):
        # Strip tokenizer markers (like Ġ or space) and convert to lowercase
        clean_token = token.replace("Ġ", "").replace(" ", "").replace("Ġ", "").lower()
        if clean_token == target_word:
            target_idx = t_idx
            break
            
    if target_idx == -1:
        print(f"Error: Target word '{target_word}' not found in sentence: '{text}'")
        print("Tokens found:", tokens)
        sys.exit(1)
        
    print(f"Sentence {s_idx + 1}: Found '{target_word}' at token index {target_idx} ('{tokens[target_idx]}')")
    
    # Extract activation vector at every layer
    for layer in range(len(hidden_states)):
        # Shape: [hidden_dim] (576 for SmolLM2)
        vec = hidden_states[layer][0, target_idx].float().cpu().numpy()
        vectors.append(vec)
        labels.append({
            "sent_idx": s_idx,
            "layer": layer,
            "category": sent["category"],
            "text": text,
            "token": tokens[target_idx]
        })

vectors = np.array(vectors, dtype=np.float32)
print(f"Collected activations matrix: {vectors.shape}")

# ----------------------------------------------------------------------
# 3. PCA DIMENSIONALITY REDUCTION
# ----------------------------------------------------------------------
print("Performing PCA reduction to 2D...")
pca = PCA(n_components=2, random_state=42)
xy = pca.fit_transform(vectors)
print("PCA complete!")

# ----------------------------------------------------------------------
# 4. BUILD INTERACTIVE PLOTLY VISUALIZATION
# ----------------------------------------------------------------------
fig = go.Figure()

# For each sentence, plot the layer-by-layer trajectory as a connected line
for s_idx, sent in enumerate(sentences):
    # Filter points belonging to this sentence
    indices = [i for i, lbl in enumerate(labels) if lbl["sent_idx"] == s_idx]
    xs = xy[indices, 0]
    ys = xy[indices, 1]
    
    # Create hover labels
    hover_texts = []
    for idx in indices:
        lbl = labels[idx]
        hover_texts.append(
            f"<b>Sentence:</b> '{lbl['text']}'<br>"
            f"<b>Category:</b> {lbl['category']}<br>"
            f"<b>Layer:</b> {lbl['layer']} / 30<br>"
            f"<b>Coordinates:</b> ({xs[idx - indices[0]]:.2f}, {ys[idx - indices[0]]:.2f})"
        )
    
    # Add line+markers trace
    fig.add_trace(go.Scatter(
        x=xs,
        y=ys,
        mode="lines+markers+text",
        name=sent["category"],
        line=dict(color=sent["color"], width=2.5),
        marker=dict(size=9, symbol="circle", line=dict(width=1, color="white")),
        text=[f"L{labels[idx]['layer']}" for idx in indices],
        textposition="top center",
        textfont=dict(size=9, color="#475569"),
        hovertext=hover_texts,
        hoverinfo="text"
    ))

# Custom Premium White Layout Styling
fig.update_layout(
    title=dict(
        text="<b>LLM Semantic Trajectory: Layer-by-Layer Word Contextualization</b><br><span style='font-size:12px;color:#64748b;font-weight:normal;'>Tracing the target token 'bank' as it gains context from Layer 0 (static embedding) to Layer 30 in SmolLM2-135M</span>",
        font=dict(size=18, color="#0f172a", family="Inter, sans-serif")
    ),
    template="plotly_white",
    width=1100,
    height=750,
    legend=dict(
        title="Context Category",
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    xaxis=dict(
        title="Principal Component 1 (Semantic Context)",
        showgrid=True,
        gridcolor="#f1f5f9",
        zeroline=False,
        showticklabels=True
    ),
    yaxis=dict(
        title="Principal Component 2 (Secondary Variance)",
        showgrid=True,
        gridcolor="#f1f5f9",
        zeroline=False,
        showticklabels=True
    ),
    margin=dict(l=50, r=50, t=120, b=50),
    hoverlabel=dict(
        bgcolor="#ffffff",
        bordercolor="#cbd5e1",
        font=dict(color="#0f172a", size=12)
    )
)

out_path = "html/llm_layer_trajectory.html"
fig.write_html(out_path, include_plotlyjs=True)
print(f"Saved interactive layer trajectory visualizer successfully -> {out_path}")
print("Experiment complete!")
