"""
4-way Embedding Perturbation Lab with Steering & Backend Autocomplete
──────────────────────────────────────────────────────────────────────
Column A : Original Qwen2.5-0.5B (unmodified)
Column B : Perturbed  +0.0001 on dim[0] and dim[1] of token ' the' (id=279)
Column C : Perturbed  +0.001  on dim[0] and dim[1] of token ' the' (id=279)
Column D : Perturbed  +0.01   on dim[0] and dim[1] of token ' the' (id=279)

Same input · seed=42 · temperature=0.0 (greedy) for all four.
Interactive steering: predicts next token candidates and allows the user
to manually select token paths, step-auto, or backend auto-complete.

Usage:
    cd /Users/m4_pro/Desktop/jama/embedding-atlas
    PYTHONUNBUFFERED=1 /Users/m4_pro/Desktop/jama/ent-slm/.venv/bin/python scripts/perturb_chat.py
Then open http://localhost:8765
"""

import os, sys, json, torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import http.server, socketserver

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MODEL_DIR        = "cache/local_qwen_0.5b"
PERTURB_TOKEN_ID = 279  # ' the' (space + the)
PERTURB_DIMS     = [0, 1]  # which 2 dimensions to change

# Four variants: original, small, medium, large deltas
DELTAS = {
    "orig":   [0.0,    0.0   ],
    "small":  [+0.0001,+0.0001],
    "medium": [+0.001, +0.001],
    "large":  [+0.01,  +0.01 ],
}

MAX_NEW_TOKENS = 200
SEED           = 42
PORT           = 8765
# ─────────────────────────────────────────────────────────────────────────────

print(f"\nDevice: cpu")

# ── Load model ONCE ───────────────────────────────────────────────────────────
print(f"\nLoading Qwen from {MODEL_DIR} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model     = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True)
model.eval()
print(f"  ✓ Loaded  dtype={model.dtype}")

# ── Clone embedding weight for each perturbed variant ─────────────────────────
print(f"\nBuilding perturbed embedding clones ...")
base_emb = model.get_input_embeddings().weight  # shape [151936, 896], bfloat16

patch_info = {}
emb_weights = {}
for name, deltas in DELTAS.items():
    w = base_emb.clone().detach()
    with torch.no_grad():
        for dim, delta in zip(PERTURB_DIMS, deltas):
            orig_val = float(w[PERTURB_TOKEN_ID, dim])
            w[PERTURB_TOKEN_ID, dim] += delta
            new_val  = float(w[PERTURB_TOKEN_ID, dim])
            if name not in patch_info:
                patch_info[name] = []
            patch_info[name].append({
                "dim": dim, "orig": orig_val, "new": new_val, "delta": delta
            })
    emb_weights[name] = w
    tag = f"Δ={deltas[0]:+.4f}" if deltas[0] != 0 else "unmodified"
    print(f"  [{name:6s}] {tag}")

# ── Thin wrapper that swaps embedding weight for generate() ───────────────────
class PerturbedModel:
    def __init__(self, base_model, emb_weight):
        self._model = base_model
        self._emb_w = emb_weight

    def generate(self, *args, **kwargs):
        orig = self._model.get_input_embeddings().weight.data
        self._model.get_input_embeddings().weight.data = self._emb_w
        try:
            return self._model.generate(*args, **kwargs)
        finally:
            self._model.get_input_embeddings().weight.data = orig

models = {
    "orig":   PerturbedModel(model, emb_weights["orig"]),
    "small":  PerturbedModel(model, emb_weights["small"]),
    "medium": PerturbedModel(model, emb_weights["medium"]),
    "large":  PerturbedModel(model, emb_weights["large"]),
}
print(f"\n✓ All 4 variants ready.\n")

# ── Repetition Penalty Helper ────────────────────────────────────────────────
def apply_repetition_penalty(logits, input_ids, penalty=1.1):
    unique_ids = torch.unique(input_ids[0])
    for uid in unique_ids:
        uid_item = uid.item()
        if uid_item < len(logits):
            val = logits[uid_item].item()
            if val < 0:
                logits[uid_item] = val * penalty
            else:
                logits[uid_item] = val / penalty
    return logits

# ── Single step helper ────────────────────────────────────────────────────────
def step_variant(variant: str, history: list[dict], gen_ids: list[int], orig_probs_list: list[torch.Tensor] = None) -> tuple[dict, list[torch.Tensor]]:
    torch.manual_seed(SEED)
    text = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(text, return_tensors="pt")["input_ids"]  # Shape [1, prompt_len]
    
    if gen_ids:
        gen_ids_tensor = torch.tensor([gen_ids], dtype=torch.long)
        input_ids = torch.cat([prompt_ids, gen_ids_tensor], dim=1)
    else:
        input_ids = prompt_ids
        
    input_len = prompt_ids.shape[1]
    attention_mask = torch.ones_like(input_ids)
    
    with torch.no_grad():
        orig = model.get_input_embeddings().weight.data
        model.get_input_embeddings().weight.data = emb_weights[variant]
        try:
            outputs = model(input_ids, attention_mask=attention_mask)
            # Logits for all prediction steps (from prompt-end to the next token candidates)
            pred_logits = outputs.logits[0, input_len - 1 :, :]
        finally:
            model.get_input_embeddings().weight.data = orig
            
    # Apply repetition penalty to all step logits
    for j in range(pred_logits.shape[0]):
        step_input_ids = input_ids[:, : input_len + j]
        pred_logits[j] = apply_repetition_penalty(pred_logits[j], step_input_ids, penalty=1.1)
        
    # Softmax for all steps
    pred_probs = torch.softmax(pred_logits.float(), dim=-1) # Shape: [L + 1, vocab_size]
    
    # Calculate metadata for all generated tokens so far
    steps_metadata = []
    for j in range(len(gen_ids)):
        selected_id = gen_ids[j]
        probs_j = pred_probs[j]
        selected_prob = probs_j[selected_id].item()
        
        # Get original probability
        if orig_probs_list is not None and j < len(orig_probs_list):
            orig_prob = orig_probs_list[j][selected_id].item()
        else:
            orig_prob = selected_prob if variant == "orig" else 0.0
            
        delta = selected_prob - orig_prob
        
        # Compute top 20 candidates for this specific step j
        top_probs_j, top_ids_j = torch.topk(probs_j, 20)
        top_20_j = []
        for p, idx in zip(top_probs_j.tolist(), top_ids_j.tolist()):
            raw_tok = tokenizer.decode([idx])
            disp_tok = raw_tok.replace(" ", "␣").replace("\n", "\\n").replace("\t", "\\t")
            if not disp_tok:
                disp_tok = f"<{idx}>"
                
            if orig_probs_list is not None and j < len(orig_probs_list):
                c_orig_prob = orig_probs_list[j][idx].item()
            else:
                c_orig_prob = p if variant == "orig" else 0.0
            c_delta = p - c_orig_prob
            
            top_20_j.append({
                "token": disp_tok,
                "id": idx,
                "prob": p,
                "orig_prob": c_orig_prob,
                "delta": c_delta
            })
            
        steps_metadata.append({
            "token": tokenizer.decode([selected_id]),
            "id": selected_id,
            "prob": selected_prob,
            "orig_prob": orig_prob,
            "delta": delta,
            "top_20": top_20_j
        })
        
    # Get alternative candidates for the next step (last row in pred_probs)
    next_probs = pred_probs[-1]
    top_probs, top_ids = torch.topk(next_probs, 20)
    
    top_20 = []
    for p, idx in zip(top_probs.tolist(), top_ids.tolist()):
        raw_tok = tokenizer.decode([idx])
        disp_tok = raw_tok.replace(" ", "␣").replace("\n", "\\n").replace("\t", "\\t")
        if not disp_tok:
            disp_tok = f"<{idx}>"
            
        if orig_probs_list is not None and len(gen_ids) < len(orig_probs_list):
            c_orig_prob = orig_probs_list[len(gen_ids)][idx].item()
        else:
            c_orig_prob = p if variant == "orig" else 0.0
        c_delta = p - c_orig_prob
        
        top_20.append({
            "token": disp_tok,
            "id": idx,
            "prob": p,
            "orig_prob": c_orig_prob,
            "delta": c_delta
        })
        
    is_eos = False
    if gen_ids and gen_ids[-1] == tokenizer.eos_token_id:
        is_eos = True
        
    res_payload = {
        "steps_metadata": steps_metadata,
        "top_20": top_20,
        "is_eos": is_eos
    }
    
    return res_payload, pred_probs

# ── Complete sequence generation helper ───────────────────────────────────────
def complete_variant(variant: str, history: list[dict], gen_ids: list[int]) -> list[int]:
    torch.manual_seed(SEED)
    text = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(text, return_tensors="pt")["input_ids"]
    
    if gen_ids:
        gen_ids_tensor = torch.tensor([gen_ids], dtype=torch.long)
        input_ids = torch.cat([prompt_ids, gen_ids_tensor], dim=1)
    else:
        input_ids = prompt_ids
        
    if gen_ids and gen_ids[-1] == tokenizer.eos_token_id:
        return gen_ids
        
    remaining_tokens = MAX_NEW_TOKENS - len(gen_ids)
    if remaining_tokens <= 0:
        return gen_ids
        
    attention_mask = torch.ones_like(input_ids)
    
    with torch.no_grad():
        orig = model.get_input_embeddings().weight.data
        model.get_input_embeddings().weight.data = emb_weights[variant]
        try:
            out = model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=remaining_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )
        finally:
            model.get_input_embeddings().weight.data = orig
            
    input_len = prompt_ids.shape[1]
    full_gen_ids = out[0][input_len:].tolist()
    return full_gen_ids

# ── Build INFO payload (served to UI) ────────────────────────────────────────
INFO = json.dumps({
    "token":      tokenizer.decode([PERTURB_TOKEN_ID]),
    "token_id":   PERTURB_TOKEN_ID,
    "dims":       PERTURB_DIMS,
    "variants": {
        "orig":   {"label": "Original",  "delta": DELTAS["orig"],   "color": "#38bdf8"},
        "small":  {"label": "+0.0001",   "delta": DELTAS["small"],  "color": "#a78bfa"},
        "medium": {"label": "+0.001",    "delta": DELTAS["medium"], "color": "#fb923c"},
        "large":  {"label": "+0.01",     "delta": DELTAS["large"],  "color": "#f472b6"},
    },
    "patch_info": patch_info,
    "seed": SEED,
})

# ── UI HTML ───────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Embedding Perturbation Lab — 4-way</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg:       #0d0f14;
    --surf:     #161922;
    --surf2:    #1e2230;
    --border:   #2a2f3e;
    --accent:   #6c63ff;
    --text:     #e4e7f0;
    --muted:    #8b92a9;
    --c-orig:   #38bdf8;
    --c-small:  #a78bfa;
    --c-medium: #fb923c;
    --c-large:  #f472b6;
  }

  html, body { height: 100%; overflow: hidden; }
  body {
    font-family: 'Inter', sans-serif;
    background: var(--bg);
    color: var(--text);
    display: flex;
    flex-direction: column;
  }

  /* HEADER */
  header {
    flex-shrink: 0;
    padding: 14px 24px;
    background: linear-gradient(135deg,#1a1d2e,#0d0f14);
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
  }
  header h1 {
    font-size: 17px; font-weight: 700;
    background: linear-gradient(135deg,#6c63ff,#ff6584);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
  }
  .badge {
    font-size: 11px; padding: 3px 10px; border-radius: 99px; font-weight: 600;
  }
  .badge-orig   { background:rgba(56,189,248,.12);   color:var(--c-orig);   border:1px solid rgba(56,189,248,.3); }
  .badge-small  { background:rgba(167,139,250,.12);  color:var(--c-small);  border:1px solid rgba(167,139,250,.3); }
  .badge-medium { background:rgba(251,146,60,.12);   color:var(--c-medium); border:1px solid rgba(251,146,60,.3); }
  .badge-large  { background:rgba(244,114,182,.12);  color:var(--c-large);  border:1px solid rgba(244,114,182,.3); }
  .chip {
    margin-left: auto;
    font-size: 11px; color: var(--muted);
    background: var(--surf2); border: 1px solid var(--border);
    border-radius: 8px; padding: 5px 12px;
    font-family: 'JetBrains Mono', monospace;
  }

  /* 4-COLUMN GRID */
  main {
    flex: 1; min-height: 0;
    display: grid;
    grid-template-columns: repeat(4, 1fr);
  }
  .col {
    display: flex; flex-direction: column;
    border-right: 1px solid var(--border);
    min-height: 0;
  }
  .col:last-child { border-right: none; }

  .col-hdr {
    flex-shrink: 0;
    padding: 10px 16px;
    background: var(--surf);
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 9px;
  }
  .dot { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
  .dot-orig   { background:var(--c-orig);   box-shadow:0 0 7px rgba(56,189,248,.5); }
  .dot-small  { background:var(--c-small);  box-shadow:0 0 7px rgba(167,139,250,.5); }
  .dot-medium { background:var(--c-medium); box-shadow:0 0 7px rgba(251,146,60,.5); }
  .dot-large  { background:var(--c-large);  box-shadow:0 0 7px rgba(244,114,182,.5); }
  .col-hdr .lbl { font-size: 13px; font-weight: 600; }
  .col-hdr .sub { font-size: 11px; color: var(--muted); margin-left: 2px; }

  /* MESSAGES */
  .msgs {
    flex: 1; overflow-y: auto; padding: 16px;
    display: flex; flex-direction: column; gap: 12px;
    scrollbar-width: thin; scrollbar-color: var(--border) transparent;
  }
  .msg {
    border-radius: 11px; padding: 10px 14px;
    font-size: 13.5px; line-height: 1.6;
    max-width: 97%; animation: fadeUp .2s ease;
  }
  @keyframes fadeUp { from{opacity:0;transform:translateY(5px)} to{opacity:1;transform:none} }
  .msg.user   { background:rgba(108,99,255,.15); border:1px solid rgba(108,99,255,.25); align-self:flex-end; color:#c8c5ff; }
  .msg.orig   { background:rgba(56,189,248,.07);   border:1px solid rgba(56,189,248,.18); align-self:flex-start; }
  .msg.small  { background:rgba(167,139,250,.07);  border:1px solid rgba(167,139,250,.18); align-self:flex-start; }
  .msg.medium { background:rgba(251,146,60,.07);   border:1px solid rgba(251,146,60,.18);  align-self:flex-start; }
  .msg.large  { background:rgba(244,114,182,.07);  border:1px solid rgba(244,114,182,.18); align-self:flex-start; }
  .msg.thinking {
    background:var(--surf2); border:1px dashed var(--border);
    align-self:flex-start; color:var(--muted); font-style:italic;
    display:flex; align-items:center; gap:9px;
  }
  .spin {
    width:14px; height:14px; border-radius:50%; flex-shrink:0;
    border:2px solid var(--border); border-top-color:var(--accent);
    animation:spin .6s linear infinite;
  }
  @keyframes spin { to{transform:rotate(360deg)} }

  /* INTERACTIVE TOKEN SPANS */
  .tok-span {
    transition: all 0.15s ease;
    border-radius: 3px;
    padding: 1px 0;
  }
  .tok-span.interactive-mode {
    cursor: pointer;
  }
  .tok-span.interactive-mode:hover {
    background: rgba(108, 99, 255, 0.22);
    text-shadow: 0 0 1px currentColor;
  }
  .tok-span.selected {
    background: rgba(108, 99, 255, 0.45) !important;
    font-weight: 500;
  }
  .tok-low-conf {
    border-bottom: 1px dashed rgba(248, 113, 113, 0.7);
    background: rgba(248, 113, 113, 0.04);
  }

  /* Probability Shift Highlights */
  .tok-prob-up {
    border-bottom: 2px solid rgba(74, 222, 128, 0.7);
    background: rgba(74, 222, 128, 0.06);
  }
  .tok-prob-down {
    border-bottom: 2px solid rgba(251, 146, 60, 0.7);
    background: rgba(251, 146, 60, 0.06);
  }

  /* Blinking cursor placeholder */
  .cursor-placeholder {
    display: inline-block;
    width: 2px;
    height: 14px;
    background: var(--accent);
    margin-left: 2px;
    vertical-align: middle;
    animation: blink 1s step-end infinite;
  }
  @keyframes blink {
    from, to { background-color: transparent }
    50% { background-color: var(--accent) }
  }

  /* PERSISTENT TOKEN INSPECTOR CARD */
  .inspector-card {
    margin-top: 12px;
    background: rgba(0, 0, 0, 0.35);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px;
    font-size: 12px;
    animation: fadeIn 0.2s ease;
  }
  .inspector-hdr {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid var(--border);
    padding-bottom: 6px;
    margin-bottom: 8px;
    color: var(--muted);
  }
  .inspector-title {
    font-weight: 600;
    color: var(--text);
  }
  .inspector-token-badge {
    background: var(--surf2);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 2px 6px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
  }
  .inspector-list {
    max-height: 180px;
    overflow-y: auto;
    scrollbar-width: thin;
  }
  .prob-row {
    display: flex;
    align-items: center;
    font-size: 11px;
    margin-bottom: 4px;
    font-family: 'JetBrains Mono', monospace;
  }
  .prob-row:last-child {
    margin-bottom: 0;
  }
  .prob-row.clickable {
    cursor: pointer;
    border-radius: 4px;
    padding: 2px 4px;
    transition: background 0.15s ease;
  }
  .prob-row.clickable:hover {
    background: rgba(108, 99, 255, 0.12);
  }
  .prob-tok {
    width: 90px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    color: var(--text);
  }
  .prob-bar-container {
    flex: 1;
    height: 8px;
    background: var(--surf2);
    border-radius: 4px;
    margin: 0 8px;
    overflow: hidden;
  }
  .prob-bar {
    height: 100%;
    border-radius: 4px;
    transition: width 0.3s ease;
  }
  .prob-bar.orig   { background: var(--c-orig); }
  .prob-bar.small  { background: var(--c-small); }
  .prob-bar.medium { background: var(--c-medium); }
  .prob-bar.large  { background: var(--c-large); }
  
  .prob-pct {
    width: 45px;
    text-align: right;
    color: var(--muted);
  }
  .prob-row.top-choice .prob-tok {
    font-weight: 600;
  }
  .prob-row.top-choice .prob-pct {
    color: var(--text);
    font-weight: 600;
  }
  .prob-row.selected-tok {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 4px;
    padding: 1px 4px;
  }

  /* Delta Badges in Candidate List */
  .delta-badge {
    font-size: 9px;
    padding: 1px 5px;
    border-radius: 3px;
    font-weight: 600;
    margin-left: 8px;
    display: inline-block;
  }
  .delta-badge-pos {
    background: rgba(74, 222, 128, 0.15);
    color: #4ade80;
    border: 1px solid rgba(74, 222, 128, 0.25);
  }
  .delta-badge-neg {
    background: rgba(251, 146, 60, 0.15);
    color: #fb923c;
    border: 1px solid rgba(251, 146, 60, 0.25);
  }

  .multi-prob-container {
    display: flex;
    gap: 6px;
    margin-left: auto;
    align-items: center;
  }
  .prob-badge {
    font-size: 10px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    padding: 2px 6px;
    border-radius: 4px;
    min-width: 45px;
    text-align: center;
  }
  .prob-badge.orig {
    background: rgba(56, 189, 248, 0.12);
    color: var(--c-orig);
    border: 1px solid rgba(56, 189, 248, 0.25);
  }
  .prob-badge.small {
    background: rgba(167, 139, 250, 0.12);
    color: var(--c-small);
    border: 1px solid rgba(167, 139, 250, 0.25);
  }
  .prob-badge.medium {
    background: rgba(251, 146, 60, 0.12);
    color: var(--c-medium);
    border: 1px solid rgba(251, 146, 60, 0.25);
  }
  .prob-badge.large {
    background: rgba(244, 114, 182, 0.12);
    color: var(--c-large);
    border: 1px solid rgba(244, 114, 182, 0.25);
  }

  /* STEP NAVIGATOR BAR */
  #navigator {
    flex-shrink: 0;
    display: none;
    align-items: center;
    justify-content: center;
    gap: 15px;
    padding: 10px 24px;
    border-top: 1px solid var(--border);
    background: rgba(22, 25, 34, 0.8);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    user-select: none;
  }
  #navigator.show {
    display: flex;
  }
  #nav-label {
    font-size: 13px;
    font-weight: 600;
    color: var(--text);
    min-width: 140px;
    text-align: center;
    font-family: 'JetBrains Mono', monospace;
  }
  .nav-btn {
    background: var(--surf2);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 6px 14px;
    border-radius: 8px;
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
  }
  .nav-btn:hover:not(:disabled) {
    background: var(--border);
    border-color: var(--accent);
    box-shadow: 0 0 10px rgba(108, 99, 255, 0.2);
  }
  .nav-btn:disabled {
    opacity: 0.3;
    cursor: not-allowed;
    filter: none !important;
    box-shadow: none !important;
  }
  #stepBtn {
    background: linear-gradient(135deg, #10b981, #059669);
    border: none;
    font-weight: 600;
    color: #fff;
    box-shadow: 0 0 10px rgba(16, 185, 129, 0.2);
  }
  #stepBtn:hover:not(:disabled) {
    filter: brightness(1.1);
    box-shadow: 0 0 14px rgba(16, 185, 129, 0.45);
  }
  #completeBtn {
    background: linear-gradient(135deg, #6c63ff, #8b5cf6);
    border: none;
    font-weight: 600;
    color: #fff;
    box-shadow: 0 0 10px rgba(108, 99, 255, 0.25);
  }
  #completeBtn:hover:not(:disabled) {
    filter: brightness(1.1);
    box-shadow: 0 0 14px rgba(108, 99, 255, 0.5);
  }

  /* FOOTER */
  footer {
    flex-shrink: 0;
    padding: 12px 20px;
    background: var(--surf);
    border-top: 1px solid var(--border);
    display: flex; gap: 10px; align-items: flex-end;
  }
  textarea {
    flex: 1; background: var(--surf2); color: var(--text);
    border: 1px solid var(--border); border-radius: 10px;
    padding: 9px 13px; font-family: 'Inter', sans-serif; font-size: 13.5px;
    resize: none; outline: none; min-height: 42px; max-height: 110px;
    transition: border-color .2s;
  }
  textarea:focus { border-color: var(--accent); }
  .btn { padding: 9px 18px; border-radius: 10px; font-size: 13.5px; font-weight: 600;
         cursor: pointer; border: none; transition: all .2s; white-space: nowrap; }
  .btn-send {
    background: linear-gradient(135deg,#6c63ff,#8b5cf6); color:#fff;
    box-shadow: 0 0 14px rgba(108, 99, 255, 0.3);
  }
  .btn-send:hover:not(:disabled) { filter:brightness(1.1); transform:translateY(-1px); }
  .btn-send:disabled { opacity:.4; cursor:not-allowed; transform:none; }
  .btn-clear { background:var(--surf2); color:var(--muted); border:1px solid var(--border); }
  .btn-clear:hover { background:var(--border); color:var(--text); }

  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
</style>
</head>
<body>

<header>
  <h1>🔬 Embedding Perturbation Lab</h1>
  <span class="badge badge-orig">● Original</span>
  <span class="badge badge-small">● +0.0001</span>
  <span class="badge badge-medium">● +0.001</span>
  <span class="badge badge-large">● +0.01</span>
  <div class="chip" id="chipInfo">loading…</div>
</header>

<main>
  <div class="col" id="col-orig">
    <div class="col-hdr">
      <div class="dot dot-orig"></div>
      <span class="lbl" style="color:var(--c-orig)">Original</span>
      <span class="sub">unmodified weights</span>
    </div>
    <div class="msgs" id="msgs-orig"></div>
  </div>
  <div class="col" id="col-small">
    <div class="col-hdr">
      <div class="dot dot-small"></div>
      <span class="lbl" style="color:var(--c-small)">Perturbed +0.0001</span>
      <span class="sub">dim[0,1] += 0.0001</span>
    </div>
    <div class="msgs" id="msgs-small"></div>
  </div>
  <div class="col" id="col-medium">
    <div class="col-hdr">
      <div class="dot dot-medium"></div>
      <span class="lbl" style="color:var(--c-medium)">Perturbed +0.001</span>
      <span class="sub">dim[0,1] += 0.001</span>
    </div>
    <div class="msgs" id="msgs-medium"></div>
  </div>
  <div class="col" id="col-large">
    <div class="col-hdr">
      <div class="dot dot-large"></div>
      <span class="lbl" style="color:var(--c-large)">Perturbed +0.01</span>
      <span class="sub">dim[0,1] += 0.01</span>
    </div>
    <div class="msgs" id="msgs-large"></div>
  </div>
</main>

<div id="navigator">
  <button class="nav-btn" id="backBtn" onclick="stepBack()">◀ Step Back</button>
  <span id="nav-label">Step 0 / 200</span>
  <button class="nav-btn" id="stepBtn" onclick="stepAll()">Next Token (All 4) ▶</button>
  <button class="nav-btn" id="completeBtn" onclick="completeAll()">Complete Response (Auto) ✦</button>
  <span style="color: var(--border); margin: 0 4px;">|</span>
  <span style="font-size: 11px; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">View Mode:</span>
  <button class="nav-btn" id="viewTextBtn" style="background: var(--accent); border-color: var(--accent);" onclick="setViewMode('text')">Text</button>
  <button class="nav-btn" id="viewProbBtn" onclick="setViewMode('prob')">Prob %</button>
  <button class="nav-btn" id="viewDeltaBtn" onclick="setViewMode('delta')">Delta Δ%</button>
</div>

<footer>
  <textarea id="inp" rows="1" placeholder="Type a message… (Enter to send, Shift+Enter for newline)"></textarea>
  <button class="btn btn-clear" onclick="clearAll()">Clear</button>
  <button class="btn btn-send" id="sendBtn" onclick="send()">Send ✦</button>
</footer>

<script>
const COLS = ['orig','small','medium','large'];
const els  = {};
COLS.forEach(k => { els[k] = document.getElementById('msgs-' + k); });
const sendBtn = document.getElementById('sendBtn');

let history = [];
let generatedIds = { orig: [], small: [], medium: [], large: [] };
let activeThinkingSpans = {};
let isEOS = { orig: false, small: false, medium: false, large: false };
let isCompleting = false;
let latestCandidates = { orig: null, small: null, medium: null, large: null };
let viewMode = 'text'; // 'text', 'prob', 'delta'
let isAutocompleteModeGlobal = false;
let selectedStepIdx = -1; // -1 means we are at the end (generation mode)

function formatPct(val) {
  if (val >= 0.001) {
    return (val * 100).toFixed(1) + '%';
  }
  return '—';
}

function getProbForVariant(variant, idx, tid, tstr) {
  const placeholder = activeThinkingSpans[variant];
  if (!placeholder || !placeholder.stepsMetadata || !placeholder.stepsMetadata[idx]) return 0;
  
  const step = placeholder.stepsMetadata[idx];
  let found = step.top_20.find(c => c.id === tid);
  if (found) return found.prob;
  
  found = step.top_20.find(c => c.token === tstr);
  if (found) return found.prob;
  
  return 0;
}

function selectStepAllColumns(idx) {
  selectedStepIdx = idx;
  updateGlobalStepControls();
  
  COLS.forEach(c => {
    const placeholder = activeThinkingSpans[c];
    if (!placeholder) return;
    
    // Highlight the span in column c
    const spans = placeholder.spanContainer.querySelectorAll('.tok-span');
    spans.forEach((sp, i) => {
      if (i === idx) sp.classList.add('selected');
      else sp.classList.remove('selected');
    });
    
    // Check if column c reached this step
    const step = placeholder.stepsMetadata ? placeholder.stepsMetadata[idx] : null;
    if (!step) {
      placeholder.inspectorCard.innerHTML = '<div style="color:var(--muted); font-style:italic; padding: 4px 0">This column did not reach this step.</div>';
      return;
    }
    
    const tokenDisplay = step.token.replace(" ", "␣");
    
    // Draw the inspector header
    placeholder.inspectorCard.innerHTML = `
      <div class="inspector-hdr">
        <span class="inspector-title">📊 Step ${idx + 1} candidates</span>
        <span class="inspector-token-badge" title="Token ID: ${step.id}">"${tokenDisplay}" (id=${step.id})</span>
      </div>
      <div class="inspector-list"></div>
    `;
    
    const listContainer = placeholder.inspectorCard.querySelector('.inspector-list');
    
    // Build union of candidates for this step
    const candidateMap = new Map();
    COLS.forEach(col => {
      const p = activeThinkingSpans[col];
      if (p && p.stepsMetadata && p.stepsMetadata[idx]) {
        p.stepsMetadata[idx].top_20.forEach(item => {
          if (!candidateMap.has(item.id)) {
            candidateMap.set(item.id, { id: item.id, token: item.token });
          }
        });
      }
    });
    
    const unionCandidates = Array.from(candidateMap.values());
    unionCandidates.forEach(cand => {
      cand.probs = {};
      COLS.forEach(col => {
        cand.probs[col] = getProbForVariant(col, idx, cand.id, cand.token);
      });
      cand.maxProb = Math.max(...COLS.map(col => cand.probs[col]));
    });
    unionCandidates.sort((a, b) => b.maxProb - a.maxProb);
    
    // Render the union candidates
    unionCandidates.forEach(cand => {
      const isSelected = step && cand.id === step.id;
      const row = document.createElement('div');
      row.className = 'prob-row clickable' + (isSelected ? ' selected-tok' : '');
      
      const orig_pct = formatPct(cand.probs['orig']);
      const small_pct = formatPct(cand.probs['small']);
      const medium_pct = formatPct(cand.probs['medium']);
      const large_pct = formatPct(cand.probs['large']);
      
      row.innerHTML = `
        <span class="prob-tok" title="${cand.token}">${cand.token}</span>
        <div class="multi-prob-container">
          <span class="prob-badge orig" title="Original">${orig_pct}</span>
          <span class="prob-badge small" title="+0.0001">${small_pct}</span>
          <span class="prob-badge medium" title="+0.001">${medium_pct}</span>
          <span class="prob-badge large" title="+0.01">${large_pct}</span>
        </div>
      `;
      
      // Enforce selection ONLY for column c on click!
      row.onclick = async () => {
        if (isCompleting) return;
        
        // Truncate generatedIds for column c up to the clicked step (idx)
        if (generatedIds[c].length > idx) {
          generatedIds[c] = generatedIds[c].slice(0, idx);
          generatedIds[c].push(cand.id);
          isEOS[c] = false;
        }
        
        // Truncate other columns to step idx + 1 to keep them in sync and trigger regeneration
        COLS.forEach(col => {
          if (col !== c) {
            if (generatedIds[col].length > idx) {
              generatedIds[col] = generatedIds[col].slice(0, idx + 1);
              isEOS[col] = false;
            }
          }
        });
        
        selectedStepIdx = -1; // Reset selection index after a candidate click!
        
        if (isAutocompleteModeGlobal) {
          await completeAll();
        } else {
          await fetchNextStep();
        }
      };
      
      listContainer.appendChild(row);
    });
  });
}

function showLatestCandidatesGlobal() {
  selectedStepIdx = -1;
  COLS.forEach(c => {
    const placeholder = activeThinkingSpans[c];
    if (placeholder) {
      const spans = placeholder.spanContainer.querySelectorAll('.tok-span');
      spans.forEach(sp => sp.classList.remove('selected'));
    }
  });
  showNextTokenCandidates();
  updateGlobalStepControls();
}

function showNextTokenCandidates() {
  COLS.forEach(k => {
    const placeholder = activeThinkingSpans[k];
    if (!placeholder) return;
    
    placeholder.inspectorCard.innerHTML = '';
    
    if (isEOS[k]) {
      placeholder.inspectorCard.innerHTML = '<div style="color:var(--muted); font-style:italic; padding: 4px 0">Reached End of Text (EOS)</div>';
      return;
    }
    
    const idx = generatedIds[k].length;
    
    placeholder.inspectorCard.innerHTML = `
      <div class="inspector-hdr">
        <span class="inspector-title">📊 Candidates for Step ${idx + 1}</span>
      </div>
      <div class="inspector-list"></div>
    `;
    
    const listContainer = placeholder.inspectorCard.querySelector('.inspector-list');
    
    if (placeholder.latestTop20) {
      placeholder.latestTop20.forEach(item => {
        const row = document.createElement('div');
        row.className = 'prob-row clickable';
        
        // Compute probabilities for other columns if they are at the same length idx
        const probs = {};
        COLS.forEach(col => {
          if (generatedIds[col].length === idx && activeThinkingSpans[col] && activeThinkingSpans[col].latestTop20) {
            const found = activeThinkingSpans[col].latestTop20.find(c => c.id === item.id);
            probs[col] = found ? found.prob : 0;
          } else {
            probs[col] = 0;
          }
        });
        
        const orig_pct = formatPct(probs['orig']);
        const small_pct = formatPct(probs['small']);
        const medium_pct = formatPct(probs['medium']);
        const large_pct = formatPct(probs['large']);
        
        row.innerHTML = `
          <span class="prob-tok" title="${item.token}">${item.token}</span>
          <div class="multi-prob-container">
            <span class="prob-badge orig" title="Original">${orig_pct}</span>
            <span class="prob-badge small" title="+0.0001">${small_pct}</span>
            <span class="prob-badge medium" title="+0.001">${medium_pct}</span>
            <span class="prob-badge large" title="+0.01">${large_pct}</span>
          </div>
        `;
        
        row.onclick = async () => {
          if (isCompleting) return;
          
          if (!isEOS[k]) {
            generatedIds[k].push(item.id);
          }
          isAutocompleteModeGlobal = false; // Reset to manual stepping when choosing a candidate manually
          await fetchNextStep();
        };
        
        listContainer.appendChild(row);
      });
    }
  });
}

function renderTokensAndBind(k, colData, isAutocomplete) {
  const placeholder = activeThinkingSpans[k];
  placeholder.spanContainer.innerHTML = '';
  placeholder.stepsMetadata = colData.steps_metadata; // Save steps metadata
  
  colData.steps_metadata.forEach((s, idx) => {
    const span = document.createElement('span');
    span.className = 'tok-span interactive-mode';
    
    // Save metadata on dataset for dynamic toggling
    span.dataset.token = s.token;
    span.dataset.prob = s.prob;
    span.dataset.origProb = s.orig_prob;
    span.dataset.delta = s.delta;
    span.dataset.variant = k;
    
    const pPct = (s.prob * 100).toFixed(1) + '%';
    const oPct = (s.orig_prob * 100).toFixed(1) + '%';
    const dSign = s.delta >= 0 ? '+' : '';
    const dPct = dSign + (s.delta * 100).toFixed(1) + '%';
    
    if (k !== 'orig') {
      span.title = `Probability: ${pPct} (Original: ${oPct}, Delta: ${dPct})`;
      if (s.delta >= 0.02) {
        span.classList.add('tok-prob-up');
      } else if (s.delta <= -0.02) {
        span.classList.add('tok-prob-down');
      }
    } else {
      span.title = `Probability: ${pPct}`;
    }
    
    if (s.prob < 0.4 && !span.classList.contains('tok-prob-up') && !span.classList.contains('tok-prob-down')) {
      span.classList.add('tok-low-conf');
    }
    
    // Format label according to active view mode
    let label = s.token;
    if (viewMode === 'prob') {
      label += ` (${(s.prob * 100).toFixed(0)}%)`;
    } else if (viewMode === 'delta') {
      if (k !== 'orig') {
        label += ` (${dSign}${(s.delta * 100).toFixed(0)}%)`;
      }
    }
    span.textContent = label;
    
    // Click handler to select this step for all columns
    span.onclick = () => {
      selectStepAllColumns(idx);
    };
    
    placeholder.spanContainer.appendChild(span);
  });
}


fetch('/info').then(r=>r.json()).then(d=>{
  document.getElementById('chipInfo').innerHTML =
    `token <strong style="color:#e4e7f0">"${d.token}"</strong> &nbsp;·&nbsp; ` +
    `dim[${d.dims.join(',')}] &nbsp;·&nbsp; seed=${d.seed}`;
});

function addMsg(col, text, cls) {
  const d = document.createElement('div');
  d.className = 'msg ' + cls;
  d.textContent = text;
  els[col].appendChild(d);
  els[col].scrollTop = els[col].scrollHeight;
  return d;
}

function addThinkingPlaceholder(col) {
  const d = document.createElement('div');
  d.className = 'msg ' + col;
  
  const spanContainer = document.createElement('div');
  spanContainer.className = 'span-container';
  spanContainer.style.whiteSpace = 'pre-wrap';
  spanContainer.style.wordBreak = 'break-word';
  
  const cursor = document.createElement('span');
  cursor.className = 'cursor-placeholder';
  
  const inspectorCard = document.createElement('div');
  inspectorCard.className = 'inspector-card';
  inspectorCard.innerHTML = '<div style="color:var(--muted)">Predicting first token…</div>';
  
  d.appendChild(spanContainer);
  d.appendChild(cursor);
  d.appendChild(inspectorCard);
  
  els[col].appendChild(d);
  els[col].scrollTop = els[col].scrollHeight;
  return {
    bubble: d,
    spanContainer: spanContainer,
    cursor: cursor,
    inspectorCard: inspectorCard
  };
}

async function selectToken(variant, tokenId) {
  if (isEOS[variant] || isCompleting) return;
  generatedIds[variant].push(tokenId);
  await fetchNextStep();
}

async function stepAll() {
  if (isCompleting) return;
  
  if (selectedStepIdx !== -1) {
    // Navigate inspection highlight forward
    const currentMaxLen = Math.max(...COLS.map(k => generatedIds[k].length));
    if (selectedStepIdx < currentMaxLen - 1) {
      selectStepAllColumns(selectedStepIdx + 1);
    } else {
      showLatestCandidatesGlobal();
    }
    return;
  }
  
  isAutocompleteModeGlobal = false; // Reset to manual stepping mode when stepping manually
  
  let advanced = false;
  COLS.forEach(k => {
    if (!isEOS[k] && generatedIds[k].length < 200 && latestCandidates[k]) {
      generatedIds[k].push(latestCandidates[k].id);
      advanced = true;
    }
  });
  
  if (advanced) {
    await fetchNextStep();
  }
}

async function stepBack() {
  if (isCompleting) return;
  
  if (selectedStepIdx !== -1) {
    // Navigate inspection highlight backward
    if (selectedStepIdx > 0) {
      selectStepAllColumns(selectedStepIdx - 1);
    } else {
      showLatestCandidatesGlobal();
    }
    return;
  }
  
  isAutocompleteModeGlobal = false; // Reset to manual stepping mode when stepping manually
  
  let backed = false;
  COLS.forEach(k => {
    if (generatedIds[k].length > 0) {
      generatedIds[k].pop();
      isEOS[k] = false; // Reset EOS state since we stepped back
      backed = true;
    }
  });
  
  if (backed) {
    await fetchNextStep();
  }
}

async function completeAll() {
  if (isCompleting) return;
  
  if (selectedStepIdx !== -1) {
    // Truncate all columns to selectedStepIdx + 1 before completing
    COLS.forEach(k => {
      if (generatedIds[k].length > selectedStepIdx) {
        generatedIds[k] = generatedIds[k].slice(0, selectedStepIdx + 1);
        isEOS[k] = false;
      }
    });
    selectedStepIdx = -1;
  }
  
  if (isAllEOS()) return;
  
  isAutocompleteModeGlobal = true;
  isCompleting = true;
  document.getElementById('backBtn').disabled = true;
  document.getElementById('stepBtn').disabled = true;
  document.getElementById('completeBtn').disabled = true;
  sendBtn.disabled = true;
  
  // Show autocomplete status
  COLS.forEach(k => {
    if (!isEOS[k]) {
      activeThinkingSpans[k].inspectorCard.innerHTML = '<div style="color:var(--muted); font-style:italic">Generating remaining sequence…</div>';
    }
  });
  
  try {
    const res = await fetch('/complete', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        history: history,
        generated: generatedIds
      })
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    
    // Update generatedIds and render completed lists
    COLS.forEach(k => {
      const colData = data[k];
      generatedIds[k] = colData.steps_metadata.map(s => s.id);
      isEOS[k] = colData.is_eos;
      
      const placeholder = activeThinkingSpans[k];
      placeholder.cursor.style.display = 'none'; // hide cursor on autocomplete
      
      renderTokensAndBind(k, colData, true);
      
      placeholder.latestTop20 = colData.top_20;
      
      if (colData.top_20 && colData.top_20.length > 0) {
        latestCandidates[k] = colData.top_20[0];
      } else {
        latestCandidates[k] = null;
      }
    });

    // Automatically select the last token of the sequence to show candidates below the chat
    const lastIdx = Math.max(...COLS.map(k => generatedIds[k].length)) - 1;
    if (lastIdx >= 0) {
      selectStepAllColumns(lastIdx);
    } else {
      COLS.forEach(k => {
        activeThinkingSpans[k].inspectorCard.innerHTML = '<div style="color:var(--muted); font-style:italic; padding: 4px 0">Generation completed. Click any word to inspect candidates.</div>';
      });
    }
    
  } catch(e) {
    COLS.forEach(k => {
      activeThinkingSpans[k].inspectorCard.innerHTML = `<span style="color:#f87171">⚠ Autocomplete Error: ${e.message}</span>`;
    });
  }
  
  isCompleting = false;
  updateGlobalStepControls();
  sendBtn.disabled = false;
}

function isAllEOS() {
  return COLS.every(k => isEOS[k]);
}

function updateGlobalStepControls() {
  const currentMaxLen = Math.max(...COLS.map(k => generatedIds[k].length));
  
  if (selectedStepIdx !== -1) {
    document.getElementById('nav-label').textContent = `Inspecting Step ${selectedStepIdx + 1} / ${currentMaxLen}`;
    document.getElementById('backBtn').disabled = false;
    document.getElementById('stepBtn').disabled = false;
    document.getElementById('completeBtn').disabled = false;
  } else {
    document.getElementById('nav-label').textContent = `Step ${currentMaxLen} / 200`;
    document.getElementById('backBtn').disabled = (currentMaxLen === 0);
    
    const allFinished = COLS.every(k => isEOS[k] || generatedIds[k].length >= 200);
    document.getElementById('stepBtn').disabled = allFinished;
    document.getElementById('completeBtn').disabled = allFinished;
  }
  
  if (history.length > 0) {
    document.getElementById('navigator').classList.add('show');
  } else {
    document.getElementById('navigator').classList.remove('show');
  }
}

async function fetchNextStep() {
  try {
    const res = await fetch('/step', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        history: history,
        generated: generatedIds
      })
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    
    // Update state and render
    COLS.forEach(k => {
      const colData = data[k];
      const placeholder = activeThinkingSpans[k];
      
      isEOS[k] = colData.is_eos;
      
      // Toggle cursor visibility
      placeholder.cursor.style.display = isEOS[k] ? 'none' : 'inline-block';
      
      renderTokensAndBind(k, colData, false);
      
      placeholder.latestTop20 = colData.top_20;
      
      // Save top candidate for auto-stepping
      if (colData.top_20 && colData.top_20.length > 0) {
        latestCandidates[k] = colData.top_20[0];
      } else {
        latestCandidates[k] = null;
      }
    });
    
    showNextTokenCandidates();
    
    // Update global controls
    updateGlobalStepControls();
    sendBtn.disabled = false; // Always re-enable Send button so they can enter a new prompt
    
    // Auto-scroll
    COLS.forEach(k => { els[k].scrollTop = els[k].scrollHeight; });
    
  } catch(e) {
    COLS.forEach(k => {
      activeThinkingSpans[k].inspectorCard.innerHTML = `<span style="color:#f87171">⚠ Error: ${e.message}</span>`;
    });
    sendBtn.disabled = false;
  }
}

async function send() {
  const text = document.getElementById('inp').value.trim();
  if (!text || sendBtn.disabled) return;

  // Append previous assistant response if it exists, to preserve chat context
  const origContainer = activeThinkingSpans['orig'] ? activeThinkingSpans['orig'].spanContainer : null;
  if (origContainer) {
    const prevText = origContainer.textContent;
    history.push({role: 'assistant', content: prevText});
  }

  history.push({role:'user', content: text});
  document.getElementById('inp').value = '';
  autoResize();

  COLS.forEach(k => addMsg(k, '👤 ' + text, 'user'));
  
  // Reset steering states
  generatedIds = { orig: [], small: [], medium: [], large: [] };
  isEOS = { orig: false, small: false, medium: false, large: false };
  isCompleting = false;
  isAutocompleteModeGlobal = false;
  latestCandidates = { orig: null, small: null, medium: null, large: null };
  selectedStepIdx = -1;
  
  // Create message bubbles
  activeThinkingSpans = {};
  COLS.forEach(k => {
    activeThinkingSpans[k] = addThinkingPlaceholder(k);
  });

  sendBtn.disabled = true;
  document.getElementById('navigator').classList.remove('show');

  // Trigger step 1
  await fetchNextStep();
}

function clearAll() {
  history = [];
  generatedIds = { orig: [], small: [], medium: [], large: [] };
  isEOS = { orig: false, small: false, medium: false, large: false };
  isCompleting = false;
  latestCandidates = { orig: null, small: null, medium: null, large: null };
  activeThinkingSpans = {};
  selectedStepIdx = -1;
  COLS.forEach(k => { els[k].innerHTML = ''; });
  document.getElementById('navigator').classList.remove('show');
  sendBtn.disabled = false;
}

function autoResize() {
  const t = document.getElementById('inp');
  t.style.height = 'auto';
  t.style.height = Math.min(t.scrollHeight, 110) + 'px';
}
document.getElementById('inp').addEventListener('input', autoResize);
document.getElementById('inp').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});

function setViewMode(mode) {
  viewMode = mode;
  
  // Update button highlights
  const btns = {
    text: document.getElementById('viewTextBtn'),
    prob: document.getElementById('viewProbBtn'),
    delta: document.getElementById('viewDeltaBtn')
  };
  
  Object.keys(btns).forEach(k => {
    if (k === mode) {
      btns[k].style.background = 'var(--accent)';
      btns[k].style.borderColor = 'var(--accent)';
    } else {
      btns[k].style.background = 'var(--surf2)';
      btns[k].style.borderColor = 'var(--border)';
    }
  });
  
  updateSpanTexts();
}

function updateSpanTexts() {
  const spans = document.querySelectorAll('.tok-span');
  spans.forEach(span => {
    const rawToken = span.dataset.token;
    const prob = parseFloat(span.dataset.prob);
    const delta = parseFloat(span.dataset.delta);
    const variant = span.dataset.variant;
    if (!rawToken) return;
    
    let label = rawToken;
    if (viewMode === 'prob') {
      label += ` (${(prob * 100).toFixed(0)}%)`;
    } else if (viewMode === 'delta') {
      if (variant !== 'orig') {
        const dSign = delta >= 0 ? '+' : '';
        label += ` (${dSign}${(delta * 100).toFixed(0)}%)`;
      }
    }
    span.textContent = label;
  });
}
</script>
</body>
</html>"""

# ── HTTP Handler ──────────────────────────────────────────────────────────────
class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_GET(self):
        if self.path in ('/', '/index.html'):
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')
            self.send_header('Content-Length', len(HTML.encode()))
            self.end_headers()
            self.wfile.write(HTML.encode())
        elif self.path == '/info':
            self._ok('application/json', INFO.encode())
        else:
            self._send(404, 'text/plain', b'Not found')

    def do_POST(self):
        if self.path == '/step':
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            hist = body.get('history', [])
            generated = body.get('generated', {})
            
            # Print log
            last = repr(hist[-1]['content'][:60]) if hist else ''
            print(f"\n[STEP] {len(hist)} turn(s) | {last}")
            
            results = {}
            # Step for 'orig' first to cache its probabilities
            orig_gen = generated.get("orig", [])
            print(f"  → orig … gen_ids={orig_gen}", end=' ', flush=True)
            orig_res, orig_probs_tensor = step_variant("orig", hist, orig_gen)
            results["orig"] = orig_res
            print("✓")
            
            orig_probs_list = [orig_probs_tensor[j] for j in range(orig_probs_tensor.shape[0])]
            
            # Step other variants, passing orig_probs_list
            for name in ('small', 'medium', 'large'):
                variant_gen = generated.get(name, [])
                print(f"  → {name} … gen_ids={variant_gen}", end=' ', flush=True)
                variant_res, _ = step_variant(name, hist, variant_gen, orig_probs_list=orig_probs_list)
                results[name] = variant_res
                print("✓")
                
            self._ok('application/json', json.dumps(results).encode())
            
        elif self.path == '/complete':
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            hist = body.get('history', [])
            generated = body.get('generated', {})
            
            # Print log
            last = repr(hist[-1]['content'][:60]) if hist else ''
            print(f"\n[COMPLETE] {len(hist)} turn(s) | {last}")
            
            # 1. Generate full token lists for each variant in backend
            full_ids = {}
            for name in ('orig', 'small', 'medium', 'large'):
                variant_gen = generated.get(name, [])
                print(f"  → Generating full sequence for {name} …", end=' ', flush=True)
                full_ids[name] = complete_variant(name, hist, variant_gen)
                print(f"done, length={len(full_ids[name])}")
                
            # 2. Compute metadata for all steps
            results = {}
            print(f"  → orig steps metadata …", end=' ', flush=True)
            orig_res, orig_probs_tensor = step_variant("orig", hist, full_ids["orig"])
            results["orig"] = orig_res
            print("✓")
            
            orig_probs_list = [orig_probs_tensor[j] for j in range(orig_probs_tensor.shape[0])]
            
            for name in ('small', 'medium', 'large'):
                print(f"  → {name} steps metadata …", end=' ', flush=True)
                variant_res, _ = step_variant(name, hist, full_ids[name], orig_probs_list=orig_probs_list)
                results[name] = variant_res
                print("✓")
                
            self._ok('application/json', json.dumps(results).encode())
        else:
            self._send(404, 'text/plain', b'Not found')

    def _ok(self, ct, body):   self._send(200, ct, body)
    def _send(self, code, ct, body):
        self.send_response(code)
        self.send_header('Content-Type', ct)
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)


if __name__ == '__main__':
    socketserver.TCPServer.allow_reuse_address = True
    print(f"\n{'='*60}")
    print(f"  🔬 Embedding Perturbation Lab — 4-way comparison")
    print(f"  Token : '{tokenizer.decode([PERTURB_TOKEN_ID])}'  (id={PERTURB_TOKEN_ID})")
    print(f"  Dims  : {PERTURB_DIMS}")
    print(f"  A) Original     Δ = 0")
    print(f"  B) Small        Δ = +0.0001")
    print(f"  C) Medium       Δ = +0.001")
    print(f"  D) Large        Δ = +0.01")
    print(f"  Seed  : {SEED}   Temp : 0.0 (greedy)")
    print(f"{'='*60}")
    print(f"\n  Open → http://localhost:{PORT}\n")
    with socketserver.TCPServer(('', PORT), Handler) as srv:
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print('\nShutting down.')
            sys.exit(0)
