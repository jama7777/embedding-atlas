import os
import sys
import json
import re
import random
import numpy as np
from sklearn.decomposition import PCA
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from transformers import AutoTokenizer
from rank_bm25 import BM25Okapi

sys.path.append("scripts")
from embed_loader import get_token_embeddings

# ----------------------------------------------------------------------
# 1. SETUP PATHS AND CONFIG
# ----------------------------------------------------------------------
ALL_DOCS_DIR = "all_documents"
QUESTIONS_FILE = os.path.join(ALL_DOCS_DIR, "questions.jsonl")
CACHE_DIR = "cache"
HTML_DIR = "html"
QDRANT_PATH = os.path.join(CACHE_DIR, "qdrant_db")
COLLECTION_NAME = "enterprise_rag_bench"
TOTAL_TARGET_DOCS = 5000

os.makedirs(HTML_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Read HF token from .env
def load_token():
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if tok:
        return tok
    if os.path.exists(".env"):
        for line in open(".env"):
            m = re.match(r'(?:export\s+)?(\w+)\s*=\s*["\']?([^"\'\s]+)', line.strip())
            if m and ("HF" in m.group(1).upper() or "HUGG" in m.group(1).upper()):
                return m.group(2)
    return None

HF_TOKEN = load_token()

# ----------------------------------------------------------------------
# 2. SCAN QUESTIONS AND COMPILE THE 5K DOCUMENT SLICE
# ----------------------------------------------------------------------
print("Scanning questions.jsonl for golden documents...")
questions = []
gold_doc_ids = set()
with open(QUESTIONS_FILE, "r") as f:
    for line in f:
        q = json.loads(line.strip())
        questions.append(q)
        for doc_id in q.get("expected_doc_ids", []):
            gold_doc_ids.add(doc_id)

print(f"Found {len(questions)} questions referencing {len(gold_doc_ids)} unique gold documents.")

# Walk directories and catalog all documents
print("Walking all_documents/ directory to catalog files...")
source_types = ["slack", "gmail", "linear", "google_drive", "hubspot", "fireflies", "github", "jira", "confluence"]
catalog = {st: [] for st in source_types}
gold_files = []

for root, _, files in os.walk(ALL_DOCS_DIR):
    for f in files:
        if not f.endswith(".txt") or "__" not in f:
            continue
        parts = f.split("__", 1)
        doc_id = parts[0]
        # Identify source type based on folder path
        st_match = None
        for st in source_types:
            if f"/{st}/" in root or root.endswith(f"/{st}"):
                st_match = st
                break
        if not st_match:
            continue
        
        full_path = os.path.join(root, f)
        doc_record = {
            "id": doc_id,
            "filename": f,
            "path": full_path,
            "source_type": st_match,
            "is_gold": doc_id in gold_doc_ids
        }
        
        if doc_record["is_gold"]:
            gold_files.append(doc_record)
        else:
            catalog[st_match].append(doc_record)

print(f"Located {len(gold_files)} gold files on disk.")

# Calculate how many additional files we need
needed = TOTAL_TARGET_DOCS - len(gold_files)
if needed < 0:
    print(f"Warning: Gold files ({len(gold_files)}) exceed target size of {TOTAL_TARGET_DOCS}. Keeping all gold files.")
    selected_docs = gold_files
else:
    # Proportional sampling from non-gold files across source types
    total_non_gold = sum(len(lst) for lst in catalog.values())
    sampled_docs = []
    for st, lst in catalog.items():
        if not lst:
            continue
        # Allocate proportionally
        prop = len(lst) / total_non_gold
        st_needed = int(round(needed * prop))
        st_needed = min(st_needed, len(lst))
        random.seed(42)  # Deterministic sampling
        sampled_docs.extend(random.sample(lst, st_needed))
    
    # Fill remaining due to rounding
    still_needed = TOTAL_TARGET_DOCS - (len(gold_files) + len(sampled_docs))
    if still_needed > 0:
        all_remaining = []
        for lst in catalog.values():
            for doc in lst:
                if doc not in sampled_docs:
                    all_remaining.append(doc)
        sampled_docs.extend(random.sample(all_remaining, min(still_needed, len(all_remaining))))
        
    selected_docs = gold_files + sampled_docs

print(f"Compiled target document slice! Total documents: {len(selected_docs)} ({len(gold_files)} gold, {len(selected_docs) - len(gold_files)} sampled).")

# Map document ID to record
id_to_record = {doc["id"]: doc for doc in selected_docs}

# ----------------------------------------------------------------------
# 3. EMBED THE 5K DOCUMENTS USING CACHED GEMMA TABLE (CBOW)
# ----------------------------------------------------------------------
print("\nLoading Gemma token-embedding table...")
emb_table, _, _ = get_token_embeddings("google/gemma-2-2b", token=HF_TOKEN)
print("Loaded token-embedding table successfully. Shape:", emb_table.shape)

print("Loading Gemma tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b", token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

def embed_text(text):
    ids = tokenizer.encode(text, truncation=True, max_length=1024)
    if not ids:
        return np.zeros(emb_table.shape[1], dtype=np.float32)
    # Mean Continuous Bag-of-Words
    vectors = emb_table[ids]
    mean_vec = vectors.mean(axis=0)
    norm = np.linalg.norm(mean_vec)
    if norm > 0:
        mean_vec = mean_vec / norm
    return mean_vec.astype(np.float32)

print("\nGenerating Continuous Bag-of-Words embeddings and preparing texts for BM25...")
embeddings = []
texts = []
tokenized_corpus = []

for i, doc in enumerate(selected_docs):
    with open(doc["path"], "r", errors="ignore") as f:
        content = f.read(10000).strip() # read up to 10k chars to avoid dilution
    texts.append(content)
    vec = embed_text(content)
    embeddings.append(vec)
    doc["snippet"] = content[:350] + ("..." if len(content) > 350 else "")
    
    # Tokenize for BM25 (lowercased, alphanumeric tokens)
    clean_tokens = re.findall(r'\b\w+\b', content.lower())
    tokenized_corpus.append(clean_tokens)
    
    if (i + 1) % 1000 == 0:
        print(f"  Processed {i + 1}/5000 documents")

embeddings = np.array(embeddings, dtype=np.float32)
print("Embedding generation complete! Matrix shape:", embeddings.shape)

# ----------------------------------------------------------------------
# 4. INITIALIZE PERSISTENT LOCAL QDRANT DB & BM25 OKAPI
# ----------------------------------------------------------------------
print(f"\nInitializing local persistent Qdrant DB at {QDRANT_PATH}...")
q_client = QdrantClient(path=QDRANT_PATH)

# Recreate collection to ensure clean state
if q_client.collection_exists(COLLECTION_NAME):
    q_client.delete_collection(COLLECTION_NAME)

q_client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=int(embeddings.shape[1]), distance=Distance.COSINE),
)

print("Upserting document vectors and metadata to Qdrant...")
points = []
for i, doc in enumerate(selected_docs):
    points.append(PointStruct(
        id=i,
        vector=embeddings[i].tolist(),
        payload={
            "id": doc["id"],
            "filename": doc["filename"],
            "source_type": doc["source_type"],
            "is_gold": bool(doc["is_gold"]),
            "snippet": doc["snippet"]
        }
    ))

# Ingest in batches
BATCH_SIZE = 1000
for i in range(0, len(points), BATCH_SIZE):
    q_client.upsert(
        collection_name=COLLECTION_NAME,
        wait=True,
        points=points[i:i+BATCH_SIZE]
    )
print(f"Successfully indexed {len(points)} documents in Qdrant!")

print("\nBuilding the rank-bm25 index on document text tokens...")
bm25 = BM25Okapi(tokenized_corpus)
print("BM25 index constructed successfully!")

# ----------------------------------------------------------------------
# 5. PCA DIMENSIONALITY REDUCTION FOR PLOTLY
# ----------------------------------------------------------------------
print("\nPerforming fast PCA to compute 2D mapping coordinates...")
pca = PCA(n_components=2, random_state=42)
xy = pca.fit_transform(embeddings).astype(np.float32)
print("PCA reduction complete!")

# ----------------------------------------------------------------------
# 6. EXECUTE RETRIEVAL FOR ALL 500 QUESTIONS (EMBEDDING & BM25)
# ----------------------------------------------------------------------
print("\nRunning RAG retrieval for all 500 questions under BOTH modes...")
q_retrievals_embed = {}
q_retrievals_bm25 = {}

for idx, q in enumerate(questions):
    q_text = q["question"]
    
    # 1. Gemma Embedding search via Qdrant
    q_vec = embed_text(q_text)
    res = q_client.query_points(
        collection_name=COLLECTION_NAME,
        query=q_vec.tolist(),
        limit=10
    )
    
    retrieved_embed = []
    for hit in res.points:
        retrieved_embed.append({
            "idx": hit.id,
            "id": hit.payload["id"],
            "score": round(hit.score, 4)
        })
    q_retrievals_embed[q["question_id"]] = retrieved_embed
    
    # 2. BM25 Keyword Search
    q_tokens = re.findall(r'\b\w+\b', q_text.lower())
    scores = bm25.get_scores(q_tokens)
    top_bm25_indices = np.argsort(scores)[::-1][:10]
    
    retrieved_bm25 = []
    for r_idx in top_bm25_indices:
        score = scores[r_idx]
        retrieved_bm25.append({
            "idx": int(r_idx),
            "id": selected_docs[r_idx]["id"],
            "score": round(float(score), 4)
        })
    q_retrievals_bm25[q["question_id"]] = retrieved_bm25

    if (idx + 1) % 100 == 0:
        print(f"  Ran retrieval for {idx + 1}/500 questions")

# ----------------------------------------------------------------------
# 7. GENERATE THE HTML DASHBOARD
# ----------------------------------------------------------------------
print("\nGenerating interactive HTML embedding atlas...")

# Define color palette for source types
PALETTE = {
    "slack": "#10b981",          # Emerald
    "gmail": "#f97316",          # Orange
    "linear": "#a855f7",         # Purple
    "google_drive": "#3b82f6",   # Blue
    "hubspot": "#eab308",        # Yellow
    "fireflies": "#06b6d4",      # Cyan
    "github": "#4b5563",         # Gray
    "jira": "#ec4899",           # Pink
    "confluence": "#f43f5e"      # Rose
}

# Compile data points for Plotly trace construction
docs_data = []
for i, doc in enumerate(selected_docs):
    docs_data.append({
        "idx": i,
        "id": doc["id"],
        "x": round(float(xy[i, 0]), 4),
        "y": round(float(xy[i, 1]), 4),
        "source": doc["source_type"],
        "is_gold": bool(doc["is_gold"]),
        "filename": doc["filename"],
        "snippet": doc["snippet"]
    })

# Format question list with expected doc indices and dual-mode hits
questions_data = []
for q in questions:
    # Resolve gold doc indices on our map
    gold_idxs = []
    for doc_id in q.get("expected_doc_ids", []):
        if doc_id in id_to_record:
            idx = selected_docs.index(id_to_record[doc_id])
            gold_idxs.append(idx)
            
    questions_data.append({
        "qid": q["question_id"],
        "type": q["question_type"],
        "question": q["question"],
        "gold_answer": q["gold_answer"],
        "facts": q.get("answer_facts", []),
        "gold_doc_ids": q.get("expected_doc_ids", []),
        "gold_idxs": gold_idxs,
        "hits_embed": q_retrievals_embed[q["question_id"]],
        "hits_bm25": q_retrievals_bm25[q["question_id"]]
    })

TEMPLATE = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>EnterpriseRAG-Bench Embedding Atlas</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
  <style>
    :root {
      --bg-dark: #f8fafc;
      --panel-dark: #ffffff;
      --panel-border: #e2e8f0;
      --text-main: #0f172a;
      --text-muted: #475569;
      --brand: #0284c7;
      --gold: #d97706;
      --hit: #dc2626;
    }
    body {
      font-family: 'Inter', sans-serif;
      margin: 0;
      background: var(--bg-dark);
      color: var(--text-main);
      display: flex;
      height: 100vh;
      overflow: hidden;
    }
    #sidebar {
      width: 480px;
      background: var(--panel-dark);
      border-right: 1px solid var(--panel-border);
      display: flex;
      flex-direction: column;
      box-shadow: 4px 0 16px rgba(15,23,42,0.06);
      z-index: 10;
    }
    #header {
      padding: 16px 24px;
      border-bottom: 1px solid var(--panel-border);
      background: rgba(248, 250, 252, 0.8);
    }
    h1 {
      font-size: 20px;
      font-weight: 700;
      margin: 0 0 4px;
      color: var(--brand);
      letter-spacing: -0.02em;
    }
    h1 span {
      color: var(--text-main);
      font-weight: 400;
    }
    .subtitle {
      font-size: 11px;
      color: var(--text-muted);
      margin: 0;
    }
    #controls {
      padding: 16px 24px;
      border-bottom: 1px solid var(--panel-border);
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .control-group {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    label {
      font-size: 10px;
      font-weight: 700;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    select, input {
      background: #f1f5f9;
      border: 1px solid var(--panel-border);
      color: var(--text-main);
      padding: 8px 12px;
      border-radius: 8px;
      font-size: 13px;
      outline: none;
      transition: border 0.2s;
    }
    select:focus, input:focus {
      border-color: var(--brand);
    }
    .filter-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 8px;
      margin-top: 4px;
    }
    .filter-btn {
      background: #f8fafc;
      border: 1px solid var(--panel-border);
      border-radius: 6px;
      padding: 6px 10px;
      font-size: 11px;
      font-weight: 500;
      color: var(--text-muted);
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: all 0.2s;
    }
    .filter-btn.active {
      color: var(--text-main);
      border-color: var(--brand);
      background: rgba(2, 132, 199, 0.08);
    }
    .filter-color {
      width: 10px;
      height: 10px;
      border-radius: 50%;
    }
    .mode-toggle {
      display: flex;
      background: #f1f5f9;
      padding: 3px;
      border-radius: 8px;
      border: 1px solid var(--panel-border);
    }
    .mode-btn {
      flex: 1;
      border: none;
      background: transparent;
      color: var(--text-muted);
      padding: 8px 0;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      border-radius: 6px;
      transition: all 0.2s;
    }
    .mode-btn.active {
      background: var(--brand);
      color: #fff;
    }
    #content {
      flex: 1;
      padding: 20px 24px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    .card {
      background: rgba(248, 250, 252, 0.8);
      border: 1px solid var(--panel-border);
      border-radius: 10px;
      padding: 14px 18px;
    }
    .card-title {
      font-size: 11px;
      font-weight: 600;
      color: var(--brand);
      text-transform: uppercase;
      margin-bottom: 6px;
      letter-spacing: 0.04em;
    }
    .question-text {
      font-size: 14px;
      font-weight: 500;
      line-height: 1.5;
    }
    .meta-badge {
      display: inline-block;
      padding: 3px 8px;
      border-radius: 8px;
      font-size: 10px;
      font-weight: 600;
      margin-top: 8px;
      background: var(--panel-border);
      color: var(--brand);
    }
    .gold-box {
      border-color: var(--gold);
      background: rgba(217, 119, 6, 0.04);
    }
    .gold-box .card-title {
      color: var(--gold);
    }
    .facts-list {
      margin: 6px 0 0;
      padding-left: 16px;
      font-size: 12px;
      color: var(--text-muted);
      line-height: 1.5;
    }
    .facts-list li {
      margin-bottom: 4px;
    }
    #evaluation-card {
      display: flex;
      align-items: center;
      gap: 16px;
    }
    .score-ring {
      width: 64px;
      height: 64px;
      border-radius: 50%;
      border: 4px solid var(--hit);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 18px;
      font-weight: 700;
      color: var(--text-main);
      flex-shrink: 0;
    }
    .score-ring.success {
      border-color: #10b981;
    }
    .eval-desc {
      font-size: 12px;
      color: var(--text-muted);
      line-height: 1.4;
    }
    .eval-desc strong {
      color: var(--text-main);
    }
    #map-container {
      flex: 1;
      position: relative;
      background: #ffffff;
    }
    #plot {
      width: 100%;
      height: 100%;
    }
  </style>
</head>
<body>

  <div id="sidebar">
    <div id="header">
      <h1>EnterpriseRAG<span>Atlas</span></h1>
      <p class="subtitle">Redwood Inference Multi-Retrieval Evaluation</p>
    </div>
    
    <div id="controls">
      <div class="control-group">
        <label for="q-select">Select Benchmark Question</label>
        <select id="q-select" onchange="loadQuestion(this.value)"></select>
      </div>
      
      <div class="control-group">
        <label for="search-input">Search Question Filter</label>
        <input id="search-input" type="text" placeholder="Filter questions..." oninput="filterQuestions(this.value)">
      </div>
      
      <div class="control-group">
        <label>Active Retrieval Search Mode</label>
        <div class="mode-toggle">
          <button class="mode-btn active" id="btn-embed" onclick="setSearchMode('embedding')">Embedding (Gemma)</button>
          <button class="mode-btn" id="btn-bm25" onclick="setSearchMode('bm25')">BM25 Keyword</button>
        </div>
      </div>
      
      <div class="control-group">
        <label>Filter Document Sources</label>
        <div class="filter-grid" id="filters-container">
          <!-- Populated dynamically -->
        </div>
      </div>
    </div>
    
    <div id="content">
      <div class="card">
        <div class="card-title">RAG Query</div>
        <div class="question-text" id="q-text">Select a question to begin simulation</div>
        <div class="meta-badge" id="q-meta">Type: --</div>
      </div>
      
      <div class="card" id="evaluation-card" style="display:none;">
        <div class="score-ring" id="eval-score">0%</div>
        <div class="eval-desc" id="eval-desc">
          <strong>Retrieval Recall@10</strong><br>
          Gold Documents retrieved: <span id="eval-ratio">0/0</span>.
        </div>
      </div>
      
      <div class="card gold-box" id="gold-card" style="display:none;">
        <div class="card-title">Gold Ground-Truth Answer</div>
        <div class="question-text" id="gold-answer" style="font-size:13px; color:var(--text-main);">--</div>
        <div class="card-title" style="margin-top: 12px; font-size:10px;">Expected Answer Facts</div>
        <ul class="facts-list" id="gold-facts"></ul>
      </div>
    </div>
  </div>
  
  <div id="map-container">
    <div id="plot"></div>
  </div>

  <script>
    const D = {docs_data};
    const Q = {questions_data};
    const PALETTE = {palette_json};
    
    const gd = document.getElementById("plot");
    let activeQuestionId = "";
    let activeSearchMode = "embedding"; // embedding or bm25
    const filterStates = {};
    
    // ----------------------------------------------------------------------
    // INITIALIZE CONTROL SELECTS, FILTERS, AND BUTTONS
    // ----------------------------------------------------------------------
    const select = document.getElementById("q-select");
    function renderDropdown(items) {
      select.innerHTML = '<option value="">-- Choose a Question --</option>' +
        items.map(q => `<option value="${q.qid}">[${q.qid}] ${q.question.substring(0, 70)}...</option>`).join("");
      select.value = activeQuestionId;
    }
    renderDropdown(Q);
    
    function filterQuestions(val) {
      const filtered = Q.filter(q => q.question.toLowerCase().includes(val.toLowerCase()) || q.qid.toLowerCase().includes(val.toLowerCase()));
      renderDropdown(filtered);
    }
    
    // Populate checkboxes/filters in sidebar
    const filterBox = document.getElementById("filters-container");
    Object.keys(PALETTE).forEach((src, idx) => {
      filterStates[src] = true; // by default visible
      filterBox.innerHTML += `
        <button class="filter-btn active" id="f-btn-${src}" onclick="toggleSource('${src}')">
          <div class="filter-color" style="background:${PALETTE[src]}"></div>
          <span>${src}</span>
        </button>
      `;
    });

    // ----------------------------------------------------------------------
    // CONSTRUCT BASE PLOTLY SCATTER TRACES
    // ----------------------------------------------------------------------
    const sourceGroups = {};
    D.forEach(doc => {
      if (!sourceGroups[doc.source]) sourceGroups[doc.source] = {x: [], y: [], text: [], ids: []};
      sourceGroups[doc.source].x.push(doc.x);
      sourceGroups[doc.source].y.push(doc.y);
      sourceGroups[doc.source].text.push(`[${doc.source}] ${doc.filename}<br><br>${doc.snippet}`);
      sourceGroups[doc.source].ids.push(doc.id);
    });

    const traces = [];
    const sourceTraceIndices = {};
    
    Object.keys(sourceGroups).forEach(src => {
      sourceTraceIndices[src] = traces.length;
      traces.push({
        x: sourceGroups[src].x,
        y: sourceGroups[src].y,
        text: sourceGroups[src].text,
        mode: 'markers',
        name: src,
        hoverinfo: 'text',
        marker: {
          size: 4.5,
          color: PALETTE[src],
          opacity: 0.65
        }
      });
    });

    // Highlighted Gold Documents Trace
    const goldTraceIdx = traces.length;
    traces.push({
      x: [], y: [], text: [],
      mode: 'markers+text',
      name: '★ Gold Documents',
      textposition: 'top center',
      textfont: { size: 10, color: '#d97706' },
      hoverinfo: 'text',
      marker: {
        size: 14,
        color: '#d97706',
        symbol: 'star',
        line: { width: 1.5, color: '#1e293b' }
      }
    });

    // Highlighted Qdrant/BM25 Hits Trace
    const hitTraceIdx = traces.length;
    traces.push({
      x: [], y: [], text: [],
      mode: 'markers+text',
      name: '◆ Search Hits',
      textposition: 'bottom center',
      textfont: { size: 9, color: '#dc2626' },
      hoverinfo: 'text',
      marker: {
        size: 11,
        color: '#dc2626',
        symbol: 'diamond',
        line: { width: 1, color: '#1e293b' }
      }
    });

    const layout = {
      backgroundColor: '#ffffff',
      paper_bgcolor: '#ffffff',
      plot_bgcolor: '#ffffff',
      showlegend: false,
      margin: { l: 0, r: 0, b: 0, t: 0 },
      hoverlabel: {
        bgcolor: '#ffffff',
        bordercolor: '#cbd5e1',
        font: { color: '#0f172a', size: 12 }
      },
      xaxis: { showgrid: false, zeroline: false, visible: false },
      yaxis: { showgrid: false, zeroline: false, visible: false }
    };

    Plotly.newPlot(gd, traces, layout, { responsive: true });

    // ----------------------------------------------------------------------
    // DYNAMIC FILTERS AND SEARCH MODES HANDLERS
    // ----------------------------------------------------------------------
    function toggleSource(src) {
      filterStates[src] = !filterStates[src];
      const btn = document.getElementById(`f-btn-${src}`);
      const trIdx = sourceTraceIndices[src];
      
      if (filterStates[src]) {
        btn.classList.add("active");
        Plotly.restyle(gd, { visible: true }, [trIdx]);
      } else {
        btn.classList.remove("active");
        Plotly.restyle(gd, { visible: "legendonly" }, [trIdx]);
      }
    }
    
    function setSearchMode(mode) {
      activeSearchMode = mode;
      document.getElementById("btn-embed").classList.toggle("active", mode === "embedding");
      document.getElementById("btn-bm25").classList.toggle("active", mode === "bm25");
      if (activeQuestionId) {
        loadQuestion(activeQuestionId);
      }
    }

    function loadQuestion(qid) {
      activeQuestionId = qid;
      if (!qid) {
        // Clear all Highlights
        Plotly.restyle(gd, { x: [[]], y: [[]], text: [[]] }, [goldTraceIdx, hitTraceIdx]);
        document.getElementById("q-text").textContent = "Select a question to begin simulation";
        document.getElementById("q-meta").textContent = "Type: --";
        document.getElementById("evaluation-card").style.display = "none";
        document.getElementById("gold-card").style.display = "none";
        return;
      }
      
      const q = Q.find(item => item.qid === qid);
      if (!q) return;

      // Update Sidebar UI
      document.getElementById("q-text").textContent = q.question;
      document.getElementById("q-meta").textContent = "Type: " + q.type.toUpperCase();
      document.getElementById("gold-answer").textContent = q.gold_answer;
      document.getElementById("gold-facts").innerHTML = q.facts.map(f => `<li>${f}</li>`).join("");
      
      document.getElementById("gold-card").style.display = "block";
      document.getElementById("evaluation-card").style.display = "flex";

      // 1. Plot Gold Documents
      const goldXs = [], goldYs = [], goldTexts = [];
      q.gold_idxs.forEach(idx => {
        const pt = D[idx];
        goldXs.push(pt.x);
        goldYs.push(pt.y);
        goldTexts.push(`[GOLD DOC] ${pt.filename}`);
      });
      Plotly.restyle(gd, { x: [goldXs], y: [goldYs], text: [goldTexts] }, [goldTraceIdx]);

      // 2. Select corresponding search mode hits
      const hits = activeSearchMode === "embedding" ? q.hits_embed : q.hits_bm25;
      const hitXs = [], hitYs = [], hitTexts = [];
      let retrievedGoldCount = 0;
      const expectedSet = new Set(q.gold_doc_ids);
      
      hits.forEach((hit, rIdx) => {
        const pt = D[hit.idx];
        hitXs.push(pt.x);
        hitYs.push(pt.y);
        const isGoldMatch = expectedSet.has(hit.id);
        if (isGoldMatch) retrievedGoldCount++;
        
        hitTexts.push(`[${activeSearchMode === "embedding" ? 'Gemma' : 'BM25'} Rank ${rIdx + 1} - Score: ${hit.score}] ${pt.filename}<br>${isGoldMatch ? '★ MATCHED GOLD DOCUMENT' : '❌ DISTRACTOR NOISE'}`);
      });
      Plotly.restyle(gd, { x: [hitXs], y: [hitYs], text: [hitTexts] }, [hitTraceIdx]);

      // 3. Compute Evaluation Recall@10
      const ratioStr = `${retrievedGoldCount}/${q.gold_doc_ids.length}`;
      const pctVal = q.gold_doc_ids.length > 0 ? Math.round((retrievedGoldCount / q.gold_doc_ids.length) * 100) : 100;
      
      const ring = document.getElementById("eval-score");
      ring.textContent = `${pctVal}%`;
      if (pctVal === 100) {
        ring.className = "score-ring success";
      } else {
        ring.className = "score-ring";
      }
      document.getElementById("eval-ratio").textContent = ratioStr;

      // 4. Zoom to Area
      const allXs = goldXs.concat(hitXs);
      const allYs = goldYs.concat(hitYs);
      if (allXs.length > 0) {
        const xMin = Math.min(...allXs), xMax = Math.max(...allXs);
        const yMin = Math.min(...allYs), yMax = Math.max(...allYs);
        const padX = Math.max((xMax - xMin) * 0.4, 0.5);
        const padY = Math.max((yMax - yMin) * 0.4, 0.5);
        
        Plotly.relayout(gd, {
          "xaxis.range": [xMin - padX, xMax + padX],
          "yaxis.range": [yMin - padY, yMax + padY]
        });
      }
    }
  </script>
</body>
</html>
"""

# Inject data structures into template
html_out = TEMPLATE.replace("{docs_data}", json.dumps(docs_data))
html_out = html_out.replace("{questions_data}", json.dumps(questions_data))
html_out = html_out.replace("{palette_json}", json.dumps(PALETTE))

out_path = os.path.join(HTML_DIR, "atlas_enterprise_rag.html")
with open(out_path, "w") as f:
    f.write(html_out)

print(f"\nSaved interactive embedding atlas successfully -> {out_path} ({os.path.getsize(out_path)//1_000_000} MB)")
print("Done. Launch and explore the RAG dashboard!")
