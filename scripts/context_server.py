import http.server
import socketserver
import json
import os
import time
import sys
import re
import random
import threading
import gzip
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA

PORT = 8095
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Detecting device... Selected: {device.upper()}")

# ----------------------------------------------------------------------
# 1. LOAD MODEL
# ----------------------------------------------------------------------
model_local = "cache/local_qwen_0.5b"

if os.path.exists(model_local):
    print(f"Loading Model OFFLINE from local folder: '{model_local}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_local, local_files_only=True)
    dtype = torch.bfloat16 if device in ["mps", "cuda"] else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_local,
        torch_dtype=dtype,
        local_files_only=True,
        attn_implementation="eager"
    ).to(device)
else:
    print("Loading Model online: Qwen/Qwen2.5-0.5B-Instruct...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        attn_implementation="eager"
    ).to(device)

model.eval()
print("Model loaded successfully!")
sys.stdout.flush()

# ----------------------------------------------------------------------
# 2. HTTP SERVER REQUEST HANDLER
# ----------------------------------------------------------------------
class ModelServerHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"   # needed for SSE streaming

    def log_message(self, format, *args):
        # Quiet requests logging for readability
        pass

    def do_OPTIONS(self):
        """Handle CORS preflight requests from the browser."""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _serve_file(self, file_path, content_type):
        """Serve a file with Content-Length, gzip compression if supported, and chunked I/O."""
        if not os.path.exists(file_path):
            self.send_response(404)
            self.send_header("Content-Length", "9")
            self.end_headers()
            self.wfile.write(b"Not Found")
            return

        # Check if the client accepts gzip
        accept_enc = self.headers.get("Accept-Encoding", "")
        use_gzip = "gzip" in accept_enc

        with open(file_path, "rb") as f:
            raw = f.read()

        cache_control = "no-cache, no-store, must-revalidate" if file_path.endswith(".html") else "public, max-age=3600"

        if use_gzip:
            body = gzip.compress(raw, compresslevel=6)
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache_control)
            self.end_headers()
        else:
            body = raw
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache_control)
            self.end_headers()

        # Stream in 64KB chunks so bytes reach the browser immediately
        CHUNK = 65536
        for i in range(0, len(body), CHUNK):
            self.wfile.write(body[i:i + CHUNK])
        self.wfile.flush()

    def do_GET(self):
        if self.path in ("/", "/index.html", "/context_ui.html"):
            self._serve_file("html/context_ui.html", "text/html; charset=utf-8")
        elif self.path == "/plotly.min.js":
            self._serve_file("html/plotly.min.js", "application/javascript")
        else:
            self.send_response(404)
            self.send_header("Content-Length", "9")
            self.end_headers()
            self.wfile.write(b"Not Found")

    def do_POST(self):
        if self.path == "/count_tokens":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            params = json.loads(post_data.decode('utf-8'))
            
            system_instruction = params.get("system_instruction", "")
            context_text = params.get("context_text", "")
            question = params.get("question", "")
            
            # Count elements individually
            sys_tokens = len(tokenizer(system_instruction)["input_ids"]) if system_instruction else 0
            ctx_tokens = len(tokenizer(context_text)["input_ids"]) if context_text else 0
            q_tokens = len(tokenizer(question)["input_ids"]) if question else 0
            
            # Formulate full template context
            messages = []
            if system_instruction:
                messages.append({"role": "system", "content": system_instruction})
            
            user_content = ""
            if context_text:
                user_content += f"Context:\n{context_text}\n\n"
            if question:
                user_content += f"Question:\n{question}"
                
            if user_content:
                messages.append({"role": "user", "content": user_content})
            
            if messages:
                formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                total_tokens = len(tokenizer(formatted_prompt)["input_ids"])
            else:
                total_tokens = 0
                
            response_payload = {
                "system": sys_tokens,
                "context": ctx_tokens,
                "question": q_tokens,
                "total": total_tokens
            }
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            body = json.dumps(response_payload).encode('utf-8')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path == "/chat":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            params = json.loads(post_data.decode('utf-8'))
            
            system_instruction = params.get("system_instruction", "")
            context_text = params.get("context_text", "")
            question = params.get("question", "")
            temperature = float(params.get("temperature", 0.1))
            max_new_tokens = int(params.get("max_new_tokens", 150))
            
            # Formulate chat format
            messages = []
            if system_instruction:
                messages.append({"role": "system", "content": system_instruction})
                
            user_content = ""
            if context_text:
                user_content += f"Context:\n{context_text}\n\n"
            if question:
                user_content += f"Question:\n{question}"
                
            if user_content:
                messages.append({"role": "user", "content": user_content})
                
            if not messages:
                self.send_response(400)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return

            # ---------------------------------------------------------------
            # STREAMING CHAT HANDLER — manual autoregressive loop with scores
            # ---------------------------------------------------------------
            try:
                formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
                prompt_len = inputs["input_ids"].shape[1]

                # --- Send SSE headers before starting generation ---
                # NOTE: We do NOT set Transfer-Encoding: chunked because
                # Python's http.server does not write actual chunked frames.
                # Instead we use Connection: close — browser reads until socket closes.
                # To prevent Python http.server from automatically injecting
                # Transfer-Encoding: chunked under HTTP/1.1 (which corrupts the SSE stream),
                # we write the status line and headers directly to the socket.
                self.wfile.write(b"HTTP/1.1 200 OK\r\n")
                self.wfile.write(b"Content-Type: text/event-stream\r\n")
                self.wfile.write(b"Cache-Control: no-cache, no-store, must-revalidate\r\n")
                self.wfile.write(b"Connection: close\r\n")
                self.wfile.write(b"X-Accel-Buffering: no\r\n")
                self.wfile.write(b"Access-Control-Allow-Origin: *\r\n")
                self.wfile.write(b"\r\n")
                self.wfile.flush()

                client_alive = True

                def _send_event(obj):
                    nonlocal client_alive
                    if not client_alive:
                        return
                    try:
                        line = "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"
                        self.wfile.write(line.encode("utf-8"))
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        client_alive = False

                # ---------------------------------------------------------------
                # Manual autoregressive token-by-token loop (with KV-cache)
                # Gives us per-step logits → probabilities → top alternatives
                # ---------------------------------------------------------------
                gen_start = time.time()
                all_token_texts = []
                all_token_data  = []   # [{text, index, elapsed_ms, prob, top_alts}]
                token_index = 0

                input_ids      = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
                past_key_values = None
                eos_id = tokenizer.eos_token_id

                # Decode prompt tokens once to map them in visualization
                prompt_tokens_decoded = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]

                with torch.no_grad():
                    for step in range(max_new_tokens):
                        model_inputs = {
                            "input_ids": input_ids if past_key_values is None else input_ids[:, -1:],
                            "attention_mask": attention_mask,
                            "past_key_values": past_key_values,
                            "use_cache": True,
                        }
                        outputs = model(**model_inputs, output_attentions=True)
                        logits = outputs.logits[:, -1, :]   # [1, vocab_size]
                        past_key_values = outputs.past_key_values

                        # Extract attention weights paid to the prompt tokens (first prompt_len tokens)
                        # Qwen / Llama attention format is tuple of length num_layers.
                        # We use the final layer (attentions[-1]) of shape [1, num_heads, q_len, k_len]
                        last_layer_attns = outputs.attentions[-1]
                        # Average across attention heads
                        mean_attn = last_layer_attns[0].mean(dim=0) # [q_len, k_len]
                        if past_key_values is None:
                            # Step 0: query_len = prompt_len, key_len = prompt_len
                            step_attn = mean_attn[-1, :prompt_len]
                        else:
                            # Step > 0: query_len = 1, key_len = prompt_len + step
                            step_attn = mean_attn[0, :prompt_len]

                        # Move to CPU float and round to 4 decimals
                        attn_weights_list = [round(float(w), 4) for w in step_attn.cpu().float().numpy().tolist()]

                        # Apply temperature
                        if temperature > 0.0:
                            logits = logits / temperature

                        probs = torch.softmax(logits, dim=-1)[0]  # [vocab_size]

                        # Sample or greedy
                        if temperature > 0.0:
                            next_token_id = torch.multinomial(probs, 1).item()
                        else:
                            next_token_id = probs.argmax().item()

                        # Selected token probability
                        chosen_prob = float(probs[next_token_id].item())

                        # Top-5 alternatives (including chosen)
                        top5_probs, top5_ids = torch.topk(probs, 5)
                        top_alternatives = []
                        for alt_prob, alt_id in zip(top5_probs.tolist(), top5_ids.tolist()):
                            alt_text = tokenizer.decode([alt_id], skip_special_tokens=False)
                            alt_text = alt_text.replace("\n", "\\n").replace("\t", "\\t")
                            top_alternatives.append({
                                "token": alt_text,
                                "token_id": alt_id,
                                "prob": round(alt_prob, 4),
                                "is_chosen": (alt_id == next_token_id)
                            })

                        # Decode chosen token text
                        token_text = tokenizer.decode([next_token_id], skip_special_tokens=False)

                        # Stop on EOS
                        if next_token_id == eos_id:
                            break

                        # Clean display text
                        display_text = tokenizer.decode([next_token_id], skip_special_tokens=True)

                        elapsed_ms = int((time.time() - gen_start) * 1000)
                        all_token_texts.append(display_text)

                        tok_record = {
                            "text": display_text,
                            "token_id": next_token_id,
                            "index": token_index,
                            "elapsed_ms": elapsed_ms,
                            "prob": round(chosen_prob, 4),
                            "top_alternatives": top_alternatives,
                            "attention_weights": attn_weights_list
                        }
                        all_token_data.append(tok_record)

                        # SSE event with full per-token data
                        _send_event({
                            "type": "token",
                            "token": display_text,
                            "index": token_index,
                            "elapsed_ms": elapsed_ms,
                            "prob": round(chosen_prob, 4),
                            "top_alternatives": top_alternatives,
                            "attention_weights": attn_weights_list
                        })

                        # Extend input_ids and attention_mask for next step
                        next_token_tensor = torch.tensor([[next_token_id]], device=device)
                        input_ids = torch.cat([input_ids, next_token_tensor], dim=1)
                        attention_mask = torch.cat([
                            attention_mask,
                            torch.ones((1, 1), dtype=attention_mask.dtype, device=device)
                        ], dim=1)

                        token_index += 1

                generation_time = time.time() - gen_start
                response_text = "".join(all_token_texts)
                completion_tokens = token_index
                tokens_per_second = completion_tokens / max(generation_time, 1e-5)


                # --------------------------------------------------------------
                # B. PERFORM REAL-TIME VISUALIZATIONS GENERATION
                # --------------------------------------------------------------
                # 1. Relevance Heatmap: compute sentence similarities to question
                sentence_ends = re.compile(r'(?<=[.!?]) +')
                raw_sentences = sentence_ends.split(context_text)
                sentences = [s.strip() for s in raw_sentences if s.strip()]
                
                relevance_data = []
                coords_payload = []
                
                if sentences and question:
                    # Embed user question
                    q_ids = tokenizer(question, return_tensors="pt")["input_ids"].to(device)
                    with torch.no_grad():
                        q_embs = model.get_input_embeddings()(q_ids)[0]
                        q_emb = q_embs.mean(dim=0).float().cpu().numpy() # Cast to float32 for NumPy
                    q_norm = np.linalg.norm(q_emb)
                    if q_norm > 0:
                        q_emb = q_emb / q_norm
                    
                    # Convert query embedding to 8-dimensional footprint (binned average of coords)
                    bins = 8
                    bin_size = len(q_emb) // bins
                    q_fingerprint = [float(np.mean(q_emb[i*bin_size : (i+1)*bin_size])) for i in range(bins)]
                        
                    # Embed sentences
                    sentence_records = []
                    for sent in sentences:
                        s_ids = tokenizer(sent, return_tensors="pt")["input_ids"].to(device)
                        with torch.no_grad():
                            s_embs = model.get_input_embeddings()(s_ids)[0]
                            s_emb = s_embs.mean(dim=0).float().cpu().numpy() # Cast to float32 for NumPy
                        s_norm = np.linalg.norm(s_emb)
                        if s_norm > 0:
                            s_emb = s_emb / s_norm
                        
                        sim = float(np.dot(s_emb, q_emb))
                        # Normalize cosine similarity [-1, 1] to [0, 1] visual scale
                        relevance = max(0.0, (sim + 1.0) / 2.0)
                        
                        # Convert sentence embedding to footprint
                        s_fingerprint = [float(np.mean(s_emb[i*bin_size : (i+1)*bin_size])) for i in range(bins)]
                        
                        sentence_records.append({
                            "text": sent,
                            "relevance": relevance,
                            "fingerprint": s_fingerprint
                        })
                    relevance_data = sentence_records
                    
                    # 2. Token Atlas 2D PCA Sampling
                    all_tokens_for_pca = []
                    
                    def add_tokens(text_source, category):
                        t_ids = tokenizer(text_source)["input_ids"]
                        if not t_ids:
                            return
                        t_ids_tensor = torch.tensor([t_ids], dtype=torch.long, device=device)
                        with torch.no_grad():
                            t_embs = model.get_input_embeddings()(t_ids_tensor)[0].float().cpu().numpy() # Cast to float32 for NumPy
                        for idx, tid in enumerate(t_ids):
                            tok_str = tokenizer.decode([tid])
                            # Clean up for presentation
                            tok_str = tok_str.replace("Ġ", " ").replace("\n", "\\n")
                            all_tokens_for_pca.append({
                                "token": tok_str,
                                "vector": t_embs[idx],
                                "category": category
                            })
                            
                    # Add system instruction tokens
                    if system_instruction:
                        add_tokens(system_instruction, "System Instructions")
                        
                    # Add question tokens
                    add_tokens(question, "User Question")
                    
                    # Add context tokens (sampled to avoid PCA crowding)
                    sorted_records = sorted(sentence_records, key=lambda x: x["relevance"], reverse=True)
                    top_records = sorted_records[:5]
                    top_texts = set(r["text"] for r in top_records)
                    
                    # Add top relevant sentences
                    for r in top_records:
                        add_tokens(r["text"], "Context (Highly Relevant)")
                        
                    # Add a random sample of other sentences
                    other_records = [r for r in sentence_records if r["text"] not in top_texts]
                    if other_records:
                        random.seed(42)
                        sampled_others = random.sample(other_records, min(3, len(other_records)))
                        for r in sampled_others:
                            add_tokens(r["text"], "Context (Other)")
                            
                    # Fit PCA
                    if len(all_tokens_for_pca) > 1:
                        vectors = np.array([item["vector"] for item in all_tokens_for_pca], dtype=np.float32)
                        pca = PCA(n_components=2, random_state=42)
                        xy = pca.fit_transform(vectors)
                        for idx, item in enumerate(all_tokens_for_pca):
                            coords_payload.append({
                                "token": item["token"],
                                "x": float(xy[idx, 0]),
                                "y": float(xy[idx, 1]),
                                "category": item["category"]
                            })
                else:
                    # Fallback if context is empty
                    relevance_data = [{"text": context_text, "relevance": 1.0}] if context_text else []
                
                # Send done event with all stats + visualization
                done_payload = {
                    "type": "done",
                    "response": response_text,
                    "prompt_tokens": prompt_len,
                    "completion_tokens": completion_tokens,
                    "generation_time": generation_time,
                    "tokens_per_second": tokens_per_second,
                    "device": device.upper(),
                    "visualization": {
                        "tokens_atlas": coords_payload,
                        "relevance_heatmap": relevance_data,
                        "question_fingerprint": q_fingerprint if (sentences and question) else [],
                        "token_data": all_token_data,  # Full per-token generation data for cinema viz
                        "prompt_tokens": prompt_tokens_decoded # Input context token strings for visual mapping
                    }
                }
                _send_event(done_payload)
                # Final SSE sentinel
                if client_alive:
                    try:
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        pass

            except Exception as e:
                import traceback
                traceback.print_exc()
                # Try sending error via SSE (headers may already be sent)
                try:
                    err_evt = "data: " + json.dumps({"type": "error", "message": str(e)}) + "\n\n"
                    self.wfile.write(err_evt.encode("utf-8"))
                    self.wfile.flush()
                except Exception:
                    pass

# ----------------------------------------------------------------------
# 3. MAIN RUNNER
# ----------------------------------------------------------------------
if __name__ == "__main__":
    server_address = ("", PORT)
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    socketserver.ThreadingTCPServer.daemon_threads = True
    with socketserver.ThreadingTCPServer(server_address, ModelServerHandler) as httpd:
        print(f"\n==================================================")
        print(f"Context Explorer Server Running at: http://localhost:{PORT}")
        print(f"Press Ctrl+C to terminate.")
        print(f"==================================================")
        sys.stdout.flush()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            sys.exit(0)
