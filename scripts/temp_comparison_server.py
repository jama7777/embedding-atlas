import http.server
import socketserver
import json
import os
import re
import sys
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA

PORT = 8766
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------------------------------------------------
# 1. LOAD MODEL
# ----------------------------------------------------------------------
model_local = "cache/local_qwen_0.5b"

if os.path.exists(model_local):
    print(f"Loading Model OFFLINE from local folder: '{model_local}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_local, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_local, output_hidden_states=True, local_files_only=True).to(device)
else:
    print("Loading Model online: Qwen/Qwen2.5-0.5B-Instruct...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", output_hidden_states=True).to(device)
model.eval()
print("Model loaded successfully!")

# ----------------------------------------------------------------------
# 2. GENERATION & ACTIVATIONS EXTRACTION PIPELINE
# ----------------------------------------------------------------------
def generate_and_trace(model, tokenizer, prompt, target_word, temperature=0.7):
    # Formulate instruction prompt using chat template
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Tokenize prompt inputs
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    prompt_len = input_ids.shape[1]
    
    generated_tokens = []
    step_predictions = []
    
    max_new_tokens = 100
    
    for step in range(max_new_tokens):
        # Run forward pass to get logits for next token
        with torch.no_grad():
            outputs = model(input_ids)
            
        next_token_logits = outputs.logits[0, -1, :]
        
        # Softmax probabilities (standard T=1 for display)
        probs = torch.softmax(next_token_logits.float(), dim=-1)
        
        # Top 15 probabilities
        top_probs, top_ids = torch.topk(probs, 15)
        
        # Extract candidate tokens and probabilities
        top_candidates = []
        for prob, tid in zip(top_probs, top_ids):
            tok_text = tokenizer.decode([tid.item()])
            top_candidates.append({
                "token": tok_text,
                "prob": float(prob.item())
            })
            
        # Sample or greedy selection based on temperature parameter
        if temperature > 0:
            scaled_logits = next_token_logits / max(temperature, 1e-5)
            probs_sampled = torch.softmax(scaled_logits.float(), dim=-1)
            next_token_id = torch.multinomial(probs_sampled, num_samples=1)[0].item()
        else:
            next_token_id = top_ids[0].item()
            
        generated_tokens.append(next_token_id)
        
        # Find the rank and probability of next_token_id in the original probs
        sorted_probs, sorted_ids = torch.sort(probs, descending=True)
        chosen_rank = int((sorted_ids == next_token_id).nonzero(as_tuple=True)[0].item() + 1)
        chosen_prob = float(probs[next_token_id].item())
        
        # Decode the actual token generated at this step
        actual_tok_text = tokenizer.decode([next_token_id])
        
        # Record step predictions
        step_predictions.append({
            "token": actual_tok_text,
            "token_id": next_token_id,
            "chosen_prob": chosen_prob,
            "chosen_rank": chosen_rank,
            "candidates": top_candidates
        })
        
        # Append to input_ids
        input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], device=device)], dim=1)
        
        # Stop on End of Sequence (EOS) token
        if next_token_id == tokenizer.eos_token_id:
            break
            
    # Run a final forward pass over the full prompt+response sequence to gather hidden_states
    with torch.no_grad():
        outputs = model(input_ids)
    hidden_states = outputs.hidden_states
    
    # Get all tokens for layer trajectory mapping
    all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    # Scan for target word occurrences in all_tokens
    target_positions = []
    cleaned_tokens = []
    for idx, token in enumerate(all_tokens):
        clean = token.replace("Ġ", "").replace(" ", "").replace("Ġ", "").replace("Ġ", "").lower()
        clean = re.sub(r'[^a-zA-Z0-9]', '', clean)
        cleaned_tokens.append(clean)
        
        if clean == target_word.lower() and len(clean) > 0:
            target_positions.append(idx)
            
    # Substring search fallback if no exact matches found
    if not target_positions:
        for idx, clean in enumerate(cleaned_tokens):
            if target_word.lower() in clean and len(clean) > 0:
                target_positions.append(idx)
                break
                
    # Fallback to last token if still empty
    if not target_positions:
        target_positions = [len(all_tokens) - 1]
        
    num_layers = len(hidden_states)
    
    # Gather activation vectors across layers for all target word occurrences
    flat_vectors = []
    for pos in target_positions:
        for layer in range(num_layers):
            vec = hidden_states[layer][0, pos].float().cpu().numpy()
            flat_vectors.append(vec)
            
    flat_vectors = np.array(flat_vectors, dtype=np.float32)
    
    # Run PCA to project to 2D
    pca = PCA(n_components=2, random_state=42)
    flat_xy = pca.fit_transform(flat_vectors)
    
    # Re-organize coordinates
    occurrences_data = []
    for o_idx, pos in enumerate(target_positions):
        coords_x = []
        coords_y = []
        for l_idx in range(num_layers):
            flat_idx = o_idx * num_layers + l_idx
            coords_x.append(float(flat_xy[flat_idx, 0]))
            coords_y.append(float(flat_xy[flat_idx, 1]))
            
        # Get localized surrounding context
        context_start = max(0, pos - 4)
        context_end = min(len(all_tokens), pos + 5)
        context_words = []
        for c_i in range(context_start, context_end):
            tok_dec = tokenizer.decode([input_ids[0][c_i].item()]).strip()
            if c_i == pos:
                context_words.append(f"<{tok_dec}>")
            else:
                context_words.append(tok_dec)
        context_str = " ".join(context_words)
        
        origin = "prompt" if pos < prompt_len else "response"
        
        occurrences_data.append({
            "token_index": pos,
            "token_str": all_tokens[pos],
            "context": context_str,
            "origin": origin,
            "x": coords_x,
            "y": coords_y
        })
        
    # Get final response text
    response_ids = input_ids[0][prompt_len:]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
    
    return {
        "response": response_text,
        "occurrences": occurrences_data,
        "num_layers": num_layers,
        "generation_steps": bookkeeping_clean_steps(step_predictions)
    }

def bookkeeping_clean_steps(steps):
    # Minor cleanup to ensure JSON compliance and safe tokens
    cleaned = []
    for s in steps:
        cleaned_candidates = []
        for c in s["candidates"]:
            cleaned_candidates.append({
                "token": c["token"] if c["token"] else "<space>",
                "prob": c["prob"]
            })
        cleaned.append({
            "token": s["token"] if s["token"] else "<space>",
            "chosen_prob": s.get("chosen_prob", 0.0),
            "chosen_rank": s.get("chosen_rank", 1),
            "candidates": cleaned_candidates
        })
    return cleaned

# ----------------------------------------------------------------------
# 2.1 STEP-BY-STEP DECODING HELPERS
# ----------------------------------------------------------------------
def step_variant(history: list[dict], gen_ids: list[int], temperature: float):
    torch.manual_seed(42)
    text = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
    
    if gen_ids:
        gen_ids_tensor = torch.tensor([gen_ids], dtype=torch.long, device=device)
        input_ids = torch.cat([prompt_ids, gen_ids_tensor], dim=1)
    else:
        input_ids = prompt_ids
        
    input_len = prompt_ids.shape[1]
    
    with torch.no_grad():
        outputs = model(input_ids)
        pred_logits = outputs.logits[0, input_len - 1 :, :]
        
    pred_probs = torch.softmax(pred_logits.float(), dim=-1)
    
    steps_metadata = []
    for j in range(len(gen_ids)):
        selected_id = gen_ids[j]
        probs_j = pred_probs[j]
        selected_prob = probs_j[selected_id].item()
        
        sorted_probs_j, sorted_ids_j = torch.sort(probs_j, descending=True)
        selected_rank = int((sorted_ids_j == selected_id).nonzero(as_tuple=True)[0].item() + 1)
        
        top_probs_j, top_ids_j = torch.topk(probs_j, 15)
        top_15_j = []
        for p, idx in zip(top_probs_j.tolist(), top_ids_j.tolist()):
            top_15_j.append({
                "token": tokenizer.decode([idx]),
                "prob": p
            })
            
        steps_metadata.append({
            "token": tokenizer.decode([selected_id]),
            "chosen_prob": selected_prob,
            "chosen_rank": selected_rank,
            "candidates": top_15_j
        })
        
    next_probs = pred_probs[-1]
    top_probs, top_ids = torch.topk(next_probs, 15)
    
    top_15 = []
    for p, idx in zip(top_probs.tolist(), top_ids.tolist()):
        top_15.append({
            "token": tokenizer.decode([idx]),
            "prob": p
        })
        
    next_token_logits = pred_logits[-1]
    if temperature > 0:
        torch.manual_seed(42 + len(gen_ids))
        scaled_logits = next_token_logits / max(temperature, 1e-5)
        probs_sampled = torch.softmax(scaled_logits.float(), dim=-1)
        next_token_id = torch.multinomial(probs_sampled, num_samples=1)[0].item()
    else:
        next_token_id = top_ids[0].item()
        
    is_eos = False
    if gen_ids and gen_ids[-1] == tokenizer.eos_token_id:
        is_eos = True
        
    return {
        "steps_metadata": steps_metadata,
        "top_15": top_15,
        "next_token_id": next_token_id,
        "is_eos": is_eos
    }

def complete_variant(history: list[dict], gen_ids: list[int], temperature: float):
    current_gen_ids = list(gen_ids)
    max_new_tokens = 100 - len(gen_ids)
    if max_new_tokens <= 0:
        return gen_ids
        
    for step in range(max_new_tokens):
        res = step_variant(history, current_gen_ids, temperature)
        if res["is_eos"]:
            break
        current_gen_ids.append(res["next_token_id"])
        if res["next_token_id"] == tokenizer.eos_token_id:
            break
            
    return current_gen_ids

# ----------------------------------------------------------------------
# 3. HTTP SERVER REQUEST HANDLER
# ----------------------------------------------------------------------
class ModelServerHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Quiet requests log output for readability
        pass

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html" or self.path == "/temp_comparison.html":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            
            html_path = "html/temp_comparison.html"
            if os.path.exists(html_path):
                with open(html_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.wfile.write(b"<h1>Error: html/temp_comparison.html not found!</h1>")
        elif self.path == "/plotly.min.js":
            self.send_response(200)
            self.send_header("Content-type", "application/javascript")
            self.end_headers()
            
            js_path = "html/plotly.min.js"
            if os.path.exists(js_path):
                with open(js_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.wfile.write(b"console.error('plotly.min.js not found');")
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

    def do_POST(self):
        if self.path == "/chat":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            params = json.loads(post_data.decode('utf-8'))
            
            prompt = params.get("prompt", "")
            target_word = params.get("target_word", "bank").strip()
            temp_b = float(params.get("temperature", 0.7))
            
            if not prompt or not target_word:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Prompt and Target Word are required"}).encode('utf-8'))
                return
                
            print(f"\n[POST /chat] Prompt: '{prompt}' | Target Word: '{target_word}' | Sample Temp: {temp_b}")
            
            try:
                print("  Running Model A (Greedy: Temp = 0.0)...")
                res_a = generate_and_trace(model, tokenizer, prompt, target_word, temperature=0.0)
                
                print(f"  Running Model B (Sampling: Temp = {temp_b})...")
                res_b = generate_and_trace(model, tokenizer, prompt, target_word, temperature=temp_b)
                
                response_payload = {
                    "model_a": res_a,
                    "model_b": res_b
                }
                
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response_payload).encode('utf-8'))
                print("  Inference and PCA complete! JSON response sent.")
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_response(500)
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
        elif self.path == "/step":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            params = json.loads(post_data.decode('utf-8'))
            
            history = params.get("history", [])
            generated = params.get("generated", {})
            temp_b = float(params.get("temperature", 0.7))
            
            try:
                res_a = step_variant(history, generated.get("model_a", []), temperature=0.0)
                res_b = step_variant(history, generated.get("model_b", []), temperature=temp_b)
                
                response_payload = {
                    "model_a": {
                        "generation_steps": res_a["steps_metadata"],
                        "next_token_id": res_a["next_token_id"],
                        "top_15": res_a["top_15"],
                        "is_eos": res_a["is_eos"],
                        "full_ids": generated.get("model_a", [])
                    },
                    "model_b": {
                        "generation_steps": res_b["steps_metadata"],
                        "next_token_id": res_b["next_token_id"],
                        "top_15": res_b["top_15"],
                        "is_eos": res_b["is_eos"],
                        "full_ids": generated.get("model_b", [])
                    }
                }
                
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response_payload).encode('utf-8'))
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_response(500)
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
        elif self.path == "/complete":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            params = json.loads(post_data.decode('utf-8'))
            
            history = params.get("history", [])
            generated = params.get("generated", {})
            temp_b = float(params.get("temperature", 0.7))
            
            try:
                full_ids_a = complete_variant(history, generated.get("model_a", []), temperature=0.0)
                full_ids_b = complete_variant(history, generated.get("model_b", []), temperature=temp_b)
                
                res_a = step_variant(history, full_ids_a, temperature=0.0)
                res_b = step_variant(history, full_ids_b, temperature=temp_b)
                
                response_payload = {
                    "model_a": {
                        "generation_steps": res_a["steps_metadata"],
                        "next_token_id": res_a["next_token_id"],
                        "top_15": res_a["top_15"],
                        "is_eos": res_a["is_eos"],
                        "full_ids": full_ids_a
                    },
                    "model_b": {
                        "generation_steps": res_b["steps_metadata"],
                        "next_token_id": res_b["next_token_id"],
                        "top_15": res_b["top_15"],
                        "is_eos": res_b["is_eos"],
                        "full_ids": full_ids_b
                    }
                }
                
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response_payload).encode('utf-8'))
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_response(500)
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))

# ----------------------------------------------------------------------
# 4. MAIN RUNNER
# ----------------------------------------------------------------------
if __name__ == "__main__":
    server_address = ("", PORT)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(server_address, ModelServerHandler) as httpd:
        print(f"\n==================================================")
        print(f"LLM Temperature Comparison Server Running at: http://localhost:{PORT}")
        print(f"Press Ctrl+C to terminate.")
        print(f"==================================================")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            sys.exit(0)
