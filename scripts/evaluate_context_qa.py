import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    model_dir = "cache/local_qwen_0.5b"
    
    if not os.path.exists(model_dir):
        print(f"Error: Model directory '{model_dir}' not found. Please run 'python3 scripts/download_chat_models.py' first.")
        return

    # Check for hardware acceleration (MPS for macOS, CUDA for NVIDIA, otherwise CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.bfloat16 # or torch.float32/float16
        print("Using Apple Silicon GPU acceleration (MPS) with bfloat16.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16
        print("Using NVIDIA GPU acceleration (CUDA) with bfloat16.")
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        print("Using CPU with float32.")

    print(f"\nLoading model and tokenizer from '{model_dir}'...")
    start_time = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        device_map=None
    ).to(device)
    
    print(f"Loaded successfully in {time.time() - start_time:.2f} seconds!")

    # 1. Model context window and max generation tokens info
    config = model.config
    context_window = getattr(config, "max_position_embeddings", 32768)
    
    print("\n" + "="*50)
    print("MODEL SPECIFICATIONS & LIMITS (PRACTICAL & THEORETICAL)")
    print("="*50)
    print(f"Model Name:          Qwen2.5-0.5B-Instruct")
    print(f"Parameters:          ~494 Million")
    print(f"Theoretical Context: {context_window} tokens")
    print(f"Practical Context:   We can feed up to {context_window} tokens, though memory on consumer hardware")
    print(f"                     may restrict long contexts (especially on CPU or limited VRAM).")
    print(f"Max Generation Limit: In Hugging Face, 'max_new_tokens' is customizable up to the total sequence length")
    print(f"                     (context_tokens + generated_tokens <= {context_window}).")
    print("="*50)

    # 2. Feed text and ask questions
    sample_context = (
        "Project Aetheris is a next-generation neural interface developed by NeuroLink Labs in the year 2042. "
        "It operates by mapping quantum state fluctuations in synaptic vesicles, specifically targeting the CA3 region of the hippocampus. "
        "Unlike previous systems that required invasive surgical microelectrodes, Aetheris utilizes a non-invasive sprayable "
        "graphene-based mesh (called AeroMesh) that self-assembles upon entering the bloodstream via inhalation. "
        "The project is currently led by Dr. Helena Vance, who previously headed the synaptic mapping initiative at CERN. "
        "A key vulnerability of the AeroMesh is its sensitivity to strong localized magnetic fields (above 2.5 Tesla), "
        "which can cause the graphene structures to lose coherence and dissolve harmlessly into biological carbon compounds."
    )
    
    sample_questions = [
        "What is the name of the sprayable graphene-based mesh used in Project Aetheris?",
        "Who is the lead scientist of Project Aetheris and where did they work previously?",
        "What is a key vulnerability of Project Aetheris's mesh, and at what threshold does it fail?",
        "How is the mesh introduced to the body?"
    ]

    print("\n--- SAMPLE CONTEXT ---")
    print(sample_context)
    print("-" * 22)

    for i, question in enumerate(sample_questions, 1):
        print(f"\n[Question {i}]: {question}")
        
        # Formulate chat format
        messages = [
            {"role": "system", "content": "You are a precise assistant. Answer the user's question using ONLY the provided context."},
            {"role": "user", "content": f"Context:\n{sample_context}\n\nQuestion:\n{question}"}
        ]
        
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
        
        prompt_tokens_len = inputs["input_ids"].shape[1]
        
        # Run inference
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.1,  # Low temperature for factual QA
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )
            
        generated_ids = outputs[0][prompt_tokens_len:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        response_tokens_len = len(generated_ids)
        
        print(f"Prompt length: {prompt_tokens_len} tokens")
        print(f"Response length: {response_tokens_len} tokens")
        print(f"Answer:\n{response}")
        print("-" * 40)

if __name__ == "__main__":
    main()
