import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# Define models to save locally
MODELS = {
    "SmolLM2-360M-Instruct": {
        "repo": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "local_dir": "cache/local_smollm2_360m"
    },
    "Qwen2.5-0.5B-Instruct": {
        "repo": "Qwen/Qwen2.5-0.5B-Instruct",
        "local_dir": "cache/local_qwen_0.5b"
    }
}

for name, cfg in MODELS.items():
    print(f"\n==================================================")
    print(f"Downloading {name} from HF Hub...")
    print(f"==================================================")
    
    tokenizer = AutoTokenizer.from_pretrained(cfg["repo"])
    model = AutoModelForCausalLM.from_pretrained(cfg["repo"], output_hidden_states=True)
    
    print(f"\nSaving {name} locally to: '{cfg['local_dir']}'...")
    os.makedirs(cfg["local_dir"], exist_ok=True)
    tokenizer.save_pretrained(cfg["local_dir"])
    model.save_pretrained(cfg["local_dir"])
    
    print(f"Successfully saved {name} locally.")

print("\nAll models have been serialized locally on disk successfully!")
