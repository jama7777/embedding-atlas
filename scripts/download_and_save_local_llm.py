import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# ----------------------------------------------------------------------
# Config & Paths
# ----------------------------------------------------------------------
model_name = "HuggingFaceTB/SmolLM2-135M"
local_dir = "cache/local_smollm2_135m"

print(f"Downloading pre-trained model and tokenizer: '{model_name}'...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)

print(f"\nSaving model and tokenizer locally to: '{local_dir}'...")
os.makedirs(local_dir, exist_ok=True)
tokenizer.save_pretrained(local_dir)
model.save_pretrained(local_dir)

print("\nModel saved successfully! You can now load it fully offline using:")
print(f"  tokenizer = AutoTokenizer.from_pretrained('{local_dir}')")
print(f"  model = AutoModelForCausalLM.from_pretrained('{local_dir}')")
