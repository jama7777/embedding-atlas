"""
Why a tokenizer is FROZEN once a model is trained on it.

We simulate a tiny 'model': an embedding table where each row (ID) has
learned the meaning of one word, according to tokenizer A's numbering.
Then we retrain the tokenizer (-> tokenizer B, different IDs) and feed
B's IDs into the model that was trained with A. Watch it break.
"""
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def train(corpus):
    tk = Tokenizer(BPE(unk_token="[UNK]"))
    tk.pre_tokenizer = Whitespace()
    tk.train_from_iterator(corpus, BpeTrainer(vocab_size=200,
                            special_tokens=["[UNK]"], show_progress=False))
    return tk

base  = ["the cat sat on the mat", "the dog sat on the log",
         "a cat and a dog play in the park"] * 30
extra = ["pizza tomato cheese basil oven naples",
         "espresso barista crema roast beans aroma"] * 30

A = train(base)            # original tokenizer
B = train(base + extra)    # RE-trained on almost-the-same data

# ---- "train" the model: each embedding row learns what its ID means,
#      using tokenizer A's numbering. We store the word as the 'meaning'. ----
model_memory = {}                      # id -> meaning the model learned (under A)
for word, idx in A.get_vocab().items():
    model_memory[idx] = word           # row `idx` now means `word`

def model_reads(token_ids):
    """The frozen model interprets each ID via what it learned under A."""
    return [model_memory.get(i, "??") for i in token_ids]

sentence = "the cat sat on the mat"
print("Sentence:", repr(sentence))
print()

# CASE 1 — use the tokenizer the model was trained with (A). Correct.
ids_A = A.encode(sentence).ids
print("USING TOKENIZER A (the one the model was trained on):")
print("   ids        :", ids_A)
print("   model reads :", model_reads(ids_A), " <- correct\n")

# CASE 2 — use the RE-trained tokenizer (B) with the SAME frozen model.
ids_B = B.encode(sentence).ids
print("USING RE-TRAINED TOKENIZER B (same model, frozen rows):")
print("   ids        :", ids_B)
print("   model reads :", model_reads(ids_B), " <- GARBAGE")
print()
print("Same English sentence. Tokenizer B's IDs point at rows the model")
print("filled with DIFFERENT words -> the meaning is scrambled.")
