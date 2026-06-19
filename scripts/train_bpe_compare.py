"""
Does adding text to the TRAINING corpus change the IDs a BPE tokenizer assigns?
We train two BPE tokenizers with the SAME algorithm/settings:
  A: on a base corpus
  B: on the base corpus + a couple extra paragraphs
...then compare the IDs they give to the same words.
"""
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

base = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "a cat and a dog play in the park",
    "cats and dogs are common pets",
    "the park has a big green lawn",
] * 20   # repeat so frequencies are stable

# the SAME corpus, plus two extra paragraphs heavy on new words
extra = [
    "pizza pizza pizza tomato cheese basil oven naples margherita",
    "espresso espresso barista crema roast beans aroma morning ritual",
] * 20

def train(corpus):
    tk = Tokenizer(BPE(unk_token="[UNK]"))
    tk.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=200, special_tokens=["[UNK]"], show_progress=False)
    tk.train_from_iterator(corpus, trainer)
    return tk

A = train(base)            # corpus A
B = train(base + extra)    # corpus A + extra paragraphs

print(f"vocab size  A={A.get_vocab_size()}   B={B.get_vocab_size()}")
print()
print(f"{'word':10} {'id in A':>8} {'id in B':>8}   same?")
for w in ["the", "cat", "dog", "park", "sat", "mat"]:
    ia = A.token_to_id(w)
    ib = B.token_to_id(w)
    same = "yes" if ia == ib else "NO  <-- shifted"
    print(f"{w:10} {str(ia):>8} {str(ib):>8}   {same}")

print()
# encode the SAME sentence with each -> different id sequences
s = "the cat sat on the mat"
print("sentence:", repr(s))
print("  A ids:", A.encode(s).ids)
print("  B ids:", B.encode(s).ids)
