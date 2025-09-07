from tokenizers import ByteLevelBPETokenizer
from pathlib import Path
import os

# Paths
DATA_PATH = "data/mrwiki_text.txt"
TOKENIZER_DIR = "data/marathi_bpe_tokenizer"
OUTPUT_PATH = "data/mrwiki_text_tokenized.txt"

# Load the trained tokenizer
print("Loading custom Marathi BPE tokenizer...")
tokenizer = ByteLevelBPETokenizer(
    os.path.join(TOKENIZER_DIR, "vocab.json"),
    os.path.join(TOKENIZER_DIR, "merges.txt")
)

# Tokenize the text file line by line
print(f"Tokenizing {DATA_PATH} ...")
with open(DATA_PATH, encoding="utf-8") as fin, open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        encoded = tokenizer.encode(line)
        fout.write(" ".join(map(str, encoded.ids)) + "\n")

print(f"Done. Tokenized output written to {OUTPUT_PATH}")
