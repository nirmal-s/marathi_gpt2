from tokenizers import ByteLevelBPETokenizer
from pathlib import Path
import os

# Paths
DATA_PATH = "data/mrwiki_text.txt"
TOKENIZER_DIR = "data/marathi_bpe_tokenizer"
VOCAB_SIZE = 32000

# Make sure output directory exists
os.makedirs(TOKENIZER_DIR, exist_ok=True)

# Initialize a Byte-Pair Encoding (BPE) tokenizer
print("Training ByteLevelBPETokenizer on Marathi Wikipedia text...")
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(files=[DATA_PATH], vocab_size=VOCAB_SIZE, min_frequency=2, special_tokens=[
    "<s>", "<pad>", "</s>", "<unk>", "<mask>"
])

tokenizer.save_model(TOKENIZER_DIR)
print(f"Tokenizer trained and saved to {TOKENIZER_DIR}")
