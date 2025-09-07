from datasets import load_dataset, Dataset
from transformers import LlamaTokenizer
import os

# Paths
DATA_PATH = "data/mrwiki_text.txt"
OUTPUT_PATH = "data/marathi_llama2_dataset"
TOKENIZER_NAME = "meta-llama/Llama-2-7b-hf"  # You can change to another Llama-2 variant if needed

# Load plain text
with open(DATA_PATH, encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

# Create Hugging Face Dataset
marathi_dataset = Dataset.from_dict({"text": lines})
marathi_dataset.save_to_disk(OUTPUT_PATH)

# Load Llama-2 tokenizer
print("Loading Llama-2 tokenizer...")
tokenizer = LlamaTokenizer.from_pretrained(TOKENIZER_NAME)

# Tokenize dataset

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

print("Tokenizing dataset...")
tokenized_dataset = marathi_dataset.map(tokenize_function, batched=True, num_proc=4)
tokenized_dataset.save_to_disk(OUTPUT_PATH + "_tokenized")

print("Done. Tokenized dataset saved to:", OUTPUT_PATH + "_tokenized")
