# Marathi LLM Project Setup



# BolBot: Marathi GPT-2 LLM

BolBot is an open-source project to train a GPT-2 style large language model (LLM) for the Marathi language, using Wikipedia dumps and a custom tokenizer. The codebase includes data preparation, tokenizer training, model architecture, and training scripts.

---

This project aims to train a large language model (LLM) on Marathi Wikipedia dumps.

## Data Download Instructions

**Note:** The actual Wikipedia data files are not included in this repository. To download the latest Marathi Wikipedia dump, run:

```bash
wget https://dumps.wikimedia.org/mrwiki/latest/mrwiki-latest-pages-articles.xml.bz2 -O data/mrwiki-latest-pages-articles.xml.bz2
bunzip2 -k data/mrwiki-latest-pages-articles.xml.bz2
```

Or use the provided script:

```bash
./download_marathi_wikipedia.sh
```

This will place the dump in the `data/` directory (which is gitignored).

## Steps
1. Download Marathi Wikipedia dump
2. Extract and preprocess text
3. Tokenize and clean data
4. Choose and configure model architecture
5. Train or fine-tune the model
6. Evaluate and deploy

## Data Directory
All raw and processed data will be stored in the `data/` directory.

---

Next step: Download the latest Marathi Wikipedia dump from https://dumps.wikimedia.org/mrwiki/latest/ (file: `mrwiki-latest-pages-articles.xml.bz2`).
