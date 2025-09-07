#!/usr/bin/env bash
# download_marathi_wikipedia.sh
# Downloads and extracts the latest Marathi Wikipedia dump

set -e

DATA_DIR="$(dirname "$0")/data"
DUMP_URL="https://dumps.wikimedia.org/mrwiki/latest/mrwiki-latest-pages-articles.xml.bz2"
DUMP_FILE="$DATA_DIR/mrwiki-latest-pages-articles.xml.bz2"
EXTRACTED_FILE="$DATA_DIR/mrwiki-latest-pages-articles.xml"

mkdir -p "$DATA_DIR"

if [ ! -f "$DUMP_FILE" ]; then
    echo "Downloading Marathi Wikipedia dump..."
    wget -O "$DUMP_FILE" "$DUMP_URL"
else
    echo "Dump file already exists: $DUMP_FILE"
fi

if [ ! -f "$EXTRACTED_FILE" ]; then
    echo "Extracting dump..."
    bunzip2 -k "$DUMP_FILE"
else
    echo "Extracted file already exists: $EXTRACTED_FILE"
fi

echo "Done. Files are in $DATA_DIR"
