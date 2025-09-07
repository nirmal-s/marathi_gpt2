import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Usage: python extract_wiki_text.py data/mrwiki-latest-pages-articles.xml data/mrwiki_text.txt

def clean_text(text):
    # Remove MediaWiki markup, templates, and HTML tags (basic)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\{\{.*?\}\}', '', text, flags=re.DOTALL)
    text = re.sub(r'\[\[].*?\]\]', '', text)
    text = re.sub(r'\[\[].*?\[\[]', '', text)
    text = re.sub(r'\[\[].*?\]', '', text)
    text = re.sub(r'\[\[].*?\[\[', '', text)
    text = re.sub(r'\[\[].*?\]', '', text)
    text = re.sub(r'\[\[].*?\[\[', '', text)
    text = re.sub(r'\[\[].*?\]', '', text)
    text = re.sub(r'\[\[].*?\[\[', '', text)
    text = re.sub(r'\[\[].*?\]', '', text)
    text = re.sub(r'\[\[].*?\[\[', '', text)
    text = re.sub(r'\[\[].*?\]', '', text)
    text = re.sub(r'\[\[].*?\[\[', '', text)
    text = re.sub(r'\[\[].*?\]', '', text)
    text = re.sub(r'\[\[].*?\[\[', '', text)
    text = re.sub(r'\[\[].*?\]', '', text)
    text = re.sub(r'\[\[].*?\[\[', '', text)
    text = re.sub(r'\[\[].*?\]', '', text)
    text = re.sub(r'\[\[].*?\[\[', '', text)
    text = re.sub(r'\[\[].*?\]', '', text)
    text = re.sub(r'\[\[].*?\[\[', '', text)
    text = re.sub(r'\[\[].*?\]', '', text)
    text = re.sub(r'\[\[].*?\[\[', '', text)
    text = re.sub(r'\[\[].*?\]', '', text)
    text = re.sub(r'\[\[].*?\[\[', '', text)
    text = re.sub(r'\[\[].*?\]', '', text)
    text = re.sub(r'\[\[].*?\[\[', '', text)
    text = re.sub(r'\[\[].*?\]', '', text)
    text = re.sub(r'\[\[].*?\[\[', '', text)
    text = re.sub(r'\[\[].*?\]', '', text)
    text = re.sub(r'\[\[].*?\[\[', '', text)
    text = re.sub(r'\[\[].*?\]', '', text)
    text = re.sub(r'\[\[].*?\[\[', '', text)
    text = re.sub(r'\[\[].*?\]', '', text)
    text = re.sub(r'\[\[].*?\[\[', '', text)
    text = re.sub(r'\[\[].*?\]', '', text)
    text = re.sub(r'\[\[].*?\[\[', '', text)
    text = re.sub(r'\[\[].*?\]', '', text)
    text = re.sub(r'\[\[].*?\[\[', '', text)
    text = re.sub(r'\[\[].*?\]', '', text)
    text = re.sub(r'\[\[].*?\[\[', '', text)
    text = re.sub(r'\[\[].*?\]', '', text)
    text = re.sub(r'\[\[].*?\[\[', '', text)
    text = re.sub(r'\[\[].*?\]', '', text)
    text = re.sub(r'\[\[].*?\[\[', '', text)
    text = re.sub(r'\[\[].*?\]', '', text)
    text = re.sub(r'\[\[].*?\[\[', '', text)
    text = re.sub(r'\[\[].*?\]', '', text)
    text = re.sub(r'\[\[].*?\[\[', '', text)
    text = re.sub(r'\[\[].*?\]', '', text)
    text = re.sub(r'\[\[].*?\[\[', '', text)
    text = re.sub(r'\[\[].*?\]', '', text)
    text = re.sub(r'\[\[].*?\[\[', '', text)
    text = re.sub(r'\[\[].*?\]', '', text)
    text = re.sub(r'\[\[].*?\[\[', '', text)
    text = re.sub(r'\[\[].*?\]', '', text)
    text = re.sub(r'\[\[].*?\[\[', '', text)
    text = re.sub(r'\[\[].*?\]', '', text)
    text = re.sub(r'\[\[].*?\[\[', '', text)
    text = re.sub(r'\[\[].*?\]', '', text)
    text = re.sub(r'\[\[].*?\[\[', '', text)
    text = re.sub(r'\[\[].*?\]', '', text)
    text = re.sub(r'\[\[].*?\[\[', '', text)
    text = re.sub(r'\[\[].*?\]', '', text)
    text = re.sub(r'\[\[].*?\[\[', '', text)
    text = re.sub(r'\[\[].*?\]', '', text)
    text = re.sub(r'\[\[].*?\[\[', '', text)
    text = re.sub(r'\[\[].*?\]', '', text)
    text = re.sub(r'\[\[].*?\[\[', '', text)
    text = re.sub(r'\[\[].*?\]', '', text)
    text = re.sub(r'\[\[].*?\[\[', '', text)
    text = re.sub(r'\[\[].*?\]', '', text)
    text = re.sub(r'\[\[].*?\[\[', '', text)
    text = re.sub(r'\[\[].*?\]', '', text)
    text = re.sub(r'\[\[].*?\[\[', '', text)
    text = re.sub(r'\[\[].*?\]', '', text)
    return text.strip()

def extract_text(xml_path, out_path):
    print(f"Extracting text from {xml_path} ...")
    context = ET.iterparse(xml_path, events=("end",))
    page_count = 0
    written_count = 0
    debug_limit = 10
    first_page_tags = None
    nsmap = {'mw': 'http://www.mediawiki.org/xml/export-0.11/'}
    with open(out_path, "w", encoding="utf-8") as out:
        for event, elem in context:
            tag = elem.tag
            if tag.endswith('page'):
                page_count += 1
                if first_page_tags is None:
                    first_page_tags = [child.tag for child in elem]
                    print(f"First <page> child tags: {first_page_tags}")
                # Use namespace-aware tags
                title = elem.findtext("mw:title", namespaces=nsmap)
                ns_val = elem.findtext("mw:ns", namespaces=nsmap)
                text = elem.findtext("mw:revision/mw:text", namespaces=nsmap)
                if page_count <= debug_limit:
                    print(f"Page {page_count}: title={title!r}, ns={ns_val!r}, text_len={len(text) if text else 0}")
                if ns_val == "0" and text:
                    cleaned = clean_text(text)
                    if cleaned:
                        out.write(cleaned + "\n\n")
                        written_count += 1
                elem.clear()
    print(f"Processed {page_count} pages. Wrote {written_count} articles to {out_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_wiki_text.py <input_xml> <output_txt>")
        sys.exit(1)
    extract_text(sys.argv[1], sys.argv[2])
