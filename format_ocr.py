"""
Format raw OCR text into a readable structured document.

Usage:
    python format_ocr.py input.ocr [output.txt]
    python format_ocr.py --from-training training/example_95.json [--doc 0] [output.txt]
"""

import argparse
import json
import re
import sys
from pathlib import Path


def format_ocr_text(raw: str) -> str:
    """Transform raw OCR text into readable formatted text."""
    # Replace literal \n with actual newlines
    text = raw.replace('\\n', '\n')

    # Collapse multiple spaces/tabs
    text = re.sub(r'[ \t]+', ' ', text)

    # Collapse 3+ newlines into 2
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Strip trailing spaces per line
    text = '\n'.join(line.strip() for line in text.splitlines())

    # Add blank line before section headers for readability
    lines = text.splitlines()
    result = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        is_header = False
        if stripped and len(stripped) >= 3:
            # ALL CAPS line
            if stripped == stripped.upper() and any(c.isalpha() for c in stripped):
                is_header = True
            # Numbered sections: "1.", "1.1 ", "Článek", "Čl."
            if re.match(r'^(\d+\.(?:\d+)?)\s', stripped):
                is_header = True
            if re.match(r'^(článek|čl\.)\s', stripped, re.IGNORECASE):
                is_header = True

        if is_header and i > 0 and result and result[-1].strip():
            result.append('')
        result.append(line)

    return '\n'.join(result).strip()


def main():
    parser = argparse.ArgumentParser(description='Format OCR text')
    parser.add_argument('input', help='Input file (.ocr, .txt, or .json)')
    parser.add_argument('output', nargs='?', help='Output file (default: stdout)')
    parser.add_argument('--from-training', action='store_true',
                        help='Input is a training JSON file')
    parser.add_argument('--doc', type=int, default=0,
                        help='Document index in training JSON (default: 0)')
    parser.add_argument('--all-docs', action='store_true',
                        help='Format all documents from training JSON')
    args = parser.parse_args()

    input_path = Path(args.input)

    if args.from_training or input_path.suffix == '.json':
        with open(input_path) as f:
            data = json.load(f)
        docs = data.get('input', data).get('documents', [])

        if args.all_docs:
            parts = []
            for doc in docs:
                fname = doc['filename']
                formatted = format_ocr_text(doc['ocr_text'])
                parts.append(f"{'='*60}\n=== {fname} ===\n{'='*60}\n\n{formatted}")
            text = '\n\n'.join(parts)
        else:
            if args.doc >= len(docs):
                print(f'Error: doc index {args.doc} out of range (0-{len(docs)-1})')
                sys.exit(1)
            doc = docs[args.doc]
            print(f'Formatting: {doc["filename"]} ({len(doc["ocr_text"])} chars)',
                  file=sys.stderr)
            text = format_ocr_text(doc['ocr_text'])
    else:
        with open(input_path) as f:
            raw = f.read()
        text = format_ocr_text(raw)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(text)
        print(f'Saved to {args.output} ({len(text.splitlines())} lines)',
              file=sys.stderr)
    else:
        print(text)


if __name__ == '__main__':
    main()
