import fitz  # pymupdf
import json
import argparse
import os
import re
from typing import List, Dict

def clean_text(text: str) -> str:
    """Basic text cleaning."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_text_from_pdf(pdf_path: str, chunk_by_page: bool = True) -> List[Dict[str, str]]:
    """
    Extracts text from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file.
        chunk_by_page: If True, each page is a separate chunk.
        
    Returns:
        List of dictionaries containing text and metadata (page number).
    """
    doc = fitz.open(pdf_path)
    extracted_data = []

    print(f"Extracting text from {pdf_path}...")
    
    for page_num, page in enumerate(doc):
        text = page.get_text()
        cleaned_text = clean_text(text)
        
        if cleaned_text:
            extracted_data.append({
                "page": page_num + 1,
                "text": cleaned_text,
                "source": os.path.basename(pdf_path)
            })
            
    return extracted_data

def main():
    parser = argparse.ArgumentParser(description="Extract text from PDF for Lancer RPG ingestion.")
    parser = argparse.ArgumentParser(description="Extract text from PDF for Lancer RPG ingestion.")
    parser.add_argument("--pdf_path", type=str, required=True, help="Path to the PDF file.")
    parser.add_argument("--output_path", type=str, default="dataset/raw_extracted.json", help="Path to save the extracted JSON.")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    data = extract_text_from_pdf(args.pdf_path)
    
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        
    print(f"Extracted {len(data)} pages. Saved to {args.output_path}")

if __name__ == "__main__":
    main()

