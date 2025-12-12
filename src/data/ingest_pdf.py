import sys
import os
import json
import argparse
import re
import yaml
from typing import List, Dict

try:
    import fitz  # pymupdf
except ImportError:
    print("❌ Error: PyMuPDF (fitz) is not installed.")
    print("Please run: pip install pymupdf")
    sys.exit(1)

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
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")
        
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
    parser = argparse.ArgumentParser(description="Extract text from PDF for TTRPG ingestion.")
    parser.add_argument("--config", type=str, help="Path to config YAML (optional).")
    parser.add_argument("--pdf_path", type=str, help="Path to the PDF file (overrides config).")
    parser.add_argument("--output_path", type=str, default="dataset/raw_extracted.json", help="Path to save the extracted JSON.")
    
    args = parser.parse_args()
    
    pdf_path = args.pdf_path
    output_path = args.output_path
    
    # Load from config if provided
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
            ingest_config = config.get("ingest", {})
            
            # Variables for path formatting
            path_vars = {
                "project_name": config.get("project_name", "default"),
                "dataset_tag": config.get("dataset_tag", "v1")
            }
            
            if not pdf_path:
                pdf_path = ingest_config.get("pdf_path")
            
            # Resolve raw_output_path with variables
            if args.output_path == "dataset/raw_extracted.json" and ingest_config.get("raw_output_path"):
                raw_out = ingest_config.get("raw_output_path")
                output_path = raw_out.format(**path_vars)

    if not pdf_path:
        parser.error("pdf_path must be provided via argument or config file.")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        data = extract_text_from_pdf(pdf_path)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        print(f"Extracted {len(data)} pages. Saved to {output_path}")
    except Exception as e:
        print(f"Error during extraction: {e}")

if __name__ == "__main__":
    main()
