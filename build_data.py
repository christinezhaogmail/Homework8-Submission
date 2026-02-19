#!/usr/bin/env python3
"""
Phase 1-3: Data collection, summarization, and preference dataset creation.

This script:
- Downloads training papers from ArXiv (or uses existing PDFs)
- Generates two summaries per paper using LLaMA
- Creates preference labels using ROUGE-L scores
"""
import os
import glob
import re

from data_utils import (
    download_arxiv_pdfs,
    save_papers_to_json,
    extract_text_and_figures,
    PaperData,
)
from summarization import generate_summaries_for_papers
from reward_model import auto_label_preferences_with_rouge


def extract_abstract_from_text(text: str) -> str:
    """
    Extract the abstract from paper text.

    Looks for text between "Abstract" and common section headers.

    Args:
        text: Full paper text

    Returns:
        Extracted abstract or first 500 chars if not found
    """
    # Common patterns for abstract section
    abstract_patterns = [
        r'Abstract\s*\n+(.*?)\n+(?:1\.|Introduction|Keywords|1\s+Introduction)',
        r'ABSTRACT\s*\n+(.*?)\n+(?:1\.|Introduction|Keywords|1\s+Introduction)',
        r'Abstract\s*[:\-]?\s*\n+(.*?)\n+\d+\.?\s*[A-Z]',  # Abstract followed by numbered section
    ]

    for pattern in abstract_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            abstract = match.group(1).strip()
            # Clean up: remove excessive whitespace
            abstract = re.sub(r'\s+', ' ', abstract)
            # Limit length to reasonable abstract size
            if len(abstract) > 100 and len(abstract) < 3000:
                return abstract

    # Fallback: use first 500 characters after skipping potential title
    lines = text.split('\n')
    # Skip first few lines (likely title/authors)
    text_start = '\n'.join(lines[5:]) if len(lines) > 5 else text
    return text_start[:500].strip()


# Configuration
ARXIV_QUERY = "cs.LG"  # Computer Science - Machine Learning
NUM_TRAIN_PAPERS = 10

# File paths
PDF_DIR = "data/pdfs"
TRAIN_META_JSON = "train_papers.json"
TRAIN_SUMMARY_JSON = "summaries_train.json"
REWARD_JSONL = "reward_data.jsonl"


def main():
    print("=== PHASE 1-3: Data collection, summarization, preference dataset ===\n")

    # Check if PDFs exist
    if not os.path.exists(PDF_DIR) or len(os.listdir(PDF_DIR)) < NUM_TRAIN_PAPERS:
        print(f"Downloading {NUM_TRAIN_PAPERS} training papers from ArXiv...")
        train_papers = download_arxiv_pdfs(ARXIV_QUERY, NUM_TRAIN_PAPERS, PDF_DIR)
        save_papers_to_json(train_papers, TRAIN_META_JSON)
    else:
        print(f"Using existing PDFs from {PDF_DIR}")
        # Extract papers from existing PDFs
        pdf_files = glob.glob(f"{PDF_DIR}/*.pdf")[:NUM_TRAIN_PAPERS]
        print(f"Found {len(pdf_files)} PDF files")

        train_papers = []
        for pdf_path in pdf_files:
            arxiv_id = os.path.basename(pdf_path).replace(".pdf", "")
            text, figure_captions = extract_text_and_figures(pdf_path)

            # Extract abstract from the paper text
            abstract = extract_abstract_from_text(text)

            train_papers.append(
                PaperData(
                    arxiv_id=arxiv_id,
                    title=f"Paper {arxiv_id}",
                    abstract=abstract,
                    text=text,
                    figure_captions=figure_captions,
                )
            )
            print(f"  Extracted abstract for {arxiv_id}: {len(abstract)} chars")
        save_papers_to_json(train_papers, TRAIN_META_JSON)

    # Generate summaries
    print("\nGenerating summaries for training papers...")
    summary_records = generate_summaries_for_papers(train_papers, TRAIN_SUMMARY_JSON)

    # Build reward modeling data with chosen/rejected labels
    print("\nBuilding reward modeling dataset via ROUGE-based preference labels...")
    auto_label_preferences_with_rouge(summary_records, REWARD_JSONL)

    print("\n=== Data preparation complete! ===")


if __name__ == "__main__":
    main()
