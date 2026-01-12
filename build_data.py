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

from data_utils import (
    download_arxiv_pdfs,
    save_papers_to_json,
    extract_text_and_figures,
    PaperData,
)
from summarization import generate_summaries_for_papers
from reward_model import auto_label_preferences_with_rouge


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
            train_papers.append(
                PaperData(
                    arxiv_id=arxiv_id,
                    title=f"Paper {arxiv_id}",
                    abstract="",
                    text=text,
                    figure_captions=figure_captions,
                )
            )
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
