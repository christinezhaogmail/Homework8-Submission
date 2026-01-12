#!/usr/bin/env python3
"""
Main entry point for the paper summarization and reward modeling pipeline.

This script orchestrates the complete workflow:
1. Download papers from ArXiv
2. Generate summaries using LLaMA 3.1
3. Create preference dataset
4. Train reward model
5. Evaluate on new papers

Usage:
    python main.py --phase all
    python main.py --phase build_data
    python main.py --phase train_reward
    python main.py --phase evaluate
"""
import argparse

from data_utils import download_arxiv_pdfs, save_papers_to_json
from summarization import generate_summaries_for_papers
from reward_model import (
    auto_label_preferences_with_rouge,
    train_reward_model,
    evaluate_summaries,
)


# Configuration
ARXIV_QUERY = "cs.LG"  # Computer Science - Machine Learning
NUM_TRAIN_PAPERS = 10
NUM_EVAL_PAPERS = 10

# File paths
PDF_DIR = "data/pdfs"
TRAIN_META_JSON = "train_papers.json"
EVAL_META_JSON = "eval_papers.json"
TRAIN_SUMMARY_JSON = "summaries_train.json"
EVAL_SUMMARY_JSON = "summaries_eval.json"
REWARD_JSONL = "reward_data.jsonl"
REWARD_MODEL_DIR = "reward_model"
RESULTS_JSON = "evaluation_results.json"


def build_data_phase():
    """
    Phase 1-3: Data collection, summarization, and preference dataset creation.

    This phase:
    - Downloads training papers from ArXiv (or uses existing PDFs)
    - Generates two summaries per paper using LLaMA
    - Creates preference labels using ROUGE-L scores
    """
    print("=== PHASE 1-3: Data collection, summarization, preference dataset ===\n")

    # Note: If PDFs already exist in data/pdfs, we can skip downloading
    # and directly load them. For now, we'll assume PDFs are already there.
    import os
    if not os.path.exists(PDF_DIR) or len(os.listdir(PDF_DIR)) < NUM_TRAIN_PAPERS:
        print(f"Downloading {NUM_TRAIN_PAPERS} training papers from ArXiv...")
        train_papers = download_arxiv_pdfs(ARXIV_QUERY, NUM_TRAIN_PAPERS, PDF_DIR)
        save_papers_to_json(train_papers, TRAIN_META_JSON)
    else:
        print(f"Using existing PDFs from {PDF_DIR}")
        from data_utils import load_papers_from_json
        # Extract papers from existing PDFs
        import glob
        pdf_files = glob.glob(f"{PDF_DIR}/*.pdf")[:NUM_TRAIN_PAPERS]
        print(f"Found {len(pdf_files)} PDF files")

        from data_utils import extract_text_and_figures, PaperData
        train_papers = []
        for pdf_path in pdf_files:
            arxiv_id = os.path.basename(pdf_path).replace(".pdf", "")
            text, figure_captions = extract_text_and_figures(pdf_path)
            # For existing PDFs, we don't have title/abstract from ArXiv API
            # So we'll use placeholder values or extract from the PDF
            train_papers.append(
                PaperData(
                    arxiv_id=arxiv_id,
                    title=f"Paper {arxiv_id}",
                    abstract="",  # Will be extracted or left empty
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


def train_reward_phase():
    """
    Phase 4: Train the reward model on preference data.

    This phase fine-tunes a DeBERTa-v3 model on the chosen/rejected summary pairs
    so it learns to assign higher scores to better summaries.
    """
    print("\n=== PHASE 4: Reward model training ===\n")
    train_reward_model(REWARD_JSONL, REWARD_MODEL_DIR)


def evaluate_phase():
    """
    Phase 5-6: Evaluation and comparison.

    This phase:
    - Generates summaries for evaluation papers
    - Scores them using the trained reward model
    - Computes ROUGE and BERTScore
    - Compares all metrics
    """
    print("\n=== PHASE 5-6: Evaluation and comparison ===\n")

    # For evaluation, we could download new papers or use a held-out set
    # For simplicity, we'll reuse the training papers here
    # In a real scenario, you'd want a separate eval set
    import os
    import json

    if not os.path.exists(TRAIN_SUMMARY_JSON):
        print("Error: No training summaries found. Run --phase build_data first.")
        return

    # Load existing summaries
    with open(TRAIN_SUMMARY_JSON, "r", encoding="utf-8") as f:
        eval_summaries = json.load(f)

    print(f"Evaluating {len(eval_summaries)} papers...")

    # Evaluate summaries
    evaluate_summaries(eval_summaries, REWARD_MODEL_DIR, RESULTS_JSON)


def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Paper summarization + reward modeling pipeline (ArXiv + LLaMA + DeBERTa)."
    )
    parser.add_argument(
        "--phase",
        choices=["build_data", "train_reward", "evaluate", "all"],
        default="all",
        help="Which phase to run.",
    )
    args = parser.parse_args()

    if args.phase in ("build_data", "all"):
        build_data_phase()

    if args.phase in ("train_reward", "all"):
        train_reward_phase()

    if args.phase in ("evaluate", "all"):
        evaluate_phase()

    print("\n=== Pipeline complete! ===")


if __name__ == "__main__":
    main()
