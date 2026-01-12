#!/usr/bin/env python3
"""
Phase 5-6: Evaluation and comparison.

This script:
- Loads summaries from training
- Scores them using the trained reward model
- Computes ROUGE and BERTScore
- Compares all metrics
"""
import os
import json

from reward_model import evaluate_summaries


# File paths
TRAIN_SUMMARY_JSON = "summaries_train.json"
REWARD_MODEL_DIR = "reward_model"
RESULTS_JSON = "evaluation_results.json"


def main():
    print("\n=== PHASE 5-6: Evaluation and comparison ===\n")

    if not os.path.exists(TRAIN_SUMMARY_JSON):
        print("Error: No training summaries found. Run build_data.py first.")
        return

    # Load existing summaries
    with open(TRAIN_SUMMARY_JSON, "r", encoding="utf-8") as f:
        eval_summaries = json.load(f)

    print(f"Evaluating {len(eval_summaries)} papers...")

    # Evaluate summaries
    evaluate_summaries(eval_summaries, REWARD_MODEL_DIR, RESULTS_JSON)

    print("\n=== Evaluation complete! ===")


if __name__ == "__main__":
    main()
