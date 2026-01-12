#!/usr/bin/env python3
"""
Phase 4: Train the reward model on preference data.

This script fine-tunes a DeBERTa-v3 model on the chosen/rejected summary pairs
so it learns to assign higher scores to better summaries.
"""
from reward_model import train_reward_model


# File paths
REWARD_JSONL = "reward_data.jsonl"
REWARD_MODEL_DIR = "reward_model"


def main():
    print("\n=== PHASE 4: Reward model training ===\n")
    train_reward_model(REWARD_JSONL, REWARD_MODEL_DIR)
    print("\n=== Training complete! ===")


if __name__ == "__main__":
    main()
