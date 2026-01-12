"""
Reward model training and evaluation utilities.
"""
import json
from typing import List, Dict, Any

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
)
from trl import RewardTrainer
import evaluate


# Configuration
REWARD_MODEL_NAME = "microsoft/deberta-v3-base"
REWARD_NUM_EPOCHS = 3
REWARD_BATCH_SIZE = 4

# Device detection: CUDA > MPS > CPU
if torch.cuda.is_available():
    LLAMA_DEVICE = "cuda"
elif torch.backends.mps.is_available():
    LLAMA_DEVICE = "mps"
else:
    LLAMA_DEVICE = "cpu"


def auto_label_preferences_with_rouge(
    summary_records: List[Dict[str, Any]],
    output_path: str = "reward_data.jsonl"
) -> None:
    """
    Use ROUGE-L against the abstract as a weak preference label.

    This automatically creates preference pairs where:
    - chosen = summary with higher ROUGE-L vs abstract
    - rejected = the other summary

    Writes reward_data.jsonl with 'chosen' and 'rejected' fields suitable for RewardTrainer.

    Args:
        summary_records: List of dictionaries with 'abstract', 'summary_1', 'summary_2'
        output_path: Path to save the JSONL file
    """
    rouge = evaluate.load("rouge")
    data_for_jsonl = []

    for rec in summary_records:
        ref = rec["abstract"]
        s1 = rec["summary_1"]
        s2 = rec["summary_2"]

        # Compute per-summary ROUGE-L scores
        s1_score = rouge.compute(predictions=[s1], references=[ref])["rougeL"]
        s2_score = rouge.compute(predictions=[s2], references=[ref])["rougeL"]

        if s1_score >= s2_score:
            chosen, rejected = s1, s2
        else:
            chosen, rejected = s2, s1

        data_for_jsonl.append(
            {
                "chosen": chosen,
                "rejected": rejected,
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        for item in data_for_jsonl:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Wrote preference data to {output_path} ({len(data_for_jsonl)} pairs).")


def load_reward_dataset(path: str) -> Dataset:
    """
    Load reward modeling dataset from JSONL file.

    Args:
        path: Path to JSONL file with 'chosen' and 'rejected' fields

    Returns:
        HuggingFace Dataset object
    """
    dataset = load_dataset("json", data_files=path, split="train")
    return dataset


def train_reward_model(data_path: str = "reward_data.jsonl", output_dir: str = "reward_model"):
    """
    Train a reward model on preference data using TRL's RewardTrainer.

    The model learns to assign higher scores to preferred summaries (chosen)
    and lower scores to rejected summaries.

    Args:
        data_path: Path to JSONL file with preference pairs
        output_dir: Directory to save the trained model
    """
    print("Loading reward model + tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        REWARD_MODEL_NAME,
        num_labels=1,
    )

    dataset = load_reward_dataset(data_path)

    def preprocess(examples):
        """Tokenize chosen and rejected summaries separately."""
        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }

        for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
            tok_chosen = tokenizer(
                chosen,
                truncation=True,
                padding="max_length",
                max_length=512,
            )
            tok_rejected = tokenizer(
                rejected,
                truncation=True,
                padding="max_length",
                max_length=512,
            )

            new_examples["input_ids_chosen"].append(tok_chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(tok_chosen["attention_mask"])
            new_examples["input_ids_rejected"].append(tok_rejected["input_ids"])
            new_examples["attention_mask_rejected"].append(tok_rejected["attention_mask"])

        return new_examples

    dataset = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=REWARD_BATCH_SIZE,
        num_train_epochs=REWARD_NUM_EPOCHS,
        evaluation_strategy="no",
        save_strategy="epoch",
        logging_steps=10,
        fp16=torch.cuda.is_available(),
    )

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    print("Training reward model...")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Reward model saved to {output_dir}")


def score_summaries_with_reward_model(
    summaries: List[str],
    reward_model,
    reward_tokenizer,
) -> List[float]:
    """
    Compute scalar reward scores for each summary.

    Args:
        summaries: List of summary texts
        reward_model: Trained reward model
        reward_tokenizer: Tokenizer for the reward model

    Returns:
        List of reward scores (one per summary)
    """
    reward_model.eval()
    scores = []

    # Get the device of the model
    model_device = next(reward_model.parameters()).device

    for s in summaries:
        inputs = reward_tokenizer(
            s,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512,
        ).to(model_device)

        with torch.no_grad():
            out = reward_model(**inputs)
            # out.logits shape: [batch, 1]
            score = out.logits.squeeze().item()
        scores.append(score)

    return scores


def evaluate_summaries(
    summary_records: List[Dict[str, Any]],
    reward_model_dir: str = "reward_model",
    output_path: str = "evaluation_results.json"
) -> List[Dict[str, Any]]:
    """
    Evaluate summaries using ROUGE, BERTScore, and the trained reward model.

    Args:
        summary_records: List of dicts with 'arxiv_id', 'title', 'abstract', 'summary_1', 'summary_2'
        reward_model_dir: Directory containing the trained reward model
        output_path: Path to save evaluation results

    Returns:
        List of evaluation results for each paper
    """
    # Load reward model
    print("Loading reward model for evaluation...")
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_dir)
    reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_dir)
    reward_model.to(LLAMA_DEVICE)
    print(f"Reward model loaded on device: {LLAMA_DEVICE}")

    # Load evaluation metrics
    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")

    results = []

    for rec in summary_records:
        ref = rec["abstract"]
        s1 = rec["summary_1"]
        s2 = rec["summary_2"]

        # ROUGE
        r1 = rouge.compute(predictions=[s1], references=[ref])
        r2 = rouge.compute(predictions=[s2], references=[ref])

        # BERTScore
        b1 = bertscore.compute(
            predictions=[s1],
            references=[ref],
            lang="en",
        )
        b2 = bertscore.compute(
            predictions=[s2],
            references=[ref],
            lang="en",
        )

        # Reward model scores
        scores = score_summaries_with_reward_model(
            [s1, s2],
            reward_model,
            reward_tokenizer,
        )
        rm1, rm2 = scores

        result_entry = {
            "arxiv_id": rec["arxiv_id"],
            "title": rec["title"],
            "abstract": ref,
            "summary_1": s1,
            "summary_2": s2,
            "rouge_1": r1,
            "rouge_2": r2,
            "bertscore_1": b1,
            "bertscore_2": b2,
            "reward_1": rm1,
            "reward_2": rm2,
        }
        results.append(result_entry)

        print("\n=== Paper", rec["arxiv_id"], "===")
        print("Reward scores: S1 =", rm1, " | S2 =", rm2)
        print("ROUGE-L: S1 =", r1["rougeL"], " | S2 =", r2["rougeL"])
        print("BERTScore F1: S1 =",
              b1["f1"][0],
              "| S2 =",
              b2["f1"][0])

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved detailed evaluation results to {output_path}")
    print("You can now inspect where reward scores agree/disagree with ROUGE/BERTScore.")

    return results
