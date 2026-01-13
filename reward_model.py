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


class ModelWrapper(torch.nn.Module):
    """Wrapper to filter out unsupported arguments for encoder models."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, **kwargs):
        # Remove arguments not supported by encoder models
        kwargs.pop('use_cache', None)
        return self.model(**kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


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
    # Load tokenizer and store a clean copy for saving later
    tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_NAME)
    print(f"Loaded tokenizer type: {tokenizer.__class__.__name__}")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    # Keep a reference to the original tokenizer before training
    original_tokenizer_path = REWARD_MODEL_NAME

    base_model = AutoModelForSequenceClassification.from_pretrained(
        REWARD_MODEL_NAME,
        num_labels=1,
    )

    # Wrap model to filter unsupported arguments
    model = ModelWrapper(base_model)

    dataset = load_reward_dataset(data_path)
    print(f"Loaded dataset with {len(dataset)} examples")
    print(f"Dataset columns: {dataset.column_names}")
    if len(dataset) > 0:
        print(f"First example: {dataset[0]}")

    # RewardTrainer handles tokenization internally, so we just pass the text fields

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=REWARD_BATCH_SIZE,
        num_train_epochs=REWARD_NUM_EPOCHS,
        eval_strategy="no",
        save_strategy="epoch",
        logging_steps=10,
        fp16=False,
        bf16=torch.cuda.is_available(),
        remove_unused_columns=False,
    )

    # Add attributes required by TRL RewardTrainer (from RewardConfig)
    training_args.model_init_kwargs = {}
    training_args.eos_token = None
    training_args.pad_token = None
    training_args.max_length = 4096
    training_args.chat_template_path = None
    training_args.disable_dropout = False
    training_args.pad_to_multiple_of = None
    training_args.dataset_num_proc = None
    training_args.center_rewards_coefficient = None
    training_args.activation_offloading = False

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Training reward model...")
    trainer.train()

    # Save the base model (unwrapped)
    base_model.save_pretrained(output_dir)

    # Save the original tokenizer (reload fresh to avoid contamination)
    print(f"Saving clean tokenizer from {original_tokenizer_path}...")
    import os
    import shutil

    # Use a temporary directory to get clean tokenizer files
    temp_dir = f"{output_dir}_temp_tokenizer"
    os.makedirs(temp_dir, exist_ok=True)

    # Download fresh tokenizer to temp directory
    clean_tokenizer = AutoTokenizer.from_pretrained(original_tokenizer_path, cache_dir=temp_dir)

    # Save to temp first
    clean_tokenizer.save_pretrained(temp_dir)

    # Copy only the essential tokenizer files (not any contaminated metadata)
    essential_files = [
        "tokenizer_config.json",
        "vocab.txt",  # DeBERTa uses vocab.txt
        "special_tokens_map.json",
        "tokenizer.json",
    ]

    for filename in essential_files:
        src = os.path.join(temp_dir, filename)
        dst = os.path.join(output_dir, filename)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  Copied: {filename}")

    # Clean up temp directory
    shutil.rmtree(temp_dir, ignore_errors=True)

    # CRITICAL FIX: Clean up tokenizer_config.json to remove SentencePiece contamination
    tokenizer_config_path = os.path.join(output_dir, "tokenizer_config.json")
    if os.path.exists(tokenizer_config_path):
        print("Cleaning tokenizer_config.json to remove SentencePiece contamination...")
        with open(tokenizer_config_path, 'r') as f:
            config = json.load(f)

        # Remove SentencePiece-related fields that cause Mistral warnings
        contaminated_fields = ["vocab_type", "sp_model_kwargs"]
        for field in contaminated_fields:
            if field in config:
                print(f"  Removing contaminated field: {field} = {config[field]}")
                del config[field]

        # Write back the cleaned config
        with open(tokenizer_config_path, 'w') as f:
            json.dump(config, f, indent=2)

    # CRITICAL: Remove any contaminating files that shouldn't be in DeBERTa tokenizer
    contaminating_files = [
        "spm.model",  # SentencePiece (LLaMA/Mistral)
        "merges.txt",  # BPE (GPT-2/Mistral)
        "vocab.json",  # BPE vocab (GPT-2/Mistral)
    ]

    for filename in contaminating_files:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            print(f"WARNING: Removing contaminating file: {filename}")
            os.remove(filepath)

    print(f"Reward model saved to {output_dir}")
    print(f"Tokenizer type: {clean_tokenizer.__class__.__name__}")

    # Verify the saved tokenizer can be loaded correctly
    print("Verifying saved tokenizer...")
    verification_tokenizer = AutoTokenizer.from_pretrained(output_dir)
    print(f"Verified tokenizer type: {verification_tokenizer.__class__.__name__}")


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
            max_length=4096,
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
        ref = rec.get("abstract", "")
        s1 = rec["summary_1"]
        s2 = rec["summary_2"]

        # Check if abstract is empty - if so, skip ROUGE/BERTScore or use fallback
        if not ref or ref.strip() == "":
            print(f"\nWarning: Empty abstract for {rec['arxiv_id']}, using first 500 chars of text as reference")
            # Use first 500 characters of paper text as fallback reference
            ref = rec.get("text", "")[:500] if rec.get("text") else ""

        # Only compute ROUGE/BERTScore if we have a valid reference
        if ref and ref.strip():
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
        else:
            # No valid reference available
            print(f"Warning: No valid reference for {rec['arxiv_id']}, skipping ROUGE/BERTScore")
            r1 = {"rouge1": None, "rouge2": None, "rougeL": None}
            r2 = {"rouge1": None, "rouge2": None, "rougeL": None}
            b1 = {"precision": [None], "recall": [None], "f1": [None]}
            b2 = {"precision": [None], "recall": [None], "f1": [None]}

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
        if r1["rougeL"] is not None:
            print("ROUGE-L: S1 =", r1["rougeL"], " | S2 =", r2["rougeL"])
            print("BERTScore F1: S1 =",
                  b1["f1"][0],
                  "| S2 =",
                  b2["f1"][0])
        else:
            print("ROUGE-L: N/A (no valid reference)")
            print("BERTScore F1: N/A (no valid reference)")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved detailed evaluation results to {output_path}")
    print("You can now inspect where reward scores agree/disagree with ROUGE/BERTScore.")

    return results
