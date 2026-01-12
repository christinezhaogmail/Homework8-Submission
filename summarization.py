"""
Summarization utilities using LLaMA 3.1 model with unsloth optimization.
"""
import json
from typing import List, Dict, Any

# Try to import unsloth FIRST, before transformers
try:
    import unsloth
    from unsloth import FastLanguageModel
    USE_UNSLOTH = True
except (ImportError, NotImplementedError):
    USE_UNSLOTH = False
    print("Warning: unsloth not available, using standard transformers (slower)")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from data_utils import PaperData


# Configuration
LLAMA_MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
# Fallback model for non-unsloth environments
LLAMA_MODEL_FALLBACK = "meta-llama/Meta-Llama-3-8B-Instruct"

# Device detection: CUDA > MPS > CPU
if torch.cuda.is_available():
    LLAMA_DEVICE = "cuda"
elif torch.backends.mps.is_available():
    LLAMA_DEVICE = "mps"
else:
    LLAMA_DEVICE = "cpu"

MAX_INPUT_TOKENS = 4096
MAX_NEW_TOKENS = 256


def load_llama():
    """
    Load the LLaMA model using unsloth's FastLanguageModel for optimized inference.
    Falls back to standard transformers if unsloth is not available.

    Returns:
        Tuple of (model, tokenizer)
    """
    if USE_UNSLOTH:
        print(f"Loading LLaMA model with unsloth: {LLAMA_MODEL_NAME}")
        # Use unsloth's FastLanguageModel for optimized 4-bit loading
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=LLAMA_MODEL_NAME,
            max_seq_length=MAX_INPUT_TOKENS,
            dtype=None,  # Auto-detect dtype
            load_in_4bit=True,  # Use 4-bit quantization
        )
        # Set FastLanguageModel to inference mode for faster generation
        FastLanguageModel.for_inference(model)
    else:
        print(f"Loading LLaMA model with standard transformers: {LLAMA_MODEL_FALLBACK}")
        print("Note: For faster inference on GPU, run this on a system with NVIDIA/AMD/Intel GPU")

        # Use standard transformers with 4-bit quantization if available
        tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_FALLBACK, use_fast=False)

        if LLAMA_DEVICE == "cuda":
            # Use 4-bit quantization on CUDA
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model = AutoModelForCausalLM.from_pretrained(
                LLAMA_MODEL_FALLBACK,
                quantization_config=bnb_config,
                device_map="auto",
                dtype=torch.float16,
            )
        elif LLAMA_DEVICE == "mps":
            # On MPS (Mac M4), use float16 without quantization
            print(f"Running on MPS (Apple Silicon) - device: {LLAMA_DEVICE}")
            print("Note: For faster inference, run this on a system with NVIDIA/AMD/Intel GPU")
            model = AutoModelForCausalLM.from_pretrained(
                LLAMA_MODEL_FALLBACK,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            # Move model to MPS device
            model = model.to(LLAMA_DEVICE)
        else:
            # On CPU, use float32 without quantization
            print("Running on CPU - this will be very slow. Consider using a GPU server.")
            model = AutoModelForCausalLM.from_pretrained(
                LLAMA_MODEL_FALLBACK,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )

    # Important for LLaMA chat-style models:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def build_multimodal_text(paper: PaperData, max_chars: int = 6000) -> str:
    """
    Construct a text input that includes title, abstract, body text, and figure captions.

    This creates a "multimodal" representation by incorporating both textual content
    and figure caption information.

    Args:
        paper: PaperData object containing paper information
        max_chars: Maximum characters to include from paper body

    Returns:
        Formatted multimodal text string
    """
    figs = "\n".join(f"- {c}" for c in paper.figure_captions[:5])  # up to 5 captions
    body = paper.text
    if len(body) > max_chars:
        body = body[:max_chars]

    multimodal = (
        f"Title: {paper.title}\n\n"
        f"Abstract:\n{paper.abstract}\n\n"
        f"Selected figure captions:\n{figs}\n\n"
        f"Paper excerpt:\n{body}\n"
    )
    return multimodal


def llama_generate_summary(
    model,
    tokenizer,
    multimodal_text: str,
    prompt_style: str = "generic",
    temperature: float = 0.3,
    top_p: float = 0.9,
) -> str:
    """
    Generate a single summary from LLaMA using a chat-style prompt.

    Args:
        model: LLaMA model
        tokenizer: Model tokenizer
        multimodal_text: Input text including paper content and figure captions
        prompt_style: Style of prompt ("generic", "figure_aware", or other)
        temperature: Sampling temperature for generation
        top_p: Top-p (nucleus) sampling parameter

    Returns:
        Generated summary text
    """
    if prompt_style == "generic":
        system_prompt = (
            "You are an expert research assistant. Summarize the following research paper "
            "clearly and concisely for a graduate student."
        )
    elif prompt_style == "figure_aware":
        system_prompt = (
            "You are an expert research assistant. Summarize the following research paper, "
            "explicitly incorporating the information conveyed by the figures and their captions."
        )
    else:
        system_prompt = (
            "You are a helpful academic assistant. Provide a detailed yet concise summary of the paper."
        )

    # LLaMA 3.1 chat template format
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Summarize the following research paper excerpt:\n\n{multimodal_text}"}
    ]

    # Use tokenizer's chat template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(
        [prompt],
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    )

    # Get the device of the model (handles device_map="auto" cases)
    model_device = next(model.parameters()).device
    inputs = inputs.to(model_device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    # Extract only the generated summary (remove the prompt)
    if prompt in decoded:
        summary = decoded.replace(prompt, "").strip()
    else:
        summary = decoded.strip()

    return summary


def generate_summaries_for_papers(papers: List[PaperData], output_path: str = "summaries.json") -> List[Dict[str, Any]]:
    """
    Generate two summaries for each paper using different prompt styles and sampling parameters.

    Args:
        papers: List of PaperData objects
        output_path: Path to save the summaries JSON file

    Returns:
        List of dictionaries containing paper info and both summaries
    """
    model, tokenizer = load_llama()
    results = []

    for idx, paper in enumerate(papers):
        print(f"\n=== Summarizing paper {idx+1}/{len(papers)}: {paper.arxiv_id} ===")
        multimodal_text = build_multimodal_text(paper)

        # Summary 1: generic prompt, low temperature
        s1 = llama_generate_summary(
            model,
            tokenizer,
            multimodal_text,
            prompt_style="generic",
            temperature=0.3,
            top_p=0.9,
        )

        # Summary 2: figure-aware prompt, higher temp
        s2 = llama_generate_summary(
            model,
            tokenizer,
            multimodal_text,
            prompt_style="figure_aware",
            temperature=0.8,
            top_p=0.95,
        )

        results.append(
            {
                "arxiv_id": paper.arxiv_id,
                "title": paper.title,
                "abstract": paper.abstract,
                "text": paper.text,  # Include full text for fallback reference
                "summary_1": s1,
                "summary_2": s2,
            }
        )

        print("Summary 1 (generic):", s1[:200], "...")
        print("Summary 2 (figure-aware):", s2[:200], "...")

    # Save raw summaries
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results
