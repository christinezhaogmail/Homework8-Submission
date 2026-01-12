# Paper Summarization with Reward Modeling

**Week 8 Assignment: Multimodal Summarization and Reward Modeling**

This project implements an end-to-end pipeline for generating and evaluating academic paper summaries using large language models (LLaMA 3.1) and reward modeling (DeBERTa-v3). The system demonstrates how to align AI-generated summaries with human preferences through reinforcement learning from human feedback (RLHF) techniques.

## Table of Contents

- [Overview](#overview)
- [Learning Objectives](#learning-objectives)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Phases](#pipeline-phases)
- [Configuration](#configuration)
- [Output Files](#output-files)
- [Evaluation Metrics](#evaluation-metrics)
- [Example Results](#example-results)
- [Troubleshooting](#troubleshooting)

## Overview

Effective summarization is critical in research because it distills large, complex documents into concise overviews that highlight key insights. However, automatically evaluating the quality of generated summaries is challenging. Traditional metrics like ROUGE and BERTScore rely on lexical overlap and can miss nuances like semantic correctness or coherence.

This project addresses this gap by:
1. Generating multiple candidate summaries using LLaMA 3.1 (8B) Instruct model
2. Creating preference labels based on ROUGE scores
3. Training a reward model to predict human-aligned quality scores
4. Comparing reward model predictions with traditional metrics

## Learning Objectives

- Generate abstractive summaries of academic documents using LLaMA 3.1 (8B) Instruct
- Collect two candidate summaries per paper and create preference labels
- Prepare datasets of summary pairs for reward model training
- Train a reward model (DeBERTa-v3) on preference data
- Evaluate summaries using ROUGE, BERTScore, and the trained reward model
- Analyze alignment between reward model scores and automatic metrics

## Project Structure

```
Homework8-Submission/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── Class 8 Homework.ipynb        # Jupyter notebook with instructions
│
├── data/
│   └── pdfs/                     # Academic papers in PDF format (10 papers)
│       ├── 2601.05103v1.pdf
│       ├── 2601.05104v1.pdf
│       └── ...
│
├── data_utils.py                 # PDF parsing and ArXiv downloading
├── summarization.py              # LLaMA model loading and summary generation
├── reward_model.py               # Reward model training and evaluation
└── main.py                       # Main pipeline orchestration
```

### Module Descriptions

#### `data_utils.py`
- **Purpose**: Handle data collection and PDF processing
- **Key Functions**:
  - `download_arxiv_pdfs()`: Download papers from ArXiv
  - `extract_text_and_figures()`: Extract text and figure captions from PDFs
  - `save_papers_to_json()` / `load_papers_from_json()`: Serialize paper data

#### `summarization.py`
- **Purpose**: Generate summaries using LLaMA 3.1 model
- **Key Functions**:
  - `load_llama()`: Load optimized 4-bit quantized LLaMA model using unsloth
  - `build_multimodal_text()`: Combine paper text with figure captions
  - `llama_generate_summary()`: Generate summaries with different prompting strategies
  - `generate_summaries_for_papers()`: Process multiple papers

#### `reward_model.py`
- **Purpose**: Train and use reward models for summary quality prediction
- **Key Functions**:
  - `auto_label_preferences_with_rouge()`: Create preference labels from ROUGE scores
  - `train_reward_model()`: Fine-tune DeBERTa-v3 on preference data
  - `score_summaries_with_reward_model()`: Score summaries using trained model
  - `evaluate_summaries()`: Compute all metrics (ROUGE, BERTScore, reward scores)

#### `main.py`
- **Purpose**: Orchestrate the complete pipeline
- **Phases**: Data collection, summarization, reward training, evaluation

## Installation

### Prerequisites

- Python 3.11 or higher
- **For GPU server (recommended)**:
  - CUDA-capable NVIDIA/AMD/Intel GPU
  - 16GB+ VRAM recommended
  - Uses optimized unsloth for fast 4-bit inference
- **For Mac M4/Apple Silicon testing**:
  - Code will fall back to standard transformers
  - Model loading will be slower
  - Use GPU server for actual training/evaluation
- HuggingFace Hub token (for accessing LLaMA models)
- OpenAI API key (optional, for additional features)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Homework8-Submission
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export HF_HUB_TOKEN="your_huggingface_token"
export OPENAI_API_KEY="your_openai_key"  # Optional
```

### Key Dependencies

- `unsloth`: Optimized LLaMA model loading with 4-bit quantization
- `transformers`: HuggingFace transformers library
- `trl`: Transformer Reinforcement Learning for reward modeling
- `torch`: PyTorch deep learning framework
- `datasets`: HuggingFace datasets library
- `evaluate`: Evaluation metrics (ROUGE, BERTScore)
- `PyMuPDF` (fitz): PDF text extraction
- `arxiv`: ArXiv API client

## Usage

### Quick Start (All Phases)

Run the complete pipeline:

```bash
python main.py --phase all
```

### Run Individual Phases

**Phase 1-3: Data Collection and Summarization**
```bash
python main.py --phase build_data
```
This will:
- Use existing PDFs in `data/pdfs/` (or download from ArXiv if needed)
- Generate two summaries per paper using different prompting strategies
- Create preference labels based on ROUGE-L scores
- Save results to `summaries_train.json` and `reward_data.jsonl`

**Phase 4: Train Reward Model**
```bash
python main.py --phase train_reward
```
This will:
- Load preference data from `reward_data.jsonl`
- Fine-tune DeBERTa-v3 model on chosen/rejected summary pairs
- Save trained model to `reward_model/` directory

**Phase 5-6: Evaluation**
```bash
python main.py --phase evaluate
```
This will:
- Load trained reward model
- Compute ROUGE, BERTScore, and reward scores for all summaries
- Save comprehensive results to `evaluation_results.json`


## Pipeline Phases

### Phase 1: Data Collection
- **Input**: PDFs in `data/pdfs/` or ArXiv query
- **Process**: Extract text and figure captions from academic papers
- **Output**: `train_papers.json` with structured paper data

### Phase 2: Summary Generation
- **Model**: `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`
- **Strategy**: Generate two summaries per paper:
  1. **Summary 1**: Generic prompt, low temperature (0.3) - More focused
  2. **Summary 2**: Figure-aware prompt, high temperature (0.8) - More creative
- **Output**: `summaries_train.json`

### Phase 3: Preference Labeling
- **Method**: Use ROUGE-L scores against paper abstracts
- **Logic**: Summary with higher ROUGE-L = "chosen", other = "rejected"
- **Output**: `reward_data.jsonl` with preference pairs

### Phase 4: Reward Model Training
- **Base Model**: `microsoft/deberta-v3-base`
- **Training**: 3 epochs with batch size 4
- **Objective**: Learn to assign higher scores to preferred summaries
- **Output**: Trained model in `reward_model/` directory

### Phase 5-6: Evaluation & Comparison
- **Metrics Computed**:
  - **ROUGE-1, ROUGE-2, ROUGE-L**: Lexical overlap with abstract
  - **BERTScore**: Semantic similarity using BERT embeddings
  - **Reward Score**: Trained model's quality prediction
- **Analysis**: Compare metric agreement and disagreement cases
- **Output**: `evaluation_results.json`

## Configuration

### Key Parameters (in respective modules)

**Summarization** (`summarization.py`):
```python
LLAMA_MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
MAX_INPUT_TOKENS = 2048
MAX_NEW_TOKENS = 256
```

**Reward Model** (`reward_model.py`):
```python
REWARD_MODEL_NAME = "microsoft/deberta-v3-base"
REWARD_NUM_EPOCHS = 3
REWARD_BATCH_SIZE = 4
```

**Data Collection** (`main.py`):
```python
ARXIV_QUERY = "cs.LG"  # ArXiv category
NUM_TRAIN_PAPERS = 10
NUM_EVAL_PAPERS = 10
```

## Output Files

| File | Description |
|------|-------------|
| `train_papers.json` | Structured data for training papers (text, abstract, figures) |
| `summaries_train.json` | Two summaries per paper with metadata |
| `reward_data.jsonl` | Preference pairs (chosen/rejected) for training |
| `reward_model/` | Trained DeBERTa-v3 reward model weights |
| `evaluation_results.json` | Complete evaluation metrics for all summaries |

## Evaluation Metrics

### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
- Measures lexical overlap between generated and reference summaries
- **ROUGE-1**: Unigram overlap
- **ROUGE-2**: Bigram overlap
- **ROUGE-L**: Longest common subsequence
- **Range**: 0-1 (higher is better)

### BERTScore
- Measures semantic similarity using BERT embeddings
- Captures meaning beyond surface-level word matching
- Computes precision, recall, and F1
- **Range**: 0-1 (higher is better)

### Reward Model Score
- Learned metric from preference data
- Predicts human-aligned quality judgments
- Captures aspects like coherence, informativeness, factual consistency
- **Range**: Uncalibrated (higher = better quality)

## Example Results

After running the pipeline, `evaluation_results.json` contains entries like:

```json
{
  "arxiv_id": "2601.05103v1",
  "title": "Paper Title",
  "abstract": "Original abstract...",
  "summary_1": "First generated summary...",
  "summary_2": "Second generated summary...",
  "rouge_1": {"rouge1": 0.45, "rouge2": 0.23, "rougeL": 0.38},
  "rouge_2": {"rouge1": 0.42, "rouge2": 0.21, "rougeL": 0.35},
  "bertscore_1": {"precision": [0.89], "recall": [0.87], "f1": [0.88]},
  "bertscore_2": {"precision": [0.86], "recall": [0.84], "f1": [0.85]},
  "reward_1": 2.34,
  "reward_2": 1.87
}
```

### Interpretation

- **High ROUGE + High Reward**: Summary is both lexically similar and high-quality
- **Low ROUGE + High Reward**: Summary paraphrases well but uses different words
- **High ROUGE + Low Reward**: Summary copies text but lacks coherence
- **BERTScore**: Often correlates better with reward scores than ROUGE

## Troubleshooting

### Common Issues

**1. Unsloth Not Supported on Mac M4/Apple Silicon**
```
Error: NotImplementedError: Unsloth currently only works on NVIDIA, AMD and Intel GPUs.

Solution: The code automatically falls back to standard transformers
- Mac M4 will use standard transformers (slower but works)
- For actual training/evaluation, use a GPU server
- No code changes needed, fallback is automatic
```

**2. Out of Memory Error**
```
Solution: Reduce batch size or use smaller model
- Set REWARD_BATCH_SIZE = 2 in reward_model.py
- Use gradient accumulation for effective larger batch
```

**3. Model Download Fails**
```
Solution: Check HuggingFace token
export HF_HUB_TOKEN="your_token"
huggingface-cli login
```

**4. CUDA Out of Memory**
```
Solution: Enable CPU offloading or use smaller sequences
- Reduce MAX_INPUT_TOKENS in summarization.py
- Use CPU: Set LLAMA_DEVICE = "cpu"
```

**5. PDF Extraction Issues**
```
Solution: Some PDFs may have complex layouts
- Check data/pdfs/ for corrupted files
- Manually verify text extraction quality
```

### Performance Tips

1. **Use GPU**: CUDA significantly speeds up inference and training
2. **Batch Processing**: Process multiple summaries together when possible
3. **Caching**: Save intermediate results to avoid recomputation
4. **4-bit Quantization**: Already enabled via unsloth for memory efficiency

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{paper_summarization_reward,
  title={Paper Summarization with Reward Modeling},
  author={Class 8 Homework},
  year={2026},
  howpublished={\url{https://github.com/your-repo/homework8-submission}}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Unsloth**: For optimized LLaMA model loading
- **HuggingFace**: For transformers, datasets, and TRL libraries
- **Meta AI**: For LLaMA 3.1 model
- **Microsoft**: For DeBERTa-v3 model

## Contact

For questions or issues, please open an issue on the GitHub repository or contact the course instructors.

---

**Note**: This project is for educational purposes as part of Week 8 homework on multimodal summarization and reward modeling.
