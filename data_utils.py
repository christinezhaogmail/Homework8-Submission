"""
Data utilities for downloading papers from ArXiv and extracting text/figures from PDFs.
"""
import os
import json
from dataclasses import dataclass
from typing import List, Tuple

import arxiv
import fitz  # PyMuPDF


@dataclass
class PaperData:
    """Data structure for storing paper information."""
    arxiv_id: str
    title: str
    abstract: str
    text: str
    figure_captions: List[str]


def download_arxiv_pdfs(query: str, num_papers: int, out_dir: str) -> List[PaperData]:
    """
    Download papers from ArXiv and extract their content.

    Args:
        query: ArXiv query category (e.g., "cs.LG")
        num_papers: Number of papers to download
        out_dir: Output directory for PDFs

    Returns:
        List of PaperData objects containing paper information
    """
    os.makedirs(out_dir, exist_ok=True)

    search = arxiv.Search(
        query=f"cat:{query}",
        max_results=num_papers,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    papers: List[PaperData] = []
    for result in search.results():
        arxiv_id = result.get_short_id()
        title = result.title
        abstract = result.summary
        pdf_path = os.path.join(out_dir, f"{arxiv_id}.pdf")

        print(f"Downloading {arxiv_id}: {title[:80]}...")
        result.download_pdf(filename=pdf_path)

        text, figure_captions = extract_text_and_figures(pdf_path)
        papers.append(
            PaperData(
                arxiv_id=arxiv_id,
                title=title,
                abstract=abstract,
                text=text,
                figure_captions=figure_captions,
            )
        )

    return papers


def extract_text_and_figures(pdf_path: str) -> Tuple[str, List[str]]:
    """
    Extract text and figure captions from a PDF file.

    This is a simple extractor that treats any line starting with 'Figure' or 'Fig.'
    as a caption. It's not perfect but works well for most academic papers.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Tuple of (full_text, list of figure captions)
    """
    doc = fitz.open(pdf_path)
    all_text_parts = []
    figure_captions = []

    for page in doc:
        t = page.get_text("text")
        all_text_parts.append(t)

        for line in t.splitlines():
            stripped = line.strip()
            if stripped.startswith("Figure ") or stripped.startswith("Fig. "):
                figure_captions.append(stripped)

    doc.close()
    full_text = "\n".join(all_text_parts)
    return full_text, figure_captions


def save_papers_to_json(papers: List[PaperData], path: str):
    """
    Save paper data to a JSON file.

    Args:
        papers: List of PaperData objects
        path: Output JSON file path
    """
    data = []
    for p in papers:
        data.append(
            {
                "arxiv_id": p.arxiv_id,
                "title": p.title,
                "abstract": p.abstract,
                "text": p.text,
                "figure_captions": p.figure_captions,
            }
        )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_papers_from_json(path: str) -> List[PaperData]:
    """
    Load paper data from a JSON file.

    Args:
        path: Input JSON file path

    Returns:
        List of PaperData objects
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    papers = []
    for d in data:
        papers.append(
            PaperData(
                arxiv_id=d["arxiv_id"],
                title=d["title"],
                abstract=d["abstract"],
                text=d["text"],
                figure_captions=d["figure_captions"],
            )
        )
    return papers
