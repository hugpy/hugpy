# pdf_pre_ocr_extractor.py
import logging
logging.basicConfig(level=logging.WARNING)
logging.getLogger("pdfminer").setLevel(logging.WARNING)
logging.getLogger("pdfplumber").setLevel(logging.WARNING)
import pdfplumber
from PyPDF2 import PdfReader

# -------------------------
# QUALITY SCORING
# -------------------------
def score_text_quality(text: str) -> float:
    if not text:
        return 0.0
    length = len(text.strip())
    if length == 0:
        return 0.0
    alpha_ratio = sum(c.isalpha() for c in text) / length
    printable_ratio = sum(c.isprintable() for c in text) / length
    words = text.split()
    avg_word_len = (
        sum(len(w) for w in words) / len(words)
        if words else 0
    )
    score = 0
    score += min(alpha_ratio * 2, 1.0)
    score += printable_ratio
    score += min(avg_word_len / 6, 1.0)
    return score / 3


# -------------------------
# CLEANING
# -------------------------
def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", "")
    text = "\n".join(line.strip() for line in text.splitlines())
    return text


# -------------------------
# EXTRACTION METHODS (generators)
# -------------------------
def extract_with_pdfplumber(pdf_path: str, first_page: int = None, last_page: int = None):
    """Extract text from PDF using pdfplumber, yield page by page."""
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        start = first_page or 0
        end = (last_page or total_pages - 1) + 1
        
        for i in range(start, min(end, total_pages)):
            page = pdf.pages[i]
            try:
                text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
            except Exception:
                text = ""
            text = normalize_text(text)
            score = score_text_quality(text)
            yield {
                "page_index": i,
                "text": text,
                "score": score,
                "method": "pdfplumber"
            }


def extract_with_pypdf2(pdf_path: str, first_page: int = None, last_page: int = None):
    """Extract text from PDF using PyPDF2, yield page by page."""
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    start = first_page or 0
    end = (last_page or total_pages - 1) + 1
    
    for i in range(start, min(end, total_pages)):
        page = reader.pages[i]
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = normalize_text(text)
        score = score_text_quality(text)
        yield {
            "page_index": i,
            "text": text,
            "score": score,
            "method": "pypdf2"
        }


# -------------------------
# MERGE STRATEGY (streaming)
# -------------------------
def merge_page_results(pages_a_gen, pages_b_gen):
    """Merge two page generators, yield best of each pair."""
    pages_a = list(pages_a_gen)
    pages_b = list(pages_b_gen)
    
    for i in range(max(len(pages_a), len(pages_b))):
        pa = pages_a[i] if i < len(pages_a) else None
        pb = pages_b[i] if i < len(pages_b) else None
        if pa and pb:
            best = pa if pa["score"] >= pb["score"] else pb
        else:
            best = pa or pb
        yield best


# -------------------------
# MAIN PIPELINE
# -------------------------
def extract_pdf_pre_ocr(
    pdf_path: str,
    first_page: int = None,
    last_page: int = None,
    page_threshold: float = 0.5,
    doc_threshold: float = 0.6
):
    """
    Extract text from PDF pages (specific range or all).
    
    Args:
        pdf_path: Path to PDF
        first_page: Start page (0-indexed, None = page 0)
        last_page: End page inclusive (0-indexed, None = last page)
        page_threshold: Score threshold for OCR trigger
        doc_threshold: Document-level score threshold
        
    Returns:
        Dict with merged pages, scores, OCR needs
    """
    # Dual extraction with page range
    pages_plumber = extract_with_pdfplumber(pdf_path, first_page, last_page)
    pages_pypdf2 = extract_with_pypdf2(pdf_path, first_page, last_page)
    
    # Merge and collect
    merged_pages = list(merge_page_results(pages_plumber, pages_pypdf2))
    
    # Compute document score
    scores = [p["score"] for p in merged_pages if p["text"].strip()]
    doc_score = sum(scores) / len(scores) if scores else 0
    
    # Determine OCR needs
    ocr_pages = [
        p["page_index"]
        for p in merged_pages
        if p["score"] < page_threshold
    ]
    requires_full_ocr = doc_score < doc_threshold
    
    return {
        "pages": merged_pages,
        "doc_score": doc_score,
        "ocr_pages": ocr_pages,
        "requires_full_ocr": requires_full_ocr
    }


# -------------------------
# CONVENIENCE: Extract single page
# -------------------------
def extract_single_pdf_page_text(pdf_path: str, page_index: int):
    """Extract a specific page only."""
    result = extract_pdf_pre_ocr(pdf_path, first_page=page_index, last_page=page_index)
    return next((item.get('text') for item in result["pages"] if item), None)
