import logging
import os.path as osp
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional, List, Dict, Any
from bs4 import BeautifulSoup
# abstract_ocr (paddle OCR) and abstract_webtools (scraping: selenium/playwright)
# are heavy, OPTIONAL extras. They are imported lazily inside the extractor
# wrappers below so `import abstract_hugpy_dev` works on a base / phone install
# without them. Install with abstract_hugpy_dev[ocr] / [web] to use these.
from hugpy_media.seo.pdf_utils import _analyze, PDFSeoReport
from typing import Literal

from PyPDF2 import PdfReader
from pydantic import BaseModel, ConfigDict

from hugpy_engine.runtime import runner_for
from hugpy_engine.schemas.chat_schemas import ChatRequest, ChatResult
from hugpy_platform.constants import DEFAULT_CHAT_MODEL, SOURCEKIND

from hugpy_media.pdfs.utils import extract_single_pdf_page_text
from hugpy_media.vision.vision_coder import deepcoder_image_analysis
from hugpy_media.whisper_model.src.model.execute import transcribe_file

logger = logging.getLogger(__name__)

def get_num_pdf_pages(pdf_path):
    reader = PdfReader(pdf_path)
    return len(reader.pages)
# ---- schemas ---------------------------------------------------------------


class GenParams(BaseModel):
    model_config = ConfigDict(frozen=True)
    max_new_tokens: int = 100
    temperature: float = 0.6
    top_p: float = 0.95
    use_chat_template: bool = False
    messages: Optional[List[Dict[str, str]]] = None
    do_sample: bool = False
    unbounded:bool=True

    def to_kwargs(self) -> dict:
        return self.model_dump()

@dataclass(frozen=True)
class AnalyzePresets:
    """How _analyze is parameterized for a given scope."""
    scope: str = "full"
    summary_preset: str = "article"
    keyword_preset: str = "seo"


# ---- extractor registry ----------------------------------------------------

# An extractor turns a source (path or url) into plain text.
Extractor = Callable[[str], str]
_EXTRACTORS: dict[str, Extractor] = {}


def register_extractor(kind: str, fn: Extractor) -> None:
    if kind in _EXTRACTORS:
        raise ValueError(f"extractor {kind!r} already registered")
    _EXTRACTORS[kind] = fn


def get_extractor(kind: str) -> Extractor:
    if kind not in _EXTRACTORS:
        raise KeyError(f"unknown extractor {kind!r}; have {sorted(_EXTRACTORS)}")
    return _EXTRACTORS[kind]


# Concrete extractors. Each one is the only thing that knows how its source
# becomes text. If a new format shows up, add one extractor here, done.

def _image_extractor(path: str) -> str:
    from abstract_ocr import paddle_image          # extra: abstract_hugpy_dev[ocr]
    return paddle_image(path)

def _website_extractor(url: str) -> str:
    from abstract_webtools import get_soup_text     # extra: abstract_hugpy_dev[web]
    return get_soup_text(url)

register_extractor("image", _image_extractor)
register_extractor("audio", transcribe_file)
register_extractor("video", transcribe_file)
register_extractor("website", _website_extractor)

def _website_body_text(url: str) -> str:
    from abstract_webtools import get_body_from_url  # extra: abstract_hugpy_dev[web]
    soup = BeautifulSoup(get_body_from_url(url), "html.parser")
    lines = [line for line in soup.text.split("\n") if line]
    return "\n".join(lines)




def _pdf_full_text(path: str) -> str:
    """Whole-PDF text. The page-level SEO report is a separate operation."""
    parts = []
    for page_num in range(get_num_pdf_pages(path)):
        parts.append(extract_single_pdf_page_text(pdf_path=path, page_index=page_num))
    return "\n\n".join(parts)

register_extractor("pdf", _pdf_full_text)

def source_to_text(source: str, kind: str | None = None) -> str:
    """
    Convert an incoming source into plain text.

    For kind='text', the source is already text and should not be extracted.
    For file/url-backed kinds, dispatch through the extractor registry.
    """
    if kind in (None, "", "text"):
        return source

    return get_extractor(kind)(source)
# ---- core operations -------------------------------------------------------

def summarize(source: str, kind: str = None, presets: AnalyzePresets = AnalyzePresets()) -> dict:
    """Run the SEO analyzer over text extracted from `source`."""
    logger.info(kind)

    text = source_to_text(source, kind)

    report = _analyze(
        text,
        scope=presets.scope,
        summary_preset=presets.summary_preset,
        keyword_preset=presets.keyword_preset,
    )

    return report.to_dict()

async def analyze(
    source: str,
    kind: SOURCEKIND = "text",
    prompt: str = "Please analyze the following content.",
    params: GenParams | None = None,
    model_key: str = DEFAULT_CHAT_MODEL,
) -> ChatResult:
    params = params or GenParams()
    text = source_to_text(source, kind)
    params = params.model_copy(update={
        "messages": [{"role": "user", "content": f"{prompt}\n\n{text}"}]
    })

    # GenParams carries some fields ChatRequest doesn't know about
    # (use_chat_template is a runner-internal toggle). Strip them here.
    payload = params.model_dump(exclude={"use_chat_template"})

    req = ChatRequest(model_key=model_key, **payload)
    runner = runner_for(model_key)
    return await runner.run(req)

# ---- PDF: the one operation that's actually different ----------------------

def summarize_pdf_by_page(path: str) -> dict:
    """PDF gets its own per-page summary because PDFSeoReport is page-structured."""
    report = PDFSeoReport()
    for page_num in range(get_num_pdf_pages(path)):
        text = extract_single_pdf_page_text(pdf_path=path, page_index=page_num)
        report.pages.append(
            _analyze(
                text,
                scope=f"page:{page_num}",
                summary_preset="brief",
                keyword_preset="long_tail",
            )
        )
    return report.to_dict()


async def analyze_pdf_by_page(
    path: str,
    prompt: str = "Please analyze this PDF page",
    params: GenParams | None = None,
    model_key: str = 'DeepCoder-14B',
) -> List[Any]:
    """Generate one analysis per page. Returns a list, indexed by page number."""
    results: List[Any] = []
    for page_num in range(get_num_pdf_pages(path)):
        text = extract_single_pdf_page_text(pdf_path=path, page_index=page_num)
        page_prompt = f"{prompt} (page {page_num})\n\n{text}"
        params = params.model_copy(update={
            "messages": [{"role": "user", "content": page_prompt}]
        })

        # GenParams carries some fields ChatRequest doesn't know about
        # (use_chat_template is a runner-internal toggle). Strip them here.
        payload = params.model_dump(exclude={"use_chat_template"})

        req = ChatRequest(model_key=model_key, **payload)
        runner = runner_for(model_key) 
        res = await runner.run(req)
        results.append(res.text)
    prompt = f"please provide a full analysis of the following sumaries:\n{results}"

    params = params.model_copy(update={
        "messages": [{"role": "user", "content": prompt}]
    })

    # GenParams carries some fields ChatRequest doesn't know about
    # (use_chat_template is a runner-internal toggle). Strip them here.
    payload = params.model_dump(exclude={"use_chat_template"})

    req = ChatRequest(model_key=model_key, **payload)
    res = await runner.run(req)
    return res.text



# ---- image analysis: doesn't go through text, separate path ----------------

def image_analysis(path: str, prompt: str = "Please describe the following text", max_new_tokens: int = 100):
    return deepcoder_image_analysis(
        image_path=path,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
    )


# ---- back-compat shims -----------------------------------------------------
# Keep the old names working so nothing downstream breaks while you migrate.
# Mark them deprecated; delete in a follow-up pass.


def summarize_text(path=None, prompt="Please summarize the text",
                model_key: str = DEFAULT_CHAT_MODEL, **kw):
    return analyze(path, "text", prompt=prompt, model_key=model_key,
                   params=GenParams(**_filter_gen_kw(kw)))

def get_pdf_text(path):
    return [
        {"page_num": i, "text": extract_single_pdf_page_text(pdf_path=path, page_index=i)}
        for i in range(get_num_pdf_pages(path))
    ]

def summarize_pdf(path):       return summarize_pdf_by_page(path)
def analyze_pdf(path=None, prompt="Please summarize the pdf component",
                model_key: str = DEFAULT_CHAT_MODEL, **kw):
    return analyze_pdf_by_page(path, prompt=prompt, model_key=model_key,
                   params=GenParams(**_filter_gen_kw(kw)))

def image_to_text(path):       return get_extractor("image")(path)
##def summarize_image(path):      return summarize(path, "image")  # legacy: was always image-backed
def summarize_image(path=None, prompt="Please summarize the following",
                  model_key: str = DEFAULT_CHAT_MODEL, **kw):
    return analyze(path, "image", prompt=prompt, model_key=model_key,
                   params=GenParams(**_filter_gen_kw(kw)))

def video_to_text(path):       return get_extractor("video")(path)
##def summarize_video(path):     return summarize(path, "video")
def summarize_video(path=None, prompt="Please summarize the video transcription",
                  model_key: str = DEFAULT_CHAT_MODEL, **kw):
    return analyze(path, "video", prompt=prompt, model_key=model_key,
                   params=GenParams(**_filter_gen_kw(kw)))

def audio_to_text(path):       return get_extractor("audio")(path)
##def summarize_audio(path):     return summarize(path, "audio")
def summarize_audio(path=None, prompt="Please summarize the audio transcription",
                  model_key: str = DEFAULT_CHAT_MODEL, **kw):
    return analyze(path, "audio", prompt=prompt, model_key=model_key,
                   params=GenParams(**_filter_gen_kw(kw)))

def website_to_text(url):
    from abstract_webtools import get_soup_text     # extra: abstract_hugpy_dev[web]
    return get_soup_text(url)
def website_body_to_text(url):      return get_extractor("website")(url)
##def summarize_website(url):         return summarize(url, "website")
def summarize_website(url=None, prompt="Please summarize the website",
                    model_key: str = DEFAULT_CHAT_MODEL, **kw):
    return analyze(url, "website", prompt=prompt, model_key=model_key,
                   params=GenParams(**_filter_gen_kw(kw)))


_GEN_KEYS = {"max_new_tokens", "temperature", "top_p", "use_chat_template", "messages", "do_sample"}

def _filter_gen_kw(kw: dict) -> dict:
    """Drop unknown kwargs so legacy callers passing extras don't blow up GenParams."""
    return {k: v for k, v in kw.items() if k in _GEN_KEYS}

