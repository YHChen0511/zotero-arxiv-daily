from __future__ import annotations

import io
import json
import re
import tarfile
from datetime import date, timedelta
from html.parser import HTMLParser
from time import sleep
from types import SimpleNamespace
from typing import Any
from urllib.parse import urljoin

import arxiv
import pymupdf
import requests
import tiktoken
from loguru import logger
from openai import OpenAI
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm

from .construct_email import render_hf_email
from .protocol import Paper
from .retriever.arxiv_retriever import (
    extract_text_from_html,
    extract_text_from_pdf,
    extract_text_from_tar,
)
from .utils import send_email


HF_DAILY_API = "https://huggingface.co/api/daily_papers"
SOURCE_DOWNLOAD_TIMEOUT = (10, 120)
SOURCE_DOWNLOAD_CHUNK_SIZE = 1024 * 1024
SOURCE_DOWNLOAD_RETRIES = 4
SOURCE_RETRY_STATUSES = {429, 500, 502, 503, 504}
ARXIV_REQUEST_HEADERS = {"User-Agent": "zotero-arxiv-daily/1.0"}
FIGURE_DOWNLOAD_TIMEOUT = (10, 60)
PDF_DOWNLOAD_TIMEOUT = (10, 120)
PDF_RENDER_SCALE = 2
PDF_MAX_FALLBACK_PAGES = 4
PDF_FALLBACK_PAGE_HEIGHT_RATIO = 0.72
PDF_FIGURE_MARGIN = 12
ARCHITECTURE_FIGURE_KEYWORDS = {
    "architecture": 8,
    "model architecture": 10,
    "framework": 8,
    "pipeline": 8,
    "overview": 7,
    "method": 6,
    "approach": 5,
    "network": 5,
    "module": 4,
    "system": 4,
    "schematic": 4,
    "workflow": 4,
    "overall": 4,
}


class HfArxivPaper:
    def __init__(self, metadata: dict[str, Any], arxiv_id: str):
        self._arxiv_id = arxiv_id
        self.title = str(metadata.get("title") or arxiv_id)
        self.summary = str(
            metadata.get("summary") or metadata.get("ai_summary") or ""
        )
        self.authors = [
            SimpleNamespace(name=name)
            for name in _extract_hf_author_names(metadata)
        ]
        self.entry_id = f"https://arxiv.org/abs/{arxiv_id}"
        self.pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

    def get_short_id(self) -> str:
        return self._arxiv_id

    def source_url(self) -> str:
        return f"https://arxiv.org/e-print/{self._arxiv_id}"


def get_hf_daily_papers(date_str: str) -> list[dict[str, Any]]:
    response = requests.get(f"{HF_DAILY_API}?date={date_str}", timeout=(10, 60))
    response.raise_for_status()
    return response.json()


def get_target_date(today: date | None = None) -> str:
    today = today or date.today()
    if today.weekday() == 0:
        return (today - timedelta(days=3)).isoformat()
    return (today - timedelta(days=1)).isoformat()


def normalize_hf_keywords(raw_value: Any) -> list[str]:
    if raw_value is None:
        return []

    candidates: list[str] = []
    if isinstance(raw_value, str):
        candidates = re.split(r"[,;|/\n，、]+", raw_value)
    elif isinstance(raw_value, dict):
        candidates = [str(value) for value in raw_value.values() if value is not None]
    elif isinstance(raw_value, list):
        for item in raw_value:
            if isinstance(item, str):
                candidates.append(item)
            elif isinstance(item, dict):
                for key in ("name", "label", "tag", "id"):
                    value = item.get(key)
                    if isinstance(value, str):
                        candidates.append(value)
                        break
            elif item is not None:
                candidates.append(str(item))
    else:
        candidates = [str(raw_value)]

    normalized = []
    for keyword in candidates:
        text = str(keyword).strip()
        if text and text.lower() not in {"none", "null", "n/a"}:
            normalized.append(text)
    return list(dict.fromkeys(normalized))


def normalize_arxiv_id(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", str(arxiv_id).strip())


def _extract_hf_author_names(metadata: dict[str, Any]) -> list[str]:
    names = []
    for author in metadata.get("authors") or []:
        if isinstance(author, dict):
            name = author.get("name")
        else:
            name = str(author)
        if name:
            names.append(str(name))
    return names


def fetch_code_url(arxiv_id: str) -> str | None:
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=0.1)
    session.mount("https://", HTTPAdapter(max_retries=retries))
    try:
        paper_list = session.get(
            f"https://paperswithcode.com/api/v1/papers/?arxiv_id={arxiv_id}",
            timeout=(10, 30),
        ).json()
    except Exception as exc:
        logger.debug(f"Error when searching code for {arxiv_id}: {exc}")
        return None

    if paper_list.get("count", 0) == 0:
        return None

    paper_id = paper_list["results"][0]["id"]
    try:
        repo_list = session.get(
            f"https://paperswithcode.com/api/v1/papers/{paper_id}/repositories/",
            timeout=(10, 30),
        ).json()
    except Exception as exc:
        logger.debug(f"Error when searching repositories for {arxiv_id}: {exc}")
        return None

    if repo_list.get("count", 0) == 0:
        return None
    return repo_list["results"][0]["url"]


def _extract_figures_from_tex(content: str) -> list[dict[str, str]]:
    figures = []
    fig_blocks = re.findall(
        r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}", content, flags=re.DOTALL
    )
    for block in fig_blocks:
        caption_match = re.search(r"\\caption\{(.*?)\}", block, flags=re.DOTALL)
        caption = caption_match.group(1).strip() if caption_match else ""
        caption = re.sub(r"[\n\r\t]+", " ", caption)

        img_match = re.search(r"\\includegraphics(?:\[.*?\])?\{(.*?)\}", block)
        if img_match:
            figures.append({"file": img_match.group(1).strip(), "caption": caption})
    return figures


class _ArxivHtmlFigureParser(HTMLParser):
    def __init__(self, base_url: str):
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.figures: list[dict[str, str]] = []
        self._figure_depth = 0
        self._figcaption_depth = 0
        self._caption_parts: list[str] = []
        self._images: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {key.lower(): value or "" for key, value in attrs}
        tag = tag.lower()
        if tag == "figure":
            if self._figure_depth == 0:
                self._caption_parts = []
                self._images = []
            self._figure_depth += 1
        elif self._figure_depth and tag == "figcaption":
            self._figcaption_depth += 1
        elif self._figure_depth and tag == "img":
            src = attr_map.get("src")
            if src:
                self._images.append(
                    {
                        "file": src,
                        "url": urljoin(self.base_url, src),
                        "caption": attr_map.get("alt", ""),
                    }
                )

    def handle_data(self, data: str) -> None:
        if self._figcaption_depth:
            self._caption_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "figcaption" and self._figcaption_depth:
            self._figcaption_depth -= 1
        elif tag == "figure" and self._figure_depth:
            self._figure_depth -= 1
            if self._figure_depth == 0:
                caption = re.sub(r"\s+", " ", " ".join(self._caption_parts)).strip()
                for image in self._images:
                    self.figures.append(
                        {
                            "file": image["file"],
                            "url": image["url"],
                            "caption": caption or image["caption"],
                            "source": "html",
                        }
                    )


def _html_url_for_paper(paper: arxiv.Result) -> str:
    return paper.entry_id.replace("/abs/", "/html/")


def extract_figures_from_html(paper: arxiv.Result) -> list[dict[str, str]]:
    html_url = _html_url_for_paper(paper)
    try:
        response = requests.get(
            html_url,
            timeout=FIGURE_DOWNLOAD_TIMEOUT,
            headers=ARXIV_REQUEST_HEADERS,
        )
        if response.status_code == 404:
            return []
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.debug(f"Failed to fetch arXiv HTML figures for {_source_paper_id(paper)}: {exc}")
        return []

    parser = _ArxivHtmlFigureParser(html_url)
    try:
        parser.feed(response.text)
    except Exception as exc:
        logger.debug(f"Failed to parse arXiv HTML figures for {_source_paper_id(paper)}: {exc}")
        return []
    return parser.figures


def _score_figure_candidate(figure: dict[str, str], index: int) -> int:
    text = f"{figure.get('caption', '')} {figure.get('file', '')}".lower()
    score = 0
    for keyword, weight in ARCHITECTURE_FIGURE_KEYWORDS.items():
        if keyword in text:
            score += weight

    figure_number = re.search(r"\bfig(?:ure)?\.?\s*(\d+)\b", text)
    if figure_number:
        number = int(figure_number.group(1))
        if number == 1:
            score += 4
        elif number == 2:
            score += 2

    if index < 3:
        score += 3 - index

    if any(word in text for word in ("appendix", "supplementary", "dataset", "benchmark")):
        score -= 3

    return score


def _select_best_figure(figures: list[dict[str, str]]) -> dict[str, str] | None:
    if not figures:
        return None
    return max(
        enumerate(figures),
        key=lambda indexed: _score_figure_candidate(indexed[1], indexed[0]),
    )[1]


def _is_embeddable_image(content: bytes) -> bool:
    return content.startswith(b"\x89PNG\r\n\x1a\n") or content.startswith(b"\xff\xd8\xff")


def _download_figure_image(url: str) -> bytes | None:
    try:
        response = requests.get(
            url,
            timeout=FIGURE_DOWNLOAD_TIMEOUT,
            headers=ARXIV_REQUEST_HEADERS,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.debug(f"Failed to download figure image {url}: {exc}")
        return None

    content = response.content
    content_type = response.headers.get("Content-Type", "").lower()
    if "application/pdf" in content_type or url.lower().split("?")[0].endswith(".pdf"):
        return _pdf_bytes_to_png(content)
    if _is_embeddable_image(content):
        return content
    logger.debug(f"Skipping unsupported figure image type from {url}: {content_type}")
    return None


def extract_image_content_from_html(
    paper: arxiv.Result,
    selected_figure: str | None = None,
    figures: list[dict[str, str]] | None = None,
) -> bytes | None:
    figures = figures if figures is not None else extract_figures_from_html(paper)
    if not figures:
        return None

    selected = None
    if selected_figure:
        selected_base = selected_figure.split("/")[-1]
        selected = next(
            (
                figure
                for figure in figures
                if figure.get("file", "").split("/")[-1] == selected_base
            ),
            None,
        )

    if selected is None:
        selected = _select_best_figure(figures)

    if selected is None or not selected.get("url"):
        return None
    return _download_figure_image(selected["url"])


def _download_pdf_bytes(paper: arxiv.Result) -> bytes | None:
    if not paper.pdf_url:
        return None
    try:
        response = requests.get(
            paper.pdf_url,
            timeout=PDF_DOWNLOAD_TIMEOUT,
            headers=ARXIV_REQUEST_HEADERS,
        )
        response.raise_for_status()
        return response.content
    except requests.RequestException as exc:
        logger.debug(f"Failed to download PDF for {_source_paper_id(paper)}: {exc}")
        return None


def _score_pdf_page_for_architecture(page: pymupdf.Page, page_index: int) -> int:
    text = page.get_text("text").lower()
    score = 0
    for keyword, weight in ARCHITECTURE_FIGURE_KEYWORDS.items():
        if keyword in text:
            score += weight
    if re.search(r"\bfig(?:ure)?\.?\s*1\b", text):
        score += 4
    if re.search(r"\bfig(?:ure)?\.?\s*2\b", text):
        score += 2
    score += max(0, 4 - page_index)
    return score


def _rect_from_bbox(page: pymupdf.Page, bbox: tuple[float, float, float, float]) -> pymupdf.Rect:
    page_rect = page.rect
    x0, y0, x1, y1 = bbox
    return pymupdf.Rect(
        max(page_rect.x0, x0),
        max(page_rect.y0, y0),
        min(page_rect.x1, x1),
        min(page_rect.y1, y1),
    )


def _expand_rect(page: pymupdf.Page, rect: pymupdf.Rect, margin: float) -> pymupdf.Rect:
    return _rect_from_bbox(
        page,
        (
            rect.x0 - margin,
            rect.y0 - margin,
            rect.x1 + margin,
            rect.y1 + margin,
        ),
    )


def _text_from_block(block: dict[str, Any]) -> str:
    text_parts = []
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            text_parts.append(str(span.get("text", "")))
    return " ".join(text_parts)


def _score_caption(text: str) -> int:
    normalized = text.lower()
    score = 0
    if re.search(r"\bfig(?:ure)?\.?\s*\d+\b", normalized):
        score += 8
    for keyword, weight in ARCHITECTURE_FIGURE_KEYWORDS.items():
        if keyword in normalized:
            score += weight
    return score


def _rects_overlap_horizontally(left: pymupdf.Rect, right: pymupdf.Rect) -> bool:
    overlap = min(left.x1, right.x1) - max(left.x0, right.x0)
    return overlap > min(left.width, right.width) * 0.25


def _nearest_caption_rect(
    image_rect: pymupdf.Rect,
    text_blocks: list[dict[str, Any]],
    page: pymupdf.Page,
) -> tuple[pymupdf.Rect | None, int]:
    best_rect = None
    best_score = 0
    max_gap = page.rect.height * 0.18

    for block in text_blocks:
        text = _text_from_block(block)
        score = _score_caption(text)
        if score == 0:
            continue

        text_rect = _rect_from_bbox(page, block["bbox"])
        if not _rects_overlap_horizontally(image_rect, text_rect):
            continue

        vertical_gap = min(
            abs(text_rect.y0 - image_rect.y1),
            abs(image_rect.y0 - text_rect.y1),
        )
        if vertical_gap > max_gap:
            continue

        score -= int(vertical_gap / 20)
        if score > best_score:
            best_rect = text_rect
            best_score = score

    return best_rect, best_score


def _select_pdf_figure_clip(
    document: pymupdf.Document,
    max_pages: int,
) -> tuple[pymupdf.Page, pymupdf.Rect] | None:
    best: tuple[int, pymupdf.Page, pymupdf.Rect] | None = None

    for page_index in range(max_pages):
        page = document.load_page(page_index)
        page_area = page.rect.width * page.rect.height
        if page_area <= 0:
            continue

        blocks = page.get_text("dict").get("blocks", [])
        text_blocks = [block for block in blocks if block.get("type") == 0 and block.get("bbox")]
        image_blocks = [
            block
            for block in blocks
            if block.get("type") == 1
            and block.get("bbox")
            and pymupdf.Rect(block["bbox"]).width * pymupdf.Rect(block["bbox"]).height
            >= page_area * 0.02
        ]
        page_score = _score_pdf_page_for_architecture(page, page_index)

        for block in image_blocks:
            image_rect = _rect_from_bbox(page, block["bbox"])
            area_ratio = (image_rect.width * image_rect.height) / page_area
            score = page_score + int(area_ratio * 80)
            if image_rect.y0 < page.rect.y0 + page.rect.height * 0.65:
                score += 4

            caption_rect, caption_score = _nearest_caption_rect(image_rect, text_blocks, page)
            clip = image_rect
            score += caption_score
            if caption_rect is not None:
                clip = clip | caption_rect

            clip = _expand_rect(page, clip, PDF_FIGURE_MARGIN)
            if best is None or score > best[0]:
                best = (score, page, clip)

    if best is None:
        return None
    return best[1], best[2]


def extract_image_content_from_pdf(paper: arxiv.Result) -> bytes | None:
    pdf_bytes = _download_pdf_bytes(paper)
    if pdf_bytes is None:
        return None

    try:
        document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        if document.page_count == 0:
            return None
        max_pages = min(document.page_count, PDF_MAX_FALLBACK_PAGES)
        selected = _select_pdf_figure_clip(document, max_pages)
        if selected is None:
            best_index = max(
                range(max_pages),
                key=lambda index: _score_pdf_page_for_architecture(
                    document.load_page(index),
                    index,
                ),
            )
            page = document.load_page(best_index)
            rect = page.rect
            selected = (
                page,
                pymupdf.Rect(
                    rect.x0,
                    rect.y0,
                    rect.x1,
                    rect.y0 + rect.height * PDF_FALLBACK_PAGE_HEIGHT_RATIO,
                ),
            )

        page, clip = selected
        pixmap = page.get_pixmap(
            matrix=pymupdf.Matrix(PDF_RENDER_SCALE, PDF_RENDER_SCALE),
            clip=clip,
        )
        return pixmap.tobytes("png")
    except Exception as exc:
        logger.debug(f"Failed to extract PDF figure for {_source_paper_id(paper)}: {exc}")
        return None


def _retry_after_seconds(response: requests.Response) -> float | None:
    retry_after = getattr(response, "headers", {}).get("Retry-After")
    if retry_after is None:
        return None
    try:
        return max(float(retry_after), 0)
    except ValueError:
        return None


def _source_paper_id(paper: arxiv.Result) -> str:
    try:
        return paper.get_short_id()
    except Exception:
        return getattr(paper, "entry_id", "unknown")


def _extract_source_archive_content(paper: arxiv.Result) -> bytes | None:
    paper_id = _source_paper_id(paper)
    try:
        source_url = paper.source_url()
    except Exception as exc:
        logger.warning(f"Failed to get source URL for {paper_id}: {exc}")
        return None

    if source_url is None:
        logger.warning(f"No source URL available for {paper_id}.")
        return None

    for attempt in range(SOURCE_DOWNLOAD_RETRIES):
        try:
            with requests.get(
                source_url,
                stream=True,
                timeout=SOURCE_DOWNLOAD_TIMEOUT,
                headers=ARXIV_REQUEST_HEADERS,
            ) as response:
                if response.status_code == 404:
                    logger.warning(f"Source for {paper_id} not found.")
                    return None

                if (
                    response.status_code in SOURCE_RETRY_STATUSES
                    and attempt < SOURCE_DOWNLOAD_RETRIES - 1
                ):
                    wait = _retry_after_seconds(response) or 5 * (attempt + 1)
                    logger.warning(
                        f"arXiv source download returned HTTP {response.status_code} "
                        f"for {paper_id}; retrying in {wait:.0f}s"
                    )
                    sleep(wait)
                    continue

                response.raise_for_status()
                buffer = io.BytesIO()
                for chunk in response.iter_content(chunk_size=SOURCE_DOWNLOAD_CHUNK_SIZE):
                    if chunk:
                        buffer.write(chunk)

                content = buffer.getvalue()
                expected_size = response.headers.get("Content-Length")
                if expected_size is not None and len(content) != int(expected_size):
                    raise requests.exceptions.ChunkedEncodingError(
                        f"retrieval incomplete: got only {len(content)} "
                        f"out of {expected_size} bytes"
                    )

                return content
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code == 404:
                logger.warning(f"Source for {paper_id} not found.")
                return None

            if status_code in SOURCE_RETRY_STATUSES and attempt < SOURCE_DOWNLOAD_RETRIES - 1:
                response = exc.response
                wait = (_retry_after_seconds(response) if response is not None else None) or 5 * (
                    attempt + 1
                )
                logger.warning(
                    f"arXiv source download returned HTTP {status_code} for {paper_id}; "
                    f"retrying in {wait:.0f}s"
                )
                sleep(wait)
                continue

            logger.warning(f"Error when downloading source for {paper_id}: {exc}")
            return None
        except (requests.RequestException, ValueError) as exc:
            if attempt < SOURCE_DOWNLOAD_RETRIES - 1:
                wait = 5 * (attempt + 1)
                logger.warning(
                    f"Error when downloading source for {paper_id}: {exc}; "
                    f"retrying in {wait}s"
                )
                sleep(wait)
                continue
            logger.warning(f"Error when downloading source for {paper_id}: {exc}")
            return None

    return None


def _pdf_bytes_to_png(pdf_bytes: bytes) -> bytes | None:
    try:
        document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        page = document.load_page(0)
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
        return pixmap.tobytes("png")
    except Exception as exc:
        logger.warning(f"Failed to convert PDF figure to PNG: {exc}")
        return None


def extract_image_content(
    paper: arxiv.Result,
    selected_figure: str | None = None,
    html_figures: list[dict[str, str]] | None = None,
) -> bytes | None:
    image_content = extract_image_content_from_html(
        paper,
        selected_figure=selected_figure,
        figures=html_figures,
    )
    if image_content is not None:
        return image_content

    image_content = extract_image_content_from_pdf(paper)
    if image_content is not None:
        return image_content

    source_content = _extract_source_archive_content(paper)
    if source_content is None:
        return None

    try:
        with tarfile.open(fileobj=io.BytesIO(source_content), mode="r:*") as tar:
            image_files = [
                member
                for member in tar.getmembers()
                if member.isfile()
                and member.name.lower().endswith((".png", ".jpg", ".jpeg", ".pdf"))
            ]
            if not image_files:
                return None

            target_member = None
            if selected_figure:
                selected_base = re.sub(
                    r"\.(pdf|eps|png|jpg|jpeg)$",
                    "",
                    selected_figure.split("/")[-1],
                    flags=re.IGNORECASE,
                )
                for image in image_files:
                    image_base = re.sub(
                        r"\.(pdf|png|jpg|jpeg)$",
                        "",
                        image.name.split("/")[-1],
                        flags=re.IGNORECASE,
                    )
                    if image_base == selected_base:
                        target_member = image
                        break

            if target_member is None:
                image_files.sort(key=lambda image: image.size, reverse=True)
                target_member = image_files[0]

            file_obj = tar.extractfile(target_member)
            if file_obj is None:
                return None
            content = file_obj.read()
            if target_member.name.lower().endswith(".pdf"):
                return _pdf_bytes_to_png(content)
            return content
    except tarfile.ReadError:
        logger.debug(f"Source for {paper.get_short_id()} is not a tar file.")
    except Exception as exc:
        logger.warning(f"Error extracting image for {paper.get_short_id()}: {exc}")
    return None


def generate_bilingual_summary(
    paper: Paper,
    openai_client: OpenAI,
    llm_params: dict[str, Any],
    hf_keywords: list[str],
    figures: list[dict[str, str]],
) -> dict[str, Any]:
    context_text = f"Title: {paper.title}\nAbstract: {paper.abstract}\n"
    if paper.full_text:
        context_text += f"Paper content preview: {paper.full_text[:4000]}\n"

    figures_text = "No figures extracted from the paper."
    if figures:
        figures_text = (
            "Figures found in paper:\n"
            + json.dumps(figures, ensure_ascii=False, indent=2)
            + "\nPrefer an architecture/framework/pipeline/overview figure. "
            + "Select its file value in selected_figure."
        )

    prompt = f"""
Analyze the following academic paper as a research assistant. The goal is to help
the reader quickly decide whether this HF Daily paper deserves deeper follow-up.

Return a valid JSON object with these keys:
- "problem": {{"cn": "...", "en": "..."}}
- "solution": {{"cn": "...", "en": "..."}}
- "result": {{"cn": "...", "en": "..."}}
- "keywords": {{"cn": ["..."], "en": ["..."]}}
- "selected_figure": "filename" or null

Field meaning:
- problem: combine a 30-second summary with problem formulation. Mention the
  paper's input, output, and optimization or evaluation target when inferable.
- solution: combine method delta, evidence quality, and hidden assumptions. Say
  what changed versus the closest prior work, whether the experiments support
  the claim, and what assumptions the method depends on.
- result: combine reading decision and research follow-up. Include whether to
  add it to a deep-reading list (yes/no/maybe), reproduction difficulty, three
  follow-up ideas, and risk scores for novelty, reliability, and practicality
  on a 1-5 scale.

Keep each field concise but information-dense. Distinguish paper claims from
your inference. If evidence is missing from the provided text, say so. Do not
use markdown code fences.

{figures_text}

Paper content:
{context_text}
"""

    enc = tiktoken.encoding_for_model("gpt-4o")
    prompt_tokens = enc.encode(prompt)[:6000]
    prompt = enc.decode(prompt_tokens)

    try:
        response = openai_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful research assistant. Output strictly valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            **llm_params.get("generation_kwargs", {}),
        )
        raw = response.choices[0].message.content
        raw = raw.replace("```json", "").replace("```", "").strip()
        summary = json.loads(raw)
    except Exception as exc:
        logger.warning(f"Failed to generate bilingual summary for {paper.url}: {exc}")
        summary = {
            "problem": {"cn": "生成失败", "en": "Generation failed"},
            "solution": {"cn": "生成失败", "en": "Generation failed"},
            "result": {"cn": "生成失败", "en": "Generation failed"},
            "keywords": {"cn": [], "en": []},
            "selected_figure": None,
        }

    raw_keywords = summary.get("keywords")
    if isinstance(raw_keywords, dict):
        cn_keywords = normalize_hf_keywords(raw_keywords.get("cn"))
        en_keywords = normalize_hf_keywords(raw_keywords.get("en"))
    else:
        normalized = normalize_hf_keywords(raw_keywords)
        cn_keywords = normalized
        en_keywords = normalized

    if hf_keywords:
        if not cn_keywords:
            cn_keywords = hf_keywords
        if not en_keywords:
            en_keywords = hf_keywords

    summary["keywords"] = {
        "cn": list(dict.fromkeys(cn_keywords)),
        "en": list(dict.fromkeys(en_keywords)),
    }
    return summary


def convert_arxiv_result_to_paper(raw_paper: arxiv.Result) -> Paper:
    full_text = extract_text_from_html(raw_paper)
    if full_text is None:
        full_text = extract_text_from_pdf(raw_paper)
    if full_text is None:
        full_text = extract_text_from_tar(raw_paper)
    return Paper(
        source="huggingface",
        title=raw_paper.title,
        authors=[author.name for author in raw_paper.authors],
        abstract=raw_paper.summary,
        url=raw_paper.entry_id,
        pdf_url=raw_paper.pdf_url,
        full_text=full_text,
    )


def convert_hf_metadata_to_paper(
    metadata: dict[str, Any],
    arxiv_id: str,
    raw_paper: arxiv.Result | HfArxivPaper | None = None,
) -> tuple[Paper, arxiv.Result | HfArxivPaper]:
    paper_source = raw_paper or HfArxivPaper(metadata, arxiv_id)
    paper = convert_arxiv_result_to_paper(paper_source)

    if not paper.title or paper.title == arxiv_id:
        paper.title = str(metadata.get("title") or arxiv_id)
    if not paper.abstract:
        paper.abstract = str(metadata.get("summary") or metadata.get("ai_summary") or "")
    if not paper.authors:
        paper.authors = _extract_hf_author_names(metadata)

    return paper, paper_source


def run_hf_daily_flow(config: Any, openai_client: OpenAI | None = None) -> None:
    date_str = config.executor.get("hf_date") or get_target_date()
    max_papers = config.executor.get("hf_max_paper_num", 10)
    logger.info(f"Fetching HuggingFace Daily Papers for {date_str}")

    try:
        hf_data = get_hf_daily_papers(date_str)
    except Exception as exc:
        logger.error(f"Failed to fetch HuggingFace daily papers: {exc}")
        return

    if not hf_data:
        logger.info("No HuggingFace papers found for this date.")
        return

    hf_data = hf_data[:max_papers]
    hf_items = []
    for item in hf_data:
        metadata = item.get("paper", {})
        arxiv_id = metadata.get("id")
        if arxiv_id:
            hf_items.append((metadata, arxiv_id))

    if not hf_items:
        logger.info("No valid arXiv IDs found in HuggingFace daily papers.")
        return

    logger.info("Using HuggingFace paper metadata directly.")

    if openai_client is None:
        openai_client = OpenAI(
            api_key=config.llm.api.key,
            base_url=config.llm.api.base_url,
        )

    processed_papers = []
    for metadata, arxiv_id in tqdm(hf_items, desc="Processing HF papers"):
        normalized_id = normalize_arxiv_id(arxiv_id)

        try:
            paper, paper_source = convert_hf_metadata_to_paper(
                metadata,
                normalized_id,
            )
            html_figures = extract_figures_from_html(paper_source)
            tex_figures = _extract_figures_from_tex(paper.full_text or "")
            figures = html_figures + tex_figures
            hf_keywords = normalize_hf_keywords(
                metadata.get("tags") or metadata.get("keywords")
            )
            summary = generate_bilingual_summary(
                paper=paper,
                openai_client=openai_client,
                llm_params=config.llm,
                hf_keywords=hf_keywords,
                figures=figures,
            )
            image_content = extract_image_content(
                paper_source,
                summary.get("selected_figure"),
                html_figures=html_figures,
            )

            processed_papers.append(
                {
                    "title": paper.title,
                    "authors": paper.authors,
                    "score": metadata.get("upvotes", 0),
                    "arxiv_id": normalized_id,
                    "pdf_url": paper.pdf_url,
                    "code_url": fetch_code_url(normalized_id),
                    "bilingual_summary": summary,
                    "image_content": image_content,
                }
            )
        except Exception as exc:
            logger.error(f"Error processing HuggingFace paper {arxiv_id}: {exc}")

    if not processed_papers:
        logger.info("No HuggingFace papers processed successfully.")
        return

    html, attachments = render_hf_email(processed_papers, date_str)
    if config.executor.get("debug"):
        debug_path = config.executor.get("hf_debug_html_path", "test.html")
        with open(debug_path, "w", encoding="utf-8") as file:
            file.write(html)

    logger.info("Sending HuggingFace Daily email")
    send_email(
        config,
        html,
        attachments=attachments,
        subject=f"HuggingFace Daily Papers {date_str}",
    )
    logger.info("HuggingFace Daily email sent successfully")
