from __future__ import annotations

import io
import json
import re
import tarfile
from contextlib import ExitStack
from datetime import date, timedelta
from tempfile import TemporaryDirectory
from typing import Any
from urllib.error import HTTPError

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


def _extract_source_archive_content(paper: arxiv.Result) -> bytes | None:
    with ExitStack() as stack:
        tmpdirname = stack.enter_context(TemporaryDirectory())
        try:
            source_path = paper.download_source(dirpath=tmpdirname)
        except HTTPError as exc:
            if exc.code == 404:
                logger.warning(f"Source for {paper.get_short_id()} not found.")
                return None
            raise
        except Exception as exc:
            logger.warning(f"Error when downloading source for {paper.get_short_id()}: {exc}")
            return None

        try:
            with open(source_path, "rb") as file:
                return file.read()
        except Exception as exc:
            logger.warning(f"Error when reading source archive for {paper.get_short_id()}: {exc}")
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
    paper: arxiv.Result, selected_figure: str | None = None
) -> bytes | None:
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

    figures_text = "No figures extracted from latex source."
    if figures:
        figures_text = (
            "Figures found in paper:\n"
            + json.dumps(figures, ensure_ascii=False, indent=2)
            + "\nSelect the most representative figure filename in selected_figure."
        )

    prompt = f"""
Analyze the following academic paper and provide a structured summary in Chinese and English.

Return a valid JSON object with these keys:
- "problem": {{"cn": "...", "en": "..."}}
- "solution": {{"cn": "...", "en": "..."}}
- "result": {{"cn": "...", "en": "..."}}
- "keywords": {{"cn": ["..."], "en": ["..."]}}
- "selected_figure": "filename" or null

Keep each description concise. Do not use markdown code fences.

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
    full_text = extract_text_from_tar(raw_paper)
    if full_text is None:
        full_text = extract_text_from_html(raw_paper)
    if full_text is None:
        full_text = extract_text_from_pdf(raw_paper)
    return Paper(
        source="huggingface",
        title=raw_paper.title,
        authors=[author.name for author in raw_paper.authors],
        abstract=raw_paper.summary,
        url=raw_paper.entry_id,
        pdf_url=raw_paper.pdf_url,
        full_text=full_text,
    )


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

    client = arxiv.Client()
    unique_arxiv_ids = list(dict.fromkeys(arxiv_id for _, arxiv_id in hf_items))
    arxiv_results: dict[str, arxiv.Result] = {}
    for i in tqdm(range(0, len(unique_arxiv_ids), 20), desc="Fetching HF arXiv metadata"):
        batch_ids = unique_arxiv_ids[i : i + 20]
        try:
            results = list(client.results(arxiv.Search(id_list=batch_ids)))
        except Exception as exc:
            logger.error(f"Failed to fetch arXiv metadata for {batch_ids}: {exc}")
            continue
        for result in results:
            short_id = result.get_short_id()
            arxiv_results[short_id] = result
            arxiv_results[normalize_arxiv_id(short_id)] = result

    if openai_client is None:
        openai_client = OpenAI(
            api_key=config.llm.api.key,
            base_url=config.llm.api.base_url,
        )

    processed_papers = []
    for metadata, arxiv_id in tqdm(hf_items, desc="Processing HF papers"):
        normalized_id = normalize_arxiv_id(arxiv_id)
        raw_paper = arxiv_results.get(arxiv_id) or arxiv_results.get(normalized_id)
        if raw_paper is None:
            logger.warning(f"arXiv ID {arxiv_id} not found in arXiv API.")
            continue

        try:
            paper = convert_arxiv_result_to_paper(raw_paper)
            figures = _extract_figures_from_tex(paper.full_text or "")
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
            image_content = extract_image_content(raw_paper, summary.get("selected_figure"))

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
