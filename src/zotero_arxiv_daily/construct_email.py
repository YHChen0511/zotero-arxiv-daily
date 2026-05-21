from __future__ import annotations

import html
import math
import re
import uuid

from .protocol import Paper


framework = """
<!DOCTYPE HTML>
<html>
<head>
  <style>
    .star-wrapper {
      font-size: 1.3em;
      line-height: 1;
      display: inline-flex;
      align-items: center;
    }
    .half-star {
      display: inline-block;
      width: 0.5em;
      overflow: hidden;
      white-space: nowrap;
      vertical-align: middle;
    }
    .full-star {
      vertical-align: middle;
    }
  </style>
</head>
<body>

<div>
    __CONTENT__
</div>

<br><br>
<div>
To unsubscribe, remove your email in your Github Action setting.
</div>

</body>
</html>
"""


def get_empty_html() -> str:
    return """
  <table border="0" cellpadding="0" cellspacing="0" width="100%" style="font-family: Arial, sans-serif; border: 1px solid #ddd; border-radius: 8px; padding: 16px; background-color: #f9f9f9;">
  <tr>
    <td style="font-size: 20px; font-weight: bold; color: #333;">
        No Papers Today. Take a Rest!
    </td>
  </tr>
  </table>
  """


def get_block_html(
    title: str,
    authors: str,
    rate: str,
    tldr: str,
    pdf_url: str,
    affiliations: str | None = None,
    code_url: str | None = None,
    arxiv_id: str | None = None,
) -> str:
    code = (
        f'<a href="{code_url}" style="display: inline-block; text-decoration: none; font-size: 14px; font-weight: bold; color: #fff; background-color: #5bc0de; padding: 8px 16px; border-radius: 4px; margin-left: 8px;">Code</a>'
        if code_url
        else ""
    )
    arxiv = (
        f"""
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>arXiv ID:</strong> <a href="https://arxiv.org/abs/{arxiv_id}" target="_blank">{arxiv_id}</a>
        </td>
    </tr>
"""
        if arxiv_id
        else ""
    )
    block_template = """
    <table border="0" cellpadding="0" cellspacing="0" width="100%" style="font-family: Arial, sans-serif; border: 1px solid #ddd; border-radius: 8px; padding: 16px; background-color: #f9f9f9;">
    <tr>
        <td style="font-size: 20px; font-weight: bold; color: #333;">
            {title}
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #666; padding: 8px 0;">
            {authors}
            <br>
            <i>{affiliations}</i>
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>Relevance:</strong> {rate}
        </td>
    </tr>
    {arxiv}
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>TLDR:</strong> {tldr}
        </td>
    </tr>

    <tr>
        <td style="padding: 8px 0;">
            <a href="{pdf_url}" style="display: inline-block; text-decoration: none; font-size: 14px; font-weight: bold; color: #fff; background-color: #d9534f; padding: 8px 16px; border-radius: 4px;">PDF</a>
            {code}
        </td>
    </tr>
</table>
"""
    return block_template.format(
        title=title,
        authors=authors,
        rate=rate,
        tldr=tldr,
        pdf_url=pdf_url,
        affiliations=affiliations,
        code=code,
        arxiv=arxiv,
    )


def get_hf_block_html(
    title: str,
    authors: str,
    score: int,
    arxiv_id: str,
    problem: str,
    solution: str,
    result: str,
    keywords: str,
    pdf_url: str,
    code_url: str | None = None,
    image_cid: str | None = None,
) -> str:
    code_btn = (
        f'<a href="{code_url}" class="btn-link" style="text-decoration: none; color: #3498db; font-size: 13px; font-weight: 600; margin-left: 10px;">Code</a>'
        if code_url
        else ""
    )

    image_html = ""
    if image_cid:
        image_html = f"""
        <div class="paper-image" style="margin-bottom: 20px; border-radius: 8px; overflow: hidden; border: 1px solid #eee;">
            <img src="cid:{image_cid}" alt="Paper Figure" style="width: 100%; height: auto; display: block;">
        </div>
        """

    return f"""
    <div class="paper-card" style="border: 1px solid #eee; border-radius: 12px; margin-bottom: 24px; background-color: #fff; box-shadow: 0 2px 8px rgba(0,0,0,0.03); overflow: hidden; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
        <div class="paper-header" style="background-color: #fafafa; padding: 16px 20px; border-bottom: 1px solid #eee;">
            <div class="paper-title" style="margin: 0; font-size: 18px; font-weight: 700; color: #2c3e50; line-height: 1.4;">
                {title}
            </div>
            <div class="paper-keywords" style="margin-top: 8px;">
                {keywords}
            </div>
            <div class="paper-meta" style="font-size: 13px; color: #7f8c8d; margin-top: 6px; display: flex; align-items: center; gap: 10px;">
                <span class="paper-badge" style="background-color: #e0f7fa; color: #006064; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; text-transform: uppercase;">Arxiv: {arxiv_id}</span>
                <span>{score} Upvotes</span>
            </div>
            <div style="font-size: 12px; color: #999; margin-top: 4px;">{authors}</div>
        </div>
        <div class="paper-body" style="padding: 20px;">
            {image_html}

            <div class="summary-section" style="margin-bottom: 16px; border-bottom: 1px solid #f9f9f9; padding-bottom: 12px;">
                <span class="summary-label" style="display: block; font-size: 11px; font-weight: 700; color: #95a5a6; text-transform: uppercase; margin-bottom: 6px;">Quick Read / Problem</span>
                {problem}
            </div>
            <div class="summary-section" style="margin-bottom: 16px; border-bottom: 1px solid #f9f9f9; padding-bottom: 12px;">
                <span class="summary-label" style="display: block; font-size: 11px; font-weight: 700; color: #95a5a6; text-transform: uppercase; margin-bottom: 6px;">Method / Evidence</span>
                {solution}
            </div>
            <div class="summary-section" style="margin-bottom: 16px; border-bottom: 1px solid #f9f9f9; padding-bottom: 12px;">
                <span class="summary-label" style="display: block; font-size: 11px; font-weight: 700; color: #95a5a6; text-transform: uppercase; margin-bottom: 6px;">Decision / Follow-up</span>
                {result}
            </div>
        </div>
        <div class="paper-footer" style="padding: 12px 20px; background-color: #fdfdfd; border-top: 1px solid #f0f0f0; display: flex; justify-content: space-between; align-items: center;">
            <div class="actions" style="display: flex; gap: 10px;">
                <a href="{pdf_url}" class="btn-link" style="text-decoration: none; color: #3498db; font-size: 13px; font-weight: 600;">Original Paper</a>
                {code_btn}
            </div>
        </div>
    </div>
    """


def get_stars(score: float) -> str:
    full_star = '<span class="full-star">*</span>'
    half_star = '<span class="half-star">*</span>'
    low = 6
    high = 8
    if score <= low:
        return ""
    if score >= high:
        return full_star * 5

    interval = (high - low) / 10
    star_num = math.ceil((score - low) / interval)
    full_star_num = int(star_num / 2)
    half_star_num = star_num - full_star_num * 2
    return (
        '<div class="star-wrapper">'
        + full_star * full_star_num
        + half_star * half_star_num
        + "</div>"
    )


def render_email(papers: list[Paper]) -> str:
    parts = []
    if len(papers) == 0:
        return framework.replace("__CONTENT__", get_empty_html())

    for p in papers:
        rate = round(p.score, 1) if p.score is not None else "Unknown"
        author_list = [a for a in p.authors]
        if len(author_list) <= 5:
            authors = ", ".join(author_list)
        else:
            authors = ", ".join(author_list[:3] + ["..."] + author_list[-2:])

        if p.affiliations is not None:
            affiliations = ", ".join(p.affiliations[:5])
            if len(p.affiliations) > 5:
                affiliations += ", ..."
        else:
            affiliations = "Unknown Affiliation"

        parts.append(
            get_block_html(
                title=p.title,
                authors=authors,
                rate=str(rate),
                tldr=p.tldr or p.abstract,
                pdf_url=p.pdf_url or p.url,
                affiliations=affiliations,
            )
        )

    content = "<br>" + "</br><br>".join(parts) + "</br>"
    return framework.replace("__CONTENT__", content)


def render_hf_email(papers: list[dict], date_str: str) -> tuple[str, dict[str, bytes]]:
    wrapper = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <style>
            .text-cn {{ font-size: 15px; line-height: 1.6; color: #2c3e50; margin-bottom: 6px; font-weight: 500; }}
            .text-en {{ font-size: 13px; line-height: 1.5; color: #7f8c8d; font-style: italic; }}
        </style>
    </head>
    <body style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f4f9; color: #333; margin: 0; padding: 20px;">
    <div class="container" style="max-width: 800px; margin: 0 auto; background-color: #ffffff; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.05);">
        <div class="header" style="background-color: #000; color: #fff; padding: 24px; text-align: center;">
            <h1 style="margin: 0; font-size: 24px; font-weight: 600;">HuggingFace Daily Papers</h1>
            <div class="date" style="font-size: 14px; opacity: 0.8; margin-top: 8px;">{date}</div>
        </div>
        <div style="text-align:center; padding: 20px 0; color: #555; font-style: italic;">
             Daily Selection of Top Arxiv Papers
        </div>
        <div class="content" style="padding: 24px;">
            {content}
        </div>
        <div style="text-align:center; color:#999;font-size:12px; padding:20px;">
            Generated by Zotero-Arxiv-Daily
        </div>
    </div>
    </body>
    </html>
    """

    if len(papers) == 0:
        return wrapper.format(date=date_str, content=get_empty_html()), {}

    parts = []
    attachments: dict[str, bytes] = {}
    for p in papers:
        cid = None
        if p.get("image_content"):
            cid = str(uuid.uuid4())
            attachments[cid] = p["image_content"]

        def format_bi(section_name: str) -> str:
            data = p.get("bilingual_summary", {}).get(section_name, {})
            if not isinstance(data, dict):
                data = {"cn": str(data), "en": str(data)}
            cn = data.get("cn", "N/A")
            en = data.get("en", "N/A")
            return f'<div class="text-cn">{cn}</div><div class="text-en">{en}</div>'

        def normalize_keywords(raw_value: object) -> list[str]:
            if isinstance(raw_value, list):
                candidates = raw_value
            elif isinstance(raw_value, str):
                candidates = re.split(r"[,;|/\n，、]+", raw_value)
            else:
                return []
            return [
                str(keyword).strip()
                for keyword in candidates
                if str(keyword).strip()
                and str(keyword).strip().lower() not in {"none", "null", "n/a"}
            ]

        def render_keyword_badges(keyword_list: list[str]) -> str:
            if not keyword_list:
                return (
                    '<span style="display: inline-block; padding: 3px 10px; border-radius: 999px; '
                    'background-color: #fff4db; color: #8a5a00; border: 1px dashed #f2d299; '
                    'font-size: 12px; font-weight: 600; opacity: 0.7;">N/A</span>'
                )
            return "".join(
                (
                    '<span style="display: inline-block; margin: 0 8px 8px 0; padding: 4px 10px; '
                    'background-color: #fff4db; color: #8a5a00; border: 1px solid #f2d299; '
                    'border-radius: 999px; font-size: 12px; font-weight: 700;">'
                    f"{html.escape(keyword)}</span>"
                )
                for keyword in keyword_list
            )

        keyword_data = p.get("bilingual_summary", {}).get("keywords", {})
        if isinstance(keyword_data, dict):
            raw_keywords = normalize_keywords(keyword_data.get("cn")) + normalize_keywords(
                keyword_data.get("en")
            )
        else:
            raw_keywords = normalize_keywords(keyword_data)

        keywords = []
        seen = set()
        for keyword in raw_keywords:
            key = keyword.lower()
            if key not in seen:
                keywords.append(keyword)
                seen.add(key)

        parts.append(
            get_hf_block_html(
                title=p.get("title", "Unknown Title"),
                authors=", ".join(p.get("authors", [])),
                score=p.get("score", 0),
                arxiv_id=p.get("arxiv_id", ""),
                problem=format_bi("problem"),
                solution=format_bi("solution"),
                result=format_bi("result"),
                keywords=f'<div style="line-height: 1.4;">{render_keyword_badges(keywords)}</div>',
                pdf_url=p.get("pdf_url", "#"),
                code_url=p.get("code_url"),
                image_cid=cid,
            )
        )

    return wrapper.format(date=date_str, content="".join(parts)), attachments
