from paper import ArxivPaper
import math
from tqdm import tqdm
from email.header import Header
from email.mime.text import MIMEText
from email.utils import parseaddr, formataddr
import smtplib
import datetime
import time
from loguru import logger
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import uuid

framework = """
<!DOCTYPE HTML>
<html>
<head>
  <style>
    .star-wrapper {
      font-size: 1.3em; /* 调整星星大小 */
      line-height: 1; /* 确保垂直对齐 */
      display: inline-flex;
      align-items: center; /* 保持对齐 */
    }
    .half-star {
      display: inline-block;
      width: 0.5em; /* 半颗星的宽度 */
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


def get_empty_html():
    block_template = """
  <table border="0" cellpadding="0" cellspacing="0" width="100%" style="font-family: Arial, sans-serif; border: 1px solid #ddd; border-radius: 8px; padding: 16px; background-color: #f9f9f9;">
  <tr>
    <td style="font-size: 20px; font-weight: bold; color: #333;">
        No Papers Today. Take a Rest!
    </td>
  </tr>
  </table>
  """
    return block_template


def get_block_html(
    title: str,
    authors: str,
    rate: str,
    arxiv_id: str,
    abstract: str,
    pdf_url: str,
    code_url: str = None,
    affiliations: str = None,
):
    code = (
        f'<a href="{code_url}" style="display: inline-block; text-decoration: none; font-size: 14px; font-weight: bold; color: #fff; background-color: #5bc0de; padding: 8px 16px; border-radius: 4px; margin-left: 8px;">Code</a>'
        if code_url
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
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>arXiv ID:</strong> <a href="https://arxiv.org/abs/{arxiv_id}" target="_blank">{arxiv_id}</a>
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>TLDR:</strong> {abstract}
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
        arxiv_id=arxiv_id,
        abstract=abstract,
        pdf_url=pdf_url,
        code=code,
        affiliations=affiliations,
    )


def get_hf_block_html(
    title: str,
    authors: str,
    score: int,
    arxiv_id: str,
    problem: str,
    solution: str,
    result: str,
    pdf_url: str,
    code_url: str = None,
    image_cid: str = None,
):
    """
    Generate the HTML block for a HuggingFace paper with bilingual content and optional image.
    problem, solution, result are expected to be HTML strings (with embedded CN/EN styling).
    """
    code_btn = (
        f'<a href="{code_url}" class="btn-link" style="text-decoration: none; color: #3498db; font-size: 13px; font-weight: 600; margin-left: 10px;">Code</a>'
        if code_url
        else ""
    )

    image_html = ""
    if image_cid:
        image_html = f"""
        <!-- Paper Image -->
        <div class="paper-image" style="margin-bottom: 20px; border-radius: 8px; overflow: hidden; border: 1px solid #eee;">
            <img src="cid:{image_cid}" alt="Paper Figure" style="width: 100%; height: auto; display: block;">
        </div>
        """

    block_template = f"""
    <!-- Paper Card -->
    <div class="paper-card" style="border: 1px solid #eee; border-radius: 12px; margin-bottom: 24px; background-color: #fff; box-shadow: 0 2px 8px rgba(0,0,0,0.03); overflow: hidden; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
        <div class="paper-header" style="background-color: #fafafa; padding: 16px 20px; border-bottom: 1px solid #eee;">
            <div class="paper-title" style="margin: 0; font-size: 18px; font-weight: 700; color: #2c3e50; line-height: 1.4;">
                {title}
            </div>
            <div class="paper-meta" style="font-size: 13px; color: #7f8c8d; margin-top: 6px; display: flex; align-items: center; gap: 10px;">
                <span class="paper-badge" style="background-color: #e0f7fa; color: #006064; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; text-transform: uppercase;">Arxiv: {arxiv_id}</span>
                <span>🔥 {score} Upvotes</span>
            </div>
            <div style="font-size: 12px; color: #999; margin-top: 4px;">{authors}</div>
        </div>
        <div class="paper-body" style="padding: 20px;">
            {image_html}

            <div class="summary-section" style="margin-bottom: 16px; border-bottom: 1px solid #f9f9f9; padding-bottom: 12px;">
                <span class="summary-label" style="display: block; font-size: 11px; font-weight: 700; color: #95a5a6; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;">🧐 The Problem / 问题背景</span>
                {problem}
            </div>
            
            <div class="summary-section" style="margin-bottom: 16px; border-bottom: 1px solid #f9f9f9; padding-bottom: 12px;">
                <span class="summary-label" style="display: block; font-size: 11px; font-weight: 700; color: #95a5a6; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;">💡 The Solution / 核心方案</span>
                {solution}
            </div>
            
            <div class="summary-section" style="margin-bottom: 16px; border-bottom: 1px solid #f9f9f9; padding-bottom: 12px;">
                <span class="summary-label" style="display: block; font-size: 11px; font-weight: 700; color: #95a5a6; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;">🚀 Key Result / 主要结果</span>
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
    return block_template


def get_stars(score: float):
    full_star = '<span class="full-star">⭐</span>'
    half_star = '<span class="half-star">⭐</span>'
    low = 6
    high = 8
    if score <= low:
        return ""
    elif score >= high:
        return full_star * 5
    else:
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


def render_email(papers: list[ArxivPaper]):
    parts = []
    if len(papers) == 0:
        return framework.replace("__CONTENT__", get_empty_html())

    for p in tqdm(papers, desc="Rendering Email"):
        rate = get_stars(p.score)
        author_list = [a.name for a in p.authors]
        num_authors = len(author_list)

        if num_authors <= 5:
            authors = ", ".join(author_list)
        else:
            authors = ", ".join(author_list[:3] + ["..."] + author_list[-2:])
        if p.affiliations is not None:
            affiliations = p.affiliations[:5]
            affiliations = ", ".join(affiliations)
            if len(p.affiliations) > 5:
                affiliations += ", ..."
        else:
            affiliations = "Unknown Affiliation"
        parts.append(
            get_block_html(
                p.title,
                authors,
                rate,
                p.arxiv_id,
                p.tldr,
                p.pdf_url,
                p.code_url,
                affiliations,
            )
        )
        time.sleep(10)

    content = "<br>" + "</br><br>".join(parts) + "</br>"
    return framework.replace("__CONTENT__", content)


def render_hf_email(papers: list, date_str: str) -> tuple[str, dict]:
    """
    Render the HF Daily Papers email.
    Returns (html_content, image_attachments_map)
    image_attachments_map: {cid: image_bytes}
    """
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
            Generated by Zotero-Arxiv-Daily | <a href="#" style="color:#999;">Unsubscribe</a>
        </div>
    </div>
    </body>
    </html>
    """

    parts = []
    attachments = {}

    if len(papers) == 0:
        return wrapper.format(date=date_str, content=get_empty_html()), {}

    for p in papers:
        # p is expected to be a dict or object with:
        # title, authors (list), score, arxiv_id, pdf_url, code_url
        # bilingual_summary: { 'problem': {'cn':..., 'en':...}, 'solution':..., 'result':... }
        # image_content: bytes | None

        cid = None
        if p.get("image_content"):
            cid = str(uuid.uuid4())
            attachments[cid] = p["image_content"]

        # Helper to format bilingual text
        def format_bi(section_name):
            data = p.get("bilingual_summary", {}).get(section_name, {})
            cn = data.get("cn", "N/A")
            en = data.get("en", "N/A")
            return f'<div class="text-cn">{cn}</div><div class="text-en">{en}</div>'

        author_str = ", ".join(p.get("authors", []))

        block = get_hf_block_html(
            title=p.get("title", "Unknown Title"),
            authors=author_str,
            score=p.get("score", 0),
            arxiv_id=p.get("arxiv_id", ""),
            problem=format_bi("problem"),
            solution=format_bi("solution"),
            result=format_bi("result"),
            pdf_url=p.get("pdf_url", "#"),
            code_url=p.get("code_url"),
            image_cid=cid,
        )
        parts.append(block)

    final_html = wrapper.format(date=date_str, content="".join(parts))
    return final_html, attachments


def send_email(
    sender: str,
    receiver: str,
    password: str,
    smtp_server: str,
    smtp_port: int,
    html: str,
    attachments: dict = None,
):
    def _format_addr(s):
        name, addr = parseaddr(s)
        return formataddr((Header(name, "utf-8").encode(), addr))

    if attachments:
        msg = MIMEMultipart("related")
        msg_html = MIMEText(html, "html", "utf-8")
        msg.attach(msg_html)

        for cid, content in attachments.items():
            img = MIMEImage(content)
            img.add_header("Content-ID", f"<{cid}>")
            msg.attach(img)
    else:
        msg = MIMEText(html, "html", "utf-8")

    msg["From"] = _format_addr("Github Action <%s>" % sender)
    msg["To"] = _format_addr("You <%s>" % receiver)
    today = datetime.datetime.now().strftime("%Y/%m/%d")
    # Use different subject if it's the specific HF email?
    # For now, let's allow the caller to set subject or keep generic.
    # The requirement implied subject "HuggingFace Daily Papers [Date]" for HF
    # But this function is generic. Let's start with just generic Subject or rely on caller to customize?
    # Current signature doesn't take subject. I'll stick to generic or infer.

    # If the html contains "HuggingFace", let's change subject?
    if "HuggingFace Daily Papers" in html:
        msg["Subject"] = Header(f"HuggingFace Daily Papers {today}", "utf-8").encode()
    else:
        msg["Subject"] = Header(f"Daily arXiv {today}", "utf-8").encode()

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
    except Exception as e:
        logger.warning(f"Failed to use TLS. {e}")
        logger.warning(f"Try to use SSL.")
        server = smtplib.SMTP_SSL(smtp_server, smtp_port)

    server.login(sender, password)
    server.sendmail(sender, [receiver], msg.as_string())
    server.quit()
