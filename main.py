import arxiv


def _get_pdf_url_patch(links) -> str:
    """
    Finds the PDF link among a result's links and returns its URL.
    Should only be called once for a given `Result`, in its constructor.
    After construction, the URL should be available in `Result.pdf_url`.
    """
    pdf_urls = [link.href for link in links if "pdf" in link.href]
    if len(pdf_urls) == 0:
        return None
    return pdf_urls[0]


arxiv.Result._get_pdf_url = _get_pdf_url_patch

import argparse
import os
import sys
from dotenv import load_dotenv
import requests

load_dotenv(override=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pyzotero import zotero
from recommender import rerank_paper
from construct_email import render_email, send_email, render_hf_email
from tqdm import trange, tqdm
from loguru import logger
from gitignore_parser import parse_gitignore
from tempfile import mkstemp
from paper import ArxivPaper
from llm import set_global_llm
import feedparser


def get_zotero_corpus(id: str, key: str) -> list[dict]:
    zot = zotero.Zotero(id, "user", key)
    collections = zot.everything(zot.collections())
    collections = {c["key"]: c for c in collections}
    corpus = zot.everything(
        zot.items(itemType="conferencePaper || journalArticle || preprint")
    )
    corpus = [c for c in corpus if c["data"]["abstractNote"] != ""]

    def get_collection_path(col_key: str) -> str:
        if p := collections[col_key]["data"]["parentCollection"]:
            return get_collection_path(p) + "/" + collections[col_key]["data"]["name"]
        else:
            return collections[col_key]["data"]["name"]

    for c in corpus:
        paths = [get_collection_path(col) for col in c["data"]["collections"]]
        c["paths"] = paths
    return corpus


def filter_corpus(corpus: list[dict], pattern: str) -> list[dict]:
    _, filename = mkstemp()
    with open(filename, "w") as file:
        file.write(pattern)
    matcher = parse_gitignore(filename, base_dir="./")
    new_corpus = []
    for c in corpus:
        match_results = [matcher(p) for p in c["paths"]]
        if not any(match_results):
            new_corpus.append(c)
    os.remove(filename)
    return new_corpus


def get_arxiv_paper(query: str, debug: bool = False) -> list[ArxivPaper]:
    client = arxiv.Client(num_retries=10, delay_seconds=10)
    feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")
    if "Feed error for query" in feed.feed.title:
        raise Exception(f"Invalid ARXIV_QUERY: {query}.")
    if not debug:
        papers = []
        all_paper_ids = [
            i.id.removeprefix("oai:arXiv.org:")
            for i in feed.entries
            if i.arxiv_announce_type == "new"
        ]
        bar = tqdm(total=len(all_paper_ids), desc="Retrieving Arxiv papers")
        for i in range(0, len(all_paper_ids), 20):
            search = arxiv.Search(id_list=all_paper_ids[i : i + 20])
            batch = [ArxivPaper(p) for p in client.results(search)]
            bar.update(len(batch))
            papers.extend(batch)
        bar.close()

    else:
        logger.debug("Retrieve 5 arxiv papers regardless of the date.")
        search = arxiv.Search(
            query="cat:cs.AI", sort_by=arxiv.SortCriterion.SubmittedDate
        )
        papers = []
        for i in client.results(search):
            papers.append(ArxivPaper(i))
            if len(papers) == 5:
                break

    return papers


parser = argparse.ArgumentParser(description="Recommender system for academic papers")


def add_argument(*args, **kwargs):
    def get_env(key: str, default=None):
        # handle environment variables generated at Workflow runtime
        # Unset environment variables are passed as '', we should treat them as None
        v = os.environ.get(key)
        if v == "" or v is None:
            return default
        return v

    parser.add_argument(*args, **kwargs)
    arg_full_name = kwargs.get("dest", args[-1][2:])
    env_name = arg_full_name.upper()
    env_value = get_env(env_name)
    if env_value is not None:
        # convert env_value to the specified type
        if kwargs.get("type") == bool:
            env_value = env_value.lower() in ["true", "1"]
        else:
            env_value = kwargs.get("type")(env_value)
        parser.set_defaults(**{arg_full_name: env_value})


# if __name__ == "__main__":

#     add_argument("--zotero_id", type=str, help="Zotero user ID")
#     add_argument("--zotero_key", type=str, help="Zotero API key")
#     add_argument(
#         "--zotero_ignore",
#         type=str,
#         help="Zotero collection to ignore, using gitignore-style pattern.",
#     )
#     add_argument(
#         "--send_empty",
#         type=bool,
#         help="If get no arxiv paper, send empty email",
#         default=False,
#     )
#     add_argument(
#         "--max_paper_num",
#         type=int,
#         help="Maximum number of papers to recommend",
#         default=100,
#     )
#     add_argument("--arxiv_query", type=str, help="Arxiv search query")
#     add_argument("--smtp_server", type=str, help="SMTP server")
#     add_argument("--smtp_port", type=int, help="SMTP port")
#     add_argument("--sender", type=str, help="Sender email address")
#     add_argument("--receiver", type=str, help="Receiver email address")
#     add_argument("--sender_password", type=str, help="Sender email password")
#     add_argument(
#         "--use_llm_api",
#         type=bool,
#         help="Use OpenAI API to generate TLDR",
#         default=False,
#     )
#     add_argument(
#         "--openai_api_key",
#         type=str,
#         help="OpenAI API key",
#         default=None,
#     )
#     add_argument(
#         "--openai_api_base",
#         type=str,
#         help="OpenAI API base URL",
#         default="https://api.openai.com/v1",
#     )
#     add_argument(
#         "--model_name",
#         type=str,
#         help="LLM Model Name",
#         default="gpt-4o",
#     )
#     add_argument(
#         "--language",
#         type=str,
#         help="Language of TLDR",
#         default="English",
#     )
#     parser.add_argument("--debug", action="store_true", help="Debug mode")
#     args = parser.parse_args()
#     assert (
#         not args.use_llm_api or args.openai_api_key is not None
#     )  # If use_llm_api is True, openai_api_key must be provided
#     if args.debug:
#         logger.remove()
#         logger.add(sys.stdout, level="DEBUG")
#         logger.debug("Debug mode is on.")
#     else:
#         logger.remove()
#         logger.add(sys.stdout, level="INFO")

#     logger.info("Retrieving Zotero corpus...")
#     corpus = get_zotero_corpus(args.zotero_id, args.zotero_key)
#     logger.info(f"Retrieved {len(corpus)} papers from Zotero.")
#     if args.zotero_ignore:
#         logger.info(f"Ignoring papers in:\n {args.zotero_ignore}...")
#         corpus = filter_corpus(corpus, args.zotero_ignore)
#         logger.info(f"Remaining {len(corpus)} papers after filtering.")
#     logger.info("Retrieving Arxiv papers...")
#     papers = get_arxiv_paper(args.arxiv_query, args.debug)
#     if len(papers) == 0:
#         logger.info(
#             "No new papers found. Yesterday maybe a holiday and no one submit their work :). If this is not the case, please check the ARXIV_QUERY."
#         )
#         if not args.send_empty:
#             exit(0)
#     else:
#         logger.info("Reranking papers...")
#         papers = rerank_paper(papers, corpus)
#         if args.max_paper_num != -1:
#             papers = papers[: args.max_paper_num]
#         if args.use_llm_api:
#             logger.info("Using OpenAI API as global LLM.")
#             set_global_llm(
#                 api_key=args.openai_api_key,
#                 base_url=args.openai_api_base,
#                 model=args.model_name,
#                 lang=args.language,
#             )
#         else:
#             logger.info("Using Local LLM as global LLM.")
#             set_global_llm(lang=args.language)

#     html = render_email(papers)
#     logger.info("Sending email...")
#     send_email(
#         args.sender,
#         args.receiver,
#         args.sender_password,
#         args.smtp_server,
#         args.smtp_port,
#         html,
#     )
#     logger.success(
#         "Email sent successfully! If you don't receive the email, please check the configuration and the junk box."
#     )


def get_hf_daily_papers(date: str) -> list[dict]:
    """
    Fetch daily papers from HuggingFace API.
    Returns a list of dicts with 'paper' key containing metadata.
    """
    url = f"https://huggingface.co/api/daily_papers?date={date}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def run_hf_daily_flow(args):
    """
    Execute the HuggingFace Daily Papers workflow:
    1. Fetch HF daily papers.
    2. Convert to ArxivPaper objects (for caching, etc.).
    3. Generate bilingual summaries.
    4. Extract figures.
    5. Render and send email.
    """
    import datetime

    # Use yesterday's date if not specified, or today?
    # User said "obtain previous day's daily_papers".
    # If running today (Jan 5), we likely want Jan 4?
    # Or just use the date passed or default to "yesterday".
    # Let's default to yesterday as per request "前一日".
    yesterday = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    logger.info(f"Fetching HuggingFace Daily Papers for {yesterday}...")
    if args.date:
        yesterday = args.date
    # Initialize Arxiv Client
    client = arxiv.Client()

    try:
        hf_data = get_hf_daily_papers(yesterday)
    except Exception as e:
        logger.error(f"Failed to fetch HF papers: {e}")
        return

    if not hf_data:
        logger.info("No HuggingFace papers found for this date.")
        return

    logger.info(f"Found {len(hf_data)} papers from HF.")

    papers_to_process = []
    # Limit to top 5 or so? HF daily usually has ~10-20. Let's process top 5 for cost/time?
    # User didn't specify limit, but let's stick to a reasonable number.
    hf_data = hf_data[:5]

    for item in tqdm(hf_data, desc="Processing HF Papers"):
        p_meta = item["paper"]
        arxiv_id = p_meta.get("id")
        if not arxiv_id:
            continue

        # Create ArxivPaper object to reuse logic (cache, source download)
        # We might need to construct a mock result or fetch from arxiv to get full metadata
        # if HF metadata is insufficient. HF metadata seems rich enough for basic info,
        # but ArxivPaper expects an arxiv.Result.
        # Let's fetch the actual arxiv result to ensure consistency and use ArxivPaper methods transparently.
        try:
            search = arxiv.Search(id_list=[arxiv_id])
            results = list(client.results(search))
            if not results:
                logger.warning(f"Arxiv ID {arxiv_id} not found in Arxiv API.")
                continue
            paper_obj = ArxivPaper(results[0])

            # Use HF upvotes as score
            paper_obj.score = p_meta.get("upvotes", 0)

            # Generate bilingual summary
            summary_dict = paper_obj.bilingual_summary

            # Extract image
            img_bytes = paper_obj.image_content

            papers_to_process.append(
                {
                    "title": paper_obj.title,
                    "authors": [a.name for a in paper_obj.authors],
                    "score": paper_obj.score,
                    "arxiv_id": paper_obj.arxiv_id,
                    "pdf_url": paper_obj.pdf_url,
                    "code_url": paper_obj.code_url,
                    "bilingual_summary": summary_dict,
                    "image_content": img_bytes,
                }
            )

        except Exception as e:
            logger.error(f"Error processing paper {arxiv_id}: {e}")
            continue

    if not papers_to_process:
        logger.info("No papers processed successfully.")
        return

    # Render Email
    html, attachments = render_hf_email(papers_to_process, yesterday)
    with open("test.html", "w", encoding="utf-8") as f:
        f.write(html)
    # Send Email
    subject = f"HuggingFace Daily Papers {yesterday}"
    logger.info(f"Sending HF Daily Email: {subject}...")
    # Note: send_email in construct_email.py handles subject logic internaly based on content
    send_email(
        args.sender,
        args.receiver,
        args.sender_password,
        args.smtp_server,
        args.smtp_port,
        html,
        attachments,
    )
    logger.success("HF Daily Email sent successfully!")


if __name__ == "__main__":

    add_argument("--zotero_id", type=str, help="Zotero user ID")
    add_argument("--zotero_key", type=str, help="Zotero API key")
    add_argument(
        "--zotero_ignore",
        type=str,
        help="Zotero collection to ignore, using gitignore-style pattern.",
    )
    add_argument(
        "--send_empty",
        type=bool,
        help="If get no arxiv paper, send empty email",
        default=False,
    )
    add_argument(
        "--max_paper_num",
        type=int,
        help="Maximum number of papers to recommend",
        default=100,
    )
    add_argument("--arxiv_query", type=str, help="Arxiv search query")
    add_argument("--smtp_server", type=str, help="SMTP server")
    add_argument("--smtp_port", type=int, help="SMTP port")
    add_argument("--sender", type=str, help="Sender email address")
    add_argument("--receiver", type=str, help="Receiver email address")
    add_argument("--sender_password", type=str, help="Sender email password")
    add_argument(
        "--openai_api_key",
        type=str,
        help="OpenAI API key",
        default=None,
    )
    add_argument(
        "--openai_api_base",
        type=str,
        help="OpenAI API base URL",
        default="https://api.openai.com/v1",
    )
    add_argument(
        "--model_name",
        type=str,
        help="LLM Model Name",
        default="gpt-4o",
    )
    add_argument(
        "--language",
        type=str,
        help="Language of TLDR",
        default="English",
    )
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()

    # Initialize Global LLM
    if args.openai_api_key:
        logger.info("Using OpenAI API as global LLM.")
        set_global_llm(
            api_key=args.openai_api_key,
            base_url=args.openai_api_base,
            model=args.model_name,
            lang=args.language,
        )
    else:
        # If no API key, some features will fail.
        logger.warning("No OpenAI API Key provided. LLM features will fail.")
        set_global_llm(lang=args.language)

    if args.debug:
        logger.remove()
        logger.add(sys.stdout, level="DEBUG")
        logger.debug("Debug mode is on.")
    else:
        logger.remove()
        logger.add(sys.stdout, level="INFO")

    # 1. Run HuggingFace Daily Flow
    logger.info("=== Starting HuggingFace Daily Papers Flow ===")
    run_hf_daily_flow(args)
    logger.info("=== Finished HuggingFace Daily Papers Flow ===")

    # 2. Run Existing Arxiv Flow
    logger.info("=== Starting Arxiv Recommendation Flow ===")

    logger.info("Retrieving Zotero corpus...")
    corpus = get_zotero_corpus(args.zotero_id, args.zotero_key)
    logger.info(f"Retrieved {len(corpus)} papers from Zotero.")
    if args.zotero_ignore:
        logger.info(f"Ignoring papers in:\n {args.zotero_ignore}...")
        corpus = filter_corpus(corpus, args.zotero_ignore)
        logger.info(f"Remaining {len(corpus)} papers after filtering.")
    logger.info("Retrieving Arxiv papers...")
    papers = get_arxiv_paper(args.arxiv_query, args.debug)
    if len(papers) == 0:
        logger.info(
            "No new papers found. Yesterday maybe a holiday and no one submit their work :). If this is not the case, please check the ARXIV_QUERY."
        )
        if not args.send_empty:
            # If HF flow ran, we might not want to exit?
            # But legacy logic exits here.
            # Let's keep it but just log.
            pass
    else:
        logger.info("Reranking papers...")
        papers = rerank_paper(papers, corpus)
        if args.max_paper_num != -1:
            papers = papers[: args.max_paper_num]

        html = render_email(papers)
        logger.info("Sending email...")
        send_email(
            args.sender,
            args.receiver,
            args.sender_password,
            args.smtp_server,
            args.smtp_port,
            html,
        )
        logger.success(
            "Email sent successfully! If you don't receive the email, please check the configuration and the junk box."
        )

    logger.info("=== Finished Arxiv Recommendation Flow ===")
