from types import SimpleNamespace

import requests

import zotero_arxiv_daily.hf_daily as hf_daily


class StubResponse:
    def __init__(
        self,
        status_code: int,
        chunks: list[bytes] | None = None,
        headers: dict[str, str] | None = None,
        text: str = "",
        content: bytes | None = None,
    ):
        self.status_code = status_code
        self._chunks = chunks or []
        self.headers = headers or {}
        self.text = text
        self.content = content if content is not None else b"".join(self._chunks)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"{self.status_code} Error",
                response=SimpleNamespace(status_code=self.status_code),
            )

    def iter_content(self, chunk_size: int):
        yield from self._chunks


class StubPaper:
    def __init__(self, source_url: str = "https://arxiv.org/e-print/2605.12345"):
        self._source_url = source_url
        self.entry_id = "https://arxiv.org/abs/2605.12345"
        self.pdf_url = "https://arxiv.org/pdf/2605.12345"

    def source_url(self):
        return self._source_url

    def get_short_id(self):
        return "2605.12345v1"


def test_extract_source_archive_content_retries_incomplete_download(monkeypatch):
    responses = [
        StubResponse(200, [b"abc"], {"Content-Length": "6"}),
        StubResponse(200, [b"abc", b"def"], {"Content-Length": "6"}),
    ]

    monkeypatch.setattr(hf_daily, "sleep", lambda _: None)
    monkeypatch.setattr(hf_daily.requests, "get", lambda *a, **kw: responses.pop(0))

    assert hf_daily._extract_source_archive_content(StubPaper()) == b"abcdef"


def test_extract_source_archive_content_returns_none_after_429(monkeypatch):
    monkeypatch.setattr(hf_daily, "SOURCE_DOWNLOAD_RETRIES", 1)
    monkeypatch.setattr(hf_daily.requests, "get", lambda *a, **kw: StubResponse(429))

    assert hf_daily._extract_source_archive_content(StubPaper()) is None


def test_extract_figures_from_html_parses_caption_and_resolves_url(monkeypatch):
    html = """
    <html><body>
      <figure>
        <img src="x1.png" alt="overview">
        <figcaption>Figure 1: Model architecture overview.</figcaption>
      </figure>
    </body></html>
    """
    monkeypatch.setattr(
        hf_daily.requests,
        "get",
        lambda *a, **kw: StubResponse(200, text=html),
    )

    figures = hf_daily.extract_figures_from_html(StubPaper())

    assert figures == [
        {
            "file": "x1.png",
            "url": "https://arxiv.org/html/x1.png",
            "caption": "Figure 1: Model architecture overview.",
            "source": "html",
        }
    ]


def test_extract_image_content_from_html_prefers_architecture_caption(monkeypatch):
    png_bytes = b"\x89PNG\r\n\x1a\nfake"
    requested_urls = []

    def fake_get(url, *args, **kwargs):
        requested_urls.append(url)
        return StubResponse(
            200,
            content=png_bytes,
            headers={"Content-Type": "image/png"},
        )

    monkeypatch.setattr(hf_daily.requests, "get", fake_get)
    figures = [
        {
            "file": "result.png",
            "url": "https://arxiv.org/html/result.png",
            "caption": "Figure 3: Benchmark results.",
            "source": "html",
        },
        {
            "file": "arch.png",
            "url": "https://arxiv.org/html/arch.png",
            "caption": "Figure 2: Overall model architecture.",
            "source": "html",
        },
    ]

    image = hf_daily.extract_image_content_from_html(StubPaper(), figures=figures)

    assert image == png_bytes
    assert requested_urls == ["https://arxiv.org/html/arch.png"]


def test_extract_image_content_uses_html_before_pdf_or_source(monkeypatch):
    monkeypatch.setattr(
        hf_daily,
        "extract_image_content_from_html",
        lambda *a, **kw: b"\x89PNG\r\n\x1a\nhtml",
    )
    monkeypatch.setattr(
        hf_daily,
        "extract_image_content_from_pdf",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("PDF should not run")),
    )
    monkeypatch.setattr(
        hf_daily,
        "_extract_source_archive_content",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("source should not run")),
    )

    assert hf_daily.extract_image_content(StubPaper()) == b"\x89PNG\r\n\x1a\nhtml"


def test_run_hf_daily_flow_uses_hf_metadata_without_arxiv_api_by_default(monkeypatch):
    sent = []
    metadata = {
        "id": "2605.18747",
        "title": "Code as Agent Harness",
        "summary": "Survey summary",
        "authors": [{"name": "Test Author"}],
        "upvotes": 159,
    }

    class UnexpectedClient:
        def __init__(self, **kwargs):
            raise AssertionError("arXiv API should not be called by default")

        def results(self, search):
            raise AssertionError("arXiv API should not be called by default")

    monkeypatch.setattr(hf_daily, "get_hf_daily_papers", lambda date: [{"paper": metadata}])
    monkeypatch.setattr(hf_daily.arxiv, "Client", UnexpectedClient)
    monkeypatch.setattr(hf_daily, "extract_text_from_html", lambda paper: "full text")
    monkeypatch.setattr(hf_daily, "extract_text_from_pdf", lambda paper: None)
    monkeypatch.setattr(hf_daily, "extract_text_from_tar", lambda paper: None)
    monkeypatch.setattr(hf_daily, "extract_figures_from_html", lambda paper: [])
    monkeypatch.setattr(hf_daily, "extract_image_content", lambda *a, **kw: None)
    monkeypatch.setattr(hf_daily, "fetch_code_url", lambda arxiv_id: None)
    monkeypatch.setattr(
        hf_daily,
        "generate_bilingual_summary",
        lambda **kwargs: {
            "problem": {"cn": "问题", "en": "Problem"},
            "solution": {"cn": "方法", "en": "Solution"},
            "result": {"cn": "结果", "en": "Result"},
            "keywords": {"cn": [], "en": []},
            "selected_figure": None,
        },
    )
    monkeypatch.setattr(
        hf_daily,
        "send_email",
        lambda config, html, attachments=None, subject=None: sent.append(
            (html, attachments, subject)
        ),
    )

    config = SimpleNamespace(
        executor={"hf_date": "2026-05-19", "hf_max_paper_num": 1, "debug": False},
        llm={},
    )

    hf_daily.run_hf_daily_flow(config, openai_client=object())

    assert len(sent) == 1
    html, attachments, subject = sent[0]
    assert "Code as Agent Harness" in html
    assert subject == "HuggingFace Daily Papers 2026-05-19"
