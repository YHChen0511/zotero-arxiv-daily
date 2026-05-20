"""Tests for ArxivRetriever."""

import time
from types import SimpleNamespace

import feedparser

from zotero_arxiv_daily.retriever.arxiv_retriever import ArxivRetriever, _run_with_hard_timeout
import zotero_arxiv_daily.retriever.arxiv_retriever as arxiv_retriever


class FeedEntry(dict):
    def __init__(self, paper_id: str, announce_type: str = "new"):
        super().__init__(arxiv_announce_type=announce_type)
        self.id = f"oai:arXiv.org:{paper_id}"
        self.title = f"Paper {paper_id}"


def _sleep_and_return(value: str, delay_seconds: float) -> str:
    time.sleep(delay_seconds)
    return value


def _raise_runtime_error() -> None:
    raise RuntimeError("boom")


def test_arxiv_retriever(config, mock_feedparser, monkeypatch):
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)

    # The RSS fixture gives us paper IDs.  After feedparser, the code calls
    # arxiv.Client().results(search) which makes real HTTP requests.  We mock
    # the arxiv Client so the test stays offline.
    new_entries = [
        e for e in mock_feedparser.entries
        if e.get("arxiv_announce_type", "new") == "new"
    ]
    paper_ids = [e.id.removeprefix("oai:arXiv.org:") for e in new_entries]

    # Build fake ArxivResult-like objects matching each RSS entry
    fake_results = []
    for entry in new_entries:
        pid = entry.id.removeprefix("oai:arXiv.org:")
        fake_results.append(SimpleNamespace(
            title=entry.title,
            authors=[SimpleNamespace(name="Test Author")],
            summary="Test abstract",
            pdf_url=f"https://arxiv.org/pdf/{pid}",
            entry_id=f"https://arxiv.org/abs/{pid}",
            source_url=lambda pid=pid: f"https://arxiv.org/e-print/{pid}",
        ))

    class FakeClient:
        def __init__(self, **kw):
            pass
        def results(self, search):
            return iter(fake_results)

    monkeypatch.setattr(arxiv_retriever.arxiv, "Client", FakeClient)

    # Skip file downloads in convert_to_paper
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_html", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_pdf", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_tar", lambda paper: None)

    retriever = ArxivRetriever(config)
    papers = retriever.retrieve_papers()

    assert len(papers) == len(new_entries)
    assert set(p.title for p in papers) == set(e.title for e in new_entries)


def test_arxiv_retriever_skips_persistent_429_batch(config, monkeypatch):
    entries = [FeedEntry(f"2605.{index:05d}v1") for index in range(21)]
    fake_feed = SimpleNamespace(feed=SimpleNamespace(title="ok"), entries=entries)
    warnings: list[str] = []

    class FakeClient:
        def __init__(self, **kw):
            pass

        def results(self, search):
            if search.id_list[0] == "2605.00000v1":
                raise arxiv_retriever.arxiv.HTTPError("https://export.arxiv.org/api/query", 0, 429)
            return iter([
                SimpleNamespace(
                    title="Paper 2605.00020v1",
                    authors=[SimpleNamespace(name="Test Author")],
                    summary="Test abstract",
                    pdf_url="https://arxiv.org/pdf/2605.00020v1",
                    entry_id="https://arxiv.org/abs/2605.00020v1",
                    source_url=lambda: "https://arxiv.org/e-print/2605.00020v1",
                )
            ])

    monkeypatch.setattr(arxiv_retriever.feedparser, "parse", lambda _: fake_feed)
    monkeypatch.setattr(arxiv_retriever.arxiv, "Client", FakeClient)
    monkeypatch.setattr(arxiv_retriever, "sleep", lambda _: None)
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_html", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_pdf", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_tar", lambda paper: None)

    papers = ArxivRetriever(config).retrieve_papers()

    assert len(papers) == 1
    assert papers[0].title == "Paper 2605.00020v1"
    assert any("Skipping arXiv batch 0" in warning for warning in warnings)


def test_run_with_hard_timeout_returns_value():
    result = _run_with_hard_timeout(
        _sleep_and_return, ("done", 0.01), timeout=5, operation="test op", paper_title="paper"
    )
    assert result == "done"


def test_run_with_hard_timeout_returns_none_on_timeout(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    result = _run_with_hard_timeout(
        _sleep_and_return, ("done", 1.0), timeout=0.01, operation="test op", paper_title="paper"
    )
    assert result is None
    assert "timed out" in warnings[0]


def test_run_with_hard_timeout_returns_none_on_failure(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    result = _run_with_hard_timeout(
        _raise_runtime_error, (), timeout=5, operation="test op", paper_title="paper"
    )
    assert result is None
    assert "boom" in warnings[0]
