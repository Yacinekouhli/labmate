from datetime import UTC, datetime

import pytest

from labmate.tools.docs import OfficialDocsBackend, fetch_docs


def fixed_now() -> datetime:
    return datetime(2026, 5, 25, 12, 0, tzinfo=UTC)


def test_catalog_lookup_returns_official_docs_with_provenance() -> None:
    backend = OfficialDocsBackend(now=fixed_now)

    result = fetch_docs(
        "tabular csv missing values",
        backend=backend,
        max_results=3,
    )

    assert result.backend == "official_docs"
    assert result.retrieved_at == "2026-05-25T12:00:00Z"
    assert result.documents
    assert result.documents[0].title == "pandas.read_csv"
    assert result.documents[0].provenance_url == result.documents[0].url
    assert result.documents[0].fetched is False
    assert any("matched" in signal for signal in result.documents[0].matched_signals)
    assert "pass url" in result.warnings[-1]


def test_huggingface_catalog_scope_filters_results() -> None:
    backend = OfficialDocsBackend(name="huggingface", now=fixed_now)

    result = fetch_docs("trainer evaluation checkpoints", backend=backend)

    assert result.documents
    assert all(document.source == "huggingface" for document in result.documents)
    assert result.documents[0].title == "Hugging Face Transformers Trainer"


def test_exact_url_fetch_extracts_title_and_snippet() -> None:
    requested_urls: list[str] = []

    def fake_fetch(url: str) -> str:
        requested_urls.append(url)
        return """
<!doctype html>
<html>
  <head><title>Pipeline — scikit-learn documentation</title></head>
  <body>
    <h1>Pipeline</h1>
    <p>Pipeline chains preprocessing transformers and an estimator.</p>
    <p>Use ColumnTransformer for heterogeneous tabular feature columns.</p>
  </body>
</html>
"""

    backend = OfficialDocsBackend(fetch=fake_fetch, now=fixed_now)

    result = fetch_docs(
        "pipeline columntransformer tabular",
        backend=backend,
        url="https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html",
    )

    assert requested_urls == [
        "https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html"
    ]
    assert result.documents[0].title == "Pipeline — scikit-learn documentation"
    assert result.documents[0].source == "scikit-learn"
    assert result.documents[0].fetched is True
    assert result.documents[0].content_chars is not None
    assert "ColumnTransformer" in result.documents[0].snippet
    assert any("body" in signal for signal in result.documents[0].matched_signals)
    assert "exact documentation URL" in result.warnings[0]


def test_exact_url_fetch_rejects_non_http_urls() -> None:
    backend = OfficialDocsBackend(now=fixed_now)

    with pytest.raises(ValueError, match="absolute http"):
        fetch_docs("pipeline", backend=backend, url="file:///tmp/docs.html")
