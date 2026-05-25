from datetime import UTC, datetime

import pytest

from labmate.tools.literature import (
    ArxivSearchBackend,
    LocalCorpusBackend,
    Paper,
    citation_graph,
    search_literature,
)


def fixed_now() -> datetime:
    return datetime(2026, 5, 25, 12, 0, tzinfo=UTC)


def kaggle_tabular_backend() -> LocalCorpusBackend:
    papers = (
        Paper(
            title="Tabular Prior-Data Fitted Networks",
            authors=("Noah Hollmann", "Samuel Muller"),
            year=2022,
            source="fixture",
            ids={"arxiv": "2207.01848"},
            url="https://arxiv.org/abs/2207.01848",
            abstract=(
                "Transformer-based prior-data fitted networks can provide strong "
                "baselines for small tabular classification problems."
            ),
            provenance_url="fixture://papers/tabpfn",
        ),
        Paper(
            title="TabNet: Attentive Interpretable Tabular Learning",
            authors=("Sercan O. Arik", "Tomas Pfister"),
            year=2019,
            source="fixture",
            ids={"arxiv": "1908.07442"},
            url="https://arxiv.org/abs/1908.07442",
            abstract=(
                "TabNet uses sequential attention to choose tabular features and "
                "can support interpretable supervised learning."
            ),
            provenance_url="fixture://papers/tabnet",
        ),
        Paper(
            title="ImageNet Classification with Deep Convolutional Neural Networks",
            authors=("Alex Krizhevsky", "Ilya Sutskever", "Geoffrey Hinton"),
            year=2012,
            source="fixture",
            ids={"doi": "10.1145/3065386"},
            abstract="Convolutional networks for image classification.",
            provenance_url="fixture://papers/alexnet",
        ),
    )
    return LocalCorpusBackend(
        papers,
        references={"arxiv:2207.01848": ("arxiv:1908.07442",)},
        citations={"arxiv:1908.07442": ("arxiv:2207.01848",)},
        now=fixed_now,
    )


def test_local_corpus_search_returns_recent_relevant_papers_with_provenance() -> None:
    backend = kaggle_tabular_backend()

    result = search_literature(
        "tabular classification baseline kaggle",
        backend=backend,
        max_results=5,
        since_year=2018,
    )

    assert result.backend == "local_corpus"
    assert result.retrieved_at == "2026-05-25T12:00:00Z"
    assert [paper.ids.get("arxiv") for paper in result.papers] == ["2207.01848", "1908.07442"]

    first = result.papers[0]
    assert first.title == "Tabular Prior-Data Fitted Networks"
    assert first.year == 2022
    assert first.url == "https://arxiv.org/abs/2207.01848"
    assert first.provenance_url == "fixture://papers/tabpfn"
    assert any("matched" in signal for signal in first.relevance_signals)
    assert "ImageNet" not in [paper.title for paper in result.papers]


def test_local_citation_graph_includes_reference_and_citation_edges() -> None:
    backend = kaggle_tabular_backend()

    result = citation_graph("arxiv:1908.07442", backend=backend)

    assert result.root.title == "TabNet: Attentive Interpretable Tabular Learning"
    assert [paper.ids["arxiv"] for paper in result.citations] == ["2207.01848"]
    assert result.references == ()
    assert result.edges[0].source_id == "arxiv:2207.01848"
    assert result.edges[0].target_id == "arxiv:1908.07442"
    assert result.edges[0].relation == "cited_by"


def test_local_citation_graph_rejects_unsupported_depth() -> None:
    backend = kaggle_tabular_backend()

    with pytest.raises(ValueError, match="depth=1"):
        citation_graph("arxiv:2207.01848", backend=backend, depth=2)


def test_arxiv_backend_parses_mocked_feed_and_keeps_fetcher_optional() -> None:
    requested_urls: list[str] = []

    def fake_fetch(url: str) -> str:
        requested_urls.append(url)
        return """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.12345v1</id>
    <updated>2024-01-12T09:00:00Z</updated>
    <published>2024-01-10T09:00:00Z</published>
    <title>Learning Robust Tabular Models for Noisy Competitions</title>
    <summary>
      We study robust tabular classification models under noisy labels and
      distribution shift for competition settings.
    </summary>
    <author><name>Ada Researcher</name></author>
    <author><name>Max Benchmark</name></author>
  </entry>
</feed>
"""

    backend = ArxivSearchBackend(fetch=fake_fetch, now=fixed_now)

    result = backend.search("tabular noisy classification", max_results=3, since_year=2024)

    assert requested_urls
    assert "search_query=all%3Atabular+noisy+classification" in requested_urls[0]
    assert result.backend == "arxiv"
    assert result.retrieved_at == "2026-05-25T12:00:00Z"
    assert len(result.papers) == 1

    paper = result.papers[0]
    assert paper.title == "Learning Robust Tabular Models for Noisy Competitions"
    assert paper.authors == ("Ada Researcher", "Max Benchmark")
    assert paper.year == 2024
    assert paper.ids == {"arxiv": "2401.12345v1"}
    assert paper.url == "http://arxiv.org/abs/2401.12345v1"
    assert paper.provenance_url == requested_urls[0]
    assert paper.snippet is not None
    assert "noisy labels" in paper.snippet
    assert any("matched" in signal for signal in paper.relevance_signals)


def test_search_results_serialize_to_json_ready_dicts() -> None:
    backend = kaggle_tabular_backend()

    payload = search_literature("tabular", backend=backend, max_results=1).to_dict()

    assert payload["query"] == "tabular"
    assert payload["papers"][0]["ids"] == {"arxiv": "2207.01848"}
    assert payload["papers"][0]["authors"] == ["Noah Hollmann", "Samuel Muller"]
