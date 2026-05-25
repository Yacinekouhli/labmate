"""Literature search and citation graph primitives.

The first implementation keeps the interfaces dependency-free and explicit so
CLI and MCP wrappers can reuse the same contracts later. Network access is
optional: backends accept injectable fetchers, and tests can run entirely from
fixtures.
"""

from __future__ import annotations

import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Literal, Protocol

Fetch = Callable[[str], bytes | str]
CitationRelation = Literal["references", "cited_by"]

_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV_API_URL = "https://export.arxiv.org/api/query"


@dataclass(frozen=True)
class Paper:
    """Normalized paper record returned by all literature backends."""

    title: str
    authors: tuple[str, ...]
    year: int | None
    source: str
    ids: Mapping[str, str] = field(default_factory=dict)
    url: str | None = None
    abstract: str | None = None
    snippet: str | None = None
    published_at: str | None = None
    updated_at: str | None = None
    provenance_url: str | None = None
    relevance_signals: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "authors": list(self.authors),
            "year": self.year,
            "source": self.source,
            "ids": dict(self.ids),
            "url": self.url,
            "abstract": self.abstract,
            "snippet": self.snippet,
            "published_at": self.published_at,
            "updated_at": self.updated_at,
            "provenance_url": self.provenance_url,
            "relevance_signals": list(self.relevance_signals),
        }


@dataclass(frozen=True)
class LiteratureSearchResult:
    query: str
    backend: str
    retrieved_at: str
    papers: tuple[Paper, ...]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "backend": self.backend,
            "retrieved_at": self.retrieved_at,
            "papers": [paper.to_dict() for paper in self.papers],
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class CitationEdge:
    source_id: str
    target_id: str
    relation: CitationRelation
    evidence: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation": self.relation,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class CitationGraphResult:
    root: Paper
    backend: str
    retrieved_at: str
    references: tuple[Paper, ...]
    citations: tuple[Paper, ...]
    edges: tuple[CitationEdge, ...]
    depth: int = 1
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "root": self.root.to_dict(),
            "backend": self.backend,
            "retrieved_at": self.retrieved_at,
            "references": [paper.to_dict() for paper in self.references],
            "citations": [paper.to_dict() for paper in self.citations],
            "edges": [edge.to_dict() for edge in self.edges],
            "depth": self.depth,
            "warnings": list(self.warnings),
        }


class LiteratureSearchBackend(Protocol):
    name: str

    def search(
        self,
        query: str,
        *,
        max_results: int = 10,
        since_year: int | None = None,
    ) -> LiteratureSearchResult:
        """Search papers and return normalized records."""


class CitationGraphBackend(Protocol):
    name: str

    def citation_graph(
        self,
        paper_id: str,
        *,
        max_results: int = 20,
        depth: int = 1,
    ) -> CitationGraphResult:
        """Return references and citations for a paper."""


class LocalCorpusBackend:
    """Search and graph backend for local fixtures or checked-in corpora.

    This backend is intentionally simple but useful: it gives agents a stable
    offline substrate for benchmark packs, paper reading lists, or mocked API
    responses while keeping the output shape identical to remote backends.
    """

    name: str

    def __init__(
        self,
        papers: Sequence[Paper],
        *,
        references: Mapping[str, Sequence[str]] | None = None,
        citations: Mapping[str, Sequence[str]] | None = None,
        name: str = "local_corpus",
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.name = name
        self._papers = tuple(papers)
        self._references = {key: tuple(value) for key, value in (references or {}).items()}
        self._citations = {key: tuple(value) for key, value in (citations or {}).items()}
        self._now = now or _utcnow
        self._paper_by_key = _index_papers(self._papers)

    def search(
        self,
        query: str,
        *,
        max_results: int = 10,
        since_year: int | None = None,
    ) -> LiteratureSearchResult:
        tokens = _tokenize(query)
        scored: list[tuple[int, int, str, Paper]] = []

        for paper in self._papers:
            if since_year is not None and paper.year is not None and paper.year < since_year:
                continue

            score, signals = _score_paper(paper, tokens)
            if tokens and score == 0:
                continue

            recency = paper.year or 0
            scored.append(
                (
                    score,
                    recency,
                    paper.title.casefold(),
                    replace(paper, relevance_signals=signals),
                )
            )

        scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
        papers = tuple(item[3] for item in scored[:max_results])
        return LiteratureSearchResult(
            query=query,
            backend=self.name,
            retrieved_at=_isoformat(self._now()),
            papers=papers,
        )

    def citation_graph(
        self,
        paper_id: str,
        *,
        max_results: int = 20,
        depth: int = 1,
    ) -> CitationGraphResult:
        if depth != 1:
            raise ValueError("LocalCorpusBackend currently supports depth=1 only")

        root = self._resolve_paper(paper_id)
        root_id = _canonical_paper_id(root)
        references = self._resolve_many(self._references_for(paper_id, root), max_results)
        citations = self._resolve_many(self._citations_for(paper_id, root), max_results)

        edges = [
            CitationEdge(
                source_id=root_id,
                target_id=_canonical_paper_id(paper),
                relation="references",
                evidence=f"{root.title} lists {paper.title} as prior work.",
            )
            for paper in references
        ]
        edges.extend(
            CitationEdge(
                source_id=_canonical_paper_id(paper),
                target_id=root_id,
                relation="cited_by",
                evidence=f"{paper.title} cites or builds on {root.title}.",
            )
            for paper in citations
        )

        return CitationGraphResult(
            root=root,
            backend=self.name,
            retrieved_at=_isoformat(self._now()),
            references=tuple(references),
            citations=tuple(citations),
            edges=tuple(edges),
            depth=depth,
        )

    def _resolve_paper(self, paper_id: str) -> Paper:
        try:
            return self._paper_by_key[_normalize_key(paper_id)]
        except KeyError as exc:
            raise KeyError(f"Unknown paper id: {paper_id}") from exc

    def _resolve_many(self, paper_ids: Sequence[str], max_results: int) -> list[Paper]:
        papers: list[Paper] = []
        for paper_id in paper_ids[:max_results]:
            papers.append(self._resolve_paper(paper_id))
        return papers

    def _references_for(self, paper_id: str, root: Paper) -> tuple[str, ...]:
        return self._lookup_edges(self._references, paper_id, root)

    def _citations_for(self, paper_id: str, root: Paper) -> tuple[str, ...]:
        return self._lookup_edges(self._citations, paper_id, root)

    def _lookup_edges(
        self,
        edges: Mapping[str, Sequence[str]],
        paper_id: str,
        root: Paper,
    ) -> tuple[str, ...]:
        keys = (paper_id, *_paper_keys(root))
        for key in keys:
            value = edges.get(key) or edges.get(_normalize_key(key))
            if value is not None:
                return tuple(value)
        return ()


class ArxivSearchBackend:
    """Minimal arXiv API search backend with injectable transport."""

    name = "arxiv"

    def __init__(
        self,
        *,
        fetch: Fetch | None = None,
        api_url: str = _ARXIV_API_URL,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._fetch = fetch or _fetch_url
        self._api_url = api_url
        self._now = now or _utcnow

    def search(
        self,
        query: str,
        *,
        max_results: int = 10,
        since_year: int | None = None,
    ) -> LiteratureSearchResult:
        url = self._build_url(query, max_results)
        payload = self._fetch(url)
        xml = payload.decode("utf-8") if isinstance(payload, bytes) else payload
        papers = tuple(
            paper
            for paper in _parse_arxiv_feed(xml, query=query, provenance_url=url)
            if since_year is None or paper.year is None or paper.year >= since_year
        )

        return LiteratureSearchResult(
            query=query,
            backend=self.name,
            retrieved_at=_isoformat(self._now()),
            papers=papers[:max_results],
        )

    def _build_url(self, query: str, max_results: int) -> str:
        params = urllib.parse.urlencode(
            {
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": max_results,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
        )
        return f"{self._api_url}?{params}"


def search_literature(
    query: str,
    *,
    backend: LiteratureSearchBackend,
    max_results: int = 10,
    since_year: int | None = None,
) -> LiteratureSearchResult:
    return backend.search(query, max_results=max_results, since_year=since_year)


def citation_graph(
    paper_id: str,
    *,
    backend: CitationGraphBackend,
    max_results: int = 20,
    depth: int = 1,
) -> CitationGraphResult:
    return backend.citation_graph(paper_id, max_results=max_results, depth=depth)


def _parse_arxiv_feed(xml: str, *, query: str, provenance_url: str) -> Iterable[Paper]:
    root = ET.fromstring(xml)
    tokens = _tokenize(query)
    for entry in root.findall(f"{_ATOM}entry"):
        title = _compact_text(entry.findtext(f"{_ATOM}title"))
        abstract = _compact_text(entry.findtext(f"{_ATOM}summary"))
        published_at = _compact_text(entry.findtext(f"{_ATOM}published"))
        updated_at = _compact_text(entry.findtext(f"{_ATOM}updated"))
        url = _compact_text(entry.findtext(f"{_ATOM}id"))
        authors = tuple(
            _compact_text(author.findtext(f"{_ATOM}name"))
            for author in entry.findall(f"{_ATOM}author")
            if _compact_text(author.findtext(f"{_ATOM}name"))
        )
        arxiv_id = _arxiv_id_from_url(url)
        paper = Paper(
            title=title,
            authors=authors,
            year=_year_from_iso8601(published_at),
            source="arxiv",
            ids={"arxiv": arxiv_id} if arxiv_id else {},
            url=url,
            abstract=abstract,
            snippet=_snippet(abstract, tokens),
            published_at=published_at,
            updated_at=updated_at,
            provenance_url=provenance_url,
        )
        _, signals = _score_paper(paper, tokens)
        yield replace(paper, relevance_signals=signals)


def _fetch_url(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "labmate/0.1.0 (https://github.com/Yacinekouhli/labmate)"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return response.read()


def _index_papers(papers: Sequence[Paper]) -> dict[str, Paper]:
    indexed: dict[str, Paper] = {}
    for paper in papers:
        for key in _paper_keys(paper):
            indexed[_normalize_key(key)] = paper
    return indexed


def _paper_keys(paper: Paper) -> tuple[str, ...]:
    keys = [paper.title]
    keys.extend(paper.ids.values())
    keys.extend(f"{name}:{value}" for name, value in paper.ids.items())
    if paper.url:
        keys.append(paper.url)
    return tuple(keys)


def _canonical_paper_id(paper: Paper) -> str:
    for key in ("doi", "arxiv", "semantic_scholar", "openalex"):
        if key in paper.ids:
            return f"{key}:{paper.ids[key]}"
    if paper.ids:
        name, value = next(iter(paper.ids.items()))
        return f"{name}:{value}"
    return _normalize_key(paper.title)


def _score_paper(paper: Paper, tokens: tuple[str, ...]) -> tuple[int, tuple[str, ...]]:
    if not tokens:
        return 0, ()

    fields = {
        "title": paper.title,
        "abstract": paper.abstract or "",
        "snippet": paper.snippet or "",
    }
    score = 0
    signals: list[str] = []
    for field_name, value in fields.items():
        text = value.casefold()
        matched = [token for token in tokens if token in text]
        if not matched:
            continue
        weight = 3 if field_name == "title" else 1
        score += len(matched) * weight
        signals.append(f"matched {', '.join(sorted(set(matched)))} in {field_name}")

    if paper.year is not None:
        signals.append(f"published in {paper.year}")
    if paper.provenance_url:
        signals.append(f"retrieved from {paper.source}")

    return score, tuple(signals)


def _snippet(text: str | None, tokens: tuple[str, ...], *, window: int = 180) -> str | None:
    if not text:
        return None
    compact = _compact_text(text)
    lowered = compact.casefold()
    matches = [lowered.find(token) for token in tokens if lowered.find(token) >= 0]
    if not matches:
        return compact[:window]
    start = max(0, min(matches) - window // 3)
    end = min(len(compact), start + window)
    return compact[start:end]


def _tokenize(query: str) -> tuple[str, ...]:
    return tuple(token for token in re.findall(r"[a-zA-Z0-9]+", query.casefold()) if len(token) > 2)


def _normalize_key(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().casefold())


def _compact_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def _arxiv_id_from_url(url: str | None) -> str | None:
    if not url:
        return None
    return url.rstrip("/").rsplit("/", maxsplit=1)[-1] or None


def _year_from_iso8601(value: str | None) -> int | None:
    if not value:
        return None
    match = re.match(r"(\d{4})", value)
    return int(match.group(1)) if match else None


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _isoformat(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
