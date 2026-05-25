"""Read-only framework documentation lookup and exact-page fetching."""

from __future__ import annotations

import re
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from html.parser import HTMLParser

Fetch = Callable[[str], bytes | str]


@dataclass(frozen=True)
class DocReference:
    title: str
    url: str
    source: str
    description: str | None = None
    snippet: str | None = None
    provenance_url: str | None = None
    matched_signals: tuple[str, ...] = ()
    fetched: bool = False
    content_chars: int | None = None
    topics: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "url": self.url,
            "source": self.source,
            "description": self.description,
            "snippet": self.snippet,
            "provenance_url": self.provenance_url,
            "matched_signals": list(self.matched_signals),
            "fetched": self.fetched,
            "content_chars": self.content_chars,
            "topics": list(self.topics),
        }


@dataclass(frozen=True)
class DocsFetchResult:
    query: str
    backend: str
    retrieved_at: str
    documents: tuple[DocReference, ...]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "backend": self.backend,
            "retrieved_at": self.retrieved_at,
            "documents": [document.to_dict() for document in self.documents],
            "warnings": list(self.warnings),
        }


class OfficialDocsBackend:
    """Small official-docs backend with catalog lookup and exact URL fetch."""

    def __init__(
        self,
        *,
        name: str = "official_docs",
        fetch: Fetch | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.name = name
        self._fetch = fetch or _fetch_url
        self._now = now or _utcnow

    def fetch_docs(
        self,
        query: str,
        *,
        url: str | None = None,
        max_results: int = 5,
    ) -> DocsFetchResult:
        if max_results < 1:
            raise ValueError("max_results must be positive")

        tokens = _tokenize(query)
        warnings: list[str] = []

        if url:
            documents = (self._fetch_exact_url(url, tokens),)
            warnings.append(
                "Fetched an exact documentation URL; "
                "verify examples against local package versions."
            )
        else:
            documents = tuple(_rank_catalog(_catalog_for_backend(self.name), tokens, max_results))
            if not documents:
                warnings.append("No built-in documentation catalog entries matched the query.")
            warnings.append(
                "Catalog results include official links only; pass url to fetch current page text."
            )

        return DocsFetchResult(
            query=query,
            backend=self.name,
            retrieved_at=_isoformat(self._now()),
            documents=documents,
            warnings=tuple(warnings),
        )

    def _fetch_exact_url(self, url: str, tokens: tuple[str, ...]) -> DocReference:
        normalized_url = _normalize_url(url)
        payload = self._fetch(normalized_url)
        html = payload.decode("utf-8", errors="replace") if isinstance(payload, bytes) else payload
        title, text = _html_to_text(html)
        compact_text = _compact_text(text)
        return DocReference(
            title=title or normalized_url,
            url=normalized_url,
            source=_source_from_url(normalized_url),
            description=None,
            snippet=_snippet(compact_text, tokens),
            provenance_url=normalized_url,
            matched_signals=_matched_signals(
                fields={
                    "title": title,
                    "body": compact_text,
                },
                tokens=tokens,
            ),
            fetched=True,
            content_chars=len(compact_text),
        )


def fetch_docs(
    query: str,
    *,
    backend: OfficialDocsBackend,
    url: str | None = None,
    max_results: int = 5,
) -> DocsFetchResult:
    return backend.fetch_docs(query, url=url, max_results=max_results)


def _rank_catalog(
    catalog: Sequence[DocReference],
    tokens: tuple[str, ...],
    max_results: int,
) -> Iterable[DocReference]:
    scored: list[tuple[int, str, DocReference]] = []
    for document in catalog:
        score = _catalog_score(document, tokens)
        if tokens and score == 0:
            continue
        fields = {
            "title": document.title,
            "description": document.description or "",
            "topics": " ".join(document.topics),
            "url": document.url,
        }
        scored.append(
            (
                score,
                document.title.casefold(),
                replace(
                    document,
                    snippet=document.description,
                    provenance_url=document.url,
                    matched_signals=_matched_signals(fields=fields, tokens=tokens),
                ),
            )
        )

    scored.sort(key=lambda item: (-item[0], item[1]))
    return (item[2] for item in scored[:max_results])


def _catalog_score(document: DocReference, tokens: tuple[str, ...]) -> int:
    if not tokens:
        return 0

    fields = {
        "title": (document.title, 4),
        "topics": (" ".join(document.topics), 3),
        "description": (document.description or "", 2),
        "url": (document.url, 1),
        "source": (document.source, 1),
    }
    score = 0
    for value, weight in fields.values():
        lowered = value.casefold()
        score += sum(weight for token in tokens if token in lowered)
    return score


def _catalog_for_backend(name: str) -> tuple[DocReference, ...]:
    if name == "huggingface":
        return tuple(document for document in _DOCS_CATALOG if document.source == "huggingface")
    return _DOCS_CATALOG


def _matched_signals(*, fields: dict[str, str], tokens: tuple[str, ...]) -> tuple[str, ...]:
    signals: list[str] = []
    for field_name, value in fields.items():
        lowered = value.casefold()
        matches = sorted({token for token in tokens if token in lowered})
        if matches:
            signals.append(f"matched {', '.join(matches)} in {field_name}")
    return tuple(signals)


def _normalize_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("url must be an absolute http(s) documentation URL")
    return url


def _fetch_url(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "labmate/0.1.0 (https://github.com/Yacinekouhli/labmate)"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return response.read()


class _ReadableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self._title_parts: list[str] = []
        self._body_parts: list[str] = []
        self._tag_stack: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        self._tag_stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self.title = _compact_text(" ".join(self._title_parts))
        if tag in self._tag_stack:
            self._tag_stack.reverse()
            self._tag_stack.remove(tag)
            self._tag_stack.reverse()

    def handle_data(self, data: str) -> None:
        if not data.strip():
            return
        if self._inside_ignored_tag():
            return
        if self._tag_stack and self._tag_stack[-1] == "title":
            self._title_parts.append(data)
        else:
            self._body_parts.append(data)

    def text(self) -> str:
        return _compact_text(" ".join(self._body_parts))

    def _inside_ignored_tag(self) -> bool:
        return any(tag in {"script", "style", "noscript", "svg"} for tag in self._tag_stack)


def _html_to_text(html: str) -> tuple[str, str]:
    parser = _ReadableHTMLParser()
    parser.feed(html)
    return parser.title, parser.text()


def _snippet(text: str, tokens: tuple[str, ...], *, window: int = 360) -> str | None:
    if not text:
        return None
    lowered = text.casefold()
    matches = [lowered.find(token) for token in tokens if lowered.find(token) >= 0]
    if not matches:
        return text[:window]
    start = max(0, min(matches) - window // 3)
    end = min(len(text), start + window)
    return text[start:end]


def _source_from_url(url: str) -> str:
    host = urllib.parse.urlparse(url).netloc.casefold()
    if "huggingface.co" in host:
        return "huggingface"
    if "pytorch.org" in host:
        return "pytorch"
    if "scikit-learn.org" in host:
        return "scikit-learn"
    if "pandas.pydata.org" in host:
        return "pandas"
    if "xgboost" in host:
        return "xgboost"
    if "lightgbm" in host:
        return "lightgbm"
    return host


def _tokenize(query: str) -> tuple[str, ...]:
    return tuple(
        token for token in re.findall(r"[a-zA-Z0-9_]+", query.casefold()) if len(token) > 2
    )


def _compact_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _isoformat(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


_DOCS_CATALOG = (
    DocReference(
        title="scikit-learn Pipeline",
        url="https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html",
        source="scikit-learn",
        description="Composable estimator pipelines for preprocessing and model training.",
        topics=("pipeline", "preprocessing", "estimator", "tabular"),
    ),
    DocReference(
        title="scikit-learn ColumnTransformer",
        url=(
            "https://scikit-learn.org/stable/modules/generated/"
            "sklearn.compose.ColumnTransformer.html"
        ),
        source="scikit-learn",
        description="Apply different preprocessing transformers to tabular column subsets.",
        topics=("column transformer", "preprocessing", "tabular", "categorical"),
    ),
    DocReference(
        title="pandas.read_csv",
        url="https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html",
        source="pandas",
        description="Read CSV files with delimiter, dtype, missing-value, and parsing controls.",
        topics=("csv", "dataframe", "missing values", "dtype", "tabular"),
    ),
    DocReference(
        title="PyTorch torch.compile",
        url="https://docs.pytorch.org/docs/stable/generated/torch.compile.html",
        source="pytorch",
        description="Compile PyTorch functions and modules for optimized execution.",
        topics=("torch.compile", "pytorch", "performance", "training"),
    ),
    DocReference(
        title="PyTorch DataLoader",
        url="https://docs.pytorch.org/docs/stable/data.html",
        source="pytorch",
        description="Load datasets with batching, shuffling, workers, and pinned memory.",
        topics=("dataloader", "dataset", "batching", "training"),
    ),
    DocReference(
        title="XGBoost Parameters",
        url="https://xgboost.readthedocs.io/en/stable/parameter.html",
        source="xgboost",
        description="Tree booster, objective, regularization, GPU, and learning-task parameters.",
        topics=("xgboost", "gradient boosting", "tabular", "parameters", "gpu"),
    ),
    DocReference(
        title="LightGBM Parameters",
        url="https://lightgbm.readthedocs.io/en/stable/Parameters.html",
        source="lightgbm",
        description="LightGBM core, learning-control, objective, metric, and GPU parameters.",
        topics=("lightgbm", "gradient boosting", "tabular", "parameters", "gpu"),
    ),
    DocReference(
        title="Hugging Face Transformers Trainer",
        url="https://huggingface.co/docs/transformers/main_classes/trainer",
        source="huggingface",
        description="Trainer API for PyTorch model training, evaluation, checkpoints, and logging.",
        topics=("trainer", "transformers", "huggingface", "training", "evaluation"),
    ),
    DocReference(
        title="Hugging Face Datasets Loading",
        url="https://huggingface.co/docs/datasets/loading",
        source="huggingface",
        description="Load datasets from the Hub, local files, CSV, JSON, Parquet, and scripts.",
        topics=("datasets", "huggingface", "csv", "parquet", "loading"),
    ),
)
