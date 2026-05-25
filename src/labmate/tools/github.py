"""Read-only GitHub repository example discovery."""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime

Fetch = Callable[[str], bytes | str]

_GITHUB_API_URL = "https://api.github.com"
MAX_FILE_SNIPPET_CHARS = 1_500
MAX_RAW_FILE_CHARS = 20_000
MAX_FILE_SIZE_BYTES = 300_000
CODE_SUFFIXES = {".py", ".r", ".R", ".jl", ".sql", ".sh"}
DOC_SUFFIXES = {".md", ".rst", ".txt"}
NOTEBOOK_SUFFIX = ".ipynb"
FILE_NAME_SIGNALS = (
    "baseline",
    "train",
    "model",
    "feature",
    "submit",
    "submission",
    "inference",
    "predict",
    "eda",
    "notebook",
)


@dataclass(frozen=True)
class GitHubRepositoryExample:
    repository: str
    url: str
    description: str | None
    stars: int
    language: str | None
    topics: tuple[str, ...]
    default_branch: str | None
    updated_at: str | None
    pushed_at: str | None
    license: str | None
    provenance_url: str
    matched_signals: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "repository": self.repository,
            "url": self.url,
            "description": self.description,
            "stars": self.stars,
            "language": self.language,
            "topics": list(self.topics),
            "default_branch": self.default_branch,
            "updated_at": self.updated_at,
            "pushed_at": self.pushed_at,
            "license": self.license,
            "provenance_url": self.provenance_url,
            "matched_signals": list(self.matched_signals),
        }


@dataclass(frozen=True)
class GitHubFileExample:
    repository: str
    path: str
    url: str
    raw_url: str
    size_bytes: int | None
    snippet: str
    line_start: int
    line_end: int
    matched_signals: tuple[str, ...]
    provenance_url: str

    def to_dict(self) -> dict[str, object]:
        return {
            "repository": self.repository,
            "path": self.path,
            "url": self.url,
            "raw_url": self.raw_url,
            "size_bytes": self.size_bytes,
            "snippet": self.snippet,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "matched_signals": list(self.matched_signals),
            "provenance_url": self.provenance_url,
        }


@dataclass(frozen=True)
class GitHubExampleSearchResult:
    query: str
    backend: str
    retrieved_at: str
    repository_filter: str | None
    examples: tuple[GitHubRepositoryExample, ...]
    file_examples: tuple[GitHubFileExample, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "backend": self.backend,
            "retrieved_at": self.retrieved_at,
            "repository_filter": self.repository_filter,
            "examples": [example.to_dict() for example in self.examples],
            "file_examples": [example.to_dict() for example in self.file_examples],
            "warnings": list(self.warnings),
        }


class GitHubRepositorySearchBackend:
    """GitHub repository search backend that works without user credentials."""

    name = "github"

    def __init__(
        self,
        *,
        fetch: Fetch | None = None,
        api_url: str = _GITHUB_API_URL,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._fetch = fetch or _fetch_url
        self._api_url = api_url.rstrip("/")
        self._now = now or _utcnow

    def find_examples(
        self,
        query: str,
        *,
        repository: str | None = None,
        max_results: int = 10,
    ) -> GitHubExampleSearchResult:
        if max_results < 1:
            raise ValueError("max_results must be positive")

        tokens = _tokenize(query)
        file_examples: tuple[GitHubFileExample, ...] = ()
        if repository:
            normalized_repository = _normalize_repository(repository)
            url = f"{self._api_url}/repos/{normalized_repository}"
            payload = _json_payload(self._fetch(url))
            repository_example = _example_from_repo_payload(payload, tokens, provenance_url=url)
            examples = (repository_example,)
            file_examples, file_warnings = self._find_repository_file_examples(
                repository_example,
                tokens=tokens,
                max_results=max_results,
            )
            warnings = (
                "Repository filter inspects public files through unauthenticated GitHub APIs; "
                "cross-repository code search still needs an authenticated backend.",
                *file_warnings,
            )
        else:
            url = self._search_url(query, max_results)
            payload = _json_payload(self._fetch(url))
            examples = tuple(
                _example_from_repo_payload(item, tokens, provenance_url=url)
                for item in payload.get("items", [])[:max_results]
                if isinstance(item, dict)
            )
            warnings = (
                "Uses unauthenticated GitHub repository search; results are rate limited.",
                "Pass repository=owner/repo to inspect public files in a known repo.",
            )

        return GitHubExampleSearchResult(
            query=query,
            backend=self.name,
            retrieved_at=_isoformat(self._now()),
            repository_filter=repository,
            examples=examples,
            file_examples=file_examples,
            warnings=warnings,
        )

    def _search_url(self, query: str, max_results: int) -> str:
        params = urllib.parse.urlencode(
            {
                "q": query,
                "sort": "stars",
                "order": "desc",
                "per_page": max_results,
            }
        )
        return f"{self._api_url}/search/repositories?{params}"

    def _find_repository_file_examples(
        self,
        repository: GitHubRepositoryExample,
        *,
        tokens: tuple[str, ...],
        max_results: int,
    ) -> tuple[tuple[GitHubFileExample, ...], tuple[str, ...]]:
        if not repository.default_branch:
            return (), ("Repository metadata did not include a default branch; skipped files.",)

        tree_url = _tree_url(self._api_url, repository.repository, repository.default_branch)
        try:
            tree_payload = _json_payload(self._fetch(tree_url))
        except (OSError, ValueError) as exc:
            return (), (f"Could not fetch repository tree for file examples: {exc}",)

        warnings: list[str] = []
        if tree_payload.get("truncated") is True:
            warnings.append(
                "Repository tree response was truncated; file examples may be incomplete."
            )

        candidates = _rank_file_candidates(tree_payload, tokens)
        file_examples: list[GitHubFileExample] = []
        skipped_fetches = 0
        for candidate in candidates:
            if len(file_examples) >= max_results:
                break

            path = str(candidate["path"])
            size_bytes = candidate["size_bytes"]
            candidate_signals = tuple(
                signal for signal in candidate["matched_signals"] if isinstance(signal, str)
            )
            raw_url = _raw_url(repository.repository, repository.default_branch, path)
            try:
                raw_payload = self._fetch(raw_url)
            except OSError:
                skipped_fetches += 1
                continue

            content = (
                raw_payload.decode("utf-8", errors="replace")
                if isinstance(raw_payload, bytes)
                else raw_payload
            )
            snippet = _snippet_for_tokens(content[:MAX_RAW_FILE_CHARS], tokens)
            file_examples.append(
                GitHubFileExample(
                    repository=repository.repository,
                    path=path,
                    url=_blob_url(repository.repository, repository.default_branch, path),
                    raw_url=raw_url,
                    size_bytes=size_bytes if isinstance(size_bytes, int) else None,
                    snippet=snippet["text"],
                    line_start=snippet["line_start"],
                    line_end=snippet["line_end"],
                    matched_signals=candidate_signals + tuple(snippet["matched_signals"]),
                    provenance_url=tree_url,
                )
            )

        if skipped_fetches:
            warnings.append("Some candidate file snippets could not be fetched.")
        if not file_examples:
            warnings.append("No file-level examples matched the query in the repository tree.")
        return tuple(file_examples), tuple(warnings)


def find_github_examples(
    query: str,
    *,
    backend: GitHubRepositorySearchBackend,
    repository: str | None = None,
    max_results: int = 10,
) -> GitHubExampleSearchResult:
    return backend.find_examples(query, repository=repository, max_results=max_results)


def _example_from_repo_payload(
    payload: Mapping[str, object],
    tokens: tuple[str, ...],
    *,
    provenance_url: str,
) -> GitHubRepositoryExample:
    full_name = _as_str(payload.get("full_name")) or _as_str(payload.get("name")) or ""
    description = _as_str(payload.get("description"))
    html_url = _as_str(payload.get("html_url")) or f"https://github.com/{full_name}"
    topics = tuple(value for value in payload.get("topics", []) if isinstance(value, str))
    license_payload = payload.get("license")
    license_name = None
    if isinstance(license_payload, dict):
        license_name = _as_str(license_payload.get("spdx_id")) or _as_str(
            license_payload.get("name")
        )

    return GitHubRepositoryExample(
        repository=full_name,
        url=html_url,
        description=description,
        stars=_as_int(payload.get("stargazers_count")),
        language=_as_str(payload.get("language")),
        topics=topics,
        default_branch=_as_str(payload.get("default_branch")),
        updated_at=_as_str(payload.get("updated_at")),
        pushed_at=_as_str(payload.get("pushed_at")),
        license=license_name,
        provenance_url=provenance_url,
        matched_signals=_matched_signals(
            fields={
                "repository": full_name,
                "description": description or "",
                "topics": " ".join(topics),
                "language": _as_str(payload.get("language")) or "",
            },
            tokens=tokens,
        ),
    )


def _json_payload(payload: bytes | str) -> dict[str, object]:
    raw = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("GitHub API response must be a JSON object")
    return data


def _rank_file_candidates(
    tree_payload: Mapping[str, object],
    tokens: tuple[str, ...],
) -> list[dict[str, object]]:
    tree = tree_payload.get("tree")
    if not isinstance(tree, list):
        return []

    candidates = []
    for entry in tree:
        if not isinstance(entry, dict) or entry.get("type") != "blob":
            continue

        path = _as_str(entry.get("path"))
        if not path or not _is_supported_example_path(path):
            continue

        size = _as_int_or_none(entry.get("size"))
        if size is not None and size > MAX_FILE_SIZE_BYTES:
            continue

        score, signals = _file_candidate_score(path, tokens)
        if score <= 0:
            continue

        candidates.append(
            {
                "path": path,
                "size_bytes": size,
                "score": score,
                "matched_signals": signals,
            }
        )

    return sorted(candidates, key=lambda item: (-int(item["score"]), str(item["path"])))[:25]


def _is_supported_example_path(path: str) -> bool:
    suffix = _path_suffix(path)
    return suffix in CODE_SUFFIXES or suffix in DOC_SUFFIXES or suffix == NOTEBOOK_SUFFIX


def _file_candidate_score(path: str, tokens: tuple[str, ...]) -> tuple[int, tuple[str, ...]]:
    normalized_path = path.casefold()
    score = 0
    signals = []
    for token in tokens:
        if token in normalized_path:
            score += 8
            signals.append(f"matched {token} in path")
    for signal in FILE_NAME_SIGNALS:
        if signal in normalized_path:
            score += 3
            signals.append(f"path contains {signal}")
    if _path_suffix(path) in CODE_SUFFIXES:
        score += 2
        signals.append("code file")
    if path.casefold().startswith(("examples/", "example/", "notebooks/", "notebook/")):
        score += 2
        signals.append("example directory")
    return score, tuple(dict.fromkeys(signals))


def _snippet_for_tokens(content: str, tokens: tuple[str, ...]) -> dict[str, object]:
    lines = content.splitlines()
    if not lines:
        return {"text": "", "line_start": 1, "line_end": 1, "matched_signals": []}

    match_index = _first_matching_line(lines, tokens)
    if match_index is None:
        match_index = _first_nonempty_line(lines)

    start = max(0, match_index - 2)
    end = min(len(lines), match_index + 3)
    snippet_lines = lines[start:end]
    text = "\n".join(snippet_lines).strip()
    if len(text) > MAX_FILE_SNIPPET_CHARS:
        text = text[:MAX_FILE_SNIPPET_CHARS].rstrip() + "..."

    return {
        "text": text,
        "line_start": start + 1,
        "line_end": end,
        "matched_signals": _matched_signals(
            fields={"snippet": text},
            tokens=tokens,
        ),
    }


def _first_matching_line(lines: list[str], tokens: tuple[str, ...]) -> int | None:
    if not tokens:
        return None
    for index, line in enumerate(lines):
        lowered = line.casefold()
        if any(token in lowered for token in tokens):
            return index
    return None


def _first_nonempty_line(lines: list[str]) -> int:
    for index, line in enumerate(lines):
        if line.strip():
            return index
    return 0


def _fetch_url(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "labmate/0.1.0 (https://github.com/Yacinekouhli/labmate)",
        },
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return response.read()


def _normalize_repository(repository: str) -> str:
    normalized = repository.strip().strip("/")
    if not re.match(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", normalized):
        raise ValueError("repository must use owner/repo format")
    return normalized


def _tree_url(api_url: str, repository: str, branch: str) -> str:
    return (
        f"{api_url}/repos/{repository}/git/trees/{urllib.parse.quote(branch, safe='')}?recursive=1"
    )


def _blob_url(repository: str, branch: str, path: str) -> str:
    return (
        f"https://github.com/{repository}/blob/"
        f"{urllib.parse.quote(branch, safe='')}/{urllib.parse.quote(path, safe='/')}"
    )


def _raw_url(repository: str, branch: str, path: str) -> str:
    return (
        f"https://raw.githubusercontent.com/{repository}/"
        f"{urllib.parse.quote(branch, safe='')}/{urllib.parse.quote(path, safe='/')}"
    )


def _matched_signals(*, fields: dict[str, str], tokens: tuple[str, ...]) -> tuple[str, ...]:
    signals: list[str] = []
    for field_name, value in fields.items():
        lowered = value.casefold()
        matches = sorted({token for token in tokens if token in lowered})
        if matches:
            signals.append(f"matched {', '.join(matches)} in {field_name}")
    return tuple(signals)


def _as_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _as_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    return value if isinstance(value, int) else 0


def _as_int_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    return value if isinstance(value, int) else None


def _path_suffix(path: str) -> str:
    dot_index = path.rfind(".")
    if dot_index < 0:
        return ""
    return path[dot_index:]


def _tokenize(query: str) -> tuple[str, ...]:
    return tuple(
        token for token in re.findall(r"[a-zA-Z0-9_]+", query.casefold()) if len(token) > 2
    )


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _isoformat(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
