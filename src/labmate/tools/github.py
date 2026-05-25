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
class GitHubExampleSearchResult:
    query: str
    backend: str
    retrieved_at: str
    repository_filter: str | None
    examples: tuple[GitHubRepositoryExample, ...]
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "backend": self.backend,
            "retrieved_at": self.retrieved_at,
            "repository_filter": self.repository_filter,
            "examples": [example.to_dict() for example in self.examples],
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
        if repository:
            normalized_repository = _normalize_repository(repository)
            url = f"{self._api_url}/repos/{normalized_repository}"
            payload = _json_payload(self._fetch(url))
            examples = (_example_from_repo_payload(payload, tokens, provenance_url=url),)
            warnings = (
                "Repository filter returns repository metadata only; "
                "authenticated code search is needed for file-level snippets.",
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
                "Use repository to inspect a known repo, "
                "or authenticated code search for snippets.",
            )

        return GitHubExampleSearchResult(
            query=query,
            backend=self.name,
            retrieved_at=_isoformat(self._now()),
            repository_filter=repository,
            examples=examples,
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
