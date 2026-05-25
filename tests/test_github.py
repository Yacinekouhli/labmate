from datetime import UTC, datetime

import pytest

from labmate.tools.github import GitHubRepositorySearchBackend, find_github_examples


def fixed_now() -> datetime:
    return datetime(2026, 5, 25, 12, 0, tzinfo=UTC)


def test_github_repository_search_parses_ranked_examples() -> None:
    requested_urls: list[str] = []

    def fake_fetch(url: str) -> str:
        requested_urls.append(url)
        return """
{
  "total_count": 2,
  "items": [
    {
      "full_name": "dmlc/xgboost",
      "html_url": "https://github.com/dmlc/xgboost",
      "description": "Scalable, portable and distributed gradient boosting library.",
      "stargazers_count": 28000,
      "language": "C++",
      "topics": ["machine-learning", "gradient-boosting", "xgboost"],
      "default_branch": "master",
      "updated_at": "2026-05-01T00:00:00Z",
      "pushed_at": "2026-05-02T00:00:00Z",
      "license": {"spdx_id": "Apache-2.0"}
    }
  ]
}
"""

    backend = GitHubRepositorySearchBackend(fetch=fake_fetch, now=fixed_now)

    result = find_github_examples("xgboost tabular gradient boosting", backend=backend)

    assert requested_urls[0].startswith("https://api.github.com/search/repositories?")
    assert "q=xgboost+tabular+gradient+boosting" in requested_urls[0]
    assert result.backend == "github"
    assert result.retrieved_at == "2026-05-25T12:00:00Z"
    assert result.repository_filter is None
    assert result.examples[0].repository == "dmlc/xgboost"
    assert result.examples[0].stars == 28000
    assert result.examples[0].license == "Apache-2.0"
    assert result.examples[0].provenance_url == requested_urls[0]
    assert any("matched" in signal for signal in result.examples[0].matched_signals)
    assert "rate limited" in result.warnings[0]


def test_github_repository_filter_fetches_exact_repo_metadata() -> None:
    requested_urls: list[str] = []

    def fake_fetch(url: str) -> str:
        requested_urls.append(url)
        return """
{
  "full_name": "pytorch/examples",
  "html_url": "https://github.com/pytorch/examples",
  "description": "A set of examples around PyTorch in Vision, Text, Reinforcement Learning, etc.",
  "stargazers_count": 24000,
  "language": "Python",
  "topics": ["pytorch", "examples"],
  "default_branch": "main",
  "updated_at": "2026-05-01T00:00:00Z",
  "pushed_at": "2026-05-02T00:00:00Z",
  "license": {"spdx_id": "BSD-3-Clause"}
}
"""

    backend = GitHubRepositorySearchBackend(fetch=fake_fetch, now=fixed_now)

    result = find_github_examples(
        "pytorch training examples",
        backend=backend,
        repository="pytorch/examples",
    )

    assert requested_urls == ["https://api.github.com/repos/pytorch/examples"]
    assert result.repository_filter == "pytorch/examples"
    assert result.examples[0].repository == "pytorch/examples"
    assert result.examples[0].language == "Python"
    assert "metadata only" in result.warnings[0]


def test_github_repository_filter_rejects_invalid_format() -> None:
    backend = GitHubRepositorySearchBackend(now=fixed_now)

    with pytest.raises(ValueError, match="owner/repo"):
        find_github_examples("xgboost", backend=backend, repository="not-a-repo")
