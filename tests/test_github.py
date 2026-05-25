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
    assert result.file_examples == ()
    assert "rate limited" in result.warnings[0]


def test_github_repository_filter_fetches_exact_repo_metadata() -> None:
    requested_urls: list[str] = []

    def fake_fetch(url: str) -> str:
        requested_urls.append(url)
        if url == "https://api.github.com/repos/pytorch/examples":
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
        if url == "https://api.github.com/repos/pytorch/examples/git/trees/main?recursive=1":
            return """
{
  "truncated": false,
  "tree": [
    {"path": "mnist/main.py", "type": "blob", "size": 1400},
    {"path": "README.md", "type": "blob", "size": 3000},
    {"path": "data/model.bin", "type": "blob", "size": 1000}
  ]
}
"""
        if url == "https://raw.githubusercontent.com/pytorch/examples/main/mnist/main.py":
            return """
import torch

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        output = model(data)
"""
        raise AssertionError(f"unexpected URL: {url}")

    backend = GitHubRepositorySearchBackend(fetch=fake_fetch, now=fixed_now)

    result = find_github_examples(
        "mnist train pytorch examples",
        backend=backend,
        repository="pytorch/examples",
    )

    assert requested_urls == [
        "https://api.github.com/repos/pytorch/examples",
        "https://api.github.com/repos/pytorch/examples/git/trees/main?recursive=1",
        "https://raw.githubusercontent.com/pytorch/examples/main/mnist/main.py",
    ]
    assert result.repository_filter == "pytorch/examples"
    assert result.examples[0].repository == "pytorch/examples"
    assert result.examples[0].language == "Python"
    assert result.file_examples[0].repository == "pytorch/examples"
    assert result.file_examples[0].path == "mnist/main.py"
    assert (
        result.file_examples[0].url == "https://github.com/pytorch/examples/blob/main/mnist/main.py"
    )
    assert result.file_examples[0].raw_url == (
        "https://raw.githubusercontent.com/pytorch/examples/main/mnist/main.py"
    )
    assert result.file_examples[0].line_start == 2
    assert result.file_examples[0].line_end == 6
    assert "def train" in result.file_examples[0].snippet
    assert any("matched train" in signal for signal in result.file_examples[0].matched_signals)
    assert "public files" in result.warnings[0]


def test_github_repository_filter_keeps_metadata_when_tree_fetch_fails() -> None:
    requested_urls: list[str] = []

    def fake_fetch(url: str) -> str:
        requested_urls.append(url)
        if url == "https://api.github.com/repos/pytorch/examples":
            return """
{
  "full_name": "pytorch/examples",
  "html_url": "https://github.com/pytorch/examples",
  "description": "Examples.",
  "stargazers_count": 24000,
  "language": "Python",
  "topics": ["pytorch", "examples"],
  "default_branch": "main",
  "updated_at": "2026-05-01T00:00:00Z",
  "pushed_at": "2026-05-02T00:00:00Z",
  "license": {"spdx_id": "BSD-3-Clause"}
}
"""
        raise OSError("tree unavailable")

    backend = GitHubRepositorySearchBackend(fetch=fake_fetch, now=fixed_now)

    result = find_github_examples(
        "mnist train pytorch examples",
        backend=backend,
        repository="pytorch/examples",
    )

    assert requested_urls == [
        "https://api.github.com/repos/pytorch/examples",
        "https://api.github.com/repos/pytorch/examples/git/trees/main?recursive=1",
    ]
    assert result.examples[0].repository == "pytorch/examples"
    assert result.file_examples == ()
    assert any("Could not fetch repository tree" in warning for warning in result.warnings)


def test_github_repository_filter_rejects_invalid_format() -> None:
    backend = GitHubRepositorySearchBackend(now=fixed_now)

    with pytest.raises(ValueError, match="owner/repo"):
        find_github_examples("xgboost", backend=backend, repository="not-a-repo")
