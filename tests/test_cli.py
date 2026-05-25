import json

from labmate.cli import main


def _json_output(capsys):
    return json.loads(capsys.readouterr().out)


def test_tools_command_uses_contract_shape(capsys) -> None:
    exit_code = main(["tools"])
    payload = _json_output(capsys)

    assert exit_code == 0
    assert payload["schema_version"] == "labmate.tool.v1"
    assert payload["ok"] is True
    assert payload["tool"] == "tools"
    assert payload["result"]["tools"][0]["input_schema"]["type"] == "object"


def test_dataset_inspect_command_calls_registered_handler(tmp_path, capsys) -> None:
    dataset = tmp_path / "train.csv"
    dataset.write_text("id,target\n1,0\n2,1\n", encoding="utf-8")

    exit_code = main(["dataset-inspect", str(dataset), "--sample-size", "1"])
    payload = _json_output(capsys)

    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["tool"] == "dataset_inspect"
    assert payload["result"]["file_name"] == "train.csv"
    assert payload["result"]["sample_rows"] == [{"id": "1", "target": "0"}]


def test_literature_search_unimplemented_backend_returns_contract_failure(capsys) -> None:
    exit_code = main(["literature-search", "tabular modeling", "--backend", "openalex"])
    payload = _json_output(capsys)

    assert exit_code != 0
    assert payload["ok"] is False
    assert payload["tool"] == "literature_search"
    assert payload["error"]["code"] == "backend_not_implemented"
    assert payload["error"]["details"] == {"backend": "openalex"}


def test_docs_fetch_command_returns_catalog_results(capsys) -> None:
    exit_code = main(["docs-fetch", "xgboost gpu parameters", "--backend", "local"])
    payload = _json_output(capsys)

    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["tool"] == "docs_fetch"
    assert payload["result"]["documents"][0]["title"] == "XGBoost Parameters"
    assert payload["result"]["documents"][0]["provenance_url"].startswith("https://")


def test_benchmark_lookup_command_returns_local_catalog_results(capsys) -> None:
    exit_code = main(["benchmark-lookup", "tabular classification"])
    payload = _json_output(capsys)

    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["tool"] == "benchmark_lookup"
    assert payload["result"]["benchmarks"]
    assert payload["result"]["benchmarks"][0]["provenance_url"].startswith("https://")


def test_benchmark_lookup_command_reports_unimplemented_remote_backend(capsys) -> None:
    exit_code = main(["benchmark-lookup", "tabular", "--backend", "papers_with_code"])
    payload = _json_output(capsys)

    assert exit_code != 0
    assert payload["ok"] is False
    assert payload["tool"] == "benchmark_lookup"
    assert payload["error"]["code"] == "backend_not_implemented"
    assert payload["error"]["details"] == {"backend": "papers_with_code"}


def test_github_find_examples_command_validates_repository_filter(capsys) -> None:
    exit_code = main(["github-find-examples", "xgboost", "--repository", "invalid"])
    payload = _json_output(capsys)

    assert exit_code != 0
    assert payload["ok"] is False
    assert payload["tool"] == "github_find_examples"
    assert payload["error"]["code"] == "invalid_arguments"


def test_init_command_can_dry_run_from_packaged_resources(tmp_path, capsys) -> None:
    exit_code = main(["init", "codex", str(tmp_path)])
    payload = _json_output(capsys)

    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["tool"] == "init"
    assert payload["result"]["files"][0]["relative_destination"] == "AGENTS.md"
    assert not (tmp_path / "AGENTS.md").exists()
