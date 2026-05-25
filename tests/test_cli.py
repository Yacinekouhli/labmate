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
    assert payload["result"]["tools"][0]["usage_examples"]
    assert {"cli", "mcp"} <= {
        example["surface"] for example in payload["result"]["tools"][0]["usage_examples"]
    }


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


def test_project_scan_command_returns_project_contract(tmp_path, capsys) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "train.csv").write_text("id,target\n1,0\n", encoding="utf-8")
    (data / "test.csv").write_text("id\n2\n", encoding="utf-8")

    exit_code = main(["project-scan", str(tmp_path)])
    payload = _json_output(capsys)

    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["tool"] == "project_scan"
    assert payload["result"]["dataset_candidates"][0]["path"] == "data"
    assert payload["result"]["experiment_tracking"]["recommended_ledger_path"] == "results.tsv"
    assert payload["result"]["recommended_next_commands"][1] == f"labmate research-brief {data}"


def test_experiment_summary_command_returns_ledger_contract(tmp_path, capsys) -> None:
    ledger = tmp_path / "results.tsv"
    ledger.write_text(
        (
            "timestamp_utc\tcommit\texperiment\tmodel_family\tfeatures\tvalidation_strategy\t"
            "metric\tscore\tscore_direction\tstatus\tartifacts\tnotes\n"
            "2026-05-25T10:00:00Z\tabc123\tdummy\tdummy\tbase\tk_fold\t"
            "rmse\t1.0\tminimize\tkeep\tmodel.pkl\tbaseline\n"
        ),
        encoding="utf-8",
    )

    exit_code = main(["experiment-summary", str(ledger)])
    payload = _json_output(capsys)

    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["tool"] == "experiment_summary"
    assert payload["result"]["ledger"]["path"] == "results.tsv"
    assert payload["result"]["best_run"]["experiment"] == "dummy"


def test_research_brief_command_returns_workflow_contract(tmp_path, capsys) -> None:
    (tmp_path / "train.csv").write_text("id,feature,target\n1,10,0\n2,11,1\n", encoding="utf-8")
    (tmp_path / "test.csv").write_text("id,feature\n3,12\n", encoding="utf-8")

    exit_code = main(["research-brief", str(tmp_path), "--max-benchmarks", "1"])
    payload = _json_output(capsys)

    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["tool"] == "research_brief"
    assert payload["result"]["inferred_task"]["task_type"] == "tabular classification"
    assert payload["result"]["benchmark_context"]["benchmarks"]
    assert payload["result"]["research_plan"][0]["tool"] == "dataset_inspect"
    assert payload["result"]["research_plan"][0]["arguments"] == {
        "path": str(tmp_path),
        "sample_size": 5,
    }
    assert payload["result"]["experiment_tracking_plan"]["ledger_path"] == "results.tsv"
    assert payload["result"]["experiment_tracking_plan"]["columns"][:4] == [
        "timestamp_utc",
        "commit",
        "experiment",
        "model_family",
    ]
    assert any(
        "dataset-inspect" in command for command in payload["result"]["recommended_next_commands"]
    )


def test_literature_search_unimplemented_backend_returns_contract_failure(capsys) -> None:
    exit_code = main(["literature-search", "tabular modeling", "--backend", "openalex"])
    payload = _json_output(capsys)

    assert exit_code != 0
    assert payload["ok"] is False
    assert payload["tool"] == "literature_search"
    assert payload["error"]["code"] == "backend_not_implemented"
    assert payload["error"]["details"] == {"backend": "openalex"}


def test_citation_graph_command_returns_local_corpus_results(capsys) -> None:
    exit_code = main(["citation-graph", "arxiv:1603.02754", "--max-results", "2"])
    payload = _json_output(capsys)

    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["tool"] == "citation_graph"
    assert payload["result"]["root"]["title"] == "XGBoost: A Scalable Tree Boosting System"
    assert payload["result"]["citations"]


def test_citation_graph_command_reports_unimplemented_remote_backend(capsys) -> None:
    exit_code = main(["citation-graph", "arxiv:1603.02754", "--backend", "semantic_scholar"])
    payload = _json_output(capsys)

    assert exit_code != 0
    assert payload["ok"] is False
    assert payload["tool"] == "citation_graph"
    assert payload["error"]["code"] == "backend_not_implemented"
    assert payload["error"]["details"]["backend"] == "semantic_scholar"


def test_citation_graph_command_reports_unknown_local_paper(capsys) -> None:
    exit_code = main(["citation-graph", "arxiv:0000.00000"])
    payload = _json_output(capsys)

    assert exit_code != 0
    assert payload["ok"] is False
    assert payload["tool"] == "citation_graph"
    assert payload["error"]["code"] == "paper_not_found"


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


def test_generic_init_command_can_dry_run_from_packaged_resources(tmp_path, capsys) -> None:
    exit_code = main(["init", "cursor", str(tmp_path)])
    payload = _json_output(capsys)

    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["tool"] == "init"
    assert payload["result"]["harness"] == "generic"
    assert [file["relative_destination"] for file in payload["result"]["files"]] == [
        "AGENTS.md",
        "program.md",
        ".mcp.json",
    ]
