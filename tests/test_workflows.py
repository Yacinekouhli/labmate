import zipfile

from labmate.tools.workflows import build_research_brief


def test_research_brief_combines_dataset_and_benchmark_context(tmp_path) -> None:
    (tmp_path / "evaluation.md").write_text(
        "Submissions are scored with ROC AUC.",
        encoding="utf-8",
    )
    (tmp_path / "train.csv").write_text(
        "id,feature,target\n1,10,0\n2,11,1\n",
        encoding="utf-8",
    )
    (tmp_path / "test.csv").write_text(
        "id,feature\n3,12\n4,13\n",
        encoding="utf-8",
    )
    (tmp_path / "sample_submission.csv").write_text(
        "id,target\n3,0\n4,0\n",
        encoding="utf-8",
    )

    result = build_research_brief(tmp_path, max_benchmarks=1)

    assert result["kind"] == "ml_research_brief"
    assert result["inferred_task"]["task_type"] == "tabular classification"
    assert result["benchmark_query"] == "tabular classification auc kaggle"
    assert result["dataset_summary"]["relations"]["train_file"] == "train.csv"
    assert result["dataset_summary"]["relations"]["train_test_schema_alignment"] == {
        "common_columns": ["id", "feature"],
        "common_feature_columns": ["feature"],
        "id_columns": ["id"],
        "train_only_columns": ["target"],
        "test_only_columns": [],
        "target_columns_absent_from_test": ["target"],
        "target_columns_present_in_test": [],
    }
    assert result["benchmark_context"]["benchmarks"]
    assert result["evidence"]["target_columns"] == ["target"]
    assert result["evidence"]["target_distribution"] == {
        "column": "target",
        "profiled_row_count": 2,
        "missing_rate": 0.0,
        "unique_values_profiled": 2,
        "unique_values_truncated": False,
        "top_values": [
            {"value": "0", "count": 1, "rate": 0.5},
            {"value": "1", "count": 1, "rate": 0.5},
        ],
    }
    assert result["evidence"]["validation_columns"] == []
    assert result["evidence"]["submission_format"] == {
        "sample_submission_file": "sample_submission.csv",
        "id_columns": ["id"],
        "output_columns": ["target"],
        "sample_submission_row_count": 2,
        "test_row_count": 2,
        "row_counts_match_test": True,
    }
    assert result["evidence"]["context_files"] == [
        {"file_name": "evaluation.md", "kind": "competition_rules"}
    ]
    assert result["evidence"]["metric_hints"] == [
        {
            "metric": "roc_auc",
            "source_file": "evaluation.md",
            "matched_text": "ROC AUC",
        }
    ]
    assert result["modeling_plan"]["readiness"] == "ready_for_baseline"
    assert result["modeling_plan"]["target_columns"] == ["target"]
    assert (
        result["modeling_plan"]["target_distribution"]
        == (result["evidence"]["target_distribution"])
    )
    assert result["modeling_plan"]["id_columns"] == ["id"]
    assert result["modeling_plan"]["validation_columns"] == []
    assert result["modeling_plan"]["submission_format"] == result["evidence"]["submission_format"]
    assert result["modeling_plan"]["feature_columns"] == ["feature"]
    assert result["modeling_plan"]["suggested_metric"] == "roc_auc"
    assert result["modeling_plan"]["validation_strategy"]["name"] == "stratified_k_fold"
    assert [
        experiment["name"] for experiment in result["modeling_plan"]["baseline_experiments"]
    ] == [
        "dummy_baseline",
        "linear_baseline",
        "tree_boosting_baseline",
    ]
    assert all(
        warning["message"] != "No obvious target column detected from column names."
        for warning in result["dataset_summary"]["warnings"]
    )
    assert [action["tool"] for action in result["research_plan"]] == [
        "dataset_inspect",
        "benchmark_lookup",
        "literature_search",
        "citation_graph",
        "docs_fetch",
        "github_find_examples",
    ]
    assert result["research_plan"][0] == {
        "priority": 1,
        "tool": "dataset_inspect",
        "command": f"labmate dataset-inspect {tmp_path} --sample-size 5",
        "arguments": {"path": str(tmp_path), "sample_size": 5},
        "purpose": "Verify schema, split alignment, target hints, and leakage warnings.",
        "evidence_to_extract": [
            "target columns",
            "train/test feature alignment",
            "sample submission format",
            "dataset warnings",
        ],
    }
    assert result["research_plan"][1]["arguments"] == {
        "query": "tabular classification auc kaggle",
        "max_results": 1,
    }
    assert result["recommended_next_commands"] == [
        action["command"] for action in result["research_plan"]
    ]
    assert result["recommended_next_commands"][0].startswith("labmate dataset-inspect ")
    assert any("literature-search" in command for command in result["recommended_next_commands"])
    assert any(
        "sample submission columns target" in item for item in result["implementation_checklist"]
    )
    assert any("local context files" in item for item in result["implementation_checklist"])
    assert "planning aid" in result["warnings"][0]


def test_research_brief_respects_task_hint_and_benchmark_query(tmp_path) -> None:
    (tmp_path / "train.csv").write_text(
        "id,price\n1,100000\n2,125000\n",
        encoding="utf-8",
    )

    result = build_research_brief(
        tmp_path / "train.csv",
        task_hint="house prices regression rmse",
        max_benchmarks=1,
    )

    assert result["inferred_task"]["task_type"] == "house prices regression rmse"
    assert result["inferred_task"]["confidence"] == "user_supplied"
    assert result["benchmark_query"] == "house prices regression rmse"
    assert result["benchmark_context"]["benchmarks"][0]["name"] == (
        "House Prices - Advanced Regression Techniques"
    )
    assert result["modeling_plan"]["readiness"] == "needs_target_confirmation"
    assert result["modeling_plan"]["validation_strategy"]["name"] == "not_ready"
    assert result["modeling_plan"]["baseline_experiments"] == [
        {
            "name": "schema_confirmation",
            "model_family": "none",
            "purpose": "confirm target, ID, feature, metric, and submission columns",
            "expected_output": "documented schema decision before modeling",
        }
    ]


def test_research_brief_uses_provided_fold_column_for_validation(tmp_path) -> None:
    (tmp_path / "train.csv").write_text(
        "id,feature,fold,target\n1,10,0,0\n2,11,1,1\n3,12,0,0\n",
        encoding="utf-8",
    )

    result = build_research_brief(tmp_path / "train.csv", max_benchmarks=1)

    assert result["evidence"]["validation_columns"] == ["fold"]
    assert result["modeling_plan"]["validation_columns"] == ["fold"]
    assert result["modeling_plan"]["feature_columns"] == ["feature"]
    assert result["modeling_plan"]["validation_strategy"] == {
        "name": "provided_split_column",
        "columns": ["fold"],
        "reason": (
            "dataset already includes validation/split columns; inspect and reuse them "
            "before creating new folds"
        ),
        "metric": "roc_auc",
    }
    assert any("fold" in item for item in result["implementation_checklist"])


def test_research_brief_warns_about_imbalanced_classification_target(tmp_path) -> None:
    rows = ["id,feature,target"]
    rows.extend(f"{index},{index * 10},0" for index in range(1, 10))
    rows.append("10,100,1")
    (tmp_path / "train.csv").write_text("\n".join(rows), encoding="utf-8")

    result = build_research_brief(tmp_path / "train.csv", max_benchmarks=1)

    assert result["modeling_plan"]["target_distribution"]["top_values"][0] == {
        "value": "0",
        "count": 9,
        "rate": 0.9,
    }
    assert any("imbalanced" in warning for warning in result["warnings"])


def test_research_brief_supports_zip_archive_dataset(tmp_path) -> None:
    archive_path = tmp_path / "competition.zip"
    with zipfile.ZipFile(archive_path, mode="w") as archive:
        archive.writestr("evaluation.md", "Submissions are scored with ROC AUC.")
        archive.writestr("train.csv", "id,feature,target\n1,10,0\n2,11,1\n")
        archive.writestr("test.csv", "id,feature\n3,12\n4,13\n")
        archive.writestr("sample_submission.csv", "id,target\n3,0\n4,0\n")

    result = build_research_brief(archive_path, max_benchmarks=1)

    assert result["dataset_summary"]["kind"] == "local_dataset_archive"
    assert result["dataset_summary"]["relations"]["train_file"] == "train.csv"
    assert result["dataset_summary"]["context_files"] == [
        {
            "file_name": "evaluation.md",
            "kind": "competition_rules",
            "size_bytes": 36,
            "snippet": "Submissions are scored with ROC AUC.",
            "json_keys": [],
        }
    ]
    assert result["evidence"]["target_columns"] == ["target"]
    assert result["evidence"]["submission_format"]["sample_submission_file"] == (
        "sample_submission.csv"
    )
    assert result["modeling_plan"]["readiness"] == "ready_for_baseline"
    assert result["recommended_next_commands"][0] == (
        f"labmate dataset-inspect {archive_path} --sample-size 5"
    )
