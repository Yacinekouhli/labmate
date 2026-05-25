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
    assert all(
        warning["message"] != "No obvious target column detected from column names."
        for warning in result["dataset_summary"]["warnings"]
    )
    assert result["recommended_next_commands"][0].startswith("labmate dataset-inspect ")
    assert any("literature-search" in command for command in result["recommended_next_commands"])
    assert any("competition metric" in item for item in result["implementation_checklist"])
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
