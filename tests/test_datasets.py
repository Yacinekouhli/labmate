import gzip

from labmate.tools.datasets import inspect_local_dataset, inspect_tabular_file


def test_inspect_tabular_file_profiles_csv_for_kaggle_training_data(tmp_path) -> None:
    csv_path = tmp_path / "train.csv"
    csv_path.write_text(
        "\n".join(
            [
                "id,feature,target,fold,future_score",
                "1,10,0,0,0.12",
                "2,,1,0,0.80",
                "3,7,0,1,0.33",
            ]
        ),
        encoding="utf-8",
    )

    result = inspect_tabular_file(csv_path, sample_size=2)

    assert result["kind"] == "tabular_file"
    assert result["delimiter"] == ","
    assert result["row_count"] == 3
    assert result["row_count_status"] == "exact"
    assert result["sample_rows"] == [
        {"id": "1", "feature": "10", "target": "0", "fold": "0", "future_score": "0.12"},
        {"id": "2", "feature": "", "target": "1", "fold": "0", "future_score": "0.80"},
    ]

    columns = {column["name"]: column for column in result["columns"]}
    assert columns["id"]["role_hints"] == ["id", "unique_per_row"]
    assert columns["feature"]["missing_count"] == 1
    assert columns["feature"]["missing_rate"] == 0.333333
    assert columns["target"]["inferred_type"] == "integer"
    assert columns["target"]["top_values"] == [
        {"value": "0", "count": 2, "rate": 0.666667},
        {"value": "1", "count": 1, "rate": 0.333333},
    ]

    assert result["target_column_hints"][0]["column"] == "target"
    assert {hint["risk"] for hint in result["leakage_risk_hints"]} >= {
        "split_indicator",
        "future_information",
    }


def test_inspect_tabular_file_supports_tsv(tmp_path) -> None:
    tsv_path = tmp_path / "labels.tsv"
    tsv_path.write_text("sample_id\ttext\tlabel\n1\ta\tspam\n2\tb\tham\n", encoding="utf-8")

    result = inspect_tabular_file(tsv_path)

    assert result["delimiter"] == "\t"
    assert result["row_count"] == 2
    assert [column["name"] for column in result["columns"]] == ["sample_id", "text", "label"]
    assert result["target_column_hints"][0]["column"] == "label"


def test_inspect_tabular_file_supports_gzipped_csv(tmp_path) -> None:
    csv_path = tmp_path / "train.csv.gz"
    with gzip.open(csv_path, mode="wt", encoding="utf-8") as handle:
        handle.write("id,feature,target\n1,10,0\n2,11,1\n")

    result = inspect_tabular_file(csv_path)

    assert result["file_name"] == "train.csv.gz"
    assert result["format"] == "csv"
    assert result["compression"] == "gzip"
    assert result["delimiter"] == ","
    assert result["row_count"] == 2
    assert result["target_column_hints"][0]["column"] == "target"


def test_inspect_local_dataset_directory_adds_split_and_submission_hints(tmp_path) -> None:
    (tmp_path / "evaluation.md").write_text(
        "Submissions are evaluated using ROC AUC.",
        encoding="utf-8",
    )
    (tmp_path / "kaggle.json").write_text(
        '{"title": "Example Competition", "id": "example"}',
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

    result = inspect_local_dataset(tmp_path)

    assert result["kind"] == "local_dataset_directory"
    assert [file_info["file_name"] for file_info in result["files"]] == [
        "sample_submission.csv",
        "test.csv",
        "train.csv",
    ]
    assert result["relations"]["train_file"] == "train.csv"
    assert result["relations"]["test_file"] == "test.csv"
    assert result["relations"]["sample_submission_file"] == "sample_submission.csv"
    assert result["relations"]["sample_submission_alignment"] == {
        "common_id_columns": ["id"],
        "row_counts_match": True,
        "submission_output_columns": ["target"],
    }
    assert result["relations"]["train_test_schema_alignment"] == {
        "common_columns": ["id", "feature"],
        "common_feature_columns": ["feature"],
        "id_columns": ["id"],
        "train_only_columns": ["target"],
        "test_only_columns": [],
        "target_columns_absent_from_test": ["target"],
        "target_columns_present_in_test": [],
    }
    assert result["relations"]["likely_target_columns"] == ["target"]
    assert result["context_files"] == [
        {
            "file_name": "evaluation.md",
            "path": str(tmp_path / "evaluation.md"),
            "kind": "competition_rules",
            "size_bytes": 40,
            "snippet": "Submissions are evaluated using ROC AUC.",
        },
        {
            "file_name": "kaggle.json",
            "path": str(tmp_path / "kaggle.json"),
            "kind": "metadata",
            "size_bytes": 49,
            "snippet": '{"title": "Example Competition", "id": "example"}',
            "json_keys": ["id", "title"],
        },
    ]
    assert result["warnings"] == []


def test_inspect_local_dataset_directory_supports_gzipped_kaggle_splits(tmp_path) -> None:
    with gzip.open(tmp_path / "train.csv.gz", mode="wt", encoding="utf-8") as handle:
        handle.write("id,feature,target\n1,10,0\n2,11,1\n")
    with gzip.open(tmp_path / "test.csv.gz", mode="wt", encoding="utf-8") as handle:
        handle.write("id,feature\n3,12\n4,13\n")
    with gzip.open(tmp_path / "sample_submission.csv.gz", mode="wt", encoding="utf-8") as handle:
        handle.write("id,target\n3,0\n4,0\n")

    result = inspect_local_dataset(tmp_path)

    assert result["relations"]["train_file"] == "train.csv.gz"
    assert result["relations"]["test_file"] == "test.csv.gz"
    assert result["relations"]["sample_submission_file"] == "sample_submission.csv.gz"
    assert result["relations"]["likely_target_columns"] == ["target"]
    assert result["relations"]["train_test_schema_alignment"]["common_feature_columns"] == [
        "feature"
    ]
    assert result["warnings"] == []


def test_inspect_local_dataset_warns_when_target_is_present_in_test(tmp_path) -> None:
    (tmp_path / "train.csv").write_text(
        "id,feature,target\n1,10,0\n2,11,1\n",
        encoding="utf-8",
    )
    (tmp_path / "test.csv").write_text(
        "id,feature,target\n3,12,0\n4,13,0\n",
        encoding="utf-8",
    )

    result = inspect_local_dataset(tmp_path)

    assert result["relations"]["train_test_schema_alignment"]["target_columns_present_in_test"] == [
        "target"
    ]
    assert (
        "Likely target columns are present in both train and test files" in result["warnings"][-1]
    )


def test_large_profile_reports_bounded_row_count(tmp_path) -> None:
    csv_path = tmp_path / "train.csv"
    csv_path.write_text("id,target\n1,0\n2,1\n3,0\n", encoding="utf-8")

    result = inspect_tabular_file(csv_path, max_profile_rows=2)

    assert result["row_count"] is None
    assert result["row_count_status"] == "bounded"
    assert result["profiled_row_count"] == 2
    assert "bounded" in result["warnings"][-1]
