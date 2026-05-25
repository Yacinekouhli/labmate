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


def test_inspect_local_dataset_directory_adds_split_and_submission_hints(tmp_path) -> None:
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
    assert result["relations"]["likely_target_columns"] == ["target"]
    assert result["warnings"] == []


def test_large_profile_reports_bounded_row_count(tmp_path) -> None:
    csv_path = tmp_path / "train.csv"
    csv_path.write_text("id,target\n1,0\n2,1\n3,0\n", encoding="utf-8")

    result = inspect_tabular_file(csv_path, max_profile_rows=2)

    assert result["row_count"] is None
    assert result["row_count_status"] == "bounded"
    assert result["profiled_row_count"] == 2
    assert "bounded" in result["warnings"][-1]
