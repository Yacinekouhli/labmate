import gzip

from labmate.tools.projects import scan_local_project


def test_scan_local_project_finds_kaggle_dataset_and_entrypoints(tmp_path) -> None:
    (tmp_path / "AGENTS.md").write_text("agent notes\n", encoding="utf-8")
    (tmp_path / "program.md").write_text("program\n", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("scikit-learn\n", encoding="utf-8")
    (tmp_path / "train_model.py").write_text("print('train')\n", encoding="utf-8")
    (tmp_path / "exploration.ipynb").write_text("{}", encoding="utf-8")
    data = tmp_path / "data"
    data.mkdir()
    (data / "evaluation.md").write_text("AUC metric\n", encoding="utf-8")
    (data / "train.csv").write_text("id,feature,target\n1,10,0\n", encoding="utf-8")
    (data / "test.csv").write_text("id,feature\n2,11\n", encoding="utf-8")
    (data / "sample_submission.csv").write_text("id,target\n2,0\n", encoding="utf-8")

    result = scan_local_project(tmp_path)

    assert result["kind"] == "local_project_scan"
    assert result["scan_limits"]["truncated"] is False
    assert result["dataset_candidates"][0]["path"] == "data"
    assert result["dataset_candidates"][0]["absolute_path"] == str(data)
    assert result["dataset_candidates"][0]["kind"] == "kaggle_dataset_directory"
    assert result["dataset_candidates"][0]["roles"] == {
        "sample_submission": "sample_submission.csv",
        "test": "test.csv",
        "train": "train.csv",
    }
    assert result["dataset_candidates"][0]["recommended_command"] == (
        f"labmate research-brief {data}"
    )
    assert result["recommended_next_commands"] == [
        "labmate tools",
        f"labmate research-brief {data}",
        f"labmate dataset-inspect {data}",
    ]
    assert result["dependency_files"] == [
        {"path": "requirements.txt", "kind": "python_requirements"}
    ]
    assert result["agent_files"] == [
        {"path": "AGENTS.md", "kind": "agent_instructions"},
        {"path": "program.md", "kind": "labmate_program"},
    ]
    assert result["code_entrypoints"][0]["path"] == "train_model.py"
    assert result["warnings"] == []


def test_scan_local_project_reports_missing_dataset_candidates(tmp_path) -> None:
    (tmp_path / "README.md").write_text("no data here\n", encoding="utf-8")

    result = scan_local_project(tmp_path)

    assert result["dataset_candidates"] == []
    assert result["recommended_next_commands"] == ["labmate tools"]
    assert result["warnings"] == [
        "No local CSV/TSV dataset candidates were found within scan limits."
    ]


def test_scan_local_project_finds_gzipped_kaggle_dataset(tmp_path) -> None:
    data = tmp_path / "compressed-data"
    data.mkdir()
    with gzip.open(data / "train.csv.gz", mode="wt", encoding="utf-8") as handle:
        handle.write("id,feature,target\n1,10,0\n")
    with gzip.open(data / "test.csv.gz", mode="wt", encoding="utf-8") as handle:
        handle.write("id,feature\n2,11\n")
    with gzip.open(data / "sample_submission.csv.gz", mode="wt", encoding="utf-8") as handle:
        handle.write("id,target\n2,0\n")

    result = scan_local_project(tmp_path)

    assert result["dataset_candidates"][0]["kind"] == "kaggle_dataset_directory"
    assert result["dataset_candidates"][0]["files"] == [
        "compressed-data/sample_submission.csv.gz",
        "compressed-data/test.csv.gz",
        "compressed-data/train.csv.gz",
    ]
    assert result["dataset_candidates"][0]["roles"] == {
        "sample_submission": "sample_submission.csv.gz",
        "test": "test.csv.gz",
        "train": "train.csv.gz",
    }
