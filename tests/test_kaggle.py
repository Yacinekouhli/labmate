from pathlib import Path

import pytest

from labmate.tools.kaggle import (
    KaggleWorkflowError,
    create_kaggle_baseline,
    normalize_competition_slug,
    start_kaggle_competition,
    validate_kaggle_submission,
)


def _write_titanic_data(root: Path) -> None:
    data = root / "data"
    data.mkdir(parents=True)
    (data / "train.csv").write_text(
        "PassengerId,Age,Fare,Survived\n1,22,7.25,0\n2,38,71.28,1\n3,,8.05,1\n",
        encoding="utf-8",
    )
    (data / "test.csv").write_text(
        "PassengerId,Age,Fare\n4,35,8.05\n5,28,7.75\n",
        encoding="utf-8",
    )
    (data / "sample_submission.csv").write_text(
        "PassengerId,Survived\n4,0\n5,0\n",
        encoding="utf-8",
    )


def test_normalize_competition_slug_accepts_urls_and_slugs() -> None:
    assert normalize_competition_slug("titanic") == "titanic"
    assert (
        normalize_competition_slug("https://www.kaggle.com/competitions/titanic/data") == "titanic"
    )
    assert normalize_competition_slug("https://www.kaggle.com/c/titanic") == "titanic"


def test_normalize_competition_slug_rejects_non_kaggle_urls() -> None:
    with pytest.raises(KaggleWorkflowError) as exc_info:
        normalize_competition_slug("https://example.com/competitions/titanic")

    assert exc_info.value.code == "invalid_competition"


def test_kaggle_start_rejects_workspace_file_collision(tmp_path) -> None:
    workspace = tmp_path / "titanic"
    workspace.write_text("not a directory\n", encoding="utf-8")

    with pytest.raises(KaggleWorkflowError) as exc_info:
        start_kaggle_competition("titanic", workspace=workspace, download=False)

    assert exc_info.value.code == "invalid_arguments"


def test_kaggle_start_creates_workspace_without_download(tmp_path) -> None:
    workspace = tmp_path / "titanic"

    result = start_kaggle_competition(
        "https://www.kaggle.com/competitions/titanic",
        workspace=workspace,
        download=False,
    )

    assert result["competition"]["slug"] == "titanic"
    assert result["workspace"]["created"] is True
    assert (workspace / "data").is_dir()
    assert (workspace / "runs").is_dir()
    assert (workspace / "submissions").is_dir()
    assert (workspace / "results.tsv").is_file()
    assert (workspace / "program.md").is_file()
    assert (workspace / "reports" / "competition_brief.md").is_file()
    assert result["download"]["status"] == "skipped"
    assert result["data"]["status"] == "missing"
    assert result["submission_policy"]["status"] == "manual_approval_required"


def test_kaggle_start_degrades_when_kaggle_cli_is_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("labmate.tools.kaggle.shutil.which", lambda _name: None)

    result = start_kaggle_competition("titanic", workspace=tmp_path / "titanic")

    assert result["download"]["requested"] is True
    assert result["download"]["status"] == "unavailable"
    assert "Kaggle MCP" in result["download"]["recommended_fix"]
    assert result["workspace"]["created"] is True


def test_kaggle_start_inspects_existing_kaggle_style_data(tmp_path) -> None:
    workspace = tmp_path / "titanic"
    _write_titanic_data(workspace)

    result = start_kaggle_competition("titanic", workspace=workspace, download=False)

    assert result["data"]["status"] == "inspected"
    assert result["data"]["inspected_path"] == "data"
    assert result["data"]["research_brief"]["inferred_task"]["task_type"] == (
        "tabular classification"
    )
    assert result["data"]["research_brief"]["inferred_task"]["target_columns"] == ["Survived"]
    assert result["data"]["research_brief"]["modeling_plan"]["id_columns"] == ["PassengerId"]
    assert result["data"]["research_brief"]["modeling_plan"]["feature_columns"] == [
        "Age",
        "Fare",
    ]
    assert result["agent_handoff"]["claude_project_command"] == "/kagglethis titanic"
    assert result["next_actions"][1]["action"] == "inspect_data"
    assert result["next_actions"][3]["action"] == "create_constant_baseline"


def test_kaggle_start_is_idempotent_for_user_files(tmp_path) -> None:
    workspace = tmp_path / "titanic"
    first = start_kaggle_competition("titanic", workspace=workspace, download=False)
    second = start_kaggle_competition("titanic", workspace=workspace, download=False)

    assert first["experiment_tracking"]["created"] is True
    assert second["experiment_tracking"]["created"] is False
    file_actions = {action["path"]: action["action"] for action in second["workspace"]["files"]}
    assert file_actions["results.tsv"] == "skipped"
    assert file_actions["program.md"] == "skipped"
    assert file_actions["reports/competition_brief.md"] == "updated"


def test_kaggle_baseline_writes_submission_manifest_and_ledger(tmp_path) -> None:
    workspace = tmp_path / "titanic"
    _write_titanic_data(workspace)
    start_kaggle_competition("titanic", workspace=workspace, download=False)

    result = create_kaggle_baseline(workspace, run_name="dummy", strategy="auto")

    assert result["kind"] == "kaggle_baseline_run"
    assert result["competition"]["slug"] == "titanic"
    assert result["artifacts"]["submission_path"] == "submissions/dummy.csv"
    assert result["artifacts"]["manifest_path"] == "runs/dummy/manifest.json"
    assert result["prediction"]["fill_values"] == {"Survived": "1"}
    assert result["validation"]["status"] == "ok"
    assert result["ledger"]["appended"] is True
    assert result["ledger"]["row"]["experiment"] == "dummy"
    assert result["ledger"]["row"]["status"] == "submission_ready"
    assert (workspace / "submissions" / "dummy.csv").read_text(encoding="utf-8").splitlines() == [
        "PassengerId,Survived",
        "4,1",
        "5,1",
    ]
    assert (workspace / "runs" / "dummy" / "manifest.json").is_file()
    assert "dummy" in (workspace / "results.tsv").read_text(encoding="utf-8")


def test_kaggle_baseline_refuses_to_overwrite_existing_submission(tmp_path) -> None:
    workspace = tmp_path / "titanic"
    _write_titanic_data(workspace)
    start_kaggle_competition("titanic", workspace=workspace, download=False)
    create_kaggle_baseline(workspace, run_name="dummy")

    with pytest.raises(KaggleWorkflowError) as exc_info:
        create_kaggle_baseline(workspace, run_name="dummy")

    assert exc_info.value.code == "artifact_exists"


def test_validate_kaggle_submission_reports_schema_errors(tmp_path) -> None:
    workspace = tmp_path / "titanic"
    _write_titanic_data(workspace)
    candidate = workspace / "submissions" / "bad.csv"
    candidate.parent.mkdir()
    candidate.write_text("PassengerId,wrong\n4,0\n", encoding="utf-8")

    result = validate_kaggle_submission(candidate, workspace=workspace)

    assert result["status"] == "failed"
    assert result["schema"]["columns_match"] is False
    assert result["rows"]["row_counts_match"] is False
    assert result["errors"]
