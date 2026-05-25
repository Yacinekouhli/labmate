from labmate.tools.experiments import summarize_experiments


def test_summarize_experiments_reports_best_and_latest_runs(tmp_path) -> None:
    ledger = tmp_path / "results.tsv"
    ledger.write_text(
        (
            "timestamp_utc\tcommit\texperiment\tmodel_family\tfeatures\tvalidation_strategy\t"
            "metric\tscore\tscore_direction\tstatus\tartifacts\tnotes\n"
            "2026-05-25T10:00:00Z\taaa111\tdummy\tdummy\tbase\tstratified_k_fold\t"
            "roc_auc\t0.500\tmaximize\tkeep\tbaseline.json\tfloor\n"
            "2026-05-25T11:00:00Z\tbbb222\tlinear\tlogistic_regression\tbase\t"
            "stratified_k_fold\troc_auc\t0.620\tmaximize\tkeep\tlinear.pkl\tbest simple\n"
            "2026-05-25T12:00:00Z\tccc333\ttree\tgradient_boosting\tbase\t"
            "stratified_k_fold\troc_auc\t0.610\tmaximize\treject\ttree.pkl\toverfit\n"
        ),
        encoding="utf-8",
    )

    result = summarize_experiments(tmp_path)

    assert result["kind"] == "experiment_summary"
    assert result["status"] == "ok"
    assert result["ledger"]["path"] == "results.tsv"
    assert result["ledger"]["completed_run_count"] == 3
    assert result["metric_summary"] == {
        "primary_metric": "roc_auc",
        "score_direction": "maximize",
        "scored_run_count": 3,
        "metrics_seen": {"roc_auc": 3},
    }
    assert result["best_run"]["experiment"] == "linear"
    assert result["best_run"]["score"] == 0.62
    assert result["latest_run"]["experiment"] == "tree"
    assert result["status_counts"] == {"keep": 2, "reject": 1}
    assert result["warnings"] == []
    assert result["recommended_next_actions"][0] == "Continue logging runs in results.tsv."


def test_summarize_experiments_handles_missing_ledger(tmp_path) -> None:
    result = summarize_experiments(tmp_path)

    assert result["status"] == "not_found"
    assert result["ledger"] is None
    assert result["best_run"] is None
    assert result["warnings"]


def test_summarize_experiments_infers_minimize_direction(tmp_path) -> None:
    ledger = tmp_path / "runs.csv"
    ledger.write_text(
        (
            "timestamp_utc,commit,experiment,model_family,metric,score,status\n"
            "2026-05-25T10:00:00Z,aaa111,baseline,linear,rmse,1.2,keep\n"
            "2026-05-25T11:00:00Z,bbb222,tree,gradient_boosting,rmse,0.9,keep\n"
        ),
        encoding="utf-8",
    )

    result = summarize_experiments(ledger)

    assert result["metric_summary"]["score_direction"] == "minimize"
    assert result["best_run"]["experiment"] == "tree"
    assert result["best_run"]["score"] == 0.9
    assert any("missing recommended columns" in warning for warning in result["warnings"])
