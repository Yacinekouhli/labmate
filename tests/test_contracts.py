import json

from labmate.contracts import ExitCode, failure, response_to_json, success


def test_success_response_has_json_serializable_contract_shape() -> None:
    response = success(
        "dataset_inspect",
        {
            "dataset": "titanic",
            "rows": 891,
            "columns": ["survived", "pclass", "sex", "age"],
            "has_target": True,
        },
        metadata={"backend": "kaggle", "cached": False},
    )

    payload = response.to_dict()

    assert payload == {
        "schema_version": "labmate.tool.v1",
        "ok": True,
        "tool": "dataset_inspect",
        "exit_code": 0,
        "result": {
            "dataset": "titanic",
            "rows": 891,
            "columns": ["survived", "pclass", "sex", "age"],
            "has_target": True,
        },
        "metadata": {"backend": "kaggle", "cached": False},
    }
    assert json.loads(json.dumps(payload)) == payload
    assert response.exit_code == ExitCode.OK


def test_failure_response_has_structured_error_and_exit_code() -> None:
    response = failure(
        "literature_search",
        code="rate_limited",
        message="Semantic Scholar rate limit exceeded.",
        exit_code=ExitCode.RATE_LIMITED,
        retryable=True,
        details={"backend": "semantic_scholar", "retry_after_seconds": 60},
    )

    payload = response.to_dict()

    assert payload == {
        "schema_version": "labmate.tool.v1",
        "ok": False,
        "tool": "literature_search",
        "exit_code": 13,
        "error": {
            "code": "rate_limited",
            "message": "Semantic Scholar rate limit exceeded.",
            "retryable": True,
            "details": {"backend": "semantic_scholar", "retry_after_seconds": 60},
        },
        "metadata": {},
    }
    assert json.loads(json.dumps(payload)) == payload
    assert response.exit_code == ExitCode.RATE_LIMITED


def test_response_to_json_is_stable_and_parseable() -> None:
    response = success("benchmark_lookup", {"metric": "rmse", "task": "tabular-regression"})

    serialized = response_to_json(response)

    assert list(json.loads(serialized)) == [
        "exit_code",
        "metadata",
        "ok",
        "result",
        "schema_version",
        "tool",
    ]
