"""Read-only benchmark and task lookup primitives."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime


@dataclass(frozen=True)
class BenchmarkReference:
    name: str
    task_type: str
    dataset: str
    modality: str
    metric: str
    source: str
    url: str
    description: str
    target: str | None = None
    protocol: str | None = None
    tags: tuple[str, ...] = ()
    pitfalls: tuple[str, ...] = ()
    baseline_suggestions: tuple[str, ...] = ()
    provenance_url: str | None = None
    matched_signals: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "task_type": self.task_type,
            "dataset": self.dataset,
            "modality": self.modality,
            "metric": self.metric,
            "source": self.source,
            "url": self.url,
            "description": self.description,
            "target": self.target,
            "protocol": self.protocol,
            "tags": list(self.tags),
            "pitfalls": list(self.pitfalls),
            "baseline_suggestions": list(self.baseline_suggestions),
            "provenance_url": self.provenance_url,
            "matched_signals": list(self.matched_signals),
        }


@dataclass(frozen=True)
class BenchmarkLookupResult:
    query: str
    backend: str
    retrieved_at: str
    benchmarks: tuple[BenchmarkReference, ...]
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "backend": self.backend,
            "retrieved_at": self.retrieved_at,
            "benchmarks": [benchmark.to_dict() for benchmark in self.benchmarks],
            "warnings": list(self.warnings),
        }


class LocalBenchmarkBackend:
    """Search a small built-in benchmark catalog for offline research planning."""

    name = "local"

    def __init__(
        self,
        benchmarks: Sequence[BenchmarkReference] | None = None,
        *,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._benchmarks = tuple(benchmarks or _BENCHMARK_CATALOG)
        self._now = now or _utcnow

    def lookup(self, query: str, *, max_results: int = 10) -> BenchmarkLookupResult:
        if max_results < 1:
            raise ValueError("max_results must be positive")

        tokens = _tokenize(query)
        ranked = tuple(_rank_benchmarks(self._benchmarks, tokens, max_results))
        warnings: list[str] = [
            "Local benchmark catalog is curated and incomplete; verify rules on source pages."
        ]
        if not ranked:
            warnings.insert(0, "No local benchmark entries matched the query.")

        return BenchmarkLookupResult(
            query=query,
            backend=self.name,
            retrieved_at=_isoformat(self._now()),
            benchmarks=ranked,
            warnings=tuple(warnings),
        )


def lookup_benchmarks(
    query: str,
    *,
    backend: LocalBenchmarkBackend,
    max_results: int = 10,
) -> BenchmarkLookupResult:
    return backend.lookup(query, max_results=max_results)


def _rank_benchmarks(
    benchmarks: Sequence[BenchmarkReference],
    tokens: tuple[str, ...],
    max_results: int,
) -> Iterable[BenchmarkReference]:
    scored: list[tuple[int, str, BenchmarkReference]] = []
    for benchmark in benchmarks:
        score = _benchmark_score(benchmark, tokens)
        if tokens and score == 0:
            continue
        fields = _benchmark_fields(benchmark)
        scored.append(
            (
                score,
                benchmark.name.casefold(),
                replace(
                    benchmark,
                    provenance_url=benchmark.url,
                    matched_signals=_matched_signals(fields=fields, tokens=tokens),
                ),
            )
        )

    scored.sort(key=lambda item: (-item[0], item[1]))
    return (item[2] for item in scored[:max_results])


def _benchmark_score(benchmark: BenchmarkReference, tokens: tuple[str, ...]) -> int:
    if not tokens:
        return 0

    fields = {
        "name": (benchmark.name, 5),
        "task_type": (benchmark.task_type, 4),
        "dataset": (benchmark.dataset, 4),
        "metric": (benchmark.metric, 3),
        "tags": (" ".join(benchmark.tags), 3),
        "description": (benchmark.description, 2),
        "protocol": (benchmark.protocol or "", 2),
        "pitfalls": (" ".join(benchmark.pitfalls), 1),
    }
    score = 0
    for value, weight in fields.values():
        lowered = value.casefold()
        score += sum(weight for token in tokens if token in lowered)
    return score


def _benchmark_fields(benchmark: BenchmarkReference) -> dict[str, str]:
    return {
        "name": benchmark.name,
        "task_type": benchmark.task_type,
        "dataset": benchmark.dataset,
        "metric": benchmark.metric,
        "tags": " ".join(benchmark.tags),
        "description": benchmark.description,
        "protocol": benchmark.protocol or "",
        "pitfalls": " ".join(benchmark.pitfalls),
    }


def _matched_signals(*, fields: dict[str, str], tokens: tuple[str, ...]) -> tuple[str, ...]:
    signals: list[str] = []
    for field_name, value in fields.items():
        lowered = value.casefold()
        matches = sorted({token for token in tokens if token in lowered})
        if matches:
            signals.append(f"matched {', '.join(matches)} in {field_name}")
    return tuple(signals)


def _tokenize(query: str) -> tuple[str, ...]:
    return tuple(
        token for token in re.findall(r"[a-zA-Z0-9_]+", query.casefold()) if len(token) > 2
    )


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _isoformat(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


_BENCHMARK_CATALOG = (
    BenchmarkReference(
        name="Titanic - Machine Learning from Disaster",
        task_type="binary_classification",
        dataset="Titanic passenger survival",
        modality="tabular",
        metric="accuracy",
        source="kaggle",
        url="https://www.kaggle.com/c/titanic",
        description="Beginner tabular classification task predicting passenger survival.",
        target="Survived",
        protocol="Train on labeled passengers and submit predictions for held-out passengers.",
        tags=("kaggle", "tabular", "classification", "binary", "beginner"),
        pitfalls=(
            "PassengerId is an identifier, not a predictive feature.",
            "Cabin and Age have substantial missingness.",
            "Leaderboard feedback can encourage overfitting on a small test set.",
        ),
        baseline_suggestions=(
            "Start with simple categorical/numeric preprocessing and logistic regression.",
            "Compare tree ensembles against a dummy majority-class baseline.",
        ),
    ),
    BenchmarkReference(
        name="House Prices - Advanced Regression Techniques",
        task_type="regression",
        dataset="Ames housing sale prices",
        modality="tabular",
        metric="root_mean_squared_error_on_log_sale_price",
        source="kaggle",
        url="https://www.kaggle.com/c/house-prices-advanced-regression-techniques",
        description="Tabular regression task predicting residential house sale prices.",
        target="SalePrice",
        protocol="Train on labeled homes and submit sale-price predictions for the test set.",
        tags=("kaggle", "tabular", "regression", "housing"),
        pitfalls=(
            "The evaluation is on log-transformed sale prices.",
            "Categorical levels can appear in test that are absent from train.",
            "Data leakage can come from post-sale or target-derived features.",
        ),
        baseline_suggestions=(
            "Use robust missing-value handling and one-hot encoding.",
            "Compare linear models against gradient boosting on log-transformed targets.",
        ),
    ),
    BenchmarkReference(
        name="Home Credit Default Risk",
        task_type="binary_classification",
        dataset="Home Credit consumer loan applications",
        modality="tabular_relational",
        metric="roc_auc",
        source="kaggle",
        url="https://www.kaggle.com/c/home-credit-default-risk",
        description="Large relational tabular task for predicting repayment difficulty.",
        target="TARGET",
        protocol=(
            "Join and aggregate multiple application, bureau, credit-card, and payment tables."
        ),
        tags=("kaggle", "tabular", "classification", "credit", "relational"),
        pitfalls=(
            "Train/test feature generation must use identical aggregation logic.",
            "Application IDs are join keys and should not be treated as ordinal features.",
            "Temporal tables can leak future behavior if aggregated carelessly.",
        ),
        baseline_suggestions=(
            "Start with application_train/application_test only before multi-table features.",
            "Use cross-validated ROC AUC and compare LightGBM/XGBoost to logistic regression.",
        ),
    ),
    BenchmarkReference(
        name="Santander Customer Transaction Prediction",
        task_type="binary_classification",
        dataset="Anonymized customer transaction features",
        modality="tabular",
        metric="roc_auc",
        source="kaggle",
        url="https://www.kaggle.com/c/santander-customer-transaction-prediction",
        description="Binary classification on anonymized numeric features.",
        target="target",
        protocol="Train on anonymized tabular features and submit probabilities for test rows.",
        tags=("kaggle", "tabular", "classification", "anonymized", "auc"),
        pitfalls=(
            "Feature names do not carry semantic meaning.",
            "Validation strategy matters because public leaderboard feedback is limited.",
            "Class imbalance can make accuracy misleading.",
        ),
        baseline_suggestions=(
            "Use stratified cross-validation and ROC AUC.",
            "Try regularized linear models before gradient boosting.",
        ),
    ),
    BenchmarkReference(
        name="M5 Forecasting - Accuracy",
        task_type="hierarchical_time_series_forecasting",
        dataset="Walmart retail unit sales",
        modality="time_series_tabular",
        metric="weighted_root_mean_squared_scaled_error",
        source="kaggle",
        url="https://www.kaggle.com/c/m5-forecasting-accuracy",
        description="Hierarchical retail demand forecasting across products, stores, and dates.",
        target="unit_sales",
        protocol="Forecast future daily sales across a product/store hierarchy.",
        tags=("kaggle", "forecasting", "time_series", "retail", "hierarchical"),
        pitfalls=(
            "Calendar and price features must be aligned by date.",
            "Leakage can happen if future sales or future aggregates enter training features.",
            "The metric is hierarchy-weighted and differs from plain RMSE.",
        ),
        baseline_suggestions=(
            "Build naive seasonal and moving-average baselines first.",
            "Use rolling-origin validation before training gradient boosting or deep models.",
        ),
    ),
    BenchmarkReference(
        name="GLUE",
        task_type="natural_language_understanding",
        dataset="General Language Understanding Evaluation tasks",
        modality="text",
        metric="task_specific_aggregate",
        source="benchmark",
        url="https://gluebenchmark.com/",
        description="Collection of sentence classification, similarity, and inference tasks.",
        protocol="Evaluate on the GLUE task suite using task-specific metrics.",
        tags=("nlp", "classification", "language_understanding", "benchmark"),
        pitfalls=(
            "Each task uses a different metric and label space.",
            "Leaderboard-style aggregate scores can hide poor performance on individual tasks.",
        ),
        baseline_suggestions=(
            "Report per-task metrics, not only aggregate score.",
            "Compare against a frozen encoder or simple fine-tuning baseline.",
        ),
    ),
)
