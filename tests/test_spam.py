"""Tests for spam pipeline load, prediction polarity, and probability bounds."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.predict import load_pipeline, predict
from src.train import METRICS_PATH, PIPELINE_PATH, load_spam_csv, train

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "spam.csv"

SPAM_EXAMPLE = (
    "WINNER!! You have been selected to receive a £900 prize reward! "
    "To claim call 09061701461. Claim code KL341. Valid 12 hours only."
)
HAM_EXAMPLE = (
    "Hey, are we still on for lunch tomorrow at noon near the office?"
)


@pytest.fixture(scope="module")
def trained_metrics():
    assert DATA.is_file(), "data/spam.csv must ship with the repo"
    metrics = train(DATA)
    assert PIPELINE_PATH.is_file()
    assert METRICS_PATH.is_file()
    assert metrics["best"]["f1"] >= 0.85
    return metrics


@pytest.fixture(scope="module")
def pipeline(trained_metrics):
    return load_pipeline(PIPELINE_PATH)


def test_dataset_shape():
    df = load_spam_csv(DATA)
    assert len(df) > 1000
    assert set(df["label"].unique()) <= {"ham", "spam"}
    assert df["y"].isin([0, 1]).all()


def test_model_file_loads(pipeline):
    assert pipeline is not None
    assert hasattr(pipeline, "predict")


def test_obvious_spam_vs_ham(pipeline):
    spam = predict(SPAM_EXAMPLE, pipeline=pipeline)
    ham = predict(HAM_EXAMPLE, pipeline=pipeline)
    assert spam["label"] == "spam"
    assert ham["label"] == "ham"
    assert spam["spam_probability"] > ham["spam_probability"]
    assert spam["spam_probability"] >= 0.5
    assert ham["spam_probability"] < 0.5


def test_probability_bounds(pipeline):
    for text in (SPAM_EXAMPLE, HAM_EXAMPLE, "", "ok"):
        result = predict(text, pipeline=pipeline)
        p = result["spam_probability"]
        assert 0.0 <= p <= 1.0
        assert result["label"] in {"spam", "ham"}


def test_top_tokens_present_for_spam(pipeline):
    result = predict(SPAM_EXAMPLE, pipeline=pipeline, top_k=5)
    assert isinstance(result["top_tokens"], list)
    # Spam copy should surface at least one positively weighted token.
    assert any(t["score"] > 0 for t in result["top_tokens"])


def test_metrics_include_all_models(trained_metrics):
    names = {m["name"] for m in trained_metrics["models"]}
    assert names == {"count_mnb", "tfidf_logreg", "tfidf_linearsvc"}
    assert trained_metrics["best_model"] in names
