"""Train and compare spam classifiers; export best pipeline + metrics/plots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "spam.csv"
ARTIFACTS = ROOT / "artifacts"
ASSETS = ROOT / "assets"
PIPELINE_PATH = ARTIFACTS / "spam_pipeline.joblib"
METRICS_PATH = ARTIFACTS / "metrics.json"
CONFUSION_PATH = ASSETS / "confusion_matrix.png"
ROC_PATH = ASSETS / "roc_curve.png"

LABEL_MAP = {"ham": 0, "spam": 1}
LABEL_NAMES = {0: "ham", 1: "spam"}


def load_spam_csv(path: str | Path = DEFAULT_DATA) -> pd.DataFrame:
    """Load UCI SMS spam CSV (latin-1) and return cleaned label/text frame."""
    path = Path(path)
    df = pd.read_csv(path, encoding="latin-1")
    # Standard columns are v1 (label) and v2 (message); drop unnamed extras.
    if {"v1", "v2"}.issubset(df.columns):
        df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "text"})
    elif {"label", "text"}.issubset(df.columns):
        df = df[["label", "text"]]
    else:
        raise ValueError(f"Unexpected columns in {path}: {list(df.columns)}")

    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["label"].isin(LABEL_MAP)].dropna(subset=["text"])
    df = df[df["text"].str.len() > 0].drop_duplicates().reset_index(drop=True)
    df["y"] = df["label"].map(LABEL_MAP).astype(int)
    return df


def build_pipelines() -> dict[str, Pipeline]:
    """Three comparable text classification pipelines."""
    return {
        "count_mnb": Pipeline(
            steps=[
                (
                    "vect",
                    CountVectorizer(
                        stop_words="english",
                        ngram_range=(1, 2),
                        min_df=2,
                        max_features=20_000,
                    ),
                ),
                ("clf", MultinomialNB(alpha=0.5)),
            ]
        ),
        "tfidf_logreg": Pipeline(
            steps=[
                (
                    "vect",
                    TfidfVectorizer(
                        stop_words="english",
                        ngram_range=(1, 2),
                        min_df=2,
                        max_features=20_000,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "tfidf_linearsvc": Pipeline(
            steps=[
                (
                    "vect",
                    TfidfVectorizer(
                        stop_words="english",
                        ngram_range=(1, 2),
                        min_df=2,
                        max_features=20_000,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "clf",
                    CalibratedClassifierCV(
                        LinearSVC(dual="auto", class_weight="balanced", random_state=42),
                        method="sigmoid",
                        cv=3,
                    ),
                ),
            ]
        ),
    }


def _spam_proba(pipe: Pipeline, texts: list[str] | np.ndarray) -> np.ndarray:
    """Return P(spam) for each text."""
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(texts)
        classes = list(pipe.classes_)
        spam_idx = classes.index(1) if 1 in classes else classes.index("spam")
        return proba[:, spam_idx]
    # Fallback: decision function → sigmoid-like ranking (should not hit for our pipes).
    scores = pipe.decision_function(texts)
    scores = np.asarray(scores, dtype=float).ravel()
    return 1.0 / (1.0 + np.exp(-scores))


def _evaluate(
    name: str,
    pipe: Pipeline,
    X_test: list[str],
    y_test: np.ndarray,
) -> dict[str, Any]:
    preds = pipe.predict(X_test)
    # Normalize string labels if any.
    preds = np.asarray(
        [LABEL_MAP.get(str(p).lower(), int(p)) for p in preds], dtype=int
    )
    y_true = np.asarray(y_test, dtype=int)
    proba = _spam_proba(pipe, X_test)
    return {
        "name": name,
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "y_pred": preds.tolist(),
        "y_proba": proba.tolist(),
    }


def _save_confusion(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["ham", "spam"],
        yticklabels=["ham", "spam"],
        ax=ax,
        cbar=False,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (best model)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _save_roc(y_true: np.ndarray, y_proba: np.ndarray, auc: float, path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(fpr, tpr, color="#1f4e79", lw=2, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#888888", lw=1, linestyle="--", label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (best model)")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def train(
    data_path: str | Path = DEFAULT_DATA,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    df = load_spam_csv(data_path)
    X = df["text"].tolist()
    y = df["y"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    results: list[dict[str, Any]] = []
    fitted: dict[str, Pipeline] = {}
    for name, pipe in build_pipelines().items():
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        results.append(_evaluate(name, pipe, X_test, y_test))

    best = max(results, key=lambda r: (r["f1"], r["roc_auc"], r["accuracy"]))
    best_pipe = fitted[best["name"]]

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipe, PIPELINE_PATH)

    y_pred = np.asarray(best["y_pred"], dtype=int)
    y_proba = np.asarray(best["y_proba"], dtype=float)
    _save_confusion(y_test, y_pred, CONFUSION_PATH)
    _save_roc(y_test, y_proba, best["roc_auc"], ROC_PATH)

    models_summary = [
        {
            "name": r["name"],
            "accuracy": r["accuracy"],
            "precision": r["precision"],
            "recall": r["recall"],
            "f1": r["f1"],
            "roc_auc": r["roc_auc"],
        }
        for r in results
    ]

    metrics: dict[str, Any] = {
        "n_samples": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "spam_rate": float(y.mean()),
        "test_size": test_size,
        "random_state": random_state,
        "best_model": best["name"],
        "best": {
            "name": best["name"],
            "accuracy": best["accuracy"],
            "precision": best["precision"],
            "recall": best["recall"],
            "f1": best["f1"],
            "roc_auc": best["roc_auc"],
        },
        "models": models_summary,
        "artifacts": {
            "pipeline": str(PIPELINE_PATH.relative_to(ROOT)),
            "confusion_matrix": str(CONFUSION_PATH.relative_to(ROOT)),
            "roc_curve": str(ROC_PATH.relative_to(ROOT)),
        },
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    metrics = train()
    best = metrics["best"]
    print(
        f"Best={best['name']} | "
        f"acc={best['accuracy']:.4f} f1={best['f1']:.4f} "
        f"roc_auc={best['roc_auc']:.4f}"
    )
    print(f"Saved {PIPELINE_PATH} and {METRICS_PATH}")
    for m in metrics["models"]:
        print(
            f"  {m['name']}: acc={m['accuracy']:.4f} "
            f"f1={m['f1']:.4f} auc={m['roc_auc']:.4f}"
        )


if __name__ == "__main__":
    main()
