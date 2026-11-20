"""CLI and programmatic inference for the spam pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from src.train import LABEL_NAMES, PIPELINE_PATH

ROOT = Path(__file__).resolve().parents[1]


def load_pipeline(path: str | Path = PIPELINE_PATH) -> Pipeline:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing pipeline at {path}. Run: python -m src.train"
        )
    return joblib.load(path)


def _unwrap_estimator(clf: Any) -> Any:
    """Return the linear base estimator when wrapped in CalibratedClassifierCV."""
    if isinstance(clf, CalibratedClassifierCV):
        # Prefer a fitted calibrated classifier's first fold estimator.
        if hasattr(clf, "calibrated_classifiers_") and clf.calibrated_classifiers_:
            calibrated = clf.calibrated_classifiers_[0]
            if hasattr(calibrated, "estimator"):
                return calibrated.estimator
            if hasattr(calibrated, "base_estimator"):
                return calibrated.base_estimator
        if hasattr(clf, "estimator"):
            return clf.estimator
    return clf


def _spam_class_index(clf: Any) -> int:
    classes = list(getattr(clf, "classes_", [0, 1]))
    if 1 in classes:
        return classes.index(1)
    if "spam" in classes:
        return classes.index("spam")
    return int(np.argmax(classes))


def top_contributing_tokens(
    pipe: Pipeline,
    text: str,
    top_k: int = 8,
) -> list[dict[str, Any]]:
    """
    Rank tokens present in `text` by contribution toward the spam class.

    Uses vectorizer feature weights × model coefficients (or NB log-odds).
    """
    vect = pipe.named_steps["vect"]
    clf = pipe.named_steps["clf"]
    X = vect.transform([text])
    feature_names = np.asarray(vect.get_feature_names_out())
    indices = X.nonzero()[1]
    if len(indices) == 0:
        return []

    values = np.asarray(X[0, indices].todense()).ravel()
    base = _unwrap_estimator(clf)
    spam_idx = _spam_class_index(clf if hasattr(clf, "classes_") else base)

    if hasattr(base, "coef_"):
        coef = np.asarray(base.coef_)
        if coef.ndim == 1:
            weights = coef
        elif coef.shape[0] == 1:
            weights = coef[0]
        else:
            weights = coef[spam_idx]
        scores = values * weights[indices]
    elif hasattr(base, "feature_log_prob_"):
        # MultinomialNB: spam log-prob − ham log-prob, scaled by counts.
        log_prob = np.asarray(base.feature_log_prob_)
        ham_idx = 1 - spam_idx if log_prob.shape[0] == 2 else 0
        log_odds = log_prob[spam_idx] - log_prob[ham_idx]
        scores = values * log_odds[indices]
    else:
        return []

    order = np.argsort(-scores)
    out: list[dict[str, Any]] = []
    for i in order[:top_k]:
        out.append(
            {
                "token": str(feature_names[indices[i]]),
                "score": float(scores[i]),
            }
        )
    return out


def predict(
    text: str,
    pipeline: Pipeline | None = None,
    top_k: int = 8,
) -> dict[str, Any]:
    """Return label, spam probability, and top contributing tokens."""
    pipe = pipeline or load_pipeline()
    text = (text or "").strip()
    if not text:
        return {
            "label": "ham",
            "spam_probability": 0.0,
            "top_tokens": [],
        }

    pred = pipe.predict([text])[0]
    if str(pred).lower() in {"spam", "ham"}:
        label = str(pred).lower()
        label_id = 1 if label == "spam" else 0
    else:
        label_id = int(pred)
        label = LABEL_NAMES.get(label_id, str(pred))

    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba([text])[0]
        classes = list(pipe.classes_)
        if 1 in classes:
            spam_idx = classes.index(1)
        elif "spam" in classes:
            spam_idx = classes.index("spam")
        else:
            spam_idx = -1
        spam_p = float(proba[spam_idx]) if spam_idx >= 0 else float(proba.max())
    else:
        spam_p = 1.0 if label_id == 1 else 0.0

    spam_p = float(np.clip(spam_p, 0.0, 1.0))
    return {
        "label": label,
        "spam_probability": spam_p,
        "top_tokens": top_contributing_tokens(pipe, text, top_k=top_k),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify SMS/email text as spam or ham.")
    parser.add_argument("text", nargs="?", help="Message text to classify")
    parser.add_argument(
        "-f",
        "--file",
        type=Path,
        help="Optional path to a UTF-8 text file",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=PIPELINE_PATH,
        help="Path to spam_pipeline.joblib",
    )
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    if args.file:
        text = args.file.read_text(encoding="utf-8")
    elif args.text:
        text = args.text
    else:
        parser.error("Provide text or --file")

    result = predict(text, pipeline=load_pipeline(args.model), top_k=args.top_k)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
