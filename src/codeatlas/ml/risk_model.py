"""
ml/risk_model.py

WHY THIS FILE EXISTS
---------------------
ml/train_model.py EVALUATES the model (it prints scores and exits). The web
API needs a model it can load and use, so this file does the other half:
train on a dataset CSV, save the result to disk, load it back, score files.

It reuses load_dataset() and make_model() from train_model.py, so the saved
model is exactly the model that was evaluated there - one definition only.

HONESTY NOTE (NFR5)
--------------------
The saved model is trained on ALL rows of the CSV, so its score on those same
rows would be meaninglessly high. Its real quality is whatever
`python -m codeatlas.ml.train_model <csv>` reports (especially the
leave-one-repo-out section). The risk_score it produces is a RANKING score
between 0 and 1, not a calibrated probability: the forest is trained with
class_weight="balanced", which inflates the numbers.

Usage (from the project root):
    python -m codeatlas.ml.risk_model dataset.csv --out models/risk_model.joblib

Only load model files you created yourself: joblib files are pickles, and
loading a pickle from an untrusted source can run arbitrary code.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from codeatlas.ml.train_model import load_dataset, make_model

FORMAT_VERSION = 1
MIN_PER_CLASS = 5  # same floor train_model.py uses before it will evaluate


class ModelError(Exception):
    """A model file is missing the expected content or could not be used."""


def train_and_save(dataset_csv: str, model_path: str | Path) -> dict:
    """Train on the whole dataset, save it, and return the metadata about it."""
    X, y, names, repos = load_dataset(dataset_csv)
    n_risky = int(y.sum())
    if n_risky < MIN_PER_CLASS or len(y) - n_risky < MIN_PER_CLASS:
        raise ModelError(
            f"Too few examples of one class ({n_risky} risky of {len(y)}) to train."
        )

    model = make_model().fit(X, y)
    info = {
        "format_version": FORMAT_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feature_names": list(names),
        "n_files": int(len(y)),
        "n_risky": n_risky,
        "repos": sorted(set(repos.tolist())) if repos is not None else [],
    }
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "info": info}, path)
    return info


def load_model(model_path: str | Path) -> dict | None:
    """
    Returns the saved bundle ({"model", "info"}), or None if there is no file.
    Raises ModelError if a file exists but cannot be used - we never silently
    pretend a broken model is fine.
    """
    path = Path(model_path)
    if not path.is_file():
        return None
    try:
        bundle = joblib.load(path)
    except Exception as exc:  # corrupt file, incompatible scikit-learn, ...
        raise ModelError(
            f"Could not load the model file {path} ({type(exc).__name__}). "
            "Retrain it with: python -m codeatlas.ml.risk_model <dataset.csv>"
        ) from exc
    if (
        not isinstance(bundle, dict)
        or "model" not in bundle
        or bundle.get("info", {}).get("format_version") != FORMAT_VERSION
    ):
        raise ModelError(f"{path} is not a model file this version understands; retrain it.")
    return bundle


def score_files(bundle: dict, rows: list[dict]) -> list[float]:
    """
    One risk score (0..1) per row. Each row is a dict holding every feature
    the model was trained on (churn_count, loc, pagerank, ...).
    """
    names = bundle["info"]["feature_names"]
    missing = [n for n in names if rows and n not in rows[0]]
    if missing:
        raise ModelError(f"Cannot score: missing feature(s) {', '.join(missing)}")
    if not rows:
        return []
    X = np.array([[float(r[n]) for n in names] for r in rows])
    return [float(p) for p in bundle["model"].predict_proba(X)[:, 1]]


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the file-risk model and save it.")
    ap.add_argument("dataset", help="CSV produced by build_dataset.py")
    ap.add_argument("--out", default="models/risk_model.joblib", help="where to save the model")
    args = ap.parse_args()
    try:
        info = train_and_save(args.dataset, args.out)
    except ModelError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
    print(f"Saved model to {args.out}")
    print(f"  trained on {info['n_files']} files ({info['n_risky']} risky) "
          f"from {len(info['repos'])} repo(s): {', '.join(info['repos']) or 'unknown'}")
    print(f"  features: {', '.join(info['feature_names'])}")
    print("Remember: judge its quality with train_model.py, not by this model's score on its own training data.")


if __name__ == "__main__":
    main()
