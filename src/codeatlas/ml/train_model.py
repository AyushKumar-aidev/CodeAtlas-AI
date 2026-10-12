"""
ml/train_model.py

WHAT THIS FILE DOES, IN PLAIN TERMS
-------------------------------------
Reads the dataset made by build_dataset.py and answers one question:

    "Can git history (churn, authors, past bug fixes) predict which
     files will need bug fixes in the FUTURE - and is that better than
     a dead-simple guess?"

The simple guess (our BASELINE) is: "files that changed the most are the
riskiest". If the ML model cannot beat that, the model is not adding
value, and we must say so honestly (NFR5).

HOW WE MEASURE
--------------
ROC-AUC: a score from 0.5 to 1.0.
    0.5 = no better than flipping a coin
    1.0 = perfectly ranks every risky file above every safe one
We report it three ways:
    1. one train/test split (also gives precision & recall)
    2. repeated cross-validation (25 different splits). With only a few
       dozen files, one split is mostly luck, so we show the average
       AND how much it wobbles (+/-).
    3. LEAVE-ONE-REPO-OUT: train on all repos except one, test on the repo
       the model has never seen, and repeat for every repo. This is the
       honest test for a tool that will be pointed at brand-new repos.
       (Needs the "repo" column made by build_dataset.py.)

Usage (from the project root):
    python src/codeatlas/ml/train_model.py C:/Users/ayush/Desktop/flask_dataset.csv
"""

from __future__ import annotations

import argparse
import csv
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

FEATURES = ["churn_count", "author_count", "bug_fix_commit_count"]
SEED = 42


def load_dataset(path: str):
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    X = np.array([[float(r[f]) for f in FEATURES] for r in rows])
    y = np.array([int(r["risky"]) for r in rows])
    paths = [r["file_path"] for r in rows]
    # Older CSVs (single repo) have no "repo" column -> repos is None.
    repos = np.array([r["repo"] for r in rows]) if rows and "repo" in rows[0] else None
    return X, y, paths, repos


def make_model() -> RandomForestClassifier:
    # class_weight="balanced": risky files are the minority, so we tell the
    # forest to take them seriously instead of just predicting "safe" always.
    return RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=SEED
    )


def baseline_scores(X: np.ndarray) -> np.ndarray:
    """Baseline: risk score = churn_count (column 0). More changes = riskier."""
    return X[:, 0]


def holdout_report(X, y) -> None:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=SEED
    )
    model = make_model().fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)

    print("=== Single train/test split (70% / 30%) ===")
    print(f"train files: {len(y_tr)}  |  test files: {len(y_te)}  "
          f"(risky in test: {int(y_te.sum())})")
    print(f"Baseline (churn only)  ROC-AUC: {roc_auc_score(y_te, baseline_scores(X_te)):.3f}")
    print(f"Random Forest          ROC-AUC: {roc_auc_score(y_te, proba):.3f}")
    print(f"Random Forest precision: {precision_score(y_te, pred, zero_division=0):.3f}  "
          f"recall: {recall_score(y_te, pred, zero_division=0):.3f}")
    print("  precision = of files flagged risky, how many really were")
    print("  recall    = of truly risky files, how many we caught")


def cross_validation_report(X, y) -> None:
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    base_aucs, rf_aucs = [], []
    for tr, te in cv.split(X, y):
        model = make_model().fit(X[tr], y[tr])
        rf_aucs.append(roc_auc_score(y[te], model.predict_proba(X[te])[:, 1]))
        base_aucs.append(roc_auc_score(y[te], baseline_scores(X[te])))

    print("\n=== Repeated cross-validation (5 folds x 5 repeats = 25 tests) ===")
    print(f"Baseline (churn only)  ROC-AUC: {np.mean(base_aucs):.3f} +/- {np.std(base_aucs):.3f}")
    print(f"Random Forest          ROC-AUC: {np.mean(rf_aucs):.3f} +/- {np.std(rf_aucs):.3f}")

    diff = np.mean(rf_aucs) - np.mean(base_aucs)
    noise = np.std(np.array(rf_aucs) - np.array(base_aucs))
    if abs(diff) < noise:
        print("\nVerdict: the difference is smaller than the noise. On this data we")
        print("CANNOT claim the ML model beats the churn baseline.")
    elif diff > 0:
        print(f"\nVerdict: the Random Forest is ahead by {diff:.3f} AUC (bigger than the noise).")
    else:
        print(f"\nVerdict: the simple baseline is ahead by {-diff:.3f} AUC. The ML model is not helping here.")


def leave_one_repo_out_report(X, y, repos) -> None:
    print("\n=== Leave-one-repo-out (train on the other repos, test on the unseen one) ===")
    if repos is None or len(set(repos)) < 2:
        print("Skipped: needs a dataset built from 2+ repos (a 'repo' column).")
        return

    print(f"{'test repo':<16}{'files':>6}{'risky':>7}{'baseline AUC':>15}{'forest AUC':>13}{'diff':>8}")
    diffs = []
    for repo in sorted(set(repos)):
        test = repos == repo
        train = ~test
        # AUC is undefined if the test repo has only one class.
        if len(set(y[test])) < 2 or len(set(y[train])) < 2:
            print(f"{repo:<16}{int(test.sum()):>6}{int(y[test].sum()):>7}   skipped (only one class)")
            continue
        model = make_model().fit(X[train], y[train])
        rf = roc_auc_score(y[test], model.predict_proba(X[test])[:, 1])
        base = roc_auc_score(y[test], baseline_scores(X[test]))
        diffs.append(rf - base)
        print(f"{repo:<16}{int(test.sum()):>6}{int(y[test].sum()):>7}{base:>15.3f}{rf:>13.3f}{rf - base:>+8.3f}")

    if not diffs:
        return
    wins = sum(d > 0 for d in diffs)
    print(f"\nForest beats the baseline on {wins} of {len(diffs)} unseen repos "
          f"(average difference {np.mean(diffs):+.3f} AUC).")
    if wins == len(diffs):
        print("Verdict: the gain held on every repo the model had never seen.")
    elif wins > len(diffs) / 2:
        print("Verdict: the gain holds on most unseen repos but not all - report it that way.")
    else:
        print("Verdict: the gain does NOT generalise to unseen repos. Report that honestly.")


def feature_importance_report(X, y) -> None:
    model = make_model().fit(X, y)
    print("\n=== What the forest relied on (trained on all files) ===")
    for name, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda t: -t[1]):
        print(f"  {name:<22} {imp:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train and evaluate the file-risk model.")
    ap.add_argument("dataset", help="CSV produced by build_dataset.py")
    args = ap.parse_args()

    X, y, _, repos = load_dataset(args.dataset)
    n_risky = int(y.sum())
    print(f"Loaded {len(y)} files ({n_risky} risky, {len(y) - n_risky} not risky)\n")
    if n_risky < 5 or len(y) - n_risky < 5:
        print("Too few examples of one class to evaluate honestly. Use a bigger repo/dataset.")
        sys.exit(1)

    holdout_report(X, y)
    cross_validation_report(X, y)
    leave_one_repo_out_report(X, y, repos)
    feature_importance_report(X, y)


if __name__ == "__main__":
    main()
