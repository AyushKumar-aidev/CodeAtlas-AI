"""
ml/train_model.py

WHAT THIS FILE DOES, IN PLAIN TERMS
-------------------------------------
Reads the dataset made by build_dataset.py and answers one question:

    "Can git history (churn, authors, past bug fixes) plus the code itself
     (size, complexity, import-graph centrality) predict which files will
     need bug fixes in the FUTURE - and is that better than a dead-simple
     guess?"

The simple guesses (our BASELINES) are:
    - churn:      "files that changed the most are the riskiest"
    - complexity: "the most complex files are the riskiest"
                  (cyclomatic complexity, the baseline named in the SRS)
    - size:       "the biggest files (most lines) are the riskiest"
                  (a famously strong predictor; a fair test of whether
                  complexity/centrality add anything beyond plain size)
If the ML model cannot beat them, it is not adding value, and we must say
so honestly (NFR5).

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

GIT_FEATURES = ["churn_count", "author_count", "bug_fix_commit_count"]
CODE_FEATURES = ["loc", "complexity_total", "complexity_max", "in_degree", "pagerank"]
SEED = 42


def load_dataset(path: str):
    """
    Returns X, y, names, repos.
    `names` lists the feature columns actually found: the git columns always,
    plus the code columns if the CSV was built with them (older CSVs lack them).
    """
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    names = GIT_FEATURES + [c for c in CODE_FEATURES if rows and c in rows[0]]
    X = np.array([[float(r[f]) for f in names] for r in rows])
    y = np.array([int(r["risky"]) for r in rows])
    repos = np.array([r["repo"] for r in rows]) if rows and "repo" in rows[0] else None
    return X, y, names, repos


def make_model() -> RandomForestClassifier:
    # class_weight="balanced": risky files are the minority, so we tell the
    # forest to take them seriously instead of just predicting "safe" always.
    return RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=SEED
    )


def baselines(names) -> dict[str, int]:
    """Baseline name -> column index. Each baseline's risk score is ONE column."""
    found = {"churn": names.index("churn_count")}
    if "complexity_total" in names:
        found["complexity"] = names.index("complexity_total")
    if "loc" in names:
        found["size"] = names.index("loc")
    return found


def describe(diff: float, noise: float) -> str:
    if abs(diff) < noise:
        return "no clear difference (smaller than the noise)"
    return "forest ahead" if diff > 0 else "BASELINE ahead"


def holdout_report(X, y, names) -> None:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=SEED
    )
    model = make_model().fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)

    print("=== Single train/test split (70% / 30%) ===")
    print(f"train files: {len(y_tr)}  |  test files: {len(y_te)}  "
          f"(risky in test: {int(y_te.sum())})")
    for bname, col in baselines(names).items():
        print(f"Baseline ({bname} only)".ljust(28) + f"ROC-AUC: {roc_auc_score(y_te, X_te[:, col]):.3f}")
    print("Random Forest".ljust(28) + f"ROC-AUC: {roc_auc_score(y_te, proba):.3f}")
    print(f"Random Forest precision: {precision_score(y_te, pred, zero_division=0):.3f}  "
          f"recall: {recall_score(y_te, pred, zero_division=0):.3f}")
    print("  precision = of files flagged risky, how many really were")
    print("  recall    = of truly risky files, how many we caught")


def cross_validation_report(X, y, names) -> None:
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    base_cols = baselines(names)
    base_aucs = {b: [] for b in base_cols}
    rf_aucs = []
    for tr, te in cv.split(X, y):
        model = make_model().fit(X[tr], y[tr])
        rf_aucs.append(roc_auc_score(y[te], model.predict_proba(X[te])[:, 1]))
        for b, col in base_cols.items():
            base_aucs[b].append(roc_auc_score(y[te], X[te, col]))

    print("\n=== Repeated cross-validation (5 folds x 5 repeats = 25 tests) ===")
    for b in base_cols:
        print(f"Baseline ({b} only)".ljust(28) +
              f"ROC-AUC: {np.mean(base_aucs[b]):.3f} +/- {np.std(base_aucs[b]):.3f}")
    print("Random Forest".ljust(28) + f"ROC-AUC: {np.mean(rf_aucs):.3f} +/- {np.std(rf_aucs):.3f}")
    print("(note: these random splits mix files from the same repo - the")
    print(" leave-one-repo-out test below is the stricter, more honest one)")
    print()
    for b in base_cols:
        d = np.array(rf_aucs) - np.array(base_aucs[b])
        print(f"  forest vs {b} baseline: {d.mean():+.3f}  -> {describe(d.mean(), d.std())}")


def leave_one_repo_out_report(X, y, names, repos) -> None:
    print("\n=== Leave-one-repo-out (train on the other repos, test on the unseen one) ===")
    if repos is None or len(set(repos)) < 2:
        print("Skipped: needs a dataset built from 2+ repos (a 'repo' column).")
        return

    base_cols = baselines(names)
    git_cols = [names.index(f) for f in GIT_FEATURES]
    has_code = len(names) > len(GIT_FEATURES)

    header = f"{'test repo':<16}{'files':>6}{'risky':>7}"
    for b in base_cols:
        header += f"{b + ' base':>17}"
    if has_code:
        header += f"{'forest(git)':>13}"
    header += f"{'forest(all)':>13}"
    print(header)

    results = []  # (baseline AUCs dict, forest_git, forest_all)
    for repo in sorted(set(repos)):
        test = repos == repo
        train = ~test
        # AUC is undefined if the test repo has only one class.
        if len(set(y[test])) < 2 or len(set(y[train])) < 2:
            print(f"{repo:<16}{int(test.sum()):>6}{int(y[test].sum()):>7}   skipped (only one class)")
            continue
        base = {b: roc_auc_score(y[test], X[test, c]) for b, c in base_cols.items()}
        rf_all = roc_auc_score(
            y[test], make_model().fit(X[train], y[train]).predict_proba(X[test])[:, 1]
        )
        rf_git = None
        line = f"{repo:<16}{int(test.sum()):>6}{int(y[test].sum()):>7}"
        for b in base_cols:
            line += f"{base[b]:>17.3f}"
        if has_code:
            rf_git = roc_auc_score(
                y[test],
                make_model().fit(X[train][:, git_cols], y[train]).predict_proba(X[test][:, git_cols])[:, 1],
            )
            line += f"{rf_git:>13.3f}"
        line += f"{rf_all:>13.3f}"
        print(line)
        results.append((base, rf_git, rf_all))

    if not results:
        return
    n = len(results)
    print()
    for b in base_cols:
        wins = sum(r[2] > r[0][b] for r in results)
        avg = np.mean([r[2] - r[0][b] for r in results])
        print(f"forest(all) beats the {b} baseline on {wins} of {n} unseen repos (average {avg:+.3f} AUC)")
    if has_code:
        wins = sum(r[2] > r[1] for r in results)
        avg = np.mean([r[2] - r[1] for r in results])
        print(f"adding code features helps on {wins} of {n} unseen repos (average {avg:+.3f} AUC vs git-only)")

    best_wins = sum(all(r[2] > r[0][b] for b in base_cols) for r in results)
    print()
    if best_wins == n:
        print("Verdict: the forest beat every baseline on every repo it had never seen.")
    elif best_wins > n / 2:
        print("Verdict: the forest beats the baselines on most unseen repos but not all - report it that way.")
    else:
        print("Verdict: the forest does NOT reliably beat the baselines on unseen repos. Report that honestly.")


def feature_importance_report(X, y, names) -> None:
    model = make_model().fit(X, y)
    print("\n=== What the forest relied on (trained on all files) ===")
    for name, imp in sorted(zip(names, model.feature_importances_), key=lambda t: -t[1]):
        print(f"  {name:<22} {imp:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train and evaluate the file-risk model.")
    ap.add_argument("dataset", help="CSV produced by build_dataset.py")
    args = ap.parse_args()

    X, y, names, repos = load_dataset(args.dataset)
    n_risky = int(y.sum())
    print(f"Loaded {len(y)} files ({n_risky} risky, {len(y) - n_risky} not risky)")
    print(f"Features used: {', '.join(names)}\n")
    if n_risky < 5 or len(y) - n_risky < 5:
        print("Too few examples of one class to evaluate honestly. Use a bigger repo/dataset.")
        sys.exit(1)

    holdout_report(X, y, names)
    cross_validation_report(X, y, names)
    leave_one_repo_out_report(X, y, names, repos)
    feature_importance_report(X, y, names)


if __name__ == "__main__":
    main()
