import pytest

from codeatlas.ml.risk_model import ModelError, load_model, train_and_save
from codeatlas.services.analysis import AnalysisError, analyze_repo


def test_pipeline_on_real_git_repo_without_model(sample_repo, tmp_path):
    result = analyze_repo(sample_repo, tmp_path / "no_model.joblib")
    summary, files = result["summary"], result["files"]

    assert summary["n_files"] == 4  # app.py, pkg/__init__.py, util.py, core.py
    assert summary["scoring_method"] == "churn_baseline"
    assert summary["model"] is None
    assert len(summary["head_commit"]) == 40

    by_path = {f["path"]: f for f in files}
    util = by_path["pkg/util.py"]
    assert util["churn_count"] == 4          # 1 initial + 3 fix commits
    assert util["bug_fix_commit_count"] == 3
    assert util["author_count"] == 2         # Ada and Grace
    assert util["in_degree"] == 2            # imported by app.py and core.py
    assert util["complexity_total"] == 2     # 1 + one `if`
    assert files[0]["path"] == "pkg/util.py" and files[0]["rank"] == 1
    assert files[0]["risk_score"] == 1.0
    assert [f["rank"] for f in files] == [1, 2, 3, 4]


def test_graph_edges_are_relative_and_correct(sample_repo, tmp_path):
    graph = analyze_repo(sample_repo, tmp_path / "x.joblib")["graph"]
    edges = {(e["source"], e["target"]) for e in graph["edges"]}
    assert ("pkg/core.py", "pkg/util.py") in edges
    assert ("app.py", "pkg/util.py") in edges
    assert ("app.py", "pkg/core.py") in edges
    assert all(not s.startswith("/") and not t.startswith("/") for s, t in edges)
    assert {n["id"] for n in graph["nodes"]} >= {"app.py", "pkg/util.py"}


def test_trained_model_is_used_when_present(sample_repo, dataset_csv, tmp_path):
    model_file = tmp_path / "models" / "risk.joblib"
    info = train_and_save(str(dataset_csv), model_file)
    assert info["n_files"] == 60 and info["repos"] == ["r1", "r2"]

    result = analyze_repo(sample_repo, model_file)
    assert result["summary"]["scoring_method"] == "random_forest"
    assert result["summary"]["model"]["n_files"] == 60
    scores = [f["risk_score"] for f in result["files"]]
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert scores == sorted(scores, reverse=True)


def test_missing_model_file_means_none_but_corrupt_one_is_an_error(tmp_path):
    assert load_model(tmp_path / "absent.joblib") is None
    bad = tmp_path / "bad.joblib"
    bad.write_text("not a model")
    with pytest.raises(ModelError):
        load_model(bad)


def test_corrupt_model_fails_the_analysis_clearly(sample_repo, tmp_path):
    bad = tmp_path / "bad.joblib"
    bad.write_text("not a model")
    with pytest.raises(AnalysisError, match="Retrain"):
        analyze_repo(sample_repo, bad)


def test_non_git_folder_is_rejected(tmp_path):
    with pytest.raises(AnalysisError, match="not a git repository"):
        analyze_repo(tmp_path, tmp_path / "m.joblib")


def test_file_limit_is_enforced(sample_repo, tmp_path):
    with pytest.raises(AnalysisError, match="limit"):
        analyze_repo(sample_repo, tmp_path / "m.joblib", max_files=2)


def test_training_refuses_a_dataset_with_too_few_risky_files(tmp_path):
    path = tmp_path / "tiny.csv"
    path.write_text(
        "repo,file_path,churn_count,author_count,bug_fix_commit_count,future_bug_fixes,risky\n"
        + "\n".join(f"r,f{i}.py,{i},1,0,0,0" for i in range(20)) + "\n"
    )
    with pytest.raises(ModelError, match="Too few"):
        train_and_save(str(path), tmp_path / "m.joblib")


@pytest.mark.xfail(
    reason="Known limitation of parsing/python_parser.py + graph/builder.py: "
           "`from pkg import core` and `from . import core` only record the module "
           "'pkg' / '.', so the edge goes to pkg/__init__.py instead of pkg/core.py.",
    strict=True,
)
def test_from_package_import_submodule_should_link_to_the_submodule(tmp_path):
    from codeatlas.graph.builder import build_dependency_graph
    from codeatlas.ingestion.repo_walker import parse_repository

    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "core.py").write_text("def f():\n    pass\n")
    (tmp_path / "a.py").write_text("from pkg import core\n")

    parsed = parse_repository(str(tmp_path))
    graph = build_dependency_graph(parsed.files, str(tmp_path))
    targets = graph.edges[str(tmp_path / "a.py")]
    assert str(tmp_path / "pkg" / "core.py") in targets
