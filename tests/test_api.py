import shutil
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from codeatlas.api.main import create_app
from codeatlas.config import Settings
from codeatlas.services.jobs import JobManager
from codeatlas.services.repo_fetcher import CloneError


def _make_client(tmp_path, sample_repo, fetcher=None):
    settings = Settings(data_dir=tmp_path / "data", model_path=tmp_path / "none.joblib")

    def fake_clone(owner, name, dest: Path, timeout):
        shutil.copytree(sample_repo, dest)  # stands in for `git clone`, no network

    app = create_app(settings)
    app.state.jobs.shutdown()
    app.state.jobs = JobManager(settings, fetcher=fetcher or fake_clone)
    return TestClient(app)


def _wait_until_finished(client, owner="pallets", name="flask", timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/repos/{owner}/{name}").json()
        if body["status"] in ("done", "failed"):
            return body
        time.sleep(0.1)
    raise AssertionError("analysis did not finish in time")


def test_health_and_docs():
    client = TestClient(create_app(Settings()))
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/docs").status_code == 200


def test_full_flow_start_poll_files_graph(tmp_path, sample_repo):
    client = _make_client(tmp_path, sample_repo)

    start = client.post("/api/repos", json={"url": "https://github.com/pallets/flask"})
    assert start.status_code == 202
    assert start.json()["owner"] == "pallets"

    final = _wait_until_finished(client)
    assert final["status"] == "done"
    assert final["summary"]["n_files"] == 4
    assert final["summary"]["owner"] == "pallets"

    files = client.get("/api/repos/pallets/flask/files").json()
    assert files["total"] == 4 and files["scoring_method"] == "churn_baseline"
    assert files["items"][0]["path"] == "pkg/util.py"

    page = client.get("/api/repos/pallets/flask/files?limit=1&offset=1").json()
    assert len(page["items"]) == 1 and page["items"][0]["rank"] == 2

    graph = client.get("/api/repos/pallets/flask/graph").json()
    assert {"source": "pkg/core.py", "target": "pkg/util.py"} in graph["edges"]

    # lower-case URL resolves to the same repo (GitHub names are case-insensitive)
    assert client.get("/api/repos/PALLETS/Flask").json()["status"] == "done"


def test_results_survive_a_restart(tmp_path, sample_repo):
    client = _make_client(tmp_path, sample_repo)
    client.post("/api/repos", json={"url": "https://github.com/pallets/flask"})
    _wait_until_finished(client)

    fresh = _make_client(tmp_path, sample_repo)  # new app, same data folder
    body = fresh.get("/api/repos/pallets/flask").json()
    assert body["status"] == "done" and body["summary"]["n_files"] == 4


@pytest.mark.parametrize("url", [
    "not a url", "http://github.com/pallets/flask", "https://example.com/a/b",
    "https://github.com/pallets/flask --upload-pack=x", "file:///etc/passwd",
])
def test_bad_urls_get_422(tmp_path, sample_repo, url):
    client = _make_client(tmp_path, sample_repo)
    response = client.post("/api/repos", json={"url": url})
    assert response.status_code == 422
    assert "github" in response.json()["detail"].lower()


def test_unknown_repo_is_404(tmp_path, sample_repo):
    client = _make_client(tmp_path, sample_repo)
    for suffix in ("", "/files", "/graph"):
        assert client.get(f"/api/repos/nobody/nothing{suffix}").status_code == 404


def test_path_tricks_are_rejected(tmp_path, sample_repo):
    # The HTTP client may collapse "/.." before sending, so 404/405 are fine too;
    # what matters is that none of these is ever treated as a valid repository.
    client = _make_client(tmp_path, sample_repo)
    for path in ("/api/repos/../etc", "/api/repos/a/..", "/api/repos/a/b%2Fc",
                 "/api/repos/a/%2e%2e", "/api/repos/..%2f..%2fetc/passwd"):
        assert client.get(path).status_code in (404, 405, 422), path
    assert client.get("/api/repos/a/%2e%2e/files").status_code in (404, 422)
    assert not (tmp_path / "data").exists()  # nothing was created on disk


def test_files_before_finished_is_409(tmp_path, sample_repo):
    def slow_clone(owner, name, dest, timeout):
        time.sleep(1.0)
        shutil.copytree(sample_repo, dest)

    client = _make_client(tmp_path, sample_repo, fetcher=slow_clone)
    client.post("/api/repos", json={"url": "https://github.com/pallets/flask"})
    response = client.get("/api/repos/pallets/flask/files")
    assert response.status_code == 409
    _wait_until_finished(client)


def test_clone_failure_is_reported_not_crashed(tmp_path, sample_repo):
    def failing_clone(owner, name, dest, timeout):
        raise CloneError("Could not clone the repository.")

    client = _make_client(tmp_path, sample_repo, fetcher=failing_clone)
    client.post("/api/repos", json={"url": "https://github.com/pallets/flask"})
    final = _wait_until_finished(client)
    assert final["status"] == "failed" and "clone" in final["error"].lower()
    assert client.get("/api/repos/pallets/flask/files").status_code == 409


def test_unexpected_bug_does_not_leak_details(tmp_path, sample_repo):
    def buggy(owner, name, dest, timeout):
        raise RuntimeError("secret internal path /srv/x")

    client = _make_client(tmp_path, sample_repo, fetcher=buggy)
    client.post("/api/repos", json={"url": "https://github.com/pallets/flask"})
    final = _wait_until_finished(client)
    assert final["status"] == "failed" and "secret" not in final["error"]


def test_same_repo_twice_returns_the_running_job(tmp_path, sample_repo):
    def slow_clone(owner, name, dest, timeout):
        time.sleep(0.5)
        shutil.copytree(sample_repo, dest)

    client = _make_client(tmp_path, sample_repo, fetcher=slow_clone)
    a = client.post("/api/repos", json={"url": "https://github.com/pallets/flask"})
    b = client.post("/api/repos", json={"url": "https://github.com/Pallets/Flask"})
    assert a.status_code == b.status_code == 202
    assert len(client.app.state.jobs._jobs) == 1
    _wait_until_finished(client)


def test_cors_allows_the_dev_ui_origin(tmp_path, sample_repo):
    client = _make_client(tmp_path, sample_repo)
    response = client.get("/health", headers={"Origin": "http://localhost:5173"})
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_web_ui_is_served_at_root_and_api_routes_still_win(tmp_path, sample_repo):
    client = _make_client(tmp_path, sample_repo)
    page = client.get("/")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"]
    assert "CodeAtlas AI" in page.text
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/api/repos/nobody/nothing").status_code == 404
