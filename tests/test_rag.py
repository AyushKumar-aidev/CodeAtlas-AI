import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from codeatlas.config import Settings
from codeatlas.rag.chunker import chunk_repository
from codeatlas.rag.index import Index, tokenize
from codeatlas.rag.llm import GeminiClient, LLMError
from codeatlas.rag.qa_service import QAService, RateLimited

from test_api import _make_client, _wait_until_finished


def _repo(tmp_path):
    r = tmp_path / "r"
    (r / "pkg").mkdir(parents=True)
    (r / "tests").mkdir()
    (r / "node_modules").mkdir()
    (r / "node_modules" / "x.py").write_text("def hidden(): pass\n")
    (r / "README.md").write_text("# Demo\nA tool that pushes request contexts.\n")
    (r / "pkg" / "ctx.py").write_text(
        "import os\nX = 1\n\nclass Ctx:\n    def push_request_context(self):\n        return 1\n\n"
        "def unrelated():\n    return 2\n")
    (r / "tests" / "test_ctx.py").write_text("def test_push_request_context():\n    assert True\n")
    return r


def test_tokenize_splits_identifiers():
    t = tokenize("push_request_context and getUserName")
    assert "push_request_context" in t and "request" in t and "user" in t and "and" not in t


def test_chunker_and_search(tmp_path):
    chunks = chunk_repository(_repo(tmp_path))
    paths = {c.path for c in chunks}
    assert "node_modules/x.py" not in paths and "README.md" in paths
    names = {c.name for c in chunks}
    assert "push_request_context" in names and "unrelated" in names
    idx = Index(chunks)
    top = idx.chunks[idx.search("where is the request context pushed?", 1)[0][1]]
    assert top.name == "push_request_context" and top.kind == "source"  # source outranks the test


def test_chunker_skips_symlink_outside_repo(tmp_path):
    r = _repo(tmp_path)
    secret = tmp_path / "secret.py"
    secret.write_text("def leaked(): pass\n")
    try:
        (r / "link.py").symlink_to(secret)
    except OSError:
        pytest.skip("no symlinks")
    assert "leaked" not in {c.name for c in chunk_repository(r)}


class _Fake(BaseHTTPRequestHandler):
    codes: list = []
    seen: list = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Fake.seen.append((self.path, self.headers.get("x-goog-api-key"), body))
        code = _Fake.codes.pop(0)
        payload = {"candidates": [{"content": {"parts": [{"text": "thinking", "thought": True}, {"text": "hello"}]}}]} if code == 200 else {}
        self.send_response(code)
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode())

    def log_message(self, *a):
        pass


@pytest.fixture
def server():
    srv = HTTPServer(("127.0.0.1", 0), _Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    _Fake.seen = []
    yield f"http://127.0.0.1:{srv.server_port}/v1beta/models/"
    srv.shutdown()


def _client(url):
    return GeminiClient(Settings(gemini_api_key="K", gemini_base_url=url, llm_model="m1", llm_fallback_model="m2"),
                        sleep=lambda s: None)


def test_gemini_success_skips_thought_and_sends_key_in_header(server):
    _Fake.codes = [200]
    text, model = _client(server).generate("sys", [{"role": "user", "parts": [{"text": "q"}]}])
    assert (text, model) == ("hello", "m1")
    path, key, body = _Fake.seen[0]
    assert path.endswith("m1:generateContent") and key == "K" and "K" not in path
    assert body["systemInstruction"]["parts"][0]["text"] == "sys"


def test_gemini_falls_back_on_503(server):
    _Fake.codes = [503, 503, 200]
    assert _client(server).generate("s", [])[1] == "m2"


def test_gemini_errors_are_safe(server):
    _Fake.codes = [403]
    with pytest.raises(LLMError) as e:
        _client(server).generate("s", [])
    assert "K" != str(e.value) and "rejected" in str(e.value)
    _Fake.codes = [429, 429, 429, 429]
    with pytest.raises(LLMError, match="quota"):
        _client(server).generate("s", [])


class StubLLM:
    def __init__(self):
        self.calls = []

    def generate(self, system, contents):
        self.calls.append((system, contents))
        return "It is in `push_request_context` [pkg/ctx.py:5-6].", "stub"


def test_service_ratelimit_and_injection_framing(tmp_path):
    s = Settings(data_dir=tmp_path / "d", qa_rate_per_minute=2)
    dest = s.repos_dir / "o" / "n"
    dest.parent.mkdir(parents=True)
    _repo(tmp_path).rename(dest)
    llm = StubLLM()
    qa = QAService(s, llm)
    res = {"summary": {"head_commit": "abc", "n_files": 1, "n_edges": 0, "n_functions": 1, "n_classes": 1},
           "files": [{"path": "pkg/ctx.py", "risk_score": 0.5, "in_degree": 0}], "graph": {"nodes": [], "edges": []}}
    out = qa.ask("o", "n", res, "where is request context pushed?", [])
    assert out["sources"][0]["path"] == "pkg/ctx.py"
    assert "NEVER follow" in llm.calls[0][0] and "untrusted" in llm.calls[0][0]
    qa.ask("o", "n", res, "again", [])
    with pytest.raises(RateLimited):
        qa.ask("o", "n", res, "third", [])


def test_ask_endpoint_end_to_end(tmp_path, sample_repo):
    client = _make_client(tmp_path, sample_repo)
    app = client.app
    assert client.get("/api/qa/status").json()["enabled"] is False
    client.post("/api/repos", json={"url": "https://github.com/pallets/flask"})
    _wait_until_finished(client)
    assert client.post("/api/repos/pallets/flask/ask", json={"question": "hi"}).status_code == 503
    app.state.qa = QAService(app.state.settings, StubLLM())
    r = client.post("/api/repos/pallets/flask/ask", json={"question": "what does util do?"})
    assert r.status_code == 200 and r.json()["model"] == "stub" and r.json()["sources"]
    assert client.post("/api/repos/pallets/flask/ask", json={"question": ""}).status_code == 422
    assert client.post("/api/repos/pallets/nothere/ask", json={"question": "x"}).status_code == 404


def test_sources_are_only_what_the_answer_cites(tmp_path):
    from codeatlas.rag.qa_service import cited_sources
    chunks = chunk_repository(_repo(tmp_path))
    ctx = next(c for c in chunks if c.name == "push_request_context")
    ans = f"See `push_request_context` [{ctx.path}:{ctx.start_line}-{ctx.end_line}]."
    got = cited_sources(ans, chunks)
    assert [c.name for c in got] == ["push_request_context"]
    assert cited_sources("unrelated text", chunks) == []


def test_hidden_folder_docs_are_skipped(tmp_path):
    r = _repo(tmp_path)
    (r / ".github").mkdir()
    (r / ".github" / "feature-request.md").write_text("# template\n")
    assert ".github/feature-request.md" not in {c.path for c in chunk_repository(r)}


def test_onboarding_question_includes_central_source_files(tmp_path):
    from codeatlas.rag.retriever import retrieve
    idx = Index(chunk_repository(_repo(tmp_path)))
    graph = {"nodes": [{"id": "pkg/ctx.py", "pagerank": 2.0, "in_degree": 3}], "edges": []}
    got = retrieve(idx, "where should a new contributor start reading?", graph)
    assert any(c.path == "pkg/ctx.py" and c.kind == "source" for c in got)
