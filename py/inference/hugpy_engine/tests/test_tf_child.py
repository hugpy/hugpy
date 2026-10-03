"""Env-profiles stage 3: the transformers slot child (tf_child.py), driven with
a fake engine over real HTTP — no weights, no torch."""
import importlib.util
import json
import os
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

_PATH = os.path.join(os.path.dirname(__file__), "..", "src", "hugpy_engine", "serve", "tf_child.py")
spec = importlib.util.spec_from_file_location("tf_child_under_test", _PATH)
tf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tf)


class FakeEngine:
    n_ctx_train = 64
    last_completion_n = 0

    def render(self, messages, kw):
        return " ".join(m["content"] for m in messages)

    def encode(self, prompt):
        return prompt.split()

    def generate(self, ids, max_new, params, cancel):
        out = ["Hel", "lo ", "wor", "ld", " STOP", " more"][:max_new]
        self.last_completion_n = 0
        for p in out:
            if cancel.is_set():
                break
            self.last_completion_n += 1
            yield p


@pytest.fixture
def base():
    srv = tf.ChatServer(FakeEngine(), "M", "/models/M", ctx=None)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), tf.make_handler(srv))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", srv
    httpd.shutdown()


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, json.loads(r.read() or b"{}")


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def test_docs(base):
    url, _ = base
    assert _get(url + "/health")[0] == 200
    assert _get(url + "/v1/models")[1]["data"][0]["id"] == "M"
    props = _get(url + "/props")[1]
    assert props["model_path"] == "/models/M" and props["engine"] == "transformers"


def test_non_stream_with_stop_usage_timings(base):
    url, _ = base
    st, raw = _post(url + "/v1/chat/completions",
                    {"model": "M", "messages": [{"role": "user", "content": "hi there"}],
                     "max_tokens": 6, "stop": [" STOP"]})
    d = json.loads(raw)
    assert st == 200 and d["choices"][0]["message"]["content"] == "Hello world"
    assert d["choices"][0]["finish_reason"] == "stop"
    assert d["usage"]["prompt_tokens"] == 2 and d["timings"]["prompt_n"] == 2


def test_stream_sse_ends_with_usage_and_done(base):
    url, _ = base
    st, raw = _post(url + "/v1/chat/completions",
                    {"model": "M", "messages": [{"role": "user", "content": "x"}], "max_tokens": 3,
                     "stream": True})
    events = [ln[6:] for ln in raw.splitlines() if ln.startswith("data: ")]
    assert st == 200 and events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert text == "Hello wor" and chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert chunks[-1]["usage"]["completion_tokens"] == 3 and "timings" in chunks[-1]


def test_prompt_over_context_is_a_400(base):
    url, srv = base
    srv.ctx = 3
    st, raw = _post(url + "/v1/chat/completions",
                    {"model": "M", "messages": [{"role": "user", "content": "a b c d e"}]})
    assert st == 400 and "exceed" in raw
