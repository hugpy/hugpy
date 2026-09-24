"""GET /llm/model-metrics2 ?model= / ?suite= filters (additive; absent = every row)."""
import contextlib

import pytest
from flask import Flask

from hugpy_server.app.routes import metrics_routes


class _Cur:
    def __init__(self, log):
        self.log = log

    def execute(self, sql, params):
        self.log.append((sql, params))

    def fetchall(self):
        return []


class _DB:
    def __init__(self):
        self.log = []

    @contextlib.contextmanager
    def cursor(self):
        yield _Cur(self.log)


@pytest.fixture()
def client(monkeypatch):
    db = _DB()
    monkeypatch.setattr(metrics_routes, "_live_db", lambda: db)
    import hugpy_engine.model_index.client as mic
    monkeypatch.setattr(mic, "enabled", lambda: True)
    app = Flask(__name__)
    app.register_blueprint(metrics_routes.metrics_bp)
    return app.test_client(), db


def test_unfiltered_query_unchanged(client):
    c, db = client
    assert c.get("/llm/model-metrics2").status_code == 200
    sql, params = db.log[-1]
    assert "WHERE" not in sql and params == (500,)


def test_model_and_suite_filters(client):
    c, db = client
    r = c.get("/llm/model-metrics2?model=Qwen~Qwen2.5_VL&suite=integrity&limit=9")
    assert r.status_code == 200 and r.get_json()["rows"] == []
    sql, params = db.log[-1]
    assert "model_name = ANY(%s) OR model_name LIKE %s" in sql and "grade_suite = %s" in sql
    assert params[0] == ["Qwen~Qwen2.5_VL", "Qwen2.5_VL"]
    assert params[1] == "%~Qwen2.5\\_VL"          # LIKE metacharacters escaped
    assert params[2:] == ("integrity", 9)


def test_throughput_is_the_mean_of_calls_never_the_ema():
    # The metrics CARD still carries an EMA field (tok_per_s_avg 91.7); the
    # throughput block must ignore it and report Σtok/Σgen-s over the call
    # ledger (mean_tok_s 71.2 from the stats), labelled by call counts.
    rows = [{"model_name": "M", "worker": "aeb", "quant": "Q4", "alloc_mode": "gpu_only",
             "tok_per_s": 91.7, "tok_per_s_avg": 91.7, "n_samples": 1},
            {"model_name": "N", "worker": "aeb", "quant": "", "alloc_mode": ""}]
    stats = [{"model_name": "M", "worker": "aeb", "quant": "Q4", "alloc_mode": "gpu_only", "n_calls": 20,
              "n_rated": 18, "mean_tok_s": 71.2, "p50": 70.0, "p90": 80.0, "min": 40.0, "max": 88.0,
              "first_at": 1.0, "last_at": 2.0},
             {"model_name": "M", "worker": None, "n_calls": 20, "n_rated": 18, "mean_tok_s": 71.2}]
    by_model = metrics_routes.attach_throughput(rows, stats)
    tp0 = rows[0]["throughput"]
    # mean-of-calls (Σ/Σ), NEVER the 91.7 EMA on the card.
    assert tp0["mean_tok_s"] == 71.2 and tp0["n_calls"] == 20 and tp0["n_rated"] == 18
    assert tp0["label"] == "Σtok/Σgen-s over 18 of 20 calls"
    assert tp0["mean_tok_s"] != 91.7 and "ema" not in str(tp0).lower()
    tp = rows[1]["throughput"]
    assert tp["n_calls"] == 0 and tp["mean_tok_s"] is None
    assert tp["reason"].startswith("no calls recorded in model_calls for N on aeb")   # scoped, never bare
    assert by_model["M"]["n_calls"] == 20


def test_suites_endpoint_lists_the_registry(client):
    c, _db = client
    d = c.get("/llm/benchmark/suites").get_json()
    by = {s["name"]: s for s in d["suites"]}
    assert by["hugpy-native-v2"]["max"] == 27 and len(by["hugpy-native-v2"]["tasks"]) == 9
    assert by["hugpy-vision-v1"]["max"] == 18 and by["hugpy-vision-v1"]["tasks"][0] == "color"
    assert by["hugpy-vision-v1"]["items"]["count"][0]["expected"] == "first number == 2"
    assert by["hugpy-native-v2"]["items"]["math"][0]["expected"] == "last number == 37"
    assert by["fleet-capacity-v1"]["legacy"] is True and by["fleet-capacity-v1"]["max"] == 9
