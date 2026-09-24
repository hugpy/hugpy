"""Task-keyed grading suites: registry, checkers, call shapes, dispatch."""
import base64
import json
import io
import threading
import time

import pytest

PIL = pytest.importorskip("PIL")
pytest.importorskip("numpy")
from PIL import Image, ImageDraw  # noqa: E402

from hugpy_curation.review import fleet_grading  # noqa: E402
from hugpy_curation.review.suites import (  # noqa: E402
    SUITES, suite_by_name, suite_for_model, imagegen, vision)
from hugpy_curation.review.suites.common import choice, number_is  # noqa: E402
from hugpy_platform.constants import DEFAULT_AGENT_BRAIN  # noqa: E402


# ------------------------------------------------------------- registry ----
def test_suite_selection_by_task():
    assert suite_for_model({"primary_task": "text-generation"}).name == "hugpy-native-v2"
    assert suite_for_model({"primary_task": "text-summarization"}).name == "hugpy-native-v2"
    assert suite_for_model({"primary_task": "image-text-to-text"}).name == "hugpy-vision-v1"
    assert suite_for_model({"primary_task": "text-to-image"}).name == "hugpy-imagegen-v1"
    assert suite_for_model({"tasks": ["image-text-to-text", "text-generation"]}).name == "hugpy-vision-v1"
    for task in ("text-to-video", "image-to-video", "automatic-speech-recognition", "text-to-speech",
                 "feature-extraction", "sentence-similarity", "object-detection", "depth-estimation",
                 "image-classification", "image-segmentation", "image-to-image", "needs-classification"):
        assert suite_for_model({"primary_task": task, "tasks": [task]}) is None, task
    assert suite_by_name("hugpy-vision-v1") is SUITES["image-text-to-text"]
    with pytest.raises(KeyError):
        suite_by_name("nope")


def test_every_suite_has_three_tiers_and_enough_tasks():
    for name, minimum in (("hugpy-native-v2", 9), ("hugpy-vision-v1", 6), ("hugpy-imagegen-v1", 6)):
        suite = suite_by_name(name)
        assert len(suite.tasks) >= minimum
        for tiers in suite.tasks.values():
            assert [t[0] for t in tiers] == ["easy", "medium", "hard"]
        assert suite.max == 3 * len(suite.tasks)
    assert suite_by_name("hugpy-native-v2").tasks is fleet_grading.TASKS_TIERED


# -------------------------------------------------------- vision checkers ----
EXPECTED_VISION = {
    ("color", "easy"): ("Red.", "Blue"), ("color", "medium"): ("yellow", "blue"),
    ("color", "hard"): ("Orange", "green"), ("count", "easy"): ("2", "3"),
    ("count", "medium"): ("three", "5"), ("count", "hard"): ("9", "8"),
    ("spatial", "easy"): ("left", "right"), ("spatial", "medium"): ("Above.", "below"),
    ("spatial", "hard"): ("Below the blue circle is a triangle", "circle"),
    ("read", "easy"): ("CAT", "DOG"), ("read", "medium"): ("4827", "4872"),
    ("read", "hard"): ("Open at nine", "open at noon"), ("shape", "easy"): ("circle", "square"),
    ("shape", "medium"): ("Triangle", "square"), ("shape", "hard"): ("hexagon", "circle"),
    ("compare", "easy"): ("right", "left"), ("compare", "medium"): ("left", "right"),
    ("compare", "hard"): ("right", "left"),
}


def test_vision_images_deterministic_and_checkers():
    for task, tiers in vision.TASKS.items():
        for tier, spec, checker in tiers:
            a, b = vision.png_bytes(spec["image"]()), vision.png_bytes(spec["image"]())
            assert a == b and Image.open(io.BytesIO(a)).size == (vision.SIZE, vision.SIZE)
            good, bad = EXPECTED_VISION[(task, tier)]
            assert vision.score({"answer": good}, checker), (task, tier, good)
            assert not vision.score({"answer": bad}, checker), (task, tier, bad)


def test_choice_first_mention_wins():
    check = choice("left", ("left", "right"))
    assert check("Left.") and not check("right") and not check("right, not left") and not check("")
    assert number_is(3)("There are 3 circles") and not number_is(3)("13")


# ------------------------------------------------------ imagegen checkers ----
def _img(size=(512, 512), bg="white"):
    image = Image.new("RGB", size, bg)
    return image, ImageDraw.Draw(image)


def _shape(kind, fill="black", bg="white"):
    image, draw = _img(bg=bg)
    if kind == "square": draw.rectangle((156, 156, 356, 356), fill=fill)
    elif kind == "circle": draw.ellipse((136, 136, 376, 376), fill=fill)
    elif kind == "triangle": draw.polygon([(256, 136), (136, 376), (376, 376)], fill=fill)
    return image


def _scene_photo(seed=0):
    """A smooth, structured 'photo': gradients + blobs (not noise, not flat)."""
    import numpy as np
    y, x = np.mgrid[0:512, 0:512] / 512.0
    r = 128 + 100 * np.sin(6 * x + seed) * np.cos(4 * y)
    g = 40 + 200 * y
    b = 128 + 120 * np.cos(9 * x * y + seed)
    return Image.fromarray(np.clip(np.dstack([r, g, b]), 0, 255).astype("uint8"))


def _noise():
    import numpy as np
    return Image.fromarray(np.random.default_rng(1).integers(0, 256, (512, 512, 3), dtype=np.uint8))


def _stripes(colors, vertical):
    image, draw = _img()
    n = len(colors)
    for i, c in enumerate(colors):
        box = (i * 512 // n, 0, (i + 1) * 512 // n, 512) if vertical else (0, i * 512 // n, 512, (i + 1) * 512 // n)
        draw.rectangle(box, fill=c)
    return image


def _check(task, tier):
    return {t: c for t, _s, c in imagegen.TASKS[task]}[tier]


def test_imagegen_checkers_good_and_bad():
    flat = Image.new("RGB", (512, 512), (128, 128, 128))
    photo = _scene_photo()
    cases = {
        ("render", "easy"): (photo, flat), ("render", "medium"): (photo, _noise()),
        ("render", "hard"): (photo, _shape("square")),
        ("size", "easy"): (photo, photo.resize((768, 512))),
        ("size", "medium"): (photo.resize((768, 512)), photo),
        ("size", "hard"): (photo.resize((512, 768)), photo),
        ("color", "easy"): (_shape("square", (220, 20, 20)), _shape("square", (20, 20, 220))),
        ("color", "medium"): (_shape("circle", (30, 60, 220)), _shape("circle", (30, 60, 220), bg=(0, 0, 0))),
        ("layout", "easy"): (_stripes(["black", "white"], True), _stripes(["white", "black"], True)),
        ("layout", "medium"): (_stripes([(30, 60, 220), (30, 160, 60)], True), _stripes([(30, 160, 60), (30, 60, 220)], True)),
        ("layout", "hard"): (_stripes([(220, 20, 20), "white", (30, 60, 220)], False),
                             _stripes([(30, 60, 220), "white", (220, 20, 20)], False)),
        ("contrast", "easy"): (_shape("circle"), _shape("circle", "white", bg="black")),
        ("contrast", "medium"): (_shape("circle", "white", bg="black"), _shape("circle")),
        ("contrast", "hard"): (_shape("square", (70, 70, 70), bg=(200, 200, 200)), _shape("square", (190, 190, 190), bg=(200, 200, 200))),
        ("shape", "easy"): (_shape("square"), _shape("circle")),
        ("shape", "medium"): (_shape("circle"), _shape("square")),
        ("shape", "hard"): (_shape("triangle"), _shape("circle")),
    }
    star_good = _img(bg=(30, 160, 60))[0]; ImageDraw.Draw(star_good).regular_polygon((256, 256, 60), 5, fill=(240, 210, 20))
    cases[("color", "hard")] = (star_good, Image.new("RGB", (512, 512), (30, 160, 60)))
    assert {k for k in cases} == {(t, tier) for t, tiers in imagegen.TASKS.items() for tier, _s, _c in tiers}
    for (task, tier), (good, bad) in cases.items():
        checker = _check(task, tier)
        assert imagegen.score({"answer": good}, checker), (task, tier, "good", checker.rule)
        assert not imagegen.score({"answer": bad}, checker), (task, tier, "bad", checker.rule)
        assert not imagegen.score({"answer": None}, checker)


def test_imagegen_stats_flags_noise_and_flat():
    assert imagegen.stats(_noise())["noise_ratio"] > 0.9
    assert imagegen.stats(_scene_photo())["noise_ratio"] < 0.5
    assert not imagegen.not_degenerate(Image.new("RGB", (512, 512), "red"))
    assert imagegen.decode("not base64 at all") is None


# ------------------------------------------------------------ call shapes ----
LANE = {"model": "org/vl", "worker": "aeb", "worker_id": "w1", "quant": "q"}
VARIATION = ("standard", "gpu_only", {}, True, None, None)


class FakeClient:
    def __init__(self, responder):
        self.calls, self.responder = [], responder

    def request(self, path, method="GET", body=None, timeout=None):
        self.calls.append((path, method, body))
        return self.responder(path, body)


def _png_b64(image):
    buffer = io.BytesIO(); image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def test_vision_call_carries_base64_data_url():
    client = FakeClient(lambda p, b: {"ok": True, "text": "red",
                                      "usage": {"completion_tokens": 1, "prompt_tokens": 300}})
    spec = vision.TASKS["color"][0][1]
    out = vision.call(client, LANE, VARIATION, spec, 16)
    path, method, body = client.calls[0]
    # /v1 downcasts content arrays to str, so the image rides /ml/vision's ``images``.
    assert (path, method) == ("/ml/vision", "POST")
    url = body["images"][0]
    assert url.startswith("data:image/png;base64,")
    decoded = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert decoded.size == (vision.SIZE, vision.SIZE)
    assert "What color" in body["messages"][0]["content"] and isinstance(body["messages"][0]["content"], str)
    assert body["alloc"] == {"worker": "aeb", "alloc_mode": "gpu_only"} and body["temperature"] == 0
    assert body["max_new_tokens"] == 16 and body["model"] == "org/vl"
    assert out["answer"] == "red" and out["error"] is None


def test_vision_error_payload_is_an_error():
    client = FakeClient(lambda p, b: {"ok": False, "error": "no projector"})
    out = vision.call(client, LANE, VARIATION, vision.TASKS["color"][0][1], 16)
    assert out["answer"] == "" and "no projector" in out["error"]


def test_imagegen_call_uses_imagine_shape():
    good = _shape("square", (220, 20, 20))
    client = FakeClient(lambda p, b: {"ok": True, "images": [{"b64": _png_b64(good), "path": "/x.png",
                                                              "width": 512, "height": 512, "seed": 1234}]})
    spec = imagegen.TASKS["color"][0][1]
    out = imagegen.call(client, LANE, VARIATION, spec)
    path, method, body = client.calls[0]
    assert (path, method) == ("/ml/imagine", "POST")
    assert body["model"] == "org/vl" and "alloc" not in body     # ImageGenRequest has no alloc
    assert body["prompt"] == spec["prompt"] and body["return_b64"] is True and body["num_images"] == 1
    assert (body["width"], body["height"], body["seed"], body["num_inference_steps"]) == (512, 512, 1234, 20)
    assert out["error"] is None and out["answer"].size == (512, 512) and out["image_bytes"] > 0
    assert out["s_per_image"] is not None and imagegen.score(out, imagegen.TASKS["color"][0][2])
    # ImageGenRequest must accept every key we send (``model`` folds to model_key).
    schemas = pytest.importorskip("hugpy_media.imagegen.schemas")
    fields = set(schemas.ImageGenRequest.model_fields)
    assert set(body) - {"model"} <= fields


def test_imagegen_error_payload_is_error_not_fail():
    client = FakeClient(lambda p, b: {"ok": False, "error": "boom", "images": []})
    out = imagegen.call(client, LANE, VARIATION, imagegen.TASKS["size"][0][1])
    assert out["answer"] is None and "boom" in out["error"]


# --------------------------------------------------------------- dispatch ----
def _worker():
    return {"id": "w1", "name": "aeb", "status": "online", "admission": "approved", "serve_mode": "on",
            "max_vram_bytes": 10**11, "max_ram_bytes": 10**11, "loaded_models": []}


def _catalog(model, task="text-generation"):
    return {"model_key": model, "primary_task": task, "tasks": [task], "framework": "gguf",
            "size_bytes": 10**9, "workers": [{"worker_id": "w1", "designated": True}]}


def _run(catalog, responder, **kw):
    events = []

    def route(path, body):
        if path.startswith("/models"): return {"models": catalog}
        if path.startswith("/llm/serving/"): return {}
        if path.startswith("/llm/workers/"): return {"ok": True}
        return responder(path, body)
    client = FakeClient(route)
    fleet_grading.run_capacity_benchmark(client, [_worker()], 32, threading.Event(),
                                         lambda kind, value: events.append((kind, value)), **kw)
    return client, events


def test_text_detail_shape_unchanged():
    client, events = _run([_catalog("org/txt", "text-generation")],
                          lambda p, b: {"choices": [{"message": {"content": "37"}}]})
    results = [v for k, v in events if k == "result" and v.get("status") == "complete"]
    assert results and results[0]["max"] == 27 and "grade_suite" not in results[0]
    for task, entry in results[0]["detail"].items():
        # judge-reviewed shape: per-task entry gained format/revised counts.
        assert set(entry) == {"tier", "max", "format", "revised", "history"} and entry["max"] == 3
        assert [h["tier"] for h in entry["history"]] == ["easy", "medium", "hard"]
    bodies = [b for p, _m, b in client.calls if p == "/v1/chat/completions"]
    assert all(isinstance(b["messages"][0]["content"], str) for b in bodies)


def test_vision_dispatch_detail_shape_matches_text():
    _client, events = _run([_catalog("org/vl", "image-text-to-text")],
                           lambda p, b: {"ok": True, "text": "red"})
    results = [v for k, v in events if k == "result" and v.get("status") == "complete"]
    assert results and results[0]["grade_suite"] == "hugpy-vision-v1" and results[0]["max"] == 18
    assert results[0]["grade_task"] == "image-text-to-text"
    first = results[0]["detail"]["color"]["history"][0]
    assert {k: first[k] for k in ("tier", "pass")} == {"tier": "easy", "pass": True}   # + expected/actual (additive)
    assert first["actual"] == "red" and first["expected"].startswith("first of") and "why" not in first
    for entry in results[0]["detail"].values():
        # SAME unified shape as text (test_text_detail_shape_unchanged): the
        # judge-review fields (format/revised) are shared across suites.
        assert set(entry) == {"tier", "max", "format", "revised", "history"}
    calls = [v for k, v in events if k == "call"]
    assert len(calls) == 18 and {c["grade"] for c in calls} <= {"PASS", "FAIL"}


def test_imagegen_dispatch_one_row_per_model_and_throughput():
    red = _png_b64(_shape("square", (220, 20, 20)))
    client, events = _run([_catalog("org/sd", "text-to-image")],
                          lambda p, b: time.sleep(0.002) or {"ok": True, "images": [{"b64": red}]})
    results = [v for k, v in events if k == "result"]
    assert len(results) == 1
    done = results[0]
    assert done["status"] == "complete" and done["worker"] == "hugpy-placed" and done["grade_suite"] == "hugpy-imagegen-v1"
    assert done["s_per_image"] is not None and done["images_per_s"] is not None
    assert done["detail"]["color"]["history"][0]["pass"] is True
    paths = [p for p, _m, _b in client.calls]
    assert "/ml/imagine" in paths and not any(p.startswith("/llm/workers/") for p in paths)  # no cold reset
    assert all("alloc" not in b for p, _m, b in client.calls if p == "/ml/imagine")


def test_no_suite_models_are_skipped_not_graded():
    client, events = _run([_catalog("org/vid", "text-to-video")], lambda p, b: pytest.fail("no inference"))
    results = [v for k, v in events if k == "result"]
    assert len(results) == 1
    row = results[0]
    assert (row["model"], row["status"], row["grade"], row["failure_class"], row["reason"]) == (
        "org/vid", "skipped", None, "no_suite", "no suite for task text-to-video")
    assert row["persist"] is False
    assert not any(p.startswith("/llm/workers/") for p, _m, _b in client.calls)


def test_forced_suite_and_optional_judge():
    red = _png_b64(_shape("square", (220, 20, 20)))
    catalog = [_catalog("org/sd", "text-to-image"),
               {**_catalog("org/judge", "image-text-to-text"), "mmproj_bytes": 1, "workers": []}]
    worker = _worker(); worker["loaded_models"] = ["org/judge"]

    def responder(path, body):
        if path == "/ml/imagine": return {"ok": True, "images": [{"b64": red}]}
        return {"ok": True, "text": "yes"}
    for with_judge in (False, True):
        events = []
        client = FakeClient(lambda p, b: {"models": catalog} if p.startswith("/models") else
                            {} if p.startswith("/llm/serving/") else {"ok": True} if p.startswith("/llm/workers/") else responder(p, b))
        fleet_grading.run_capacity_benchmark(client, [worker], 32, threading.Event(),
                                             lambda k, v: events.append((k, v)), model_ids=["org/sd"],
                                             with_judge=with_judge)
        done = [v for k, v in events if k == "result" and v.get("status") == "complete"][0]
        assert done["judge"]["model"] == DEFAULT_AGENT_BRAIN and done["judge"]["counted"] is with_judge
        assert ("judge_color" in done["detail"]) is with_judge
        assert done["max"] == (36 if with_judge else 18)
        judge_bodies = [b for p, _m, b in client.calls if p == "/ml/vision"]
        assert judge_bodies and all(b["alloc"] == {"worker": "aeb"} for b in judge_bodies)


# ------------------------------------------------------ bounded run loop ----
from hugpy_curation.review.suites import common  # noqa: E402

FAST = {"call_s": 0.6, "cold_load_cap_s": 0.6, "model_s": 4.0}


def _two_workers():
    a = _worker(); b = {**_worker(), "id": "w2", "name": "cmp"}
    return [a, b]


def _catalog2(model, task="text-generation", size=10**9):
    return {**_catalog(model, task), "size_bytes": size,
            "workers": [{"worker_id": "w1", "designated": True}, {"worker_id": "w2", "designated": True}]}


def _run_bounded(catalog, responder, workers=None, stop=None, **kw):
    events = []

    def route(path, body):
        if path.startswith("/models"): return {"models": catalog}
        if path.startswith("/llm/serving/"): return {}
        if path.startswith("/llm/workers"): return {"ok": True} if "/" in path[len("/llm/workers"):] else {"workers": []}
        if path.startswith("/llm/model-metrics2"): return {"rows": []}
        return responder(path, body)
    client = FakeClient(route)
    started = time.monotonic()
    fleet_grading.run_capacity_benchmark(client, workers or [_worker()], 16, stop or fleet_grading.BenchmarkControl(),
                                         lambda kind, value: events.append((kind, value)),
                                         budgets=kw.pop("budgets", FAST), **kw)
    return client, events, time.monotonic() - started


def _results(events):
    return [v for k, v in events if k == "result"]


def test_never_answering_lane_times_out_and_run_continues():
    hang = threading.Event()

    def responder(path, body):
        if body and body.get("model") == "org/hang":
            hang.wait(30)            # never answers within the budget
        return {"choices": [{"message": {"content": "37"}}]}
    catalog = [_catalog("org/hang"), _catalog("org/ok")]
    catalog[0]["size_bytes"], catalog[1]["size_bytes"] = 1, 2
    _c, events, elapsed = _run_bounded(catalog, responder)
    hang.set()
    rows = _results(events)
    hung = [r for r in rows if r["model"] == "org/hang" and r.get("failure_class")]
    assert hung and all(r["failure_class"] == "timeout" for r in hung)
    assert "cold-load" in hung[0]["reason"] and hung[0]["evidence"]["phase"] == "cold-load"
    assert any(r["model"] == "org/ok" and r["status"] == "complete" for r in rows)
    assert elapsed < FAST["model_s"] + 6        # bounded by the model budget, not the hang
    assert any(v.get("remaining_budget_s") is not None for k, v in events if k == "progress")


def test_hard_load_failure_skips_every_remaining_lane_of_the_model():
    detail = json.dumps({"error": {"message": "load failed", "load_failure": {
        "class": "hard_load_failure", "loader_stderr": "gguf_init: bad magic\nmore"}}})

    def responder(path, body):
        if body["model"] == "org/bad":
            raise fleet_grading.FleetError("HTTP 500 at /v1/chat/completions: " + detail)
        return {"choices": [{"message": {"content": "37"}}]}
    client, events, _ = _run_bounded([_catalog2("org/bad"), _catalog2("org/good", size=2 * 10**9)], responder,
                                     workers=_two_workers())
    bad = [r for r in _results(events) if r["model"] == "org/bad"]
    assert bad[0]["failure_class"] == "hard_load_failure" and bad[0]["evidence"]["loader_stderr_first_line"] == "gguf_init: bad magic"
    skipped = bad[1:]
    assert skipped and all(r["reason"].startswith("skipped: hard_load_failure on aeb — gguf_init: bad magic") for r in skipped)
    # exactly one inference attempt for the bad model; the good model still graded
    assert sum(1 for p, _m, b in client.calls if p == "/v1/chat/completions" and b["model"] == "org/bad") == 1
    assert any(r["model"] == "org/good" and r["status"] == "complete" for r in _results(events))
    calls = [v for k, v in events if k == "call" and v.get("failure_class") == "hard_load_failure"]
    assert calls and calls[0]["task"] == "benchmark:hard_load_failure"


def test_vram_fit_skips_only_that_lane():
    def responder(path, body):
        if body["alloc"]["alloc_mode"] == "gpu_only":
            raise fleet_grading.FleetError("HTTP 500 at /v1/chat/completions: vram_fit: model does not fit")
        return {"choices": [{"message": {"content": "37"}}]}
    _c, events, _ = _run_bounded([_catalog("org/m")], responder)
    rows = _results(events)
    assert [r.get("failure_class") for r in rows if r["alloc_mode"] == "gpu_only"] == ["vram_fit"]
    assert any(r["alloc_mode"] == "ram_only" and r["status"] == "complete" for r in rows)


def test_unreachable_worker_is_skipped_for_the_rest_of_the_run():
    def responder(path, body):
        if body["alloc"]["worker"] == "aeb":
            raise ConnectionRefusedError("[Errno 111] Connection refused")
        return {"choices": [{"message": {"content": "37"}}]}
    catalog = [_catalog2("org/a", size=1), _catalog2("org/b", size=2)]
    client, events, _ = _run_bounded(catalog, responder, workers=_two_workers())
    aeb = [r for r in _results(events) if r["worker"] == "aeb"]
    assert all(r["failure_class"] == "unreachable" for r in aeb)
    assert any(r["reason"].startswith("skipped: worker aeb unreachable") for r in aeb if r["model"] == "org/b")
    attempts = [b for p, _m, b in client.calls if p == "/v1/chat/completions" and b.get("alloc", {}).get("worker") == "aeb"]
    assert len(attempts) == fleet_grading.UNREACHABLE_STRIKES
    assert any(r["worker"] == "cmp" and r["status"] == "complete" for r in _results(events))


def test_cold_load_capacity_retries_are_bounded(monkeypatch):
    monkeypatch.setattr(common, "CAPACITY_BACKOFF_S", (0.01,))
    tries = []

    def responder(path, body):
        tries.append(1)
        raise fleet_grading.FleetError('HTTP 503 at /v1/chat/completions: {"error": {"message": "cold_load_capacity: 2 loads in flight"}}')
    _c, events, elapsed = _run_bounded([_catalog("org/m")], responder, budgets={**FAST, "cold_load_cap_s": 5.0})
    rows = [r for r in _results(events) if r.get("failure_class")]
    assert rows and rows[0]["failure_class"] == "cold_load_capacity" and "exhausted" in rows[0]["reason"]
    assert len(tries) <= 2 * (common.CAPACITY_RETRIES + 1) and elapsed < 5


def test_cancel_is_honored_inside_a_waiting_call():
    control = fleet_grading.BenchmarkControl(); release = threading.Event()

    def responder(path, body):
        threading.Timer(0.3, control.set).start()
        release.wait(30)
        return {"choices": [{"message": {"content": "37"}}]}
    _c, events, elapsed = _run_bounded([_catalog("org/m")], responder, stop=control,
                                       budgets={**FAST, "cold_load_cap_s": 20.0, "model_s": 30.0})
    release.set()
    assert elapsed < 5
    assert [r["failure_class"] for r in _results(events) if r.get("failure_class")] == ["cancelled"]


def test_resume_skips_recently_graded_lanes_unless_forced():
    now = time.time()
    recent = {"rows": [{"model_name": "org/m", "quant": "runtime default", "alloc_mode": "gpu_only", "worker": "aeb",
                        "grade": 80.0, "graded_at": now - 3600}]}
    for force, expected in ((False, 1), (True, 0)):
        events = []

        def route(path, body):
            if path.startswith("/models"): return {"models": [_catalog("org/m")]}
            if path.startswith("/llm/model-metrics2"): return recent
            if path.startswith("/llm/serving/") or path.startswith("/llm/workers/"): return {}
            return {"choices": [{"message": {"content": "37"}}]}
        fleet_grading.run_capacity_benchmark(FakeClient(route), [_worker()], 16, fleet_grading.BenchmarkControl(),
                                             lambda k, v: events.append((k, v)), budgets=FAST, resume=True, force=force)
        resumed = [r for r in _results(events) if r.get("failure_class") == "resume"]
        assert len(resumed) == expected
        if resumed: assert resumed[0]["alloc_mode"] == "gpu_only" and resumed[0]["persist"] is False


def test_cheapest_and_hot_models_run_first():
    worker = _worker(); worker["loaded_models"] = ["org/hot-big"]
    catalog = [_catalog("org/big"), _catalog("org/small"), _catalog("org/hot-big")]
    catalog[0]["size_bytes"], catalog[1]["size_bytes"], catalog[2]["size_bytes"] = 9 * 10**10, 10**8, 9 * 10**10
    _c, events, _ = _run_bounded(catalog, lambda p, b: {"choices": [{"message": {"content": "37"}}]}, workers=[worker])
    order = list(dict.fromkeys(r["model"] for r in _results(events)))
    assert order == ["org/hot-big", "org/small", "org/big"]


def test_classify_reads_structured_evidence():
    cls, ev = common.classify('HTTP 500 at /v1: {"error": {"request_id": "v1-abc", "load_failure": {"class": "vram_fit"}}}')
    assert cls == "vram_fit" and ev["request_id"] == "v1-abc" and ev["load_failure_class"] == "vram_fit"
    assert common.classify("model is held by admission (faulty_model)")[0] == "held"
    assert common.classify("HTTP 404 at /v1: nope")[0] == "http_error"
