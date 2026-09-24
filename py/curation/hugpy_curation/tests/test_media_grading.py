"""Three-tier media graders: image-to-text and text-to-image (2026-09-23)."""
import base64

from hugpy_curation.review import fleet_grading
from hugpy_curation.review import media_grading as mg

VISION_ANSWERS = {
    ("color", "easy"): "Red", ("color", "medium"): "blue", ("color", "hard"): "Orange.",
    ("count", "easy"): "2", ("count", "medium"): "5", ("count", "hard"): "9",
    ("shape", "easy"): "circle", ("shape", "medium"): "Triangle", ("shape", "hard"): "green",
    ("spatial", "easy"): "above", ("spatial", "medium"): "right", ("spatial", "hard"): "green",
    ("read_text", "easy"): "7", ("read_text", "medium"): "305", ("read_text", "hard"): "HUGPY",
}


def test_png_roundtrip_and_color_names():
    cv = mg.Canvas(64, 32).rect(0, 0, 32, 32, mg.BLUE)
    img = mg.decode_png(cv.png())
    assert (img.w, img.h) == (64, 32)
    assert img.get(5, 5) == mg.BLUE and img.get(60, 5) == mg.WHITE
    for rgb, name in ((mg.RED, "red"), (mg.GREEN, "green"), (mg.BLUE, "blue"),
                      (mg.YELLOW, "yellow"), (mg.ORANGE, "orange"), (mg.PURPLE, "purple"),
                      (mg.BLACK, "black"), (mg.WHITE, "white"), ((128, 128, 128), "gray")):
        assert mg.color_name(rgb) == name


def test_vision_images_are_what_the_questions_claim():
    # count tier images really contain n circles; shapes/colors decode.
    for tier, n in (("easy", 2), ("medium", 5), ("hard", 9)):
        build = dict((t, b) for t, _p, b, _c in mg.VISION_TASKS_TIERED["count"])[tier]
        assert mg.dark_blobs(mg.decode_png(build())) == n
    for tiers in mg.VISION_TASKS_TIERED.values():
        assert [t[0] for t in tiers] == ["easy", "medium", "hard"]
        for _tier, _prompt, build, _check in tiers:
            assert mg.decode_png(build()).w > 0


def _vision_grade(answer_for):
    order = [(c, t) for c, tiers in mg.VISION_TASKS_TIERED.items() for t, *_ in tiers]
    it = iter(order)
    seen = []
    def ask(prompt, png):
        key = next(it)
        assert png.startswith(b"\x89PNG")
        seen.append(key)
        return answer_for(key), None, 0.5, 20.0, 30, 2
    return mg.grade_vision(ask), seen


def test_vision_perfect_and_wrong_answers():
    g, seen = _vision_grade(lambda k: VISION_ANSWERS[k])
    assert (g["score"], g["max"], g["suite"]) == (15, 15, "vision") and len(seen) == 15
    g, _ = _vision_grade(lambda k: "I cannot see images")
    assert g["score"] == 0
    # contradictory answers fail ("red or blue")
    g, _ = _vision_grade(lambda k: "red or blue" if k == ("color", "easy") else VISION_ANSWERS[k])
    assert g["detail"]["color"]["tier"] == 2


def _ideal(prompt, params):
    """A 'perfect' image model built from the same canvas primitives."""
    w, h = params["width"], params["height"]
    cv = mg.Canvas(w, h)
    if "apple" in prompt:
        cv.circle(w // 2, h // 2, min(w, h) // 4, mg.RED)
    elif "solid red background" in prompt:
        cv.rect(0, 0, w, h, mg.RED)
    elif "top half is solid blue" in prompt:
        cv.rect(0, 0, w, h // 2, mg.BLUE).rect(0, h // 2, w, h, mg.YELLOW)
    elif "France" in prompt:
        cv.rect(0, 0, w // 3, h, mg.BLUE).rect(2 * w // 3, 0, w, h, mg.RED)
    elif "black circle in the center" in prompt:
        cv.circle(w // 2, h // 2, w // 5, mg.BLACK)
    elif "top-left corner" in prompt:
        cv.rect(10, 10, w // 2 - 10, h // 2 - 10, mg.BLACK)
    elif "three black dots" in prompt:
        for x in (w // 4, w // 2, 3 * w // 4):
            cv.circle(x, h // 2, w // 16, mg.BLACK)
    return base64.b64encode(cv.png()).decode(), None, 1.5


def test_imagegen_ideal_model_scores_full_and_blank_scores_low():
    g = mg.grade_imagegen(_ideal)
    assert (g["score"], g["max"], g["suite"]) == (9, 9, "imagegen"), g["detail"]
    assert g["s_per_image"] == 1.5
    gray = lambda p, params: (base64.b64encode(
        mg.Canvas(512, 512, (128, 128, 128)).png()).decode(), None, 1.0)
    g = mg.grade_imagegen(gray)
    # a fixed gray square is blank, the wrong size for 512x384, and only
    # "reproducible" — it earns the seed tier and nothing else
    assert g["score"] == 1 and g["detail"]["delivery"]["history"][2]["pass"]


def test_imagegen_errors_are_recorded_not_raised():
    g = mg.grade_imagegen(lambda p, params: (None, "ImageGenError: boom", 0.1))
    assert g["score"] == 0 and all(c["grade"] == "ERROR" for c in g["calls"])
    g = mg.grade_imagegen(lambda p, params: ("bm90IGFuIGltYWdl", None, 0.1))   # "not an image"
    assert g["score"] == 0 and g["calls"][0]["grade"] == "ERROR"
    assert g["calls"][0]["error"] not in (None, "N/A")


def test_suite_selection():
    assert mg.suite_for({"primary_task": "text-to-image"}) == "imagegen"
    assert mg.suite_for({"primary_task": "image-text-to-text"}) == "vision"
    assert mg.suite_for({"primary_task": "text-generation"}) == "text"
    assert mg.suite_for({"primary_task": "text-to-video"}) == "none"
    assert mg.suite_for({"tasks": ["text-to-image", "image-to-image"]}) == "imagegen"


class _Client:
    """Just enough central for run_capacity_benchmark over one image model."""
    def __init__(self):
        self.posts = []
    def request(self, path, method="GET", body=None, timeout=None):
        if path.startswith("/models"):
            return {"models": [
                {"model_key": "flux-mini", "primary_task": "text-to-image",
                 "workers": [{"worker_id": "w1", "designated": True}]},
                {"model_key": "tts-x", "primary_task": "text-to-speech",
                 "workers": [{"worker_id": "w1", "designated": True}]}]}
        if path.startswith("/llm/serving/"):
            return {}
        if path == "/ml/imagine":
            self.posts.append(body)
            b64, _e, _s = _ideal(body["prompt"], body)
            return {"ok": True, "images": [{"b64": b64, "width": body["width"],
                                            "height": body["height"]}]}
        raise AssertionError(f"unexpected call {method} {path}")


def test_benchmark_routes_image_models_to_the_media_grader():
    """Image models go through the suites registry to the imagegen suite
    (hugpy-imagegen-v1, /ml/imagine, one row per model, placement hugpy's);
    a task with no suite is skipped with a reason, never text-graded."""
    events = []
    c = _Client()
    workers = [{"id": "w1", "name": "ae", "status": "online", "admission": "approved"}]
    fleet_grading.run_capacity_benchmark(c, workers, 64, fleet_grading.BenchmarkControl(),
                                         lambda kind, v: events.append((kind, v)))
    results = [v for k, v in events if k == "result"]
    by = {r["model"]: r for r in results}
    flux = by["flux-mini"]
    assert flux["status"] == "complete" and flux["grade_suite"] == "hugpy-imagegen-v1"
    assert flux["grade"].endswith("/18") and flux["worker"] == "hugpy-placed"
    assert by["tts-x"]["status"] == "skipped" and by["tts-x"]["failure_class"] == "no_suite"
    assert len(c.posts) == 1 + 18                   # seat generation + 6 tasks x 3 tiers
    assert all("alloc" not in b for b in c.posts)   # placement stays hugpy's
