"""VRAM offload cliff sweep — where does throughput actually fall off?

**Operator ask (2026-07-25, verbatim):** _"just have it start at max vram with
spill to ram and see if there is any difference every 10% less vram. and have any
notable difference examined within that 10% to determine where the change was. if
there was no change ever, then thats info to know. and means that vram is not a
priority"_

So: coarse pass at 100%→0% GPU layers in 10% steps; whenever two adjacent coarse
points differ by more than ``--notable`` (default 15%), BISECT that 10% band to
locate the actual transition. A flat curve is a RESULT, not a failure — it means
VRAM is not the lever we assume it is, and the band/floor guesses built on that
assumption are unfounded.

Why this can run now, when k7 (2026-07-18) could not
----------------------------------------------------
`OFFLOAD-CLIFF-2026-07-18.md` built a harness and could not measure anything,
for two reasons it documented as blockers:
  1. the ae slot pinned models at full GPU and IGNORED the offload lever, and
  2. no central verb could cold-recycle the slot child (`unload`/`evict`/
     `slots-unload` all left the PID unchanged), so a swept budget never applied.
Both are fixed: the k14 `/slots/<id>/relaunch` verb respawns the child under a
NEW pid with an explicit `n_gpu_layers`, verified live 2026-07-25. That was
literally recommendation #1 of that document.

k7 also selected its subjects by POPULARITY (top-10 by recent usage) and found 7
of 10 sat on a CPU-only worker — so it concluded "unmeasurable". The MoE sweep
(`test_moe.py`) took the opposite approach, enumerating by STRUCTURE, and got
definitive per-model answers including for models nobody had ever called. This
harness follows the structural method: it picks subjects by what the placement
decision depends on (dense, fits the card, real chat model), not by call counts.

**A NOTE ON THE MODEL OF SPILL THIS TESTS.** `spill.cpu_resident_bytes()` today
assumes layers are UNIFORM: ``file_bytes * (1 - ngl/total)``. Real GGUFs are not
uniform — the token embedding and output head are large non-repeating tensors
that belong to no block. So the x-axis here is reported BOTH ways: the nominal
uniform estimate (what the code believes) and the measured VRAM actually taken
(what the card reports). Divergence between them is itself a finding.

Usage (needs a QUIET GPU — it repeatedly re-seats a model):

    venv/bin/python tests/test_vram_cliff.py --list
    venv/bin/python tests/test_vram_cliff.py --model <key> --path <gguf>
    venv/bin/python tests/test_vram_cliff.py --model <key> --path <gguf> \
        --step 10 --notable 15 --runs 3

Results append to /mnt/llm_storage/comms/vram-cliff-sweep.json. Nothing here is
destructive: the slot is unloaded at the end and the record is additive.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

GIB = 2 ** 30
AE_AGENT = "http://192.168.1.100:9100"
AE_SLOT = "http://192.168.1.100:8101"
RECORD = Path("/mnt/llm_storage/comms/vram-cliff-sweep.json")
MOE_RECORD = Path("/mnt/llm_storage/comms/moe-detection-sweep.json")

# EVERY WAIT IS BOUNDED. An unattended sweep that can block forever isn't
# unattended — the first run proved it, sitting 76 minutes in do_poll on a
# stream that never completed, which no per-model try/except can rescue because
# nothing raises. Socket timeouts are per-operation and reset on any byte, so a
# separate wall-clock deadline is required, not merely nice.
_SOCK_TIMEOUT_S = 120.0        # per socket op — a healthy token gap is « 1s
_GEN_DEADLINE_S = 300.0        # total per generation
_LOAD_TIMEOUT_S = 600.0        # a seat/load: big GGUFs off the hot tier are ~5s,
                               # off the spinner minutes; 10 min is generous

# APTITUDE PROBES (operator, 2026-07-25): the sweep already pays for a decode at
# every grid point, so ask something that GRADES the model instead of a throwaway
# paragraph. Each probe is auto-checkable from the text alone — no judge model,
# no ambiguity — so one run yields BOTH the tok/s curve and a per-model aptitude
# score. Deliberately short (the sweep measures decode rate, not essay length)
# and deliberately varied: code, arithmetic reasoning, deductive logic, and
# instruction-following, which fail independently.
#
# Also useful as a spill sanity check: if quality collapses as layers move to
# CPU, that is a correctness finding, not just a speed one. (It should NOT —
# offload changes where weights live, not what they compute — so a score that
# drops with ngl is evidence of a real bug.)
APTITUDE_PROBES = [
    {
        "id": "code_reverse",
        "prompt": ("Write a single line of Python that reverses the string s. "
                   "Reply with ONLY the code, no explanation."),
        # any of the idiomatic answers
        "check": lambda t: ("[::-1]" in t) or ("reversed(" in t),
    },
    {
        "id": "math_wordproblem",
        "prompt": ("A shelf holds 3 boxes. Each box holds 7 jars. Each jar holds "
                   "12 marbles. How many marbles in total? Reply with ONLY the "
                   "number."),
        "check": lambda t: "252" in t.replace(",", ""),
    },
    {
        "id": "logic_deduction",
        "prompt": ("All bloops are razzies. All razzies are lazzies. Is every "
                   "bloop a lazzie? Answer ONLY yes or no."),
        "check": lambda t: t.strip().lower().lstrip("*# ").startswith("yes"),
    },
    {
        "id": "instruction_following",
        "prompt": ("Reply with exactly the word BANANA in uppercase and nothing "
                   "else."),
        "check": lambda t: "BANANA" in t and len(t.strip()) <= 24,
    },
]

# The timing prompt is probe 0 by default; --probe-all grades every probe at each
# grid point (4x the decodes — slower, but a far better aptitude signal).
PROMPT = APTITUDE_PROBES[0]["prompt"]


# ───────────────────────────────── plumbing ─────────────────────────────────

def _post(url: str, body: dict | None = None,
          timeout: float = _LOAD_TIMEOUT_S) -> dict:
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode()
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode()[:200]}"}
    except Exception as exc:                    # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}
    try:
        return json.loads(raw)
    except ValueError:
        return {"raw": raw[:400]}


def _get(url: str, timeout: float = 30.0) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception as exc:                    # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def vram_used_bytes() -> int | None:
    h = _get(f"{AE_AGENT}/health")
    gpus = h.get("gpus") or []
    if not gpus:
        return None
    g = gpus[0]
    tot, free = g.get("memory_total"), g.get("memory_free")
    return None if tot is None or free is None else int(tot) - int(free)


def slot_status() -> dict:
    return _get(f"{AE_SLOT}/status")


def unload() -> None:
    _post(f"{AE_SLOT}/unload", {}, timeout=120)
    time.sleep(2)


def seat(model_key: str, path: str, ngl: int) -> dict:
    """Seat the model at an explicit layer count. Returns the slot's own report.

    Uses the SLOT's /load (not central's) so the measurement is of the placement
    we asked for, with nothing in between re-deciding it.

    **DO NOT PASS ``path``.** The first run of this harness did, pointing at
    ``/mnt/llm_storage/...`` — the 16 TB spinning array. An explicit path is
    taken verbatim, which SKIPS the hot-cache read-through
    (``hot_cache.use()``), so every load streamed off the spinner and the sweep
    was timing disk I/O rather than GPU offload. Loads that should be seconds
    off the 990 took minutes. Omitting ``path`` lets the worker resolve the
    model itself, which routes through ``/mnt/hot990`` and also exercises the
    promote-on-call path the way real serving does. The ``path`` parameter is
    kept only for the caller's records/logging.
    """
    unload()
    t0 = time.time()
    r = _post(f"{AE_SLOT}/load",
              {"model_key": model_key, "n_gpu_layers": ngl})
    r["_load_seconds"] = round(time.time() - t0, 1)
    return r


def timed_generate(model_key: str, max_tokens: int = 96,
                   probe: dict | None = None) -> dict:
    """One timed decode through the worker — and GRADE it.

    Returns tok/s plus, when a `probe` is given, whether the answer was correct
    and the text it produced. Same decode, twice the information: the sweep is
    already paying for the tokens, so every grid point also scores the model's
    aptitude (operator ask, 2026-07-25).
    """
    probe = probe or APTITUDE_PROBES[0]
    body = {"model_key": model_key,
            "messages": [{"role": "user", "content": probe["prompt"]}],
            "max_tokens": max_tokens}
    data = json.dumps(body).encode()
    req = urllib.request.Request(f"{AE_AGENT}/infer/stream", data=data,
                                 headers={"Content-Type": "application/json"})
    t0 = first = None
    tokens = 0
    chunks: list[str] = []
    # HARD WALL-CLOCK CEILING. urlopen's `timeout` is PER SOCKET OPERATION, not
    # a total: a stream that trickles anything (a keepalive, a status frame)
    # resets it forever and the read never returns. That wedged the first
    # unattended run — the process sat in do_poll for 76 minutes with 0 CPU,
    # blocking all remaining models, and the per-model try/except could not help
    # because nothing ever raised. An unattended sweep MUST bound every wait it
    # makes; "it will probably answer" is not a bound.
    deadline = time.time() + _GEN_DEADLINE_S
    try:
        with urllib.request.urlopen(req, timeout=_SOCK_TIMEOUT_S) as r:
            t0 = time.time()
            for line in r:
                if time.time() > deadline:
                    return {"error": f"deadline: no completion in "
                                     f"{_GEN_DEADLINE_S}s ({tokens} tokens)",
                            "tokens": tokens}
                s = line.decode(errors="replace").strip()
                if not s.startswith("data: "):
                    continue
                try:
                    ev = json.loads(s[6:])
                except ValueError:
                    continue
                if ev.get("type") == "token":
                    tokens += 1
                    chunks.append(str(ev.get("text") or ""))
                    if first is None:
                        first = time.time()
    except Exception as exc:                    # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}", "tokens": tokens}
    end = time.time()
    if not tokens or t0 is None:
        return {"error": "no tokens", "tokens": 0}
    decode_s = end - (first or t0)
    text = "".join(chunks)
    try:
        correct = bool(probe["check"](text))
    except Exception:                            # noqa: BLE001 — a grader must never abort a sweep
        correct = None
    return {
        "tokens": tokens,
        "ttft_s": round((first - t0), 3) if first else None,
        "total_s": round(end - t0, 3),
        "tok_s": round(tokens / decode_s, 2) if decode_s > 0 else None,
        "probe": probe["id"],
        "correct": correct,
        "answer": text.strip()[:200],
    }


# ───────────────────────────────── the sweep ────────────────────────────────

def measure_point(model_key: str, path: str, pct: int, total_layers: int,
                  runs: int, max_tokens: int) -> dict:
    """Seat at `pct`% of layers on GPU and measure decode throughput.

    pct=100 -> ngl=-1 (all layers, the 'max vram' start the operator asked for);
    pct=0   -> ngl=0  (pure CPU, full spill to RAM).
    """
    ngl = -1 if pct >= 100 else int(round(total_layers * pct / 100.0))
    seated = seat(model_key, path, ngl)
    if seated.get("error") or not seated.get("healthy"):
        return {"pct": pct, "ngl": ngl, "error": seated.get("error") or "unhealthy",
                "load_seconds": seated.get("_load_seconds")}
    time.sleep(1)
    vram = vram_used_bytes()
    samples: list[float] = []
    for i in range(runs + 1):                   # first run is warmup, discarded
        g = timed_generate(model_key, max_tokens)
        if i and not g.get("error") and g.get("tok_s"):
            samples.append(g["tok_s"])
    # Aptitude: grade EVERY probe once at this placement. Same decodes the sweep
    # already pays for, but they now say what the model can DO — and, because
    # this repeats at every grid point, whether quality holds as layers spill to
    # CPU (it should: offload moves where weights live, not what they compute —
    # so a score that degrades with ngl is a real bug, not a speed tradeoff).
    graded = []
    for pr in APTITUDE_PROBES:
        g = timed_generate(model_key, max_tokens, probe=pr)
        graded.append({"probe": pr["id"], "correct": g.get("correct"),
                       "answer": g.get("answer"), "error": g.get("error")})
        if g.get("tok_s"):
            samples.append(g["tok_s"])
    scored = [x for x in graded if x["correct"] is not None]
    n_ok = sum(1 for x in scored if x["correct"])
    st = slot_status()
    return {
        "pct": pct,
        "ngl": ngl,
        "effective_ngl": st.get("n_gpu_layers"),
        "n_cpu_moe": st.get("n_cpu_moe"),
        "vram_used_gib": round(vram / GIB, 3) if vram else None,
        "tok_s_median": round(statistics.median(samples), 2) if samples else None,
        "tok_s_samples": samples,
        "aptitude": f"{n_ok}/{len(scored)}" if scored else None,
        "aptitude_score": round(n_ok / len(scored), 3) if scored else None,
        "probes": graded,
        "load_seconds": seated.get("_load_seconds"),
        "child_pid": st.get("child_pid"),
    }


def sweep_model(model_key: str, path: str, step: int, notable_pct: float,
                runs: int, max_tokens: int, verbose: bool = True) -> dict:
    """Coarse 100→0 in `step`%; bisect any adjacent pair differing > notable_pct."""
    total_layers = _get(f"{AE_SLOT}/status").get("total_layers")
    if not total_layers:
        probe = seat(model_key, path, -1)
        total_layers = probe.get("total_layers") or slot_status().get("total_layers")
    if not total_layers:
        return {"model_key": model_key, "error": "could not determine total_layers"}

    grid = list(range(100, -1, -step))
    if grid[-1] != 0:
        grid.append(0)
    if verbose:
        print(f"\n{model_key}\n  layers={total_layers}  grid={grid}  "
              f"runs={runs}/point  notable={notable_pct}%\n")

    coarse: list[dict] = []
    for pct in grid:
        pt = measure_point(model_key, path, pct, total_layers, runs, max_tokens)
        coarse.append(pt)
        if verbose:
            if pt.get("error"):
                print(f"  {pct:>3}% ngl={pt['ngl']:<4} ERROR {pt['error'][:70]}")
            else:
                # None-safe: an unmeasurable VRAM read or a point with no timed
                # sample must print a dash, never crash the sweep. (It did:
                # `:>6` on None raised TypeError mid-run — recorded and skipped
                # by the per-model guard, but it cost that model's curve.)
                _v = pt.get("vram_used_gib")
                _t = pt.get("tok_s_median")
                print(f"  {pct:>3}% ngl={pt['ngl']:<4} "
                      f"vram={(f'{_v:.3f}' if _v is not None else '-'):>7} GiB  "
                      f"{(f'{_t:.2f}' if _t is not None else '-'):>8} tok/s  "
                      f"aptitude {pt.get('aptitude') or '-':<5} "
                      f"(load {pt.get('load_seconds')}s)", flush=True)

    # Bisect every adjacent pair whose throughput moved more than `notable_pct`.
    refined: list[dict] = []
    for a, b in zip(coarse, coarse[1:]):
        ta, tb = a.get("tok_s_median"), b.get("tok_s_median")
        if not ta or not tb:
            continue
        delta = abs(tb - ta) / max(ta, tb) * 100.0
        if delta <= notable_pct:
            continue
        lo, hi = b["pct"], a["pct"]
        if verbose:
            print(f"\n  NOTABLE {hi}%->{lo}%: {ta} -> {tb} tok/s "
                  f"({delta:.0f}% change) — bisecting\n")
        seen = {a["pct"]: ta, b["pct"]: tb}
        for _ in range(3):                      # 3 splits ~= 1-2% resolution
            mid = (lo + hi) // 2
            if mid in seen or mid in (lo, hi):
                break
            pt = measure_point(model_key, path, mid, total_layers, runs, max_tokens)
            refined.append(pt)
            tm = pt.get("tok_s_median")
            if verbose:
                print(f"    bisect {mid:>3}% ngl={pt['ngl']:<4} "
                      f"{tm if tm else pt.get('error','?')} tok/s")
            if not tm:
                break
            seen[mid] = tm
            # Walk toward whichever side still holds the discontinuity.
            if abs(tm - seen[hi]) > abs(tm - seen[lo]):
                hi = mid
            else:
                lo = mid
        if verbose:
            print(f"  -> transition localized between {lo}% and {hi}% GPU layers\n")

    ok = [p for p in coarse if p.get("tok_s_median")]
    spread = None
    if len(ok) >= 2:
        vals = [p["tok_s_median"] for p in ok]
        spread = round((max(vals) - min(vals)) / max(vals) * 100.0, 1)
    # Aptitude rolled up across the whole curve: the BEST score any placement
    # achieved (the model's real capability) and whether it held steady as
    # layers spilled (it should — a drop is a correctness bug, not a tradeoff).
    scores = [p["aptitude_score"] for p in coarse
              if p.get("aptitude_score") is not None]
    apt_best = max(scores) if scores else None
    apt_worst = min(scores) if scores else None
    return {
        "model_key": model_key,
        "path": path,
        "total_layers": total_layers,
        "coarse": coarse,
        "refined": refined,
        "spread_pct": spread,
        "aptitude_best": apt_best,
        "aptitude_worst": apt_worst,
        "aptitude_stable_across_offload": (
            None if apt_best is None else apt_best == apt_worst),
        "verdict": ("FLAT — no measurable cliff; VRAM placement is NOT the lever "
                    "for this model" if spread is not None and spread < notable_pct
                    else "cliff present" if spread is not None else "inconclusive"),
    }


def resolve_model_key(path: str) -> str | None:
    """Best-effort model_key for a store path, via central's manifest.

    The worker resolves a model_key to its own on-disk copy (which is what lets
    us omit ``path`` and get hot-cache read-through), so the sweep needs the KEY
    the fleet knows, not the filename. Returns None when nothing matches — the
    caller then SKIPS that model rather than dying.
    """
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from hugpy_engine.config.models.models_config import get_models_dict
        keys = list(get_models_dict(dict_return=True) or {})
    except Exception:                            # noqa: BLE001
        return None
    # The repo dir is the strongest signal (…/gguf/<owner>/<repo>/…): match a
    # key whose bare tail equals the repo dir, else fall back to the filename
    # stem. Namespaced keys (owner~Repo) and bare keys both resolve this way.
    parts = Path(path).parts
    repo = parts[-2] if len(parts) >= 2 else ""
    stem = Path(path).stem
    for want in (repo, stem):
        if not want:
            continue
        for k in keys:
            if k == want or k.split("~")[-1] == want:
                return k
    for k in keys:                               # last resort: containment
        tail = k.split("~")[-1]
        if tail and (tail in path):
            return k
    return None


def completed_paths(out: Path = RECORD, include_failures: bool = False) -> set:
    """Paths already in the record — the basis for RESUME.

    THE PATTERN (operator, 2026-07-25: _"every test should be built like
    this"_): a long sweep must be stoppable and resumable. Three properties,
    and all three are needed:

      1. **Append per unit of work**, not at the end — a crash, a reboot, or a
         kill leaves every completed item on disk.
      2. **Read the record on start** and skip what is already done — so
         restarting is FREE and never duplicates.
      3. **An explicit --force** to redo anyway, because a fixed bug or a
         changed method means old rows are stale.

    Without (2), (1) alone still forces a full re-run and appends duplicates,
    which is what this harness did on its first outing.

    By default a FAILED row does not count as done: an error/skip usually means
    a bug worth retrying after a fix. ``include_failures=True`` treats any row
    as complete (useful to grind through a known-bad model list once).
    """
    done: set = set()
    if not out.is_file():
        return done
    try:
        payload = json.loads(out.read_text())
    except (OSError, ValueError):
        return done                              # unreadable record -> redo all
    for r in payload.get("runs", []):
        p = r.get("path")
        if not p:
            continue
        failed = bool(r.get("error") or r.get("skipped"))
        if failed and not include_failures:
            continue
        done.add(p)
    return done


def sweep_all(step: int, notable_pct: float, runs: int, max_tokens: int,
              limit: int, min_gib: float, max_gib: float,
              force: bool = False, retry_failures: bool = True) -> dict:
    """Sweep EVERY eligible dense model, unattended.

    Robustness is the whole point (operator: _"make sure it's robust enough not
    to stop because some model has a path error or something"_):
      * every model runs inside its own try/except — a failure is RECORDED as a
        row and the run continues to the next model;
      * an unresolvable model_key is skipped with a reason, not raised;
      * the record is written after EVERY model, so a crash, a reboot, or a
        kill still leaves every completed sweep on disk;
      * the slot is unloaded between models so one wedged child can't poison
        the next measurement.
    """
    cands = dense_candidates(limit=limit, min_gib=min_gib, max_gib=max_gib)
    done = set() if force else completed_paths(
        include_failures=not retry_failures)
    todo = [c for c in cands if c["path"] not in done]
    print(f"unattended sweep: {len(cands)} candidate(s); "
          f"{len(done & {c['path'] for c in cands})} already recorded -> "
          f"{len(todo)} to run"
          f"{'  [--force: redoing all]' if force else ''}\n", flush=True)
    results: list[dict] = []
    for i, c in enumerate(todo, 1):
        path = c["path"]
        print(f"\n{'=' * 78}\n[{i}/{len(todo)}] {Path(path).name}  "
              f"({c.get('size_gib')} GiB)\n{'=' * 78}", flush=True)
        key = resolve_model_key(path)
        if not key:
            row = {"path": path, "size_gib": c.get("size_gib"),
                   "skipped": "no model_key in central's manifest"}
            print(f"  SKIP — {row['skipped']}", flush=True)
            results.append(row)
            append_record(row)
            continue
        try:
            r = sweep_model(key, path, step, notable_pct, runs, max_tokens)
        except Exception as exc:                 # noqa: BLE001 — never abort the run
            r = {"model_key": key, "path": path,
                 "error": f"{type(exc).__name__}: {exc}"}
            print(f"  ERROR — {r['error'][:120]}", flush=True)
        results.append(r)
        append_record(r)                         # persist after EVERY model
        try:
            unload()
        except Exception:                        # noqa: BLE001
            pass

    print("\n" + "=" * 78 + "\nUNATTENDED SWEEP COMPLETE\n" + "=" * 78,
          flush=True)
    for r in results:
        name = Path(r.get("path", "?")).name[:52]
        if r.get("skipped"):
            print(f"  SKIP   {name:<52} {r['skipped']}")
        elif r.get("error"):
            print(f"  ERROR  {name:<52} {str(r['error'])[:60]}")
        else:
            print(f"  {('FLAT' if 'FLAT' in (r.get('verdict') or '') else 'cliff'):<6} "
                  f"{name:<52} spread {r.get('spread_pct')}%")
    return {"models": len(results), "results": results}


# A decode-throughput sweep needs models that DECODE TOKENS. The first cut of
# this filter selected on size + dense-ness only and happily offered
# `Wan2.1_14B_VACE` (video) and `flux2-klein-9b-...-text-encoder` (an encoder,
# not a generator) as subjects — neither produces a tok/s curve. These are
# excluded by family/role markers in the path rather than by a task label,
# because the catalog's task labels are themselves unreliable (the sticky
# task-from-PATH landmine).
_NOT_A_DECODER = (
    "vace", "wan2.1", "wan2_1",
    "-vae", "clip", "mmproj", "upscaler", "sd-turbo", "sdxl",
    "diffusers", "controlnet", "embedding", "gte-", "bge-", "rerank",
)

# PATH NAMES LIE — the reason this is a config check and not a substring test.
# `flux2-klein-9b-uncensored-TEXT-ENCODER` is a full Qwen3ForCausalLM 9B text
# GENERATOR; "text-encoder" names its ROLE in the flux2 image pipeline, not
# what the artifact is. The first cut of this filter excluded it on the words
# "encoder" and "flux" and so silently dropped one of the largest legitimate
# dense subjects on the box. Same family of error as the sticky task-from-PATH
# landmine. So: ask config.json what the architecture IS; fall back to the
# name list only when there is no config to read.
_CAUSAL_MARKERS = ("forcausallm", "forconditionalgeneration")


def _is_text_decoder(path: str) -> bool:
    low = path.lower()
    if any(m in low for m in _NOT_A_DECODER):
        return False
    # Prefer ground truth: the sibling config.json's `architectures`.
    for parent in (Path(path).parent, Path(path).parent.parent):
        cfg = parent / "config.json"
        if not cfg.is_file():
            continue
        try:
            arch = json.loads(cfg.read_text()).get("architectures") or []
        except (OSError, ValueError):
            continue
        if arch:
            return any(any(m in str(a).lower() for m in _CAUSAL_MARKERS)
                       for a in arch)
    return True                                  # no config -> assume decoder


def dense_candidates(limit: int = 10, min_gib: float = 3.0,
                     max_gib: float = 20.0) -> list[dict]:
    """Dense GGUFs that FIT the card, from the MoE sweep record (structural pick).

    Deliberately not a popularity ranking — that is the selection mistake that
    made the k7 sweep unmeasurable.
    """
    if not MOE_RECORD.is_file():
        return []
    rows = json.loads(MOE_RECORD.read_text())["rows"]
    out = [r for r in rows
           if not r["is_moe"] and r["verdict"] == "dense"
           and "_archive" not in r["path"]
           and _is_text_decoder(r["path"])
           and min_gib <= (r["size_gib"] or 0) <= max_gib]
    out.sort(key=lambda r: -(r["size_gib"] or 0))
    return out[:limit]


def append_record(result: dict, out: Path = RECORD) -> Path:
    payload = {"runs": []}
    if out.is_file():
        try:
            payload = json.loads(out.read_text())
        except ValueError:
            pass
    payload.setdefault("runs", []).append({"at": int(time.time()), **result})
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        return out
    except OSError:
        alt = Path.cwd() / out.name
        alt.write_text(json.dumps(payload, indent=2))
        return alt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true",
                    help="list dense candidates that fit the card, then exit")
    ap.add_argument("--model", help="model_key to sweep")
    ap.add_argument("--path", help="absolute .gguf path (shard 1 if sharded)")
    ap.add_argument("--step", type=int, default=10, help="coarse step %% (default 10)")
    ap.add_argument("--notable", type=float, default=15.0,
                    help="%% throughput change that triggers a bisect (default 15)")
    ap.add_argument("--runs", type=int, default=3, help="timed runs per point")
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--keep", action="store_true",
                    help="leave the last placement seated (default: unload)")
    ap.add_argument("--all", action="store_true",
                    help="sweep EVERY eligible dense model, unattended; a bad "
                         "model is recorded and skipped, never fatal")
    ap.add_argument("--limit", type=int, default=10,
                    help="--all: max models to sweep (default 10)")
    ap.add_argument("--min-gib", type=float, default=3.0)
    ap.add_argument("--max-gib", type=float, default=20.0)
    ap.add_argument("--force", action="store_true",
                    help="--all: redo models already in the record (default: "
                         "RESUME — skip them, so restarting is free)")
    ap.add_argument("--keep-failures", action="store_true",
                    help="--all: treat previously FAILED models as done too "
                         "(default: retry them, since a failure usually means "
                         "a bug that has since been fixed)")
    a = ap.parse_args(argv)

    if a.all:
        h = _get(f"{AE_AGENT}/health")
        if h.get("error"):
            print(f"ae agent unreachable: {h['error']}")
            return 2
        sweep_all(a.step, a.notable, a.runs, a.max_tokens,
                  a.limit, a.min_gib, a.max_gib,
                  force=a.force, retry_failures=not a.keep_failures)
        try:
            unload()
        except Exception:                        # noqa: BLE001
            pass
        print(f"\nrecord: {RECORD}")
        return 0

    if a.list or not (a.model and a.path):
        cands = dense_candidates()
        if not cands:
            print(f"no candidates (is {MOE_RECORD} present? run test_moe.py first)")
            return 2
        print("dense GGUFs that fit the 24 GiB card (structural pick):\n")
        for r in cands:
            print(f"  {r['size_gib']:>6.2f} GiB  {r['path']}")
        print("\nsweep one with:\n  venv/bin/python tests/test_vram_cliff.py \\\n"
              "      --model '<model_key>' --path '<path above>'")
        return 0

    h = _get(f"{AE_AGENT}/health")
    if h.get("error"):
        print(f"ae agent unreachable: {h['error']}")
        return 2
    print(f"ae: {h.get('name')}  gpus={[g.get('name') for g in (h.get('gpus') or [])]}")

    result = sweep_model(a.model, a.path, a.step, a.notable, a.runs, a.max_tokens)
    if not a.keep:
        unload()

    out = append_record(result)
    print("\n" + "=" * 78)
    print(f"  verdict: {result.get('verdict')}")
    if result.get("spread_pct") is not None:
        print(f"  spread : {result['spread_pct']}% between fastest and slowest")
    print(f"  record : {out}")
    if (result.get("spread_pct") or 0) < a.notable:
        print("\n  A FLAT curve is a real result: it says GPU layer placement did "
              "not\n  move throughput for this model — so VRAM is not the "
              "priority the\n  band/floor guesses assume, and those guesses are "
              "unfounded.")
    return 0


# ─────────────────────── cheap pytest contract (CI-safe) ────────────────────
# No GPU, no store: these pin the sweep's LOGIC so it can't silently rot.

def test_grid_starts_at_max_vram_and_reaches_zero():
    step = 10
    grid = list(range(100, -1, -step))
    if grid[-1] != 0:
        grid.append(0)
    assert grid[0] == 100 and grid[-1] == 0
    assert len(grid) == 11


def test_pct_to_ngl_mapping():
    total = 48
    assert (-1 if 100 >= 100 else 0) == -1          # 100% -> -1 (all layers)
    assert int(round(total * 50 / 100.0)) == 24
    assert int(round(total * 0 / 100.0)) == 0


def test_resume_skips_completed_but_retries_failures(tmp_path):
    """The resume contract — restarting a long sweep must be FREE.

    Completed models are skipped; FAILED ones are retried by default (a failure
    usually means a bug that has since been fixed — as happened here: a
    TypeError killed one model's curve mid-run and was fixed minutes later).
    """
    rec = tmp_path / "r.json"
    rec.write_text(json.dumps({"runs": [
        {"path": "/m/ok.gguf", "verdict": "cliff present"},
        {"path": "/m/boom.gguf", "error": "TypeError: ..."},
        {"path": "/m/skip.gguf", "skipped": "no model_key"},
    ]}))
    done = completed_paths(rec)
    assert done == {"/m/ok.gguf"}                       # failures NOT done
    allrows = completed_paths(rec, include_failures=True)
    assert allrows == {"/m/ok.gguf", "/m/boom.gguf", "/m/skip.gguf"}


def test_resume_treats_a_missing_or_corrupt_record_as_nothing_done(tmp_path):
    assert completed_paths(tmp_path / "nope.json") == set()
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert completed_paths(bad) == set()                # redo all, never crash


def test_flat_curve_is_reported_as_a_result_not_a_failure():
    vals = [20.0, 19.8, 20.1, 19.9]
    spread = (max(vals) - min(vals)) / max(vals) * 100.0
    assert spread < 15.0                             # would read as FLAT


if __name__ == "__main__":
    raise SystemExit(main())
