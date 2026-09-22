"""Single-format transfer/ledger honesty (2026-07-17 slice).

Central mirrors WHOLE HF snapshots — the same weights in 3-5 formats, often an
fp32 duplicate — and the old provisioner shipped every one to workers while the
per-model ``bytes`` ledger summed the whole folder (whisper-large-v3 read 50GB
vs ~3GB usable; flan-t5-xl 45GB vs ~11GB). This inflation made the fleet gauge
look like a runaway download storm when nothing was transferring (the operator
scare). This suite locks the SINGLE-FORMAT selection and the effective-bytes
math against representative on-disk layouts.

Design contract exercised here:
  * safetensors (single + sharded+index) is preferred; pytorch bins / tf / flax /
    h5 / msgpack / onnx|openvino|coreml dirs / rust_model.ot / fp32 duplicates are
    dropped ONLY when a complete usable format remains.
  * pytorch-only dirs keep the bins (that is then the serving format).
  * GGUF is a NO-OP here (its effective quant is resolved elsewhere).
  * DEGRADE TO CORRECT: an unrecognized layout with no positively-complete
    torch format returns the WHOLE listing (never risk a broken model to save
    disk).

Runs under pytest:  venv/bin/python -m pytest tests/test_single_format_select.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugpy_storage import format_select as F


def _kept(files, framework="transformers"):
    return sorted(r for (r, _s) in F.select_files(files, framework=framework))


# ── representative on-disk layouts ──────────────────────────────────────────
WHISPER_LARGE_V3 = [
    ("flax_model.msgpack", 6174007324),
    ("pytorch_model.fp32-00001-of-00002.bin", 4993677094),
    ("model.fp32-00001-of-00002.safetensors", 4993448880),
    ("pytorch_model.bin", 3087394553),
    ("model.safetensors", 3087130976),          # <- the fp16 that serves
    ("pytorch_model.fp32-00002-of-00002.bin", 1180725482),
    ("model.fp32-00002-of-00002.safetensors", 1180663192),
    ("small.pt", 483617219),                     # openai-whisper .pt (kept, conservative)
    ("tokenizer.json", 2480617),
    ("pytorch_model.bin.index.fp32.json", 117957),
    ("model.safetensors.index.fp32.json", 117893),
    ("config.json", 1272),
    ("preprocessor_config.json", 340),
    ("generation_config.json", 3903),
]

FLAN_T5_XL_SHARDED = [
    ("tf_model-00001-of-00002.h5", 9970701536),
    ("flax_model-00001-of-00002.msgpack", 9969726387),
    ("pytorch_model-00001-of-00002.bin", 9449717937),
    ("model-00001-of-00002.safetensors", 9449619912),   # <- sharded st part 1
    ("pytorch_model-00002-of-00002.bin", 1949494999),
    ("model-00002-of-00002.safetensors", 1949477672),   # <- sharded st part 2
    ("tf_model-00002-of-00002.h5", 1429448928),
    ("flax_model-00002-of-00002.msgpack", 1429326464),
    ("tokenizer.json", 2424064),
    ("spiece.model", 791656),
    ("tf_model.h5.index.json", 68466),
    ("model.safetensors.index.json", 53032),            # <- the st index (needed)
    ("flax_model.msgpack.index.json", 51206),
    ("pytorch_model.bin.index.json", 50781),
    ("config.json", 1438),
]

# pytorch-only: no safetensors at all -> bins are the serving format, kept.
LED_LARGE_PYTORCH_ONLY = [
    ("pytorch_model.bin", 648000000),
    ("config.json", 1200),
    ("tokenizer.json", 500000),
    ("merges.txt", 400000),
    ("vocab.json", 900000),
]

# sentence-transformers: safetensors present + alt-runtime exports + pooling dir.
MINILM_MULTIFORMAT = [
    ("tf_model.h5", 91005696),
    ("pytorch_model.bin", 90888945),
    ("rust_model.ot", 90887379),
    ("model.safetensors", 90868376),            # <- serves
    ("tokenizer.json", 466247),
    ("vocab.txt", 231508),
    ("onnx/model.onnx", 90000000),
    ("onnx/model_quantized.onnx", 23000000),
    ("openvino/openvino_model.bin", 90000000),
    ("openvino/openvino_model.xml", 211000),
    ("1_Pooling/config.json", 190),             # <- needed sidecar (subdir)
    ("modules.json", 349),
    ("config_sentence_transformers.json", 116),
    ("sentence_bert_config.json", 53),
    ("config.json", 612),
]

GGUF_MULTIQUANT = [
    ("model-Q4_K_M.gguf", 4800000000),
    ("model-Q8_0.gguf", 8500000000),
    ("model-f16.gguf", 16000000000),
    ("mmproj-f16.gguf", 600000000),
    ("config.json", 900),
]


# ── safetensors preferred over redundant formats ────────────────────────────
def test_whisper_keeps_fp16_safetensors_drops_the_rest():
    kept = _kept(WHISPER_LARGE_V3)
    assert "model.safetensors" in kept
    assert "config.json" in kept and "tokenizer.json" in kept
    assert "preprocessor_config.json" in kept
    # every redundant weight format is gone
    for gone in ("flax_model.msgpack", "pytorch_model.bin",
                 "pytorch_model.fp32-00001-of-00002.bin",
                 "model.fp32-00001-of-00002.safetensors",
                 "model.safetensors.index.fp32.json",
                 "pytorch_model.bin.index.fp32.json"):
        assert gone not in kept, gone


def test_whisper_effective_is_a_fraction_of_the_dir_sum():
    dir_sum = sum(s for (_r, s) in WHISPER_LARGE_V3)
    eff = F.effective_bytes(WHISPER_LARGE_V3, framework="transformers")
    assert eff < dir_sum / 5           # ~3.6GB vs ~25GB
    # exactly: fp16 safetensors + small.pt + sidecars
    assert eff == (3087130976 + 483617219 + 2480617 + 1272 + 340 + 3903)


def test_flan_sharded_safetensors_keeps_shards_and_index():
    kept = _kept(FLAN_T5_XL_SHARDED)
    assert "model-00001-of-00002.safetensors" in kept
    assert "model-00002-of-00002.safetensors" in kept
    assert "model.safetensors.index.json" in kept       # the index is a needed keep
    assert "spiece.model" in kept and "tokenizer.json" in kept
    for gone in ("tf_model-00001-of-00002.h5", "flax_model-00001-of-00002.msgpack",
                 "pytorch_model-00001-of-00002.bin", "pytorch_model.bin.index.json",
                 "tf_model.h5.index.json", "flax_model.msgpack.index.json"):
        assert gone not in kept, gone


def test_flan_effective_is_the_single_safetensors_set():
    eff = F.effective_bytes(FLAN_T5_XL_SHARDED, framework="transformers")
    expect = (9449619912 + 1949477672 + 53032        # st shards + index
              + 2424064 + 791656 + 1438)             # tokenizer + spiece + config
    assert eff == expect
    assert eff < sum(s for (_r, s) in FLAN_T5_XL_SHARDED) / 3


# ── pytorch-only keeps bins (that IS the format) ────────────────────────────
def test_pytorch_only_keeps_the_bin():
    kept = _kept(LED_LARGE_PYTORCH_ONLY)
    assert "pytorch_model.bin" in kept
    assert F.effective_bytes(LED_LARGE_PYTORCH_ONLY, framework="transformers") == \
        sum(s for (_r, s) in LED_LARGE_PYTORCH_ONLY)   # nothing to drop


# ── alt-runtime dirs + rust/tf dropped, pooling sidecar kept ────────────────
def test_minilm_drops_alt_runtimes_keeps_pooling():
    kept = _kept(MINILM_MULTIFORMAT)
    assert "model.safetensors" in kept
    assert "1_Pooling/config.json" in kept              # subdir sidecar survives
    assert "modules.json" in kept
    assert "config_sentence_transformers.json" in kept
    for gone in ("tf_model.h5", "pytorch_model.bin", "rust_model.ot",
                 "onnx/model.onnx", "onnx/model_quantized.onnx",
                 "openvino/openvino_model.bin", "openvino/openvino_model.xml"):
        assert gone not in kept, gone


# ── GGUF untouched ──────────────────────────────────────────────────────────
def test_gguf_is_a_noop():
    for fw in ("gguf", "llama_cpp", "GGUF"):
        sel = F.select_files(GGUF_MULTIQUANT, framework=fw)
        assert sel == GGUF_MULTIQUANT               # identity — no quant dropped
        assert F.effective_bytes(GGUF_MULTIQUANT, framework=fw) == \
            sum(s for (_r, s) in GGUF_MULTIQUANT)


# ── degrade-to-correct: no recognizable torch format -> keep everything ─────
def test_unknown_layout_kept_whole():
    weird = [
        ("weights.pkl", 500000000),        # unrecognized weight container
        ("config.json", 1000),
        ("tokenizer.json", 200000),
    ]
    assert F.select_files(weird, framework="transformers") == weird


def test_only_alt_formats_present_is_kept_whole():
    # tf/flax only (no torch format). We can't positively name a keep-format, so
    # we must NOT strip them — that would leave a folder with NO loadable weights.
    alt_only = [
        ("tf_model.h5", 400000000),
        ("flax_model.msgpack", 400000000),
        ("config.json", 1000),
    ]
    assert F.select_files(alt_only, framework="transformers") == alt_only


def test_empty_and_framework_none():
    assert F.select_files([], framework="transformers") == []
    # framework unknown but a clean safetensors set present -> still selects.
    single = [("model.safetensors", 100), ("config.json", 10),
              ("pytorch_model.bin", 100)]
    kept = _kept(single, framework=None)
    assert "model.safetensors" in kept and "pytorch_model.bin" not in kept
