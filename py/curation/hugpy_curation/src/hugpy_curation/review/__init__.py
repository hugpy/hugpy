"""hugpy_curation.review — automated download-and-review of HF models.

Two stages, so cheap work filters the expensive work:

  1. screen  — metadata only (no weights). Fit: params, quant, bytes, context,
     estimated VRAM at the target context vs the card. Capability: task, tags,
     downloads, recency, base-model lineage, publisher trust.
  2. smoke   — survivors get downloaded and actually loaded with llama_cpp in a
     subprocess, then asked fixed probe prompts: real load time, real VRAM,
     real tokens/sec, real output.

A hugpy agent (a model the fleet already serves, over /v1/chat/completions)
reads the measured facts and returns an adopt/trial/reject verdict.

Entry points: the `hugpy-review` CLI (`hugpy-curation review`),
the /api/llm/review routes, and a systemd timer running saved criteria.

Submodules import lazily on purpose — importing this package must never pull
in huggingface_hub, llama_cpp or Flask, because the Flask app imports it at
startup and a missing optional dep would take the service down.
"""

# `judge_model`, not `judge`: a package attribute named after a submodule is a
# footgun — importing hugpy_curation.review.judge rebinds the attribute to
# the MODULE, silently replacing the function anyone had grabbed off the package.
__all__ = ["ReviewCriteria", "screen", "smoke_test", "review_one", "run",
           "report_markdown", "judge_model", "store", "push_run"]


from importlib import import_module

# submodule name -> attribute to pull out of it (None = the module itself).
# import_module, NOT `from . import store`: the latter looks the name up on the
# package first, which re-enters __getattr__ and recurses until the stack dies.
_LAZY = {
    "ReviewCriteria": (".criteria", "ReviewCriteria"),
    "screen": (".screen", "screen"),
    "smoke_test": (".smoke", "smoke_test"),
    "review_one": (".pipeline", "review_one"),
    "run": (".pipeline", "run"),
    "report_markdown": (".pipeline", "report_markdown"),
    "judge_model": (".judge", "judge"),
    "store": (".store", None),
    "push_run": (".push", "push_run"),
}


def __getattr__(name):
    try:
        module, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(name) from None
    mod = import_module(module, __name__)
    value = mod if attr is None else getattr(mod, attr)
    globals()[name] = value          # cache: __getattr__ runs once per name
    return value
