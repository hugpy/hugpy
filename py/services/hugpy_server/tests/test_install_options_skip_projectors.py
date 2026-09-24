"""The Add-models quant picker never offers a vision projector as a quant.

zerodigest/Qwen3.8-27B-Uncensored-YMQ-MTP-GGUF ships its quants under
non-standard names (``YMQ-L-TI`` …) and its projectors as
``mmproj/Qwen3.8-27B-Uncensored-vision-{Q4_K_S,Q6_K,Q8_0}.gguf``. The quant
regex grouped the projectors as "GGUF · Q6_K", that option was installed, and
central ended up holding only projector files for the model (2026-09-23).
"""
from types import SimpleNamespace as NS

from hugpy_server.app.functions.imports.options.install import _gguf_options

GB = 2 ** 30


def test_projectors_are_not_quant_options():
    files = [NS(path="Qwen3.8-27B-Uncensored-YMQ-L-TI.gguf", size=int(14.2 * GB)),
             NS(path="Qwen3.8-27B-Uncensored-YMQ-XS-Pro.gguf", size=int(10.9 * GB)),
             NS(path="mmproj/Qwen3.8-27B-Uncensored-vision-Q4_K_S.gguf", size=500_340_480),
             NS(path="mmproj/Qwen3.8-27B-Uncensored-vision-Q6_K.gguf", size=614_851_584),
             NS(path="mmproj-model-f16.gguf", size=int(1.7 * GB)),
             NS(path="README.md", size=17069)]
    opts = _gguf_options(files, free_bytes=None)
    chosen = [o.filename or "" for o in opts] + [str(o.include) for o in opts]
    assert not any("mmproj" in c for c in chosen), chosen
    assert {o.filename for o in opts} >= {"Qwen3.8-27B-Uncensored-YMQ-L-TI.gguf",
                                          "Qwen3.8-27B-Uncensored-YMQ-XS-Pro.gguf"}
