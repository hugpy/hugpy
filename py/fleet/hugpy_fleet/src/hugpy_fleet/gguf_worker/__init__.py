"""Slim GGUF-only worker agent — see ``agent.py`` for the runnable entry point.

    python -m gguf_worker --central http://192.168.1.250:7002

A drop-in member of the same LLM worker pool as ``worker_agent``, but with no
hugpy/torch/transformers imports: stdlib + flask + llama-cpp-python only. Built
for devices where the full inference stack can't install (Termux phones,
small ARM boards). Speaks the identical registration/heartbeat//infer/stream
protocol, so central needs zero changes to use one.
"""
