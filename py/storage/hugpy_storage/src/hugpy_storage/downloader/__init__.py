"""The model-download plane, OUT of the console API process.

Why this package exists (operator, 2026-07-27/28: "a download from hf in the
add models tab seems to push off all of the workers and makes the api
unstable"):

A download used to run *inside* gunicorn. Even after the fork->spawn fix, the
transfer child was parented to a gunicorn worker and the monitor / watch /
estimate threads lived in the API process — one of them walking the whole
destination tree once a second. Those threads share the pool that also serves
``/llm/workers/<id>/heartbeat``; starve it for 45s (HEARTBEAT_TIMEOUT_SECONDS)
and every worker reads ``offline``. The fix is not a better thread, it is a
different PROCESS.

Layout::

    engine.py    the transfer lifecycle (spawn child, watch, stall+resume,
                 terminal state, registry refresh) — FLASK-FREE, so the daemon
                 never imports the web stack.
    queue.py     enqueue / cancel / retry, expressed purely against the comms
                 job store + its cross-process mirror. The API routes call
                 THIS; they never start a transfer.
    daemon.py    the long-lived process: claim queued download jobs from the
                 mirror and run them. ``hugpy-downloader``.
    presence.py  the daemon's heartbeat file, so the API can tell an operator
                 "queued, waiting for downloader" instead of leaving a job
                 silently dead.

The queue is the EXISTING cross-process plumbing (comms/jobs.py +
comms/shared.py SqliteMirror), not a new IPC: the API creates a ``pending``
job of kind ``download`` and disowns it; the daemon claims it with a
compare-and-set and becomes its owner. Cancel rides the mirror's cancel flag,
exactly as it already did between gunicorn workers.
"""
