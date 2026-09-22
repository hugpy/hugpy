"""hugpy sentinel (k95): bound-exceeded detection -> case -> one-shot agent.

A small periodic checker over central's four read surfaces (/llm/jobs,
/llm/workers, /oracle/capabilities, observed /oracle/route scorecards). Each
anomaly opens ONE deduplicated case; each newly opened case spawns ONE
bounded `hugpy-agent case` run that diagnoses and documents (document-only
contract). Remedies exist as a typed whitelist but execution is gated on a
setting that defaults OFF, and the prod worker "ae" is excluded
structurally — ae cases are document+escalate only.

The sentinel is a CLIENT of central over HTTP, never an import of its
internals: central's job-state machine (comms/jobs.py) stays the single
authority on stalled/expired.
"""
