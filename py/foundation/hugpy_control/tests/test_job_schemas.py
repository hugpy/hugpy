"""job_schemas is a compat re-export of hugpy_control.jobs — same objects."""
from hugpy_control import job_schemas, jobs


def test_names_are_the_same_objects():
    for name in job_schemas.__all__:
        assert getattr(job_schemas, name) is getattr(jobs, name), name
    assert job_schemas.normalize_status("queued") == "pending"
