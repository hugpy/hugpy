"""The doctrine seam: null default, explicit install, oracle fallback, and
the judge note that reads through it (no ``hugpy_fleet`` import anywhere)."""

from __future__ import annotations

import pytest

from hugpy_curation import providers
from hugpy_curation.dossier.verdicts import doctrine_note


class Doctrine:
    def __init__(self, version):
        self.version = version


class FakeSource:
    def __init__(self, doctrine):
        self.doctrine = doctrine

    def latest(self):
        return self.doctrine


@pytest.fixture(autouse=True)
def _reset():
    from hugpy_oracle.providers import reset_providers as reset_oracle
    providers.reset_providers()
    reset_oracle()
    yield
    providers.reset_providers()
    reset_oracle()


def test_default_is_null_and_the_note_says_no_snapshot():
    src = providers.get_doctrine_source()
    assert isinstance(src, providers.NullDoctrineSource)
    assert isinstance(src, providers.DoctrineSource)
    assert src.latest() is None
    assert doctrine_note().startswith("no fleet doctrine snapshot exists yet")


def test_installed_source_drives_the_note():
    providers.set_doctrine_source(FakeSource(Doctrine("k118.7")))
    assert "fleet doctrine k118.7 is available" in doctrine_note()
    providers.set_doctrine_source(FakeSource({"version": 3}))     # mapping form
    assert "fleet doctrine 3 is available" in doctrine_note()
    providers.set_doctrine_source(None)
    assert isinstance(providers.get_doctrine_source(), providers.NullDoctrineSource)


def test_oracle_installed_doctrine_is_used_when_curation_has_none():
    from hugpy_oracle.providers import set_doctrine_source as oracle_set

    oracle_set(FakeSource(Doctrine("from-oracle")))
    assert providers.get_doctrine_source().latest().version == "from-oracle"
    assert "from-oracle" in doctrine_note()
    # An explicit curation install still wins over the oracle's.
    providers.set_doctrine_source(FakeSource(Doctrine("mine")))
    assert "fleet doctrine mine is available" in doctrine_note()


def test_a_faulting_source_is_a_note_not_a_failure():
    class Broken:
        def latest(self):
            raise ConnectionError("central down")

    providers.set_doctrine_source(Broken())
    assert "fleet doctrine unavailable (ConnectionError)" in doctrine_note()


def test_set_rejects_non_sources():
    with pytest.raises(TypeError):
        providers.set_doctrine_source(object())
