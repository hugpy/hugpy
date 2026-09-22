"""The dossier store as the oracle's ``DossierSource`` — tested against the
oracle's real Protocol and its real ``set_dossier_source`` registry."""

from __future__ import annotations

import json
import os

import pytest

from hugpy_curation.dossier import store as dstore
from hugpy_curation.dossier.dossier import ModelDossier, TrustSignals
from hugpy_curation.dossier.oracle_source import DossierStoreSource, install_dossier_source


@pytest.fixture
def dossier_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DOSSIER_DIR", str(tmp_path))
    yield str(tmp_path)


@pytest.fixture(autouse=True)
def _reset_oracle_providers():
    from hugpy_oracle.providers import reset_providers
    reset_providers()
    yield
    reset_providers()


def _file(criteria: str, hub_id: str) -> str:
    d = ModelDossier(hub_id=hub_id, criteria=criteria,
                     trust=TrustSignals(license="apache-2.0"))
    path = dstore.save(d)
    assert path
    return path


def test_source_satisfies_the_oracle_protocol():
    from hugpy_oracle import providers as op

    source = DossierStoreSource()
    assert isinstance(source, op.DossierSource)      # runtime_checkable Protocol


def test_root_follows_the_store_and_pin_wins(dossier_dir, tmp_path):
    assert DossierStoreSource().root() == dossier_dir
    pinned = str(tmp_path / "elsewhere")
    assert DossierStoreSource(pinned).root() == pinned


def test_iter_dossiers_yields_store_records_and_skips_sidecars(dossier_dir):
    a = _file("nightly", "org/alpha")
    b = _file("vision", "org/beta")
    dstore.save_radar("nightly", [{"hub_id": "org/gem"}], detail="x")
    # A corrupt file and a stray tmp file are the store's problem, not the ledger's.
    with open(os.path.join(dossier_dir, "nightly", "broken.json"), "w") as fh:
        fh.write("{not json")
    with open(os.path.join(dossier_dir, "nightly", "org__zeta.json.tmp"), "w") as fh:
        fh.write("{}")

    records = list(DossierStoreSource().iter_dossiers())
    assert [(r["criteria"], r["hub_id"], r["path"]) for r in records] == [
        ("nightly", "org/alpha", a), ("vision", "org/beta", b)]
    for r in records:
        assert set(r) == {"criteria", "hub_id", "payload", "path"}
        assert isinstance(r["payload"], dict)
        assert r["payload"]["hub_id"] == r["hub_id"]
        assert r["payload"]["trust"]["license"] == "apache-2.0"


def test_missing_root_is_empty_not_an_error(tmp_path):
    source = DossierStoreSource(str(tmp_path / "nowhere"))
    assert list(source.iter_dossiers()) == []


def test_install_registers_with_the_oracle_and_the_ledger_sees_it(dossier_dir):
    from hugpy_oracle import providers as op

    path = _file("nightly", "org/alpha")
    source = install_dossier_source()
    assert op.get_dossier_source() is source
    assert op.get_dossier_source().root() == dossier_dir
    records = list(op.get_dossier_source().iter_dossiers())
    assert [r["path"] for r in records] == [path]
    # The oracle's own consumer reads through the same seam.
    from hugpy_oracle.interim_ledger import DiscoveryDossierSource
    ok, detail, root = DiscoveryDossierSource().probe()
    assert ok and root == dossier_dir, detail


def test_package_install_providers_reports_what_it_wired(dossier_dir):
    import hugpy_curation
    from hugpy_oracle import providers as op

    report = hugpy_curation.install_providers()
    assert report["dossier_source"] == {"installed": True, "root": dossier_dir}
    assert report["doctrine_source"]["installed"] is False
    assert isinstance(op.get_dossier_source(), DossierStoreSource)
