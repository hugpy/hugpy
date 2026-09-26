"""pkg_promote.build_members refuses a hugpy-server member that carries the
LIVE console bundle while react/ui/src changed since it was built (the
pipeline never runs npm). A member with a different bundle (an older version
rebuilt for rollback) is not judged against today's React source."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pkg_promote as P  # noqa: E402

BC = P.BW._console_tool()


def _setup(tmp_path, monkeypatch):
    src = tmp_path / "react" / "ui" / "src"
    src.mkdir(parents=True)
    (src / "A.jsx").write_text("1\n")
    live = tmp_path / "live" / "console_dist"
    live.mkdir(parents=True)
    (live / "index.html").write_text("<html>v2</html>")
    BC.write_source_stamp(live, src)
    monkeypatch.setattr(P.BW, "REACT_UI_SRC", src)
    monkeypatch.setattr(P.BW, "CONSOLE_DIST", live)
    monkeypatch.setattr(P.BW.console_stale_reason, "__defaults__", (src, live))
    member = tmp_path / "member" / "hugpy_server"
    (member / "src" / "hugpy_server" / "console_dist").mkdir(parents=True)
    return src, live, member


def test_live_bundle_member_rebuilds_when_the_ui_changed(tmp_path, monkeypatch):
    src, live, member = _setup(tmp_path, monkeypatch)
    (member / "src" / "hugpy_server" / "console_dist" / "index.html").write_text("<html>v2</html>")
    P.console_fresh_or_rebuild([member])                    # fresh: passes
    (src / "A.jsx").write_text("edited, never rebuilt\n")
    rebuilt = []
    def _rebuild(stale):
        rebuilt.append(stale)
        BC.write_source_stamp(live, src)
    monkeypatch.setattr(P, "rebuild_console", _rebuild)
    P.console_fresh_or_rebuild([member])
    assert rebuilt and P.sha256(member / "src" / "hugpy_server" /
                                "console_dist" / "SOURCE_HASH.json") == \
        P.sha256(live / "SOURCE_HASH.json")


def test_other_bundle_member_is_not_judged(tmp_path, monkeypatch):
    src, live, member = _setup(tmp_path, monkeypatch)
    (member / "src" / "hugpy_server" / "console_dist" / "index.html").write_text("<html>v1 (rollback)</html>")
    (src / "A.jsx").write_text("edited\n")
    P.console_fresh_or_rebuild([member])                    # an older version: no verdict
