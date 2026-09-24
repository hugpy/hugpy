"""console_manifest.json shape and tools/build_console.py --check (no npm needed)."""

from __future__ import annotations

import copy
import importlib.util
import json
import re
from pathlib import Path

import pytest

PKG_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PKG_ROOT / "console_manifest.json"

_spec = importlib.util.spec_from_file_location("build_console", PKG_ROOT / "tools" / "build_console.py")
bc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bc)

EXPECTED = {  # BUILD_CONSOLE.md table
    "/": ("ui", "@hugpy/ui", ""),
    "/fleet": ("agents_ui", "@hugpy/agents-ui", "fleet"),
    "/media": ("media_intelligence_ui", "@hugpy/media-intelligence-ui", "media"),
    "/video": ("video_intelligence_ui", "@hugpy/video-intelligence-ui", "video"),
}


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_manifest_shape():
    data = _manifest()
    bc.validate_manifest(data)
    assert data["console_dist"] == "src/hugpy_server/console_dist"
    got = {m: (e["package_dir"], e["npm"], e["target"]) for m, e in data["mounts"].items()}
    assert got == EXPECTED
    for entry in data["mounts"].values():
        assert re.fullmatch(r"\d+\.\d+\.\d+", entry["version"])


def test_manifest_matches_react_packages():
    data = _manifest()
    react_root = (MANIFEST.parent / data["react_root"]).resolve()
    if not react_root.is_dir():
        pytest.skip("react/ tree not present next to this package")
    for entry in data["mounts"].values():
        pkg = json.loads((react_root / entry["package_dir"] / "package.json").read_text())
        assert pkg["name"] == entry["npm"]


@pytest.mark.parametrize("mutate, msg", [
    (lambda d: d.pop("mounts"), "missing 'mounts'"),
    (lambda d: d["mounts"].pop("/"), "root mount"),
    (lambda d: d["mounts"]["/fleet"].pop("version"), "missing ['version']"),
    (lambda d: d["mounts"]["/fleet"].update(target="agents"), "must copy to 'fleet'"),
    (lambda d: d["mounts"]["/fleet"].update(npm="agents-ui"), "@hugpy scope"),
])
def test_manifest_validation_rejects(mutate, msg):
    data = copy.deepcopy(_manifest())
    mutate(data)
    with pytest.raises(ValueError, match=re.escape(msg)):
        bc.validate_manifest(data)


def _fake_dist(tmp_path: Path) -> Path:
    cd = tmp_path / "console_dist"
    (cd / "assets").mkdir(parents=True)
    (cd / "index.html").write_text("<html>root</html>")
    (cd / "assets" / "main.js").write_text("")
    (cd / "fleet").mkdir()
    (cd / "fleet" / "index.html").write_text("<html>fleet</html>")
    (cd / "video").mkdir()  # present but no index.html
    return cd


def test_check_reports_each_mount(tmp_path, capsys):
    cd = _fake_dist(tmp_path)
    rc = bc.main(["--check", "--console-dist", str(cd)])
    out = capsys.readouterr().out
    assert rc == 1  # /media and /video lack index.html
    rows = {line.split()[0]: line for line in out.splitlines()[1:-1]}
    assert set(rows) == set(EXPECTED)
    assert "index.html: present" in rows["/"] and "files: 2" in rows["/"]
    assert "index.html: present" in rows["/fleet"]
    assert "index.html: MISSING" in rows["/media"]
    assert "index.html: MISSING" in rows["/video"]
    assert out.splitlines()[-1] == "2/4 mounts have index.html"


def test_check_only_filters(tmp_path, capsys):
    cd = _fake_dist(tmp_path)
    assert bc.main(["--check", "--only", "fleet", "--console-dist", str(cd)]) == 0
    out = capsys.readouterr().out
    assert "/fleet" in out and "/media" not in out
    assert bc.main(["--check", "--only", "/nope", "--console-dist", str(cd)]) == 2


def test_install_keeps_other_mounts(tmp_path):
    """Replacing one mount never wipes the others (root included)."""
    manifest = bc.load_manifest(MANIFEST)
    cd = _fake_dist(tmp_path)
    new_root = tmp_path / "ui_dist"
    (new_root / "fleet").mkdir(parents=True)  # ui's postbuild copy of an arm
    (new_root / "index.html").write_text("<html>new root</html>")
    (new_root / "fleet" / "index.html").write_text("<html>stale arm copy</html>")
    bc.install_dist(manifest, "/", new_root, cd)
    assert (cd / "index.html").read_text() == "<html>new root</html>"
    assert not (cd / "assets").exists()
    assert (cd / "fleet" / "index.html").read_text() == "<html>fleet</html>"
    assert (cd / "video").is_dir()

    new_media = tmp_path / "media_dist"
    new_media.mkdir()
    (new_media / "index.html").write_text("<html>media</html>")
    bc.install_dist(manifest, "/media", new_media, cd)
    assert (cd / "media" / "index.html").is_file()
    assert (cd / "index.html").read_text() == "<html>new root</html>"
    assert (cd / "fleet" / "index.html").read_text() == "<html>fleet</html>"

    with pytest.raises(bc.BuildError):
        bc.install_dist(manifest, "/video", tmp_path / "empty", cd)
    assert (cd / "video").is_dir()


# ---- console freshness: a UI edit can never silently miss a wheel ----------
import os as _os
import sys as _sys
import time as _time

WORKSPACE = PKG_ROOT.parents[2]


def _tree(tmp_path):
    src = tmp_path / "react" / "ui" / "src"
    (src / "components").mkdir(parents=True)
    (src / "components" / "A.jsx").write_text("export default 1\n")
    (src / "components" / "a.test.mjs").write_text("// test only\n")
    dist = tmp_path / "console_dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html></html>")
    return src, dist


def test_source_hash_ignores_test_files_only(tmp_path):
    src, _dist = _tree(tmp_path)
    h = bc.source_hash(src)
    (src / "components" / "a.test.mjs").write_text("// changed test\n")
    assert bc.source_hash(src) == h
    (src / "components" / "A.jsx").write_text("export default 2\n")
    assert bc.source_hash(src) != h


def test_stamp_match_is_fresh_and_any_edit_is_stale(tmp_path):
    src, dist = _tree(tmp_path)
    bc.write_source_stamp(dist, src)
    assert bc.console_staleness(src, dist) is None
    (src / "components" / "B.jsx").write_text("x\n")
    why = bc.console_staleness(src, dist)
    assert why and "STALE" in why and bc.REBUILD_HINT in why


def test_mtime_fallback_without_stamp(tmp_path):
    src, dist = _tree(tmp_path)
    old = _time.time() - 3600
    for p in src.rglob("*"):
        _os.utime(p, (old, old))
    _os.utime(dist / "index.html", (old + 60, old + 60))
    assert bc.console_staleness(src, dist) is None
    (src / "components" / "A.jsx").write_text("edited\n")
    assert "newer than" in bc.console_staleness(src, dist)
    assert "no SOURCE_HASH.json" in bc.console_staleness(src, dist, mtime_fallback=False)


def test_no_react_tree_is_not_judged(tmp_path):
    assert bc.console_staleness(tmp_path / "nope", tmp_path) is None


def _build_wheels():
    bw = WORKSPACE / "py" / "build_wheels.py"
    if not bw.is_file():
        # The sandboxed verify mirrors only the packages, not the repo-level
        # build script; the guard is exercised from the live tree.
        pytest.skip("py/build_wheels.py not present in this checkout (package-only sandbox)")
    s = importlib.util.spec_from_file_location("build_wheels_under_test", bw)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def test_build_wheels_refuses_hugpy_server_on_a_stale_console(tmp_path, monkeypatch):
    bw = _build_wheels()
    src, dist = _tree(tmp_path)
    bc.write_source_stamp(dist, src)
    (src / "components" / "A.jsx").write_text("edited without a rebuild\n")
    monkeypatch.setattr(bw, "REACT_UI_SRC", src)
    monkeypatch.setattr(bw, "CONSOLE_DIST", dist)
    monkeypatch.setattr(bw.console_stale_reason, "__defaults__", (src, dist))
    built = []
    monkeypatch.setattr(bw, "build_one", lambda *a, **k: built.append(a))
    with pytest.raises(SystemExit) as exc:
        bw.main(["--only", "hugpy-server", "--out", str(tmp_path / "out")])
    assert "refusing to build hugpy-server" in str(exc.value) and bc.REBUILD_HINT in str(exc.value)
    assert built == []
    # a package set without hugpy-server is not held up by the console
    monkeypatch.setattr(bw, "verify", lambda *a, **k: ("0.0.0", []))
    assert bw.main(["--only", "hugpy-platform", "--out", str(tmp_path / "out2"), "--no-sdist"]) == 0
    # rebuilt -> the same hugpy-server build proceeds
    bc.write_source_stamp(dist, src)
    assert bw.main(["--only", "hugpy-server", "--out", str(tmp_path / "out3"), "--no-sdist"]) == 0


def test_the_tracked_console_matches_the_react_source():
    """The live tree itself: console_dist was rebuilt from react/ui/src."""
    src = WORKSPACE / "react" / "ui" / "src"
    if not src.is_dir():
        pytest.skip("no React tree")
    assert bc.console_staleness(src, PKG_ROOT / "src" / "hugpy_server" / "console_dist") is None
