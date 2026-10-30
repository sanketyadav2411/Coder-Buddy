import io
import stat
import zipfile

import pytest

from agent.secure import (ScanConfig, UnsafeArchiveError, reset_workspace, safe_extract_zip,
                          save_uploaded_file, write_text_files)


def make_zip(entries: dict | list, *, symlink: str | None = None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        items = entries.items() if isinstance(entries, dict) else entries
        for name, data in items:
            zf.writestr(name, data)
        if symlink:
            info = zipfile.ZipInfo(symlink)
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            zf.writestr(info, "/etc/passwd")
    return buf.getvalue()


def cfg(**kw):
    return ScanConfig.from_env(kw)


def test_extracts_only_scannable_sources(tmp_path):
    data = make_zip({"src/a.py": "x = 1\n", "src/b.js": "y = 2\n", "logo.png": b"\x89PNG", "README.md": "hi",
                     "node_modules/dep.js": "eval(1)", "uv.lock": "lock"})
    extracted, skipped = safe_extract_zip(data, tmp_path / "ws", cfg())
    assert sorted(extracted) == ["src/a.py", "src/b.js"] and skipped == []
    assert not (tmp_path / "ws" / "logo.png").exists()


@pytest.mark.parametrize("name", [
    "../evil.py", "a/../../evil.py", "/etc/evil.py", "C:/evil.py", "C:\\evil.py", "..\\evil.py",
])
def test_zip_slip_and_absolute_paths_abort_the_whole_archive(tmp_path, name):
    data = make_zip([("ok.py", "x = 1"), (name, "evil")])
    with pytest.raises(UnsafeArchiveError):
        safe_extract_zip(data, tmp_path / "ws", cfg())
    assert not (tmp_path / "evil.py").exists() and not (tmp_path.parent / "evil.py").exists()


def test_symlink_entries_are_rejected(tmp_path):
    with pytest.raises(UnsafeArchiveError, match="symbolic link"):
        safe_extract_zip(make_zip({"a.py": "x"}, symlink="link.py"), tmp_path / "ws", cfg())


def test_not_a_zip_is_rejected(tmp_path):
    with pytest.raises(UnsafeArchiveError, match="not a valid zip"):
        safe_extract_zip(b"this is not a zip", tmp_path / "ws", cfg())


def test_oversized_member_is_skipped_not_extracted(tmp_path):
    data = make_zip({"big.py": "x = 1\n" * 100000, "small.py": "y = 2\n"})
    extracted, skipped = safe_extract_zip(data, tmp_path / "ws", cfg(max_file_kb=100))
    assert extracted == ["small.py"] and "larger" in skipped[0].reason
    assert not (tmp_path / "ws" / "big.py").exists()


def test_zip_bomb_by_declared_size_is_rejected(tmp_path):
    # 25 MB of highly compressible source declared against a 1 MB budget (guard trips at 20x the budget)
    data = make_zip({f"f{i}.py": "a" * 1_000_000 for i in range(25)})
    assert len(data) < 200_000                                       # tiny on the wire, huge when expanded
    with pytest.raises(UnsafeArchiveError, match="unreasonable size"):
        safe_extract_zip(data, tmp_path / "ws", cfg(max_total_mb=1, max_file_kb=2048))
    assert not (tmp_path / "ws").exists() or list((tmp_path / "ws").iterdir()) == []


def test_total_extracted_size_limit_is_enforced(tmp_path):
    c = cfg(max_total_mb=1, max_file_kb=200)
    data = make_zip({f"f{i}.py": "a" * 100_000 for i in range(12)})  # 1.2 MB of sources
    with pytest.raises(UnsafeArchiveError, match="total size"):
        safe_extract_zip(data, tmp_path / "ws", c)


def test_too_many_entries_is_rejected(tmp_path):
    data = make_zip({f"d/f{i}.txt": "x" for i in range(101)})
    with pytest.raises(UnsafeArchiveError, match="too many entries"):
        safe_extract_zip(data, tmp_path / "ws", cfg(max_files=10))


def test_file_count_cap_skips_the_rest(tmp_path):
    data = make_zip({f"f{i}.py": "x = 1" for i in range(6)})
    extracted, skipped = safe_extract_zip(data, tmp_path / "ws", cfg(max_files=4))
    assert len(extracted) == 4 and len(skipped) == 2


# ---- single-file uploads ------------------------------------------------------------------------------------
@pytest.mark.parametrize("given,saved", [("a.py", "a.py"), ("../../etc/x.py", "x.py"), ("dir\\sub\\y.js", "y.js"),
                                         ("/abs/z.c", "z.c")])
def test_save_uploaded_file_keeps_only_the_basename(tmp_path, given, saved):
    assert save_uploaded_file(given, b"x = 1", tmp_path / "ws", cfg()) == saved
    assert (tmp_path / "ws" / saved).exists() and not (tmp_path / "etc").exists()


@pytest.mark.parametrize("name", ["", "..", "a.exe", "noext"])
def test_save_uploaded_file_rejects_bad_uploads(tmp_path, name):
    with pytest.raises(UnsafeArchiveError):
        save_uploaded_file(name, b"x", tmp_path / "ws", cfg(max_file_kb=200))


def test_save_uploaded_file_rejects_oversized_uploads(tmp_path):
    # built inside the test on purpose: a huge parametrize value would become a huge test id, and
    # pytest stores the id in an environment variable (Windows allows at most 32,767 characters)
    with pytest.raises(UnsafeArchiveError, match="larger than"):
        save_uploaded_file("big.py", b"x" * 300_000, tmp_path / "ws", cfg(max_file_kb=200))


# ---- workspace helpers ----------------------------------------------------------------------------------------
def test_reset_workspace_empties_previous_contents(tmp_path):
    ws = reset_workspace(tmp_path, "upload")
    (ws / "old.py").write_text("x")
    ws2 = reset_workspace(tmp_path, "upload")
    assert ws2 == ws and list(ws2.iterdir()) == []


@pytest.mark.parametrize("name", ["", ".", "..", "a/b", "a\\b"])
def test_reset_workspace_rejects_odd_names(tmp_path, name):
    with pytest.raises(ValueError):
        reset_workspace(tmp_path, name)


def test_write_text_files_cannot_escape_the_workspace(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    write_text_files(ws, {"sub/a.py": "x = 1"})
    assert (ws / "sub" / "a.py").read_text() == "x = 1"
    with pytest.raises(ValueError):
        write_text_files(ws, {"../escape.py": "x"})
    assert not (tmp_path / "escape.py").exists()
