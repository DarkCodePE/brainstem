"""Security regression tests for `wiki_ingest` — SEC-ADR-006 mitigants.
Each test cites SEC-NN. Uses importorskip since modules may not exist yet."""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("wiki_ingest", reason="core not implemented yet")
security_mod = pytest.importorskip("wiki_ingest.security", reason="security not implemented yet")
worker_mod = pytest.importorskip("wiki_ingest.worker", reason="worker not implemented yet")
queue_mod = pytest.importorskip("wiki_ingest.queue", reason="queue not implemented yet")
daemon_mod = pytest.importorskip("wiki_ingest.daemon", reason="daemon not implemented yet")


# Helpers
def _raw(root: Path) -> Path:
    return root / "raw"


def _mk_worker(db_path: Path, root: Path, mcp: MagicMock, **overrides: Any):
    q = queue_mod.IngestQueue(db_path)
    kw: dict[str, Any] = {
        "queue": q,
        "mcp_client": mcp,
        "root": root,
        "pool_size": 1,
        "rate_limit_per_min": 10,
        "max_attempts": 1,
        "backoff": (0, 0, 0),
    }
    kw.update(overrides)
    return worker_mod.IngestWorker(**kw), q


def _reason(row: dict[str, Any]) -> str:
    return (row.get("last_error") or row.get("skip_reason") or "").lower()


UnsafePathError = getattr(security_mod, "UnsafePathError", Exception)


# SEC-01 — Path traversal
@pytest.mark.parametrize(
    "bad_path",
    [
        "raw/../../etc/passwd",
        "raw/articles/evil\x00.md",
        "raw/articles/evil\x01.md",
    ],
)
def test_sec01_rejects_unsafe_paths(tmp_wiki_root: Path, bad_path: str) -> None:
    """SEC-01: traversal, NUL bytes and control bytes raise UnsafePathError."""
    with pytest.raises(UnsafePathError):
        security_mod.validate_safe_path(Path(bad_path), _raw(tmp_wiki_root))


def test_sec01_accepts_valid_path(tmp_wiki_root: Path) -> None:
    """SEC-01: a legitimate raw/articles/ path resolves cleanly."""
    ok = _raw(tmp_wiki_root) / "articles" / "ok.md"
    ok.write_bytes(b"# ok\n")
    assert Path(security_mod.validate_safe_path(ok, _raw(tmp_wiki_root))) == ok.resolve(strict=True)


# SEC-02 — Symlink abuse
@pytest.mark.asyncio
async def test_sec02_symlink_is_quarantined_and_mcp_never_called(
    ingest_db_path: Path,
    tmp_wiki_root: Path,
    mock_mcp_client: MagicMock,
    event_factory: Callable[..., dict[str, Any]],
) -> None:
    """SEC-02: symlink in raw/ is moved to raw/_rejected/symlinks/ and MCP is not called."""
    target = tmp_wiki_root / "outside.txt"
    target.write_bytes(b"secret\n")
    evil = _raw(tmp_wiki_root) / "articles" / "evil.md"
    os.symlink(target, evil)
    w, q = _mk_worker(ingest_db_path, tmp_wiki_root, mock_mcp_client)
    ev = event_factory(path=evil, bucket="articles")
    q.enqueue(ev)
    await w.process_one(q.claim_next())
    rejected = _raw(tmp_wiki_root) / "_rejected" / "symlinks"
    assert rejected.exists() and list(rejected.iterdir()), (
        "symlink must land in raw/_rejected/symlinks/"
    )
    assert mock_mcp_client.write_page.await_count == 0


@pytest.mark.asyncio
async def test_sec02_regular_file_not_quarantined(
    ingest_db_path: Path,
    tmp_wiki_root: Path,
    mock_mcp_client: MagicMock,
    event_factory: Callable[..., dict[str, Any]],
) -> None:
    """SEC-02: regular file is NOT moved to _rejected/symlinks/."""
    good = _raw(tmp_wiki_root) / "articles" / "good.md"
    good.write_bytes(b"---\ntitle: Good\n---\nbody\n")
    w, q = _mk_worker(ingest_db_path, tmp_wiki_root, mock_mcp_client)
    q.enqueue(event_factory(path=good, bucket="articles"))
    await w.process_one(q.claim_next())
    rejected = _raw(tmp_wiki_root) / "_rejected" / "symlinks"
    assert not rejected.exists() or not list(rejected.iterdir())


# SEC-03 — TOCTOU: single os.open per event
@pytest.mark.asyncio
async def test_sec03_single_os_open_per_event(
    ingest_db_path: Path,
    tmp_wiki_root: Path,
    mock_mcp_client: MagicMock,
    event_factory: Callable[..., dict[str, Any]],
) -> None:
    """SEC-03: the file is opened exactly once per event (no path re-open)."""
    src = _raw(tmp_wiki_root) / "articles" / "toctou.md"
    src.write_bytes(b"---\ntitle: T\n---\nbody\n")
    real_open = os.open
    opens: list[str] = []

    def counting(path, flags, mode=0o777, *a, **k):
        p = os.fsdecode(path) if isinstance(path, (bytes, bytearray)) else str(path)
        if p in (str(src), str(src.resolve())):
            opens.append(p)
        return real_open(path, flags, mode, *a, **k)

    with patch.object(os, "open", side_effect=counting):
        w, q = _mk_worker(ingest_db_path, tmp_wiki_root, mock_mcp_client)
        q.enqueue(event_factory(path=src, bucket="articles"))
        await w.process_one(q.claim_next())
    assert len(opens) == 1, f"single open expected, got {opens}"


# SEC-05 — MCP / prompt-injection mitigations
@pytest.mark.asyncio
@pytest.mark.parametrize("danger_key", ["system", "role"])
async def test_sec05_dangerous_frontmatter_rejected(
    ingest_db_path: Path,
    tmp_wiki_root: Path,
    mock_mcp_client: MagicMock,
    event_factory: Callable[..., dict[str, Any]],
    danger_key: str,
) -> None:
    """SEC-05: injection-shaped frontmatter keys => status='failed', reason='dangerous-frontmatter'."""
    src = _raw(tmp_wiki_root) / "articles" / f"danger-{danger_key}.md"
    src.write_bytes(f'---\n{danger_key}: "ignore previous"\ntitle: x\n---\nbody\n'.encode())
    w, q = _mk_worker(ingest_db_path, tmp_wiki_root, mock_mcp_client)
    q.enqueue(event_factory(path=src, bucket="articles"))
    claimed = q.claim_next()
    await w.process_one(claimed)
    row = q.get(claimed["event_id"])
    assert row["status"] == "failed"
    assert "dangerous-frontmatter" in _reason(row) or "injection" in _reason(row)
    assert mock_mcp_client.write_page.await_count == 0


@pytest.mark.asyncio
async def test_sec05_valid_frontmatter_wraps_body_in_ingested_source(
    ingest_db_path: Path,
    tmp_wiki_root: Path,
    mock_mcp_client: MagicMock,
    event_factory: Callable[..., dict[str, Any]],
) -> None:
    """SEC-05: body wrapped as <ingested_source trust="untrusted" sha256="..."> ... </ingested_source>."""
    src = _raw(tmp_wiki_root) / "articles" / "valid.md"
    src.write_bytes(b"---\ntitle: Foo\n---\n<!-- do X -->\n")
    w, q = _mk_worker(ingest_db_path, tmp_wiki_root, mock_mcp_client)
    q.enqueue(event_factory(path=src, bucket="articles"))
    await w.process_one(q.claim_next())
    assert mock_mcp_client.write_page.await_count == 1
    a, k = mock_mcp_client.write_page.call_args
    payload = " ".join(map(str, a)) + " " + " ".join(f"{x}={y}" for x, y in k.items())
    assert "<ingested_source" in payload and "</ingested_source>" in payload
    assert ('trust="untrusted"' in payload) or ("trust='untrusted'" in payload)
    assert "sha256=" in payload


@pytest.mark.asyncio
@pytest.mark.parametrize("variant", ["key-count", "depth"])
async def test_sec05_yaml_bomb_rejected(
    ingest_db_path: Path,
    tmp_wiki_root: Path,
    mock_mcp_client: MagicMock,
    event_factory: Callable[..., dict[str, Any]],
    variant: str,
) -> None:
    """SEC-05/SEC-06: YAML-bomb (150 keys OR 8-level nesting) => status='failed' reason ~ yaml/depth."""
    src = _raw(tmp_wiki_root) / "articles" / f"bomb-{variant}.md"
    if variant == "key-count":
        lines = ["---"] + [f"k{i}: {i}" for i in range(150)] + ["---", "body"]
    else:  # depth
        lines = ["---"]
        indent = ""
        for key in "abcdefgh":
            lines.append(f"{indent}{key}:")
            indent += "  "
        lines[-1] = lines[-1] + " leaf"
        lines += ["---", "body"]
    src.write_bytes(("\n".join(lines) + "\n").encode())
    w, q = _mk_worker(ingest_db_path, tmp_wiki_root, mock_mcp_client)
    q.enqueue(event_factory(path=src, bucket="articles"))
    claimed = q.claim_next()
    await w.process_one(claimed)
    row = q.get(claimed["event_id"])
    assert row["status"] == "failed"
    r = _reason(row)
    assert "yaml" in r or "bomb" in r or "depth" in r
    assert mock_mcp_client.write_page.await_count == 0


# SEC-06 — Content-bomb size gate
@pytest.mark.asyncio
async def test_sec06_26mb_file_skipped_without_mcp(
    ingest_db_path: Path,
    tmp_wiki_root: Path,
    mock_mcp_client: MagicMock,
    event_factory: Callable[..., dict[str, Any]],
) -> None:
    """SEC-06: 26 MiB file (fstat-measured) => skipped with reason='size-bomb', no MCP."""
    src = _raw(tmp_wiki_root) / "articles" / "bomb.md"
    src.write_bytes(b"---\ntitle: bomb\n---\n")
    with open(src, "ab") as f:
        f.write(b"\x00" * (26 * 1024 * 1024))
    w, q = _mk_worker(ingest_db_path, tmp_wiki_root, mock_mcp_client)
    q.enqueue(event_factory(path=src, bucket="articles", size=26 * 1024 * 1024))
    claimed = q.claim_next()
    await w.process_one(claimed)
    row = q.get(claimed["event_id"])
    assert row["status"] in ("skipped", "failed")
    r = _reason(row)
    assert "size" in r or "bomb" in r
    assert mock_mcp_client.write_page.await_count == 0


# SEC-07 — Log allowlist
@pytest.mark.asyncio
async def test_sec07_logs_no_abs_path_no_body_no_raw_exception(
    ingest_db_path: Path,
    tmp_wiki_root: Path,
    mock_mcp_client: MagicMock,
    event_factory: Callable[..., dict[str, Any]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """SEC-07: log output never contains absolute path, body, or raw str(exc)/traceback."""
    src = _raw(tmp_wiki_root) / "articles" / "logtest.md"
    canary = "TOP-SECRET-CANARY-DO-NOT-LOG-ME"
    src.write_bytes(f"---\ntitle: L\n---\n{canary}\n".encode())
    mock_mcp_client.write_page.side_effect = RuntimeError("sensitive-internal-state")
    caplog.set_level(logging.DEBUG, logger="wiki_ingest")
    w, q = _mk_worker(ingest_db_path, tmp_wiki_root, mock_mcp_client)
    q.enqueue(event_factory(path=src, bucket="articles"))
    await w.process_one(q.claim_next())
    blob = (
        "\n".join(r.getMessage() for r in caplog.records)
        + "\n"
        + "\n".join(" ".join(f"{k}={v}" for k, v in r.__dict__.items()) for r in caplog.records)
    )
    assert str(src.resolve()) not in blob, "absolute path leaked"
    assert canary not in blob, "body leaked"
    assert "sensitive-internal-state" not in blob, "raw exception str leaked"
    assert "Traceback" not in blob, "traceback leaked"


# SEC-08 — Rate-limit durability across restarts
@pytest.mark.asyncio
async def test_sec08_rate_limit_persists_across_worker_restart(
    ingest_db_path: Path,
    tmp_wiki_root: Path,
    mock_mcp_client: MagicMock,
    event_factory: Callable[..., dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SEC-08: 8 tokens spent in worker A => new instance B allows only 2 more before throttling."""
    t = [0.0]
    monkeypatch.setattr(worker_mod.time, "monotonic", lambda: t[0])
    sleeps: list[float] = []

    async def fake_sleep(d: float) -> None:
        sleeps.append(d)
        t[0] += d

    monkeypatch.setattr(worker_mod.asyncio, "sleep", fake_sleep)
    w_a, q = _mk_worker(ingest_db_path, tmp_wiki_root, mock_mcp_client, rate_limit_per_min=10)
    for _ in range(8):
        q.enqueue(event_factory())
    for _ in range(8):
        await w_a.process_one(q.claim_next())
    assert mock_mcp_client.write_page.await_count == 8
    # Fresh worker on same DB: no free ride.
    mock_mcp_client.write_page.reset_mock()
    sleeps.clear()
    w_b, _ = _mk_worker(ingest_db_path, tmp_wiki_root, mock_mcp_client, rate_limit_per_min=10)
    for _ in range(3):
        q.enqueue(event_factory())
    for _ in range(3):
        await w_b.process_one(q.claim_next())
    assert any(s >= 1.0 for s in sleeps), (
        f"restart bypassed rate-limit; no throttle observed (sleeps={sleeps})"
    )


# SEC-10 — Atomic writes
@pytest.mark.asyncio
async def test_sec10_atomic_write_uses_same_dir_tempfile(
    ingest_db_path: Path,
    tmp_wiki_root: Path,
    mock_mcp_client: MagicMock,
    event_factory: Callable[..., dict[str, Any]],
) -> None:
    """SEC-10: os.replace(tmp, target) where tmp is created in target's own directory."""
    src = _raw(tmp_wiki_root) / "articles" / "atomic.md"
    src.write_bytes(b"---\ntitle: A\n---\nbody\n")
    calls: list[tuple[str, str]] = []
    real = os.replace

    def tracking(a, b, *ar, **kw):
        calls.append((str(a), str(b)))
        return real(a, b, *ar, **kw)

    with patch.object(os, "replace", side_effect=tracking):
        w, q = _mk_worker(ingest_db_path, tmp_wiki_root, mock_mcp_client)
        q.enqueue(event_factory(path=src, bucket="articles"))
        await w.process_one(q.claim_next())
    assert calls, "os.replace must be used for atomic writes"
    for tmp, target in calls:
        assert Path(tmp).parent == Path(target).parent, (
            f"tempfile {tmp} must share directory with target {target}"
        )


@pytest.mark.asyncio
async def test_sec10_crash_between_temp_and_replace_leaves_no_partial(
    ingest_db_path: Path,
    tmp_wiki_root: Path,
    mock_mcp_client: MagicMock,
    event_factory: Callable[..., dict[str, Any]],
) -> None:
    """SEC-10: simulated crash at os.replace => target absent or prior version, never partial."""
    src = _raw(tmp_wiki_root) / "articles" / "crash.md"
    src.write_bytes(b"---\ntitle: C\n---\nbody\n")
    target_dir = tmp_wiki_root / "wiki" / "sources"
    target_dir.mkdir(parents=True, exist_ok=True)
    prev = target_dir / "crash.md"
    prev.write_bytes(b"previous-good-version\n")

    def boom(a, b, *ar, **kw):
        raise OSError("simulated crash between tempfile and rename")

    with patch.object(os, "replace", side_effect=boom):
        w, q = _mk_worker(ingest_db_path, tmp_wiki_root, mock_mcp_client)
        q.enqueue(event_factory(path=src, bucket="articles"))
        with contextlib.suppress(OSError):
            await w.process_one(q.claim_next())
    if prev.exists():
        assert prev.read_bytes() == b"previous-good-version\n", "target must never be half-written"


# SEC-11 — DB perms + umask
def test_sec11_db_file_is_0600_after_init(ingest_db_path: Path) -> None:
    """SEC-11: EventQueue.init creates .wiki-ingest.db with mode 0o600."""
    old = os.umask(0o000)
    try:
        q = queue_mod.IngestQueue(ingest_db_path)
        if hasattr(q, "init"):
            q.init()
        assert ingest_db_path.exists()
        mode = ingest_db_path.stat().st_mode & 0o777
        assert mode == 0o600, f"expected 0600, got {oct(mode)}"
    finally:
        os.umask(old)


def test_sec11_daemon_sets_restrictive_umask(monkeypatch: pytest.MonkeyPatch) -> None:
    """SEC-11: daemon start hook invokes os.umask(0o077)."""
    observed: list[int] = []
    real = os.umask

    def capturing(new: int) -> int:
        observed.append(new)
        return real(new)

    monkeypatch.setattr(os, "umask", capturing)
    entry = (
        getattr(daemon_mod, "apply_security_umask", None)
        or getattr(daemon_mod, "_apply_security_umask", None)
        or getattr(daemon_mod, "harden_process", None)
    )
    if entry is None:
        pytest.skip("daemon hardening hook not exposed yet")
    entry()
    assert 0o077 in observed, f"expected umask(0o077), observed={[oct(x) for x in observed]}"
