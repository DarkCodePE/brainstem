"""Mermaid → PNG rendering + diagram caption (ADR-023 Phase 1).

LOCAL-ONLY: renders the Mermaid block ADR-022 already produces into a PNG using
``mmdc`` (mermaid-cli) driving the system Chrome — no hosted service, nothing
leaves the machine ([[ADR-018]]). Rendering is a DEGRADABLE dependency: if
``mmdc``/Chrome is missing or the render fails, every function here returns
``None``/best-effort and the caller keeps the page + text post intact
([[ADR-023]] R-1). Never raises into the ingest path.

The PNG is a first-class artifact (caller stores it under
``knowledge-base/assets/diagrams/<slug>.png``), NOT inlined into the repo page
prose. ``diagram_caption`` builds the deterministic text proxy used for
multimodal search (Phase 1 / B1).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

from wiki_repos.types import RepoRef

logger = logging.getLogger(__name__)

# (cmd, env, timeout) -> exit code. Injectable so tests never spawn node/Chrome.
RenderRunner = Callable[[list[str], "dict[str, str]", float], int]

_FENCE_RE = re.compile(r"^\s*```mermaid\s*\n(?P<src>.*?)\n```\s*$", re.DOTALL)

_CHROME_CANDIDATES = (
    "/usr/bin/google-chrome",
    "/usr/bin/google-chrome-stable",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
)


def _strip_fence(block: str) -> str:
    """Return the raw Mermaid source inside a ```mermaid fence (or the input)."""
    m = _FENCE_RE.match(block or "")
    return (m.group("src") if m else (block or "")).strip()


def _find_mmdc(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    found = shutil.which("mmdc")
    if found:
        return found
    # repo-local install fallback
    local = Path(__file__).resolve().parents[2] / "node_modules" / ".bin" / "mmdc"
    return str(local) if local.is_file() else None


def _find_chrome(explicit: str | None = None) -> str | None:
    if explicit and Path(explicit).exists():
        return explicit
    env = os.environ.get("PUPPETEER_EXECUTABLE_PATH")
    if env and Path(env).exists():
        return env
    for cand in _CHROME_CANDIDATES:
        if Path(cand).exists():
            return cand
    return shutil.which("google-chrome") or shutil.which("chromium")


def _default_runner(cmd: list[str], env: dict[str, str], timeout: float) -> int:
    return subprocess.run(  # noqa: S603 — fixed argv, no shell
        cmd, env=env, timeout=timeout, capture_output=True, check=False
    ).returncode


def mermaid_to_png(
    mermaid_block: str,
    out_path: Path,
    *,
    runner: RenderRunner | None = None,
    timeout: float = 60.0,
    mmdc_path: str | None = None,
    chrome_path: str | None = None,
) -> Path | None:
    """Render a Mermaid block to ``out_path`` (PNG). Returns the path, or ``None``
    when rendering is unavailable/failed (caller degrades — never raises).

    Drives ``mmdc`` with the system Chrome (``PUPPETEER_EXECUTABLE_PATH``) and a
    ``--no-sandbox`` puppeteer config (headless server-safe). ``runner`` is
    injectable for hermetic tests.
    """
    src = _strip_fence(mermaid_block)
    if not src:
        return None
    mmdc = _find_mmdc(mmdc_path)
    if not mmdc:
        logger.info("diagram render degrade: mmdc not found")
        return None

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run = runner or _default_runner

    tmp_dir = Path(tempfile.mkdtemp(prefix="mmdc_"))
    try:
        in_mmd = tmp_dir / "diagram.mmd"
        in_mmd.write_text(src, encoding="utf-8")
        # Headless Chrome on a server usually needs --no-sandbox.
        pptr_cfg = tmp_dir / "puppeteer.json"
        pptr_cfg.write_text(json.dumps({"args": ["--no-sandbox"]}), encoding="utf-8")

        cmd = [
            mmdc,
            "-i",
            str(in_mmd),
            "-o",
            str(out_path),
            "-b",
            "white",
            "-p",
            str(pptr_cfg),
        ]
        env = dict(os.environ)
        chrome = _find_chrome(chrome_path)
        if chrome:
            env["PUPPETEER_EXECUTABLE_PATH"] = chrome

        try:
            code = run(cmd, env, timeout)
        except subprocess.TimeoutExpired:
            logger.info("diagram render degrade: mmdc timeout after %.0fs", timeout)
            return None
        except Exception as exc:  # noqa: BLE001 — any render error degrades
            logger.info("diagram render degrade: runner error: %s", exc)
            return None

        if code != 0:
            logger.info("diagram render degrade: mmdc exit %d", code)
            return None
        if not out_path.is_file() or out_path.stat().st_size == 0:
            logger.info("diagram render degrade: no PNG produced")
            return None
        return out_path
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def diagram_caption(ref: RepoRef, graph_overview: dict | None) -> str:
    """Deterministic text proxy describing the diagram, for multimodal search.

    Built only from real graph facts (no fabrication). When no graph is
    available, returns a minimal, honest caption."""
    if not graph_overview:
        return f"Architecture diagram of {ref.owner}/{ref.repo} (code-graph unavailable)."
    totals = graph_overview.get("totals", {})
    contexts = [c.get("name", "?") for c in graph_overview.get("contexts", [])[:8]]
    hubs = [h.get("file", "?") for h in graph_overview.get("top_hubs", [])[:5]]
    parts = [f"Architecture diagram of {ref.owner}/{ref.repo}."]
    if contexts:
        parts.append(
            f"{totals.get('contexts', len(contexts))} bounded contexts: {', '.join(contexts)}."
        )
    if hubs:
        parts.append(f"Key modules (graph hubs): {', '.join(hubs)}.")
    if "encapsulation_pct" in totals:
        parts.append(f"{totals['encapsulation_pct']}% intra-context encapsulation.")
    return " ".join(parts)


def caption_markdown(ref: RepoRef, caption: str, image_relpath: str, *, date: str) -> str:
    """A standalone artifact for the diagram image — its OWN modality (a text
    proxy for search), deliberately NOT inlined into the repo source page."""
    return (
        "---\n"
        f'title: "Architecture diagram — {ref.owner}/{ref.repo}"\n'
        f"date: {date}\n"
        f'sources: ["{ref.canonical_url}"]\n'
        "origin: diagram-caption\n"
        f"image: {image_relpath}\n"
        f"repo: {ref.owner}/{ref.repo}\n"
        "tags: [diagram, architecture, repo, image]\n"
        "category: assets\n"
        "---\n\n"
        f"# Architecture diagram — {ref.owner}/{ref.repo}\n\n"
        f"![Architecture diagram]({Path(image_relpath).name})\n\n"
        f"{caption}\n"
    )
