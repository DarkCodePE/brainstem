"""Per-type body-quality scorer (ADR-048 Fase 1).

OKF ([[ADR-045]]) guarantees a page's *frontmatter* (``type`` + metadata) at the
[[ADR-035]] write boundary, but mandates **no body structure**. So page *bodies*
were never scored: a vault scan (879 pages, 2026-06-21) found 23.8% below
"genuine" — raw dumps, boilerplate, and stubs that pass every frontmatter check.

This module closes that gap with a **deterministic** scorer (no LLM, no network,
``$0``, reproducible — the same philosophy as :mod:`wiki_repos.quality`, but it
scores the *whole rendered body* against a per-type contract instead of just the
``## Evolution`` section). It dispatches on the OKF ``type`` and, for ``Source``
pages, a sub-type derived from tags/origin (``repo`` / ``paper`` / generic).

Verdict (one per page, priority ``no_signal > raw_dump > bloat > weak > genuine``):

- ``no_signal`` — stub or boilerplate-only (the only tier ADR-048 D4 will *skip*).
- ``raw_dump``  — body never went through synthesis (``<ingested_source>`` wrap,
  ``ingested-untrusted`` / ``repo-digest`` origin, or broken-extraction markers).
- ``bloat``     — a huge body (full PDF/README text shoved in as the page).
- ``weak``      — short/thin, or missing its type's required sections.
- ``genuine``   — a real synthesized body.

The numbers and thresholds here were calibrated on that scan; the scorer
reproduces its verdicts on real vault samples (see tests).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = [
    "BodyContract",
    "BodyQualityScore",
    "CONTRACTS",
    "classify_subtype",
    "parse_frontmatter",
    "score_body",
]

# --------------------------------------------------------------------------- #
# Tunables (calibrated on the 2026-06-21 vault scan)
# --------------------------------------------------------------------------- #
#: Prose floor (chars) below which a stub-shaped page is no_signal.
_STUB_PROSE = 200
#: Prose floor for entity/concept/source name-only stubs.
_TYPE_STUB_PROSE = 120
#: Body byte size above which a page is bloat (raw full-text dump).
_BLOAT_BYTES = 40_000

_STUB_RE = re.compile(r"^\s*Mentioned in \[\[", re.MULTILINE)
_FM_RE = re.compile(r"^---\n(.*?)\n---\n?(.*)$", re.DOTALL)
_SCALAR_RE = re.compile(r"^(\w[\w_]*):\s*(.*)$")
_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_WIKILINK_RE = re.compile(r"\[\[[^\]]*\]\]")

#: Markers of an unsynthesized / broken-extraction body.
_RAW_ORIGINS = frozenset({"ingested-untrusted", "repo-digest"})
_BROKEN_EXTRACT = ("could not be found", "image could not", "imagefile")
_BOILERPLATE = (
    "bootstrapped with `create-next-app`",
    "bootstrapped with [create-next-app",
    "create-next-app",
    "run the development server",
)


@dataclass(frozen=True, slots=True)
class BodyContract:
    """The body a page of a given (type, subtype) is expected to carry.

    ``required_sections`` are ``##`` headings the synthesizer emits for genuine
    pages; ``min_prose`` is the chars-of-prose floor below which the body is
    treated as thin (``weak``). ``forbid_raw`` marks types whose raw extraction
    must never *be* the page body (papers, repos)."""

    subtype: str
    required_sections: tuple[str, ...] = ()
    min_prose: int = 400
    forbid_raw: bool = False


#: Per-(sub)type body contracts (ADR-048 D1). Hung off the OKF ``type``.
CONTRACTS: dict[str, BodyContract] = {
    "repo": BodyContract("repo", ("What it is", "Capabilities"), min_prose=400, forbid_raw=True),
    "paper": BodyContract("paper", ("Abstract",), min_prose=400, forbid_raw=True),
    "source": BodyContract("source", (), min_prose=400),
    "entity": BodyContract("entity", (), min_prose=120),
    "concept": BodyContract("concept", (), min_prose=120),
    "observation": BodyContract("observation", (), min_prose=80),
}


@dataclass(frozen=True, slots=True)
class BodyQualityScore:
    """Deterministic body-quality result (ADR-048)."""

    score: float  # [0,1], informational; the verdict drives the decision
    verdict: str  # genuine | weak | bloat | raw_dump | no_signal
    subtype: str
    prose_len: int
    n_bytes: int
    notes: tuple[str, ...] = field(default_factory=tuple)


# --------------------------------------------------------------------------- #
# Parsing / classification
# --------------------------------------------------------------------------- #
def parse_frontmatter(content: str) -> tuple[dict[str, str], str, str]:
    """Light frontmatter split: ``(scalars, body, raw_fm_text)``.

    Stdlib-only (no yaml dep, mirroring the scan): scalar lines become a dict;
    the raw frontmatter text is returned too so callers can substring-match
    inline lists like ``tags: [a, b]`` without a real YAML parse."""
    m = _FM_RE.match(content)
    if not m:
        return {}, content, ""
    fm_raw, body = m.group(1), m.group(2)
    fm: dict[str, str] = {}
    for line in fm_raw.splitlines():
        mm = _SCALAR_RE.match(line)
        if mm:
            fm[mm.group(1)] = mm.group(2).strip()
    return fm, body, fm_raw


def prose_len(body: str) -> int:
    """Chars of real prose: drop headings, quotes, tables, links, list markup."""
    out: list[str] = []
    for ln in body.splitlines():
        s = ln.strip()
        if not s or s.startswith(("#", ">", "|")):
            continue
        s = _LINK_RE.sub(r"\1", s)
        s = _WIKILINK_RE.sub("", s)
        s = re.sub(r"[*_`#>-]", "", s)
        out.append(s.strip())
    return len("".join(out))


def classify_subtype(fm: dict[str, str], raw_fm: str, body: str) -> str:
    """OKF ``type`` wins first; only ``Source`` is sub-split into repo/paper.

    An Entity tagged ``github`` is still an Entity — the type, not the tags,
    decides everything except the repo/paper split inside ``Source``."""
    t = (fm.get("type") or "").lower()
    if t in ("entity", "concept", "observation"):
        return t
    if t == "source":
        tags_blob = (raw_fm + " " + body[:400]).lower()
        if "papers" in tags_blob or "arxiv_id" in raw_fm or "arxiv" in tags_blob:
            return "paper"
        if any(x in tags_blob for x in ("repo", "github", "codebase")):
            return "repo"
        return "source"
    return t or "other"


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
_VERDICT_SCORE = {"no_signal": 0.10, "raw_dump": 0.25, "bloat": 0.40, "weak": 0.55}


def _present_sections(body: str, required: tuple[str, ...]) -> int:
    if not required:
        return 0
    low = body.lower()
    return sum(
        1 for h in required if re.search(rf"^#+\s*{re.escape(h.lower())}", low, re.MULTILINE)
    )


def score_body(content: str) -> BodyQualityScore:
    """Score a full page (frontmatter + body) against its per-type contract."""
    fm, body, raw_fm = parse_frontmatter(content)
    subtype = classify_subtype(fm, raw_fm, body)
    contract = CONTRACTS.get(subtype)
    plen = prose_len(body)
    nbytes = len(content.encode("utf-8"))
    origin = (fm.get("origin") or "").lower()
    low = body.lower()
    notes: list[str] = []

    def result(verdict: str, score: float) -> BodyQualityScore:
        return BodyQualityScore(
            score=round(max(0.0, min(1.0, score)), 4),
            verdict=verdict,
            subtype=subtype,
            prose_len=plen,
            n_bytes=nbytes,
            notes=tuple(notes),
        )

    # 1. no_signal — stubs + boilerplate-only repos (the blocking tier).
    if _STUB_RE.search(body) and plen < _STUB_PROSE:
        notes.append("body is a 'Mentioned in [[…]]' stub")
        return result("no_signal", _VERDICT_SCORE["no_signal"])
    if subtype in ("entity", "concept", "source") and plen < _TYPE_STUB_PROSE:
        notes.append(f"prose {plen}c < {_TYPE_STUB_PROSE}c floor for {subtype}")
        return result("no_signal", _VERDICT_SCORE["no_signal"])
    if (
        subtype == "repo"
        and any(b in low for b in _BOILERPLATE)
        and "stars: 0" in low.replace("**", "")
    ):
        notes.append("0-star create-next-app boilerplate repo")
        return result("no_signal", _VERDICT_SCORE["no_signal"])

    # 2. raw_dump — body never went through synthesis.
    if "<ingested_source" in body or origin in _RAW_ORIGINS:
        notes.append(f"unsynthesized body (origin={origin or 'ingested_source wrap'})")
        return result("raw_dump", _VERDICT_SCORE["raw_dump"])
    if any(b in low for b in _BROKEN_EXTRACT):
        notes.append("broken-extraction markers (missing figures / OCR garbage)")
        return result("raw_dump", _VERDICT_SCORE["raw_dump"])

    # 3. bloat — full PDF/README text shoved in as the body.
    if nbytes > _BLOAT_BYTES:
        notes.append(f"{nbytes}B body exceeds {_BLOAT_BYTES}B (likely raw full-text)")
        if contract and contract.forbid_raw:
            notes.append(f"{subtype} contract forbids raw full-text as body")
        return result("bloat", _VERDICT_SCORE["bloat"])

    # 4. weak — short, or missing the type's required sections.
    if contract is not None:
        if plen < contract.min_prose:
            notes.append(f"prose {plen}c < {contract.min_prose}c floor for {subtype}")
            return result("weak", _VERDICT_SCORE["weak"])
        if contract.required_sections:
            present = _present_sections(body, contract.required_sections)
            if present < len(contract.required_sections):
                missing = len(contract.required_sections) - present
                notes.append(
                    f"missing {missing}/{len(contract.required_sections)} "
                    f"required section(s) for {subtype}: {contract.required_sections}"
                )
                return result("weak", _VERDICT_SCORE["weak"])
    elif plen < 400:  # unknown subtype: generic thinness floor
        return result("weak", _VERDICT_SCORE["weak"])

    # 5. genuine — scale score with prose richness (0.70..1.00).
    score = 0.70 + min(0.30, plen / 6000.0)
    return result("genuine", score)
