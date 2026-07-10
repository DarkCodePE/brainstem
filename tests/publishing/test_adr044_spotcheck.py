"""Tests for the ADR-044 explainer spot-check harness (parsing + gate, no LLM)."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "adr044_explainer_spotcheck.py"
_spec = importlib.util.spec_from_file_location("adr044_spotcheck", _SCRIPT)
assert _spec and _spec.loader
spotcheck = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(spotcheck)


@dataclass
class _Draft:
    body: str
    sources: tuple = field(default_factory=tuple)


@dataclass
class _Snip:
    body: str
    title: str = ""


# --------------------------------------------------------------------------- #
# generate
# --------------------------------------------------------------------------- #


def test_generate_writes_checklist_with_ac2_prescreen(tmp_path: Path) -> None:
    drafts = {
        "HNSW": _Draft("Explainer.\n\n- ➡️ point\n\nInsight?", (_Snip("hnsw concept"),)),
        "RAG": _Draft("Used by 9999+ teams.", (_Snip("rag, no numbers"),)),  # invented stat
    }
    out = tmp_path / "checklist.md"
    rc = spotcheck.generate(out, list(drafts), lambda t: drafts[t])
    assert rc == 0
    text = out.read_text()
    assert "GROUNDED ✅" in text  # the clean HNSW draft
    assert "UNGROUNDED ❌" in text and "9999" in text  # the invented stat surfaced
    assert text.count("- [ ] AC-1") == 2 and text.count("- [ ] AC-3") == 2


def test_generate_survives_a_failing_topic(tmp_path: Path) -> None:
    def draft_fn(topic: str):
        if topic == "boom":
            raise RuntimeError("router down")
        return _Draft("fine", (_Snip("x"),))

    out = tmp_path / "c.md"
    assert spotcheck.generate(out, ["ok", "boom"], draft_fn) == 0
    assert "draft generation failed" in out.read_text()


def test_cta_values_exempt_from_ac2(tmp_path: Path) -> None:
    # the 4000+ count lives only in the verbatim CTA → must NOT be flagged.
    d = _Draft(
        "Concept.\n\nLa versión detallada (leído por 4000+): https://n",
        (_Snip("concept, no numbers"),),
    )
    out = tmp_path / "c.md"
    spotcheck.generate(out, ["T"], lambda _t: d, cta={"newsletter_proof": "leído por 4000+"})
    assert "GROUNDED ✅" in out.read_text()


# --------------------------------------------------------------------------- #
# tally
# --------------------------------------------------------------------------- #


def _checklist(entries: list[tuple[str, str, str]]) -> str:
    """entries = list of (ac1_mark, ac3_mark, ac2_verdict)."""
    blocks = []
    for i, (a1, a3, a2) in enumerate(entries, 1):
        blocks.append(
            f"## {i}. topic\n\n- AC-2 (factual, auto): {a2}\n"
            f"- [{a1}] AC-1 on-archetype\n- [{a3}] AC-3 publishable\n"
        )
    return "# spot-check\n\n" + "\n---\n\n".join(blocks) + "\n"


def test_tally_all_bands_pass(tmp_path: Path) -> None:
    # 10 entries all ticked + grounded → 100% on every band → PASS (rc 0)
    p = tmp_path / "filled.md"
    p.write_text(_checklist([("x", "x", "GROUNDED")] * 10))
    assert spotcheck.tally(p) == 0


def test_tally_fails_when_a_band_below_threshold(tmp_path: Path) -> None:
    # on-archetype only 5/10 = 50% < 85% → FAIL (rc 1)
    entries = [("x", "x", "GROUNDED")] * 5 + [(" ", "x", "GROUNDED")] * 5
    p = tmp_path / "f.md"
    p.write_text(_checklist(entries))
    assert spotcheck.tally(p) == 1


def test_tally_factual_band_uses_auto_verdict(tmp_path: Path) -> None:
    # all human boxes ticked but 2/10 ungrounded → factual 80% < 90% → FAIL
    entries = [("x", "x", "GROUNDED")] * 8 + [("x", "x", "UNGROUNDED")] * 2
    p = tmp_path / "f.md"
    p.write_text(_checklist(entries))
    assert spotcheck.tally(p) == 1


def test_tally_rejects_non_checklist(tmp_path: Path) -> None:
    p = tmp_path / "nope.md"
    p.write_text("just some markdown, no checklist markers")
    assert spotcheck.tally(p) == 2
