#!/usr/bin/env python3
"""ADR-044 explainer spot-check harness — probabilistic acceptance gate.

ADR-044's acceptance criteria for the ``explainer`` archetype are probabilistic
and human-rated over N≥30 drafts:

  - AC-1 **on-archetype**   ≥ 0.85  (reads as a structured third-party explainer)
  - AC-2 **factual**        ≥ 0.90  (0 invented numbers/claims) — MACHINE pre-screened
  - AC-3 **publishable**    ≥ 0.85  (publishable with minor edits or better)

This mirrors ``scripts/ac3_spotcheck_history.py``: ``generate`` produces N drafts,
auto-scores AC-2 with :func:`wiki_publishing.explainer_quality.score_explainer`
(the only objective band), and writes a fillable Markdown checklist where a human
ticks AC-1 / AC-3; ``tally`` reads the filled checklist and gates the three bands.

Draft generation is injected (``draft_fn``) so the parsing/gate logic is testable
with zero LLM. The CLI's default ``draft_fn`` builds the live REASONING-tier
drafter — running ``generate`` for real costs one draft call per topic.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from wiki_publishing.explainer_quality import score_explainer  # noqa: E402

# Probabilistic acceptance bands (ADR-044 "Acceptance criteria").
_BANDS = {"on_archetype": 0.85, "factual": 0.90, "publishable": 0.85}

#: A draft generator: topic -> object with ``.body`` and ``.sources`` (a
#: LinkedInDraft, or any duck-typed stand-in in tests).
DraftFn = Callable[[str], Any]

_AC1_RE = re.compile(r"-\s*\[([ xX])\]\s*AC-1", re.MULTILINE)
_AC3_RE = re.compile(r"-\s*\[([ xX])\]\s*AC-3", re.MULTILINE)
_AC2_RE = re.compile(r"-\s*AC-2[^\n]*?:\s*(GROUNDED|UNGROUNDED)", re.IGNORECASE)


def _cta_exempt_values(cta: dict[str, str] | None) -> list[str]:
    """Verbatim author-supplied CTA strings exempt from AC-2 grounding."""
    return [v for v in (cta or {}).values() if v and v.strip()]


def _block(index: int, topic: str, draft: Any, exempt: Sequence[str]) -> str:
    """One checklist entry: AC-2 auto verdict + human AC-1/AC-3 boxes + the draft."""
    body = getattr(draft, "body", "") or ""
    sources = getattr(draft, "sources", ()) or ()
    ac2 = score_explainer(body, sources, exempt_values=exempt)
    if ac2.verdict == "grounded":
        ac2_line = "- AC-2 (factual, auto): GROUNDED ✅"
    else:
        nums = ", ".join(ac2.ungrounded_numbers)
        ac2_line = f"- AC-2 (factual, auto): UNGROUNDED ❌ — check: {nums}"
    return (
        f"## {index}. {topic}\n\n"
        f"{ac2_line}\n"
        "- [ ] AC-1 on-archetype (structured third-party explainer: hook → labelled "
        "blocks → named insight → closer)\n"
        "- [ ] AC-3 publishable (with minor edits or better)\n"
        "- Notes: \n\n"
        "**Draft:**\n\n"
        f"{body.rstrip()}\n"
    )


def generate(
    out_path: Path,
    topics: Sequence[str],
    draft_fn: DraftFn,
    *,
    cta: dict[str, str] | None = None,
) -> int:
    """Generate drafts for ``topics``, auto-score AC-2, write a fillable checklist."""
    exempt = _cta_exempt_values(cta)
    blocks: list[str] = []
    for i, topic in enumerate(topics, 1):
        try:
            draft = draft_fn(topic)
        except Exception as exc:  # noqa: BLE001 — one bad topic must not abort the run.
            blocks.append(f"## {i}. {topic}\n\n> ⚠️ draft generation failed: {exc}\n")
            print(f"[adr044]  {i:>2}. {topic}: FAILED ({exc})", file=sys.stderr)
            continue
        blocks.append(_block(i, topic, draft, exempt))
        print(f"[adr044]  {i:>2}. {topic}: drafted", file=sys.stderr)

    n = len(topics)
    doc = (
        "---\n"
        "title: ADR-044 explainer spot-check — probabilistic acceptance\n"
        "adr: ADR-044\n"
        'gate: "on-archetype ≥0.85, factual ≥0.90, publishable ≥0.85"\n'
        "status: pending-human-judgement\n"
        "---\n\n"
        "# ADR-044 explainer spot-check\n\n"
        "For each draft: AC-2 (factual) is pre-scored automatically — an UNGROUNDED "
        "verdict lists numbers to verify against the cited sources. Read the draft and "
        "tick `- [x] AC-1` if it reads as a structured third-party explainer, and "
        "`- [x] AC-3` if it's publishable with minor edits or better. Then run:\n\n"
        f"```bash\npython scripts/adr044_explainer_spotcheck.py --tally {out_path.name}\n```\n\n"
        "**Gate:** on-archetype ≥85%, factual ≥90%, publishable ≥85% "
        f"(N={n}; ADR-044 wants N≥30 for a real read).\n\n"
        "---\n\n" + "\n---\n\n".join(blocks) + "\n"
    )
    out_path.write_text(doc, encoding="utf-8")
    print(f"\n[adr044] checklist written → {out_path}", file=sys.stderr)
    return 0


def tally(path: Path) -> int:
    """Score a filled-in checklist against the three ADR-044 bands."""
    text = path.read_text(encoding="utf-8")
    ac1 = _AC1_RE.findall(text)
    ac3 = _AC3_RE.findall(text)
    ac2 = _AC2_RE.findall(text)
    total = len(ac1)
    if total == 0:
        print(f"[adr044] no `- [ ] AC-1` lines in {path} — is it the generated checklist?")
        return 2

    rates = {
        "on_archetype": sum(1 for m in ac1 if m.lower() == "x") / total,
        "publishable": sum(1 for m in ac3 if m.lower() == "x") / len(ac3) if ac3 else 0.0,
        "factual": sum(1 for v in ac2 if v.upper() == "GROUNDED") / len(ac2) if ac2 else 0.0,
    }
    print(f"ADR-044 explainer spot-check tally ({path.name}, N={total})")
    all_pass = True
    for band, threshold in _BANDS.items():
        rate = rates[band]
        ok = rate >= threshold
        all_pass = all_pass and ok
        print(f"  {band:<13} {rate:.0%}  (gate ≥{threshold:.0%})  {'PASS ✅' if ok else 'FAIL ❌'}")
    print(f"  verdict: {'PASS ✅' if all_pass else 'FAIL ❌'}")
    return 0 if all_pass else 1


def _live_draft_fn(cta: dict[str, str] | None) -> DraftFn:
    """Build the real explainer drafter (REASONING tier). One LLM call per topic."""
    import asyncio
    import os

    from wiki_publishing import NewsletterCTA, ProductPS  # noqa: PLC0415
    from wiki_publishing.linkedin_draft import (  # noqa: PLC0415
        LinkedInDraftGenerator,
        WikiContentSource,
    )
    from wiki_routing import default_router  # noqa: PLC0415

    wiki_root = Path(os.environ.get("WIKI_ROOT", _REPO_ROOT / "knowledge-base"))
    gen = LinkedInDraftGenerator(
        router=default_router(), content_source=WikiContentSource(wiki_root=wiki_root)
    )
    kwargs: dict[str, Any] = {"post_type": "explainer", "focus": "use"}
    cta = cta or {}
    if cta.get("newsletter_url"):
        kwargs["newsletter_cta"] = NewsletterCTA(
            url=cta["newsletter_url"], proof=cta.get("newsletter_proof") or None
        )
    if cta.get("product_name") and cta.get("product_pitch"):
        kwargs["product_ps"] = ProductPS(
            name=cta["product_name"], pitch=cta["product_pitch"], url=cta.get("product_url") or None
        )

    def run(topic: str) -> Any:
        return asyncio.run(gen.generate(topic, **kwargs))

    return run


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="ADR-044 explainer spot-check harness.")
    p.add_argument(
        "--out", type=Path, default=_REPO_ROOT / "docs" / "ADR044-explainer-spotcheck.md"
    )
    p.add_argument("--tally", type=Path, default=None, help="score a filled checklist instead.")
    p.add_argument(
        "--topics-file", type=Path, default=None, help="one topic per line (generate mode)."
    )
    p.add_argument("--newsletter-url", default="")
    p.add_argument("--newsletter-proof", default="")
    p.add_argument("--product-name", default="")
    p.add_argument("--product-pitch", default="")
    p.add_argument("--product-url", default="")
    args = p.parse_args(argv)

    if args.tally is not None:
        return tally(args.tally)

    if not args.topics_file or not args.topics_file.is_file():
        print("[adr044] generate mode needs --topics-file (one topic per line).", file=sys.stderr)
        return 2
    topics = [
        ln.strip() for ln in args.topics_file.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]
    cta = {
        "newsletter_url": args.newsletter_url,
        "newsletter_proof": args.newsletter_proof,
        "product_name": args.product_name,
        "product_pitch": args.product_pitch,
        "product_url": args.product_url,
    }
    return generate(args.out, topics, _live_draft_fn(cta), cta=cta)


if __name__ == "__main__":
    raise SystemExit(main())
