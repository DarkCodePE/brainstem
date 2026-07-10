"""``wiki_synthesis`` — in-repo port of the Hermes batch-ingest synthesis leg.

ADR-035 D3: the rich synthesis the Hermes ``wiki-batch-ingest`` cron
performed (source page + entity pages + concept pages + index entries +
log entry) moves into the repo, under tests and CI, invoked as a
post-write hook of the ingest worker.

Extraction is router-driven when a router is wired (ONE structured
call per file — summary + entities + concepts as JSON, issue #180),
with the deterministic heuristics as the degrade path on ANY failure,
and an honest ``origin`` marker either way.
"""

from wiki_synthesis.agent import SynthesisAgent, SynthesisOutcome
from wiki_synthesis.hooks import CompositePostWriteHook, SynthesisOnIngestHook
from wiki_synthesis.reconcile import Accretion, accrete_mention_page, accrete_source_page

__all__ = [
    "Accretion",
    "CompositePostWriteHook",
    "SynthesisAgent",
    "SynthesisOnIngestHook",
    "SynthesisOutcome",
    "accrete_mention_page",
    "accrete_source_page",
]
