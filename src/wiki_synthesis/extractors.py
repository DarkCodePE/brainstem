"""Deterministic entity/concept extraction — no LLM, no network.

The Hermes batch prompt asked the LLM to "extract key entities (people,
tools, projects) and concepts (ideas, patterns, frameworks)". This
module is the deterministic approximation: proper-noun sequences and
code identifiers become entities; pattern/framework-suffixed noun
phrases become concepts. Identical input always yields identical
output (Counter + stable sort), which is what makes the degrade path
testable.

This is the DEGRADE path (issue #180): when a router is available,
:mod:`wiki_synthesis.structured` does the extraction with one LLM call
and these heuristics are not consulted. To keep degrade output clean, a
generic-term blacklist plus a minimum-quality rule (multi-word proper
nouns, code identifiers, or consistently-capitalized words mentioned
≥3 times) filter the noisy acronym candidates (``LLM``, ``GPU``,
``KV``...) that polluted the first live batch.
"""

from __future__ import annotations

import re
from collections import Counter

__all__ = [
    "extract_concepts",
    "extract_entities",
    "extract_image_refs",
    "extract_urls",
    "strip_frontmatter",
]

_FRONTMATTER_RE = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)
_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
_URL_RE = re.compile(r"https?://[^\s)\]>\"']+")
# Markdown image ![alt](url) and Obsidian embed ![[file]] forms.
_IMAGE_MD_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_IMAGE_EMBED_RE = re.compile(r"!\[\[[^\]]+\]\]")

# Proper-noun sequence: capitalized words optionally joined by space/hyphen.
_PROPER_RE = re.compile(r"\b([A-Z][A-Za-z0-9+#.]*(?:[ -][A-Z][A-Za-z0-9+#.]*)*)\b")
# CamelCase / PascalCase identifier.
_CAMEL_RE = re.compile(r"\b([A-Z][a-z0-9]+(?:[A-Z][A-Za-z0-9]*)+)\b")
# snake_case / dotted identifiers from inline code.
_IDENT_RE = re.compile(r"\b([A-Za-z][\w.-]*[\w])\b")

# A concept is 1-3 lowercase words ending in a pattern/idea suffix.
_CONCEPT_RE = re.compile(
    r"\b((?:[a-z][\w-]*\s+){1,2}"
    r"(?:pattern|architecture|framework|pipeline|workflow|strategy|protocol|"
    r"approach|method|methodology|engineering|learning|management|graph|"
    r"system|loop|principle)s?)\b"
)

_STOPWORDS = frozenset(
    """the this that these those a an i it in on at we you he she they and but
    or if when what how why not no yes my your our their its is are was were
    be been with for from by as of to do does did so then there here also use
    using new one two more most some all each every after before while since
    however because although key main note example summary update today""".split()
)

_MAX_TERM_LEN = 60

# Generic technical terms with no standalone knowledge value — junk
# entity pages from the 2026-06-10 live batch (issue #180). Matched
# case-insensitively against the WHOLE candidate term.
_GENERIC_BLACKLIST = frozenset(
    """ai api apis app apps cli cpu cpus css db dbs dns gpu gpus html http
    https ide ides ip json kv llm llms os pdf ram repo repos sdk sdks sql
    ssd tldr ui url urls ux vm vms xml yaml yml""".split()
)

# Single capitalized prose words must be mentioned at least this often
# (and never appear lowercase) to qualify as an entity.
_MIN_SINGLE_WORD_MENTIONS = 3


def strip_frontmatter(text: str) -> str:
    """Drop a leading YAML frontmatter block, if present."""
    return _FRONTMATTER_RE.sub("", text, count=1)


def extract_image_refs(text: str) -> list[str]:
    """All image references, both ``![alt](url)`` and ``![[embed]]``,
    in document order (deduplicated, first occurrence wins)."""
    refs = _IMAGE_MD_RE.findall(text) + _IMAGE_EMBED_RE.findall(text)
    return _dedupe(refs)


def extract_urls(text: str) -> list[str]:
    """All http(s) URLs in document order (deduplicated). Trailing
    punctuation that markdown prose glues onto a URL is trimmed."""
    urls = [u.rstrip(".,;:!?") for u in _URL_RE.findall(text)]
    return _dedupe(urls)


def extract_entities(text: str, *, limit: int = 5) -> list[str]:
    """Deterministic entity candidates (people, tools, projects).

    Sources, in priority order:
    - CamelCase identifiers and inline-code identifiers (tool names).
    - Multi-word proper-noun sequences ("Claude Code", "Second Brain").
    - Single capitalized words that are *consistently* capitalized in
      the text (never appear lowercase) and occur at least
      ``_MIN_SINGLE_WORD_MENTIONS`` times — a cheap sentence-start
      filter.

    Candidates whose whole term is a blacklisted generic acronym
    (``LLM``, ``GPU``, ``KV``...) are dropped regardless of frequency.
    Ranked by frequency desc, then alphabetically (stable). Capped at
    ``limit``.
    """
    body = strip_frontmatter(text)
    body = _IMAGE_MD_RE.sub(" ", body)
    body = _IMAGE_EMBED_RE.sub(" ", body)
    body = _URL_RE.sub(" ", body)

    inline_codes: list[str] = []
    for code in _INLINE_CODE_RE.findall(body):
        code = code.strip()
        if 1 < len(code) <= _MAX_TERM_LEN and _IDENT_RE.fullmatch(code):
            inline_codes.append(code)
    prose = _CODE_FENCE_RE.sub(" ", body)
    prose_no_inline = _INLINE_CODE_RE.sub(" ", prose)

    counts: Counter[str] = Counter()
    lower_words = set(re.findall(r"\b[a-z][a-z0-9]+\b", prose_no_inline))

    for term in inline_codes + _CAMEL_RE.findall(prose):
        counts[term] += 1

    for term in _PROPER_RE.findall(prose_no_inline):
        term = term.strip()
        if not term or len(term) > _MAX_TERM_LEN:
            continue
        words = term.split()
        if words[0].lower() in _STOPWORDS:
            continue
        if len(words) >= 2:
            counts[term] += 1
        elif term.lower() not in lower_words and term.lower() not in _STOPWORDS:
            counts[term] += 1

    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    out: list[str] = []
    for term, count in ranked:
        if term.lower() in _GENERIC_BLACKLIST:
            continue
        if not _passes_quality(term, count):
            continue
        out.append(term)
        if len(out) >= limit:
            break
    return out


def _passes_quality(term: str, count: int) -> bool:
    """Minimum-quality rule (issue #180): keep only multi-word proper
    nouns, code identifiers (CamelCase / snake_case / dotted), or
    single capitalized words mentioned ≥ ``_MIN_SINGLE_WORD_MENTIONS``
    times."""
    if " " in term:
        return True
    if _CAMEL_RE.fullmatch(term) or "_" in term or "." in term:
        return True
    return count >= _MIN_SINGLE_WORD_MENTIONS


def extract_concepts(text: str, *, limit: int = 5) -> list[str]:
    """Deterministic concept candidates (ideas, patterns, frameworks).

    Lowercase noun phrases ending in a pattern/idea suffix word
    ("second brain architecture", "event sourcing pattern"). Ranked by
    frequency desc, then alphabetically. Capped at ``limit``.
    """
    body = strip_frontmatter(text)
    body = _CODE_FENCE_RE.sub(" ", body)
    body = _URL_RE.sub(" ", body)
    lowered = body.lower()

    counts: Counter[str] = Counter()
    for phrase in _CONCEPT_RE.findall(lowered):
        phrase = " ".join(phrase.split())
        if not phrase or len(phrase) > _MAX_TERM_LEN:
            continue
        first = phrase.split()[0]
        if first in _STOPWORDS:
            phrase = " ".join(phrase.split()[1:])
            if " " not in phrase:
                continue
        if phrase.split()[0] in _GENERIC_BLACKLIST:
            continue  # "llm pipeline" et al — generic-led phrases (issue #180).
        counts[phrase] += 1

    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [term for term, _ in ranked[:limit]]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out
