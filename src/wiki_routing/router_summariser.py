"""
``RouterSummariser`` — a ``wiki_memory.summariser.Summariser`` backed
by the ``ModelRouter`` per [PRD-004 FR-4](../../docs/PRD-004-memory-tree.md)
+ [PRD-008](../../docs/PRD-008-model-routing.md).

This is the integration win that turns the M2 seal worker from a
deterministic stub (``NullSummariser``) into an LLM-driven path
without changing any code on the seal worker side. The seal worker
already takes a ``Summariser`` in its constructor; this class
satisfies that Protocol while delegating the actual generation to the
3-tier router.

### Citation extraction (M3-S3: closes SPEC-009 OQ-2)

Two paths are supported, selected by the ``output_format`` constructor
arg:

- ``"json"`` (default) — the prompt instructs the model to emit a JSON
  object ``{"body": "...", "cited_shas": [full-sha-1, ...]}``. The
  parser is tolerant: it accepts pure JSON, JSON in a ```` ```json ````
  fence, and JSON preceded by an explanation line. Cited shas are
  validated against the input parts; unknown shas are dropped (the
  seal worker's faithfulness gate would refuse them anyway).
- ``"substring"`` — the legacy path. The prompt asks the model to cite
  with ``[[chunk:SHA8]]`` markers in prose. The parser scans the
  response for any short-sha that matches an input part.

When ``output_format="json"`` and ``strict_json=False`` (the default),
a malformed JSON response falls back to substring extraction. With
``strict_json=True`` a malformed response raises ``ValueError`` — used
by tests and by deployments that want to fail loudly on provider
drift.

### Why this lives in ``wiki_routing`` and not ``wiki_memory``

The dependency direction matters: ``wiki_memory`` declares the
``Summariser`` Protocol and **must not** import the router (that
would couple memory to a specific LLM strategy and break the
abstraction PRD-004 R-1 mitigations are built on). The router is
free to depend on ``wiki_memory`` for the seal-summary use case
because the dependency points downwards (LLM substrate depends on
the storage shape it summarises).

### Faithfulness post-condition

The seal worker re-checks the cited shas after this summariser
returns (``SealWorker._call_summariser``), so a hallucinated
citation refuses the seal cleanly. We do **not** try to enforce
faithfulness here — that's a single-source-of-truth boundary the
seal worker already owns. The JSON path raises the bar by letting
the model cite by **full** sha (no short-sha collision class),
but the substring fallback remains in place because real providers
drift in their structured-output adherence.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from typing import Literal

from wiki_memory.summariser import SummaryPart, SummaryResult
from wiki_routing.policy import TaskDescriptor
from wiki_routing.router import Message, ModelRouter

OutputFormat = Literal["json", "substring"]


_SUMMARY_SYSTEM_PROMPT_JSON = (
    "You are the Second Brain Wiki's memory-tree summariser. Given a list "
    "of child chunks (each identified by a full sha256 in its header), "
    "produce a concise Markdown summary that:\n"
    "1. Captures the union of facts in the chunks (no new claims).\n"
    "2. Cites each fact inline using [[chunk:SHA8]] (the first 8 chars of "
    "the chunk's full sha).\n"
    "3. Does NOT cite shas that were not provided.\n"
    "Keep the summary under ~3000 tokens.\n\n"
    "Respond with a JSON object of the form:\n"
    '{"body": "<your markdown summary>", '
    '"cited_shas": [<list of FULL sha strings actually used>]}\n'
    "No prose outside the JSON. Do not wrap the JSON in code fences "
    "unless your provider requires it."
)

_SUMMARY_SYSTEM_PROMPT_SUBSTRING = (
    "You are the Second Brain Wiki's memory-tree summariser. Given a list "
    "of child chunks (each identified by a short sha), produce a concise "
    "Markdown summary that:\n"
    "1. Captures the union of facts in the chunks (no new claims).\n"
    "2. Cites each fact with the short-sha in the form [[chunk:SHA8]].\n"
    "3. Does NOT cite shas that were not provided.\n"
    "Keep the summary under ~3000 tokens."
)


def _short(sha: str) -> str:
    return sha[:8]


def _render_user_prompt_json(parts: Sequence[SummaryPart]) -> str:
    """Build the user-message body for the JSON-mode path.

    Each chunk is presented under a ``### CHUNK <full-sha>`` header so
    the model has the full citation key in front of it for each fact.
    The full sha is what the JSON ``cited_shas`` list refers back to;
    the short-sha is what ends up in the rendered ``[[chunk:SHA8]]``
    citations inside ``body``.
    """
    lines = [
        "Summarise the following chunks. Cite each fact inline as "
        "[[chunk:SHA8]] (first 8 chars of the chunk's full sha). "
        "Then respond with the JSON object described in the system "
        "prompt — its `cited_shas` list contains the FULL sha of each "
        "chunk you actually cited.",
        "",
    ]
    for part in parts:
        lines.append(f"### CHUNK {part.sha256} ({part.token_count} tokens)")
        lines.append(part.body)
        lines.append("")
    return "\n".join(lines)


def _render_user_prompt_substring(parts: Sequence[SummaryPart]) -> str:
    """Build the user-message body for the legacy substring path."""
    lines = ["Summarise the following chunks. Cite each as [[chunk:SHA8]].", ""]
    for part in parts:
        lines.append(f"### chunk:{_short(part.sha256)} ({part.token_count} tokens)")
        lines.append(part.body)
        lines.append("")
    return "\n".join(lines)


def _hash(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


# Code-fence wrappers we tolerate around JSON output. Some providers
# default to fenced output even when told not to.
_FENCE_RE = re.compile(
    r"```(?:json|JSON)?\s*\n?(?P<body>.*?)\n?```",
    re.DOTALL,
)


def _extract_json_object(text: str) -> object:
    """Pull the first JSON object out of ``text``.

    Handles three shapes:

    1. Pure JSON — ``json.loads`` directly.
    2. JSON inside ```` ```json ... ``` ```` fences.
    3. JSON preceded (or trailed) by prose — the first balanced
       ``{...}`` block is parsed.

    Raises ``ValueError`` if nothing parseable is found.
    """
    stripped = text.strip()
    if not stripped:
        raise ValueError("empty response from model")

    # Path 1: pure JSON.
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Path 2: fenced code block.
    fence_match = _FENCE_RE.search(stripped)
    if fence_match is not None:
        candidate = fence_match.group("body").strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass  # fall through to the brace scan

    # Path 3: scan for the first balanced JSON object. We can't just
    # regex ``\{.*\}`` greedily because the body field may contain
    # braces; walk the string tracking string state.
    obj_str = _scan_first_object(stripped)
    if obj_str is not None:
        try:
            return json.loads(obj_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"found JSON-shaped block but it failed to parse: {e}") from e

    raise ValueError("no JSON object found in model response")


def _scan_first_object(text: str) -> str | None:
    """Return the first balanced ``{...}`` substring in ``text``.

    Respects string literals (so ``{"body": "has } in it"}`` is
    handled correctly). Returns ``None`` if no balanced object is
    found.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


class RouterSummariser:
    """``Summariser`` implementation that dispatches to ``ModelRouter``.

    Parameters
    ----------
    router:
        Live ``ModelRouter`` instance. The summariser doesn't own
        configuration — the router does — so swapping out the
        provider chain for the Reasoning tier is a one-line edit
        in the agent wiring, not here.
    intent:
        ``TaskDescriptor`` intent for the dispatched call. Defaults
        to ``"seal"`` because that is the entire reason this class
        exists. Override via the constructor if a non-seal call site
        wants to reuse the same prompt template (rare).
    caller_priority:
        Pass-through to ``TaskDescriptor``. Defaults to
        ``"background"`` because seal work is daemon-scheduled.
    output_format:
        ``"json"`` (default) drives the structured-output path that
        closes SPEC-009 OQ-2. ``"substring"`` keeps the legacy
        ``[[chunk:SHA8]]``-scanning behaviour for callers that
        explicitly want it (e.g. providers that can't reliably emit
        JSON).
    strict_json:
        Only meaningful when ``output_format="json"``. When ``True``,
        malformed JSON raises ``ValueError`` and the call fails. When
        ``False`` (default), malformed JSON falls back to substring
        extraction — the seal worker's faithfulness gate still
        enforces correctness on the way out.

    Notes
    -----
    The two parser paths converge on the same ``SummaryResult`` shape;
    the substring path stays available as a defensive fallback because
    real providers drift in their structured-output adherence (per
    SPEC-009 §"Risks and mitigations").
    """

    def __init__(
        self,
        *,
        router: ModelRouter,
        intent: str = "seal",
        caller_priority: str = "background",
        output_format: OutputFormat = "json",
        strict_json: bool = False,
    ) -> None:
        self._router = router
        # Stored as plain str rather than Intent/CallerPriority so
        # callers can override with a string from config without
        # importing the Literal types.
        self._intent = intent
        self._caller_priority = caller_priority
        if output_format not in ("json", "substring"):
            raise ValueError(f"output_format must be 'json' or 'substring', got {output_format!r}")
        self._output_format: OutputFormat = output_format
        self._strict_json = strict_json

    @property
    def output_format(self) -> OutputFormat:
        """Exposed read-only for tests."""
        return self._output_format

    @property
    def strict_json(self) -> bool:
        """Exposed read-only for tests."""
        return self._strict_json

    async def summarise(self, parts: Sequence[SummaryPart]) -> SummaryResult:
        """Run the Reasoning-tier model and shape the response into
        a ``SummaryResult``.

        Two corner cases worth calling out:

        - Empty ``parts``: we short-circuit with an empty-shape result
          (matches ``NullSummariser`` behaviour). The router is never
          called — burning a Reasoning-tier slot to summarise zero
          chunks would be silly.
        - Citations the model invented: dropped here, then re-checked
          by the seal worker's faithfulness gate. This summariser
          doesn't have to defend that boundary; we just report what
          we got.
        """
        if not parts:
            empty_body = "# Empty summary\n"
            return SummaryResult(
                body=empty_body,
                sha256=_hash(empty_body),
                parent_token_count=1,
                cited_shas=(),
            )

        if self._output_format == "json":
            system_prompt = _SUMMARY_SYSTEM_PROMPT_JSON
            user_body = _render_user_prompt_json(parts)
        else:
            system_prompt = _SUMMARY_SYSTEM_PROMPT_SUBSTRING
            user_body = _render_user_prompt_substring(parts)

        # Approx token count for the call site to feed to the policy.
        # 4-char/token heuristic mirrors the chunker.
        estimated_tokens = max(1, len(user_body) // 4)

        # Cast the literal strings via ``# type: ignore`` would be
        # ugly; just dataclass-construct it inline.
        task = TaskDescriptor(
            intent=self._intent,  # type: ignore[arg-type]
            estimated_input_tokens=estimated_tokens,
            has_image=False,
            caller_priority=self._caller_priority,  # type: ignore[arg-type]
        )

        result = await self._router.call(
            task,
            messages=[
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_body),
            ],
        )

        if self._output_format == "json":
            return self._parse_json_or_fallback(result.text, parts)
        return self._parse_substring(result.text, parts)

    # ------------------------------------------------------------------ #
    # Parsers                                                             #
    # ------------------------------------------------------------------ #

    def _parse_json_or_fallback(self, text: str, parts: Sequence[SummaryPart]) -> SummaryResult:
        """Parse a JSON-mode response.

        On any parse / shape error: if ``strict_json`` is True, raise
        ``ValueError``; otherwise delegate to the substring parser so
        the seal flow still produces a result the faithfulness gate
        can adjudicate.
        """
        try:
            parsed = _extract_json_object(text)
            body, cited_full = self._validate_json_shape(parsed, parts)
        except ValueError as e:
            if self._strict_json:
                raise ValueError(f"strict_json: failed to parse model response: {e}") from e
            return self._parse_substring(text, parts)

        return SummaryResult(
            body=body,
            sha256=_hash(body),
            parent_token_count=max(1, len(body) // 4),
            cited_shas=tuple(cited_full),
        )

    def _validate_json_shape(
        self, parsed: object, parts: Sequence[SummaryPart]
    ) -> tuple[str, list[str]]:
        """Enforce the ``{body: str, cited_shas: list[str]}`` contract
        and filter ``cited_shas`` to shas that appear in ``parts``.

        Returns ``(body, cited_full_shas)``. Raises ``ValueError`` if
        the shape is wrong.
        """
        if not isinstance(parsed, dict):
            raise ValueError(f"expected JSON object, got {type(parsed).__name__}")
        if "body" not in parsed:
            raise ValueError("JSON object missing required 'body' field")
        if "cited_shas" not in parsed:
            raise ValueError("JSON object missing required 'cited_shas' field")
        body = parsed["body"]
        if not isinstance(body, str):
            raise ValueError(f"'body' must be str, got {type(body).__name__}")
        cited_raw = parsed["cited_shas"]
        if not isinstance(cited_raw, list):
            raise ValueError(f"'cited_shas' must be list, got {type(cited_raw).__name__}")

        # Build a sha index for full-sha and short-sha lookup so the
        # model can cite by either form (full is what we asked for, but
        # being lenient on short is harmless and matches the substring
        # path's behaviour).
        full_index: dict[str, str] = {p.sha256: p.sha256 for p in parts}
        short_index: dict[str, str] = {_short(p.sha256): p.sha256 for p in parts}

        cited_full: list[str] = []
        for entry in cited_raw:
            if not isinstance(entry, str):
                # Bad entry — skip it rather than fail the whole result.
                # The seal worker's faithfulness gate is still the
                # source of truth.
                continue
            entry = entry.strip()
            if entry in full_index and full_index[entry] not in cited_full:
                cited_full.append(full_index[entry])
            elif entry in short_index and short_index[entry] not in cited_full:
                cited_full.append(short_index[entry])
            # else: hallucinated sha; drop silently.

        return body, cited_full

    def _parse_substring(self, text: str, parts: Sequence[SummaryPart]) -> SummaryResult:
        """Legacy ``[[chunk:SHA8]]`` substring path.

        Used directly when ``output_format="substring"`` and as a
        fallback when JSON parsing fails with ``strict_json=False``.
        """
        body = text
        sha = _hash(body)

        # Extract citations: any short-sha that appears in the model's
        # response AND was in the input set is considered cited.
        # Short-shas not in the input set are dropped here; the seal
        # worker's faithfulness gate would reject the summary anyway,
        # but truncating the cited tuple keeps the result honest
        # about what was actually grounded.
        input_shorts = {_short(p.sha256): p.sha256 for p in parts}
        cited: list[str] = []
        for short, full in input_shorts.items():
            if f"[[chunk:{short}]]" in body and full not in cited:
                cited.append(full)

        return SummaryResult(
            body=body,
            sha256=sha,
            parent_token_count=max(1, len(body) // 4),
            cited_shas=tuple(cited),
        )


__all__ = ["OutputFormat", "RouterSummariser"]
