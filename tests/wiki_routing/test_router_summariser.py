"""Tests for ``wiki_routing.router_summariser.RouterSummariser`` — the
``wiki_memory.summariser.Summariser`` implementation backed by the
3-tier router.

Three layers of integration are covered:

1. RouterSummariser satisfies the ``Summariser`` Protocol via
   ``isinstance``. This is the contract the seal worker depends on.
2. The structured-JSON output path (SPEC-009 OQ-2 close-out): clean
   JSON, fenced JSON, JSON-with-leading-text, malformed-JSON
   fallback, strict-mode failure, and full-sha citation handling.
3. Pairing ``RouterSummariser`` with ``SealWorker`` + ``NullWriteSink``
   produces a sealed tree node end-to-end. This is the M3-S1
   integration win — once green, swapping ``NullSummariser`` for
   ``RouterSummariser`` in the agent wiring is a one-line edit.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from wiki_agent.write_sink import NullWriteSink
from wiki_memory.chunker import chunk_text
from wiki_memory.seal_worker import VAULT_TREES_PREFIX, SealError, SealWorker
from wiki_memory.summariser import Summariser, SummaryPart
from wiki_routing.policy import RoutingPolicy
from wiki_routing.router import BackendResponse, ModelRouter, StubBackend
from wiki_routing.router_summariser import RouterSummariser
from wiki_routing.tiers import Tier


def _mk_parts(n: int) -> list[SummaryPart]:
    parts = []
    for i in range(n):
        body = f"child body {i} with content {i * i}"
        sha = hashlib.sha256(body.encode()).hexdigest()
        parts.append(SummaryPart(sha256=sha, body=body, token_count=10))
    return parts


def _shortsha(sha: str) -> str:
    return sha[:8]


def _summary_text_citing(parts: list[SummaryPart]) -> str:
    """Build a model-style response that cites every input chunk."""
    lines = ["# Summary"]
    for p in parts:
        lines.append(f"- fact from [[chunk:{_shortsha(p.sha256)}]]")
    return "\n".join(lines)


def _json_response_citing(
    parts: list[SummaryPart], *, body: str | None = None, full_shas: bool = True
) -> str:
    """Build a clean JSON-mode response citing every input chunk.

    ``full_shas=True`` cites by full sha (what the prompt requests);
    ``False`` cites by short-sha (lenient path the parser also accepts).
    """
    rendered = body or "\n".join(f"- fact from [[chunk:{_shortsha(p.sha256)}]]" for p in parts)
    shas = [p.sha256 if full_shas else _shortsha(p.sha256) for p in parts]
    return json.dumps({"body": rendered, "cited_shas": shas})


@pytest.fixture
def write_sink() -> NullWriteSink:
    return NullWriteSink()


def _router_returning(text: str, *, cost_usd: float = 0.001) -> ModelRouter:
    """Build a router whose Reasoning tier returns ``text``."""
    backend = StubBackend(
        label="stub:reasoning",
        response=BackendResponse(
            text=text,
            tokens_in=100,
            tokens_out=100,
            cost_usd=cost_usd,
        ),
        quote_usd=cost_usd,
    )
    return ModelRouter(
        policy=RoutingPolicy(),
        providers={
            Tier.REASONING: backend,
            Tier.FAST: backend,
            Tier.VISION: backend,
        },
    )


class TestProtocolConformance:
    def test_satisfies_summariser_protocol(self) -> None:
        # The integration contract the seal worker depends on.
        router = _router_returning("anything")
        rs = RouterSummariser(router=router)
        assert isinstance(rs, Summariser)

    def test_backwards_compatible_constructor(self) -> None:
        # Old call sites that pass only ``router=`` keep working —
        # output_format defaults to "json" (the new path), strict_json
        # defaults to False.
        rs = RouterSummariser(router=_router_returning("x"))
        assert rs.output_format == "json"
        assert rs.strict_json is False

    def test_rejects_invalid_output_format(self) -> None:
        with pytest.raises(ValueError):
            RouterSummariser(
                router=_router_returning("x"),
                output_format="xml",  # type: ignore[arg-type]
            )


class TestSummariseDirectSubstringMode:
    """Coverage of the legacy substring path, kept intact as the
    JSON-mode fallback."""

    @pytest.mark.asyncio
    async def test_empty_parts_short_circuit(self) -> None:
        rs = RouterSummariser(router=_router_returning("never called"), output_format="substring")
        result = await rs.summarise([])
        assert result.cited_shas == ()
        assert result.body

    @pytest.mark.asyncio
    async def test_returns_model_text_in_body(self) -> None:
        parts = _mk_parts(3)
        text = _summary_text_citing(parts)
        rs = RouterSummariser(router=_router_returning(text), output_format="substring")
        result = await rs.summarise(parts)
        assert result.body == text

    @pytest.mark.asyncio
    async def test_sha_matches_body(self) -> None:
        parts = _mk_parts(2)
        text = _summary_text_citing(parts)
        rs = RouterSummariser(router=_router_returning(text), output_format="substring")
        result = await rs.summarise(parts)
        assert result.sha256 == hashlib.sha256(result.body.encode("utf-8")).hexdigest()

    @pytest.mark.asyncio
    async def test_extracts_citations_from_text(self) -> None:
        parts = _mk_parts(4)
        # Cite only the first two; the other two should not appear in
        # cited_shas.
        cited_text = "\n".join(f"- [[chunk:{_shortsha(parts[i].sha256)}]] x" for i in range(2))
        rs = RouterSummariser(router=_router_returning(cited_text), output_format="substring")
        result = await rs.summarise(parts)
        assert set(result.cited_shas) == {parts[0].sha256, parts[1].sha256}

    @pytest.mark.asyncio
    async def test_ignores_hallucinated_citations(self) -> None:
        parts = _mk_parts(2)
        # Model invents a citation for a sha that isn't in the input.
        ghost = "deadbeefcafef00d"
        text = f"- [[chunk:{ghost}]] hallucinated\n- [[chunk:{_shortsha(parts[0].sha256)}]] real"
        rs = RouterSummariser(router=_router_returning(text), output_format="substring")
        result = await rs.summarise(parts)
        assert set(result.cited_shas) == {parts[0].sha256}


class TestSummariseDirectJsonMode:
    """SPEC-009 OQ-2 close-out: structured JSON output for citations."""

    @pytest.mark.asyncio
    async def test_json_mode_parses_clean_json(self) -> None:
        parts = _mk_parts(3)
        payload = _json_response_citing(parts)
        rs = RouterSummariser(router=_router_returning(payload))  # default = json
        result = await rs.summarise(parts)
        # Body is the JSON's body field, not the raw response.
        expected_body = "\n".join(f"- fact from [[chunk:{_shortsha(p.sha256)}]]" for p in parts)
        assert result.body == expected_body
        assert set(result.cited_shas) == {p.sha256 for p in parts}

    @pytest.mark.asyncio
    async def test_json_mode_parses_json_in_code_fence(self) -> None:
        parts = _mk_parts(2)
        inner = _json_response_citing(parts)
        fenced = f"```json\n{inner}\n```"
        rs = RouterSummariser(router=_router_returning(fenced))
        result = await rs.summarise(parts)
        assert set(result.cited_shas) == {p.sha256 for p in parts}
        # The body comes from the inner JSON, not the fence wrapper.
        assert "```" not in result.body

    @pytest.mark.asyncio
    async def test_json_mode_parses_json_with_leading_text(self) -> None:
        parts = _mk_parts(2)
        inner = _json_response_citing(parts)
        # Model preceded the JSON with an apology / explanation —
        # common with Anthropic-style refusals that still emit JSON.
        wrapped = f"Here is the summary you asked for:\n\n{inner}\n\nDone."
        rs = RouterSummariser(router=_router_returning(wrapped))
        result = await rs.summarise(parts)
        assert set(result.cited_shas) == {p.sha256 for p in parts}
        # Body is just the JSON's body, not the surrounding prose.
        assert "Here is the summary" not in result.body

    @pytest.mark.asyncio
    async def test_json_mode_falls_back_to_substring_on_malformed_json(self) -> None:
        parts = _mk_parts(2)
        # Garbage that's not JSON but contains a valid [[chunk:SHA8]]
        # marker — the substring fallback should pick it up.
        garbage = (
            "I refuse to emit JSON, here is prose:\n"
            f"- [[chunk:{_shortsha(parts[0].sha256)}]] fact\n"
        )
        rs = RouterSummariser(router=_router_returning(garbage))  # strict=False default
        result = await rs.summarise(parts)
        # Substring parser ran: body is the raw text, only parts[0] is
        # cited.
        assert result.body == garbage
        assert set(result.cited_shas) == {parts[0].sha256}

    @pytest.mark.asyncio
    async def test_strict_json_raises_on_malformed(self) -> None:
        parts = _mk_parts(2)
        garbage = "not json at all, no braces here"
        rs = RouterSummariser(
            router=_router_returning(garbage), output_format="json", strict_json=True
        )
        with pytest.raises(ValueError, match="strict_json"):
            await rs.summarise(parts)

    @pytest.mark.asyncio
    async def test_strict_json_raises_on_wrong_shape(self) -> None:
        parts = _mk_parts(2)
        # Valid JSON, wrong shape: missing 'cited_shas'.
        payload = json.dumps({"body": "summary text"})
        rs = RouterSummariser(
            router=_router_returning(payload), output_format="json", strict_json=True
        )
        with pytest.raises(ValueError, match="cited_shas"):
            await rs.summarise(parts)

    @pytest.mark.asyncio
    async def test_json_mode_drops_hallucinated_shas(self) -> None:
        parts = _mk_parts(2)
        # Model cites a sha that isn't in parts — the summariser
        # filters it out so the seal worker's faithfulness gate
        # sees only real shas and accepts the seal.
        payload = json.dumps(
            {
                "body": "summary",
                "cited_shas": [parts[0].sha256, "deadbeef" * 8],  # second is fake
            }
        )
        rs = RouterSummariser(router=_router_returning(payload))
        result = await rs.summarise(parts)
        assert set(result.cited_shas) == {parts[0].sha256}

    @pytest.mark.asyncio
    async def test_full_sha_citation(self) -> None:
        """Model cites by full 64-char sha — preserved end-to-end."""
        parts = _mk_parts(1)
        full_sha = parts[0].sha256
        assert len(full_sha) == 64  # sanity
        payload = json.dumps({"body": "single fact", "cited_shas": [full_sha]})
        rs = RouterSummariser(router=_router_returning(payload))
        result = await rs.summarise(parts)
        assert result.cited_shas == (full_sha,)

    @pytest.mark.asyncio
    async def test_json_mode_accepts_short_sha_lenient(self) -> None:
        """Even though the prompt asks for full shas, the parser
        accepts short shas to match the substring path's behaviour."""
        parts = _mk_parts(2)
        payload = _json_response_citing(parts, full_shas=False)
        rs = RouterSummariser(router=_router_returning(payload))
        result = await rs.summarise(parts)
        # The short shas were expanded back to full shas in cited_shas.
        assert set(result.cited_shas) == {p.sha256 for p in parts}

    @pytest.mark.asyncio
    async def test_json_mode_empty_parts_short_circuit(self) -> None:
        # Same short-circuit as substring mode — never calls the model.
        rs = RouterSummariser(router=_router_returning("never called"))
        result = await rs.summarise([])
        assert result.cited_shas == ()
        assert result.body

    @pytest.mark.asyncio
    async def test_substring_mode_still_works(self) -> None:
        """Explicit substring mode behaves exactly as before — used as
        the regression guard on the legacy parser."""
        parts = _mk_parts(3)
        text = _summary_text_citing(parts)
        rs = RouterSummariser(router=_router_returning(text), output_format="substring")
        result = await rs.summarise(parts)
        assert result.body == text
        assert set(result.cited_shas) == {p.sha256 for p in parts}


class TestSealWorkerIntegration:
    @pytest.mark.asyncio
    async def test_seal_with_router_summariser_produces_sealed_node(
        self,
        content_store,
        tree_store,
        write_sink: NullWriteSink,
    ) -> None:
        # Seed: two chunks in the content store and a tree node.
        chunks = chunk_text("alpha alpha.\n\nbeta beta.", target_tokens=5, hard_cap_tokens=20)
        await content_store.insert_many(source_id="src-rs", chunks=chunks)
        await tree_store.create_source_node(node_id="src-rs")

        # Build a router whose reasoning model returns a clean JSON
        # citing every chunk.
        parts = [
            SummaryPart(sha256=c.sha256, body=c.body, token_count=c.token_count) for c in chunks
        ]
        canned = _json_response_citing(parts)
        router = _router_returning(canned)
        rs = RouterSummariser(router=router)

        worker = SealWorker(
            content_store=content_store,
            tree_store=tree_store,
            write_sink=write_sink,
            summariser=rs,
        )
        result = await worker.seal_source(source_id="src-rs", node_id="src-rs")

        # Vault mirror landed at the right place.
        assert result.page_path.startswith(VAULT_TREES_PREFIX)
        assert "sources/" in result.page_path
        # Tree node was sealed.
        node = await tree_store.get("src-rs")
        assert node is not None
        assert node.sealed_at is not None
        assert node.summary_sha256 == result.summary_sha256
        # Write sink saw exactly one upsert.
        assert len(write_sink.calls) == 1
        mode, page = write_sink.calls[0]
        assert mode == "upsert"
        # Frontmatter contains the cited list — the router-driven path
        # is end-to-end equivalent to NullSummariser's behaviour.
        assert isinstance(page.frontmatter.get("cited"), list)
        assert len(page.frontmatter["cited"]) == len(chunks)

    @pytest.mark.asyncio
    async def test_json_mode_drops_hallucinated_shas_integration(
        self,
        content_store,
        tree_store,
        write_sink: NullWriteSink,
    ) -> None:
        """Integration: JSON cites a sha not in parts → RouterSummariser
        drops it → SealWorker faithfulness gate sees only real shas and
        accepts the seal (the hallucination never reaches it)."""
        chunks = chunk_text("real content here", target_tokens=10)
        await content_store.insert_many(source_id="src-h", chunks=chunks)
        await tree_store.create_source_node(node_id="src-h")

        # JSON cites every real sha + one ghost.
        real_shas = [c.sha256 for c in chunks]
        payload = json.dumps(
            {
                "body": "\n".join(f"- [[chunk:{_shortsha(s)}]] real" for s in real_shas),
                "cited_shas": [*real_shas, "deadbeef" * 8],
            }
        )

        rs = RouterSummariser(router=_router_returning(payload))
        worker = SealWorker(
            content_store=content_store,
            tree_store=tree_store,
            write_sink=write_sink,
            summariser=rs,
        )
        # No raise: the summariser filtered the ghost before the gate
        # ever saw it.
        result = await worker.seal_source(source_id="src-h", node_id="src-h")
        assert result.summary_sha256

        try:
            await worker.seal_source(source_id="src-h", node_id="src-h")
        except SealError as e:
            pytest.fail(f"unexpected SealError on clean router output: {e}")

    @pytest.mark.asyncio
    async def test_seal_rejects_hallucination_via_faithfulness_gate(
        self,
        content_store,
        tree_store,
        write_sink: NullWriteSink,
    ) -> None:
        """Substring-mode regression: a hallucinated short-sha is
        dropped by the summariser; the faithfulness gate still sees
        only real shas and accepts the seal."""
        chunks = chunk_text("real content here", target_tokens=10)
        await content_store.insert_many(source_id="src-h2", chunks=chunks)
        await tree_store.create_source_node(node_id="src-h2")

        text = (
            "\n".join(f"- [[chunk:{_shortsha(c.sha256)}]] real" for c in chunks)
            + "\n- [[chunk:deadbeef]] hallucinated"
        )

        rs = RouterSummariser(router=_router_returning(text), output_format="substring")
        worker = SealWorker(
            content_store=content_store,
            tree_store=tree_store,
            write_sink=write_sink,
            summariser=rs,
        )
        result = await worker.seal_source(source_id="src-h2", node_id="src-h2")
        assert result.summary_sha256


class TestRouterTierSelection:
    @pytest.mark.asyncio
    async def test_seal_intent_routes_to_reasoning(self) -> None:
        # Confirm RouterSummariser uses the "seal" intent so the policy
        # actually selects the REASONING tier.
        reasoning = StubBackend(
            label="stub:reasoning",
            response=BackendResponse(
                text=json.dumps({"body": "# Summary", "cited_shas": []}),
                tokens_in=1,
                tokens_out=1,
                cost_usd=0.001,
            ),
            quote_usd=0.001,
        )
        fast = StubBackend(label="stub:fast")
        router = ModelRouter(
            policy=RoutingPolicy(),
            providers={Tier.REASONING: reasoning, Tier.FAST: fast, Tier.VISION: fast},
        )
        rs = RouterSummariser(router=router)
        await rs.summarise(_mk_parts(2))
        # Reasoning saw the call; fast did not.
        assert len(reasoning.calls) == 1
        assert fast.calls == []
