"""Tests for ``wiki_routing.tiers`` — the ``Tier`` value type."""

from __future__ import annotations

from wiki_routing.tiers import Tier


class TestTier:
    def test_three_tiers_exist(self) -> None:
        # ADR-013 names three logical tiers. Anything else would mean
        # the routing matrix has drifted from the ADR.
        assert {t.name for t in Tier} == {"REASONING", "FAST", "VISION"}

    def test_tier_values_are_lowercase_strings(self) -> None:
        # The wire format on routing.toml uses lower-case identifiers
        # (PRD-008 FR-1). Equality is identity-based on the enum, but
        # callers that round-trip through TOML use .value.
        for tier in Tier:
            assert tier.value == tier.name.lower()

    def test_str_returns_value(self) -> None:
        assert str(Tier.REASONING) == "reasoning"
        assert str(Tier.FAST) == "fast"
        assert str(Tier.VISION) == "vision"

    def test_tiers_are_distinct(self) -> None:
        # Identity, not just equality — Tier is an Enum.
        assert Tier.REASONING is not Tier.FAST
        assert Tier.FAST is not Tier.VISION
        assert Tier.REASONING is not Tier.VISION

    def test_tiers_are_hashable(self) -> None:
        # Mapping keys depend on this.
        d = {Tier.REASONING: "r", Tier.FAST: "f", Tier.VISION: "v"}
        assert d[Tier.REASONING] == "r"
        assert d[Tier.FAST] == "f"
        assert d[Tier.VISION] == "v"
