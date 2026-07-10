"""
Tests for `wiki_compress.stages.url_shorten`.
"""

from __future__ import annotations

import re

from wiki_compress.stages.url_shorten import MIN_URL_LENGTH, UrlShortener


class TestThreshold:
    def test_short_url_left_alone(self) -> None:
        s = UrlShortener()
        out = s("Visit https://x.io now.")
        assert "https://x.io" in out
        assert s.url_map == {}

    def test_long_url_shortened(self) -> None:
        long_url = "https://example.com/" + "x" * 50
        assert len(long_url) >= MIN_URL_LENGTH
        s = UrlShortener()
        out = s(f"See {long_url} please.")
        assert long_url not in out
        assert re.search(r"\[url:[0-9a-f]{5}\]", out)
        assert long_url in s.url_map.values()


class TestDeterminism:
    def test_same_url_same_handle_across_instances(self) -> None:
        url = "https://example.com/" + "y" * 60
        s1 = UrlShortener()
        s2 = UrlShortener()
        out1 = s1(url)
        out2 = s2(url)
        assert out1 == out2

    def test_same_url_within_run_reuses_handle(self) -> None:
        url = "https://example.com/" + "z" * 60
        s = UrlShortener()
        out = s(f"{url} appears twice {url} here")
        handles = re.findall(r"\[url:[0-9a-f]{5}(?:_\d+)?\]", out)
        assert len(handles) == 2
        assert handles[0] == handles[1]
        assert len(s.url_map) == 1


class TestIdempotent:
    def test_rerun_on_shortened_text_is_noop(self) -> None:
        url = "https://example.com/" + "a" * 60
        s1 = UrlShortener()
        once = s1(url)
        s2 = UrlShortener()
        twice = s2(once)
        assert twice == once
        # The second shortener should not have learned anything — there
        # were no http(s) URLs left to capture.
        assert s2.url_map == {}


class TestReset:
    def test_reset_clears_state(self) -> None:
        s = UrlShortener()
        s("https://example.com/" + "q" * 60)
        assert s.url_map
        s.reset()
        assert s.url_map == {}


class TestCollision:
    def test_collision_disambiguator(self) -> None:
        """Force a 5-char prefix collision and confirm handles stay unique."""
        s = UrlShortener()
        url_a = "https://example.com/long/path/a/" + "x" * 30
        s(url_a)
        original_handle = next(iter(s.url_map.keys()))
        # Force a fake collision — pretend another URL maps to the same prefix.
        url_b = "https://example.com/long/path/b/" + "y" * 30
        # Patch the map directly to simulate a colliding handle.
        s.url_map[original_handle] = url_a  # already there, but explicit.
        # Re-seed with a synthetic colliding entry under the same handle:
        # actual sha1 collisions are vanishingly rare, so we test the logic
        # by pretending the slot is taken.
        # Reset _seen so the next call re-shortens.
        s._seen.pop(url_b, None)  # noqa: SLF001 — testing the disambiguator.
        # When we shorten url_b, the new handle must be distinct.
        # Note: in real practice the digests would differ; this test
        # exercises the disambiguator branch, not the digest itself.
        # Force the digest collision by stuffing the map:
        s.url_map["[url:00000]"] = "FAKE-COLLIDER"
        # Now shorten a URL whose digest equals 00000 — impossible to engineer
        # without breaking sha1, so we directly call the private method to
        # cover the branch.
        # Replace _handle for a moment.
        from wiki_compress.stages import url_shorten as us

        original_handle_fn = us._handle

        def _fixed_handle(url: str, disambiguator: int = 0) -> str:
            base = "00000"
            return f"[url:{base}_{disambiguator}]" if disambiguator else f"[url:{base}]"

        us._handle = _fixed_handle  # type: ignore[assignment]
        try:
            colliding_url = "https://example.com/colliding/" + "z" * 40
            shortened = s(colliding_url)
            # Disambiguator should kick in:
            assert "_" in shortened
            assert colliding_url in s.url_map.values()
        finally:
            us._handle = original_handle_fn  # type: ignore[assignment]


class TestUnicode:
    def test_long_url_with_cjk_query(self) -> None:
        # URLs do not usually contain raw CJK, but the regex must not crash
        # on text where CJK sits adjacent to a URL.
        url = "https://example.com/path/" + "x" * 40
        s = UrlShortener()
        out = s(f"参考: {url} 看看 🌟")
        assert "参考" in out
        assert "看看" in out
        assert "🌟" in out
        assert url not in out
