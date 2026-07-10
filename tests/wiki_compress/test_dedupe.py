"""
Tests for `wiki_compress.stages.dedupe`.
"""

from __future__ import annotations

from wiki_compress.stages.dedupe import SEE_ABOVE, dedupe_paragraphs
from wiki_compress.stages.preserve import mask_code_blocks


class TestExactDuplicates:
    def test_two_identical_long_paragraphs_collapse(self) -> None:
        para = "This paragraph is long enough that it exceeds the dedupe floor of forty chars."
        text = f"{para}\n\n{para}"
        out = dedupe_paragraphs(text)
        assert para in out
        assert out.count(para) == 1
        assert SEE_ABOVE in out

    def test_three_identical_paragraphs_collapse_to_one_plus_two_markers(self) -> None:
        para = "Long enough paragraph to qualify for dedupe; needs about fifty chars."
        text = "\n\n".join([para, para, para])
        out = dedupe_paragraphs(text)
        assert out.count(para) == 1
        assert out.count(SEE_ABOVE) == 2


class TestWhitespaceVariants:
    def test_paragraphs_differing_only_in_whitespace_are_duplicates(self) -> None:
        para_a = "Paragraph    with weird     whitespace patterns, definitely long enough."
        para_b = "Paragraph with weird whitespace patterns,    definitely    long    enough."
        text = f"{para_a}\n\n{para_b}"
        out = dedupe_paragraphs(text)
        assert SEE_ABOVE in out


class TestSmallParagraphs:
    def test_short_repeated_paragraphs_kept(self) -> None:
        text = "Yes.\n\nYes.\n\nYes."
        out = dedupe_paragraphs(text)
        # Below the 40-char floor — must not be collapsed.
        assert out.count("Yes.") == 3
        assert SEE_ABOVE not in out


class TestNearDuplicates:
    def test_one_char_difference_keeps_both(self) -> None:
        a = "This is a longish paragraph that talks about cats and dogs in detail."
        b = "This is a longish paragraph that talks about cats and dogs in detai!"
        out = dedupe_paragraphs(f"{a}\n\n{b}")
        assert a in out
        assert b in out
        assert SEE_ABOVE not in out


class TestProtectedSpans:
    def test_two_identical_code_blocks_not_deduped(self) -> None:
        code = "```\ndef f(): pass\n```"
        text = f"{code}\n\nIntro paragraph that is long enough to qualify normally.\n\n{code}"
        wrapped, _ = mask_code_blocks(text)
        out = dedupe_paragraphs(wrapped)
        # The code block sentinels survived dedupe — neither was collapsed.
        # We don't assert on the literal code body here (it's wrapped), but
        # the SEE_ABOVE marker must NOT appear for the code itself.
        # The intro paragraph appears only once, so no marker for it either.
        # In practice, the test ensures dedupe never touches protected spans.
        from wiki_compress.stages.preserve import release_preserved

        released = release_preserved(out)
        assert released.count("def f(): pass") == 2


class TestIdempotent:
    def test_running_dedupe_twice_is_a_noop(self) -> None:
        para = "Long enough paragraph for dedupe, with some honest content in it."
        text = f"{para}\n\n{para}\n\n{para}"
        once = dedupe_paragraphs(text)
        twice = dedupe_paragraphs(once)
        assert once == twice


class TestUnicode:
    def test_cjk_paragraph_dedupes_correctly(self) -> None:
        para = "这是一段中文段落,长度足够触发去重器的阈值。Mixed with English to pad."
        text = f"{para}\n\n{para}"
        out = dedupe_paragraphs(text)
        assert para in out
        assert SEE_ABOVE in out
        assert out.count(para) == 1

    def test_emoji_paragraph_dedupes_correctly(self) -> None:
        para = "🌟🚀 emoji paragraph 👨‍👩‍👧‍👦 with enough chars to qualify for dedupe."
        text = f"{para}\n\n{para}"
        out = dedupe_paragraphs(text)
        assert para in out
        assert SEE_ABOVE in out
