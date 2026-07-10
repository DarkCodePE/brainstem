"""
Tests for ``wiki_compress.stages.line_dedupe``.

Coverage matrix:

| Behaviour                                              | Test                                       |
| ------------------------------------------------------ | ------------------------------------------ |
| Two identical 100-char lines → second collapsed        | test_two_identical_long_lines_collapse     |
| Two identical 30-char lines (< 80 floor) → NOT col     | test_short_lines_not_collapsed             |
| Three identical lines → only the first remains         | test_three_identical_lines                 |
| First line never replaced with a marker                | test_first_occurrence_preserved            |
| Floor configurable via keyword                         | test_floor_configurable                    |
| Paragraph dedupe and line dedupe do not interfere      | test_no_interference_with_paragraph_dedupe |
| Empty input is a no-op                                 | test_empty_input_noop                      |
| ``make_line_deduper`` produces a working stage         | test_make_line_deduper_stage               |
| Idempotency                                            | test_idempotent                            |
| CJK lines preserved verbatim in marker source          | test_cjk_line_preserved                    |
| Protected code spans not touched                       | test_protected_code_block_preserved        |
| Marker line itself is not re-deduped                   | test_marker_line_not_re_deduped            |
"""

from __future__ import annotations

import hashlib

from wiki_compress.stages.dedupe import dedupe_paragraphs
from wiki_compress.stages.line_dedupe import (
    DEFAULT_MIN_LINE_LENGTH,
    dedupe_lines,
    make_line_deduper,
)
from wiki_compress.stages.preserve import mask_code_blocks, release_preserved


def _sha8(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


class TestLongLines:
    def test_two_identical_long_lines_collapse(self) -> None:
        long_line = "X" * 100
        text = f"{long_line}\nseparator line\n{long_line}"
        out = dedupe_lines(text)
        assert out.count(long_line) == 1
        assert f"[see-line:{_sha8(long_line)}]" in out

    def test_three_identical_lines(self) -> None:
        long_line = "Y" * 90
        text = "\n".join([long_line, long_line, long_line])
        out = dedupe_lines(text)
        assert out.count(long_line) == 1
        # The two later occurrences are markers.
        assert out.count(f"[see-line:{_sha8(long_line)}]") == 2

    def test_first_occurrence_preserved(self) -> None:
        long_line = "Z" * 85
        text = f"{long_line}\n{long_line}"
        out = dedupe_lines(text)
        lines = out.split("\n")
        assert lines[0] == long_line
        assert lines[1] == f"[see-line:{_sha8(long_line)}]"


class TestShortLinesFloor:
    def test_short_lines_not_collapsed(self) -> None:
        short_line = "OK status flag here"  # < 80 chars
        text = f"{short_line}\n{short_line}\n{short_line}"
        out = dedupe_lines(text)
        # Below the 80-char floor — nothing collapsed.
        assert out.count(short_line) == 3
        assert "[see-line:" not in out

    def test_floor_configurable(self) -> None:
        line = "git status row that is exactly 50 chars long ok!!"
        assert 30 < len(line) < 80
        text = f"{line}\n{line}"
        out_default = dedupe_lines(text)
        out_loose = dedupe_lines(text, min_length=30)
        # Default floor leaves them alone:
        assert out_default == text
        # Tighter floor collapses them:
        assert "[see-line:" in out_loose

    def test_floor_at_default_boundary(self) -> None:
        line_under = "a" * (DEFAULT_MIN_LINE_LENGTH - 1)
        line_at = "b" * DEFAULT_MIN_LINE_LENGTH
        text_under = f"{line_under}\n{line_under}"
        text_at = f"{line_at}\n{line_at}"
        assert "[see-line:" not in dedupe_lines(text_under)
        assert "[see-line:" in dedupe_lines(text_at)


class TestEmpty:
    def test_empty_input_noop(self) -> None:
        assert dedupe_lines("") == ""


class TestMakeLineDeduper:
    def test_make_line_deduper_stage(self) -> None:
        line = "git status row that is exactly 50 chars long ok!!"
        text = f"{line}\n{line}"
        stage = make_line_deduper(min_length=30)
        out = stage(text)
        assert "[see-line:" in out
        assert stage.__name__ == "line_dedupe_min30"


class TestIdempotency:
    def test_idempotent(self) -> None:
        long_line = "Q" * 100
        text = f"{long_line}\n{long_line}\n{long_line}"
        once = dedupe_lines(text)
        twice = dedupe_lines(once)
        assert once == twice

    def test_marker_line_not_re_deduped(self) -> None:
        long_line = "R" * 100
        once = dedupe_lines(f"{long_line}\n{long_line}")
        # Twice through must not introduce a recursive marker.
        twice = dedupe_lines(once)
        assert once == twice


class TestInteractionWithParagraphDedupe:
    def test_no_interference_with_paragraph_dedupe(self) -> None:
        # A paragraph composed of long, unique lines; the paragraph repeats.
        # Each line is > 80 chars so line-dedupe is willing to collapse it.
        line1 = "First long line of the paragraph well past the eighty char floor that line-dedupe defends."
        line2 = "Second long line, again comfortably over eighty characters so line-dedupe is happy to collapse."
        assert len(line1) > DEFAULT_MIN_LINE_LENGTH
        assert len(line2) > DEFAULT_MIN_LINE_LENGTH
        paragraph = f"{line1}\n{line2}"
        text = f"{paragraph}\n\n{paragraph}"
        # Paragraph dedupe collapses the second copy.
        para_out = dedupe_paragraphs(text)
        assert para_out.count(line1) == 1
        # Line dedupe on the same input does its job at line granularity.
        line_out = dedupe_lines(text)
        assert "[see-line:" in line_out
        # And the two passes do not interfere when stacked.
        stacked = dedupe_paragraphs(dedupe_lines(text))
        # Stacked result still contains the first occurrence of each line.
        assert line1 in stacked


class TestUnicode:
    def test_cjk_line_preserved(self) -> None:
        cjk_line = "这是一行中文文字,长度刚好超过默认阈值,以便测试去重。" * 4
        assert len(cjk_line) > DEFAULT_MIN_LINE_LENGTH
        text = f"{cjk_line}\n{cjk_line}"
        out = dedupe_lines(text)
        assert cjk_line in out
        assert out.count(cjk_line) == 1
        assert "[see-line:" in out


class TestProtectedSpans:
    def test_protected_code_block_preserved(self) -> None:
        long_line = "S" * 100
        code = f"```\n{long_line}\n{long_line}\n```"
        text = f"{code}\n\n{long_line}\n{long_line}\n"
        wrapped, _ = mask_code_blocks(text)
        out = dedupe_lines(wrapped)
        released = release_preserved(out)
        # Code block content is preserved as-is (both copies still inside).
        # The plain duplicates outside the fence are collapsed.
        # Count occurrences of the long_line: 2 inside the code block,
        # 1 outside (the second outside one is a marker).
        assert released.count(long_line) == 3
        assert "[see-line:" in released
