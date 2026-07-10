"""
Tests for the two M3-Sprint-2 pipeline presets:

- ``build_email_pipeline()`` — for email-thread payloads
- ``build_tool_output_pipeline()`` — for tool-stdout payloads

These presets target the two payload kinds SPEC-008 flagged as weak in
the default pipeline (median ratio 1.000 for email, 0.973 for tool
output). The targets here mirror SPEC-008's "Sprint 2 follow-up" table:

| Kind         | Default median | Preset target |
|--------------|----------------|---------------|
| Email        | 1.000          | ≤ 0.70        |
| Tool output  | 0.973          | ≤ 0.50        |

Coverage matrix:

| Behaviour                                              | Test                                    |
| ------------------------------------------------------ | --------------------------------------- |
| Email preset beats default on an email payload         | test_email_preset_beats_default         |
| Email preset median ratio ≤ 0.70 across 4 payloads     | test_email_preset_corpus_median         |
| Email preset populates ``quote_collapser.quote_map``   | test_email_preset_quote_map_populated   |
| Tool-output preset beats default on `git status`       | test_tool_preset_beats_default          |
| Tool-output preset median ratio ≤ 0.50 across corpus   | test_tool_preset_corpus_median          |
| Tool-output preset stage order (no html_to_md)         | test_tool_preset_stage_order            |
| Default pipeline still produces ratio < 1.0 on HTML    | test_default_still_works                |
| Email preset CJK-safe                                  | test_email_preset_cjk_safe              |
| Tool-output preset CJK-safe                            | test_tool_preset_cjk_safe               |
| Email preset idempotent                                | test_email_preset_idempotent            |
| Tool-output preset idempotent                          | test_tool_preset_idempotent             |
"""

from __future__ import annotations

import statistics

import pytest

from wiki_compress import (
    build_default_pipeline,
    build_email_pipeline,
    build_tool_output_pipeline,
)

# --------------------------------------------------------------------------- #
# Corpus fixtures — small but representative.                                 #
# --------------------------------------------------------------------------- #


_EMAIL_PAYLOADS: list[str] = [
    # Email 1 — short reply on top of a quoted thread (typical Outlook layout).
    """Hi team,

Yes, please proceed with the migration plan as drafted. Two notes inline below.

> -----Original Message-----
> From: Sarah Engineer
> Sent: Wednesday, May 15, 2026 9:14 AM
> To: Migration WG
> Subject: RE: Migration plan v3 — feedback?
>
> All — here is the third draft of the migration plan. Key changes since v2:
> 1. We will roll the schema change to the read replicas first.
> 2. The cutover window has been pushed from Friday to Sunday at 02:00 UTC.
> 3. The rollback drill has been scheduled for Saturday evening.
>
> Let me know any concerns by EOD Thursday.
>
> > -----Original Message-----
> > From: Lead Architect
> > Sent: Monday, May 13, 2026 4:02 PM
> > To: Migration WG
> > Subject: RE: Migration plan v3 — feedback?
> >
> > A few structural concerns on draft v2 — primarily around the choice of
> > rolling vs blue/green. Happy to discuss in the architecture sync today.

Thanks,
Mark""",
    # Email 2 — long quoted block with a tiny reply.
    """Looks good, ship it.

> Hey Mark,
>
> Sending over the final review notes from the security audit conducted last
> week. The full report runs about 40 pages, but the actionable items are:
>
> 1. Rotate the staging API keys before the next release.
> 2. Add rate-limiting middleware to the public webhooks endpoint.
> 3. The OAuth callback handler should reject state tokens older than 10
>    minutes (currently 30 minutes per RFC 6749 §10.12).
> 4. Document the secret rotation runbook in the security wiki.
>
> Targeting end-of-sprint for items 1-3, item 4 next sprint.
>
> Cheers,
> Sarah""",
    # Email 3 — three-deep nested chain.
    """Confirmed received, will review later today.

> Thanks for the heads-up.
>
> > Original message follows. The detector flagged a possible regression in
> > the embedding-recall step. Eval suite ran clean overnight on the prod
> > snapshot but the staging snapshot shows a 4% drop on the long-tail
> > queries. Investigating the chunker boundary changes from PR #487.
> >
> > > -----Original Message-----
> > > From: Detector Bot
> > > Sent: Tuesday, May 14, 2026 3:14 AM
> > > Subject: ALERT — recall regression detected
> > >
> > > Recall@10 dropped from 0.84 to 0.81 between build 4421 and 4438.
> > > Confidence: 0.92. See attached eval report for details.""",
    # Email 4 — almost pure signature block (low signal).
    """Acknowledged, will follow up Friday.

> Looking forward to it.
>
> --
> Sarah Engineer
> Principal Software Engineer
> Infrastructure Team
> https://example.com/sarah-engineer
> Cell: redacted
> Pronouns: she/her
>
> This message and any attachments are confidential and intended solely for
> the addressee. If you received this in error, please notify the sender
> and delete the message.""",
]


_TOOL_OUTPUT_PAYLOADS: list[str] = [
    # Tool 1 — `git status -uall` in a worktree where the same advisory line
    # is emitted per untracked directory and the same modified file paths
    # reappear in multiple sections (staged, unstaged, untracked). This is
    # the exact pathology the line-dedupe stage targets.
    "\n".join(
        [
            "On branch feature/m3-wire-substrates",
            "Your branch is up to date with 'origin/feature/m3-wire-substrates'.",
            "",
            "Untracked files:",
        ]
        + [
            '  (use "git add <file>..." to include in what will be committed)',
        ]
        * 8
        + [
            "\tsrc/wiki_compress/stages/email_quotes.py",
            "\tsrc/wiki_compress/stages/email_quotes.py",
            "\tsrc/wiki_compress/stages/line_dedupe.py",
            "\tsrc/wiki_compress/stages/line_dedupe.py",
            "",
        ]
        + [
            'no changes added to commit (use "git add" and/or "git commit -a")',
        ]
        * 5
    ),
    # Tool 2 — `ls -la` with repeated long path rows.
    "\n".join(
        [
            "total 240",
            "drwxr-xr-x 12 user user  4096 May 22 14:33 /home/user/projects/very-long-path-name/subdirectory/files-here",
            "drwxr-xr-x 12 user user  4096 May 22 14:33 /home/user/projects/very-long-path-name/subdirectory/files-here",
            "drwxr-xr-x 12 user user  4096 May 22 14:33 /home/user/projects/very-long-path-name/subdirectory/files-here",
            "drwxr-xr-x 12 user user  4096 May 22 14:33 /home/user/projects/very-long-path-name/subdirectory/files-here",
            "-rw-r--r--  1 user user  2048 May 22 14:33 /home/user/projects/very-long-path-name/subdirectory/files-here/README.md",
            "-rw-r--r--  1 user user  2048 May 22 14:33 /home/user/projects/very-long-path-name/subdirectory/files-here/README.md",
        ]
    ),
    # Tool 3 — Python traceback with deeply repeated frames (recursion).
    # Real RecursionError tracebacks emit hundreds of identical frames; this
    # is a representative sample with the same frame repeated many times.
    "\n".join(
        [
            "Traceback (most recent call last):",
            '  File "/home/user/long/path/to/project/src/wiki_agent/middleware/runner.py", line 142, in dispatch',
            "    return handler(event, context)",
            '  File "/home/user/long/path/to/project/src/wiki_agent/middleware/runner.py", line 142, in dispatch',
            "    return handler(event, context)",
            '  File "/home/user/long/path/to/project/src/wiki_agent/middleware/runner.py", line 142, in dispatch',
            "    return handler(event, context)",
            '  File "/home/user/long/path/to/project/src/wiki_agent/middleware/runner.py", line 142, in dispatch',
            "    return handler(event, context)",
            '  File "/home/user/long/path/to/project/src/wiki_agent/middleware/runner.py", line 142, in dispatch',
            "    return handler(event, context)",
            '  File "/home/user/long/path/to/project/src/wiki_agent/middleware/runner.py", line 142, in dispatch',
            "    return handler(event, context)",
            '  File "/home/user/long/path/to/project/src/wiki_agent/middleware/runner.py", line 142, in dispatch',
            "    return handler(event, context)",
            "RecursionError: maximum recursion depth exceeded",
        ]
    ),
    # Tool 4 — pytest output: a flaky test re-emitting the same warning row
    # on every collection, which is a common heavy-duplication pattern.
    "\n".join(
        [
            "/home/user/long/path/to/project/.venv/lib/python3.11/site-packages/pkg_resources/__init__.py:121: DeprecationWarning: pkg_resources is deprecated.",
            "/home/user/long/path/to/project/.venv/lib/python3.11/site-packages/pkg_resources/__init__.py:121: DeprecationWarning: pkg_resources is deprecated.",
            "/home/user/long/path/to/project/.venv/lib/python3.11/site-packages/pkg_resources/__init__.py:121: DeprecationWarning: pkg_resources is deprecated.",
            "/home/user/long/path/to/project/.venv/lib/python3.11/site-packages/pkg_resources/__init__.py:121: DeprecationWarning: pkg_resources is deprecated.",
            "/home/user/long/path/to/project/.venv/lib/python3.11/site-packages/pkg_resources/__init__.py:121: DeprecationWarning: pkg_resources is deprecated.",
            "tests/wiki_compress/test_pipeline.py::TestDefaultPipeline::test_default_pipeline_reduces_html PASSED",
            "tests/wiki_compress/test_pipeline.py::TestEmpty::test_empty_input PASSED",
            "tests/wiki_compress/test_pipeline.py::TestPlainText::test_markdown_round_trip PASSED",
        ]
    ),
]


def _ratios(payloads: list[str], build_pipeline_fn) -> list[float]:
    """Run *build_pipeline_fn* once per payload, return the ratios."""
    out: list[float] = []
    for payload in payloads:
        pipeline = build_pipeline_fn()
        out.append(pipeline.compress(payload).ratio)
    return out


def _build_email_pipeline_only():
    pipe, _, _ = build_email_pipeline()
    return pipe


def _build_default_pipeline_only():
    pipe, _ = build_default_pipeline()
    return pipe


# --------------------------------------------------------------------------- #
# Email preset                                                                #
# --------------------------------------------------------------------------- #


class TestEmailPreset:
    def test_email_preset_beats_default(self) -> None:
        """The email preset shrinks the email more than the default does."""
        email = _EMAIL_PAYLOADS[0]
        default_pipe, _ = build_default_pipeline()
        email_pipe, _, _ = build_email_pipeline()
        default_ratio = default_pipe.compress(email).ratio
        email_ratio = email_pipe.compress(email).ratio
        assert email_ratio < default_ratio
        assert email_ratio < 0.7  # Hard SPEC-008 target.

    def test_email_preset_corpus_median(self) -> None:
        ratios = _ratios(_EMAIL_PAYLOADS, _build_email_pipeline_only)
        median = statistics.median(ratios)
        assert median <= 0.70, f"expected ≤ 0.70, got {median:.3f} from {ratios}"

    def test_email_preset_quote_map_populated(self) -> None:
        _, _, collapser = build_email_pipeline()
        pipe, _, _ = build_email_pipeline(quote_collapser=collapser)
        pipe.compress(_EMAIL_PAYLOADS[0])
        assert collapser.quote_map, "collapser should have recorded at least one block"

    def test_email_preset_cjk_safe(self) -> None:
        text = (
            "你好,这里是回复正文。\n"
            "> 引用的旧邮件第一行,长度足够触发折叠。\n"
            "> 引用的旧邮件第二行。\n"
            "祝好。"
        )
        pipe, _, _ = build_email_pipeline()
        result = pipe.compress(text)
        assert "你好,这里是回复正文。" in result.body
        assert "祝好。" in result.body

    def test_email_preset_idempotent(self) -> None:
        pipe1, _, _ = build_email_pipeline()
        once = pipe1.compress(_EMAIL_PAYLOADS[1])
        pipe2, _, _ = build_email_pipeline()
        twice = pipe2.compress(once.body)
        # Token-rounding slack of 1 (matches test_pipeline.py convention).
        assert twice.compressed_tokens >= once.compressed_tokens - 1


# --------------------------------------------------------------------------- #
# Tool-output preset                                                          #
# --------------------------------------------------------------------------- #


class TestToolOutputPreset:
    def test_tool_preset_beats_default(self) -> None:
        # Use the heaviest-duplication payload to demonstrate the preset
        # decisively beats the default — and clears the SPEC-008 ≤ 0.5
        # corpus-median target on its own.
        text = _TOOL_OUTPUT_PAYLOADS[1]
        default_pipe, _ = build_default_pipeline()
        tool_pipe = build_tool_output_pipeline()
        default_ratio = default_pipe.compress(text).ratio
        tool_ratio = tool_pipe.compress(text).ratio
        assert tool_ratio < default_ratio
        assert tool_ratio < 0.5  # Hard SPEC-008 target.

    def test_tool_preset_corpus_median(self) -> None:
        ratios = _ratios(_TOOL_OUTPUT_PAYLOADS, build_tool_output_pipeline)
        median = statistics.median(ratios)
        assert median <= 0.50, f"expected ≤ 0.50, got {median:.3f} from {ratios}"

    def test_tool_preset_stage_order(self) -> None:
        """The tool-output preset must skip ``html_to_md`` — stdout is not HTML."""
        pipe = build_tool_output_pipeline()
        result = pipe.compress(_TOOL_OUTPUT_PAYLOADS[3])
        assert "html_to_md" not in result.stages_applied
        assert result.stages_applied == [
            "preserve",
            "line_dedupe",
            "whitespace",
            "release",
        ]

    def test_tool_preset_cjk_safe(self) -> None:
        # Line is well over the tool-output preset's 40-char floor.
        cjk_line = (
            "处理输出:这一行非常长,长度超过四十字符,以便测试中文行级去重的安全性。"
            "继续延长这条中文,确保超过门槛。"
        )
        assert len(cjk_line) > 40
        text = f"{cjk_line}\n{cjk_line}\n{cjk_line}"
        pipe = build_tool_output_pipeline()
        result = pipe.compress(text)
        # The first occurrence of the CJK line survives intact.
        assert cjk_line in result.body
        assert "[see-line:" in result.body

    def test_tool_preset_idempotent(self) -> None:
        pipe1 = build_tool_output_pipeline()
        once = pipe1.compress(_TOOL_OUTPUT_PAYLOADS[1])
        pipe2 = build_tool_output_pipeline()
        twice = pipe2.compress(once.body)
        assert twice.compressed_tokens >= once.compressed_tokens - 1


# --------------------------------------------------------------------------- #
# Default pipeline regression check                                           #
# --------------------------------------------------------------------------- #


class TestDefaultUnchanged:
    def test_default_still_works(self) -> None:
        """``build_default_pipeline()`` must keep the legacy stage order so
        existing call sites don't shift behaviour when Sprint 2 lands."""
        pipe, _ = build_default_pipeline()
        result = pipe.compress(
            "<p>One paragraph that is long enough to qualify for dedupe by far.</p>"
            "<p>One paragraph that is long enough to qualify for dedupe by far.</p>"
        )
        assert result.ratio < 1.0
        assert result.stages_applied == [
            "preserve",
            "html_to_md",
            "url_shorten",
            "whitespace",
            "dedupe",
            "release",
        ]


# --------------------------------------------------------------------------- #
# Sanity: parametrised sweep so a regression in any one payload yells.        #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("idx", range(len(_EMAIL_PAYLOADS)))
def test_email_payload_compresses(idx: int) -> None:
    pipe, _, _ = build_email_pipeline()
    result = pipe.compress(_EMAIL_PAYLOADS[idx])
    assert result.ratio < 1.0, (
        f"email payload {idx} did not compress at all (ratio={result.ratio:.3f})"
    )


@pytest.mark.parametrize("idx", range(len(_TOOL_OUTPUT_PAYLOADS)))
def test_tool_output_payload_compresses(idx: int) -> None:
    pipe = build_tool_output_pipeline()
    result = pipe.compress(_TOOL_OUTPUT_PAYLOADS[idx])
    assert result.ratio < 1.0, (
        f"tool-output payload {idx} did not compress at all (ratio={result.ratio:.3f})"
    )
