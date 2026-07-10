"""CLI for the knowledge-graph QA tooling (issues UA-3 / UA-4).

Usage::

    # Wiki-health report (human summary) + JSON
    sbw-wiki-qa health --graph knowledge-base/.understand-anything/knowledge-graph.json

    # Gate CI on regressions relative to a committed baseline
    sbw-wiki-qa health --graph <graph.json> --baseline <baseline.json> --fail-on-regression

    # Record the current findings as the accepted baseline
    sbw-wiki-qa health --graph <graph.json> --baseline <baseline.json> --update-baseline

    # Render the guided tour to Markdown (UA-4)
    sbw-wiki-qa tour --graph <graph.json> --out docs/.../guided-tour.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TextIO

from wiki_qa.baseline import compute_regressions, load_baseline, save_baseline
from wiki_qa.graph import load_graph
from wiki_qa.linter import HealthReport, lint
from wiki_qa.tour import render_tour


def _print_summary(report: HealthReport, stream: TextIO) -> None:
    s = report.to_dict()["summary"]
    print(
        f"[wiki-qa] orphans={s['orphans']} broken_wikilinks={s['broken_wikilinks']} "
        f"duplicate_slugs={s['duplicate_slugs']} (total={s['total']})",
        file=stream,
    )


def _cmd_health(args: argparse.Namespace) -> int:
    graph = load_graph(args.graph)
    report = lint(graph)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        _print_summary(report, sys.stderr)

    if args.update_baseline:
        if not args.baseline:
            print("[wiki-qa] --update-baseline requires --baseline PATH", file=sys.stderr)
            return 2
        save_baseline(args.baseline, report)
        print(f"[wiki-qa] baseline written to {args.baseline}", file=sys.stderr)
        return 0

    if args.fail_on_regression:
        baseline = load_baseline(args.baseline) if args.baseline else load_baseline("/nonexistent")
        regressions = compute_regressions(report, baseline)
        if regressions.has_regressions:
            print(
                f"[wiki-qa] FAIL: {regressions.total} new issue(s) vs baseline — "
                f"orphans={len(regressions.orphans)} "
                f"broken={len(regressions.broken_wikilinks)} "
                f"duplicate_slugs={len(regressions.duplicate_slugs)}",
                file=sys.stderr,
            )
            for key in (
                *regressions.orphans,
                *regressions.broken_wikilinks,
                *regressions.duplicate_slugs,
            ):
                print(f"  + {key}", file=sys.stderr)
            return 1
        print("[wiki-qa] OK: no regressions vs baseline", file=sys.stderr)

    return 0


def _cmd_tour(args: argparse.Namespace) -> int:
    data = json.loads(Path(args.graph).read_text(encoding="utf-8"))
    markdown = render_tour(data)
    if args.out:
        Path(args.out).write_text(markdown + "\n", encoding="utf-8")
        print(f"[wiki-qa] tour written to {args.out}", file=sys.stderr)
    else:
        print(markdown)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sbw-wiki-qa",
        description="QA & navigation tooling over the Understand-Anything knowledge graph.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    health = sub.add_parser("health", help="Lint the wiki for orphans / broken links / dup slugs.")
    health.add_argument("--graph", required=True, help="Path to the UA knowledge-graph JSON.")
    health.add_argument("--baseline", help="Path to the accepted-issues baseline JSON.")
    health.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit non-zero if new issues appear vs the baseline.",
    )
    health.add_argument(
        "--update-baseline",
        action="store_true",
        help="Write current findings to --baseline and exit.",
    )
    health.add_argument(
        "--json", action="store_true", help="Emit the full report as JSON to stdout."
    )
    health.set_defaults(func=_cmd_health)

    tour = sub.add_parser("tour", help="Render the graph's guided tour to Markdown.")
    tour.add_argument("--graph", required=True, help="Path to the UA knowledge-graph JSON.")
    tour.add_argument("--out", help="Write Markdown here instead of stdout.")
    tour.set_defaults(func=_cmd_tour)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func = args.func
    result: int = func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
