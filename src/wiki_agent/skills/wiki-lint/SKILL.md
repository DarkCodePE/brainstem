---
name: wiki-lint
description: "Health-check the wiki for quality issues"
version: "0.1.0"
agent: lint-agent
triggers:
  - "lint"
  - "check"
  - "health"
  - "scan"
  - "validate"
inputs:
  - name: fix
    type: boolean
    required: false
    default: false
    description: "Auto-apply suggested fixes"
  - name: categories
    type: list
    required: false
    description: "Filter categories: contradictions, stale, orphans, cross_refs, frontmatter"
outputs:
  - name: issues
    type: list
    description: "List of detected issues with category, severity, and description"
  - name: pages_scanned
    type: integer
    description: "Number of pages scanned"
  - name: issues_found
    type: integer
    description: "Total issues detected"
  - name: issues_fixed
    type: integer
    description: "Issues auto-fixed (if fix=true)"
tags: [lint, quality, health-check, wiki]
---

# wiki-lint

Scan the wiki for quality issues and report findings.

## Checks performed

1. **Orphan pages**: Pages with no inbound links from other pages
2. **Invalid frontmatter**: Missing required fields (type, title, date, sources, tags, origin) — `type` is OKF-mandatory (ADR-045 §9.2)
3. **Missing cross-references**: Broken wikilinks or missing bidirectional refs
4. **Stale pages**: Pages not updated in over 90 days
5. **Contradictions**: Conflicting claims across entity/concept pages

## Severity levels

- **high**: Invalid frontmatter, broken links, contradictions
- **medium**: Orphan pages, missing bidirectional cross-references
- **low**: Stale information, minor style issues
