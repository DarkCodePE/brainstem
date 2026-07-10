"""Response schemas for Wiki Deep Agent subagents.

Pydantic models that define the structured output each subagent
returns to the orchestrator via Deep Agents' ``response_format``.
The orchestrator receives validated JSON instead of free-form text.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# ------------------------------------------------------------------
# Ingest subagent
# ------------------------------------------------------------------


class IngestResult(BaseModel):
    """Structured result from the ingest subagent."""

    summary_page: str = Field(
        description="Path to the created summary page (e.g. wiki/sources/my-article.md)"
    )
    pages_created: list[str] = Field(
        default_factory=list, description="Paths of newly created wiki pages"
    )
    pages_updated: list[str] = Field(
        default_factory=list, description="Paths of updated wiki pages"
    )
    entities_extracted: list[str] = Field(
        default_factory=list, description="Names of extracted entities"
    )
    concepts_extracted: list[str] = Field(
        default_factory=list, description="Names of extracted concepts"
    )
    lessons_learned: list[str] = Field(
        default_factory=list, description="Schema lessons discovered during ingestion"
    )


# ------------------------------------------------------------------
# Query subagent
# ------------------------------------------------------------------


class QueryResult(BaseModel):
    """Structured result from the query subagent."""

    answer: str = Field(description="Synthesised answer with inline citations")
    citations: list[str] = Field(
        default_factory=list, description="Wiki page paths cited in the answer"
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Confidence score 0-1 based on source coverage"
    )
    filed_path: str | None = Field(
        default=None, description="Path if answer was filed as wiki page, null otherwise"
    )


# ------------------------------------------------------------------
# Lint subagent
# ------------------------------------------------------------------


class LintIssue(BaseModel):
    """A single issue detected by the lint subagent."""

    category: str = Field(
        description="orphan | invalid_frontmatter | missing_cross_ref | stale | contradiction"
    )
    severity: str = Field(description="high | medium | low")
    page_path: str = Field(description="Path to the affected wiki page")
    description: str = Field(description="Clear description of the issue")
    auto_fixed: bool = Field(default=False, description="Whether this issue was auto-fixed")


class LintResult(BaseModel):
    """Structured result from the lint subagent."""

    issues: list[LintIssue] = Field(default_factory=list, description="All detected issues")
    pages_scanned: int = Field(default=0, description="Number of pages scanned")
    issues_fixed: int = Field(default=0, description="Number of issues auto-fixed")


# ------------------------------------------------------------------
# Index subagent
# ------------------------------------------------------------------


class IndexResult(BaseModel):
    """Structured result from the index subagent."""

    entries_added: int = Field(default=0, description="New entries added to index.md")
    entries_updated: int = Field(default=0, description="Existing entries updated")
    stale_removed: int = Field(default=0, description="Stale entries removed")
    backlinks_added: list[str] = Field(
        default_factory=list, description="Pages where backlinks were added"
    )
    broken_links: list[str] = Field(default_factory=list, description="Broken wikilinks detected")


# ------------------------------------------------------------------
# Capture subagent
# ------------------------------------------------------------------


class CaptureResult(BaseModel):
    """Structured result from the capture subagent."""

    obs_id: str = Field(description="Assigned observation ID (e.g. OBS-2026-04-13-001)")
    category: str = Field(
        description="product-gap | process-insight | tool-learning | architecture-insight | research-finding"
    )
    confidence: str = Field(description="high | medium | low")
    file_path: str = Field(description="Path to the observations file")


# ------------------------------------------------------------------
# Review subagent
# ------------------------------------------------------------------


class ThemeCluster(BaseModel):
    """A cluster of related observations identified by the review subagent."""

    theme_name: str = Field(description="Name of the thematic cluster")
    obs_ids: list[str] = Field(description="OBS-IDs in this cluster")
    pattern_strength: int = Field(description="Number of independent observations")
    proposed_graduation: str = Field(description="schema-rule | concept-page | entity-page")
    rationale: str = Field(description="Why this cluster should graduate")


class ReviewResult(BaseModel):
    """Structured result from the review subagent."""

    observations_reviewed: int = Field(default=0, description="Total observations reviewed")
    themes: list[ThemeCluster] = Field(default_factory=list, description="Themed clusters found")
    unmatched_count: int = Field(default=0, description="Observations without a clear pattern")
