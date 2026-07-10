---
name: wiki-ingest
description: "Ingest a source document into the wiki knowledge base"
version: "0.1.0"
agent: ingest-agent
triggers:
  - "ingest"
  - "process"
  - "add source"
  - "import"
inputs:
  - name: source_path
    type: string
    required: true
    description: "Path to the source document relative to raw/"
  - name: force
    type: boolean
    required: false
    default: false
    description: "Re-ingest even if source was already processed"
outputs:
  - name: summary_page
    type: string
    description: "Path to the created summary page"
  - name: pages_created
    type: list
    description: "Paths of newly created wiki pages"
  - name: pages_updated
    type: list
    description: "Paths of updated wiki pages"
  - name: entities_extracted
    type: list
    description: "Names of extracted entities"
  - name: concepts_extracted
    type: list
    description: "Names of extracted concepts"
tags: [ingest, source-processing, wiki]
---

# wiki-ingest

Process a source document through the full ingest pipeline.

## Steps

1. Read the source document from `raw/`
2. Classify the source type (article, paper, bookmark, voice_note)
3. Generate a summary page at `wiki/sources/<slug>.md`
4. Extract and create/update entity pages at `wiki/entities/`
5. Extract and create/update concept pages at `wiki/concepts/`
6. Add entries to `wiki/index.md`
7. Append operation details to `wiki/log.md`

## Quality checks

- Every page has valid YAML frontmatter (type, title, date, sources, tags, origin)
- `type` is the OKF-mandatory field (ADR-045 §9.2); write_page injects it by folder if omitted
- Every entity/concept mention uses `[[wikilink]]` syntax
- Summary is factual and traces to the source document
- Index is updated with all new pages
