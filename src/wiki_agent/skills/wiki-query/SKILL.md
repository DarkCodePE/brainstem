---
name: wiki-query
description: "Answer questions using wiki knowledge base content"
version: "0.1.0"
agent: query-agent
triggers:
  - "query"
  - "ask"
  - "what is"
  - "how does"
  - "explain"
inputs:
  - name: question
    type: string
    required: true
    description: "Natural language question to answer"
  - name: file_answer
    type: boolean
    required: false
    default: false
    description: "Whether to file the answer as a wiki page"
  - name: max_pages
    type: integer
    required: false
    default: 10
    description: "Maximum number of wiki pages to read"
outputs:
  - name: answer
    type: string
    description: "Synthesised answer with citations"
  - name: citations
    type: list
    description: "Wiki page paths cited in the answer"
  - name: filed_path
    type: string
    description: "Path if answer was filed as wiki page (null otherwise)"
tags: [query, search, answer, wiki]
---

# wiki-query

Search the wiki and synthesise a grounded answer with citations.

## Steps

1. Search `wiki/index.md` for relevant pages matching the question
2. Read the top matching pages (up to max_pages)
3. Synthesise an answer grounded in wiki content
4. Cite every factual claim with `(source: wiki/path/to/page.md)`
5. Optionally file the answer at `wiki/answers/<slug>.md`

## Error handling

- If no relevant pages found, return NO_RELEVANT_PAGES error
- Never hallucinate information not present in the wiki
- Clearly state gaps when wiki coverage is partial
