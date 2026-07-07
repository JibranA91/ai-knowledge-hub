# Wiki Schema

This document defines the conventions the AI must follow when maintaining this wiki.

## Page Types

Every page must declare a `type:` field in its YAML frontmatter. This is how the server identifies the page's role — it drives provenance stamping, deletion cleanup, and recalibration safety, **independent of folder name**.

| Type | Role | Frontmatter |
|------|------|-------------|
| `source_summary` | Summary of an ingested source document | `type: source_summary` |
| `concept` | Topic / concept knowledge page | `type: concept` |
| `entity` | Named entity (person, team, product, system, process) | `type: entity` |
| `rca` | Root-cause analysis for an incident or bug | `type: rca` |
| `query_result` | Saved Q&A page (written by the system, not the AI) | `type: query_result` |

The planner picks the type for each page it plans, and the server stamps it into frontmatter at write time. Never invent new type values.

---

## System-Injected Fields

The fields below are **stamped by the server** after each page is written. Never fabricate, guess, or fill in values for them — follow the instructions in the table exactly.

| Field | Pages | Instruction |
|-------|-------|-------------|
| `type` | All pages | The server stamps this from the planner's choice. You may include it in your draft as documentation, but the server is authoritative. |
| `date_ingested` | `type: source_summary` only | Include the **exact placeholder** `<system-injected>`. The server replaces it with today's date. On subsequent updates the server preserves the original date — never a different one. |
| `uploaded_file` | `type: source_summary` only | **Omit this field entirely.** The server detects its absence and injects the real filename. If you include it, the server skips injection and whatever you wrote stays. |

For all other date values (incident dates, due dates, etc.) extract them verbatim from the source document. If the document does not contain a date, leave the field blank — never infer or approximate.

---

## Directory Structure

```
wiki/
├── sources/      — One summary page per ingested source document
├── concepts/     — Topic and concept pages (the main knowledge layer)
├── entities/     — Named entities: people, teams, products, systems
├── rca/          — Root-cause analyses for incidents, failures, and bugs
└── queries/      — Saved question answers (written by the system, not the AI)
```

Folder names are conventions, not contracts — the server identifies page roles by `type:` frontmatter, not folder path. Orgs may rename or restructure these folders by editing this schema, as long as every page still declares one of the five recognised `type:` values.

## Page Naming

- Lowercase, words separated by hyphens: `data-pipeline-overview.md`
- Be specific: `aws-bedrock-setup.md` not `setup.md`

---

## Source Summary Page Format (`sources/`)

```markdown
---
title: "[Document Title]"
tags: [tag1, tag2]
date_ingested: <system-injected>
type: source_summary
entities: []
related: []
---

# [Document Title]

**Source**: original-filename.pdf
**Date ingested**: <system-injected>

## Summary
2–3 paragraph summary of the document.

## Key Points
- Most important takeaway
- Second key point

## Related Pages
- [[concept-name]] — reason this concept is relevant
```

**Rules**:
- Include `date_ingested: <system-injected>` and `**Date ingested**: <system-injected>` exactly as shown — the server replaces both.
- Do **not** include an `uploaded_file:` field — the server injects it automatically.

---

## Concept Page Format (`concepts/`)

```markdown
---
title: "[Concept Name]"
type: concept
tags: [tag1, tag2]
entities: []
related: []
---

# [Concept Name]

## Overview
Clear definition and context.

## Details
Deeper explanation, examples, nuances.

## Open Questions
- Things that are unclear or need more sources

## Sources
- [[source-title]] — what this source contributes
```

---

## Entity Page Format (`entities/`)

```markdown
---
title: "[Entity Name]"
type: entity
category: person | team | product | system | process
tags: [tag1, tag2]
related: []
---

# [Entity Name]

**Type**: person | team | product | system | process
**Also known as**: [aliases, if any]

## Description
Concise description of who or what this entity is.

## Role / Responsibilities
What this entity does or is responsible for.

## Related Concepts
- [[concept-name]] — relationship to this entity

## Sources
- [[source-title]] — what was learned from this source
```

---

## RCA Page Format (`rca/`)

```markdown
---
title: "[RCA: Brief Incident Title]"
type: rca
tags: [rca, tag1, tag2]
date: YYYY-MM-DD
severity: critical | high | medium | low
status: open | resolved | monitoring
related: []
---

# RCA: [Brief Incident Title]

**Date**: YYYY-MM-DD
**Severity**: critical | high | medium | low
**Status**: open | resolved | monitoring
**Owner**: [person or team]

## Summary
1–2 sentence description of what happened and its impact.

## Timeline
- `HH:MM` — Event description

## Root Cause
Clear, specific explanation of the underlying cause.

## Contributing Factors
- Factor one

## Impact
Describe affected systems, users, or processes and the duration.

## Resolution
What was done to resolve the incident.

## Action Items
| Action | Owner | Due Date | Status |
|--------|-------|----------|--------|
| Description | Team | YYYY-MM-DD | open |

## Lessons Learned
- Key takeaway

## Related Pages
- [[concept-name]] — relevant concept
```

**Rules for RCA dates**:
- Only include Timeline and Action Items if they appear in the source document
- Only include `date:` and `**Date**:` if the exact incident date appears in the source document.
- Only populate `Due Date` cells in the Action Items table if a due date is stated in the source.
- Never infer, approximate, or fabricate any date — leave the field blank if it is not in the source.

---

## Cross-linking

Use `[[page-name]]` syntax (without the `.md` extension) to link between pages.
Always cross-link when you mention an entity or concept that has its own page.

---

## Audit Log Entry Format

The planner emits a single `log_entry` per ingest, which the server appends to the audit log:

```
## [<system-injected date>] ingest | Document Title
Brief description of what was ingested and which pages were updated.
```

The date in log entries is replaced by the server at write time — include any date placeholder and the server will overwrite it.

---

## Ingest Checklist

When ingesting a new source:
1. Create `sources/<slug>.md` summary page
2. Update or create relevant `concepts/` and `entities/` pages
3. If the source describes an incident, failure, or bug investigation, create or update the relevant `rca/<slug>.md` page
4. Add cross-links between related pages

---

## Contradictions

The planner assigns one of three resolution modes to each conflict. The writer must follow the assigned mode exactly.

| Mode | Meaning | What to do |
|------|---------|------------|
| `surface` | Preserve both claims | Add a `## Conflicting Sources` section (see template below) |
| `new_wins` | New document's claim replaces the old | Overwrite the existing claim with the new content; do not preserve the old text |
| `existing_wins` | Existing page's claim is kept | Leave the existing claim unchanged; add a one-sentence note that the new source disagrees |

### `surface` — Conflicting Sources Section

```markdown
## Conflicting Sources

- **[Source A]** states: "[claim from A]"
- **[Source B]** states: "[claim from B]"

Resolution: [brief explanation of which claim is used or why both are preserved]
```

Never silently overwrite an existing claim — always follow the resolution mode the planner assigned.
