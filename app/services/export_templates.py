"""Static text templates bundled into wiki exports.

`EXPORT_README_TEMPLATE` is the `README.md` written into the export zip
alongside the self-contained `retriever.py`. It is rendered with ``str.format``
in `wiki_engine.export_wiki`, so literal ``{`` / ``}`` must be doubled.

Placeholders: ``{embeddings_section}``, ``{embedding_model}``,
``{embedding_dimensions}``.
"""

EXPORT_README_TEMPLATE = """\
# Wiki Export — Retriever Guide

This export contains all wiki pages, the knowledge graph, and (optionally) pre-computed vector
embeddings. The bundled `retriever.py` lets you find relevant pages for a question without
connecting to any LLM or external service.

---

## Prerequisites

- Python 3.9+
- `numpy` — optional; required only for cosine/hybrid retrieval

```bash
pip install numpy   # optional
```

---

## Quick Start

```bash
# BM25 keyword retrieval (no dependencies)
python retriever.py "What is our vacation policy?"

# Hybrid retrieval with a pre-computed query vector (requires numpy)
python retriever.py "What is our vacation policy?" --query-vector vec.json
```

---

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--top-k N` | 5 | Number of pages to return |
| `--graph-hops N` | 1 | Expand results with N-hop graph neighbours |
| `--bm25-weight F` | 0.4 | Weight for BM25 keyword score |
| `--cosine-weight F` | 0.6 | Weight for cosine similarity score |
| `--query-vector FILE` | — | JSON file with a pre-computed query vector |
| `--export-dir PATH` | `.` | Path to the unzipped export directory |

---

## Python API

```python
from retriever import retrieve

results = retrieve("What is our data retention policy?", top_k=5, graph_hops=1)
for r in results:
    print(r["path"], r["score"])
    # feed r["content"] to your LLM
```

With a pre-computed query vector:

```python
results = retrieve(
    "What is our data retention policy?",
    query_vector=[0.12, -0.34, ...],   # float list from your embedding model
    bm25_weight=0.3,
    cosine_weight=0.7,
)
```

---

## What is BM25?

BM25 scores pages by how frequently your query terms appear, adjusted for page length and how
rare each term is across all pages (TF-IDF style). No ML model is required — it works immediately
out of the box with pure Python.

Before weighting, BM25 scores are min-max normalized to a 0–1 range (each page's score divided by
the highest BM25 score for that query). This puts them on the same scale as cosine similarity, so
`--bm25-weight` and `--cosine-weight` are directly comparable and the defaults behave as intended.
This matches the live system's hybrid search.

Raise `--bm25-weight` for exact-term or technical queries (e.g. error codes, API names).

---

## What is cosine similarity?

Cosine similarity measures how closely aligned two meaning vectors are, regardless of magnitude.
It captures conceptual and paraphrase matches that keyword search misses — e.g. "vacation" matches
"paid time off". Requires `numpy` and a pre-computed `query_vector`.

Raise `--cosine-weight` for conceptual or paraphrase queries.

---

## Embeddings in This Export

{embeddings_section}

---

## Generating a Query Vector with Bedrock

You **must** use the same embedding model and produce vectors of the same dimensions as those
stored in this export, otherwise cosine similarity scores will be meaningless or numpy will
raise a shape error.

```python
import json, boto3

client = boto3.client("bedrock-runtime", region_name="us-east-1")
response = client.invoke_model(
    modelId="{embedding_model}",
    body=json.dumps({{"inputText": "your question here"}}),
)
query_vector = json.loads(response["body"].read())["embedding"]
# len(query_vector) should equal {embedding_dimensions}

# Save to file for CLI use
with open("vec.json", "w") as f:
    json.dump(query_vector, f)
```

---

## Tuning Weights

| Scenario | Recommended settings |
|----------|---------------------|
| Exact term / code / error lookup | `--bm25-weight 0.8 --cosine-weight 0.2` |
| Conceptual / paraphrase query | `--bm25-weight 0.2 --cosine-weight 0.8` |
| General (default) | `--bm25-weight 0.4 --cosine-weight 0.6` |

The defaults mirror the live system's hybrid search configuration.
"""
