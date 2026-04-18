<div align="center">

<img src="https://github.com/ather-ops/CineSense-AI/blob/main/02-Assets/Light%20Purple%20and%20White%20Gradient%20Grainy%20Simple%20Abstract%20Offline%20Twitch%20Banner.png" alt="CineSense AI Banner" width="100%" />

<br/>
<br/>

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white&labelColor=0d1117)
![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-MiniLM-2ECC71?style=flat-square&labelColor=0d1117)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-E67E22?style=flat-square&labelColor=0d1117)
![NLTK](https://img.shields.io/badge/NLTK-sent__tokenize-9B59B6?style=flat-square&labelColor=0d1117)
![Pandas](https://img.shields.io/badge/Pandas-Pipeline-150458?style=flat-square&logo=pandas&logoColor=white&labelColor=0d1117)
![Status](https://img.shields.io/badge/Status-Active%20Daily%20Commits-27AE60?style=flat-square&labelColor=0d1117)
![Sprint](https://img.shields.io/badge/Sprint-Day%205-2ECC71?style=flat-square&labelColor=0d1117)
![Destination](https://img.shields.io/badge/Target-Chrome%20Extension-4285F4?style=flat-square&logo=google-chrome&logoColor=white&labelColor=0d1117)
![License](https://img.shields.io/badge/License-MIT-F39C12?style=flat-square&labelColor=0d1117)

<br/>

> **Semantic search for Netflix — powered by sentence-level chunking, vector embeddings, and ChromaDB. Built daily. Shipping as a Chrome Extension.**

</div>

---

## What is CineSense AI

CineSense AI is a production-grade semantic search system built on the Netflix titles dataset (8,800+ titles). Instead of keyword matching, it understands the *intent* behind a query. Search for "dark psychological thriller with unreliable narrator" and it returns the closest semantic matches — not titles that happen to contain those exact words.

This repository is a daily-commit build log. Every day, one new capability is added to the pipeline. The final destination is a Chrome Extension that sits inside Netflix itself and replaces the search bar with a semantic engine.

**RAG Foundation:** [Cortex\_RAG](https://github.com/ather-ops/Cortex_RAG) — learn the complete embedding pipeline from one-hot encoding through SentenceTransformers before diving into this repo.

---

## Why This Project Exists

Most "recommendation engines" in portfolios are collaborative filtering wrappers around scikit-learn. CineSense AI is different. It is being built the way a production ML system is built: real vector store, proper sentence-level chunking via NLTK, compound metadata filtering, and a retrieval pipeline heading toward a live browser extension.

The goal is not to demonstrate that you can use a library. The goal is to build something that works in the real world.

---

## Pipeline Architecture

<div align="center">
  <img src="https://github.com/ather-ops/CineSense-AI/blob/main/02-Assets/architecture.svg?raw=true" alt="CineSense AI Pipeline Architecture" width="72%" />
</div>

---

## Repository Structure

<div align="center">
  <img src="https://github.com/ather-ops/CineSense-AI/blob/main/02-Assets/repo_structure1.svg?raw=true" alt="CineSense AI Repository Structure" width="72%" />
</div>

---

## Roadmap and Priorities

Work is executed in strict priority order. The next item on this list is always the only item being worked on.

| Priority | Task | Status |
|---|---|---|
| 1 | `ingestion.py` — end-to-end pipeline to ChromaDB, production-ready | In progress |
| 2 | Unit tests for ingestion — chunking correctness, metadata schema, batch insert | Pending |
| 3 | `rag_engine.py` — retrieval pipeline with LLM re-ranking layer | Pending |
| 4 | `app.py` — Streamlit UI wrapping the full search pipeline | Pending |
| 5 | FastAPI layer — REST endpoint for external access | Pending |
| 6 | Chrome Extension scaffold — Manifest V3, Netflix DOM injection | Pending |
| 7 | Chrome Extension UI — live semantic search overlay inside Netflix | Pending |

Progress is tracked commit by commit. Every merge to `main` moves exactly one item from Pending to Done.

---

## Testing

Tests live in `03-Core/tests/`. The goal is to cover the ingestion and retrieval logic before adding any new pipeline layers — not after.

**What is currently tested:**

Tests are being added alongside `ingestion.py` as it is written. The first test suite covers:

- `fill_missing()` — verifies year columns use median, categoricals fill with `"unknown"`, no nulls remain after cleaning
- `sentence_chunk()` — verifies chunks never exceed `max_sentences`, all chunks are non-empty strings, complete sentences are preserved
- ChromaDB insert — verifies chunk count matches expected, metadata fields are present and correctly typed, IDs follow `show_id_chunk_N` format

**Why tests are being written now, not later:**

Ingestion logic is the foundation everything else depends on. If `sentence_chunk` silently cuts sentences mid-word, or if metadata types cause `$gte` filters to fail at query time, those bugs compound through every layer above. Catching them at the ingestion layer costs one test. Catching them at the Chrome Extension layer costs days.

**Running the tests:**

```bash
pip install pytest
pytest 03-Core/tests/ -v
```

---

## Build Log — Day by Day

---

### Day 1 — Data Foundation and Exploration

**Objective:** Load, clean, and understand the dataset before touching any ML.

**What was built:**
- CSV ingestion with full error handling
- Automated null strategy: year columns use median, numeric columns use mean, categoricals filled with `"unknown"`
- EDA dashboard — five visualizations saved to `04-Visuals/`: content type pie, top countries bar, top genres bar, release growth line chart, rating distribution count plot
- Consistent palette: `#2ECC71` green / `#2C3E50` slate across all plots

**Key decisions:**
- Genre extraction uses `.str.split(',').explode()` — each title belongs to multiple genres stored in a single comma-separated field
- Year nulls use median not mean — resistant to outlier release years skewing the imputed value

---

### Day 2 — Embeddings and Vector Store

**Objective:** Convert text into dense vectors and store them in a queryable vector database.

**What was built:**
- Combined text field construction: `title + director + cast + listed_in + description`
- Embeddings via `SentenceTransformer("all-MiniLM-L6-v2")` — 384-dimensional dense vectors
- ChromaDB collection with full metadata per document: `title`, `type`, `country`, `release_year`, `added_year`, `rating`, `listed_in`
- Batch insert at size 100 to handle the full corpus without memory pressure

**Key decisions:**
- ChromaDB over FAISS: metadata filtering is native — no separate filter layer needed
- `added_year` tracked separately because content added to Netflix years after release carries different recommendation context

---

### Day 3 — Advanced Retrieval and Filtered Search

**Objective:** Semantic similarity plus structured metadata filters applied simultaneously.

**What was built:**
- `advanced_netflix_search()` with parameters: `genre`, `min_year`, `max_year`, `rating`, `movie_type`, `top_k`
- Single conditions passed directly; multiple conditions wrapped in ChromaDB's `$and` operator
- Filters: `$contains`, `$gte`, `$lte`, `$eq`
- Four query patterns tested: basic semantic, genre + year, rating + type, year range
- CSV export via `save_search_results()`

**Example queries tested:**

| Query | Filters | Notes |
|---|---|---|
| "Action Thriller" | none | Pure semantic match |
| "Romantic Comedy" | genre=Romantic, year>=2020 | Compound filter |
| "Documentary" | rating=PG-13, type=Movie | Type-scoped |
| "Crime Drama" | year 2019–2021 | Time-scoped |

---

### Day 4 — Random Chunking (Experimental)

**Objective:** First attempt at splitting long documents before embedding.

**What happened:** Built a basic chunking approach manually — word-count splitting without sentence-boundary awareness. Functioned but produced incomplete sentences mid-chunk, which degrades embedding quality. This was a deliberate proof-of-concept iteration that led directly to the Day 5 overhaul.

---

### Day 5 — Proper Sentence-Level Chunking and Full Refactor

**Objective:** Replace the experimental Day 4 chunking with a linguistically correct implementation. Full code review and refactor across the entire pipeline.

**What was built:**

`sentence_chunk(text, max_sentences=2)` — splits combined text using NLTK's `sent_tokenize` for true sentence-boundary detection. Chunks contain exactly 2 complete sentences. No mid-sentence cuts.

Full pipeline refactor:
- Type hints added to all functions
- Metadata values explicitly cast (`int`, `str`) before ChromaDB insert — prevents type mismatch errors at query time
- Chunk IDs migrated to `show_id_chunk_N` format — unique, traceable, human-readable
- `advanced_netflix_search()` upgraded with title-level deduplication and matching chunk preview display
- `fill_missing()` extracted as a named typed function
- Magic values extracted as named constants: `GREEN`, `SLATE`, `BATCH_SIZE`

**Day 4 vs Day 5 chunking comparison:**

| Aspect | Day 4 | Day 5 |
|---|---|---|
| Split strategy | Word-count fixed window | Sentence-boundary detection |
| Library | Manual slicing | NLTK `sent_tokenize` |
| Sentence integrity | Cuts mid-sentence | Always complete sentences |
| Chunk ID format | `chunk_{i}` | `{show_id}_chunk_{N}` |
| Metadata typing | Implicit | Explicit cast per field |
| Result deduplication | None | Title-level dedup in retrieval |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Data | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Tokenization | NLTK `sent_tokenize` |
| Embeddings | SentenceTransformers `all-MiniLM-L6-v2` |
| Vector Store | ChromaDB |
| Testing | pytest |
| UI (upcoming) | Streamlit |
| API (upcoming) | FastAPI |
| Extension (upcoming) | Chrome Extension Manifest V3 |

---

## Getting Started

```bash
git clone https://github.com/ather-ops/CineSense-AI.git
cd CineSense-AI
pip install -r requirements.txt
```

Place `netflix_titles.csv` in `01-Data/` then run:

```bash
python 03-Core/ingestion.py
```

To run a search:

```python
from 03-Core.rag_engine import advanced_netflix_search

results = advanced_netflix_search(
    collection, model,
    query_text="psychological thriller with a twist ending",
    genre="Thrillers",
    min_year=2018,
    top_k=5
)
```

---

## Learn RAG From Scratch

**[Cortex\_RAG](https://github.com/ather-ops/Cortex_RAG)** — annotated notebooks covering the full embedding pipeline from one-hot encoding through SentenceTransformers, built for progressive learning.

---

## Author

**Ather Assadullah** — Self-taught AI/ML engineer, Kashmir, India.

Building production-grade ML systems independently, one commit at a time.

[![GitHub](https://img.shields.io/badge/GitHub-ather--ops-181717?style=flat-square&logo=github&labelColor=0d1117)](https://github.com/ather-ops)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-ather--assadullah-0A66C2?style=flat-square&logo=linkedin&labelColor=0d1117)](https://linkedin.com/in/ather-assadullah-164492301)
[![Portfolio](https://img.shields.io/badge/Portfolio-Live-27AE60?style=flat-square&labelColor=0d1117)](https://portofolio-eight-fawn.vercel.app)

---

## License

MIT License

---

<div align="center">

Built with focus. Committed daily. Shipping to production.

</div>
