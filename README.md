<div align="center">

<img src="https://github.com/ather-ops/CineSense-AI/blob/main/02-Assets/Cinesense%20Github%20Design.jpg" alt="CineSense AI Banner" width="100%" />

<br/>
<br/>

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white&labelColor=1e2a3a)
![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-all--MiniLM--L6--v2-2ECC71?style=flat-square&labelColor=1a2a1a)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-E67E22?style=flat-square&labelColor=2a1a0a)
![NLTK](https://img.shields.io/badge/NLTK-Chunking-9B59B6?style=flat-square&labelColor=1a0a2a)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Pipeline-150458?style=flat-square&logo=pandas&logoColor=white&labelColor=0a0a2a)

![Status](https://img.shields.io/badge/Status-Active%20Daily%20Commits-27AE60?style=flat-square&labelColor=0a1a0a)
![Sprint](https://img.shields.io/badge/Sprint-Day%204%20of%20N-E74C3C?style=flat-square&labelColor=2a0a0a)
![License](https://img.shields.io/badge/License-MIT-F39C12?style=flat-square&labelColor=2a1a00)
![Destination](https://img.shields.io/badge/Destination-Chrome%20Extension-4285F4?style=flat-square&logo=google-chrome&logoColor=white&labelColor=0a0a2a)

<br/>

> **Semantic search for Netflix — powered by vector embeddings, sentence-level chunking, ChromaDB, and a RAG pipeline being built one commit at a time. Final destination: a Chrome Extension.**

</div>

---

## What is CineSense AI

CineSense AI is a production-grade semantic search system built on top of the Netflix titles dataset (8,800+ titles). Instead of keyword matching, it understands the *intent* behind your query. Search for "dark psychological thriller with unreliable narrator" and it retrieves the closest semantic matches — not titles that happen to contain those exact words.

This repository is an active, daily-commit build log. Every day, a new capability is added: from raw embeddings to sentence-level chunking, LLM-powered re-ranking, and eventually a fully deployed Chrome Extension that sits inside Netflix itself.

If you want to understand RAG (Retrieval-Augmented Generation) from the ground up — embeddings, vector stores, pipelines — follow the foundation repository first:

**RAG Foundation:** [Cortex\_RAG](https://github.com/ather-ops/Cortex_RAG) — covers one-hot encoding through SentenceTransformers, step by step.

---

## Why This Project Exists

Most "recommendation engines" you see in portfolios are collaborative filtering wrappers around scikit-learn. CineSense AI is different. It is being built the way a production ML system is built: with a real vector store, semantic embeddings, sentence-level document chunking, metadata filtering, and an end-to-end retrieval pipeline heading toward a live browser extension.

The goal is not to demonstrate that you can use a library. The goal is to build something that works in the real world.

---

## Current Architecture

<div align="center">
<img src="https://github.com/ather-ops/CineSense-AI/blob/main/02-Assets/architecture.svg" alt="CineSense AI Pipeline Architecture" width="68%" />
</div>

---

## Build Log — Day by Day

This section is updated with every commit. Each day adds a layer to the pipeline.

---

### Day 1 — Data Foundation and Exploration

**Objective:** Load, clean, and understand the dataset before touching any ML.

**What was built:**
- CSV ingestion with full error handling (FileNotFoundError, generic exceptions)
- Automated null-value strategy: numeric columns filled with mean or median (year columns use median), categorical columns filled with `"unknown"`
- Complete EDA dashboard — five production-quality visualizations:
  - Content type distribution (Movies vs TV Shows) — pie chart
  - Top 10 content-producing countries — horizontal bar chart
  - Top 10 most popular genres (exploded from comma-separated field) — bar chart
  - Content release growth over time — line chart with fill
  - Audience segment distribution by rating — count plot
- Theme: custom green/slate palette (`#2ECC71`, `#2C3E50`) applied consistently across all plots

**Key decisions:**
- Genre extraction required `.str.split(',').explode()` because each title can belong to multiple genres stored in a single field
- Year-based nulls use median (not mean) to avoid skew from outlier release years

---

### Day 2 — Embeddings and Vector Store

**Objective:** Convert text into dense vectors and store them in a queryable vector database.

**What was built:**
- Combined text field construction: `title + director + cast + listed_in + description` concatenated into a single semantic string per title
- Embedding generation using `SentenceTransformer("all-MiniLM-L6-v2")` — fast, lightweight, strong semantic quality at 384 dimensions
- ChromaDB collection initialized with full metadata per document:
  - `title`, `type`, `country`, `release_year`, `added_year`, `rating`, `listed_in`, `description`
  - `added_year` extracted by parsing the `date_added` field (fallback: `release_year`)
- Batch insert at size 100 to handle the full 8,800+ record corpus without memory issues

**Key decisions:**
- ChromaDB chosen over FAISS for this stage: it handles metadata filtering natively without a separate filtering layer
- `added_year` computed separately from `release_year` because content added to Netflix years after release is a meaningful signal for recommendation context

---

### Day 3 — Advanced Retrieval and Filtered Search

**Objective:** Build a search function that goes beyond "nearest neighbor" — supporting compound metadata filters alongside semantic similarity.

**What was built:**
- `advanced_netflix_search()` function with the following filter parameters:
  - `genre` — `$contains` match on the `listed_in` field
  - `min_year` / `max_year` — `$gte` / `$lte` range filters on `release_year`
  - `rating` — exact match filter
  - `movie_type` — `"Movie"` or `"TV Show"` filter
  - `top_k` — configurable result count
- Compound filter logic: single conditions pass directly; multiple conditions are wrapped in ChromaDB's `$and` operator
- Four test queries covering: basic semantic search, genre + year filter, rating + type filter, year range filter
- Search results export to CSV via `save_search_results()`

**Example queries tested:**

| Query | Filters | Result |
|---|---|---|
| "Action Thriller" | none | Top 3 semantic matches |
| "Romantic Comedy" | genre=Romantic, year>=2020 | Filtered recent romcoms |
| "Documentary" | rating=PG-13, type=Movie | Filtered documentaries |
| "Crime Drama" | year 2019-2021 | Time-scoped crime content |

---

### Day 4 — Document Chunking

**Objective:** Handle long combined-text documents properly by splitting them at sentence boundaries before embedding — improving retrieval precision on verbose entries.

**What was built:**

`chunk_text_by_sentences(text, target_words=400, min_words=100)` — splits any document at natural sentence boundaries using NLTK's `sent_tokenize`. Targets 400 words per chunk. Orphaned final segments below 100 words are merged into the preceding chunk rather than stored as standalone noise.

`should_chunk(text, max_words=500)` — lightweight gate function. Documents under 500 words skip the chunking process entirely and are inserted as-is, preserving performance on the majority of the dataset where descriptions are short.

The metadata schema was extended to track `chunk_index` and `total_chunks` per stored vector, allowing the retrieval layer to surface exactly which portion of a title matched the query — and enabling future LLM re-ranking to reconstruct full document context from its parts.

IDs migrated from `show_id` to `chunk_{i}` format to handle the one-to-many relationship between titles and their stored vectors.

**Chunking design decisions:**

| Decision | Rationale |
|---|---|
| Sentence-boundary split (NLTK) over fixed-word split | Preserves grammatical completeness — critical for meaningful embeddings |
| 500-word gate (`should_chunk`) | Most Netflix descriptions are short; skipping chunking on them avoids unnecessary overhead |
| Merge short final segments | Prevents low-signal orphan chunks from polluting search results |
| `chunk_index` + `total_chunks` in metadata | Enables future context reconstruction for LLM re-ranking |

---

### Coming Next

| Day | Planned Work |
|---|---|
| Day 5 | LLM integration — re-rank results using an LLM layer, natural language answer generation |
| Day 6 | API layer — FastAPI endpoint wrapping the full search pipeline |
| Day 7+ | Chrome Extension scaffold, Netflix DOM injection, live overlay UI |

---

## Repository Structure

```
CineSense-AI/
|
|-- 01-Notebooks/
|   |-- day1_eda.ipynb
|   |-- day2_embeddings.ipynb
|   |-- day3_retrieval.ipynb
|   |-- day4_chunking.ipynb
|
|-- 02-Assets/
|   |-- Cinesense Github Design.jpg
|   |-- architecture.svg
|
|-- 03-Data/
|   |-- netflix_titles.csv
|
|-- 04-Extension/              # Chrome Extension (in progress)
|
|-- README.md
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Data | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Tokenization | NLTK (`sent_tokenize`) |
| Embeddings | SentenceTransformers (`all-MiniLM-L6-v2`) |
| Vector Store | ChromaDB |
| API (upcoming) | FastAPI |
| Extension (upcoming) | Chrome Extension Manifest V3 |

---

## Getting Started

**Clone the repository:**

```bash
git clone https://github.com/ather-ops/CineSense-AI.git
cd CineSense-AI
```

**Install dependencies:**

```bash
pip install pandas numpy matplotlib seaborn sentence-transformers chromadb nltk
```

**Run the notebook:**

Open `01-Notebooks/day4_chunking.ipynb` in Jupyter and run all cells. Place `netflix_titles.csv` in the `03-Data/` directory before running.

**Run a search:**

```python
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

CineSense AI sits on top of a RAG foundation. If you want to understand how embeddings work before jumping into this repository, start here:

**[Cortex\_RAG](https://github.com/ather-ops/Cortex_RAG)** — A structured learning repository covering the complete embedding pipeline from one-hot encoding through SentenceTransformers, with annotated notebooks and progressive complexity.

---

## Author

**Ather Assadullah** — Self-taught AI/ML engineer based in Kashmir, India.

Building production-grade ML systems independently, one commit at a time.

- GitHub: [@ather-ops](https://github.com/ather-ops)
- LinkedIn: [ather-assadullah-164492301](https://linkedin.com/in/ather-assadullah-164492301)
- Portfolio: [portofolio-eight-fawn.vercel.app](https://portofolio-eight-fawn.vercel.app)

---

## License

This project is licensed under the MIT License.

---

<div align="center">

Built with focus. Committed daily. Shipping to production.

</div>
