<div align="center">

<img src="https://raw.githubusercontent.com/ather-ops/CineSense-AI/main/02-Assets/Cinesense%20Github%20Design.jpg" alt="CineSense AI Banner" width="100%" />

<br/>
<br/>

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white&labelColor=0d1117)
![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-MiniLM-2ECC71?style=flat-square&labelColor=0d1117)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-E67E22?style=flat-square&labelColor=0d1117)
![NLTK](https://img.shields.io/badge/NLTK-Chunking-9B59B6?style=flat-square&labelColor=0d1117)
![Pandas](https://img.shields.io/badge/Pandas-Pipeline-150458?style=flat-square&logo=pandas&logoColor=white&labelColor=0d1117)
![Status](https://img.shields.io/badge/Status-Active%20Daily%20Commits-27AE60?style=flat-square&labelColor=0d1117)
![Sprint](https://img.shields.io/badge/Sprint-Day%204-E74C3C?style=flat-square&labelColor=0d1117)
![Destination](https://img.shields.io/badge/Target-Chrome%20Extension-4285F4?style=flat-square&logo=google-chrome&logoColor=white&labelColor=0d1117)
![License](https://img.shields.io/badge/License-MIT-F39C12?style=flat-square&labelColor=0d1117)

<br/>

> **Semantic search for Netflix — powered by vector embeddings, sentence-level chunking, and ChromaDB. Built daily. Shipping as a Chrome Extension.**

</div>

---

## What is CineSense AI

CineSense AI is a production-grade semantic search system built on the Netflix titles dataset (8,800+ titles). Instead of keyword matching, it understands the *intent* behind a query. Search for "dark psychological thriller with unreliable narrator" and it returns the closest semantic matches — not titles that happen to contain those exact words.

This repository is a daily-commit build log. Every day, one new capability is added to the pipeline. The final destination is a Chrome Extension that sits inside Netflix itself and replaces the search bar with a semantic engine.

If you want to understand RAG (Retrieval-Augmented Generation) from first principles before exploring this repo, start here:

**RAG Foundation:** [Cortex\_RAG](https://github.com/ather-ops/Cortex_RAG) — covers the complete embedding pipeline from one-hot encoding through SentenceTransformers, step by step.

---

## Why This Project Exists

Most "recommendation engines" in portfolios are collaborative filtering wrappers around scikit-learn. CineSense AI is different. It is being built the way a production ML system is built: real vector store, semantic embeddings, sentence-level document chunking, compound metadata filtering, and a retrieval pipeline heading toward a live browser extension.

The goal is not to demonstrate that you can use a library. The goal is to build something that works in the real world.

---

## Pipeline Architecture

<div align="center">
  <img src="https://raw.githubusercontent.com/ather-ops/CineSense-AI/main/02-Assets/architecture.svg" alt="CineSense AI Pipeline Architecture" width="72%" />
</div>

---

## Repository Structure

<div align="center">
  <img src="https://raw.githubusercontent.com/ather-ops/CineSense-AI/main/02-Assets/repo_structure.svg" alt="CineSense AI Repository Structure" width="72%" />
</div>

---

## Build Log — Day by Day

This section is updated with every commit. Each day adds one layer to the pipeline.

---

### Day 1 — Data Foundation and Exploration

**Objective:** Load, clean, and understand the dataset before touching any ML.

**What was built:**
- CSV ingestion with full error handling (`FileNotFoundError`, generic exceptions)
- Automated null strategy: numerics filled with mean or median (year columns use median to avoid skew from outlier release years), categoricals filled with `"unknown"`
- EDA dashboard — five production-quality visualizations saved to `04-Visuals/`:
  - Content type distribution (Movies vs TV Shows) — pie chart
  - Top 10 content-producing countries — horizontal bar
  - Top 10 most popular genres (exploded from comma-separated field) — bar chart
  - Content release growth over time — line chart with area fill
  - Audience segment distribution by rating — count plot
- Consistent palette applied: `#2ECC71` green / `#2C3E50` slate

**Key decisions:**
- Genre extraction uses `.str.split(',').explode()` — each title belongs to multiple genres stored in one field
- Year-based nulls use median not mean — resistant to outlier release years pulling the imputed value

---

### Day 2 — Embeddings and Vector Store

**Objective:** Convert text into dense vectors and store them in a queryable vector database.

**What was built:**
- Combined text field: `title + director + cast + listed_in + description` per title
- Embeddings via `SentenceTransformer("all-MiniLM-L6-v2")` — 384-dimensional dense vectors
- ChromaDB collection with full metadata per document: `title`, `type`, `country`, `release_year`, `added_year`, `rating`, `listed_in`, `description`
- `added_year` parsed separately from `date_added` field (fallback: `release_year`)
- Batch insert at size 100 to handle the full corpus without memory pressure

**Key decisions:**
- ChromaDB over FAISS: handles metadata filtering natively — no separate filter layer required
- `added_year` tracked separately because content added to Netflix years post-release carries different context than its original release year

---

### Day 3 — Advanced Retrieval and Filtered Search

**Objective:** Build a compound search function — semantic similarity and structured metadata filters applied simultaneously.

**What was built:**
- `advanced_netflix_search()` with parameters: `genre`, `min_year`, `max_year`, `rating`, `movie_type`, `top_k`
- Single conditions passed directly to ChromaDB; multiple conditions wrapped in `$and`
- Filter operators in use: `$contains`, `$gte`, `$lte`, exact match
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

### Day 4 — Document Chunking

**Objective:** Improve retrieval precision on long documents by splitting at sentence boundaries before embedding.

**What was built:**

`chunk_text_by_sentences(text, target_words=400, min_words=100)` — splits any document at natural sentence boundaries using NLTK `sent_tokenize`. Targets 400 words per chunk. Orphaned segments below 100 words are merged into the preceding chunk rather than stored as noise.

`should_chunk(text, max_words=500)` — gate function. Documents under 500 words skip chunking entirely. The majority of Netflix descriptions are short — this preserves performance where chunking adds no value.

Metadata schema extended with `chunk_index` and `total_chunks` per vector. IDs migrated from `show_id` to `chunk_{i}` format to handle the one-to-many title-to-chunk relationship.

**Design decisions:**

| Decision | Reason |
|---|---|
| Sentence-boundary split over fixed-word split | Preserves grammatical completeness — critical for meaningful embeddings |
| 500-word gate | Most descriptions are short; chunking them adds overhead with no retrieval benefit |
| Merge short trailing segments | Prevents low-signal orphan chunks from polluting results |
| `chunk_index` + `total_chunks` in metadata | Enables future LLM re-ranking to reconstruct full document context |

---

### Coming Next

| Day | Planned |
|---|---|
| Day 5 | LLM integration — re-rank results, natural language answer generation |
| Day 6 | FastAPI layer — REST endpoint wrapping the full search pipeline |
| Day 7+ | Chrome Extension scaffold, Netflix DOM injection, live overlay UI |

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
| Entry Point | `app.py` |
| API (upcoming) | FastAPI |
| Extension (upcoming) | Chrome Extension Manifest V3 |

---

## Getting Started

```bash
git clone https://github.com/ather-ops/CineSense-AI.git
cd CineSense-AI
pip install pandas numpy matplotlib seaborn sentence-transformers chromadb nltk
```

Place `netflix_titles.csv` in `01-Data/` then open `03-Core/day4_chunking.ipynb` and run all cells.

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

CineSense AI is built on top of a RAG foundation. To understand how the embedding pipeline works before jumping into this repo:

**[Cortex\_RAG](https://github.com/ather-ops/Cortex_RAG)** — annotated notebooks covering the complete pipeline from one-hot encoding through SentenceTransformers, built for progressive learning.

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
