<div align="center">

<img src="https://github.com/ather-ops/CineSense-AI/blob/main/02-Assets/2.png" alt="CineSense AI Banner" width="100%" />

<br/>
<br/>

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-orange?style=for-the-badge)
![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-NLP-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active%20Development-brightgreen?style=for-the-badge)
![Days](https://img.shields.io/badge/Sprint-Day%203%20of%20N-red?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

<br/>

> **Semantic search for Netflix — powered by vector embeddings, ChromaDB, and a RAG pipeline being built one commit at a time. Final destination: a Chrome Extension.**

</div>

---

## What is CineSense AI

CineSense AI is a production-grade semantic search system built on top of the Netflix titles dataset (8,800+ titles). Instead of keyword matching, it understands the *intent* behind your query. Search for "dark psychological thriller with unreliable narrator" and it retrieves the closest semantic matches — not titles that happen to contain those exact words.

This repository is an active, daily-commit build log. Every day, a new capability is added: from raw embeddings to chunking, LLM-powered re-ranking, and eventually a fully deployed Chrome Extension that sits inside Netflix itself.

If you want to understand RAG (Retrieval-Augmented Generation) from the ground up — embeddings, vector stores, pipelines — follow the foundation repository first:

**RAG Foundation:** [Cortex\_RAG](https://github.com/ather-ops/Cortex_RAG) — covers one-hot encoding through SentenceTransformers, step by step.

---

## Why This Project Exists

Most "recommendation engines" you see in portfolios are collaborative filtering wrappers around scikit-learn. CineSense AI is different. It is being built the way a production ML system is built: with a real vector store, semantic embeddings, metadata filtering, and an end-to-end retrieval pipeline heading toward a live browser extension.

The goal is not to demonstrate that you can use a library. The goal is to build something that works in the real world.

---

## Current Architecture

```
netflix_titles.csv  (8,800+ titles)
        |
        v
   [ Data Cleaning & EDA ]
   Null handling, type normalization, visual dashboards
        |
        v
   [ Feature Engineering ]
   Combined text: title + director + cast + genres + description
        |
        v
   [ Embedding Generation ]
   SentenceTransformer: all-MiniLM-L6-v2
   384-dimensional dense vectors per title
        |
        v
   [ Vector Store: ChromaDB ]
   Persistent collection with rich metadata
   (type, country, year, rating, genre)
        |
        v
   [ Advanced Retrieval Engine ]
   Semantic query + metadata filters ($and, $gte, $lte, $contains)
   Returns ranked results with full context
```

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

### Coming Next

| Day | Planned Work |
|---|---|
| Day 4 | Document chunking — split long descriptions into overlapping chunks for finer-grained retrieval |
| Day 5 | LLM integration — re-rank results using an LLM layer, natural language answer generation |
| Day 6+ | API layer (FastAPI), Chrome Extension scaffold, Netflix DOM injection |

---

## Repository Structure

```
CineSense-AI/
|
|-- 01-Notebooks/
|   |-- day1_eda.ipynb
|   |-- day2_embeddings.ipynb
|   |-- day3_retrieval.ipynb
|
|-- 02-Assets/
|   |-- Cinesense Github Design.jpg
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
pip install pandas numpy matplotlib seaborn sentence-transformers chromadb
```

**Run the notebook:**

Open `01-Notebooks/day3_retrieval.ipynb` in Jupyter and run all cells. Place `netflix_titles.csv` in the `03-Data/` directory before running.

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
