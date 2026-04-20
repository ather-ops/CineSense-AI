<div align="center">

<img src="https://github.com/ather-ops/CineSense-AI/blob/main/02-Assets/Light%20Purple%20and%20White%20Gradient%20Grainy%20Simple%20Abstract%20Offline%20Twitch%20Banner.png" alt="CineSense AI Banner" width="100%" />

<br/>
<br/>

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white&labelColor=0d1117)
![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-MiniLM-2ECC71?style=flat-square&labelColor=0d1117)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Persistent-E67E22?style=flat-square&labelColor=0d1117)
![Gemini](https://img.shields.io/badge/Gemini-2.5%20Flash-8957E5?style=flat-square&logo=google&logoColor=white&labelColor=0d1117)
![NLTK](https://img.shields.io/badge/NLTK-sent__tokenize-9B59B6?style=flat-square&labelColor=0d1117)
![Pandas](https://img.shields.io/badge/Pandas-Pipeline-150458?style=flat-square&logo=pandas&logoColor=white&labelColor=0d1117)
![Status](https://img.shields.io/badge/Status-Active%20Daily%20Commits-27AE60?style=flat-square&labelColor=0d1117)
![Sprint](https://img.shields.io/badge/Sprint-Day%206-8957E5?style=flat-square&labelColor=0d1117)
![Destination](https://img.shields.io/badge/Target-Chrome%20Extension-4285F4?style=flat-square&logo=google-chrome&logoColor=white&labelColor=0d1117)
![License](https://img.shields.io/badge/License-MIT-F39C12?style=flat-square&labelColor=0d1117)

<br/>

> **Semantic search for Netflix — sentence-level chunking, ChromaDB vector store, and Gemini 2.5 Flash generating natural language recommendations. Built daily. Shipping as a Chrome Extension.**

</div>

---

## What is CineSense AI

CineSense AI is a production-grade RAG system built on the Netflix titles dataset (8,800+ titles). You describe what you want to watch in plain language. The pipeline encodes your query into a vector, retrieves the most semantically relevant chunks from ChromaDB, and sends them to Gemini 2.5 Flash — which returns ranked recommendations with per-title reasoning.

This repository is a daily-commit build log. Every day, one new capability is added. The final destination is a Chrome Extension that sits inside Netflix itself.

**RAG Foundation:** [Cortex\_RAG](https://github.com/ather-ops/Cortex_RAG) — learn the complete embedding pipeline from one-hot encoding through SentenceTransformers before diving into this repo.

---

## Why This Project Exists

Most "recommendation engines" in portfolios are collaborative filtering wrappers around scikit-learn. CineSense AI is different. It is a full RAG pipeline — real vector store, sentence-level chunking, compound metadata filtering, and an LLM layer generating grounded answers from retrieved context. Not a toy. Not a wrapper. Built to ship.

---

## Pipeline Architecture

<div align="center">
  <img src="https://github.com/ather-ops/CineSense-AI/blob/main/02-Assets/architecture.svg?raw=true" alt="CineSense AI Pipeline Architecture" width="72%" />
</div>

---

## Repository Structure

<div align="center">
  <img src=" https://github.com/ather-ops/CineSense-AI/blob/main/02-Assets/Repo-Struture.png" alt="CineSense AI Repository Structure" width="72%" />
</div>

---

## Roadmap and Priorities

Work is executed in strict priority order. The next item on this list is always the only item being worked on.

| Priority | Task | Status |
|---|---|---|
| 1 | `ingestion.py` — full pipeline to ChromaDB, production-ready | Done |
| 2 | `rag_engine.py` — retrieval + Gemini 2.5 Flash LLM layer | Done |
| 3 | Unit tests — ingestion correctness, metadata schema, retrieval smoke tests | In progress |
| 4 | `app.py` — Streamlit UI wrapping the full RAG pipeline | Pending |
| 5 | FastAPI layer — REST endpoint for external access | Pending |
| 6 | Chrome Extension scaffold — Manifest V3, Netflix DOM injection | Pending |
| 7 | Chrome Extension UI — live semantic search overlay inside Netflix | Pending |

Progress is tracked commit by commit. Every merge to `main` moves exactly one item from Pending to Done.

---

## Testing

Tests live in `03-Core/tests/`. The goal is to cover ingestion and retrieval logic before adding any new pipeline layers — not after.

**Current test coverage:**

- `fill_missing()` — year columns use median, categoricals fill with `"unknown"`, no nulls remain
- `sentence_chunk()` — chunks never exceed `max_sentences`, all chunks are non-empty, sentences are complete
- ChromaDB insert — chunk count matches expected, metadata fields are present and correctly typed, IDs follow `show_id_chunk_N` format
- `retrieve()` — returns results for a known query, compound filters reduce result count vs. no filter

**Running tests:**

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
- EDA dashboard — five visualizations saved to `04-Visuals/`
- Consistent palette: `#2ECC71` green / `#2C3E50` slate

**Key decisions:**
- Genre extraction uses `.str.split(',').explode()` — each title belongs to multiple genres in one field
- Year nulls use median — resistant to outlier release years skewing the imputed value

---

### Day 2 — Embeddings and Vector Store

**Objective:** Convert text into dense vectors and store them in a queryable vector database.

**What was built:**
- Combined text field: `title + director + cast + listed_in + description`
- Embeddings via `SentenceTransformer("all-MiniLM-L6-v2")` — 384-dimensional dense vectors
- ChromaDB collection with full per-document metadata
- Batch insert at size 100

**Key decisions:**
- ChromaDB over FAISS: metadata filtering is native — no separate filter layer needed

---

### Day 3 — Advanced Retrieval and Filtered Search

**Objective:** Semantic similarity plus structured metadata filters applied simultaneously.

**What was built:**
- `retrieve()` with compound `$and` filter support: genre, year range, rating, type
- Four query patterns tested: basic semantic, genre + year, rating + type, year range
- CSV export via `save_search_results()`

---

### Day 4 — Random Chunking (Experimental)

**Objective:** First attempt at splitting long documents before embedding.

Built a basic word-count splitting approach. Produced incomplete sentences mid-chunk, which degrades embedding quality. Deliberate proof-of-concept that led to the Day 5 overhaul.

---

### Day 5 — Sentence-Level Chunking and Full Refactor

**Objective:** Replace experimental chunking with linguistically correct implementation. Full code review.

**What was built:**

`sentence_chunk(text, max_sentences=2)` — NLTK `sent_tokenize` for true sentence-boundary detection. No mid-sentence cuts. Chunk IDs: `show_id_chunk_N`.

Full pipeline refactor: type hints on all functions, explicit metadata type casting, named constants, title-level deduplication in retrieval.

**Day 4 vs Day 5:**

| Aspect | Day 4 | Day 5 |
|---|---|---|
| Split strategy | Fixed word-count | Sentence-boundary detection |
| Sentence integrity | Cuts mid-sentence | Always complete |
| Chunk IDs | `chunk_{i}` | `{show_id}_chunk_{N}` |
| Metadata typing | Implicit | Explicit cast per field |

---

### Day 6 — LLM Layer: Gemini 2.5 Flash

**Objective:** Add a language model on top of the retrieval layer to generate natural language recommendations grounded in retrieved context.

**What was built:**

`rag_engine.py` — the complete RAG answer generation layer:

`build_context(results)` — deduplicates retrieved chunks by title and formats them as a numbered context block. Prevents the LLM from seeing the same title twice with conflicting chunk content.

`rag_answer(llm, query, results)` — constructs a grounded prompt from the retrieved context and sends it to Gemini 2.5 Flash. The model is instructed to recommend exactly 3 titles from the retrieved set with one-sentence reasoning per title. It cannot hallucinate titles that were not retrieved — the context is the only source.

`cinesense(query, ...)` — the top-level interface combining retrieval and generation in a single call.

ChromaDB upgraded from `Client()` (in-memory) to `PersistentClient(path="./chroma_data")` — the vector store now survives restarts. `ingestion.py` runs once; `rag_engine.py` loads the persisted store on every subsequent query.

**Security:** The API key is loaded from the `GEMINI_API_KEY` environment variable — never hardcoded in source.

**Prompt design:**

```
You are CineSense AI, a Netflix recommendation assistant powered by Gemini 2.5 Flash.

A user is looking for: '{query}'

Based only on the titles below, recommend the top 3 most relevant matches.
For each recommendation explain in one sentence why it fits the query.

RETRIEVED TITLES:
{context}
```

The phrase "based only on the titles below" is load-bearing — it prevents the LLM from recommending Netflix titles it knows from training that were not retrieved. This keeps the output grounded to the vector store.

**Example output:**

```
Query: something emotional and heartbreaking

1. The Pursuit of Happyness (2006)
   A father's relentless struggle through homelessness makes this
   one of the most emotionally raw films on the platform.

2. Grave of the Fireflies (1988)
   A devastating wartime story told through two siblings —
   widely considered one of the most heartbreaking films ever made.

3. Room (2015)
   A mother and son's life in captivity and the trauma of re-entering
   the world delivers sustained emotional weight throughout.
```

---

## Getting Started

```bash
git clone https://github.com/ather-ops/CineSense-AI.git
cd CineSense-AI
pip install -r requirements.txt
```

Set your Gemini API key:

```bash
export GEMINI_API_KEY=your_key_here
```

Run ingestion once to build and persist the vector store:

```bash
python 03-Core/ingestion.py
```

Then run queries through the RAG engine:

```bash
python 03-Core/rag_engine.py
```

Or call directly from Python:

```python
from rag_engine import load_resources, cinesense

embed_model, collection, llm = load_resources()

cinesense(
    "a slow-burn psychological thriller",
    collection=collection,
    embed_model=embed_model,
    llm=llm,
    genre="Thrillers",
    min_year=2015,
)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Data | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Tokenization | NLTK `sent_tokenize` |
| Embeddings | SentenceTransformers `all-MiniLM-L6-v2` |
| Vector Store | ChromaDB `PersistentClient` |
| LLM | Gemini 2.5 Flash (`google-generativeai`) |
| Testing | pytest |
| UI (upcoming) | Streamlit |
| Extension (upcoming) | Chrome Extension Manifest V3 |

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
