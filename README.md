<div align="center">

<img src="https://github.com/ather-ops/CineSense-AI/blob/main/02-Assets/Light%20Purple%20and%20White%20Gradient%20Grainy%20Simple%20Abstract%20Offline%20Twitch%20Banner.png?raw=true" alt="CineSense AI Banner" width="100%" />

<br/>
<br/>

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white&labelColor=0d1117)
![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-MiniLM-2ECC71?style=flat-square&labelColor=0d1117)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Persistent-E67E22?style=flat-square&labelColor=0d1117)
![Gemini](https://img.shields.io/badge/Gemini-2.5%20Flash-8957E5?style=flat-square&logo=google&logoColor=white&labelColor=0d1117)
![NLTK](https://img.shields.io/badge/NLTK-sent__tokenize-9B59B6?style=flat-square&labelColor=0d1117)
![FastAPI](https://img.shields.io/badge/FastAPI-In%20Progress-009688?style=flat-square&logo=fastapi&logoColor=white&labelColor=0d1117)
![Streamlit](https://img.shields.io/badge/Streamlit-In%20Progress-FF4B4B?style=flat-square&logo=streamlit&logoColor=white&labelColor=0d1117)
![Status](https://img.shields.io/badge/Status-Active%20Daily%20Commits-27AE60?style=flat-square&labelColor=0d1117)
![Sprint](https://img.shields.io/badge/Sprint-Day%2011-8957E5?style=flat-square&labelColor=0d1117)
![MVP](https://img.shields.io/badge/MVP%20Launch-April%2025-E74C3C?style=flat-square&labelColor=0d1117)
![License](https://img.shields.io/badge/License-MIT-F39C12?style=flat-square&labelColor=0d1117)

<br/>

> **Semantic search for Netflix — sentence-level chunking, ChromaDB vector store, and Gemini 2.5 Flash generating natural language recommendations. MVP launching April 25.**

</div>

---

## What is CineSense AI

CineSense AI is a production-grade RAG system built on the Netflix titles dataset (8,800+ titles). You describe what you want to watch in plain language. The pipeline encodes your query into a vector, retrieves the most semantically relevant chunks from ChromaDB, and sends them to Gemini 2.5 Flash — which returns ranked recommendations with per-title reasoning.

This repository is a daily-commit build log. Every day, one new capability is added. The final destination is a Chrome Extension that sits inside Netflix itself.

**RAG Foundation:** [Cortex\_RAG](https://github.com/ather-ops/Cortex_RAG) — learn the complete embedding pipeline from one-hot encoding through SentenceTransformers before diving into this repo.

---

## MVP Launch — April 25

The core RAG pipeline (ingestion + retrieval + LLM) is complete. The final two layers before MVP are being added now:

| Layer | Status | ETA |
|---|---|---|
| `ingestion.py` — full pipeline to ChromaDB | Done | — |
| `rag_engine.py` — retrieval + Gemini 2.5 Flash | Done | — |
| EDA visuals — 5 production charts | Done | — |
| `FastAPI` — REST endpoint wrapping the RAG pipeline | In progress | April 24 |
| `app.py` — Streamlit UI | In progress | April 24–25 |
| MVP live | — | **April 25** |

---

## Pipeline Architecture

<div align="center">
  <img src="https://github.com/ather-ops/CineSense-AI/blob/main/02-Assets/architecture.svg?raw=true" alt="CineSense AI Pipeline Architecture" width="72%" />
</div>

---

## Repository Structure

<div align="center">
  <img src="https://github.com/ather-ops/CineSense-AI/blob/main/02-Assets/Repo-Struture.png?raw=true" alt="CineSense AI Repository Structure" width="72%" />
</div>

---

## EDA Visuals

Five production charts generated from the raw Netflix dataset during ingestion. Saved to `04-Visuals/`.

<div align="center">

<table>
<tr>
<td align="center" width="50%">
<img src="https://github.com/ather-ops/CineSense-AI/blob/main/04-Visuals/Content%20_Type%20_%20Movies_%20vs_Tv%20_shows.png?raw=true" width="100%" alt="Content Type Distribution"/>
<br/><sub>Content Type — Movies vs TV Shows</sub>
</td>
<td align="center" width="50%">
<img src="https://github.com/ather-ops/CineSense-AI/blob/main/04-Visuals/Content_Growth_over_the_Year.png?raw=true" width="100%" alt="Content Growth Over the Years"/>
<br/><sub>Content Growth Over the Years</sub>
</td>
</tr>
<tr>
<td align="center" width="50%">
<img src="https://github.com/ather-ops/CineSense-AI/blob/main/04-Visuals/Top_10_Countries.png?raw=true" width="100%" alt="Top 10 Content-Producing Countries"/>
<br/><sub>Top 10 Content-Producing Countries</sub>
</td>
<td align="center" width="50%">
<img src="https://github.com/ather-ops/CineSense-AI/blob/main/04-Visuals/Top_10_genres.png?raw=true" width="100%" alt="Top 10 Genres"/>
<br/><sub>Top 10 Most Popular Genres</sub>
</td>
</tr>
<tr>
<td align="center" colspan="2">
<img src="https://github.com/ather-ops/CineSense-AI/blob/main/04-Visuals/Segment_Distribution.png?raw=true" width="50%" alt="Audience Segment Distribution"/>
<br/><sub>Audience Segment Distribution by Rating</sub>
</td>
</tr>
</table>

</div>

---

## Roadmap and Priorities

Work is executed in strict priority order. The next item on this list is always the only item being worked on.

| Priority | Task | Status |
|---|---|---|
| 1 | `ingestion.py` — full pipeline to ChromaDB, production-ready | Done |
| 2 | `rag_engine.py` — retrieval + Gemini 2.5 Flash LLM layer | Done |
| 3 | EDA visuals — 5 charts saved to `04-Visuals/` | Done |
| 4 | Unit tests — ingestion correctness, metadata schema, retrieval smoke tests | In progress |
| 5 | `FastAPI` — REST endpoint wrapping the full RAG pipeline | In progress |
| 6 | `app.py` — Streamlit UI | In progress |
| 7 | Chrome Extension scaffold — Manifest V3, Netflix DOM injection | Pending |
| 8 | Chrome Extension UI — live semantic search overlay inside Netflix | Pending |

---

## Testing

Tests live in `03-Core/tests/`. Written alongside the pipeline — not after.

**Coverage:**
- `fill_missing()` — year columns use median, categoricals fill with `"unknown"`, no nulls remain
- `sentence_chunk()` — chunks never exceed `max_sentences`, all non-empty, sentences always complete
- ChromaDB insert — chunk count matches expected, metadata fields present and correctly typed, IDs follow `show_id_chunk_N` format
- `retrieve()` — returns results for a known query, compound filters reduce result count vs no filter

```bash
pip install pytest
pytest 03-Core/tests/ -v
```

---

## Build Log — Day by Day

---

### Days 1–3 — Data, Embeddings, Retrieval

**Day 1:** CSV ingestion, null strategy (`fill_missing`), EDA dashboard — five visualizations saved to `04-Visuals/`.

**Day 2:** Combined text field construction, `SentenceTransformer("all-MiniLM-L6-v2")` embeddings (384-dim), ChromaDB collection, batch insert at size 100.

**Day 3:** `retrieve()` with compound `$and` filter support across genre, year range, rating, and type. Four query patterns tested. CSV export.

---

### Days 4–5 — Chunking

**Day 4:** Experimental word-count splitting. Produced incomplete sentences mid-chunk — deliberate proof-of-concept.

**Day 5:** Replaced with `sentence_chunk(text, max_sentences=2)` using NLTK `sent_tokenize`. Full pipeline refactor: type hints, explicit metadata casting, named constants, title-level deduplication in retrieval.

| Aspect | Day 4 | Day 5 |
|---|---|---|
| Split strategy | Fixed word-count | Sentence-boundary detection |
| Sentence integrity | Cuts mid-sentence | Always complete |
| Chunk IDs | `chunk_{i}` | `{show_id}_chunk_{N}` |
| Metadata typing | Implicit | Explicit cast per field |

---

### Day 6 — LLM Layer: Gemini 2.5 Flash

`rag_engine.py` — the complete RAG answer generation layer.

`build_context(results)` — deduplicates retrieved chunks by title, formats as a numbered context block.

`rag_answer(llm, query, results)` — constructs a grounded prompt and sends it to Gemini 2.5 Flash. The model recommends exactly 3 titles from the retrieved set with one-sentence reasoning each. It cannot hallucinate titles that were not retrieved — the context is the only source.

ChromaDB upgraded from `Client()` to `PersistentClient(path="./chroma_data")` — the vector store now survives restarts. `ingestion.py` runs once; `rag_engine.py` loads the persisted store on every subsequent query.

API key loaded from `GEMINI_API_KEY` environment variable — never hardcoded in source.

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
   A mother and son's captivity and their trauma of re-entering
   the world delivers sustained emotional weight throughout.
```

---

### Days 7–11 — EDA Visuals, Refactoring, UI and API

Days 7 through 11 covered production hardening of the ingestion pipeline, EDA chart exports to `04-Visuals/`, code reviews across `ingestion.py` and `rag_engine.py`, and the start of the FastAPI and Streamlit layers.

FastAPI endpoint and Streamlit UI are in active development. MVP launches **April 25**.

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

Query the RAG engine:

```bash
python 03-Core/rag_engine.py
```

Or from Python:

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
| API | FastAPI (in progress) |
| UI | Streamlit (in progress) |
| Testing | pytest |
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

Built with focus. Committed daily. MVP launching April 25.

</div>
