# =============================================================================
# CineSense AI — rag_engine.py
# Author: ather-ops | github.com/ather-ops/CineSense-AI
# =============================================================================
import os
import chromadb
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# ── Constants ─────────────────────────────────────────────────────────────────
CHROMA_PATH = "./chroma_data"
COLLECTION  = "netflix_titles"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL   = "models/gemini-2.5-flash-preview-04-17"
TOP_K       = 5


def load_resources():
    """
    Load everything needed for CineSense AI.
    API key is read from Streamlit Secrets first, then env variable.
    Returns: embed_model, collection, llm
    """
    # ── API key: Streamlit Secrets takes priority ──────────────────────────
    try:
        import streamlit as st
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        api_key = os.getenv("GEMINI_API_KEY", "")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found.\n"
            "Streamlit Cloud: App Settings > Secrets > add GEMINI_API_KEY\n"
            "Local: export GEMINI_API_KEY=your_key in terminal"
        )

    # ── Embedding model (runs locally, no API needed) ─────────────────────
    embed_model = SentenceTransformer(EMBED_MODEL)

    # ── ChromaDB (reads from disk — must run notebook first) ──────────────
    client     = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=COLLECTION)

    # ── Gemini LLM ────────────────────────────────────────────────────────
    genai.configure(api_key=api_key)
    llm = genai.GenerativeModel(LLM_MODEL)

    print(f"CineSense loaded — {collection.count():,} chunks in ChromaDB")
    return embed_model, collection, llm


def retrieve(collection, embed_model, query,
             genre=None, min_year=None, max_year=None,
             rating=None, movie_type=None, top_k=TOP_K):
    """
    Semantic search in ChromaDB with optional metadata filters.
    Returns raw ChromaDB results dict.
    """
    query_emb  = embed_model.encode([query])[0]
    conditions = []

    if genre:       conditions.append({"listed_in":    {"$contains": genre}})
    if min_year:    conditions.append({"release_year": {"$gte": min_year}})
    if max_year:    conditions.append({"release_year": {"$lte": max_year}})
    if rating:      conditions.append({"rating":       {"$eq": rating}})
    if movie_type:  conditions.append({"type":         {"$eq": movie_type}})

    where = (
        {"$and": conditions} if len(conditions) > 1
        else conditions[0]   if len(conditions) == 1
        else None
    )

    return collection.query(
        query_embeddings=[query_emb.tolist()],
        where=where,
        n_results=top_k,
        include=["metadatas", "distances"],
    )


def build_context(results):
    """Format ChromaDB results into clean text for the LLM prompt."""
    seen, context, rank = set(), "", 1
    for meta in results["metadatas"][0]:
        title = meta["title"]
        if title in seen:
            continue
        seen.add(title)
        context += (
            f"{rank}. {title} ({meta['release_year']})\n"
            f"   Genre: {meta['listed_in']}\n"
            f"   Rating: {meta['rating']} | Type: {meta['type']}\n\n"
        )
        rank += 1
    return context


def rag_answer(llm, query, results):
    """Send retrieved context + query to Gemini and return the answer."""
    context = build_context(results)
    if not context.strip():
        return "No matching titles found. Try a different query or remove filters."

    prompt = (
        f"You are CineSense AI, a Netflix recommendation assistant.\n\n"
        f"User wants: '{query}'\n\n"
        f"Recommend the top 3 most relevant titles from the list below.\n"
        f"For each title write one sentence explaining why it fits.\n"
        f"Be friendly and concise. Only use titles from the list.\n\n"
        f"TITLES:\n{context}"
    )
    return llm.generate_content(prompt).text


def cinesense(query, collection, embed_model, llm,
              genre=None, min_year=None, max_year=None,
              rating=None, movie_type=None, top_k=TOP_K):
    """Full RAG pipeline: Retrieve -> Build context -> Generate answer."""
    results = retrieve(
        collection, embed_model, query,
        genre=genre, min_year=min_year, max_year=max_year,
        rating=rating, movie_type=movie_type, top_k=top_k,
    )
    answer = rag_answer(llm, query, results)
    return results, answer
