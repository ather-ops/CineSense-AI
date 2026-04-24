# =============================================================================
# CineSense AI — rag_engine.py
# RAG retrieval layer: ChromaDB semantic search + Gemini 2.5 Flash LLM answers
# Author: ather-ops
# =============================================================================

# ── Step 1: Imports ──────────────────────────────────────────────────────────
import os
import sys
import chromadb
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# Add parent directory to path for local imports if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Step 2: Get API key from Streamlit Secrets or .env file ─────────────────
def get_api_key():
    """Get API key from Streamlit Secrets (cloud) or .env file (local)"""
    
    # Try Streamlit Secrets first (for cloud deployment)
    try:
        import streamlit as st
        api_key = st.secrets["GEMINI_API_KEY"]
        print("API key loaded from Streamlit Secrets")
        return api_key
    except:
        pass
    
    # Try .env.txt file (for local development)
    try:
        from dotenv import load_dotenv
        load_dotenv('.env.txt')
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            print("API key loaded from .env.txt file")
            return api_key
    except:
        pass
    
    # If no API key found
    raise EnvironmentError(
        "\n" + "="*60 + "\n"
        "GEMINI_API_KEY not found!\n\n"
        "For Streamlit Cloud Deployment:\n"
        "  Go to Settings -> Secrets and add:\n"
        "  GEMINI_API_KEY = 'your_api_key_here'\n\n"
        "For Local Development:\n"
        "  Create a .env.txt file with:\n"
        "  GEMINI_API_KEY=your_api_key_here\n"
        "="*60
    )

# ── Constants ─────────────────────────────────────────────────────────────────
# Get the root directory (where app.py is located)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(ROOT_DIR, "chroma_data")
COLLECTION = "netflix_titles"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "models/gemini-2.5-flash"
TOP_K = 5

# ── Step 3: Load models and vector store ──────────────────────────────────────
def load_resources():
    """
    Load the embedding model, connect to the persisted ChromaDB collection,
    and configure the Gemini LLM. Returns (embed_model, collection, llm).
    """
    # Get API key
    api_key = get_api_key()

    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL)

    print(f"Connecting to ChromaDB at: {CHROMA_PATH}")
    os.makedirs(CHROMA_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    try:
        collection = client.get_collection(name=COLLECTION)
        print(f"Collection '{COLLECTION}' found with {collection.count():,} chunks")
    except:
        raise EnvironmentError(
            f"Collection '{COLLECTION}' not found in ChromaDB.\n"
            "Please run ingestion.py first to create the vector store."
        )

    print("Configuring Gemini...")
    genai.configure(api_key=api_key)
    llm = genai.GenerativeModel(LLM_MODEL)

    print("All resources loaded successfully!")
    return embed_model, collection, llm


# ── Step 4: Semantic retrieval ────────────────────────────────────────────────
def retrieve(
    collection,
    embed_model,
    query: str,
    genre:      str | None = None,
    min_year:   int | None = None,
    max_year:   int | None = None,
    rating:     str | None = None,
    movie_type: str | None = None,
    top_k:      int        = TOP_K,
) -> dict:
    """
    Query ChromaDB with the embedded query vector and optional metadata filters.
    Supports compound $and filters across genre, year range, rating, and type.
    """
    query_emb = embed_model.encode([query])[0]
    conditions = []

    if genre:
        conditions.append({"listed_in": {"$contains": genre}})
    if min_year:
        conditions.append({"release_year": {"$gte": min_year}})
    if max_year:
        conditions.append({"release_year": {"$lte": max_year}})
    if rating:
        conditions.append({"rating": {"$eq": rating}})
    if movie_type:
        conditions.append({"type": {"$eq": movie_type}})

    where = (
        {"$and": conditions} if len(conditions) > 1
        else conditions[0] if len(conditions) == 1
        else None
    )

    return collection.query(
        query_embeddings=[query_emb.tolist()],
        where=where,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )


# ── Step 5: Build LLM context ─────────────────────────────────────────────────
def build_context(results: dict) -> str:
    """
    Deduplicate retrieved chunks by title and format them as a
    numbered context block for the LLM prompt.
    """
    seen = set()
    context = ""
    rank = 1

    for meta in results["metadatas"][0]:
        title = meta["title"]
        if title in seen:
            continue
        seen.add(title)
        context += (
            f"{rank}. {title} ({meta['release_year']})\n"
            f"   Genre  : {meta['listed_in']}\n"
            f"   Rating : {meta['rating']} | Type: {meta['type']}\n\n"
        )
        rank += 1

    return context


# ── Step 6: RAG answer generation ────────────────────────────────────────────
def rag_answer(llm, query: str, results: dict) -> str:
    """
    Build a prompt from the retrieved context and send it to Gemini 2.5 Flash.
    Returns the model's recommendation text.
    """
    context = build_context(results)

    prompt = (
        f"You are CineSense AI, a Netflix recommendation assistant powered by Gemini 2.5 Flash.\n\n"
        f"A user is looking for: '{query}'\n\n"
        f"Based only on the titles below, recommend the top 3 most relevant matches.\n"
        f"For each recommendation explain in one sentence why it fits the query.\n\n"
        f"RETRIEVED TITLES:\n{context}"
    )

    response = llm.generate_content(prompt)
    return response.text


# ── Step 7: Main search interface ────────────────────────────────────────────
def cinesense(
    query: str,
    collection=None,
    embed_model=None,
    llm=None,
    genre: str | None = None,
    min_year: int | None = None,
    max_year: int | None = None,
    rating: str | None = None,
    movie_type: str | None = None,
    top_k: int = TOP_K,
) -> tuple[dict, str]:
    """
    Full RAG pipeline: retrieve from ChromaDB → build context → generate answer.
    Returns (chroma_results, llm_answer).
    """
    results = retrieve(
        collection, embed_model, query,
        genre=genre, min_year=min_year, max_year=max_year,
        rating=rating, movie_type=movie_type, top_k=top_k,
    )
    answer = rag_answer(llm, query, results)

    print("\n" + "=" * 60)
    print(f"  Query: {query}")
    print("=" * 60)
    print(answer)

    return results, answer


# ── Entry point (for testing) ───────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Testing CineSense AI Engine")
    print("=" * 60)
    
    embed_model, collection, llm = load_resources()

    # Test queries
    cinesense("something emotional and heartbreaking",
              collection=collection, embed_model=embed_model, llm=llm)

    cinesense("a gripping crime thriller set in the 2000s",
              collection=collection, embed_model=embed_model, llm=llm,
              genre="Crime", min_year=2000, max_year=2010)

    cinesense("feel-good family movie for the weekend",
              collection=collection, embed_model=embed_model, llm=llm,
              movie_type="Movie", rating="PG")
