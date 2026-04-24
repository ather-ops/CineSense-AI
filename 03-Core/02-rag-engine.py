# =============================================================================
# CineSense AI — rag_engine.py
# =============================================================================

import os
import chromadb
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# Constants
CHROMA_PATH = "./chroma_data"
COLLECTION = "netflix_titles"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "models/gemini-2.5-flash"
TOP_K = 5

def load_resources():
    """Load embedding model, ChromaDB collection, and Gemini LLM"""
    
    # Get API key from Streamlit Secrets
    try:
        import streamlit as st
        api_key = st.secrets["GEMINI_API_KEY"]
    except:
        # For local testing
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise Exception("GEMINI_API_KEY not found. Add to Streamlit Secrets or .env file")
    
    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL)
    
    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=COLLECTION)
    
    print("Configuring Gemini...")
    genai.configure(api_key=api_key)
    llm = genai.GenerativeModel(LLM_MODEL)
    
    print(f"Ready! Collection has {collection.count():,} chunks")
    return embed_model, collection, llm

def retrieve(collection, embed_model, query, genre=None, min_year=None, max_year=None, rating=None, movie_type=None, top_k=5):
    """Search ChromaDB for relevant content"""
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
    
    where = {"$and": conditions} if len(conditions) > 1 else (conditions[0] if conditions else None)
    
    return collection.query(
        query_embeddings=[query_emb.tolist()],
        where=where,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

def build_context(results):
    """Format search results for LLM"""
    seen = set()
    context = ""
    rank = 1
    
    for meta in results["metadatas"][0]:
        title = meta["title"]
        if title in seen:
            continue
        seen.add(title)
        context += f"{rank}. {title} ({meta['release_year']})\n   Genre: {meta['listed_in']}\n   Rating: {meta['rating']}\n\n"
        rank += 1
    
    return context

def rag_answer(llm, query, results):
    """Generate answer using Gemini"""
    context = build_context(results)
    prompt = f"""You are CineSense AI. User query: '{query}'

Based ONLY on these titles, recommend top 3 matches:

{context}

Respond with friendly recommendations and explain why each fits."""
    
    response = llm.generate_content(prompt)
    return response.text

def cinesense(query, collection, embed_model, llm, genre=None, min_year=None, max_year=None, rating=None, movie_type=None, top_k=5):
    """Main RAG pipeline"""
    results = retrieve(collection, embed_model, query, genre, min_year, max_year, rating, movie_type, top_k)
    answer = rag_answer(llm, query, results)
    return results, answer
