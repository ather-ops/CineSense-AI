
"""
CineSense AI - Blue Pink Chat Interface
Author: ather-ops
"""

import os
import html
import streamlit as st
import chromadb
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "./chroma_data"
COLLECTION = "netflix_titles"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "models/gemini-2.5-flash"

st.set_page_config(
    page_title="CineSense AI",
    page_icon="C",
    layout="centered"
)

st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(circle at top left, rgba(59,130,246,0.25), transparent 32%),
        radial-gradient(circle at top right, rgba(236,72,153,0.22), transparent 30%),
        linear-gradient(180deg, #050816 0%, #020617 100%);
}

.block-container {
    max-width: 780px;
    padding-top: 1.4rem;
    padding-bottom: 2rem;
}

header, footer, #MainMenu {
    visibility: hidden;
}

.hero {
    text-align: center;
    padding: 28px 12px 20px;
    margin-bottom: 16px;
}

.hero-badge {
    display: inline-block;
    font-size: 11px;
    letter-spacing: .5px;
    color: #bfdbfe;
    border: 1px solid rgba(147,197,253,0.28);
    background: rgba(15,23,42,0.75);
    padding: 6px 14px;
    border-radius: 999px;
    margin-bottom: 14px;
}

.hero-title {
    font-size: 40px;
    line-height: 1.05;
    font-weight: 800;
    letter-spacing: -1.4px;
    margin: 0;
    background: linear-gradient(90deg, #60a5fa, #e879f9, #fb7185);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-family: Inter, system-ui, sans-serif;
}

.hero-subtitle {
    color: #94a3b8;
    font-size: 14px;
    font-style: italic;
    margin-top: 12px;
    line-height: 1.7;
}

.hero-small {
    color: #475569;
    font-size: 11px;
    margin-top: 8px;
}

.quick-row {
    margin: 10px 0 20px;
}

.stButton > button {
    background: rgba(15,23,42,0.78) !important;
    color: #cbd5e1 !important;
    border: 1px solid rgba(148,163,184,0.16) !important;
    border-radius: 999px !important;
    font-size: 12px !important;
    padding: 7px 13px !important;
    width: 100% !important;
    transition: all .18s ease !important;
}

.stButton > button:hover {
    border-color: rgba(236,72,153,0.75) !important;
    color: #f9a8d4 !important;
    background: rgba(30,41,59,0.9) !important;
}

.user-msg {
    width: fit-content;
    max-width: 78%;
    margin: 12px 0 12px auto;
    padding: 12px 16px;
    border-radius: 20px 20px 5px 20px;
    background: linear-gradient(135deg, #2563eb, #db2777);
    color: #ffffff;
    font-size: 14px;
    line-height: 1.6;
    box-shadow: 0 12px 32px rgba(219,39,119,0.16);
}

.ai-msg {
    width: fit-content;
    max-width: 88%;
    margin: 12px auto 12px 0;
    padding: 14px 17px;
    border-radius: 20px 20px 20px 5px;
    background: rgba(15,23,42,0.88);
    color: #dbeafe;
    font-size: 14px;
    line-height: 1.75;
    border: 1px solid rgba(96,165,250,0.16);
    border-left: 3px solid #60a5fa;
    box-shadow: 0 14px 35px rgba(15,23,42,0.34);
    white-space: pre-wrap;
}

.empty-box {
    text-align: center;
    margin: 34px 0 26px;
    padding: 28px 18px;
    border: 1px solid rgba(148,163,184,0.12);
    border-radius: 24px;
    background: rgba(15,23,42,0.45);
}

.empty-title {
    color: #e0f2fe;
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 8px;
}

.empty-text {
    color: #64748b;
    font-size: 13px;
    line-height: 1.7;
}

.footer-note {
    text-align: center;
    color: #334155;
    font-size: 10px;
    margin-top: 28px;
    font-family: monospace;
}

div[data-testid="stChatInput"] {
    border-radius: 999px;
}

textarea {
    background: rgba(15,23,42,0.95) !important;
    color: #e2e8f0 !important;
    border: 1px solid rgba(96,165,250,0.22) !important;
    border-radius: 18px !important;
}

</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading CineSense AI...")
def load_all():
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        api_key = os.getenv("GEMINI_API_KEY", "")

    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in Streamlit Secrets.")

    genai.configure(api_key=api_key)

    embed_model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=COLLECTION)
    llm = genai.GenerativeModel(LLM_MODEL)

    return embed_model, collection, llm


def retrieve_titles(query, collection, embed_model, top_k=5):
    query_emb = embed_model.encode([query])[0]
    results = collection.query(
        query_embeddings=[query_emb.tolist()],
        n_results=top_k,
        include=["metadatas", "distances"]
    )
    return results


def build_context(results):
    seen = set()
    lines = []

    for meta in results["metadatas"][0]:
        title = str(meta.get("title", "Unknown"))
        if title in seen:
            continue

        seen.add(title)
        year = meta.get("release_year", "Unknown")
        genre = meta.get("listed_in", "Unknown")
        rating = meta.get("rating", "Unknown")
        content_type = meta.get("type", "Unknown")

        lines.append(
            f"- {title} ({year})\\n"
            f"  Genre: {genre}\\n"
            f"  Rating: {rating} | Type: {content_type}"
        )

    return "\\n\\n".join(lines)


def generate_answer(query, results, llm):
    context = build_context(results)

    if not context.strip():
        return "I could not find strong matches. Try describing the mood, genre, or era differently."

    prompt = f"""
You are CineSense AI, a movie recommendation assistant.

User query:
{query}

Retrieved Netflix titles:
{context}

Recommend the top 3 most relevant titles.
For each title, write:
1. Title and year
2. One short reason why it matches

Rules:
- Only use titles from the retrieved list.
- Be concise.
- Friendly tone.
"""

    response = llm.generate_content(prompt)
    return response.text


def answer_query(query, collection, embed_model, llm):
    results = retrieve_titles(query, collection, embed_model)
    answer = generate_answer(query, results, llm)
    return answer


st.markdown("""
<div class="hero">
    <div class="hero-badge">Phase 1 Live · RAG Movie Intelligence</div>
    <h1 class="hero-title">CineSense AI</h1>
    <div class="hero-subtitle">
        Tell me your mood, memory, genre, or vibe.<br/>
        CineSense searches Netflix titles by meaning, not keywords.
    </div>
    <div class="hero-small">Powered by SentenceTransformers · ChromaDB · Gemini</div>
</div>
""", unsafe_allow_html=True)

try:
    embed_model, collection, llm = load_all()
except Exception as e:
    st.error(f"Engine error: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

quick_prompts = [
    "something emotional",
    "dark crime thriller",
    "funny late night movie",
    "mind bending sci-fi",
    "old classic movie",
    "feel good family film",
    "hidden gem",
    "short series to binge"
]

st.markdown('<div class="quick-row">', unsafe_allow_html=True)
cols = st.columns(4)
clicked_query = None

for i, prompt in enumerate(quick_prompts):
    with cols[i % 4]:
        if st.button(prompt, key=f"quick_{i}"):
            clicked_query = prompt

st.markdown('</div>', unsafe_allow_html=True)

if not st.session_state.messages:
    st.markdown("""
    <div class="empty-box">
        <div class="empty-title">What are you in the mood to watch?</div>
        <div class="empty-text">
            Try: emotional drama, old thriller, sci-fi with twists,<br/>
            or just describe a feeling.
        </div>
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.messages:
    safe_content = html.escape(msg["content"])

    if msg["role"] == "user":
        st.markdown(f'<div class="user-msg">{safe_content}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="ai-msg">{safe_content}</div>', unsafe_allow_html=True)

typed_query = st.chat_input("Ask CineSense what to watch...")

query = clicked_query or typed_query

if query:
    query = query.strip()

    if query:
        st.session_state.messages.append({"role": "user", "content": query})

        with st.spinner("CineSense is thinking..."):
            try:
                answer = answer_query(query, collection, embed_model, llm)
            except Exception as e:
                answer = f"Error: {str(e)}"

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

if st.session_state.messages:
    if st.button("Clear chat", key="clear_chat"):
        st.session_state.messages = []
        st.rerun()

st.markdown("""
<div class="footer-note">
CineSense AI · blue/pink interface · ather-ops
</div>
""", unsafe_allow_html=True)
