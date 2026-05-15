"""
CineSense AI - Perfect Blue Pink Dark Interface
Author: ather-ops
"""

import os
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
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
/* FULL BLACK BACKGROUND EVERYWHERE */
html, body, .stApp, 
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stBottomBlockContainer"],
section, main, div {
    background-color: #000000 !important;
}

.block-container {
    max-width: 780px;
    padding: 2rem 1rem 4rem;
    background-color: #000000 !important;
}

header, footer, #MainMenu, [data-testid="stToolbar"] {
    visibility: hidden;
    display: none;
}

/* HERO HEADER */
.hero {
    text-align: center;
    padding: 36px 20px 32px;
    margin-bottom: 24px;
    background-color: #000000;
}

.hero-badge {
    display: inline-block;
    font-size: 16px;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: #ec4899;
    border: 1px solid rgba(236,72,153,0.40);
    background-color: rgba(236,72,153,0.08);
    padding: 7px 18px;
    border-radius: 999px;
    margin-bottom: 18px;
}

.hero-title {
    font-size: 72px;
    line-height: 1.1;
    font-weight: 900;
    letter-spacing: -2px;
    margin: 0;
    background: linear-gradient(90deg, #3b82f6 0%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-family: "Inter", system-ui, sans-serif;
}

.hero-subtitle {
    color: #6b7280;
    font-size: 15px;
    font-style: italic;
    margin-top: 16px;
    line-height: 1.8;
}

/* QUICK PROMPTS */
.quick-row {
    margin: 16px 0 28px;
}

.stButton > button {
    background-color: #000000 !important;
    color: #9ca3af !important;
    border: 1px solid rgba(236,72,153,0.30) !important;
    border-radius: 999px !important;
    font-size: 12px !important;
    padding: 9px 16px !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #3b82f6, #ec4899) !important;
    color: #ffffff !important;
    border-color: transparent !important;
    transform: translateY(-2px);
}

/* USER MESSAGE - PINK */
.user-msg {
    display: block;
    width: fit-content;
    max-width: 72%;
    margin: 16px 0 16px auto;
    padding: 14px 20px;
    border-radius: 24px 24px 6px 24px;
    background-color: #ec4899;
    color: #ffffff;
    font-size: 14.5px;
    line-height: 1.7;
    font-weight: 500;
    box-shadow: 0 8px 24px rgba(236,72,153,0.35);
}

/* AI MESSAGE - BLUE PINK GRADIENT WITH BORDERS */
.ai-msg {
    display: block;
    width: fit-content;
    max-width: 86%;
    margin: 16px auto 16px 0;
    padding: 16px 20px;
    border-radius: 24px 24px 24px 6px;
    background: linear-gradient(135deg, rgba(59,130,246,0.15), rgba(236,72,153,0.15));
    color: #f3f4f6;
    font-size: 14.5px;
    line-height: 1.85;
    border: 1px solid #3b82f6;
    border-right: 1px solid #ec4899;
    box-shadow: 
        0 0 0 1px rgba(59,130,246,0.20),
        0 12px 32px rgba(0,0,0,0.40);
    white-space: pre-wrap;
}

/* EMPTY STATE */
.empty-box {
    text-align: center;
    margin: 48px 0 36px;
    padding: 36px 24px;
    border: 1px solid rgba(236,72,153,0.25);
    border-radius: 32px;
    background-color: rgba(236,72,153,0.05);
}

.empty-title {
    color: #f9fafb;
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 12px;
}

.empty-text {
    color: #6b7280;
    font-size: 14px;
    line-height: 1.8;
}

/* CHAT INPUT - DARK TRANSPARENT BACKGROUND */
[data-testid="stChatInputContainer"],
[data-testid="stChatInput"],
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] form,
[data-testid="stBottomBlockContainer"] {
    background-color: #000000 !important;
    background: #000000 !important;
}

[data-testid="stChatInput"] textarea {
    background-color: rgba(0,0,0,0.85) !important;
    color: #f9fafb !important;
    border: 1px solid rgba(236,72,153,0.50) !important;
    border-radius: 26px !important;
    caret-color: #ec4899 !important;
    font-size: 15px !important;
    padding: 15px 20px !important;
    box-shadow: 
        0 0 0 1px rgba(59,130,246,0.25),
        inset 0 2px 8px rgba(0,0,0,0.40) !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #4b5563 !important;
}

[data-testid="stChatInput"] textarea:focus {
    border: 1px solid #3b82f6 !important;
    box-shadow: 
        0 0 0 2px rgba(59,130,246,0.35),
        inset 0 2px 8px rgba(0,0,0,0.40) !important;
}

/* SEND BUTTON */
[data-testid="stChatInput"] button {
    background: linear-gradient(135deg, #3b82f6, #ec4899) !important;
    color: #ffffff !important;
    border-radius: 999px !important;
    border: none !important;
    box-shadow: 0 4px 16px rgba(236,72,153,0.40) !important;
}

[data-testid="stChatInput"] button:hover {
    background: linear-gradient(135deg, #2563eb, #db2777) !important;
    box-shadow: 0 6px 24px rgba(236,72,153,0.55) !important;
}

/* FOOTER */
.footer {
    text-align: center;
    color: #374151;
    font-size: 10px;
    margin-top: 36px;
    font-family: monospace;
}

/* FORCE BLACK EVERYWHERE */
* {
    scrollbar-color: #ec4899 #000000;
}

::-webkit-scrollbar {
    background: #000000;
}

::-webkit-scrollbar-thumb {
    background: #ec4899;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="🎬 Loading CineSense AI...")
def load_engine():
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        api_key = os.getenv("GEMINI_API_KEY", "")

    if not api_key:
        raise ValueError("GEMINI_API_KEY not found")

    genai.configure(api_key=api_key)

    embed = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    coll = client.get_collection(name=COLLECTION)
    llm = genai.GenerativeModel(LLM_MODEL)

    return embed, coll, llm


def get_answer(query, collection, embed_model, llm):
    q_emb = embed_model.encode([query])[0]

    results = collection.query(
        query_embeddings=[q_emb.tolist()],
        n_results=5,
        include=["metadatas"]
    )

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

        lines.append(f"- {title} ({year})\n  Genre: {genre}\n  Rating: {rating}")

    context = "\n\n".join(lines)

    if not context.strip():
        return "No strong matches found. Try different words."

    prompt = f"""You are CineSense AI.

User query: {query}

Retrieved titles:
{context}

Recommend top 3. For each: title, year, one reason.
Be concise and friendly. Only use retrieved titles."""

    response = llm.generate_content(prompt)
    return response.text


# HEADER
st.markdown("""
<div class="hero">
    <div class="hero-badge">Phase 1 Live</div>
    <h1 class="hero-title">CineSense AI</h1>
    <div class="hero-subtitle">
        Describe your mood. Get Netflix recommendations.<br/>
        Powered by RAG, ChromaDB, Gemini.
    </div>
</div>
""", unsafe_allow_html=True)

# LOAD
try:
    embed_model, collection, llm = load_engine()
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# SESSION
if "messages" not in st.session_state:
    st.session_state.messages = []

# QUICK PROMPTS
prompts = [
    "something emotional",
    "dark thriller",
    "funny movie",
    "mind bending",
    "old classic",
    "feel good",
    "hidden gem",
    "binge series"
]

st.markdown('<div class="quick-row">', unsafe_allow_html=True)
cols = st.columns(4)
clicked = None

for i, p in enumerate(prompts):
    with cols[i % 4]:
        if st.button(p, key=f"q{i}"):
            clicked = p

st.markdown('</div>', unsafe_allow_html=True)

# EMPTY STATE
if not st.session_state.messages:
    st.markdown("""
    <div class="empty-box">
        <div class="empty-title">What do you want to watch?</div>
        <div class="empty-text">
            Try: emotional drama, thriller, sci-fi, or just a feeling.
        </div>
    </div>
    """, unsafe_allow_html=True)

# MESSAGES
for msg in st.session_state.messages:
    safe = msg["content"].replace("<", "&lt;").replace(">", "&gt;")

    if msg["role"] == "user":
        st.markdown(f'<div class="user-msg">{safe}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="ai-msg">{safe}</div>', unsafe_allow_html=True)

# INPUT
typed = st.chat_input("Ask CineSense...")

query = clicked or typed

if query:
    query = query.strip()
    if query:
        st.session_state.messages.append({"role": "user", "content": query})

        with st.spinner("Thinking..."):
            try:
                answer = get_answer(query, collection, embed_model, llm)
            except Exception as e:
                answer = f"Error: {str(e)}"

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

# CLEAR
if st.session_state.messages:
    if st.button("Clear", key="clr"):
        st.session_state.messages = []
        st.rerun()

# FOOTER
st.markdown('<div class="footer">CineSense AI · ather-ops</div>', unsafe_allow_html=True)
