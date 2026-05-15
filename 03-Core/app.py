"""
CineSense AI - Clean Premium Interface
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
/* FULL BLACK BACKGROUND */
html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stBottomBlockContainer"],
section, main {
    background-color: #000000 !important;
}

.block-container {
    max-width: 820px;
    padding: 2.5rem 1rem 5rem;
}

header, footer, #MainMenu, [data-testid="stToolbar"] {
    display: none;
}

/* BIG CLEAN HERO */
.hero {
    text-align: center;
    padding: 40px 20px 36px;
    margin-bottom: 28px;
    position: relative;
}

/* Lighting glow behind title */
.hero::before {
    content: "";
    position: absolute;
    top: 0;
    left: 50%;
    transform: translateX(-50%);
    width: 500px;
    height: 180px;
    background: radial-gradient(circle, rgba(59,130,246,0.15), transparent 70%);
    filter: blur(40px);
    z-index: 0;
}

.hero-badge {
    display: inline-block;
    font-size: 12px;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #ec4899;
    border: 1px solid rgba(236,72,153,0.40);
    background-color: rgba(236,72,153,0.08);
    padding: 8px 18px;
    border-radius: 999px;
    margin-bottom: 20px;
    z-index: 1;
    position: relative;
}

.hero-title {
    font-size: 68px;
    line-height: 1.05;
    font-weight: 900;
    letter-spacing: -2.5px;
    margin: 0;
    background: linear-gradient(135deg, #60a5fa 0%, #ec4899 50%, #a855f7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-family: "Inter", "Helvetica Neue", system-ui, sans-serif;
    font-style: italic;
    position: relative;
    z-index: 1;
    text-shadow: 0 0 80px rgba(59,130,246,0.35);
}

.hero-subtitle {
    color: #8b5cf6;
    font-size: 22px;
    font-style: italic;
    font-weight: 400;
    margin-top: 18px;
    line-height: 1.6;
    z-index: 1;
    position: relative;
    letter-spacing: -0.3px;
}

.hero-subtitle-secondary {
    color: #6b7280;
    font-size: 14px;
    font-style: italic;
    margin-top: 10px;
    z-index: 1;
    position: relative;
}

/* QUICK PROMPTS */
.quick-row {
    margin: 18px 0 30px;
}

.stButton > button {
    background-color: #000000 !important;
    color: #9ca3af !important;
    border: 1px solid rgba(236,72,153,0.25) !important;
    border-radius: 999px !important;
    font-size: 12px !important;
    padding: 10px 14px !important;
    width: 100% !important;
    transition: all 0.22s ease !important;
    font-weight: 500;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #3b82f6, #ec4899) !important;
    color: #ffffff !important;
    border-color: transparent !important;
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(236,72,153,0.40);
}

/* USER MESSAGE - SOLID PINK */
.user-msg {
    display: block;
    width: fit-content;
    max-width: 70%;
    margin: 18px 0 18px auto;
    padding: 14px 22px;
    border-radius: 24px 24px 6px 24px;
    background-color: #ec4899;
    color: #ffffff;
    font-size: 15px;
    line-height: 1.7;
    font-weight: 500;
    box-shadow: 0 10px 30px rgba(236,72,153,0.40);
}

/* AI MESSAGE - CLEAN BLUE PINK GLOW */
.ai-msg {
    display: block;
    width: fit-content;
    max-width: 84%;
    margin: 18px auto 18px 0;
    padding: 18px 24px;
    border-radius: 24px 24px 24px 6px;
    background: linear-gradient(135deg, rgba(59,130,246,0.12), rgba(236,72,153,0.12));
    color: #f3f4f6;
    font-size: 15px;
    line-height: 1.85;
    border: 1px solid rgba(59,130,246,0.35);
    border-right: 1px solid rgba(236,72,153,0.35);
    box-shadow:
        0 0 50px rgba(59,130,246,0.08),
        0 16px 40px rgba(0,0,0,0.50);
    white-space: pre-wrap;
    backdrop-filter: blur(4px);
}

/* EMPTY STATE */
.empty-box {
    text-align: center;
    margin: 50px 0 40px;
    padding: 40px 24px;
    border: 1px solid rgba(236,72,153,0.20);
    border-radius: 32px;
    background-color: rgba(236,72,153,0.04);
}

.empty-title {
    color: #f9fafb;
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 14px;
    font-style: italic;
}

.empty-text {
    color: #6b7280;
    font-size: 14px;
    line-height: 1.8;
    font-style: italic;
}

/* CHAT INPUT */
[data-testid="stChatInputContainer"],
[data-testid="stChatInput"],
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] form {
    background-color: #000000 !important;
}

[data-testid="stChatInput"] textarea {
    background-color: rgba(0,0,0,0.90) !important;
    color: #f9fafb !important;
    border: 1px solid rgba(236,72,153,0.45) !important;
    border-radius: 28px !important;
    caret-color: #ec4899 !important;
    font-size: 15px !important;
    padding: 16px 22px !important;
    font-style: italic;
    box-shadow:
        0 0 0 1px rgba(59,130,246,0.20),
        0 0 60px rgba(59,130,246,0.12);
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #4b5563 !important;
    font-style: italic;
}

/* SEND BUTTON */
[data-testid="stChatInput"] button {
    background: linear-gradient(135deg, #3b82f6, #ec4899) !important;
    color: #ffffff !important;
    border-radius: 999px !important;
    border: none !important;
    box-shadow: 0 0 30px rgba(236,72,153,0.35);
}

/* FOOTER */
.footer {
    text-align: center;
    color: #374151;
    font-size: 10px;
    margin-top: 40px;
    font-family: monospace;
    font-style: italic;
}

/* SCROLLBAR */
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
    except:
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


# --- HEADER ---
st.markdown("""
<div class="hero">
    <div class="hero-badge">Phase 1 Live</div>
    <h1 class="hero-title">CineSense AI</h1>
    <div class="hero-subtitle">Your Netflix recommendation engine</div>
    <div class="hero-subtitle-secondary">Describe your mood. Get what to watch.</div>
</div>
""", unsafe_allow_html=True)

# --- LOAD ---
try:
    embed_model, collection, llm = load_engine()
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# --- SESSION ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- QUICK PROMPTS ---
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

# --- EMPTY STATE ---
if not st.session_state.messages:
    st.markdown("""
    <div class="empty-box">
        <div class="empty-title">What do you want to watch?</div>
        <div class="empty-text">
            Try: emotional drama, thriller, sci-fi, or just a feeling.
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- MESSAGES ---
for msg in st.session_state.messages:
    safe = msg["content"].replace("<", "&lt;").replace(">", "&gt;")

    if msg["role"] == "user":
        st.markdown(f'<div class="user-msg">{safe}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="ai-msg">{safe}</div>', unsafe_allow_html=True)

# --- INPUT ---
typed = st.chat_input("Ask CineSense anything...")

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

# --- CLEAR ---
if st.session_state.messages:
    if st.button("Clear", key="clr"):
        st.session_state.messages = []
        st.rerun()

# --- FOOTER ---
st.markdown('<div class="footer">CineSense AI · ather-ops</div>', unsafe_allow_html=True)
