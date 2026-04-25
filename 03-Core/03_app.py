"""
CineSense AI - Streamlit Chat Interface
Author: ather-ops | github.com/ather-ops/CineSense-AI
"""
import importlib.util
from pathlib import Path

RAG_PATH = Path(__file__).resolve().parent / "02_rag_engine.py"

spec = importlib.util.spec_from_file_location("rag_engine", RAG_PATH)
rag_engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rag_engine)

load_resources = rag_engine.load_resources
cinesense = rag_engine.cinesense
import os
import sys
import subprocess

# AUTO-BUILD DATABASE IF MISSING
# This checks if chroma_data exists. If not, it runs pipeline.py automatically on the server.
if not os.path.exists("../chroma_data"):
    print("First-time setup: Building vector database...")
    print("Please wait, this takes 2-3 minutes on the first launch.")
    try:
        # Run the pipeline script located in the same folder
        subprocess.run([sys.executable, "01_pipeline.py"], check=True)
        print("Database created successfully!")
    except Exception as e:
        print(f"Error creating database: {e}")

st.set_page_config(
    page_title="CineSense AI",
    page_icon="C",
    layout="centered"
)

st.markdown("""
<style>
.stApp {
    background: #000000;
}
.block-container {
    max-width: 700px;
    padding-top: 2rem;
}
header, footer, #MainMenu {
    visibility: hidden;
}
.top-header {
    display: flex;
    align-items: center;
    gap: 15px;
    padding-bottom: 15px;
    border-bottom: 1px solid #1a1a1a;
    margin-bottom: 20px;
}
.top-header img {
    width: 40px;
    height: 40px;
    border-radius: 8px;
    border: 1px solid #222;
}
.top-title {
    font-size: 18px;
    font-weight: 700;
    color: #ffffff;
    font-family: sans-serif;
}
.top-subtitle {
    font-size: 11px;
    color: #666;
    font-family: sans-serif;
}
.top-badge {
    margin-left: auto;
    font-size: 10px;
    color: #666;
    border: 1px solid #222;
    border-radius: 20px;
    padding: 4px 12px;
    font-family: monospace;
}
.user-message {
    background: #0f0f0f;
    border: 1px solid #1f1f1f;
    border-radius: 12px 12px 4px 12px;
    padding: 10px 14px;
    color: #e0e0e0;
    font-size: 14px;
    line-height: 1.6;
    max-width: 80%;
    margin: 0 0 12px auto;
    font-family: sans-serif;
}
.assistant-message {
    background: #0a0a0a;
    border: 1px solid #1a1a1a;
    border-left: 2px solid #f97316;
    border-radius: 4px 12px 12px 12px;
    padding: 12px 14px;
    color: #d0d0d0;
    font-size: 14px;
    line-height: 1.7;
    white-space: pre-wrap;
    max-width: 90%;
    margin: 0 auto 12px 0;
    font-family: sans-serif;
}
.message-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 4px;
    font-family: sans-serif;
}
.label-user {
    color: #444;
    text-align: right;
}
.label-assistant {
    color: #f97316;
}
.stButton>button {
    background: #0a0a0a !important;
    border: 1px solid #1f1f1f !important;
    border-radius: 18px !important;
    color: #666 !important;
    font-size: 11px !important;
    padding: 5px 12px !important;
    width: 100% !important;
    font-family: sans-serif !important;
}
.stButton>button:hover {
    border-color: #f97316 !important;
    color: #f97316 !important;
}
div[data-testid="column"]:last-child .stButton>button {
    background: #f97316 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    padding: 10px 20px !important;
}
div[data-testid="column"]:last-child .stButton>button:hover {
    background: #e8650f !important;
}
.stTextInput>div>div>input {
    background: #0a0a0a !important;
    border: 1px solid #1f1f1f !important;
    border-radius: 10px !important;
    color: #e0e0e0 !important;
    font-size: 14px !important;
    caret-color: #f97316 !important;
    padding: 10px 14px !important;
}
.stTextInput>div>div>input::placeholder {
    color: #333 !important;
}
label {
    color: #444 !important;
    font-size: 11px !important;
}
</style>
""", unsafe_allow_html=True)

LOGO_URL = "https://raw.githubusercontent.com/ather-ops/CineSense-AI/main/Assets/logo.jpg"

st.markdown(f"""
<div class="top-header">
  <img src="{LOGO_URL}" alt="CineSense AI Logo"/>
  <div>
    <div class="top-title">CineSense AI</div>
    <div class="top-subtitle">Netflix Intelligence - Gemini 2.0 + RAG</div>
  </div>
  <div class="top-badge">8,800 titles</div>
</div>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading CineSense AI engine...")
def get_engine():
    return load_resources()

try:
    embed_model, collection, llm = get_engine()
except Exception as e:
    st.error(f"Engine initialization failed: {e}")
    st.info("Make sure you have run pipeline.py and added GEMINI_API_KEY to Streamlit Secrets")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "prefill" not in st.session_state:
    st.session_state.prefill = ""

if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center;padding:50px 0 30px;font-family:sans-serif">
      <div style="font-size:22px;font-weight:700;margin-bottom:8px;color:#1a1a1a">
        What are you in the mood to watch?</div>
      <div style="font-size:13px;color:#333;line-height:1.7">
        Ask in plain English - mood, genre, era, or feeling.<br>
        CineSense searches 8,800 Netflix titles semantically.
      </div>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="message-label label-user">You</div>'
                f'<div class="user-message">{msg["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="message-label label-assistant">CineSense AI</div>'
                f'<div class="assistant-message">{msg["content"]}</div>',
                unsafe_allow_html=True
            )

QUICK_PROMPTS = [
    "Something emotional",
    "Thriller keep me awake",
    "Funny late night show",
    "Mind-bending sci-fi",
    "Feel-good family film",
    "Dark crime drama",
    "Hidden gem",
    "Short series to binge"
]

cols = st.columns(4)
for idx, prompt in enumerate(QUICK_PROMPTS):
    with cols[idx % 4]:
        if st.button(prompt, key=f"prompt_{idx}"):
            st.session_state.prefill = prompt
            st.rerun()

st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

input_col, button_col = st.columns([5, 1])
with input_col:
    user_query = st.text_input(
        "",
        placeholder="Ask anything about Netflix...",
        value=st.session_state.prefill,
        key="query_input",
        label_visibility="collapsed"
    )
with button_col:
    send_button = st.button("Send", key="send_btn")

with st.expander("Advanced Filters"):
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        content_type = st.selectbox("Type", ["All", "Movie", "TV Show"], key="filter_type")
    with filter_col2:
        content_rating = st.selectbox("Rating", ["All", "G", "PG", "PG-13", "R", "TV-14", "TV-MA"], key="filter_rating")
    with filter_col3:
        genre_keyword = st.text_input("Genre", placeholder="e.g. Crime", key="filter_genre")
    
    year_range = st.slider("Release Year", 1990, 2024, (2000, 2024), key="filter_years")

def get_active_filters():
    filters = {}
    if content_type != "All":
        filters["movie_type"] = content_type
    if content_rating != "All":
        filters["rating"] = content_rating
    if genre_keyword.strip():
        filters["genre"] = genre_keyword.strip()
    
    year_min, year_max = year_range
    if year_min > 1990:
        filters["min_year"] = year_min
    if year_max < 2024:
        filters["max_year"] = year_max
    
    return filters

query = (user_query or "").strip()
if (send_button or st.session_state.prefill) and query:
    st.session_state.prefill = ""
    st.session_state.messages.append({"role": "user", "content": query})
    
    with st.spinner("Searching Netflix catalog..."):
        try:
            active_filters = get_active_filters()
            results, answer = cinesense(
                query,
                collection,
                embed_model,
                llm,
                **active_filters
            )
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Error processing request: {str(e)}"
            })
    
    st.rerun()

if st.session_state.messages:
    st.markdown('<div style="height:5px"></div>', unsafe_allow_html=True)
    if st.button("Clear conversation", key="clear_btn"):
        st.session_state.messages = []
        st.rerun()

st.markdown("""
<div style="text-align:center;margin-top:30px;font-size:10px;color:#1a1a1a;font-family:sans-serif">
CineSense AI · RAG + ChromaDB + Gemini 2.0 · ather-ops
</div>
""", unsafe_allow_html=True)
