"""
CineSense AI - Streamlit Chat Interface
Author: ather-ops | github.com/ather-ops/CineSense-AI
"""
import streamlit as st
import os
import chromadb
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "./chroma_data"
COLLECTION = "netflix_titles"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gemini-1.5-flash"

st.set_page_config(page_title="CineSense AI", page_icon="C", layout="centered")

st.markdown("""
<style>
.stApp{background:#000}.block-container{max-width:700px;padding-top:2rem}
header,footer,#MainMenu{visibility:hidden}
.user-message{background:#0f0f0f;border:1px solid #333;border-radius:12px;padding:10px;color:#fff;max-width:80%;margin-left:auto;margin-bottom:10px}
.assistant-message{background:#0a0a0a;border-left:3px solid #f97316;border-radius:12px;padding:10px;color:#fff;max-width:90%;margin-bottom:10px}
.stButton>button{background:#333!important;color:#fff!important;border-radius:10px!important}
.stTextInput>div>div>input{background:#0a0a0a!important;color:#fff!important;border:1px solid #333!important}
</style>
""", unsafe_allow_html=True)

st.title("CineSense AI")
st.markdown("Netflix Intelligence - Gemini 1.5 + RAG")

@st.cache_resource(show_spinner="Loading AI...")
def load_all():
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except:
        api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("Add GEMINI_API_KEY in Streamlit Secrets!")
    embed_model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=COLLECTION)
    genai.configure(api_key=api_key)
    llm = genai.GenerativeModel(LLM_MODEL)
    return embed_model, collection, llm

try:
    embed_model, collection, llm = load_all()
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

def build_context(results):
    context = ""
    seen = set()
    for meta in results["metadatas"][0]:
        title = meta["title"]
        if title not in seen:
            seen.add(title)
            context += f"- {title} ({meta['release_year']})\n  Genre: {meta['listed_in']}\n\n"
    return context

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    role = "user-message" if msg["role"] == "user" else "assistant-message"
    st.markdown(f'<div class="{role}">{msg["content"]}</div>', unsafe_allow_html=True)

query = st.text_input("Ask about Netflix...", key="q")

if st.button("Send") and query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.spinner("Searching..."):
        query_emb = embed_model.encode([query])[0]
        results = collection.query(query_embeddings=[query_emb.tolist()], n_results=5)
        context = build_context(results)
        prompt = f"You are CineSense AI. Recommend top 3 titles for: \'{query}\'\n\n{context}"
        answer = llm.generate_content(prompt).text
        st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()

if st.session_state.messages:
    if st.button("Clear"):
        st.session_state.messages = []
        st.rerun()
