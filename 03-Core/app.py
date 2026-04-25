"""
CineSense AI - Netflix Recommendation Engine
Author: ather-ops
"""
import streamlit as st
import os
import chromadb
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="CineSense AI", page_icon="C", layout="centered")

# CSS
st.markdown("""
<style>
.stApp{background:#000}
.block-container{max-width:800px;padding-top:1rem}
header,footer,#MainMenu{visibility:hidden}
.header{text-align:center;padding:30px 0 20px}
.header h1{color:#fff;font-size:28px;font-weight:700;margin:0}
.header p{color:#888;font-size:14px;font-style:italic;margin:8px 0 0 0}
.user-msg{background:#1a1a1a;border-radius:18px 18px 4px 18px;padding:12px 16px;color:#fff;max-width:75%;margin:8px 0 8px auto}
.ai-msg{background:#0d0d0d;border-radius:18px 18px 18px 4px;padding:12px 16px;color:#ddd;max-width:85%;margin:8px 0;border-left:3px solid #f97316}
.stTextInput>div>div>input{background:#1a1a1a!important;border:1px solid #333!important;border-radius:25px!important;color:#fff!important;padding:12px 20px!important}
.stButton>button{background:#f97316!important;color:#fff!important;border:none!important;border-radius:25px!important;padding:10px 30px!important}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header"><h1>CineSense AI</h1><p>Your Netflix recommendation engine powered by Gemini AI</p></div>', unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_resources():
    # Get API key
    try:
        api_key = st.secrets["AIzaSyBuiI2-k6zdQTMqdU0QKXL1V0lIluV9Lc4"]
    except:
        api_key = os.getenv("GEMINI_API_KEY", "")

    if not api_key:
        st.error("Add GEMINI_API_KEY in Streamlit Secrets!")
        st.stop()

    # Load models
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="./chroma_data")
    collection = client.get_collection(name="netflix_titles")

    # Configure Gemini with NEW API
    genai.configure(api_key=api_key)

    # List available models and pick the first working one
    available_models = []
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            available_models.append(m.name)

    if not available_models:
        st.error("No models available with this API key!")
        st.stop()

    # Use first available model
    model_name = available_models[0]
    st.success(f"Using model: {model_name}")

    llm = genai.GenerativeModel(model_name)

    return embed_model, collection, llm, model_name

embed_model, collection, llm, model_name = load_resources()

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    role = "user-msg" if msg["role"] == "user" else "ai-msg"
    st.markdown(f'<div class="{role}">{msg["content"]}</div>', unsafe_allow_html=True)

query = st.text_input("", placeholder="Ask anything about Netflix...", key="q", label_visibility="collapsed")

if st.button("Send") and query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("Searching..."):
        try:
            # Search ChromaDB
            query_emb = embed_model.encode([query])[0]
            results = collection.query(query_embeddings=[query_emb.tolist()], n_results=5)

            # Build context
            seen = set()
            titles = []
            for meta in results["metadatas"][0]:
                t = meta["title"]
                if t not in seen:
                    seen.add(t)
                    titles.append(f"• {t} ({meta['release_year']})")

            context = "\n".join(titles)
            prompt = f"Recommend these Netflix titles for: '{query}'\n\n{context}\n\nBe brief."

            # Generate answer
            response = llm.generate_content(prompt)
            answer = response.text

        except Exception as e:
            answer = f"Error: {str(e)}"

        st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()

if st.session_state.messages and st.button("Clear"):
    st.session_state.messages = []
    st.rerun()

st.markdown(f'<div style="text-align:center;margin-top:30px;font-size:10px;color:#444">CineSense AI · Model: {model_name} · ather-ops</div>', unsafe_allow_html=True)
