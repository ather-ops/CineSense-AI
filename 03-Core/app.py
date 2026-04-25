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
def load_all():
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except:
        api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        st.error("Add API key in Secrets!")
        st.stop()
    
    genai.configure(api_key=api_key)
    embed = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="./chroma_data")
    coll = client.get_collection(name="netflix_titles")
    llm = genai.GenerativeModel("models/gemini-2.5-flash")
    return embed, coll, llm

embed_model, collection, llm = load_all()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    role = "user-msg" if msg["role"] == "user" else "ai-msg"
    st.markdown(f'<div class="{role}">{msg["content"]}</div>', unsafe_allow_html=True)

query = st.text_input("", placeholder="Ask about Netflix...", key="q", label_visibility="collapsed")

if st.button("Send") and query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.spinner("Thinking..."):
        try:
            q_emb = embed_model.encode([query])[0]
            res = collection.query(query_embeddings=[q_emb.tolist()], n_results=5)
            seen = set()
            titles = []
            for m in res["metadatas"][0]:
                if m["title"] not in seen:
                    seen.add(m["title"])
                    titles.append(f"• {m['title']} ({m['release_year']})")
            ctx = "\n".join(titles)
            p = f"Recommend for: '{query}'\n\n{ctx}\n\nBe brief."
            ans = llm.generate_content(p).text
        except Exception as e:
            ans = f"Error: {str(e)}"
        st.session_state.messages.append({"role": "assistant", "content": ans})
    st.rerun()

if st.session_state.messages and st.button("Clear"):
    st.session_state.messages = []
    st.rerun()

st.markdown('<div style="text-align:center;margin-top:30px;font-size:10px;color:#444">CineSense AI · models/gemini-2.5-flash · ather-ops</div>', unsafe_allow_html=True)
