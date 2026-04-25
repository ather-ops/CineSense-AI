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
LLM_MODEL = "gemini-2.0-flash'

st.set_page_config(page_title="CineSense AI", page_icon="C", layout="centered")

st.markdown("""
<style>
.stApp { background: #000000; }
.block-container { max-width: 800px; padding-top: 1rem; }
header, footer, #MainMenu { visibility: hidden; }
.header { text-align: center; padding: 30px 0 20px; }
.header h1 { color: #ffffff; font-size: 28px; font-weight: 700; margin: 0; font-family: sans-serif; }
.header p { color: #888888; font-size: 14px; font-style: italic; margin: 8px 0 0 0; }
.user-msg { background: #1a1a1a; border-radius: 18px 18px 4px 18px; padding: 12px 16px; color: #ffffff; font-size: 14px; max-width: 75%; margin: 8px 0; margin-left: auto; }
.ai-msg { background: #0d0d0d; border-radius: 18px 18px 18px 4px; padding: 12px 16px; color: #dddddd; font-size: 14px; max-width: 85%; margin: 8px 0; border-left: 3px solid #f97316; }
.stTextInput > div > div > input { background: #1a1a1a !important; border: 1px solid #333333 !important; border-radius: 25px !important; color: #ffffff !important; padding: 12px 20px !important; }
.stButton > button { background: #f97316 !important; color: #ffffff !important; border: none !important; border-radius: 25px !important; padding: 10px 30px !important; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header"><h1>CineSense AI</h1><p>Your Netflix recommendation engine powered by Gemini AI</p></div>', unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_ai():
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except:
        api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        st.error("Add GEMINI_API_KEY in Streamlit Secrets")
        st.stop()
    embed_model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=COLLECTION)
    genai.configure(api_key=api_key)
    llm = genai.GenerativeModel(LLM_MODEL)
    return embed_model, collection, llm

embed_model, collection, llm = load_ai()

# Quick prompts - NO AUTOMATIC RERUN
st.markdown('<hr style="border-top: 1px solid #1a1a1a; margin: 20px 0;">', unsafe_allow_html=True)
cols = st.columns(4)
quick = ["Emotional", "Thriller", "Comedy", "Sci-Fi", "Crime", "Family", "Hidden Gem", "Binge"]
for i, q in enumerate(quick):
    with cols[i % 4]:
        if st.button(q, key=f"btn_{i}"):
            st.session_state['user_query'] = q

# Initialize query variable
query = ""

# Check if there's a queued query
if 'user_query' in st.session_state:
    query = st.session_state.pop('user_query')

# Get user input
user_input = st.text_input("", placeholder="Ask me anything about Netflix...", key="main_input", label_visibility="collapsed")

if user_input:
    query = user_input

# Display chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-msg">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="ai-msg">{msg["content"]}</div>', unsafe_allow_html=True)

# Process query only when button is clicked
send = st.button("Send")

if query and send:
    st.session_state.messages.append({"role": "user", "content": query})
    
    with st.spinner("Thinking..."):
        try:
            query_emb = embed_model.encode([query])[0]
            results = collection.query(query_embeddings=[query_emb.tolist()], n_results=5)
            
            seen = set()
            titles = []
            for meta in results["metadatas"][0]:
                title = meta["title"]
                if title not in seen:
                    seen.add(title)
                    titles.append(f"• {title} ({meta['release_year']})")
            
            context = "\n".join(titles)
            prompt = f"Recommend these Netflix titles for: \'{query}\'\n\n{context}\n\nGive brief, friendly response."
            answer = llm.generate_content(prompt).text
        except Exception as e:
            answer = f"Error: {str(e)}"
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
    
    # Clear query to prevent repeat
    query = ""

if st.session_state.messages:
    if st.button("Clear Chat"):
        st.session_state.messages = []
