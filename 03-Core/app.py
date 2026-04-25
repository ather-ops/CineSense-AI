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

# ============================================
# PRO CSS STYLES
# ============================================
st.markdown("""
<style>
.stApp { background: #000000; }
.block-container { max-width: 800px; padding-top: 1rem; }
header, footer, #MainMenu { visibility: hidden; }

/* Header */
.header { text-align: center; padding: 30px 0 20px; }
.header h1 { 
    color: #ffffff; 
    font-size: 28px; 
    font-weight: 700;
    margin: 0;
    font-family: sans-serif;
}
.header p { 
    color: #888888; 
    font-size: 14px;
    font-style: italic;  /* Italic subheading */
    margin: 8px 0 0 0;
}

/* Chat messages */
.user-msg { 
    background: #1a1a1a; 
    border-radius: 18px 18px 4px 18px; 
    padding: 12px 16px; 
    color: #ffffff; 
    font-size: 14px;
    max-width: 75%;
    margin: 8px 0;
    margin-left: auto;
    line-height: 1.5;
}
.ai-msg { 
    background: #0d0d0d; 
    border-radius: 18px 18px 18px 4px; 
    padding: 12px 16px; 
    color: #dddddd; 
    font-size: 14px;
    max-width: 85%;
    margin: 8px 0;
    border-left: 3px solid #f97316;
    line-height: 1.6;
}

/* Input box */
.stTextInput > div > div > input { 
    background: #1a1a1a !important; 
    border: 1px solid #333333 !important; 
    border-radius: 25px !important; 
    color: #ffffff !important; 
    padding: 12px 20px !important;
    font-size: 15px !important;
}
.stTextInput > div > div > input::placeholder { 
    color: #555555 !important; 
}

/* Send button */
.stButton > button { 
    background: #f97316 !important; 
    color: #ffffff !important; 
    border: none !important; 
    border-radius: 25px !important; 
    padding: 10px 30px !important;
    font-weight: 600 !important;
}
.stButton > button:hover { 
    background: #ea6c0c !important; 
}

/* Quick prompts */
.quick-btn { 
    background: #1a1a1a !important; 
    border: 1px solid #333333 !important; 
    color: #888888 !important; 
    border-radius: 20px !important;
    font-size: 12px !important;
    padding: 6px 14px !important;
}
.quick-btn:hover { 
    border-color: #f97316 !important; 
    color: #f97316 !important;
}

/* Divider */
.divider { border-top: 1px solid #1a1a1a; margin: 20px 0; }

/* Footer */
.footer { text-align: center; color: #444444; font-size: 11px; margin-top: 30px; }
</style>
""", unsafe_allow_html=True)

# ============================================
# HEADER WITH ITALIC SUBTITLE
# ============================================
st.markdown("""
<div class="header">
    <h1>CineSense AI</h1>
    <p>Your Netflix recommendation engine powered by Gemini AI</p>
</div>
""", unsafe_allow_html=True)

# ============================================
# LOAD AI RESOURCES
# ============================================
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

# ============================================
# QUICK PROMPTS
# ============================================
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

quick_prompts = ["Emotional", "Thriller", "Comedy", "Sci-Fi", "Crime", "Family", "Hidden Gem", "Binge"]
cols = st.columns(len(quick_prompts))
for i, prompt in enumerate(quick_prompts):
    with cols[i]:
        if st.button(prompt, key=f"qp{i}", help=f"Search for {prompt} movies"):
            st.session_state['q'] = prompt
            st.rerun()

# ============================================
# CHAT HISTORY
# ============================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-msg">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="ai-msg">{msg["content"]}</div>', unsafe_allow_html=True)

# ============================================
# INPUT SECTION
# ============================================
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

col1, col2 = st.columns([5, 1])
with col1:
    query = st.text_input("", placeholder="Ask me anything about Netflix...", 
                         key="q", label_visibility="collapsed")
with col2:
    send = st.button("Send")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    
    with st.spinner("Thinking..."):
        query_emb = embed_model.encode([query])[0]
        results = collection.query(query_embeddings=[query_emb.tolist()], n_results=5)
        
        # Build recommendations
        seen = set()
        titles = []
        for meta in results["metadatas"][0]:
            title = meta["title"]
            if title not in seen:
                seen.add(title)
                titles.append(f"• {title} ({meta['release_year']}) - {meta['listed_in']}")
        
        context = "\n".join(titles)
        prompt = f"As CineSense AI, recommend these Netflix titles for: \'{query}\'\n\n{context}\n\nGive a brief, friendly response."
        
        try:
            answer = llm.generate_content(prompt).text
        except Exception as e:
            answer = f"Error: {str(e)}"
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
    
    st.rerun()

# ============================================
# CLEAR CHAT
# ============================================
if st.session_state.messages:
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ============================================
# FOOTER
# ============================================
st.markdown("""
<div class="footer">
    CineSense AI · RAG + ChromaDB + Gemini · ather-ops
</div>
""", unsafe_allow_html=True)
