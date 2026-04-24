# =============================================================================
# CineSense AI - Streamlit Chat Interface
# Professional Chat UI with Black Background
# Author: ather-ops
# =============================================================================

import streamlit as st
import os
from dotenv import load_dotenv
from rag_engine import load_resources, cinesense

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="CineSense AI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for black background and professional chat styling
st.markdown("""
<style>
    /* Main container styling */
    .stApp {
        background-color: #000000;
    }
    
    /* Main background */
    .main {
        background-color: #000000;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Chat message containers */
    .chat-message {
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    
    /* User message styling */
    .user-message {
        background-color: #1a1a1a;
        border-left: 3px solid #E50914;
        margin-left: 20%;
    }
    
    /* Assistant message styling */
    .assistant-message {
        background-color: #0a0a0a;
        border-left: 3px solid #4CAF50;
        margin-right: 20%;
    }
    
    /* Message text */
    .message-text {
        color: #ffffff;
        font-size: 1rem;
        line-height: 1.5;
        font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    
    /* Sender label */
    .sender-label {
        font-weight: bold;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
    }
    
    .user-label {
        color: #E50914;
    }
    
    .assistant-label {
        color: #4CAF50;
    }
    
    /* Input area styling */
    .stTextInput > div > div > input {
        background-color: #1a1a1a;
        color: #ffffff;
        border: 1px solid #333333;
        border-radius: 8px;
        padding: 0.75rem;
        font-size: 1rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #E50914;
        box-shadow: none;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #E50914;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        background-color: #b8070f;
        transform: translateY(-2px);
    }
    
    /* Clear button styling */
    .clear-btn > button {
        background-color: #333333;
    }
    
    .clear-btn > button:hover {
        background-color: #444444;
        transform: translateY(-2px);
    }
    
    /* Mood buttons container */
    .mood-buttons-container {
        margin: 1rem 0;
        padding: 0.5rem;
        border-radius: 8px;
        background-color: #0a0a0a;
    }
    
    .mood-label {
        color: #888888;
        font-size: 0.8rem;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    /* Header styling */
    .cine-header {
        text-align: center;
        padding: 1.5rem;
        border-bottom: 1px solid #1a1a1a;
        margin-bottom: 2rem;
    }
    
    .logo-text {
        font-size: 2.5rem;
        font-weight: bold;
        color: #E50914;
        letter-spacing: 2px;
    }
    
    .tagline {
        color: #888888;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    /* Divider */
    .divider {
        border-top: 1px solid #1a1a1a;
        margin: 2rem 0 1rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666666;
        font-size: 0.8rem;
        border-top: 1px solid #1a1a1a;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'resources_loaded' not in st.session_state:
    st.session_state.resources_loaded = False
if 'embed_model' not in st.session_state:
    st.session_state.embed_model = None
if 'collection' not in st.session_state:
    st.session_state.collection = None
if 'llm' not in st.session_state:
    st.session_state.llm = None

# Header with logo
st.markdown("""
<div class="cine-header">
    <div class="logo-text">✦ CineSense AI ✦</div>
    <div class="tagline">Next Level Gen AI Tool by Ather-Ops</div>
</div>
""", unsafe_allow_html=True)

# Load RAG resources
def load_rag_resources():
    """Load embedding model, ChromaDB collection, and LLM"""
    with st.spinner("Initializing CineSense AI..."):
        try:
            embed_model, collection, llm = load_resources()
            st.session_state.embed_model = embed_model
            st.session_state.collection = collection
            st.session_state.llm = llm
            st.session_state.resources_loaded = True
            st.success("CineSense AI is ready!")
            return True
        except Exception as e:
            st.error(f"Failed to load resources: {str(e)}")
            st.info("Make sure GEMINI_API_KEY is set in .env file")
            return False

# Load resources if not already loaded
if not st.session_state.resources_loaded:
    if load_rag_resources():
        st.rerun()

# Mood buttons section
st.markdown('<div class="mood-buttons-container">', unsafe_allow_html=True)
st.markdown('<div class="mood-label">Quick Recommendations</div>', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)

mood_queries = {
    "Emotional": "Recommend something emotional and heartbreaking",
    "Action": "Find me an intense action thriller",
    "Comedy": "Show me funny comedy movies or TV shows",
    "Romance": "I want a romantic love story",
    "Documentary": "Recommend an interesting documentary"
}

with col1:
    if st.button("Emotional", key="mood1", use_container_width=True):
        st.session_state.mood_query = mood_queries["Emotional"]
with col2:
    if st.button("Action", key="mood2", use_container_width=True):
        st.session_state.mood_query = mood_queries["Action"]
with col3:
    if st.button("Comedy", key="mood3", use_container_width=True):
        st.session_state.mood_query = mood_queries["Comedy"]
with col4:
    if st.button("Romance", key="mood4", use_container_width=True):
        st.session_state.mood_query = mood_queries["Romance"]
with col5:
    if st.button("Documentary", key="mood5", use_container_width=True):
        st.session_state.mood_query = mood_queries["Documentary"]

st.markdown('</div>', unsafe_allow_html=True)

# Display chat messages
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <div class="sender-label user-label">You</div>
            <div class="message-text">{message["content"]}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <div class="sender-label assistant-label">CineSense AI</div>
            <div class="message-text">{message["content"]}</div>
        </div>
        """, unsafe_allow_html=True)

# Handle mood button queries
if 'mood_query' in st.session_state:
    query = st.session_state.mood_query
    del st.session_state.mood_query
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Get response
    with st.spinner("CineSense AI is analyzing..."):
        try:
            results, answer = cinesense(
                query=query,
                collection=st.session_state.collection,
                embed_model=st.session_state.embed_model,
                llm=st.session_state.llm,
                top_k=5
            )
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {str(e)}"})
    
    st.rerun()

# Chat input area
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

with st.container():
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_input(
            "Message",
            placeholder="Ask for recommendations... (e.g., 'Recommend something emotional and heartbreaking')",
            key="user_input",
            label_visibility="collapsed"
        )
    
    with col2:
        send_button = st.button("Send", key="send_btn", use_container_width=True)
    
    # Clear chat button
    col_clear1, col_clear2, col_clear3 = st.columns([2, 1, 2])
    with col_clear2:
        if st.button("Clear Chat", key="clear_btn", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# Process user input
if send_button and user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Get response
    with st.spinner("CineSense AI is thinking..."):
        try:
            results, answer = cinesense(
                query=user_input,
                collection=st.session_state.collection,
                embed_model=st.session_state.embed_model,
                llm=st.session_state.llm,
                top_k=5
            )
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {str(e)}"})
    
    st.rerun()

# Footer
st.markdown("""
<div class="footer">
    Powered by Gemini 2.5 Flash | ChromaDB | Sentence Transformers
</div>
""", unsafe_allow_html=True)
