"""
"""
CineSense AI — Streamlit Chat UI
Author: ather-ops | github.com/ather-ops/CineSense-AI
Run: streamlit run app.py
"""
import streamlit as st
from rag_engine import load_resources, cinesense

st.set_page_config(page_title="CineSense AI", page_icon="C", layout="centered")

st.markdown("""
<style>
.stApp{background:#000}
.block-container{max-width:680px;padding-top:1.5rem}
header,footer,#MainMenu{visibility:hidden}
.top{display:flex;align-items:center;gap:12px;padding-bottom:12px;
     border-bottom:1px solid #1a1a1a;margin-bottom:18px}
.top img{width:38px;height:38px;border-radius:9px;object-fit:cover;
         border:1px solid #1e1e1e}
.top-title{font-size:17px;font-weight:700;color:#fff;
           font-family:sans-serif;letter-spacing:-.3px}
.top-sub{font-size:11px;color:#333;font-family:sans-serif}
.top-chip{margin-left:auto;font-size:10px;color:#333;
          border:1px solid #1e1e1e;border-radius:20px;
          padding:3px 10px;font-family:monospace}
.umsg{background:#0e0e0e;border:1px solid #1e1e1e;
      border-radius:14px 14px 3px 14px;padding:10px 14px;
      color:#e2e8f0;font-size:13.5px;line-height:1.55;
      max-width:80%;margin:0 0 10px auto;font-family:sans-serif}
.amsg{background:#080808;border:1px solid #161616;
      border-left:2px solid #f97316;
      border-radius:3px 14px 14px 14px;padding:12px 14px;
      color:#d1d5db;font-size:13.5px;line-height:1.7;
      white-space:pre-wrap;max-width:92%;
      margin:0 auto 10px 0;font-family:sans-serif}
.lbl{font-size:10px;font-weight:600;letter-spacing:.8px;
     text-transform:uppercase;margin-bottom:3px;font-family:sans-serif}
.lbl-u{color:#333;text-align:right}
.lbl-a{color:#f97316}
.stButton>button{background:#090909 !important;
    border:1px solid #1a1a1a !important;border-radius:20px !important;
    color:#555 !important;font-size:11px !important;
    padding:4px 11px !important;width:100% !important;
    font-family:sans-serif !important}
.stButton>button:hover{border-color:#f97316 !important;color:#f97316 !important}
div[data-testid="column"]:last-child .stButton>button{
    background:#f97316 !important;color:#fff !important;
    border:none !important;border-radius:10px !important;
    font-weight:700 !important;font-size:13px !important;padding:9px 20px !important}
div[data-testid="column"]:last-child .stButton>button:hover{
    background:#ea6c0c !important;color:#fff !important}
.stTextInput>div>div>input{background:#0a0a0a !important;
    border:1px solid #1e1e1e !important;border-radius:10px !important;
    color:#e2e8f0 !important;font-size:14px !important;
    caret-color:#f97316 !important;padding:9px 13px !important}
.stTextInput>div>div>input::placeholder{color:#2a2a2a !important}
label{color:#333 !important;font-size:11px !important}
</style>
""", unsafe_allow_html=True)

LOGO = ("https://raw.githubusercontent.com/ather-ops/CineSense-AI"
        "/main/02-Assets/30%20Best%20Star%20Logo%20Design%20Ideas%20You%20Should%20Check.jpg")

st.markdown(f"""
<div class="top">
  <img src="{LOGO}" alt="logo"/>
  <div>
    <div class="top-title">CineSense AI</div>
    <div class="top-sub">Netflix Intelligence — Gemini 2.5 + RAG</div>
  </div>
  <div class="top-chip">8,800 titles</div>
</div>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading CineSense AI...")
def get_engine():
    return load_resources()

try:
    embed_model, collection, llm = get_engine()
except Exception as e:
    st.error(f"Engine error: {e}")
    st.stop()

if "msgs" not in st.session_state: st.session_state.msgs = []
if "fill" not in st.session_state: st.session_state.fill = ""

if not st.session_state.msgs:
    st.markdown("""
    <div style="text-align:center;padding:44px 0 28px;font-family:sans-serif">
      <div style="font-size:20px;font-weight:700;margin-bottom:6px;color:#1a1a1a">
        What are you in the mood to watch?</div>
      <div style="font-size:12.5px;color:#222;line-height:1.65">
        Ask in plain English — mood, genre, era, feeling.<br>
        CineSense searches 8,800 Netflix titles semantically.</div>
    </div>""", unsafe_allow_html=True)
else:
    for m in st.session_state.msgs:
        if m["role"] == "user":
            st.markdown(f'<div class="lbl lbl-u">You</div><div class="umsg">{m["content"]}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="lbl lbl-a">CineSense AI</div><div class="amsg">{m["content"]}</div>',
                        unsafe_allow_html=True)

MOODS = ["Something emotional", "Thriller keep me awake",
         "Funny late night show", "Mind-bending sci-fi",
         "Feel-good family film", "Dark crime drama",
         "Hidden gem", "Short series to binge"]
cols = st.columns(4)
for i, mood in enumerate(MOODS):
    with cols[i % 4]:
        if st.button(mood, key=f"m{i}"):
            st.session_state.fill = mood
            st.rerun()

st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)
ic, bc = st.columns([5, 1])
with ic:
    query = st.text_input("", placeholder="Ask anything about Netflix...",
                          value=st.session_state.fill,
                          key="qinput", label_visibility="collapsed")
with bc:
    send = st.button("Send", key="sbtn")

with st.expander("Filters (optional)"):
    c1, c2, c3 = st.columns(3)
    with c1: st.selectbox("Type",   ["All","Movie","TV Show"], key="ft")
    with c2: st.selectbox("Rating", ["All","G","PG","PG-13","R","TV-14","TV-MA"], key="fr")
    with c3: st.text_input("Genre keyword", placeholder="e.g. Crime", key="fg")
    st.slider("Year range", 1990, 2024, (2000, 2024), key="fy")

def get_filters():
    kw = {}
    if st.session_state.ft != "All":    kw["movie_type"] = st.session_state.ft
    if st.session_state.fr != "All":    kw["rating"]     = st.session_state.fr
    if st.session_state.fg.strip():     kw["genre"]      = st.session_state.fg.strip()
    y1, y2 = st.session_state.fy
    if y1 > 1990: kw["min_year"] = y1
    if y2 < 2024: kw["max_year"] = y2
    return kw

q = (query or "").strip()
if (send or st.session_state.fill) and q:
    st.session_state.fill = ""
    st.session_state.msgs.append({"role": "user", "content": q})
    with st.spinner("Searching..."):
        try:
            _, ans = cinesense(q, collection, embed_model, llm, **get_filters())
            st.session_state.msgs.append({"role": "assistant", "content": ans})
        except Exception as e:
            st.session_state.msgs.append({"role": "assistant", "content": f"Error: {e}"})
    st.rerun()

if st.session_state.msgs:
    st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)
    if st.button("Clear chat", key="clr"):
        st.session_state.msgs = []
        st.rerun()

st.markdown("""<div style="text-align:center;margin-top:24px;font-size:10px;
color:#141414;font-family:sans-serif">
CineSense AI  ·  RAG + ChromaDB + Gemini 2.5  ·  ather-ops</div>""",
unsafe_allow_html=True)
