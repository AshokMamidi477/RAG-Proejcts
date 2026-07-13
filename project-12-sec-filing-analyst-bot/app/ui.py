"""ui.py — Streamlit chat interface"""
import uuid, requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="SEC Filing Analyst", page_icon="📈", layout="wide")
st.title("📈 SEC Filing Analyst Bot")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Filters")
    ticker = st.text_input("Filter by ticker (optional)", placeholder="e.g. AAPL")
    if st.button("New Conversation"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption("Sources: " + " | ".join(msg["sources"]))

if prompt := st.chat_input("Ask about the filings..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching filings..."):
            try:
                resp = requests.post(f"{API_URL}/chat", json={
                    "session_id": st.session_state.session_id,
                    "question": prompt,
                    "ticker_filter": ticker or None,
                }, timeout=30)
                data = resp.json()
                st.markdown(data["answer"])
                if data["sources"]:
                    st.caption("Sources: " + " | ".join(data["sources"]))
                st.session_state.messages.append({
                    "role": "assistant", "content": data["answer"], "sources": data["sources"]
                })
            except Exception as e:
                st.error(f"Error: {e}")
