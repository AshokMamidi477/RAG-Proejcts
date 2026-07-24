"""ui.py — Streamlit chat interface"""
import uuid, requests
import streamlit as st

API_URL = "http://localhost:8000"
st.set_page_config(page_title="Customer Support", page_icon="💬", layout="centered")
st.title("💬 Customer Support")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("How can I help you?"):
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("user"): st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("..."):
            try:
                resp = requests.post(f"{API_URL}/chat",
                    json={"session_id":st.session_state.session_id,"message":prompt},timeout=30)
                data = resp.json()
                st.markdown(data["response"])
                if data.get("escalated"):
                    st.warning("Your query has been escalated to a human agent.")
                st.session_state.messages.append({"role":"assistant","content":data["response"]})
            except Exception as e:
                st.error(f"Error: {e}")
