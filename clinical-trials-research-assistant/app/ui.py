"""ui.py — Streamlit frontend"""
import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Clinical Trials Research Assistant", page_icon="🔬", layout="wide")
st.title("🔬 Clinical Trials Research Assistant")
st.caption("Ask questions across your clinical trial corpus — get grounded answers with citations.")

with st.sidebar:
    st.header("Settings")
    k = st.slider("Documents to retrieve (k)", 2, 10, 5)
    st.markdown("---")
    st.caption("Make sure the FastAPI server is running on port 8000.")

question = st.text_input("Ask a question about the trial corpus",
    placeholder="e.g. What were the primary endpoints of trials studying diabetes treatment?")

if st.button("Search", type="primary") and question:
    with st.spinner("Searching corpus..."):
        try:
            resp = requests.post(f"{API_URL}/query", json={"question": question, "k": k}, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            st.markdown("### Answer")
            st.markdown(data["answer"])
            if data["sources"]:
                st.markdown("### Sources")
                for src in data["sources"]:
                    st.markdown(f"- `{src}`")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Make sure uvicorn is running on port 8000.")
        except Exception as e:
            st.error(f"Error: {e}")
