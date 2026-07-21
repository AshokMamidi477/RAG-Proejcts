"""ui.py — Streamlit legal research interface"""
import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Legal Precedent Finder", page_icon="⚖️", layout="wide")
st.title("⚖️ Legal Precedent Finder")
st.caption("Ask legal questions — get answers grounded in case law and statutes.")
st.warning("This tool is for research purposes only and does not constitute legal advice.")

question = st.text_area("Your legal question",
    placeholder="e.g. What are the requirements for establishing a duty of care in negligence claims?",
    height=100)

if st.button("Research", type="primary") and question:
    with st.spinner("Searching case law..."):
        try:
            resp = requests.post(f"{API_URL}/search", json={"question": question}, timeout=45)
            resp.raise_for_status()
            data = resp.json()
            st.markdown("### Research Results")
            st.markdown(data["answer"])
            if data.get("sources"):
                with st.expander("View source excerpts"):
                    for i, src in enumerate(data["sources"], 1):
                        st.text(f"Source {i}: {src}")
        except Exception as e:
            st.error(f"Error: {e}")
