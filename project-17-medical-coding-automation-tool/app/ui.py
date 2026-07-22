"""ui.py — Streamlit coder review interface"""
import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Medical Coding Tool", page_icon="🏥", layout="wide")
st.title("🏥 Medical Coding Automation Tool")
st.caption("AI-suggested ICD-10 codes for review and confirmation by a certified coder.")
st.warning("AI suggestions must be reviewed and confirmed by a certified medical coder before use.")

note = st.text_area("Clinical Note", height=200,
    placeholder="e.g. 58-year-old male presenting with chest pain radiating to left arm...")

if st.button("Suggest ICD-10 Codes", type="primary") and note:
    with st.spinner("Retrieving and analysing codes..."):
        try:
            resp = requests.post(f"{API_URL}/suggest-codes",
                                 json={"clinical_note": note}, timeout=30)
            resp.raise_for_status()
            suggestions = resp.json()["suggestions"]

            st.subheader(f"{len(suggestions)} Code Suggestions")
            for i, s in enumerate(suggestions, 1):
                conf = s.get("confidence", 0)
                colour = "green" if conf >= 0.8 else "orange" if conf >= 0.6 else "red"
                with st.container():
                    col1, col2, col3 = st.columns([1, 4, 1])
                    col1.markdown(f"**`{s.get('code','?')}`**")
                    col2.write(s.get("description", ""))
                    col2.caption(s.get("reasoning", ""))
                    col3.markdown(f":{colour}[{conf*100:.0f}%]")
                    st.divider()
        except Exception as e:
            st.error(f"Error: {e}")
