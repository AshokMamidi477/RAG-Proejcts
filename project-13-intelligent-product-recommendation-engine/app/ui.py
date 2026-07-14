"""ui.py — Streamlit demo"""
import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Product Recommender", page_icon="🛒", layout="wide")
st.title("🛒 Intelligent Product Recommendation Engine")
st.caption("Semantic search + LLM re-ranking — describe what you need in plain language.")

with st.sidebar:
    category = st.selectbox("Filter by category (optional)",
        ["All", "Electronics", "Toys", "Clothing", "Home & Garden", "Sports", "Beauty"])
    context  = st.text_area("Additional context (optional)",
        placeholder="e.g. Budget is under $50, buying as a gift for a 7-year-old boy")

query = st.text_input("What are you looking for?",
    placeholder="e.g. Something to help my kid learn coding without a screen")

if st.button("Find Products", type="primary") and query:
    with st.spinner("Finding the best matches..."):
        try:
            resp = requests.post(f"{API_URL}/recommend", json={
                "query": query,
                "context": context,
                "category": None if category == "All" else category,
            }, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            if not data["recommendations"]:
                st.info("No recommendations found. Try a different query.")
            else:
                for i, rec in enumerate(data["recommendations"], 1):
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**#{i} — {rec.get('name', 'Product')}**")
                            st.caption(rec.get("reason", ""))
                            st.write(rec.get("description", ""))
                        with col2:
                            st.metric("Price", f"${rec.get('price', '—')}")
                            st.metric("Rating", f"⭐ {rec.get('rating', '—')}")
                        st.divider()
        except Exception as e:
            st.error(f"Error: {e}")
