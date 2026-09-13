import streamlit as st
from project import app  # reuse the compiled LangGraph app from project.py

st.set_page_config(page_title="Auto Parts Assistant", page_icon="🔧")
st.title("🔧 Auto Parts Recommendation Assistant")
st.caption("A LangGraph agent using a local LLM (Ollama) + vector search")

query = st.text_input(
    "What part are you looking for?",
    placeholder="e.g. Recommend brake pads for a 2018 Honda Civic"
)

if st.button("Search", type="primary") and query:
    with st.spinner("Searching..."):
        initial_state = {
            "query": query,
            "intent": {},
            "is_ambiguous": False,
            "candidates": [],
            "ranked": [],
            "response": "",
        }
        result = app.invoke(initial_state)

    st.markdown("### Recommendation")
    st.write(result["response"])

    # Transparency/debugging: show what was understood and what was retrieved
    with st.expander("🔍 Extracted intent"):
        st.json(result["intent"])

    with st.expander("📦 Candidates before vs. after ranking"):
        st.write("**Retrieved candidates (vector similarity order):**")
        for c in result["candidates"]:
            st.write(f"- {c.get('part_name')} | {c.get('compatible_make')} {c.get('compatible_model')} | ${c.get('price_usd')}")

        st.write("**Re-ranked results (after scoring):**")
        for c in result["ranked"]:
            st.write(f"- {c.get('part_name')} | {c.get('compatible_make')} {c.get('compatible_model')} | ${c.get('price_usd')}")