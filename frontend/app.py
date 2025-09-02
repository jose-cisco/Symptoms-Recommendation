import streamlit as st
import requests
import os

# --------------------------------------
# Detect API endpoint (local vs deployed)
# --------------------------------------
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Symptoms Recommender",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Symptoms Recommender System")
st.markdown("Enter a symptom and get recommended associated symptoms.")

# Input box
symptom = st.text_input("Type a symptom (e.g., fever, cough, headache):")

if st.button("Recommend"):
    if not symptom.strip():
        st.warning("Please enter a symptom.")
    else:
        try:
            response = requests.get(f"{API_URL}/recommend", params={"symptom": symptom})
            if response.status_code == 200:
                data = response.json()
                if data["recommendations"]:
                    st.success(f"Recommendations for **{symptom}**:")
                    for rec in data["recommendations"]:
                        st.write(f"- {rec}")
                else:
                    st.info("No recommendations found.")
            else:
                st.error("Error from API. Please try again later.")
        except Exception as e:
            st.error(f"Could not connect to API: {e}")

st.markdown("---")
st.caption("Backend powered by FastAPI • Frontend powered by Streamlit")