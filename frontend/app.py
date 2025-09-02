import streamlit as st
import requests

st.set_page_config(page_title="Symptoms Recommendation", layout="centered")

st.title("🤒 Symptoms Recommendation System")
st.write("Enter a symptom and get recommended associated symptoms.")

# API URL (backend on Render)
API_URL = "https://your-fastapi-backend.onrender.com/recommend"

symptom = st.text_input("Enter Symptom", "")

if st.button("Find Recommendations"):
    if symptom.strip():
        try:
            response = requests.get(API_URL, params={"symptom": symptom})
            if response.status_code == 200:
                data = response.json()
                if data["recommendations"]:
                    st.success("Recommendations found!")
                    for rec in data["recommendations"]:
                        st.write(f"- {rec}")
                else:
                    st.warning("No recommendations found.")
            else:
                st.error(f"API Error {response.status_code}")
        except Exception as e:
            st.error(f"Error connecting to API: {e}")
    else:
        st.warning("Please enter a symptom first.")