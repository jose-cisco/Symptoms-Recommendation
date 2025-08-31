# 🤖 Symptoms Recommender API

This project provides a **FastAPI-based Symptoms Recommendation API** using Association Rules (Apriori).  
It loads a SQLite database (`AI_Symptomps_Note.db`) with symptoms and allows querying for related symptoms.

---

## 🚀 Features
- Input a symptom (or partial string)
- Get back associated symptoms from the dataset
- Runs as a REST API with automatic Swagger UI (`/docs`)
  
---

## 📂 Project Structure
- main.py # FastAPI app
- requirements.txt # Dependencies
- Procfile # Deployment process file (Render/Railway)
- AI_Symptomps_Note.db # Placeholder SQLite DB 
- README.md # Documentation
