from fastapi import FastAPI, Query
from pydantic import BaseModel
import pandas as pd
import sqlite3
from typing import List

app = FastAPI(
    title="🩺 Symptoms Recommender API",
    description="Recommend associated symptoms using Apriori association rules.",
    version="1.0.0",
)

class RecommendationResponse(BaseModel):
    input: str
    recommendations: List[str]
    message: str

def load_data():
    conn = sqlite3.connect("AI_Symptomps_Note.db")
    df = pd.read_sql_query("SELECT * FROM df_filtered", conn)
    conn.close()
    return df

Symptoms = load_data()

def find_associated_symptoms(input_symptom, Symptoms):
    symptoms_pairs = Symptoms[['symptoms_note_clean', 'search_term']].dropna()
    mask = (
        symptoms_pairs['search_term'].str.contains(input_symptom, case=False, na=False) |
        symptoms_pairs['symptoms_note_clean'].str.contains(input_symptom, case=False, na=False)
    )
    matching_rows = symptoms_pairs[mask]
    if matching_rows.empty:
        return []
    associated = []
    for _, row in matching_rows.iterrows():
        raw = str(row['symptoms_note_clean'])
        parts = [p.strip() for p in raw.split(',') if p.strip()]
        associated.extend(parts)
    return list(set([s for s in associated if s.lower() != input_symptom.lower()]))

@app.get("/")
def root():
    return {"message": "Welcome to the Symptoms Recommender API. Visit /docs for Swagger UI."}

@app.get(
    "/recommend",
    response_model=RecommendationResponse,
    summary="Get associated symptoms",
    description="Provide a symptom and receive a list of related symptoms."
)
def recommend(
    symptom: str = Query(..., description="Symptom to search for", example="fever")
):
    associated = find_associated_symptoms(symptom, Symptoms)
    if not associated:
        return {"input": symptom, "recommendations": [], "message": "No matches found"}
    return {"input": symptom, "recommendations": associated, "message": "Success"}