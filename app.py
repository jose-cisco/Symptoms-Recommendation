from fastapi import FastAPI, Query
import pandas as pd
import sqlite3
from mlxtend.frequent_patterns import apriori, association_rules

app = FastAPI(title="Symptoms Recommender API", version="1.0")

# -----------------------------
# Utility Functions
# -----------------------------
def preprocess_symptoms(Symptoms):
    Symptoms['symptoms_note_clean'] = Symptoms['symptoms_note_clean'].fillna('')
    Symptoms['symptoms_list'] = Symptoms['symptoms_note_clean'].str.split(',')

    all_symptoms = set()
    for symptoms in Symptoms['symptoms_list']:
        if isinstance(symptoms, list):
            cleaned = [s.strip() for s in symptoms if s and s.strip()]
            all_symptoms.update(cleaned)

    symptoms_encoded = pd.DataFrame(0, index=Symptoms.index, columns=list(all_symptoms))
    for idx, symptoms in enumerate(Symptoms['symptoms_list']):
        if isinstance(symptoms, list):
            for s in symptoms:
                if s and s.strip():
                    symptoms_encoded.loc[idx, s.strip()] = 1
    return symptoms_encoded.astype(bool)

def generate_association_rules(symptoms_encoded, min_support=0.2, min_confidence=0.8):
    frequent_itemsets = apriori(symptoms_encoded, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    return rules

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

    input_lower = input_symptom.lower()
    unique_associated, seen = [], set()
    for s in associated:
        if s:
            s_clean = s.strip()
            if s_clean and input_lower != s_clean.lower() and s_clean.lower() not in seen:
                unique_associated.append(s_clean)
                seen.add(s_clean.lower())
    return unique_associated

# -----------------------------
# Load Data once
# -----------------------------
def load_data():
    conn = sqlite3.connect("AI_Symptomps_Note.db")
    df = pd.read_sql_query("SELECT * FROM df_filtered", conn)
    conn.close()
    return df

Symptoms = load_data()
symptoms_encoded = preprocess_symptoms(Symptoms)
rules = generate_association_rules(symptoms_encoded)

# -----------------------------
# API Endpoints
# -----------------------------
@app.get("/")
def root():
    return {"message": "Welcome to the Symptoms Recommender API", "endpoints": ["/recommend"]}

@app.get("/recommend")
def recommend(symptom: str = Query(..., description="Symptom string to search for")):
    associated = find_associated_symptoms(symptom, Symptoms)
    if not associated:
        return {"input": symptom, "recommendations": [], "message": "No matches found"}
    return {"input": symptom, "recommendations": associated}