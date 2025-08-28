import streamlit as st
import pandas as pd
import sqlite3
from sklearn.preprocessing import MinMaxScaler
from mlxtend.frequent_patterns import apriori, association_rules

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

    symptoms_encoded = symptoms_encoded.fillna(0)

    # Scale (although Apriori usually works fine with binary)
    scaler = MinMaxScaler()
    symptoms_scaled = pd.DataFrame(
        scaler.fit_transform(symptoms_encoded),
        columns=symptoms_encoded.columns
    )
    return symptoms_scaled.astype(bool)

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
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Symptoms Recommender", page_icon="🩺", layout="centered")
st.title("🩺 Symptoms Recommender System (Apriori-based)")
st.write("Enter any symptom(s) and get associated symptoms/diseases based on Apriori rules.")

@st.cache_data
def load_data():
    conn = sqlite3.connect("AI_Symptomps_Note.db")
    df = pd.read_sql_query("SELECT * FROM df_filtered", conn)
    conn.close()
    return df

# Load data
Symptoms = load_data()

# Precompute rules (optional – for display only)
symptoms_encoded = preprocess_symptoms(Symptoms)
rules = generate_association_rules(symptoms_encoded)

# User input
user_input = st.text_input("Enter symptom(s):", placeholder="e.g. ไอ, มีเสมหะคัดจมูก or fever")

if st.button("🔍 Recommend"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter at least one symptom")
    else:
        associated = find_associated_symptoms(user_input, Symptoms)
        if not associated:
            st.error("❌ No associated symptoms found.")
        else:
            st.success("✅ Associated Symptoms Found")
            st.write("### Related Symptoms / Possible Conditions")
            st.write(associated)

            # Optional: show related rules for transparency
            matched_rules = rules[rules['antecedents'].apply(lambda x: user_input.lower() in [s.lower() for s in x])]
            if not matched_rules.empty:
                st.write("### Matching Association Rules")
                st.dataframe(matched_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])