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
## 📂 Working On Symptoms Data And Symptoms Recommendation Demo
- AI_Symptomps_Note.db # Placeholder SQLite DB 
- Data Engineering.ipynb — Extracted relevant explanation to the symptoms information.  
  This phase includes:
  - Processing and cleaning irrelevant information from extracted symptoms data  
  - Transforming the cleaned data into a structured DataFrame  
  - Loading the transformed data into the SQLite database for querying by the API
## 🧩 Symptoms Recommendation Process

This module and notebook implement the **core workflow** for discovering and analyzing relationships between symptoms using data-driven techniques.

### 1️⃣ `preprocess_symptoms`
Cleans and standardizes raw symptom notes:
- Removes irrelevant or non-symptom text  
- Normalizes symptom terminology  
- Converts structured symptom entries into a DataFrame for analysis  

### 2️⃣ `generate_association_rules`
Uses **Apriori Algorithm** to find associations between symptoms:
- Generates frequent itemsets based on co-occurrence patterns  
- Derives association rules (`antecedents → consequents`) with support, confidence, and lift metrics  
- Identifies potential symptom relationships  

### 3️⃣ `select_features`
Selects key features and filters significant symptom relationships:
- Applies thresholding on metrics (e.g., confidence ≥ 0.6, lift ≥ 1.0)  
- Keeps only high-impact or relevant associations  
- Reduces noise and redundant patterns  

### 4️⃣ `implement_symptom_analysis`
Integrates all previous steps:
- Combines cleaned and structured data  
- Applies association rule mining  
- Selects important features  
- Produces a **Symptoms Recommendation Table** showing other symptoms most commonly linked to a given symptom or case  

---

## 📓 Symptoms_Recommendation.ipynb

The Jupyter Notebook serves as a complete analytical environment for:
- Testing and validating the symptom preprocessing and association rule generation pipeline  
- Iteratively tuning thresholds for improved recommendation accuracy  

It performs:
1. Data extraction and cleaning from raw symptom notes  
2. Feature transformation into a binary symptom matrix  
3. Association rule generation using the **Apriori** method  
4. Feature selection and importance ranking  
5. Generation of symptom association recommendations  

This notebook can be run independently to update the symptom database or validate model performance before deployment via FastAPI.
The Jupyter notebook which consits of Data Engineering.ipynb and Symptoms_Recommendation.ipynb can be excuted to see the progress and reslut through each of the block lines by press play (F5) to excute each lines of the code.
