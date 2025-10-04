from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import sqlite3
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "AI_Symptomps_Note.db"
TABLE_NAME = "df_filtered"

# -------------------------
# Notebook function
# -------------------------
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
    unique_associated = []
    seen = set()
    for s in associated:
        if not s:
            continue
        s_clean = s.strip()
        if s_clean == "":
            continue
        if input_lower != s_clean.lower() and s_clean.lower() not in seen:
            unique_associated.append(s_clean)
            seen.add(s_clean.lower())

    return unique_associated

# -------------------------
# Frontend + Backend
# -------------------------

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
        <head>
            <title>Symptoms Recommendation</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    background: #f9f9f9;
                }
                .container {
                    text-align: center;
                    font-size: 20px;
                    background: white;
                    padding: 30px;
                    border-radius: 12px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                    width: 600px;
                }
                input, button {
                    padding: 10px;
                    font-size: 18px;
                    margin-top: 10px;
                    width: 80%;
                }
                button {
                    background: #007BFF;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    cursor: pointer;
                }
                button:hover {
                    background: #0056b3;
                }
                .results {
                    margin-top: 20px;
                    text-align: left;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h2>🩺 Symptoms Recommendation System</h2>
                <form action="/recommend" method="post">
                    <input type="text" name="symptoms" placeholder="e.g. back pain, ปวดเมื่อยกล้ามเนื้อทั่วๆ" required>
                    <br>
                    <button type="submit">🔍 Recommend</button>
                </form>
            </div>
        </body>
    </html>
    """

@app.post("/recommend", response_class=HTMLResponse)
async def recommend(symptoms: str = Form(...)):
    # Load dataset each time (like Jupyter)
    conn = sqlite3.connect(DB_PATH)
    Symptoms = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
    conn.close()

    associated = find_associated_symptoms(symptoms, Symptoms)

    # Build results HTML
    results_html = "<ul>"
    for s in associated:
        results_html += f"<li>{s}</li>"
    results_html += "</ul>" if associated else "<p>No associated symptoms found.</p>"

    # Auto-download results as text file
    file_content = f"Results for '{symptoms}':\n" + "\n".join(associated)
    download_script = f"""
    <script>
        var blob = new Blob([{repr(file_content)}], {{ type: 'text/plain' }});
        var link = document.createElement('a');
        link.href = window.URL.createObjectURL(blob);
        link.download = "recommendations.txt";
        link.click();
    </script>
    """

    return f"""
    <html>
        <head>
            <title>Symptoms Recommendation</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    background: #f9f9f9;
                }}
                .container {{
                    text-align: center;
                    font-size: 20px;
                    background: white;
                    padding: 30px;
                    border-radius: 12px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                    width: 700px;
                }}
                .results {{
                    margin-top: 20px;
                    text-align: left;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h2>Results for '{symptoms}'</h2>
                <div class="results">
                    <h3>Associated Symptoms:</h3>
                    {results_html}
                </div>
                <a href="/">⬅️ Try Again</a>
            </div>
            {download_script}
        </body>
    </html>
    """
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)