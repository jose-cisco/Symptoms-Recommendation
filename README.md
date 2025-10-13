# 🩺 Symptoms Recommendation System (FastAPI + Apriori + Similarity)

A powerful hybrid **Symptom Recommendation System** that helps identify associated or possible next symptoms based on a given patient’s input.  
It combines **Association Rules (Apriori)** and **Symptom Similarity Analysis** to provide interpretable and intelligent recommendations.

---

## 🚀 Features

- **Hybrid Recommendation Engine**
  - Association Rules (Apriori)
  - Symptom Co-occurrence Similarity
  - Frequency-based Fallback

- **FastAPI Backend**
  - REST API with `/recommend`, `/download_json`, and `/health`
  - OpenAPI docs automatically generated at `/docs`

- **Interactive HTML Frontend**
  - Built-in web interface for entering symptoms
  - Beautiful, centered layout with dynamic results
  - Auto-downloads JSON results file

- **SQLite Database Integration**
  - Reads symptoms data directly from a `.db` file
  - Auto-preprocessing and encoding at runtime

---

## 🧠 Tech Stack

| Component | Technology |
|------------|-------------|
| Backend | [FastAPI](https://fastapi.tiangolo.com/) |
| Frontend | HTML + JavaScript (embedded in FastAPI) |
| Database | SQLite |
| ML / Data Mining | `mlxtend`, `pandas`, `numpy` |
| File Downloads | JSON auto-generation |
| Server | `uvicorn` |

---

## 📦 Installation

### 1️⃣ Clone Repository
```bash
git clone https://github.com/yourusername/symptoms-recommendation.git
cd symptoms-recommendation
```

### 2️⃣ Create Virtual Environment
```bash
conda create -n Symptoms_Recoomendation python=3.10 -y
conda activate Symptoms_Recoomendation
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Add Your Database
Place your SQLite database file (e.g., `Symptoms Database.db`) in the project directory.  
Update these constants in `main.py` if needed:
```python
RAW_DATA_PATH = "Symptoms Database.db"
SQL_QUERY = "SELECT * FROM ai_symptom_picker"
```

---

## ▶️ Run the Server

Start FastAPI using Uvicorn:

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Then open:

👉 **Frontend UI:** [http://127.0.0.1:8000](http://127.0.0.1:8000)  
👉 **API Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)  
👉 **Health Check:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## 💻 Web Interface

The HTML interface allows users to:

1. Input symptoms (comma-separated, e.g. `cough, sore throat`)
2. Select recommendation method:
   - `hybrid` (rules + similarity)
   - `rules` (Apriori only)
   - `similarity` (co-occurrence only)
3. View real-time recommendations
4. Auto-download the JSON results file

![Frontend Example](assets/frontend.png)

---

## 🧩 API Endpoints

| Method | Endpoint | Description |
|--------|-----------|-------------|
| `GET` | `/` | HTML frontend interface |
| `GET` | `/health` | System and model diagnostics |
| `POST` | `/recommend` | Get recommended symptoms |
| `POST` | `/download_json` | Download recommendations as JSON file |
| `GET` | `/docs` | Swagger API documentation |

### Example Request
```bash
curl -X POST "http://127.0.0.1:8000/recommend"      -H "Content-Type: application/json"      -d '{
           "symptoms": "ปวดท้อง, คลื่นไส้",
           "age": 25,
           "gender": "female",
           "top_n": 10,
           "method": "hybrid"
         }'
```

### Example Response
```json
{
  "timestamp": "2025-10-13T21:45:00",
  "input_symptoms": ["ปวดท้อง", "คลื่นไส้"],
  "recommendations": [
    {"symptom": "เบื่ออาหาร"},
    {"symptom": "อาเจียน"},
    {"symptom": "ท้องอืด"}
  ],
  "patient_info": {
    "age": "25",
    "gender": "female"
  },
  "total_recommendations": 3,
  "message": "Success"
}
```

---

## 🩹 Health Check Example

Check model readiness and system status:
```bash
curl http://127.0.0.1:8000/health
```

Example Response:
```json
{
  "status": "ok",
  "total_symptoms": 154,
  "association_rules": 462,
  "model_loaded": true,
  "mlxtend_available": true
}
```
---

## 🖼️ Results 

---
![image alt](https://github.com/jose-cisco/Symptoms-Recommendation/blob/main/messageImage_1760343260123_0.jpg?raw=true)
![image alt](https://github.com/jose-cisco/Symptoms-Recommendation/blob/main/messageImage_1760343269847_0.jpg?raw=true)
![image alt](https://github.com/jose-cisco/Symptoms-Recommendation/blob/main/messageImage_1760343278164_0.jpg?raw=true)
---

## 📤 Auto JSON Download
After generating results via the web interface, click **Download JSON**,  
and your output will automatically be saved (e.g. `recommendations_20251013_214500.json`).

---

## 🧪 Development Notes

- Automatically builds Apriori association rules and co-occurrence matrices.
- If `mlxtend` isn’t installed, gracefully skips rule generation.
- Falls back to frequency-based suggestions if both methods fail.
- Compatible with multilingual symptom notes (Thai + English).

---

## 🏗️ Folder Structure

```
├── main.py                   # Main FastAPI app
├── Symptoms Database.db       # SQLite DB file
├── requirements.txt           # Dependencies
├── assets/
│   └── frontend.png           # Optional screenshot for README
└── README.md
```

---

## 💡 Future Enhancements

- Add disease prediction alongside symptom recommendation  
- Improve NLP-based preprocessing  
- Add multi-language support (EN/TH)  
- Deploy via Docker or Cloud Run  

---
