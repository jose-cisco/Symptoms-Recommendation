"""
FastAPI Symptom Recommendation System with Auto-Download
Deploy symptom recommendations as REST API with JSON download capability
"""

"""
FastAPI Symptom Recommendation System with Auto-Download
Deploy symptom recommendations as REST API with JSON download capability
"""

from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import sqlite3
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import traceback
import sys
import warnings
from io import BytesIO

warnings.filterwarnings('ignore')

# Try to import ML libraries, but don't crash if they fail
MLXTEND_AVAILABLE = False
try:
    from mlxtend.frequent_patterns import apriori, association_rules
    MLXTEND_AVAILABLE = True
except ImportError:
    # This warning is included in the recommender class as well
    pass 

# ============================================================================
# DATA MODELS
# ============================================================================

class SymptomRequest(BaseModel):
    symptoms: str = Field(..., description="Comma-separated symptoms (e.g., 'ไอ, เจ็บคอ')")
    age: Optional[int] = Field(None, ge=0, le=120)
    gender: Optional[str] = Field(None, description="e.g., 'male', 'female'")
    top_n: int = Field(10, ge=1, le=50)
    method: str = Field('hybrid', description="'rules', 'similarity', or 'hybrid' (default)")

class SymptomRecommendation(BaseModel):
    # Removed: score: float (as requested)
    symptom: str

class RecommendationResponse(BaseModel):
    timestamp: str
    input_symptoms: List[str]
    recommendations: List[SymptomRecommendation]
    patient_info: Dict[str, Optional[str]]
    total_recommendations: int
    message: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    total_symptoms: int
    association_rules: int
    model_loaded: bool
    error: Optional[str] = None
    mlxtend_available: bool

# ============================================================================
# SYMPTOM RECOMMENDER CLASS (Updated Logic)
# ============================================================================

class SymptomRecommender:
    """Symptom recommendation engine with ultra-robust error handling"""
    
    def __init__(self, symptoms_data):
        self.data = symptoms_data
        self.symptoms_encoded = None
        self.rules = None
        self.symptom_vectors = None
        self.all_symptoms_list = []
        self.initialization_error = None
        self.symptom_frequency = {}
        
    def preprocess_data(self):
        """Preprocess and encode symptoms"""
        try:
            # Find symptom column
            symptom_col = None
            # Prioritize 'search_term' as per the notebook's usage
            for col in ['search_term', 'symptoms_note_clean', 'symptoms', 'symptom', 'symptom_text', 'chief_complaint']:
                if col in self.data.columns:
                    symptom_col = col
                    break
            
            if symptom_col is None:
                print(f"Available columns: {list(self.data.columns)}")
                raise ValueError(f"No symptom column found. Please check your database.")
            
            print(f"✓ Using column: '{symptom_col}'")
            self.data['symptoms_clean'] = self.data[symptom_col].astype(str).fillna('')
            
            # Split and clean
            self.data['symptoms_list'] = self.data['symptoms_clean'].apply(
                lambda x: [s.strip() for s in str(x).split(',') if s.strip()]
            )
            
            # Get unique symptoms
            all_symptoms = set()
            symptom_counts = {}
            
            for symptoms in self.data['symptoms_list']:
                for symptom in symptoms:
                    if symptom:
                        all_symptoms.add(symptom)
                        symptom_counts[symptom] = symptom_counts.get(symptom, 0) + 1
            
            self.symptom_frequency = symptom_counts
            self.all_symptoms_list = sorted(list(all_symptoms))
            
            print(f"✓ Found {len(self.all_symptoms_list)} unique symptoms")
            
            # Create binary matrix
            symptoms_encoded = pd.DataFrame(
                0, 
                index=range(len(self.data)), 
                columns=self.all_symptoms_list
            )
            
            for idx, symptoms in enumerate(self.data['symptoms_list']):
                for symptom in symptoms:
                    if symptom in self.all_symptoms_list:
                        symptoms_encoded.loc[idx, symptom] = 1
            
            self.symptoms_encoded = symptoms_encoded
            print(f"✓ Encoded matrix shape: {self.symptoms_encoded.shape}")
            return self.symptoms_encoded
            
        except Exception as e:
            print(f"✗ ERROR in preprocess_data: {e}")
            traceback.print_exc()
            raise
    
    def build_association_rules(self):
        """Build association rules (safe version)"""
        try:
            if not MLXTEND_AVAILABLE:
                print("⊘ Skipping association rules (mlxtend not available)")
                self.rules = None
                return None
            
            print("Building association rules...")
            symptoms_bool = self.symptoms_encoded.astype(bool)
            
            # Try with very low support
            for min_sup in [0.005, 0.002, 0.001]:
                try:
                    frequent_itemsets = apriori(
                        symptoms_bool, 
                        min_support=min_sup,
                        use_colnames=True,
                        max_len=4
                    )
                    
                    if len(frequent_itemsets) > 0:
                        print(f"✓ Found {len(frequent_itemsets)} itemsets (support={min_sup})")
                        break
                except:
                    continue
            
            if len(frequent_itemsets) == 0:
                print("⊘ No frequent itemsets found")
                self.rules = None
                return None
            
            # Generate rules
            self.rules = association_rules(
                frequent_itemsets,
                metric="confidence",
                min_threshold=0.1
            )
            
            if len(self.rules) > 0:
                self.rules['score'] = self.rules['confidence'] * self.rules['lift']
            
            print(f"✓ Generated {len(self.rules)} association rules")
            return self.rules
            
        except Exception as e:
            print(f"⊘ Association rules failed: {e}")
            self.rules = None
            return None
    
    def build_symptom_vectors(self):
        """Build co-occurrence vectors"""
        try:
            print("Building symptom vectors...")
            co_occurrence = self.symptoms_encoded.T.dot(self.symptoms_encoded)
            
            # Normalize by frequency
            symptom_freq = self.symptoms_encoded.sum(axis=0).replace(0, 1)
            self.symptom_vectors = co_occurrence.div(symptom_freq, axis=0).fillna(0)
            
            print(f"✓ Symptom vectors ready: {self.symptom_vectors.shape}")
            return self.symptom_vectors
            
        except Exception as e:
            print(f"✗ ERROR in build_symptom_vectors: {e}")
            self.symptom_vectors = None
            return None
    
    def recommend_symptoms(self, input_symptoms, age=None, gender=None, top_n=10, method='similarity'):
        """Generate recommendations with fallback"""
        try:
            # Parse input
            if isinstance(input_symptoms, str):
                input_symptoms = [s.strip() for s in input_symptoms.split(',') if s.strip()]
            
            # Validate symptoms
            valid_symptoms = [s for s in input_symptoms if s in self.all_symptoms_list]
            invalid_symptoms = [s for s in input_symptoms if s not in self.all_symptoms_list]
            
            message = None
            if invalid_symptoms:
                message = f"Unknown symptoms: {', '.join(invalid_symptoms[:3])}"
                if len(invalid_symptoms) > 3:
                    message += f" (+{len(invalid_symptoms)-3} more)"
            
            if not valid_symptoms:
                return {
                    'input_symptoms': input_symptoms,
                    'recommendations': [],
                    'message': f"No matching symptoms. Try: {', '.join(self.all_symptoms_list[:5])}...",
                    'patient_info': {'age': age, 'gender': gender}
                }
            
            recommendations = {}
            
            # Try rules first
            if method in ['rules', 'hybrid'] and self.rules is not None:
                try:
                    rule_recs = self._recommend_from_rules(valid_symptoms)
                    # Increased weight for rules in hybrid (0.7)
                    rule_weight = 0.7 if method == 'hybrid' else 1.0
                    for symptom, score in rule_recs.items():
                        recommendations[symptom] = recommendations.get(symptom, 0) + score * rule_weight
                except Exception as e:
                    print(f"Rules method failed: {e}")
            
            # Always try similarity
            if method in ['similarity', 'hybrid'] and self.symptom_vectors is not None:
                try:
                    sim_recs = self._recommend_from_similarity(valid_symptoms)
                    # Reduced weight for similarity in hybrid (0.3)
                    sim_weight = 0.3 if method == 'hybrid' else 1.0
                    for symptom, score in sim_recs.items():
                        recommendations[symptom] = recommendations.get(symptom, 0) + score * sim_weight
                except Exception as e:
                    print(f"Similarity method failed: {e}")
            
            # Fallback: frequency-based
            if not recommendations and not (method in ['rules', 'similarity'] and (self.rules is None and self.symptom_vectors is None)):
                recommendations = self._recommend_by_frequency(valid_symptoms)
                if not message:
                    message = "Using frequency-based recommendations (fallback)"
            
            # Remove input symptoms
            for symptom in valid_symptoms:
                recommendations.pop(symptom, None)
            
            # Sort and return
            sorted_recs = sorted(
                recommendations.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_n]
            
            # NOTE: Score is intentionally dropped from the final output format here
            return {
                'input_symptoms': valid_symptoms,
                'recommendations': [
                    {'symptom': s} # Removed score field
                    for s, score in sorted_recs
                ],
                'message': message,
                'patient_info': {'age': age, 'gender': gender}
            }
            
        except Exception as e:
            print(f"ERROR in recommend_symptoms: {e}")
            traceback.print_exc()
            # Return empty but valid response
            return {
                'input_symptoms': [],
                'recommendations': [],
                'message': f"Error: {str(e)}",
                'patient_info': {'age': age, 'gender': gender}
            }

    def _recommend_from_rules(self, input_symptoms):
        """Recommend from association rules (using max confidence/score if multiple rules apply)"""
        recommendations = {}
        if self.rules is None or len(self.rules) == 0:
            return recommendations
        
        for _, rule in self.rules.iterrows():
            antecedents = set(rule['antecedents'])
            consequents = set(rule['consequents'])
            
            # Check if any input symptom is in the rule's antecedent set
            if antecedents.intersection(set(input_symptoms)):
                for symptom in consequents:
                    if symptom not in input_symptoms:
                        score = rule.get('score', rule['confidence'])
                        recommendations[symptom] = max(
                            recommendations.get(symptom, 0), 
                            score
                        )
        return recommendations

    def _recommend_from_similarity(self, input_symptoms):
        """
        Recommend from co-occurrence using summation to aggregate evidence.
        """
        recommendations = {}
        if self.symptom_vectors is None:
            return recommendations
        
        for input_symptom in input_symptoms:
            if input_symptom in self.symptom_vectors.index:
                similarities = self.symptom_vectors.loc[input_symptom]
                for symptom, score in similarities.items():
                    if symptom != input_symptom and score > 0:
                        recommendations[symptom] = recommendations.get(symptom, 0) + score
                        
        return recommendations
    
    def _recommend_by_frequency(self, input_symptoms):
        """Fallback: recommend most common symptoms"""
        recommendations = {}
        for symptom, count in self.symptom_frequency.items():
            if symptom not in input_symptoms:
                # Use normalized frequency as a score
                recommendations[symptom] = count / len(self.data)
        return recommendations

    def fit(self):
        """Train the recommendation system"""
        print("Building recommendation system...")
        self.preprocess_data()
        self.build_association_rules()
        self.build_symptom_vectors()
        print("Recommendation system ready!")
        return self

# ============================================================================
# GLOBAL DATA & MODEL INITIALIZATION
# ============================================================================

RECOMMENDER: Optional[SymptomRecommender] = None
DATA_LOAD_ERROR: Optional[str] = None
RAW_DATA_PATH = 'Symptoms Database.db'
SQL_QUERY = "SELECT * FROM ai_symptom_picker" 

print("="*60)
print("INITIALIZING SYMPTOM RECOMMENDER MODEL")
print("="*60)

try:
    print(f"Connecting to database: {RAW_DATA_PATH}...")
    # 1. Establish a connection
    connection = sqlite3.connect(RAW_DATA_PATH)
    
    # 2. Fetch data
    data = pd.read_sql_query(SQL_QUERY, connection)
    connection.close()
    
    print(f"Successfully loaded {len(data)} records. Training model...")
    
    # 3. Initialize and fit the Recommender 
    RECOMMENDER = SymptomRecommender(data)
    RECOMMENDER.fit()
    
    print("Model training complete.")

except Exception as e:
    DATA_LOAD_ERROR = f"Failed to load data or initialize model: {e}"
    print(DATA_LOAD_ERROR)
    print("Traceback:")
    traceback.print_exc(file=sys.stdout)

# ============================================================================
# FASTAPI APP & ENDPOINTS
# ============================================================================

app = FastAPI(
    title="Symptom Recommender API",
    description="A service to recommend next possible symptoms based on user input (Hybrid Association Rules and Similarity).",
    version="1.0.2" # Version updated for scoring/weight change
)

# CORS Middleware 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
def health_check():
    """Check the health and readiness of the model"""
    
    model_loaded = RECOMMENDER is not None and DATA_LOAD_ERROR is None
    
    rules_count = 0
    total_symptoms = 0
    
    if model_loaded:
        rules_count = len(RECOMMENDER.rules) if RECOMMENDER.rules is not None else 0
        total_symptoms = len(RECOMMENDER.all_symptoms_list)
        
    return HealthResponse(
        status="ok" if model_loaded else "error",
        total_symptoms=total_symptoms,
        association_rules=rules_count,
        model_loaded=model_loaded,
        error=DATA_LOAD_ERROR,
        mlxtend_available=MLXTEND_AVAILABLE
    )


@app.post("/recommend", response_model=RecommendationResponse, tags=["Prediction"])
def recommend_symptoms(request: SymptomRequest):
    """
    Get the next set of recommended symptoms based on the input.
    """
    if RECOMMENDER is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model not loaded. Error: {DATA_LOAD_ERROR}"
        )

    try:
        # Generate recommendations using the globally loaded model
        result = RECOMMENDER.recommend_symptoms(
            input_symptoms=request.symptoms,
            age=request.age,
            gender=request.gender,
            top_n=request.top_n,
            method=request.method
        )
        
        # Format the final response
        return RecommendationResponse(
            timestamp=datetime.now().isoformat(),
            input_symptoms=result.get('input_symptoms', []),
            recommendations=result['recommendations'],
            patient_info={
                'age': str(request.age) if request.age is not None else 'N/A',
                'gender': request.gender if request.gender is not None else 'N/A'
            },
            total_recommendations=len(result['recommendations']),
            message=result.get('message', 'Success')
        )

    except Exception as e:
        print("Error during recommendation:", e)
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during recommendation."
        )


@app.post("/download_json", tags=["Prediction"])
async def download_json(request: SymptomRequest):
    """
    Get the recommendation response and return it as a downloadable JSON file.
    """
    try:
        # Call the core recommendation logic
        response_model = recommend_symptoms(request)
        
        # Convert the Pydantic model to a dictionary
        response_data = response_model.model_dump()
        
        # Convert dictionary to JSON string
        json_content = json.dumps(response_data, ensure_ascii=False, indent=4)
        
        # Use a BytesIO buffer to hold the file content
        content_buffer = BytesIO(json_content.encode('utf-8'))
        
        filename = f"recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        return FileResponse(
            path=content_buffer,
            media_type="application/json",
            filename=filename,
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
            }
        )
        
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"detail": e.detail})
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"An unexpected error occurred during file generation: {e}"})


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index():
    """Basic HTML interface for quick testing"""
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Symptom Recommender</title>
    <style>
        body {{ 
            font-family: Arial, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background-color: #f0f2f5; 
        }}
        .container {{ 
            max-width: 800px; 
            margin: 0 auto; 
            background: white; 
            padding: 30px; 
            border-radius: 8px; 
            box-shadow: 0 4px 12px rgba(0,0,0,0.1); 
        }}
        h1 {{ 
            color: #2c3e50; 
            text-align: center; 
            margin-bottom: 25px; 
            border-bottom: 2px solid #3498db; 
            padding-bottom: 10px;
        }}
        form {{ 
            display: grid; 
            grid-template-columns: 1fr 1fr; 
            gap: 20px; 
            margin-bottom: 30px; 
            padding: 20px; 
            border: 1px solid #dcdcdc; 
            border-radius: 6px; 
            background-color: #fafafa;
        }}
        .form-group {{ display: flex; flex-direction: column; }}
        label {{ font-weight: 600; margin-bottom: 5px; color: #34495e; }}
        input[type="text"], input[type="number"], select {{ 
            padding: 10px; 
            border: 1px solid #bdc3c7; 
            border-radius: 4px; 
            font-size: 16px; 
            box-sizing: border-box;
        }}
        button {{ 
            grid-column: 1 / 3; 
            padding: 12px; 
            background-color: #3498db; 
            color: white; 
            border: none; 
            border-radius: 4px; 
            font-size: 18px; 
            cursor: pointer; 
            transition: background-color 0.3s; 
        }}
        button:hover {{ background-color: #2980b9; }}
        
        #download-btn {{
            background-color: #2ecc71; 
            margin-top: 10px;
        }}
        #download-btn:hover {{ background-color: #27ae60; }}

        .results-section {{ border-top: 2px solid #dcdcdc; padding-top: 20px; display: none; }}
        .results-section.show {{ display: block; }}
        h2 {{ color: #2980b9; border-bottom: 1px solid #ecf0f1; padding-bottom: 5px; margin-bottom: 15px; }}
        .result-item {{ 
            display: block; /* Simpler block list as score is removed */
            padding: 10px 0; 
            border-bottom: 1px dashed #ecf0f1; 
        }}
        .result-item:last-child {{ border-bottom: none; }}
        .message {{ margin-top: 15px; padding: 10px; background-color: #fef7e0; border: 1px solid #f39c12; color: #c0392b; border-radius: 4px; }}
        .health-status {{ text-align: center; font-size: 1.1em; padding: 10px; border-radius: 4px; margin-bottom: 20px; font-weight: bold; }}
        .health-status.ok {{ background-color: #e8f8f5; color: #27ae60; border: 1px solid #2ecc71; }}
        .health-status.error {{ background-color: #fbecec; color: #c0392b; border: 1px solid #e74c3c; }}
        .input-info {{ font-style: italic; color: #7f8c8d; margin-bottom: 15px; }}
    </style>
</head>
<body>
    <div class="container">
        <a href="/docs" style="float:right; text-decoration: none; color:#3498db; font-weight: bold;">API Docs</a>
        <h1>Symptom Recommender System</h1>
        
        <div id="health_check" class="health-status">Checking Model Status...</div>

        <form id="recommender-form">
            <div class="form-group">
                <label for="symptoms">Symptoms (Comma-separated, e.g., ไอ, เจ็บคอ)</label>
                <input type="text" id="symptoms" name="symptoms" value="ปวดท้อง" required>
            </div>
            <div class="form-group">
                <label for="top_n">Top N Recommendations</label>
                <input type="number" id="top_n" name="top_n" value="10" min="1" max="50">
            </div>
            <div class="form-group">
                <label for="age">Age (Optional)</label>
                <input type="number" id="age" name="age" min="0" max="120">
            </div>
            <div class="form-group">
                <label for="gender">Gender (Optional)</label>
                <select id="gender" name="gender">
                    <option value="">N/A</option>
                    <option value="male">male</option>
                    <option value="female">female</option>
                </select>
            </div>
            <div class="form-group">
                <label for="method">Method</label>
                <select id="method" name="method">
                    <option value="hybrid" selected>hybrid (Rules: 70% + Similarity: 30%)</option>
                    <option value="rules">rules (Association Rules)</option>
                    <option value="similarity">similarity (Co-occurrence)</option>
                </select>
            </div>
            <div class="form-group"></div>

            <button type="submit">Get Recommendations</button>
        </form>

        <div id="results" class="results-section">
            <h2>Recommendations</h2>
            <div id="input_display" class="input-info"></div>
            <div id="recommendations-list"></div>
            <div id="message_display" class="message"></div>
            <button id="download-btn" disabled>Download JSON</button>
        </div>
    </div>

    <script>
        const API_URL = window.location.origin;
        let lastSuccessfulPayload = null;
        
        // Health Check on load
        async function checkHealth() {{
            const healthStatusDiv = document.getElementById('health_check');
            try {{
                const response = await fetch(`${{API_URL}}/health`);
                const result = await response.json();
                
                if (result.status === 'ok') {{
                    healthStatusDiv.classList.add('ok');
                    healthStatusDiv.innerHTML = `✅ Model Ready. Found ${{result.total_symptoms}} unique symptoms, ${{result.association_rules}} rules.`;
                }} else {{
                    healthStatusDiv.classList.add('error');
                    healthStatusDiv.innerHTML = `❌ Model Error. ${{(result.error || 'Check server logs for details.')}}`;
                }}
            }} catch (err) {{
                healthStatusDiv.classList.add('error');
                healthStatusDiv.innerHTML = '❌ Network Error: Could not connect to API.';
            }}
        }}
        
        checkHealth();

        // Function to handle recommendation and display
        document.getElementById('recommender-form').onsubmit = async (e) => {{
            e.preventDefault();
            
            const form = e.target;
            const payload = {{
                symptoms: form.symptoms.value,
                age: form.age.value ? parseInt(form.age.value) : null,
                gender: form.gender.value || null,
                top_n: parseInt(form.top_n.value),
                method: form.method.value
            }};
            
            lastSuccessfulPayload = null;
            document.getElementById('download-btn').disabled = true;

            try {{
                const response = await fetch(`${{API_URL}}/recommend`, {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify(payload)
                }});
                
                if (!response.ok) {{
                    const errorData = await response.json();
                    throw new Error(errorData.detail || `HTTP error! Status: ${{response.status}}`);
                }}

                const result = await response.json();
                
                const list = document.getElementById('recommendations-list');
                const inputDisplay = document.getElementById('input_display');
                const messageDisplay = document.getElementById('message_display');

                list.innerHTML = '';
                
                // Update input display
                inputDisplay.innerHTML = `
                    <strong>Input Symptoms:</strong> ${{result.input_symptoms.join(', ') || 'None'}} 
                    | <strong>Patient:</strong> Age: ${{result.patient_info.age}}, Gender: ${{result.patient_info.gender}}
                    | <strong>Method:</strong> ${{form.method.value}}
                `;

                // Update message display
                if (result.message && result.message !== 'Success') {{
                    messageDisplay.style.display = 'block';
                    messageDisplay.textContent = `Message: ${{result.message}}`;
                }} else {{
                    messageDisplay.style.display = 'none';
                }}
                
                // Update recommendations list (No score displayed)
                if (result.recommendations.length === 0) {{
                    list.innerHTML += '<p>No recommendations found.</p>';
                    document.getElementById('download-btn').disabled = true;
                }} else {{
                    result.recommendations.forEach((rec, i) => {{
                        list.innerHTML += `
                            <div class="result-item">
                                <span>${{i+1}}. ${{rec.symptom}}</span>
                            </div>
                        `;
                    }});
                    // Enable download button and store payload
                    lastSuccessfulPayload = payload;
                    document.getElementById('download-btn').disabled = false;
                }}
                
                document.getElementById('results').classList.add('show');
                
            }} catch (err) {{
                console.error(err);
                alert('Request Error: ' + err.message);
                document.getElementById('download-btn').disabled = true;
            }}
        }};
        
        // Function to handle JSON download
        document.getElementById('download-btn').onclick = async () => {{
            if (!lastSuccessfulPayload) return;

            try {{
                // Use a standard POST request to the new /download_json endpoint
                const response = await fetch(`${{API_URL}}/download_json`, {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify(lastSuccessfulPayload)
                }});
                
                if (!response.ok) {{
                    const errorData = await response.json();
                    throw new Error(errorData.detail || `HTTP error! Status: ${{response.status}}`);
                }}
                
                // Get the file content as a blob
                const blob = await response.blob();
                
                // Read filename from header (or use a default)
                const contentDisposition = response.headers.get('Content-Disposition');
                let filename = 'recommendation_data.json';
                if (contentDisposition && contentDisposition.indexOf('filename=') !== -1) {{
                    // Simple regex to extract filename (handles quotes)
                    const match = contentDisposition.match(/filename=["']?([^"']+)["']?$/i);
                    if (match && match[1]) {{
                        filename = match[1];
                    }}
                }}
                
                // Create a temporary link element to trigger the download
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.style.display = 'none';
                a.href = url;
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);

            }} catch (err) {{
                console.error(err);
                alert('Download Error: ' + err.message);
            }}
        }};
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html)


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("STARTING SERVER")
    print("="*60)
    print("🌐 http://localhost:8000")
    print("📚 http://localhost:8000/docs")
    print("🏥 http://localhost:8000/health")
    print("\n" + "="*60 + "\n")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)