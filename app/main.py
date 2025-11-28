# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
from typing import List
import os

app = FastAPI(
    title="Skin Cancer Classification API",
    description="Cilt kanseri sınıflandırması için ML API",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model yükleme
MODEL_PATH = "app/models/cancer_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model yüklendi: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Model yüklenemedi: {e}")
    model = None

# Request model
class PredictionInput(BaseModel):
    features: List[float] = Field(..., min_items=1000, max_items=1000)
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": [0.0] * 1000  # 1000 feature
            }
        }

# Response model
class PredictionOutput(BaseModel):
    prediction: int  # 0 = benign, 1 = malignant
    prediction_label: str  # "benign" veya "malignant"
    confidence: float
    probabilities: dict

@app.get("/")
def root():
    return {
        "message": "Skin Cancer Classification API",
        "status": "running",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model yüklenmedi")
    
    try:
        # Features'ı numpy array'e çevir
        features = np.array(data.features).reshape(1, -1)
        
        # Tahmin yap
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Sonuç
        result = {
            "prediction": int(prediction),
            "prediction_label": "malignant" if prediction == 1 else "benign",
            "confidence": float(max(probabilities)),
            "probabilities": {
                "benign": float(probabilities[0]),
                "malignant": float(probabilities[1])
            }
        }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Tahmin hatası: {str(e)}")

@app.get("/model-info")
def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model yüklenmedi")
    
    return {
        "model_type": type(model).__name__,
        "n_estimators": model.n_estimators if hasattr(model, 'n_estimators') else None,
        "n_features": model.n_features_in_ if hasattr(model, 'n_features_in_') else None,
        "n_classes": model.n_classes_ if hasattr(model, 'n_classes_') else None,
    }