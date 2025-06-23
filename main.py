from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
import numpy as np
import os
import requests
import base64
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://skiniq-frontend.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# --- Download & load 3 models ---
MODELS = {}
CLASS_MAP = {}

MODEL_CONFIGS = {
    "skin_disease": {
        "url": "https://huggingface.co/Armand345/skiniq-model/resolve/main/skin_model.pth",
        "path": "skin_model.pth",
        "classes": [
            "Melanocytic nevi", "Melanoma", "Benign keratosis",
            "Basal cell carcinoma", "Actinic keratoses", "Vascular lesions", "Dermatofibroma"
        ]
    },
    "acne": {
        "url": "https://huggingface.co/Armand345/skiniq-model/resolve/main/acne_model.pth",
        "path": "acne_model.pth",
        "classes": ["No acne", "Mild acne", "Moderate acne", "Severe acne"]
    },
    "pigmentation": {
        "url": "https://huggingface.co/Armand345/skiniq-model/resolve/main/pigmentation_model.pth",
        "path": "pigmentation_model.pth",
        "classes": ["No pigmentation", "Melasma", "Hyperpigmentation", "Hypopigmentation"]

import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import ImageUpload from '../components/ImageUpload';
import { supabase } from '../supabaseClient';

const ScanPage = () => {
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    supabase.auth.getUser()
      .then(({ data: { user } }) => { if (!user) navigate('/login'); });
  }, [navigate]);

  const handleScan = async () => {
    if (!image) return;
    setLoading(true);

    const formData = new FormData();
    formData.append("file", image);

    try {
      console.log("🔍 Sending scan request...");
      const response = await fetch("https://skiniq-backend-ej69.onrender.com/analyze-skin", {
        method: "POST",
        body: formData,
      });
      console.log("👉 Response received:", response);

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status} – ${errorText}`);
      }

      const result = await response.json();
      console.log("✅ Scan result:", result);
      localStorage.setItem("scanResult", JSON.stringify(result));
      navigate('/result');
    } catch (err) {
      console.error("❌ Scan failed:", err);
      alert(`Something went wrong: ${err.message}`);
    } finally {
      setLoading(false);
    }
}

def load_model(config):
    if not os.path.exists(config["path"]):
        r = requests.get(config["url"])
        with open(config["path"], "wb") as f:
            f.write(r.content)
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(config["classes"]))
    model.load_state_dict(torch.load(config["path"], map_location="cpu"))
    model.eval()
    return model

for name, config in MODEL_CONFIGS.items():
    MODELS[name] = load_model(config)
    CLASS_MAP[name] = config["classes"]

@app.on_event("startup")
async def init():
    time.sleep(2)
    print("✅ All models loaded and backend is ready.")

@app.post("/analyze-skin")
async def analyze_skin(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        tensor = transform(image).unsqueeze(0)

        best_result = None

        for model_name, model in MODELS.items():
            with torch.no_grad():
                output = model(tensor)
                probs = torch.nn.functional.softmax(output[0], dim=0)
                top_prob, top_class = torch.max(probs, 0)
                class_name = CLASS_MAP[model_name][top_class.item()]

                result = {
                    "model": model_name,
                    "condition": class_name,
                    "confidence": float(top_prob.item()),
                    "recommendation": "Please consult a dermatologist for confirmation."
                }

                if best_result is None or result["confidence"] > best_result["confidence"]:
                    best_result = result

        return best_result

    except Exception as e:
        print("❌ Backend error:", str(e))
        return {"error": str(e)}

@app.get("/")
def root():
    return {"status": "OK"}

  };

  return (
    <div style={{ fontFamily: 'Segoe UI', padding: '2rem', textAlign: 'center', maxWidth: '600px', margin: 'auto' }}>
      <h1 style={{ fontSize: '2.5rem', color: '#006E3C' }}>AI‑Powered Skin Scan</h1>
      <ImageUpload image={image} setImage={setImage} />
      <button
        onClick={handleScan}
        disabled={!image || loading}
        style={{
          marginTop: '2rem',
          padding: '1rem 2rem',
          fontSize: '1rem',
          background: '#28a745',
          color: 'white',
          border: 'none',
          borderRadius: '5px',
          cursor: image && !loading ? 'pointer' : 'not-allowed',
          opacity: image && !loading ? 1 : 0.6
        }}
      >
        {loading ? 'Analyzing…' : 'Analyze'}
      </button>
      <p style={{ marginTop: '2rem', color: '#777' }}>📸 Tip: good lighting = better results.</p>
    </div>
  );
};

export default ScanPage;
