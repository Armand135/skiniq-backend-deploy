from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
import requests
import os

app = FastAPI()

# Allow CORS for frontend calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Download model from Hugging Face if not exists
MODEL_URL = "https://huggingface.co/YOUR_USERNAME/skiniq-model/resolve/main/skin_model.pth"
MODEL_PATH = "skin_model.pth"

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Hugging Face...")
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)

# Load model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 7)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

CLASS_NAMES = [
    "Melanocytic nevi", "Melanoma", "Benign keratosis",
    "Basal cell carcinoma", "Actinic keratoses", "Vascular lesions", "Dermatofibroma"
]

@app.get("/")
def read_root():
    return {"message": "SkinIQ Inference API is running."}

@app.post("/analyze-skin/")
async def analyze_skin(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        top_prob, top_class = torch.max(probs, 0)

    return {
        "condition": CLASS_NAMES[top_class.item()],
        "confidence": float(top_prob.item()),
        "recommendation": "Please consult a dermatologist for confirmation."
    }