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
      allow_origins=["*"],
      allow_credentials=True,
      allow_methods=["*"],
      allow_headers=["*"],
)

CLASS_NAMES = [
    "Melanocytic nevi", "Melanoma", "Benign keratosis",
    "Basal cell carcinoma", "Actinic keratoses", "Vascular lesions", "Dermatofibroma"
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

MODEL_URL = "https://huggingface.co/Armand345/skiniq-model/resolve/main/skin_model.pth"
MODEL_PATH = "skin_model.pth"

if not os.path.exists(MODEL_PATH):
    print("📥 Downloading skin model...")
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

def generate_gradcam(image_tensor, model, target_class):
    model.zero_grad()
    features = []
    grads = []

    def forward_hook(module, input, output):
        features.append(output.detach())

    def backward_hook(module, grad_in, grad_out):
        grads.append(grad_out[0].detach())

    final_conv = model.layer4[1].conv2
    handle_f = final_conv.register_forward_hook(forward_hook)
    handle_b = final_conv.register_backward_hook(backward_hook)

    output = model(image_tensor)
    one_hot = torch.zeros((1, output.size()[-1]))
    one_hot[0][target_class] = 1
    output.backward(gradient=one_hot)

    gradients = grads[0][0]
    activations = features[0][0]
    weights = torch.mean(gradients, dim=(1, 2))
    cam = torch.zeros(activations.shape[1:], dtype=torch.float32)

    for i, w in enumerate(weights):
        cam += w * activations[i, :, :]

    cam = np.maximum(cam.numpy(), 0)
    cam = cam / cam.max()
    cam = np.uint8(255 * cam)
    cam = Image.fromarray(cam).resize((224, 224))

    handle_f.remove()
    handle_b.remove()
    return cam

@app.on_event("startup")
async def startup_event():
    time.sleep(2)
    print("✅ Backend is ready.")

@app.post("/analyze-skin")
async def analyze_skin(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(tensor)
            probs = torch.nn.functional.softmax(output[0], dim=0)
            top_prob, top_class = torch.max(probs, 0)

        cam = generate_gradcam(tensor, model, top_class.item())
        cam = cam.convert("RGBA")
        orig = image.resize((224, 224)).convert("RGBA")
        heatmap = Image.blend(orig, cam, alpha=0.5)

        buffered = io.BytesIO()
        heatmap.save(buffered, format="PNG")
        cam_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {
            "condition": CLASS_NAMES[top_class.item()],
            "confidence": float(top_prob.item()),
            "recommendation": "Please consult a dermatologist for confirmation.",
            "gradcam": cam_base64
        }

    except Exception as e:
        print("❌ Backend error:", str(e))
        return {"error": str(e)}

@app.get("/")
def root():
    return {"status": "OK"}
