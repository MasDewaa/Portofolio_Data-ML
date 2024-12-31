"""
    The provided code defines a FastAPI endpoint for a Convolutional Neural Network (CNN) image
    classifier using a pre-trained ResNet18 model in PyTorch for predicting classes of images.
    :return: The endpoint `/predict` returns a JSON response containing the predicted class name and
    confidence score for the input image. The response structure is as follows:
"""
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
import torch
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import ResNet18_Weights
from PIL import Image
import json
import io

router = APIRouter()

# Path ke model dan label
model_path = "models/stanford_dogs_model_ResNet18.pth"
label_path = "models/class_labels.json"

# Inisialisasi model sesuai arsitektur
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, 120)  # Output untuk 120 class

# Load state_dict
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

# Remove 'model.' prefix if present
checkpoint = {k.replace('model.', ''): v for k, v in checkpoint.items()}

# Load state_dict ke model
model.load_state_dict(checkpoint)
model.eval()

# Load label kelas
with open(label_path, "r") as f:
    class_labels = json.load(f)

# Transformasi gambar
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Sesuaikan ukuran input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@router.get("/")
def read_root():
    return {"message": "Welcome to the CNN Classifier API using PyTorch!"}

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Membaca file gambar
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = transform(image).unsqueeze(0)  # Tambahkan batch dimension

        # Prediksi menggunakan model
        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()

        # Ambil nama kelas
        class_name = class_labels[str(predicted_class)]

        return {
            "class": class_name,
            "confidence": float(confidence)
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
