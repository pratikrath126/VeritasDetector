import torch
import timm
import numpy as np
from PIL import Image
import io

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224
MODEL_PATH = "model.pth"

_model = None
_device = None

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def load_model():
    """Load model once and cache it globally"""
    global _model, _device

    _device = get_device()
    print(f"Loading model on device: {_device}")

    _model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2)

    checkpoint = torch.load(MODEL_PATH, map_location=_device)
    _model.load_state_dict(checkpoint['model_state_dict'])
    _model = _model.to(_device)
    _model.eval()

    val_acc = checkpoint.get('val_accuracy', None)
    if isinstance(val_acc, (int, float)):
        print(f"Model loaded successfully (Training accuracy: {val_acc:.2f}%)")
    else:
        print("Model loaded successfully")
    return _model

def preprocess_image(image_bytes):
    """Convert image bytes to model input tensor"""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    img_np = np.array(img, dtype=np.float32) / 255.0

    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    img_np = (img_np - mean) / std

    img_tensor = torch.FloatTensor(img_np).permute(2, 0, 1).unsqueeze(0)
    return img_tensor

def predict_image(image_bytes):
    """
    Run deepfake detection on image bytes.
    Returns: {"label": "Real"|"Fake", "confidence": float, "scores": dict}
    """
    global _model, _device

    if _model is None:
        load_model()

    img_tensor = preprocess_image(image_bytes).to(_device)

    with torch.no_grad():
        outputs = _model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)

        real_prob = probabilities[0][0].item()
        fake_prob = probabilities[0][1].item()

        predicted_class = torch.argmax(probabilities, dim=1).item()

    label = "Fake" if predicted_class == 1 else "Real"
    confidence = fake_prob if label == "Fake" else real_prob

    return {
        "label": label,
        "confidence": round(confidence * 100, 2),
        "scores": {
            "real": round(real_prob * 100, 2),
            "fake": round(fake_prob * 100, 2)
        }
    }
