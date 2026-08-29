import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

from class_names import CLASS_NAMES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENCODER_PATH = "models/encoder.pth"
CLASSIFIER_PATH = "models/linear_classifier.pth"


class BarlowTwinsEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = models.resnet18(weights=None)
        self.encoder.fc = nn.Identity()

    def forward(self, x):
        return self.encoder(x)


class LinearClassifier(nn.Module):
    def __init__(self, feature_dim=512, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


encoder = BarlowTwinsEncoder()
classifier = LinearClassifier(feature_dim=512, num_classes=len(CLASS_NAMES))

# --- LOAD ENCODER (raw resnet keys, no "encoder." prefix) ---
encoder_state = torch.load(ENCODER_PATH, map_location=DEVICE)
_ = encoder.encoder.load_state_dict(encoder_state)

# --- LOAD CLASSIFIER (combined checkpoint, extract fc.* only) ---
classifier_full_state = torch.load(CLASSIFIER_PATH, map_location=DEVICE)
fc_state = {
    k.replace("fc.", "", 1): v
    for k, v in classifier_full_state.items()
    if k.startswith("fc.")
}
_ = classifier.fc.load_state_dict(fc_state)

print("Encoder and classifier loaded successfully")

encoder = encoder.to(DEVICE)
classifier = classifier.to(DEVICE)
encoder.eval()
classifier.eval()

transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4467, 0.4398, 0.4066),
        std=(0.2603, 0.2566, 0.2713)
    )
])


def predict(image):
    if image is None:
        return {}, "Please upload an image."

    image = image.convert("RGB")
    x = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        features = encoder(x)
        logits = classifier(features)
        probabilities = torch.softmax(logits, dim=1)[0]

    result = {CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))}

    predicted_index = torch.argmax(probabilities).item()
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = probabilities[predicted_index].item() * 100

    message = f"Prediction: {predicted_class}\nConfidence: {confidence:.2f}%"
    return result, message


print("INFERENCE.PY FINISHED LOADING")