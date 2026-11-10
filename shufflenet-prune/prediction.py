"""
predict.py — Prédiction sur food-11/test/ avec ShuffleNet entraîné.
Produit un fichier predictions.json au format { "0": "bread", ... }
"""

import json
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

# ============================================================
# 0. CONFIGURATION
# ============================================================
TEST_DIR       = Path("food-11/test")
MODEL_PATH     = "shufflenet_food11_best.pt"
OUTPUT_JSON    = "predictions.json"

# Classes Food-11 dans l'ordre alphabétique (ordre de ImageFolder)
# Vérifié par rapport à la structure habituelle de Food-11
FOOD11_CLASSES = [
    "bread",
    "dairy",
    "dessert",
    "egg",
    "fried",
    "meat",
    "noodles-pasta",
    "rice",
    "seafood",
    "soup",
    "vegetable-fruit",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 1. TRANSFORM (identique à celle de validation)
# ============================================================
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ============================================================
# 2. CHARGEMENT DU MODÈLE
# ============================================================
model = models.shufflenet_v2_x1_5(num_classes=11)
model.fc = nn.Linear(model.fc.in_features, 11)

state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(state_dict)
model = model.to(DEVICE)
model.eval()

print(f"✓ Modèle chargé depuis {MODEL_PATH}")

# ============================================================
# 3. PRÉDICTIONS
# ============================================================
predictions = {}

# Lister et trier numériquement les images (0.jpg, 1.jpg, 2.jpg, …)
image_paths = sorted(
    TEST_DIR.glob("*.jpg"),
    key=lambda p: int(p.stem)   # tri par numéro, pas par chaîne
)

print(f"Images trouvées dans {TEST_DIR} : {len(image_paths)}")

for img_path in image_paths:
    img_name = img_path.stem          # "0", "1", "2", ...

    # Charger + transformer
    image = Image.open(img_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)  # (1, 3, 224, 224)

    # Inférence
    with torch.no_grad():
        output = model(tensor)
        pred_idx = output.argmax(dim=1).item()

    predictions[img_name] = FOOD11_CLASSES[pred_idx]

# ============================================================
# 4. SAUVEGARDE JSON
# ============================================================
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(predictions, f, indent=4, ensure_ascii=False)

print(f"✓ Prédictions sauvegardées dans {OUTPUT_JSON}")
