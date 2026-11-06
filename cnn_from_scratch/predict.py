"""
Prédiction sur le jeu de test avec le meilleur modèle, et écriture du fichier JSON à soumettre.

Lancement :
    .venv\\Scripts\\python.exe predict.py

Sortie :
    runs/predictions.json   format {"0": "bread", "1": "fried", ...}
"""

import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from train_cnn import MEAN, STD, SmallResNet

TEST_DIR = Path(r"E:\1TC\food-11\food-11\test")
CKPT = Path("runs/best_model.pt")
OUT = Path("runs/predictions.json")


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(CKPT, map_location=device)
    classes, img_size = ckpt["classes"], ckpt["img_size"]
    model = SmallResNet(num_classes=len(classes), widths=ckpt["widths"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    files = sorted(TEST_DIR.glob("*.jpg"), key=lambda p: int(p.stem))   # 0.jpg, 1.jpg, 2.jpg ...
    preds = {}
    for i in range(0, len(files), 128):
        batch_files = files[i:i + 128]
        x = torch.stack([tf(Image.open(p).convert("RGB")) for p in batch_files]).to(device)
        # Augmentation au moment du test (TTA) : image originale + image retournée, on moyenne les probabilités
        probs = model(x).softmax(1) + model(torch.flip(x, dims=[3])).softmax(1)
        for p, k in zip(batch_files, probs.argmax(1).tolist()):
            preds[p.stem] = classes[k]          # indice -> nom du dossier de la classe

    OUT.write_text(json.dumps(preds, indent=4), encoding="utf-8")
    print(f"{len(preds)} images prédites, résultat écrit dans {OUT}")


if __name__ == "__main__":
    main()
