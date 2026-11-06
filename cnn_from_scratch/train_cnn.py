"""
Food-11 — entraînement d'un petit CNN résiduel à partir de zéro (aucun poids pré-entraîné).

Lancement (depuis le dossier du projet) :
    .venv\\Scripts\\python.exe train_cnn.py

Sorties (dans le dossier runs/) :
    best_model.pt      poids du modèle ayant le meilleur F1 macro sur la validation
    log.csv            pour chaque époque : perte d'entraînement, perte de validation,
                       exactitude, F1 macro (sert à tracer les courbes du rapport)
    confusion.csv      matrice de confusion de la validation (meilleure époque)
    summary.txt        nombre de paramètres, meilleur F1, score de frugalité, F1 par classe
"""

import os
# Sous Windows, chaque processus de lecture d'images recharge les bibliothèques de calcul ;
# limiter OpenBLAS à 1 thread évite l'erreur « OpenBLAS error: Memory allocation still failed ».
# Ces lignes doivent venir AVANT l'import de torch / numpy.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import csv
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ============================================================
# 0. Configuration : tout ce qu'on peut régler est ici.
#    Pour une expérience, on ne change qu'un seul paramètre à la fois.
# ============================================================
DATA_DIR = Path(r"E:\1TC\food-11\food-11\train")  # jeu d'entraînement : 11 dossiers de classes
OUT_DIR = Path("runs")
IMG_SIZE = 128          # côté des images après redimensionnement
BATCH_SIZE = 64
EPOCHS = 40
LR = 2e-3               # taux d'apprentissage maximal (le planificateur OneCycle monte puis descend)
WEIGHT_DECAY = 0.05     # décroissance des poids : régularisation contre le surapprentissage
LABEL_SMOOTHING = 0.1   # lissage des étiquettes
VAL_RATIO = 0.1         # 10 % du jeu d'entraînement servent de validation
NUM_WORKERS = 0         # 0 = lecture des images dans le processus principal (plus lent mais sûr sous Windows :
                        # avec 2 ou 4 processus, la mémoire virtuelle du portable était épuisée -> erreur OpenBLAS)
SEED = 42               # graine aléatoire fixée pour des résultats reproductibles
WIDTHS = (32, 64, 96, 160)  # nombre de canaux des 4 étages ; ~1,33 M de paramètres.
                            # Plus large = plus précis, mais la frugalité pénalise davantage.


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 1. Données : lecture des dossiers -> séparation entraînement/validation -> augmentation
# ============================================================
# Moyenne et écart-type par canal (statistiques d'ImageNet) utilisés pour normaliser.
# Ce ne sont que six nombres, pas des poids pré-entraînés.
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0)),  # recadrage aléatoire : simule distance et cadrage
    transforms.RandomHorizontalFlip(),                         # symétrie horizontale : un plat retourné reste le même plat
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),               # luminosité/contraste/saturation/teinte : simule l'éclairage
    transforms.ToTensor(),                                     # image -> tenseur [3, H, W], valeurs entre 0 et 1
    transforms.Normalize(MEAN, STD),                           # soustraire la moyenne, diviser par l'écart-type
])
val_tf = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.14)),   # en validation, aucune transformation aléatoire : redimensionner
    transforms.CenterCrop(IMG_SIZE),           # puis garder le centre
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def stratified_split(targets, val_ratio, seed):
    """Séparation stratifiée : chaque classe est coupée dans la même proportion,
    pour que la validation ait la même répartition des 11 classes que l'entraînement."""
    rng = random.Random(seed)
    by_class = {}
    for idx, y in enumerate(targets):
        by_class.setdefault(y, []).append(idx)
    train_idx, val_idx = [], []
    for y, idxs in by_class.items():
        rng.shuffle(idxs)
        n_val = max(1, int(len(idxs) * val_ratio))
        val_idx += idxs[:n_val]
        train_idx += idxs[n_val:]
    return train_idx, val_idx


# ============================================================
# 2. Modèle : petit réseau résiduel
# ============================================================
class ResBlock(nn.Module):
    """Bloc résiduel : deux convolutions 3x3, puis on rajoute l'entrée par un « raccourci »
    (y = F(x) + x). Le raccourci laisse passer le gradient directement vers l'arrière,
    ce qui permet d'entraîner des réseaux plus profonds."""

    def __init__(self, c_in, c_out, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(inplace=True)
        # Si la forme change (nombre de canaux ou taille divisée par 2),
        # une convolution 1x1 sur le raccourci aligne les dimensions.
        self.shortcut = nn.Identity()
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride, bias=False),
                nn.BatchNorm2d(c_out),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class SmallResNet(nn.Module):
    # Formes indiquées pour widths=(32, 64, 96, 160) et des images 128x128 ; B = taille du lot.
    def __init__(self, num_classes=11, widths=(32, 64, 96, 160), dropout=0.2):
        super().__init__()
        # entrée [B, 3, 128, 128]
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], 3, 2, 1, bias=False),   # -> [B, 32, 64, 64]
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
        )
        layers = []
        c_in = widths[0]
        for i, w in enumerate(widths):
            stride = 1 if i == 0 else 2                      # à chaque nouvel étage, la taille est divisée par 2
            layers.append(ResBlock(c_in, w, stride))
            layers.append(ResBlock(w, w, 1))
            c_in = w
        self.features = nn.Sequential(*layers)               # -> [B, 160, 8, 8]
        self.pool = nn.AdaptiveAvgPool2d(1)                  # -> [B, 160, 1, 1] moyenne de chaque canal
        self.dropout = nn.Dropout(dropout)                   # pendant l'entraînement, éteint des neurones au hasard
        self.fc = nn.Linear(c_in, num_classes)               # -> [B, 11] : ce sont les logits

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x).flatten(1)                          # [B, 160]
        return self.fc(self.dropout(x))                      # pas de softmax : l'entropie croisée l'inclut déjà


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ============================================================
# 3. Évaluation : perte, exactitude, F1 macro
# ============================================================
def macro_f1(confusion):
    """confusion[i][j] = nombre d'images de vraie classe i prédites j.
    Renvoie (F1 moyen sur les classes, liste des F1 par classe)."""
    k = confusion.shape[0]
    f1s = []
    for c in range(k):
        tp = confusion[c, c].item()
        fp = confusion[:, c].sum().item() - tp   # prédit c alors que ce n'est pas c (fausse alarme)
        fn = confusion[c, :].sum().item() - tp   # c'est c mais non prédit c (oubli)
        p = tp / (tp + fp) if tp + fp > 0 else 0.0
        r = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1s.append(2 * p * r / (p + r) if p + r > 0 else 0.0)
    return sum(f1s) / k, f1s


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, num_classes):
    model.eval()
    total_loss, n = 0.0, 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(x)
            loss = loss_fn(logits, y)
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
        pred = logits.argmax(1)
        for t, p in zip(y.cpu(), pred.cpu()):
            confusion[t, p] += 1
    f1, per_class = macro_f1(confusion)
    acc = confusion.diag().sum().item() / n
    return total_loss / n, acc, f1, per_class, confusion


def frugality(f1, n_params, w=0.3):
    r = n_params / 1e6
    return f1 - w / (1 + 1 / r)


# ============================================================
# 4. Boucle d'entraînement principale
# ============================================================
def main():
    set_seed(SEED)
    OUT_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Appareil : {device}", torch.cuda.get_device_name(0) if device.type == "cuda" else "")

    # On lit deux fois le même dossier : une copie avec l'augmentation d'entraînement,
    # une avec la transformation de validation, puis on coupe avec les mêmes indices.
    full_train = datasets.ImageFolder(DATA_DIR, transform=train_tf)
    full_val = datasets.ImageFolder(DATA_DIR, transform=val_tf)
    classes = full_train.classes
    train_idx, val_idx = stratified_split(full_train.targets, VAL_RATIO, SEED)
    train_set = Subset(full_train, train_idx)
    val_set = Subset(full_val, val_idx)
    print(f"Classes ({len(classes)}) : {classes}")
    print(f"Entraînement : {len(train_set)} images, validation : {len(val_set)} images")

    train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=True, drop_last=True, persistent_workers=NUM_WORKERS > 0)
    val_loader = DataLoader(val_set, BATCH_SIZE * 2, shuffle=False, num_workers=NUM_WORKERS,
                            pin_memory=True, persistent_workers=NUM_WORKERS > 0)

    model = SmallResNet(num_classes=len(classes), widths=WIDTHS).to(device)
    n_params = count_params(model)
    print(f"Nombre de paramètres : {n_params:,} ({n_params/1e6:.2f} M)")

    loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, epochs=EPOCHS, steps_per_epoch=len(train_loader))
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")  # précision mixte : moins de mémoire, plus rapide

    # Vérification : à l'initialisation, la perte doit être proche de ln(11) ≈ 2,398
    model.eval()
    with torch.no_grad():
        x0, y0 = next(iter(val_loader))
        init_loss = nn.functional.cross_entropy(model(x0.to(device)), y0.to(device)).item()
    print(f"Perte initiale = {init_loss:.3f} (valeur théorique ln({len(classes)}) = {math.log(len(classes)):.3f})")

    best_f1 = -1.0
    with open(OUT_DIR / "log.csv", "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1", "lr", "seconds"])

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        running, seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(x)              # propagation avant : [B, 11]
                loss = loss_fn(logits, y)      # entropie croisée
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()      # rétropropagation : calcul des gradients
            scaler.step(optimizer)             # descente de gradient : mise à jour des paramètres
            scaler.update()
            scheduler.step()                   # le taux d'apprentissage avance d'un pas sur la courbe OneCycle
            running += loss.item() * x.size(0)
            seen += x.size(0)

        train_loss = running / seen
        val_loss, val_acc, val_f1, per_class, confusion = evaluate(model, val_loader, loss_fn, device, len(classes))
        secs = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]
        mark = ""
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({"model": model.state_dict(), "classes": classes, "img_size": IMG_SIZE, "widths": WIDTHS},
                       OUT_DIR / "best_model.pt")
            best_per_class = per_class
            with open(OUT_DIR / "confusion.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["true\\pred"] + classes)
                for c, row in zip(classes, confusion.tolist()):
                    w.writerow([c] + row)
            mark = "  * meilleur, sauvegardé"
        print(f"époque {epoch:2d}/{EPOCHS} | train {train_loss:.3f} | val {val_loss:.3f} "
              f"| acc {val_acc:.3f} | F1 {val_f1:.3f} | {secs:.0f}s{mark}")
        with open(OUT_DIR / "log.csv", "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.4f}", f"{val_loss:.4f}", f"{val_acc:.4f}",
                                    f"{val_f1:.4f}", f"{lr_now:.6f}", f"{secs:.1f}"])

    frug = frugality(best_f1, n_params)
    lines = [
        f"Nombre de paramètres : {n_params:,} ({n_params/1e6:.3f} M)",
        f"Meilleur F1 macro (validation) : {best_f1:.4f}",
        f"Frugalité = F1 - 0,3/(1+1/R) = {frug:.4f}",
        "F1 par classe :",
    ] + [f"  {c:16s} {s:.3f}" for c, s in zip(classes, best_per_class)]
    text = "\n".join(lines)
    print("\n" + text)
    (OUT_DIR / "summary.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":   # indispensable sous Windows pour la lecture multi-processus
    main()
