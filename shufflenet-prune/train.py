import copy
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

# ============================================================
# 0.  CONFIGURATION
# ============================================================
DATA_DIR = Path("food-11/train")          # <-- ton dossier
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-3
VAL_SPLIT = 0.2                           # 20 % pour la validation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 1.  TRANSFORMS (statistiques ImageNet car modèle pré-entraîné)
# ============================================================
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ============================================================
# 4.  COMPTAGE DES PARAMÈTRES
# ============================================================
def count_params(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters()
               if (p.requires_grad or not trainable_only))


# ============================================================
# 5.  BOUCLES TRAIN / EVAL
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    # Backbone gelé : on fige aussi les statistiques des BatchNorm
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (outputs.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        outputs = model(x)
        loss = criterion(outputs, y)
        total_loss += loss.item() * x.size(0)
        correct += (outputs.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


def main():
    # ============================================================
    # 2.  DATASET  (train uniquement → split manuel)
    # ============================================================
    # Deux ImageFolder distincts pour que chaque split ait son propre transform
    train_base = datasets.ImageFolder(DATA_DIR, transform=train_transform)
    val_base = datasets.ImageFolder(DATA_DIR, transform=val_transform)

    print(f"Classes détectées ({len(train_base.classes)}) :")
    for idx, name in enumerate(train_base.classes):
        print(f"  {idx} → {name}")
    print(f"Images totales : {len(train_base)}\n")

    # Découpage reproductible
    n_val = int(VAL_SPLIT * len(train_base))
    perm = torch.randperm(
        len(train_base), generator=torch.Generator().manual_seed(42)
    ).tolist()
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_ds = Subset(train_base, train_idx)
    val_ds = Subset(val_base, val_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=4, pin_memory=True)

    # ============================================================
    # 3.  MODÈLE : ShuffleNet V2 x0.5  (la plus petite variante)
    # ============================================================
    model = models.shufflenet_v2_x1_5(
        weights=models.ShuffleNet_V2_X1_5_Weights.IMAGENET1K_V1
    )
    num_classes = len(train_base.classes)   # 11 pour Food-11
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # ---------- RÉDUCTION N°1 : geler tout le backbone ----------
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False

    model = model.to(DEVICE)

    print(f"Paramètres totaux           : {count_params(model):,}")
    print(f"Paramètres entraînables     : {count_params(model, trainable_only=True):,}\n")

    # ============================================================
    # 6.  ENTRAÎNEMENT
    # ============================================================
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0
    best_weights = None

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        scheduler.step()

        print(f"[{epoch:02d}/{EPOCHS}] "
              f"train loss {train_loss:.4f}  acc {train_acc:.4f}  |  "
              f"val loss {val_loss:.4f}  acc {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    torch.save(model.state_dict(), "shufflenet_food11_best.pt")
    print(f"\n✓ Meilleure accuracy validation : {best_acc:.4f}")


if __name__ == "__main__":
    main()
