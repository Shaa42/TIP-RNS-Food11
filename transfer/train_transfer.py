"""
Food-11 - transfer learning with a *small* ImageNet-pretrained network, tuned for the Frugality score.

Why this exists
---------------
Frug = F1 - 0.3 / (1 + 1/R), with R = number of parameters in millions.
The penalty is steep for small R (0.35 M -> 0.078, 1.3 M -> 0.171, 4 M -> 0.24, 11 M -> 0.275), so the best
trade-off is a very small network whose features are already good. Starting from ImageNet weights lets a
~1.3 M-parameter network reach a much higher F1 than the same size trained from scratch (0.847 on validation).

Usage (from the project folder):
    uv run python transfer/train_transfer.py --data-dir path/to/food-11/train --arch shufflenet_v2_x1_0
    uv run python transfer/train_transfer.py --data-dir ... --arch shufflenet_v2_x0_5
    uv run python transfer/train_transfer.py --data-dir ... --arch mobilenet_v3_small
Add --test-dir path/to/food-11/test to also write the JSON of predictions (runs_<arch>/predictions.json).
Compare the "Frugality" line printed at the end of each run and submit the best one only once.

Outputs (runs_<arch>/): best_model.pt, log.csv, confusion.csv, summary.txt, predictions.json (optional).
"""

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

MEAN = [0.485, 0.456, 0.406]   # ImageNet statistics: must match the pretrained weights
STD = [0.229, 0.224, 0.225]


# ------------------------------------------------------------------ helpers
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stratified_split(targets, val_ratio, seed):
    """Same per-class split as the from-scratch CNN, so the validation scores are comparable."""
    rng = random.Random(seed)
    by_class = {}
    for idx, y in enumerate(targets):
        by_class.setdefault(y, []).append(idx)
    train_idx, val_idx = [], []
    for idxs in by_class.values():
        rng.shuffle(idxs)
        n_val = max(1, int(len(idxs) * val_ratio))
        val_idx += idxs[:n_val]
        train_idx += idxs[n_val:]
    return train_idx, val_idx


def macro_f1(confusion):
    """confusion[i][j] = images of true class i predicted j. Returns (macro F1, per-class F1)."""
    f1s = []
    for c in range(confusion.shape[0]):
        tp = confusion[c, c].item()
        fp = confusion[:, c].sum().item() - tp
        fn = confusion[c, :].sum().item() - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return sum(f1s) / len(f1s), f1s


def frugality(f1, n_params, w=0.3):
    return f1 - w / (1 + 1e6 / n_params)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def build_model(arch, num_classes, pretrained):
    """Load a torchvision network (ImageNet weights if pretrained) and replace its last layer."""
    model = models.get_model(arch, weights="DEFAULT" if pretrained else None)
    if hasattr(model, "fc"):                                   # ShuffleNetV2, ResNet
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        head = model.fc
    else:                                                      # MobileNetV2/V3: classifier = [..., Linear]
        last = len(model.classifier) - 1
        model.classifier[last] = nn.Linear(model.classifier[last].in_features, num_classes)
        head = model.classifier[last]
    return model, head


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, num_classes):
    model.eval()
    total, n = 0.0, 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(x)
        total += loss_fn(logits.float(), y).item() * x.size(0)
        n += x.size(0)
        for t, p in zip(y.cpu(), logits.argmax(1).cpu()):
            confusion[t, p] += 1
    f1, per_class = macro_f1(confusion)
    return total / n, confusion.diag().sum().item() / n, f1, per_class, confusion


@torch.no_grad()
def predict_test(model, classes, img_size, test_dir, out_path, device):
    """Predictions on the unlabeled test set, with flip test-time augmentation -> JSON {"0": "bread", ...}."""
    tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)), transforms.CenterCrop(img_size),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    files = sorted(Path(test_dir).glob("*.jpg"), key=lambda p: int(p.stem))
    model.eval()
    preds = {}
    for i in range(0, len(files), 128):
        batch = files[i:i + 128]
        x = torch.stack([tf(Image.open(p).convert("RGB")) for p in batch]).to(device)
        probs = model(x).softmax(1) + model(torch.flip(x, dims=[3])).softmax(1)
        for p, k in zip(batch, probs.argmax(1).tolist()):
            preds[p.stem] = classes[k]
    Path(out_path).write_text(json.dumps(preds, indent=4), encoding="utf-8")
    print(f"{len(preds)} test images -> {out_path}")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True, help="train folder: 11 class sub-folders")
    ap.add_argument("--test-dir", type=Path, default=None, help="test folder; if given, writes predictions.json")
    ap.add_argument("--arch", default="shufflenet_v2_x1_0",
                    choices=["shufflenet_v2_x0_5", "shufflenet_v2_x1_0", "mobilenet_v3_small", "mobilenet_v2"])
    ap.add_argument("--no-pretrained", action="store_true", help="random init (smoke tests only)")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3, help="peak LR of the backbone (head gets 5x)")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=0 if os.name == "nt" else 4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    out = args.out_dir or Path(f"runs_{args.arch}")
    out.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    S = args.img_size

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(S, scale=(0.4, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(),                       # light, parameter-free strong augmentation
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.2),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(S * 1.14)), transforms.CenterCrop(S),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    full_train = datasets.ImageFolder(args.data_dir, transform=train_tf)
    full_val = datasets.ImageFolder(args.data_dir, transform=val_tf)
    classes = full_train.classes
    tr_idx, va_idx = stratified_split(full_train.targets, args.val_ratio, args.seed)
    train_loader = DataLoader(Subset(full_train, tr_idx), args.batch_size, shuffle=True, drop_last=True,
                              num_workers=args.workers, pin_memory=device.type == "cuda",
                              persistent_workers=args.workers > 0)
    val_loader = DataLoader(Subset(full_val, va_idx), args.batch_size * 2, shuffle=False,
                            num_workers=args.workers, persistent_workers=args.workers > 0)
    print(f"{len(classes)} classes | train {len(tr_idx)} | val {len(va_idx)} | device {device}")

    model, head = build_model(args.arch, len(classes), pretrained=not args.no_pretrained)
    model = model.to(device)
    n_params = count_params(model)
    print(f"{args.arch}: {n_params:,} parameters ({n_params / 1e6:.3f} M) | "
          f"frugality penalty = {0.3 / (1 + 1e6 / n_params):.3f}")

    head_ids = {id(p) for p in head.parameters()}
    groups = [
        {"params": [p for p in model.parameters() if id(p) not in head_ids], "lr": args.lr},
        {"params": list(head.parameters()), "lr": args.lr * 5},
    ]
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[args.lr, args.lr * 5], epochs=args.epochs,
        steps_per_epoch=len(train_loader), pct_start=0.2)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    with open(out / "log.csv", "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1", "seconds"])

    best_f1, best_per_class = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running, seen = time.time(), 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = loss_fn(model(x).float(), y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item() * x.size(0)
            seen += x.size(0)
        val_loss, val_acc, val_f1, per_class, confusion = evaluate(model, val_loader, loss_fn, device, len(classes))
        mark = ""
        if val_f1 > best_f1:
            best_f1, best_per_class = val_f1, per_class
            torch.save({"model": model.state_dict(), "classes": classes, "img_size": S, "arch": args.arch},
                       out / "best_model.pt")
            with open(out / "confusion.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["true\\pred"] + classes)
                for c, row in zip(classes, confusion.tolist()):
                    w.writerow([c] + row)
            mark = "  * best"
        secs = time.time() - t0
        print(f"epoch {epoch:2d}/{args.epochs} | train {running / seen:.3f} | val {val_loss:.3f} "
              f"| acc {val_acc:.3f} | F1 {val_f1:.3f} | {secs:.0f}s{mark}")
        with open(out / "log.csv", "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{running / seen:.4f}", f"{val_loss:.4f}", f"{val_acc:.4f}",
                                    f"{val_f1:.4f}", f"{secs:.1f}"])

    lines = [
        f"Architecture: {args.arch} ({'random init' if args.no_pretrained else 'ImageNet-pretrained'})",
        f"Parameters: {n_params:,} ({n_params / 1e6:.3f} M)",
        f"Best validation macro-F1: {best_f1:.4f}",
        f"Frugality = F1 - 0.3/(1+1/R) = {frugality(best_f1, n_params):.4f}",
        "Per-class F1:",
    ] + [f"  {c:16s} {s:.3f}" for c, s in zip(classes, best_per_class)]
    (out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))

    if args.test_dir is not None:
        ckpt = torch.load(out / "best_model.pt", map_location=device)
        model.load_state_dict(ckpt["model"])
        predict_test(model, classes, S, args.test_dir, out / "predictions.json", device)


if __name__ == "__main__":
    main()
