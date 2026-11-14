"""Evaluate existing epoch checkpoints with deterministic multi-scale-6 TTA."""

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from train_transfer import MEAN, STD, build_model, macro_f1, stratified_split


class MultiScaleDataset(Dataset):
    def __init__(self, samples, img_size):
        self.samples = samples
        self.scales = (img_size, round(img_size * 8 / 7), round(img_size * 9 / 7))
        self.transforms = [transforms.Compose([
            transforms.Resize(scale), transforms.CenterCrop(img_size), transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)]) for scale in self.scales]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        views = []
        for transform in self.transforms:
            view = transform(image)
            views.extend((view, torch.flip(view, dims=[2])))
        return torch.stack(views), label


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, num_classes):
    model.eval()
    total_loss, count = 0.0, 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for views, targets in loader:
        batch_size, num_views = views.shape[:2]
        views = views.flatten(0, 1).to(device)
        targets = targets.to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            probabilities = model(views).float().softmax(1).view(batch_size, num_views, -1).mean(1)
        total_loss += loss_fn(probabilities.clamp_min(1e-8).log(), targets).item() * batch_size
        count += batch_size
        for target, prediction in zip(targets.cpu(), probabilities.argmax(1).cpu()):
            confusion[target, prediction] += 1
    f1, per_class = macro_f1(confusion)
    accuracy = confusion.diag().sum().item() / count
    return total_loss / count, accuracy, f1, per_class


@torch.no_grad()
def predict_test(model, test_dir, classes, img_size, device, batch_size, workers, out_path):
    files = sorted(Path(test_dir).glob("*.jpg"), key=lambda path: int(path.stem))
    loader = DataLoader(MultiScaleDataset([(path, 0) for path in files], img_size), batch_size=batch_size,
                        shuffle=False, num_workers=workers, persistent_workers=workers > 0)
    model.eval()
    predictions, offset = {}, 0
    for views, _ in loader:
        current_batch, num_views = views.shape[:2]
        probabilities = model(views.flatten(0, 1).to(device)).float().softmax(1)
        labels = probabilities.view(current_batch, num_views, -1).mean(1).argmax(1).tolist()
        for path, label in zip(files[offset:offset + current_batch], labels):
            predictions[path.stem] = classes[label]
        offset += current_batch
    out_path.write_text(json.dumps(predictions, indent=4), encoding="utf-8")
    print(f"{len(predictions)} test images -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    checkpoint_paths = sorted((args.run_dir / "checkpoints").glob("epoch_*.pt"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No epoch checkpoints found in {args.run_dir / 'checkpoints'}")
    first = torch.load(checkpoint_paths[0], map_location="cpu")
    classes, img_size, arch = first["classes"], first["img_size"], first["arch"]

    folder = datasets.ImageFolder(args.data_dir)
    _, val_indices = stratified_split(folder.targets, args.val_ratio, args.seed)
    val_samples = [folder.samples[index] for index in val_indices]
    loader = DataLoader(MultiScaleDataset(val_samples, img_size), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, persistent_workers=args.workers > 0)
    model, _ = build_model(arch, len(classes), pretrained=False)
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    print(f"multi-scale-6: Resize {img_size}/{round(img_size * 8 / 7)}/{round(img_size * 9 / 7)} + flip | "
          f"{len(val_samples)} validation images | device {device}")

    rows, best = [], None
    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        loss, accuracy, f1, per_class = evaluate(model, loader, loss_fn, device, len(classes))
        epoch = checkpoint.get("epoch", int(checkpoint_path.stem.split("_")[-1]))
        rows.append((epoch, loss, accuracy, f1))
        if best is None or f1 > best[0]:
            best = (f1, epoch, accuracy, loss, per_class, checkpoint_path)
        print(f"epoch {epoch:2d} | multi-scale-6 loss {loss:.4f} | acc {accuracy:.4f} | F1 {f1:.4f}")

    with open(args.run_dir / "multiscale6_validation.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "val_multiscale6_loss", "val_multiscale6_acc", "val_multiscale6_macro_f1"])
        writer.writerows(rows)

    f1, epoch, accuracy, loss, per_class, checkpoint_path = best
    print(f"\nBest multi-scale-6: epoch {epoch} | loss {loss:.4f} | acc {accuracy:.4f} | macro-F1 {f1:.4f}")
    for class_name, score in zip(classes, per_class):
        print(f"  {class_name:16s} {score:.3f}")

    if args.test_dir:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        predict_test(model, args.test_dir, classes, img_size, device, args.batch_size, args.workers,
                     args.run_dir / "predictions_multiscale6.json")


if __name__ == "__main__":
    main()
