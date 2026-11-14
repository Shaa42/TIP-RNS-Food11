"""Average nearby checkpoints, recalibrate BatchNorm, and evaluate with flip-2."""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.swa_utils import update_bn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_transfer import MEAN, STD, build_model, evaluate, set_seed, stratified_split


def average_states(checkpoints):
    states = [checkpoint["model"] for checkpoint in checkpoints]
    averaged = {}
    for name in states[0]:
        tensors = [state[name].cpu() for state in states]
        if tensors[0].is_floating_point():
            averaged[name] = torch.stack(tensors).mean(0)
        else:
            averaged[name] = tensors[-1].clone()
    return averaged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()

    set_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    paths = [args.run_dir / "checkpoints" / f"epoch_{epoch:02d}.pt" for epoch in args.epochs]
    checkpoints = [torch.load(path, map_location="cpu") for path in paths]
    reference = checkpoints[0]
    for checkpoint in checkpoints[1:]:
        for key in ("arch", "classes", "img_size"):
            if checkpoint[key] != reference[key]:
                raise ValueError(f"Checkpoint mismatch for {key}")

    image_size = reference["img_size"]
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.4, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.2),
    ])
    validation_transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)), transforms.CenterCrop(image_size),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    full_train = datasets.ImageFolder(args.data_dir, transform=train_transform)
    full_validation = datasets.ImageFolder(args.data_dir, transform=validation_transform)
    train_indices, validation_indices = stratified_split(full_train.targets, args.val_ratio, args.seed)
    bn_loader = DataLoader(Subset(full_train, train_indices), batch_size=args.batch_size, shuffle=False,
                           num_workers=args.workers, persistent_workers=args.workers > 0)
    validation_loader = DataLoader(Subset(full_validation, validation_indices),
                                   batch_size=args.batch_size * 2, shuffle=False,
                                   num_workers=args.workers, persistent_workers=args.workers > 0)

    model, _ = build_model(reference["arch"], len(reference["classes"]), pretrained=False)
    model.load_state_dict(average_states(checkpoints))
    model = model.to(device)
    print(f"Averaging epochs {args.epochs} on device {device}")
    print("Recalibrating BatchNorm on the training split...")
    update_bn(bn_loader, model, device=device)

    loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)
    loss, _, accuracy, f1, per_class, confusion = evaluate(
        model, validation_loader, loss_function, device, len(reference["classes"]), flip_tta=True)
    epoch_label = "_".join(str(epoch) for epoch in args.epochs)
    output_path = args.run_dir / f"averaged_epochs_{epoch_label}.pt"
    output = {
        "model": model.state_dict(),
        "classes": reference["classes"],
        "img_size": image_size,
        "arch": reference["arch"],
        "averaged_epochs": args.epochs,
        "batchnorm_recalibrated": True,
        "val_flip_tta_loss": loss,
        "val_flip_tta_acc": accuracy,
        "val_flip_tta_f1": f1,
    }
    torch.save(output, output_path)

    print(f"Validation flip-2 loss: {loss:.6f}")
    print(f"Validation flip-2 accuracy: {accuracy:.6f}")
    print(f"Validation flip-2 macro-F1: {f1:.10f}")
    for class_name, score in zip(reference["classes"], per_class):
        print(f"  {class_name:16s} {score:.3f}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
