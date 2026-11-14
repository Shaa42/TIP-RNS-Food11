"""Distill a trained ShuffleNetV2 x1.0 teacher into a compact x0.5 student."""

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from train_transfer import (MEAN, STD, build_model, count_params, evaluate, frugality,
                            predict_test, set_seed, stratified_split)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--student-arch", default="shufflenet_v2_x0_5")
    parser.add_argument("--img-size", type=int, default=None,
                        help="override the teacher checkpoint input size")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="weight of the hard-label loss; distillation weight is 1-alpha")
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--full-data", action="store_true",
                        help="train on all labeled images without validation")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("runs_shufflenet_v2_x0_5_distill_mps"))
    args = parser.parse_args()

    if not 0 <= args.alpha <= 1:
        raise ValueError("--alpha must be between 0 and 1")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")

    set_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    teacher_checkpoint = torch.load(args.teacher_checkpoint, map_location="cpu")
    classes = teacher_checkpoint["classes"]
    image_size = args.img_size or teacher_checkpoint["img_size"]
    teacher_arch = teacher_checkpoint["arch"]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.4, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.2),
    ])
    validation_transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    full_train = datasets.ImageFolder(args.data_dir, transform=train_transform)
    full_validation = datasets.ImageFolder(args.data_dir, transform=validation_transform)
    if full_train.classes != classes:
        raise ValueError("Teacher classes do not match the dataset classes")
    if args.full_data:
        train_indices, validation_indices = list(range(len(full_train))), []
    else:
        train_indices, validation_indices = stratified_split(full_train.targets, args.val_ratio, args.seed)
    train_loader = DataLoader(Subset(full_train, train_indices), batch_size=args.batch_size, shuffle=True,
                              drop_last=not args.full_data, num_workers=args.workers,
                              pin_memory=device.type == "cuda",
                              persistent_workers=args.workers > 0)
    validation_loader = None if args.full_data else DataLoader(
        Subset(full_validation, validation_indices), batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.workers, persistent_workers=args.workers > 0)

    teacher, _ = build_model(teacher_arch, len(classes), pretrained=False)
    teacher.load_state_dict(teacher_checkpoint["model"])
    teacher = teacher.to(device).eval()
    teacher.requires_grad_(False)

    student, student_head = build_model(args.student_arch, len(classes), pretrained=True)
    student = student.to(device)
    student_params = count_params(student)
    teacher_params = count_params(teacher)

    print(f"{len(classes)} classes | train {len(train_indices)} | val {len(validation_indices)} | device {device}")
    print(f"Training mode: {'100% labeled data (no validation)' if args.full_data else '90/10 split'}")
    print(f"Teacher: {teacher_arch} | {teacher_params:,} parameters | frozen")
    print(f"Student: {args.student_arch} | {student_params:,} parameters "
          f"({student_params / 1e6:.3f} M) | frugality penalty "
          f"{0.3 / (1 + 1e6 / student_params):.3f}")
    print(f"Distillation: alpha={args.alpha:g} | temperature={args.temperature:g}")
    print(f"Input size: {image_size}x{image_size}")

    head_ids = {id(parameter) for parameter in student_head.parameters()}
    parameter_groups = [
        {"params": [parameter for parameter in student.parameters() if id(parameter) not in head_ids],
         "lr": args.lr},
        {"params": list(student_head.parameters()), "lr": args.lr * 5},
    ]
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[args.lr, args.lr * 5], epochs=args.epochs,
        steps_per_epoch=len(train_loader), pct_start=0.2)
    hard_loss_function = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    args.out_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_dir = args.out_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    with open(args.out_dir / "log.csv", "w", newline="") as file:
        csv.writer(file).writerow(["epoch", "train_total_loss", "train_hard_loss", "train_soft_loss",
                                  "val_flip_tta_loss", "val_flip_tta_acc",
                                  "val_flip_tta_macro_f1", "seconds"])

    best_f1, best_per_class = -1.0, None
    for epoch in range(1, args.epochs + 1):
        student.train()
        start = time.time()
        total_sum = hard_sum = soft_sum = 0.0
        seen = 0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                teacher_logits = teacher(images).float()
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                student_logits = student(images).float()
                hard_loss = hard_loss_function(student_logits, targets)
                temperature = args.temperature
                soft_loss = F.kl_div(
                    F.log_softmax(student_logits / temperature, dim=1),
                    F.softmax(teacher_logits / temperature, dim=1),
                    reduction="batchmean") * temperature ** 2
                loss = args.alpha * hard_loss + (1 - args.alpha) * soft_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            batch_size = images.size(0)
            total_sum += loss.item() * batch_size
            hard_sum += hard_loss.item() * batch_size
            soft_sum += soft_loss.item() * batch_size
            seen += batch_size

        checkpoint = {
            "model": student.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "classes": classes,
            "img_size": image_size,
            "arch": args.student_arch,
            "teacher_arch": teacher_arch,
            "teacher_checkpoint": str(args.teacher_checkpoint),
            "distill_alpha": args.alpha,
            "distill_temperature": args.temperature,
            "full_data": args.full_data,
            "epoch": epoch,
        }
        if validation_loader is not None:
            validation_loss, _, validation_accuracy, validation_f1, per_class, confusion = evaluate(
                student, validation_loader, hard_loss_function, device, len(classes), flip_tta=True)
            checkpoint.update({"val_flip_tta_loss": validation_loss,
                               "val_flip_tta_acc": validation_accuracy,
                               "val_flip_tta_f1": validation_f1})
        torch.save(checkpoint, checkpoint_dir / f"epoch_{epoch:02d}.pt")

        mark = ""
        if validation_loader is not None and validation_f1 > best_f1:
            best_f1, best_per_class = validation_f1, per_class
            torch.save(checkpoint, args.out_dir / "best_model.pt")
            with open(args.out_dir / "confusion.csv", "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["true\\pred"] + classes)
                for class_name, row in zip(classes, confusion.tolist()):
                    writer.writerow([class_name] + row)
            mark = "  * best flip-TTA"
        elif args.full_data and epoch == args.epochs:
            torch.save(checkpoint, args.out_dir / "final_model.pt")
            mark = "  * final"

        seconds = time.time() - start
        if validation_loader is not None:
            print(f"epoch {epoch:2d}/{args.epochs} | total {total_sum / seen:.3f} "
                  f"| hard {hard_sum / seen:.3f} | soft {soft_sum / seen:.3f} "
                  f"| val TTA loss {validation_loss:.3f} | acc {validation_accuracy:.3f} "
                  f"| F1 {validation_f1:.3f} | {seconds:.0f}s{mark}")
            validation_values = [f"{validation_loss:.4f}", f"{validation_accuracy:.4f}",
                                 f"{validation_f1:.4f}"]
        else:
            print(f"epoch {epoch:2d}/{args.epochs} | total {total_sum / seen:.3f} "
                  f"| hard {hard_sum / seen:.3f} | soft {soft_sum / seen:.3f} "
                  f"| {seconds:.0f}s{mark}")
            validation_values = ["", "", ""]
        with open(args.out_dir / "log.csv", "a", newline="") as file:
            csv.writer(file).writerow([
                epoch, f"{total_sum / seen:.4f}", f"{hard_sum / seen:.4f}", f"{soft_sum / seen:.4f}",
                *validation_values, f"{seconds:.1f}"])

    summary = [
        f"Teacher: {teacher_arch} ({teacher_params:,} parameters, frozen)",
        f"Student: {args.student_arch} ({student_params:,} parameters)",
        f"Distillation: alpha={args.alpha:g}, temperature={args.temperature:g}",
        f"Input size: {image_size}x{image_size}",
    ]
    if args.full_data:
        summary += [f"Training data: all {len(train_indices)} labeled images",
                    f"Completed epochs: {args.epochs}",
                    "No validation metrics (full-data final refit)"]
    else:
        summary += [f"Best validation flip-TTA macro-F1: {best_f1:.4f}",
                    f"Student frugality: {frugality(best_f1, student_params):.4f}",
                    "Per-class F1:"]
        summary += [f"  {class_name:16s} {score:.3f}"
                    for class_name, score in zip(classes, best_per_class)]
    (args.out_dir / "summary.txt").write_text("\n".join(summary), encoding="utf-8")
    print("\n" + "\n".join(summary))

    if args.test_dir is not None:
        model_path = args.out_dir / ("final_model.pt" if args.full_data else "best_model.pt")
        best_checkpoint = torch.load(model_path, map_location=device)
        student.load_state_dict(best_checkpoint["model"])
        predict_test(student, classes, image_size, args.test_dir,
                     args.out_dir / "predictions.json", device)


if __name__ == "__main__":
    main()
