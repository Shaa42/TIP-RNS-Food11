"""Generate flip-2 Food-11 predictions from an existing checkpoint without retraining."""

import argparse
from pathlib import Path

import torch

from train_transfer import build_model, predict_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model, _ = build_model(checkpoint["arch"], len(checkpoint["classes"]), pretrained=False)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    args.output.parent.mkdir(exist_ok=True, parents=True)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Architecture: {checkpoint['arch']} | epoch {checkpoint.get('epoch', 'unknown')} | device {device}")
    predict_test(model, checkpoint["classes"], checkpoint["img_size"],
                 args.test_dir, args.output, device)


if __name__ == "__main__":
    main()
