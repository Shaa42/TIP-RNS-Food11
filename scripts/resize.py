from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def resize_dataset(
    input_dir: Path,
    output_dir: Path,
    size: tuple[int, int],
) -> tuple[int, int]:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()

    if input_dir == output_dir:
        raise ValueError("Input and output directories must be different.")

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    images = (
        path
        for path in input_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        # Avoid processing old output if it is inside the input directory.
        and not path.resolve().is_relative_to(output_dir)
    )

    resized = 0
    failed = 0

    for source in images:
        destination = output_dir / source.relative_to(input_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)

        try:
            with Image.open(source) as original:
                image_format = original.format
                image = ImageOps.exif_transpose(original).convert("RGB")
                image = image.resize(size, Image.Resampling.LANCZOS)

                if image_format:
                    image.save(destination, format=image_format)
                else:
                    image.save(destination)

            resized += 1
        except (OSError, ValueError) as error:
            print(f"Could not resize {source}: {error}", file=sys.stderr)
            failed += 1

    return resized, failed


def main() -> int:
    project_dir = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Resize a dataset of images while preserving its folder structure."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=project_dir / "food-11",
        help="Dataset directory to read (default: food-11).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "food-11-resized",
        help="Directory for resized copies (default: food-11-resized).",
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(224, 224),
        help="Output image size in pixels (default: 224 224).",
    )
    args = parser.parse_args()

    try:
        resized, failed = resize_dataset(
            args.input_dir,
            args.output_dir,
            tuple(args.size),
        )
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    print(f"Resized {resized} image(s); failed to process {failed}.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
