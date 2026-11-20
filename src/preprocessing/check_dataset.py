
"""Check dataset completeness and image-label correspondence."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "data" / "processed"
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

total_missing = 0

for split in SPLITS:
    image_dir = ROOT / split / "images"
    label_dir = ROOT / split / "labels"

    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(
            f"Missing images or labels directory for {split}"
        )

    images = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    labels = list(label_dir.glob("*.txt"))
    label_names = {p.stem for p in labels}

    missing = [
        image.name for image in images
        if image.stem not in label_names
    ]

    total_missing += len(missing)

    print(f"{split}:")
    print(f"  Images: {len(images)}")
    print(f"  Label files: {len(labels)}")
    print(f"  Missing label files: {len(missing)}")

    for name in missing[:10]:
        print(f"    Missing label: {name}")

print(f"\nTotal missing label files: {total_missing}")

if total_missing:
    raise SystemExit(1)
