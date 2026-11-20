
from pathlib import Path
from collections import Counter
import csv
import math
import shutil

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data" / "raw" / "data"
TARGET = ROOT / "data" / "processed"
REPORT = ROOT / "reports" / "preprocessing_report.csv"

SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def process_label(source_file, target_file, split, changes):
    output_lines = []

    for line_number, line in enumerate(
        source_file.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue

        values = line.split()

        if len(values) != 5:
            raise ValueError(
                f"Invalid label format: {source_file}:{line_number}"
            )

        class_id = int(values[0])
        x, y, w, h = map(float, values[1:])

        if class_id not in (0, 1):
            raise ValueError(f"Invalid class: {source_file}:{line_number}")

        if not all(math.isfinite(v) for v in (x, y, w, h)):
            raise ValueError(f"Non-finite value: {source_file}:{line_number}")

        if w <= 0 or h <= 0:
            changes.append((split, source_file.name, line_number, "removed_zero_size"))
            continue

        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2

        clipped = (
            max(0.0, min(1.0, x1)),
            max(0.0, min(1.0, y1)),
            max(0.0, min(1.0, x2)),
            max(0.0, min(1.0, y2)),
        )

        cx1, cy1, cx2, cy2 = clipped

        if cx2 <= cx1 or cy2 <= cy1:
            changes.append((split, source_file.name, line_number, "removed_outside"))
            continue

        new_x = (cx1 + cx2) / 2
        new_y = (cy1 + cy2) / 2
        new_w = cx2 - cx1
        new_h = cy2 - cy1

        if any(
            abs(a - b) > 1e-9
            for a, b in zip((x1, y1, x2, y2), clipped)
        ):
            changes.append((split, source_file.name, line_number, "clipped"))

        output_lines.append(
            f"{class_id} {new_x:.10f} {new_y:.10f} "
            f"{new_w:.10f} {new_h:.10f}"
        )

    target_file.write_text(
        "\n".join(output_lines) + ("\n" if output_lines else ""),
        encoding="utf-8",
    )


def main():
    changes = []

    for split in SPLITS:
        source_images = SOURCE / split / "images"
        source_labels = SOURCE / split / "labels"

        target_images = TARGET / split / "images"
        target_labels = TARGET / split / "labels"

        target_images.mkdir(parents=True, exist_ok=True)
        target_labels.mkdir(parents=True, exist_ok=True)

        images = sorted(
            p for p in source_images.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )

        for image in images:
            destination = target_images / image.name
            if not destination.exists():
                shutil.copy2(image, destination)

            source_label = source_labels / f"{image.stem}.txt"
            target_label = target_labels / f"{image.stem}.txt"

            if not source_label.exists():
                raise FileNotFoundError(source_label)

            process_label(source_label, target_label, split, changes)

        print(f"{split}: processed {len(images)} images")

    REPORT.parent.mkdir(parents=True, exist_ok=True)

    with REPORT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["split", "file", "line", "action"])
        writer.writerows(changes)

    print("\nPreprocessing complete.")
    print("Changes:", len(changes))
    print("Breakdown:", dict(Counter(row[3] for row in changes)))
    print("Report:", REPORT)


if __name__ == "__main__":
    main()
