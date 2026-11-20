
from pathlib import Path
import math

ROOT = Path("data/processed")
errors = []

for split in ("train", "val", "test"):
    label_dir = ROOT / split / "labels"
    checked = 0

    for file in label_dir.glob("*.txt"):
        checked += 1

        for line_no, line in enumerate(
            file.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue

            values = line.split()

            if len(values) != 5:
                errors.append(f"{file}:{line_no} - wrong number of values")
                continue

            try:
                class_id = int(values[0])
                x, y, w, h = map(float, values[1:])
            except ValueError:
                errors.append(f"{file}:{line_no} - invalid number")
                continue

            if class_id not in (0, 1):
                errors.append(f"{file}:{line_no} - invalid class")

            if not all(math.isfinite(v) for v in (x, y, w, h)):
                errors.append(f"{file}:{line_no} - non-finite coordinate")
            elif not (
                0 <= x <= 1
                and 0 <= y <= 1
                and 0 < w <= 1
                and 0 < h <= 1
                and x - w / 2 >= -1e-6
                and x + w / 2 <= 1 + 1e-6
                and y - h / 2 >= -1e-6
                and y + h / 2 <= 1 + 1e-6
            ):
                errors.append(f"{file}:{line_no} - invalid bounding box")

    print(f"{split}: checked {checked} label files")

print(f"\nTotal errors: {len(errors)}")

for error in errors[:20]:
    print(error)

# Additional diagnostic: bounding boxes outside image boundaries
minor = 0
moderate = 0
major = 0

for split in ("train", "val", "test"):
    for file in (ROOT / split / "labels").glob("*.txt"):
        for line in file.read_text(encoding="utf-8").splitlines():
            values = line.split()

            if len(values) != 5:
                continue

            try:
                x, y, w, h = map(float, values[1:])
            except ValueError:
                continue

            if not all(math.isfinite(v) for v in (x, y, w, h)):
                continue

            overflow = max(
                0,
                -(x - w / 2),
                x + w / 2 - 1,
                -(y - h / 2),
                y + h / 2 - 1,
            )

            if overflow > 0:
                if overflow <= 0.001:
                    minor += 1
                elif overflow <= 0.01:
                    moderate += 1
                else:
                    major += 1

print("\nBounding box overflow:")
print("Minor (<=0.001):", minor)
print("Moderate (0.001-0.01):", moderate)
print("Major (>0.01):", major)

# Diagnose invalid boxes not counted by the overflow check
other_issues = []

for split in ("train", "val", "test"):
    for file in (ROOT / split / "labels").glob("*.txt"):
        for line_no, line in enumerate(
            file.read_text(encoding="utf-8").splitlines(), 1
        ):
            values = line.split()

            if len(values) != 5:
                continue

            try:
                x, y, w, h = map(float, values[1:])
            except ValueError:
                continue

            if not all(math.isfinite(v) for v in (x, y, w, h)):
                continue

            overflow = max(
                0,
                -(x - w / 2),
                x + w / 2 - 1,
                -(y - h / 2),
                y + h / 2 - 1,
            )

            if overflow == 0 and not (
                0 <= x <= 1
                and 0 <= y <= 1
                and 0 < w <= 1
                and 0 < h <= 1
            ):
                other_issues.append(
                    f"{file}:{line_no} -> {line}"
                )

print("\nOther invalid bounding boxes:", len(other_issues))

for issue in other_issues[:20]:
    print(issue)

# Show the largest bounding-box boundary violations
largest = []

for split in ("train", "val", "test"):
    for file in (ROOT / split / "labels").glob("*.txt"):
        for line_no, line in enumerate(
            file.read_text(encoding="utf-8").splitlines(), 1
        ):
            values = line.split()

            if len(values) != 5:
                continue

            try:
                x, y, w, h = map(float, values[1:])
            except ValueError:
                continue

            if not all(math.isfinite(v) for v in (x, y, w, h)):
                continue

            overflow = max(
                0,
                -(x - w / 2),
                x + w / 2 - 1,
                -(y - h / 2),
                y + h / 2 - 1,
            )

            if overflow > 0:
                largest.append((overflow, split, file.name, line_no))

largest.sort(reverse=True)

print("\nLargest bounding-box violations:")
for overflow, split, name, line_no in largest[:10]:
    print(
        f"{split}/{name}, line {line_no}: "
        f"overflow={overflow:.6f}"
    )
