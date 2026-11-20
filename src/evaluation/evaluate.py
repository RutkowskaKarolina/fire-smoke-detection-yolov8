
"""Evaluate YOLOv8s on validation or held-out test data."""

import argparse
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou


def count_false_positives(
    pred_boxes,
    pred_classes,
    pred_conf,
    gt_boxes,
    gt_classes,
    match_iou=0.50,
):
    """Count FP using class-aware, one-to-one matching."""

    if len(pred_boxes) == 0:
        return 0

    if len(gt_boxes) == 0:
        return len(pred_boxes)

    ious = box_iou(pred_boxes, gt_boxes)

    matched_gt = set()
    false_positives = 0

    # Match higher-confidence predictions first.
    order = torch.argsort(pred_conf, descending=True)

    for pred_idx in order.tolist():
        best_iou = match_iou
        best_gt = None

        for gt_idx in range(len(gt_boxes)):
            if gt_idx in matched_gt:
                continue

            if int(pred_classes[pred_idx]) != int(gt_classes[gt_idx]):
                continue

            iou = float(ious[pred_idx, gt_idx])

            if iou >= best_iou:
                best_iou = iou
                best_gt = gt_idx

        if best_gt is not None:
            matched_gt.add(best_gt)
        else:
            false_positives += 1

    return false_positives


def load_yolo_labels(label_path, image_width, image_height):
    """Convert normalized YOLO xywh labels to pixel xyxy boxes."""

    boxes = []
    classes = []

    if not label_path.is_file():
        raise FileNotFoundError(
            f"Missing label file: {label_path}"
        )

    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        values = line.split()

        if len(values) != 5:
            raise ValueError(
                f"Invalid annotation in {label_path}: {line}"
            )

        class_id = int(values[0])
        x, y, w, h = map(float, values[1:])

        x1 = (x - w / 2) * image_width
        y1 = (y - h / 2) * image_height
        x2 = (x + w / 2) * image_width
        y2 = (y + h / 2) * image_height

        boxes.append([x1, y1, x2, y2])
        classes.append(class_id)

    if not boxes:
        return (
            torch.empty((0, 4), dtype=torch.float32),
            torch.empty((0,), dtype=torch.long),
        )

    return (
        torch.tensor(boxes, dtype=torch.float32),
        torch.tensor(classes, dtype=torch.long),
    )


def calculate_fp_per_image(
    model,
    data_path,
    split="val",
    conf=0.25,
    nms_iou=0.70,
    match_iou=0.50,
    imgsz=640,
):
    """Calculate FP/image, including images without objects."""

    data_path = Path(data_path).resolve()

    with data_path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file)

    dataset_root = Path(data.get("path", data_path.parent))

    if not dataset_root.is_absolute():
        dataset_root = data_path.parent / dataset_root

    image_dir = Path(data[split])

    if not image_dir.is_absolute():
        image_dir = dataset_root / image_dir

    image_dir = image_dir.resolve()

    # Expected layout: val/images and val/labels.
    label_dir = image_dir.parent / "labels"

    if not image_dir.is_dir():
        raise FileNotFoundError(
            f"Image directory not found: {image_dir}"
        )

    if not label_dir.is_dir():
        raise FileNotFoundError(
            f"Label directory not found: {label_dir}"
        )

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    image_files = sorted(
        p for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    )

    if not image_files:
        raise ValueError(
            f"No images found in {image_dir}"
        )

    # Verify labels before inference. Empty files are valid negatives.
    for image_path in image_files:
        relative_path = image_path.relative_to(image_dir)
        label_path = label_dir / relative_path.with_suffix(".txt")

        if not label_path.is_file():
            raise FileNotFoundError(
                f"Missing label for {image_path}: {label_path}"
            )

    total_fp = 0
    processed_images = 0

    print(f"\nEvaluating FP/image on {split}...")
    print(f"Images: {len(image_files)}")
    print(f"Confidence threshold: {conf}")
    print(f"NMS IoU: {nms_iou}")
    print(f"Matching IoU: {match_iou}")
    print(f"Image size: {imgsz}")

    for result in model.predict(
        source=str(image_dir),
        conf=conf,
        iou=nms_iou,
        imgsz=imgsz,
        stream=True,
        verbose=False,
    ):
        image_path = Path(result.path).resolve()
        relative_path = image_path.relative_to(image_dir)
        label_path = label_dir / relative_path.with_suffix(".txt")

        height, width = result.orig_shape

        gt_boxes, gt_classes = load_yolo_labels(
            label_path,
            width,
            height,
        )

        pred_boxes = result.boxes.xyxy.cpu()
        pred_classes = result.boxes.cls.cpu()
        pred_conf = result.boxes.conf.cpu()

        total_fp += count_false_positives(
            pred_boxes,
            pred_classes,
            pred_conf,
            gt_boxes,
            gt_classes,
            match_iou=match_iou,
        )

        processed_images += 1

    if processed_images != len(image_files):
        raise RuntimeError(
            f"Expected {len(image_files)} images, "
            f"processed {processed_images}"
        )

    fp_per_image = total_fp / processed_images

    print("\n--- FP/image results ---")
    print(f"Images evaluated: {processed_images}")
    print(f"False positives: {total_fp}")
    print(f"FP/image: {fp_per_image:.6f}")

    return fp_per_image


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate YOLOv8s detection performance."
    )

    parser.add_argument(
        "--weights",
        required=True,
        help="Path to trained YOLO checkpoint.",
    )
    parser.add_argument(
        "--data",
        default="data/processed/data.yaml",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.70,
        help="NMS IoU threshold.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
    )

    args = parser.parse_args()

    if not Path(args.weights).is_file():
        parser.error(
            f"Checkpoint not found: {args.weights}"
        )

    model = YOLO(args.weights)

    print("\n--- Standard YOLO evaluation ---")

    metrics = model.val(
        data=args.data,
        split=args.split,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        project="reports",
        name="evaluation",
        exist_ok=False,
    )

    print("\n--- Detection metrics ---")
    print(f"mAP@0.5: {metrics.box.map50:.6f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.6f}")
    print(f"Precision: {metrics.box.mp:.6f}")
    print(f"Recall: {metrics.box.mr:.6f}")

    calculate_fp_per_image(
        model=model,
        data_path=args.data,
        split=args.split,
        conf=args.conf,
        nms_iou=args.iou,
        match_iou=0.50,
        imgsz=args.imgsz,
    )


if __name__ == "__main__":
    main()
