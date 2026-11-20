# Fire and Smoke Detection — YOLOv8s Experimental Study

Master's thesis research project focused on detecting fire and smoke in images using deep learning.

The study investigates how selected training hyperparameters affect the detection performance of a fixed YOLOv8s architecture.

## 1. Research Objective

The objective is to optimize the training configuration of YOLOv8s without modifying its architecture.

The investigated hyperparameters are:

- Learning rate
- Number of training epochs
- Batch size
- Data augmentation

The baseline configuration (E0) is compared with controlled experimental variants (E1–E4). A combined configuration (E5) will be selected using validation results.

### Research Hypothesis

Compared with the baseline configuration, the optimized model should achieve:

- An improvement of at least 3 percentage points in mAP@0.5.
- Recall no more than 2 percentage points below the baseline.
- A reduction of at least 5% in false positives per image (FP/image).

These criteria will be evaluated experimentally. They are not yet confirmed.

## 2. Dataset

The project uses the *Smoke Fire Detection YOLO* dataset obtained from Kaggle.

The prepared dataset contains approximately 21,527 images:

| Split | Images |
|---|---:|
| Training | 14,122 |
| Validation | 3,099 |
| Test | 4,306 |

Classes:

- `0` — smoke
- `1` — fire

The dataset uses YOLO-format bounding-box annotations.

The dataset is not included in this repository.

The exact dataset version, licensing conditions, and redistribution permissions must be verified and documented before publication or redistribution.

### Dataset Structure

The preprocessing pipeline produces the following directory structure:

```text
data/
├── raw/
│   └── data/
│       ├── train/
│       ├── val/
│       └── test/
└── processed/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    ├── test/
    │   ├── images/
    │   └── labels/
    └── data.yaml
```

### Preprocessing

After downloading and extracting the dataset, place the original files in the following directory structure:

```text
data/raw/data/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Run the preprocessing script from the project root:

```powershell
python src/preprocessing/prepare_dataset.py
```

The script:

- Copies images into `data/processed/`.
- Validates YOLO annotation format and class identifiers.
- Clips bounding boxes to image boundaries.
- Removes invalid bounding boxes.
- Preserves the original train/validation/test split.
- Records preprocessing changes.

A detailed preprocessing report is generated at:

```text
reports/preprocessing_report.csv
```

The script uses predefined input and output paths and does not support command-line arguments.

### Dataset Validation

Before validating annotation values, check dataset completeness:

```powershell
python src/preprocessing/check_dataset.py
```

The script checks the number of images and label files in each dataset split and identifies images without corresponding annotation files.

The completeness check confirmed:

| Split | Images | Label files | Missing labels |
|---|---:|---:|---:|
| Train | 14,122 | 14,122 | 0 |
| Validation | 3,099 | 3,099 | 0 |
| Test | 4,306 | 4,306 | 0 |

All 21,527 images have corresponding label files. Empty annotation files are permitted for images without annotated objects.

This check does not detect duplicate images or data leakage between splits.

After preprocessing, validate the YOLO annotations by running:

```powershell
python src/preprocessing/validate_labels.py
```

The script checks:

- Annotation format and numerical values.
- Valid class identifiers.
- Bounding-box dimensions.
- Normalized coordinates.
- Bounding boxes extending beyond image boundaries.

The output includes the number of checked label files, detected errors, and bounding-box boundary violations.

A successful validation should report:

```text
Total errors: 0
```

The number of checked label files should also be compared with the expected dataset structure.

This script validates existing annotation files; it does not independently verify that every image has a corresponding label file.

### Dataset Configuration

A dataset configuration template is provided in:

```text
config/data.yaml
```

To configure the dataset:

1. Download the *Smoke Fire Detection YOLO* dataset from Kaggle.
2. Place the original files in `data/raw/data/`, following the structure described above.
3. Run the preprocessing script and validate the resulting annotations.
4. Copy `config/data.yaml` to `data/processed/data.yaml`.
5. In the copied file, replace the `path` value with the absolute path to your local `data/processed/` directory.
6. Verify that the `train`, `val`, and `test` directories contain the corresponding `images/` and `labels/` subdirectories.

The training configuration in `config/baseline.yaml` references `data/processed/data.yaml`.

The dataset configuration must be adapted to the execution environment, such as Windows, Linux, or Google Colab.

The class identifiers must remain unchanged: `0` for smoke and `1` for fire.

## 3. Environment and Dependencies

The project was prepared locally using:

- Python 3.12.2
- Ultralytics YOLO 8.3.200
- PyTorch
- PyCharm
- Windows PowerShell

Full experiments are planned for a GPU environment, such as Google Colab Pro.

### Installation

Create and activate a virtual environment on Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

All commands below should be executed from the project root directory.

## 4. Baseline Training (E0)

The baseline uses the YOLOv8s architecture initialized from pretrained `yolov8s.pt` weights.

The baseline configuration is stored in:

```text
config/baseline.yaml
```

To start baseline training:

```powershell
python src/training/train.py --config config/baseline.yaml
```

The baseline configuration (E0) uses the following fixed training parameters:

| Parameter | Value |
|---|---|
| Model | YOLOv8s (`yolov8s.pt`) |
| Epochs | 100 |
| Image size | 640 |
| Batch size | 16 |
| Optimizer | SGD |
| Initial learning rate (`lr0`) | 0.01 |
| Momentum | 0.937 |
| Weight decay | 0.0005 |
| Random seed | 42 |
| Deterministic training | Enabled |

The optimizer is explicitly set to `SGD` rather than `auto` to prevent automatic optimizer selection from overriding the configured learning rate.

These settings define the reference configuration for subsequent controlled experiments.

Training artifacts and model checkpoints are saved under the `models/` directory and are excluded from Git tracking.

## 5. Experimental Design

The experiments follow a controlled comparison:

| Experiment | Investigated factor |
|---|---|
| E0 | Baseline |
| E1 | Learning rate |
| E2 | Number of epochs |
| E3 | Batch size |
| E4 | Data augmentation |
| E5 | Combined configuration selected on validation data |

The YOLOv8s architecture, dataset split, and initial pretrained weights remain fixed.

The test set must not be used to select hyperparameters.

The currently implemented experimental configuration files are located in `config/experiments/`. Additional configurations will be added as the experimental study progresses.

## 6. Evaluation

The evaluation script is:

```text
src/evaluation/evaluate.py
```

It calculates:

- mAP@0.5
- mAP@0.5:0.95
- Precision
- Recall
- FP/image

### Validation

After training E0, run:

```powershell
python src/evaluation/evaluate.py --weights models/E0_baseline/weights/best.pt --data data/processed/data.yaml --split val
```

### Held-out Test

After experimental configurations have been selected and frozen, run:

```powershell
python src/evaluation/evaluate.py --weights models/E0_baseline/weights/best.pt --data data/processed/data.yaml --split test
```

The selected E5 checkpoint must also be evaluated on the held-out test set.

### FP/image Methodology

False positives are counted using class-aware one-to-one matching between predictions and ground-truth bounding boxes.

The evaluation settings are:

| Parameter | Value |
|---|---|
| Confidence threshold for FP/image | 0.25 |
| NMS IoU threshold | 0.70 |
| Ground-truth matching IoU | 0.50 |
| Image size | 640 |

Predictions are processed in descending confidence order.

Each ground-truth object can be matched to at most one prediction of the same class.

A prediction is considered a true positive if it matches an unmatched ground-truth object of the same class with IoU ≥ 0.50.

Unmatched predictions are counted as false positives.

FP/image is calculated as:

**FP/image = Total false-positive detections / Total evaluated images**

Images without annotated objects are included in the denominator.

The confidence threshold and NMS IoU threshold are held constant across experimental configurations for FP/image comparisons.

The evaluation procedure for mAP, Precision, and Recall must also be applied consistently across all experiments.

## 7. Technical Verification

The project has undergone initial local verification:

- Dataset preprocessing and annotation validation completed.
- Validation split: 3,099 images, no missing label files.
- Test split: 4,306 images, no missing label files.
- One-epoch CPU smoke test completed successfully.
- Evaluation script passed Python syntax checking.
- Basic one-to-one matching tests completed.
- FP counting tested on real validation images.

The preprocessing pipeline reported 397 annotation changes:

- 379 bounding-box clipping operations.
- 18 removals of invalid zero-sized bounding boxes.

The smoke test was conducted on a small subset of the training data and is not considered a baseline research experiment.

Full baseline training, experimental comparisons, and final held-out test evaluation remain pending.

## 8. Reproducibility

The project uses YAML configuration files and a fixed primary random seed of 42.

The dataset is not committed to the repository, and local dataset paths must be configured before running experiments.

For each full experiment, the following should be recorded:

- Experiment configuration.
- Dataset version and preprocessing procedure.
- Python, PyTorch, and Ultralytics versions.
- GPU environment.
- Initial model weights.
- Random seed.
- Training metrics and artifacts.
- Validation and test results.

To reproduce an experiment, the same dataset split, initial weights, training configuration, and evaluation protocol should be used.

The repository is under development and does not yet contain final experimental results.

## 9. Project Status

Task 03 experimental infrastructure has been implemented and partially verified through local technical tests.

The repository currently includes preprocessing, annotation validation, baseline training, experimental configurations, and evaluation scripts.

The full YOLOv8s training experiments and final scientific evaluation have not yet been completed.