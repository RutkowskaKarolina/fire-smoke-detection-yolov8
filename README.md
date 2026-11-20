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

### Dataset Structure

```text
data/
├── raw/
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

The dataset is not included in this repository.

The exact dataset version, licensing conditions, and redistribution permissions must be verified and documented before publication or redistribution.

### Preprocessing

Preprocessing includes validation of YOLO annotations, correction of bounding boxes extending beyond image boundaries, and removal of invalid zero-sized bounding boxes.

The original train/validation/test split is preserved.

Preprocessing information is recorded in `reports/preprocessing_report.csv`.

## 3. Environment and Dependencies

The project was prepared locally using:

- Python 3.12.2
- Ultralytics YOLO 8.3.200
- PyTorch
- PyCharm
- Windows PowerShell

Full experiments are planned for a GPU environment, such as Google Colab Pro.

### Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

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

The current baseline specifies 100 epochs, an image size of 640, batch size 16, and random seed 42.

**Important:** The final optimizer and learning-rate configuration must be explicitly fixed and verified before the full experiments. Ultralytics `optimizer: auto` may override the learning-rate setting.

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
| Confidence threshold | 0.25 |
| NMS IoU threshold | 0.70 |
| Ground-truth matching IoU | 0.50 |
| Image size | 640 |

Predictions are processed in descending confidence order. Each ground-truth object can be matched to at most one prediction of the same class.

Unmatched predictions are counted as false positives.

FP/image is calculated as:

**FP/image = Total false-positive detections / Total evaluated images**

Images without annotated objects are included in the denominator.

## 7. Technical Verification

The project has undergone initial local verification:

- Dataset preprocessing and annotation validation completed.
- Validation split: 3,099 images, no missing label files.
- Test split: 4,306 images, no missing label files.
- One-epoch CPU smoke test completed successfully.
- Evaluation script passed Python syntax checking.
- Basic one-to-one matching tests completed.
- FP counting tested on real validation images.

The smoke test was conducted on a small subset of the training data and is not considered a baseline research experiment.

Full baseline training, experimental comparisons, and final held-out test evaluation remain pending.

## 8. Reproducibility

The project uses YAML configuration files and a fixed primary random seed of 42.

For each full experiment, the following should be recorded:

- Experiment configuration
- Dataset version and preprocessing procedure
- Python, PyTorch, and Ultralytics versions
- GPU environment
- Initial model weights
- Random seed
- Training metrics and artifacts
- Validation and test results

The repository is under development and does not yet contain final experimental results.

## 9. Project Status

Task 03 experimental infrastructure is being prepared and technically verified.

The full YOLOv8s training experiments and final scientific evaluation have not yet been completed.