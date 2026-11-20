"""Train YOLOv8s from a YAML experiment configuration."""
import argparse
from pathlib import Path
import yaml
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/baseline.yaml')
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    model_path = cfg.pop('model')
    model = YOLO(model_path)
    model.train(**cfg)


if __name__ == '__main__':
    main()
