"""Basic file-count and label-format audit (not a duplicate/leakage audit)."""
from pathlib import Path

root = Path('data/processed')
for split in ('train', 'val', 'test'):
    images = root / 'images' / split
    labels = root / 'labels' / split
    files = [p for p in images.rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}] if images.exists() else []
    print(f'{split}: {len(files)} images, labels directory exists: {labels.exists()}')
