# Waldo Detection with Faster R-CNN + CLIP Re-ranking

This project explores small-object detection for locating Waldo in crowded images using Faster R-CNN with customized anchors and optional CLIP-based semantic re-ranking.

The project was developed for COMPSCI 682 (Neural Networks) at UMass Amherst.

---

# Overview

The goal of this project is to improve Waldo localization performance in visually cluttered scenes where the target object is small and difficult to detect.

The pipeline combines:

- Faster R-CNN with FPN backbone
- Small-object tuned anchor generation
- Optional CLIP-based semantic scoring
- IoU-based evaluation

---
# Datasets

Please refer to the waldo_merged_resplit.zip
Merge version1 and version2 images from robotflfow.

---

# Features

- Faster R-CNN object detection
- Customized anchors for small-object localization
- Optional CLIP semantic re-ranking
- Validation/Test IoU evaluation
- SLURM training support

---

# Project Structure

```text
FINAL_WALDO/
├── readme.md
├── requirements.txt
├── run_train_faster_rcnn.sh
└── train_frcnn_waldo.py
```

---

# Installation

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

---

# Dataset Format

The dataset uses COCO-style annotations.

Expected structure:

```text
waldo_merged_resplit/
├── train/
│   ├── _annotations.coco.json
│   └── images...
├── valid/
│   ├── _annotations.coco.json
│   └── images...
└── test/
    ├── _annotations.coco.json
    └── images...
```

---

# Training

Run locally:

```bash
python train_frcnn_waldo.py \
    --dataset_root ./waldo_merged_resplit \
    --output_dir ./results/run1 \
    --epochs 12 \
    --batch_size 2 \
    --img_size 512 \
    --use_clip
```

Or submit with SLURM:

```bash
sbatch run_train_faster_rcnn.sh
```

---

# CLIP Re-ranking

When enabled, CLIP computes semantic similarity between predicted image regions and a text query such as:

```text
"Waldo wearing red and white striped shirt"
```

The CLIP score is combined with Faster R-CNN confidence scores during evaluation.

Enable using:

```bash
--use_clip
```

---

# Evaluation Metrics

The project evaluates:

- Mean IoU
- Localization quality
- Bounding box overlap accuracy

---

# Dependencies

Main libraries:

- PyTorch
- Torchvision
- OpenAI CLIP
- NumPy
- Pillow
- Matplotlib
- tqdm

See `requirements.txt` for full dependency versions.

---

# Notes

- The project focuses on small-object localization challenges.
- Anchor sizes were modified to better detect tiny Waldo instances.
- CLIP is used only for semantic re-ranking during evaluation.

---

# Acknowledgements

- UMass Amherst COMPSCI 682
- PyTorch
- Torchvision
- OpenAI CLIP# waldo
