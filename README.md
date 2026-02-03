# Paralleldots Assignment Solution
# SAM2 Object Tracking with IoU Evaluation

A computer vision project implementing object tracking across video frames using Meta's Segment Anything Model 2 (SAM2), with comprehensive performance evaluation on the CMU 3D Object Recognition dataset.

## Overview

This project tackles the challenge of tracking everyday objects through image sequences. I used SAM2's video prediction capabilities to follow objects like soup cans, juice boxes, and milk cartons as they appear in different frames, then evaluated how well the tracking performed using Intersection over Union (IoU) metrics.

## What I Built

The solution implements a complete pipeline for:
- Extracting ground truth bounding boxes from segmentation masks
- Tracking objects across multiple frames using SAM2
- Batch processing 10 different object types through 400+ image pairs
- Calculating IoU scores to measure tracking accuracy
- Visualizing performance with comprehensive analytics

## Key Features

### Core Functionality
- **Automated Bounding Box Extraction**: Converts pixel masks to coordinate boxes
- **Frame-by-Frame Tracking**: Propagates object locations through image sequences
- **IoU-Based Evaluation**: Measures tracking accuracy against ground truth
- **Batch Processing**: Handles all object types automatically with error recovery

### Performance Analysis
- Per-object class statistics
- Distribution visualizations
- Success rate categorization (good/medium/poor tracking)
- Detailed performance breakdowns

## Dataset

**CMU 3D Object Recognition Dataset**
- 10 common household objects
- Multiple frames per object (50+ images each)
- Ground truth segmentation masks included
- Total: 400+ image pairs processed

Objects tracked:
- Soup cans (chowder, tomato)
- Beverages (OJ carton, soy milk, diet coke)
- Juice boxes
- Rice packages

## Results

The tracking system achieved solid performance across most object types:

| Object | Mean IoU | Performance |
|--------|----------|-------------|
| Orange Juice Carton | 0.54 | ⭐⭐⭐ Excellent |
| Pot Roast Soup | 0.52 | ⭐⭐⭐ Excellent |
| Soy Milk Carton | 0.50 | ⭐⭐⭐ Good |
| Juice Box | 0.48 | ⭐⭐ Good |
| Can (Chowder) | 0.45 | ⭐⭐ Good |
| Diet Coke | 0.41 | ⭐⭐ Moderate |
| Can (Soy Milk) | 0.41 | ⭐⭐ Moderate |
| Rice (Tuscan) | 0.38 | ⭐ Moderate |
| Can (Tomato Soup) | 0.28 | ⭐ Challenging |
| Rice Pilaf | 0.24 | ⭐ Challenging |

**Overall Statistics:**
- Mean IoU: 0.42
- Median IoU: 0.44
- Predictions with IoU > 0.5: ~35%

### What Worked Well
- Objects with distinct textures and shapes (cartons, bottles)
- Items with clear color contrast
- Larger objects with consistent appearance

### Challenges Faced
- Similar-looking cylindrical objects (soup cans)
- Items with repetitive patterns (rice packages)
- Objects that change appearance between frames

## Technical Implementation

### Key Components

**1. Bounding Box Extraction (`process_img_png_mask`)**
```python
# Converts binary masks to coordinate boxes
# Finds min/max pixel positions for tight bounding boxes
```

**2. IoU Calculation (`calculate_iou`)**
```python
# Measures overlap between predicted and ground truth boxes
# Formula: Intersection Area / Union Area
```

**3. Object Tracking (`track_item_boxes`)**
```python
# Uses SAM2's video predictor to propagate segmentations
# Handles frame-by-frame tracking with state management
```

**4. Batch Processing**
- Loops through all object types
- Processes each frame sequence
- Updates bounding boxes dynamically
- Saves predictions for evaluation

### Data Pipeline

```
Ground Truth Masks → Extract Bboxes → Track Through Frames
                                            ↓
Ground Truth CSV ← Compare & Evaluate ← Predictions CSV
                         ↓
                   IoU Results + Visualizations
```

## Setup & Installation

### Requirements
```bash
# Core dependencies
torch>=2.0.0
torchvision
opencv-python
matplotlib
pandas
numpy
pillow

# SAM2
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e .
```

### Dataset Download
```bash
# CMU 3D Object Recognition Dataset
wget http://www.cs.cmu.edu/~ehsiao/3drecognition/CMU10_3D.zip
unzip CMU10_3D.zip
```

### Model Checkpoint
```bash
# Download SAM2 Hiera-Tiny model
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt
```

## Usage

### Running the Complete Pipeline

Simply execute the Jupyter notebook cells in order:

1. **Setup & Installation** - Install dependencies and download models
2. **Data Loading** - Load dataset and initialize SAM2
3. **Batch Processing** - Track all objects across all frames
4. **Evaluation** - Calculate IoU scores and generate statistics
5. **Visualization** - Create performance plots

### Generated Outputs

The notebook creates these files in the dataset directory:
- `sam2_predictions.csv` - Model predictions (~412 rows)
- `ground_truth_boxes.csv` - Ground truth boxes (~50 rows)
- `iou_results.csv` - Merged results with IoU scores
- `iou_analysis.png` - 4-panel performance visualization

## Project Structure

```
.
├── ds_solution.ipynb          # Main notebook with complete solution
├── README.md                  # This file
├── sam2_hiera_tiny.pt        # SAM2 model checkpoint
└── CMU10_3D/
    └── data_2D/              # Dataset images and masks
        ├── *.jpg             # RGB images
        ├── *_gt.png          # Ground truth masks
        ├── sam2_predictions.csv
        ├── ground_truth_boxes.csv
        ├── iou_results.csv
        └── iou_analysis.png
```
