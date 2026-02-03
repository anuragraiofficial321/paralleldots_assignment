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

## Methodology

### Tracking Approach
1. Extract initial bounding box from ground truth mask in first frame
2. Use SAM2 video predictor to track object to next frame
3. Extract new bounding box from predicted segmentation mask
4. Repeat for all subsequent frames
5. Save predictions at each step

### Evaluation Strategy
- Match predictions with ground truth by filename normalization
- Calculate IoU for each prediction-GT pair
- Aggregate statistics by object class
- Identify performance patterns

### Challenges Solved
- **Filename Mismatch**: Ground truth and predictions had different naming conventions - fixed with regex normalization
- **DataFrame Column Handling**: Merge operations created suffix columns - resolved with explicit column mapping
- **Error Recovery**: Added try-catch blocks to handle missing masks and corrupted data

## Lessons Learned

### What I Discovered
- Object texture and color contrast significantly impact tracking performance
- SAM2 works better with distinct, well-defined objects
- Cylindrical objects with similar appearance are harder to track reliably
- Initial bounding box quality affects all subsequent frames

### Potential Improvements
- Implement temporal smoothing to reduce jitter
- Use multiple detection models for difficult objects
- Add re-initialization logic when tracking confidence drops
- Explore fine-tuning SAM2 on similar object categories

## Visualization Examples

The analysis includes:
- **IoU Distribution Histogram** - Shows overall tracking quality spread
- **Per-Object Performance Bar Chart** - Compare object classes
- **Box Plots by Class** - See variance within each category
- **Cumulative Distribution** - Understand performance thresholds

## Future Work

Possible extensions:
- [ ] Implement real-time tracking on live video
- [ ] Add support for multiple simultaneous objects
- [ ] Explore different SAM2 model sizes (base, large)
- [ ] Compare with other tracking methods (SORT, DeepSORT)
- [ ] Apply to different datasets (COCO, VOC)

## Acknowledgments

- **SAM2**: Meta's Segment Anything Model 2 for powerful segmentation
- **CMU Dataset**: Carnegie Mellon's 3D object recognition dataset
- **Assignment**: Thanks to my instructor for the interesting challenge

## License

This project is for educational purposes. Please check individual component licenses:
- SAM2: Apache 2.0
- CMU Dataset: Check dataset terms

## Contact

Feel free to reach out if you have questions or suggestions!

---

**Note**: This was developed as part of a data science course assignment focusing on computer vision and object tracking. The solution demonstrates practical application of state-of-the-art segmentation models for tracking tasks.

## Quick Start

```bash
# Clone the repo
git clone <your-repo-url>

# Install dependencies
pip install torch torchvision opencv-python matplotlib pandas numpy pillow

# Install SAM2
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2 && pip install -e . && cd ..

# Download dataset
wget http://www.cs.cmu.edu/~ehsiao/3drecognition/CMU10_3D.zip
unzip CMU10_3D.zip

# Download model
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt

# Open and run the notebook
jupyter notebook ds_solution.ipynb
```
