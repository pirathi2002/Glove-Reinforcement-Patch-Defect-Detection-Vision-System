# Glove Defect Detection using Anomalib

A comprehensive anomaly detection system for detecting defects in glove reinforcement patches (between thumb and index finger) using the Anomalib framework. This project trains 19 different anomaly detection models across 16 lighting conditions and provides interactive visualization of results.
##  Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Main Pipeline (main.py)                  │
└─────────────────────────────────────────────────────────────┘
                              │
                 ┌────────────┼────────────┐
                 │            │            │
                 ▼            ▼            ▼
         ┌──────────┐  ┌──────────┐  ┌──────────┐
         │Preprocess│  │  Train   │  │ Validate │
         └──────────┘  └──────────┘  └──────────┘
                 │            │            │
                 ▼            ▼            ▼
         ┌──────────┐  ┌──────────┐  ┌──────────┐
         │   ROI    │  │  Models  │  │ Heatmaps │
         └──────────┘  └──────────┘  └──────────┘
                              │
                              ▼
                      ┌──────────────┐
                      │   Logger &   │
                      │   Metrics    │
                      └──────────────┘
```
##  Project Overview

This system uses state-of-the-art anomaly detection models to identify defects in glove reinforcement patches. The pipeline includes:

- **Data Preprocessing**: Automated image preprocessing with ROI extraction
- **Multi-Model Training**: 19 different Anomalib models trained across 16 lighting conditions (304 model instances total)
- **Interactive Validation**: Comprehensive heatmap visualization and anomaly score analysis
- **Extensive Logging**: Complete metrics tracking, visualization, and reporting

##  Project Structure

```
glove-defect-detection/
├── config.py                      # Central configuration file
├── main.py                        # Main pipeline orchestrator
├── requirements.txt               # Python dependencies
├── README.md                      # This file
│
├── data/                          # Data directory
│   ├── train/                     # Original training data
│   │   └── acceptable/            # 16 folders (lighting conditions)
│   ├── train_images/              # Preprocessed training images
│   └── test/                      # Validation/test images
│       ├── acceptable/
│       ├── marginal/
│       └── unacceptable/
│
├── src/                           # Source code
│   ├── logger.py                  # Logging utilities
│   ├── preprocessing.py           # Image preprocessing
│   ├── roi.py                     # ROI selection
│   ├── train_models.py            # Model training
│   ├── validate.py                # Validation and visualization
│   └── utils.py                   # Utility functions
│
├── models/                        # Trained model checkpoints
│   ├── patchcore/
│   │   ├── folder_00/
│   │   ├── folder_01/
│   │   └── ...
│   └── ...
│
├── results/                       # Results and outputs
│   ├── heatmaps/                  # Generated heatmaps
│   ├── metrics/                   # Training metrics
│   └── summary_report.json        # Overall summary
│
└── logs/                          # Log files
    └── ...
```

##  Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd glove-defect-detection

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

Place your data in the following structure:
```
data/
└── train/
    └── acceptable/
        ├── folder_01/  # Lighting condition 1
        ├── folder_02/  # Lighting condition 2
        └── ...         # Up to folder_16
```

### 3. Run Preprocessing

```bash
python main.py --preprocess
```

This will:
- Resize all images to 256x256
- Apply optional preprocessing (CLAHE, denoising, etc.)
- Save preprocessed images to `data/train_images/`
- Preserve the 16-folder structure

### 4. Train Models

**Option A: Train a specific model on all folders**
```bash
python main.py --train --model patchcore
```

**Option B: Train all models (sequential)**
```bash
python main.py --train --all
```

**Option C: Train all models with parallel processing**
```bash
python main.py --train --all --parallel-folders --max-workers 4
```

**Option D: Train specific model on specific folder**
```bash
python main.py --train --model patchcore --folder 0
```

### 5. Validate and Visualize

**Interactive mode (recommended)**
```bash
python main.py --validate --interactive
```

This opens an interactive terminal interface where you can:
- Select a glove (set of 16 images)
- Choose which model to visualize (1-19)
- View heatmaps and overlays for all 16 lighting conditions
- Compare anomaly scores across images

**Non-interactive mode**
```bash
python main.py --validate --model patchcore --glove 0
```

### 6. Generate Reports

```bash
python main.py --report
```

##  Configuration

Edit `config.py` to customize:

### ROI Configuration
```python
ROI_CONFIG = {
    'x': 50,
    'y': 50,
    'w': 150,
    'h': 150,
    'debug': False
}
```

### Preprocessing Options
```python
PREPROCESSING_CONFIG = {
    'target_size': (256, 256),
    'normalize': True,
    'clahe': False,  # Contrast enhancement
    'denoise': False,
    'gaussian_blur': False,
}
```

### Training Configuration
```python
TRAINING_CONFIG = {
    'num_epochs': 50,
    'batch_size': 32,
    'learning_rate': 0.001,
    'early_stopping_patience': 10,
}
```

##  Supported Models

The system supports 19 Anomalib models:

1. **CFA** - Coupled-hypersphere-based Feature Adaptation
2. **C-Flow** - Conditional Normalizing Flow
3. **CS-Flow** - Fully Convolutional Cross-Scale-Flows
4. **DFKDE** - Deep Feature Kernel Density Estimation
5. **DFM** - Deep Feature Modeling
6. **DRAEM** - Discriminatively Trained Reconstruction Embedding
7. **DSR** - A Dual Subspace Re-Projection Network
8. **EfficientAD** - Accurate Visual Anomaly Detection
9. **FastFlow** - Fast Normalizing Flow
10. **FRE** - Feature Reconstruction Error
11. **GANomaly** - Generative Adversarial Networks for Anomaly Detection
12. **PaDiM** - Patch Distribution Modeling
13. **PatchCore** - Towards Total Recall in Industrial Anomaly Detection
14. **Reverse Distillation** - Anomaly Detection via Reverse Distillation
15. **STFPM** - Student-Teacher Feature Pyramid Matching
16. **SuperSimpleNet** - Simple and Effective Anomaly Detection
17. **U-Flow** - U-shaped Normalizing Flow
18. **VLM-AD** - Vision-Language Model for Anomaly Detection
19. **AI-VAD** - Adaptive Image Visual Anomaly Detection

Each model is trained 16 times (once per lighting condition folder).

##  Outputs

### Training Outputs

For each model-folder combination:
- **Model checkpoint**: `models/{model_name}/folder_{idx}/last.ckpt`
- **Training metrics**: `results/metrics/{model_name}/folder_{idx}/training_metrics.csv`
- **Validation metrics**: `results/metrics/{model_name}/folder_{idx}/validation_metrics.csv`
- **Summary**: `results/metrics/{model_name}/folder_{idx}/summary.json`

### Validation Outputs

For each validation run:
- **Original images**: `results/heatmaps/{model_name}/folder_{idx}/original/`
- **Heatmaps**: `results/heatmaps/{model_name}/folder_{idx}/heatmap/`
- **Overlays**: `results/heatmaps/{model_name}/folder_{idx}/overlay/`
- **Image results**: `results/metrics/{model_name}/folder_{idx}/image_results.csv`

### Metrics Tracked

- **Image-level**: AUROC, F1-Score, Accuracy, Precision, Recall
- **Pixel-level**: AUROC, F1-Score
- **Training**: Loss curves, learning rate, epoch times
- **Per-image**: Anomaly scores, predictions, thresholds

##  Visualization Features

### Interactive Validation
- Select glove and model interactively
- View all 16 images with their heatmaps
- Side-by-side comparison of original, heatmap, and overlay
- Adjustable threshold and visualization parameters

### Grid Display
- 16 images displayed in a 4x4 grid
- Each image shows: Original | Heatmap | Overlay
- Anomaly scores displayed for each image

### Comparative Analysis
- Compare all 19 models on a specific folder
- Bar charts showing relative performance
- Metrics: AUROC, F1-Score, Accuracy, etc.

##  Advanced Usage

### Full Pipeline Execution

Run the complete pipeline (preprocess → train → report):
```bash
python main.py --full-pipeline
```

### Parallel Training

**Train folders in parallel** (recommended for multi-core systems):
```bash
python main.py --train --all --parallel-folders --max-workers 8
```

**Train models in parallel** (requires significant RAM):
```bash
python main.py --train --all --parallel-models --max-workers 4
```

**Both parallel** (maximum speed, high resource usage):
```bash
python main.py --train --all --parallel-folders --parallel-models --max-workers 4
```

### Model Comparison

Compare all models on a specific folder:
```bash
python main.py --compare-models --folder 0 --metric image_AUROC
```

Compare all models across all folders:
```bash
python main.py --compare-models --metric image_AUROC
```

### Custom ROI Selection

To interactively select ROI:
```python
from src.roi import ROISelector
from pathlib import Path

selector = ROISelector()
roi_coords = selector.interactive_roi_selection(Path("path/to/sample/image.jpg"))
print(f"Selected ROI: {roi_coords}")
# Update config.py with the new coordinates
```

##  Troubleshooting

### Common Issues

**1. Out of Memory during training**
- Reduce batch size in `config.py`
- Train models sequentially instead of in parallel
- Use smaller image sizes

**2. CUDA out of memory**
- Set `accelerator='cpu'` in `TRAINING_CONFIG`
- Reduce batch size
- Train one model at a time

**3. No checkpoints found**
- Ensure training completed successfully
- Check `logs/` directory for error messages
- Verify model save directory exists

**4. Images not loading**
- Check file extensions match config
- Verify data directory structure
- Run `python main.py --verify`

##  Logging

All operations are comprehensively logged:

- **Console output**: Real-time progress and status
- **Log files**: Detailed logs in `logs/` directory
- **Metrics CSV**: Epoch-by-epoch metrics
- **Summary JSON**: Final results and statistics

To adjust logging level, edit `config.py`:
```python
LOGGING_CONFIG = {
    'log_level': 'INFO',  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    'log_to_file': True,
    'log_to_console': True,
}
```

##  Testing

Test individual components:

```bash
# Test preprocessing
python src/preprocessing.py

# Test ROI selection
python src/roi.py

# Test logger
python src/logger.py

# Test utilities
python src/utils.py
```

##  Model-Specific Notes

### PatchCore
- Best for texture-based defects
- Fast inference
- No training required (uses coreset)

### EfficientAD
- Good balance of speed and accuracy
- Lower memory footprint
- Good for real-time applications

### DRAEM
- Synthetic anomaly generation
- Works well with limited normal samples
- Longer training time

### STFPM
- Knowledge distillation approach
- Good generalization
- Requires more epochs

##  Workflow Summary

```
1. Data Collection
   └─> Place images in data/train/acceptable/

2. Preprocessing
   └─> python main.py --preprocess

3. Training
   └─> python main.py --train --all

4. Validation
   └─> python main.py --validate --interactive

5. Analysis
   └─> python main.py --report
   └─> python main.py --compare-models
```

##  Contributing

When modifying the code:
1. Follow the existing code structure
2. Add docstrings to all functions
3. Update `config.py` for new parameters
4. Test with `python main.py --verify`
5. Update this README if adding features

##  License

[Your License Here]

##  Acknowledgments

- **Anomalib**: https://github.com/openvinotoolkit/anomalib
- Built with PyTorch, OpenCV, and NumPy

##  Contact

[Your Contact Information]

---

**Note**: This README assumes you have the necessary hardware and dependencies installed. For production deployment, consider containerization with Docker for reproducibility.