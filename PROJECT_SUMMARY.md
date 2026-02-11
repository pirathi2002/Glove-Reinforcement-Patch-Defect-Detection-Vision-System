# Glove Defect Detection - Project Implementation Summary

##  Overview

This document provides a comprehensive overview of the glove defect detection system implementation, detailing all components, features, and technical specifications.

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

### Module Breakdown

#### 1. **config.py** - Central Configuration
- All project paths and directories
- ROI coordinates and settings
- Preprocessing parameters
- Training hyperparameters
- Model-specific configurations
- Logging and visualization settings

#### 2. **src/logger.py** - Comprehensive Logging
**Classes:**
- `ProjectLogger`: Main logging interface
  - Console and file output
  - Multiple log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
  - Exception tracking with full traceback
  
- `MetricsTracker`: Metrics collection and storage
  - Training metrics (loss, AUROC, F1, etc.)
  - Validation metrics
  - Per-image results
  - CSV export
  - Summary statistics
  
- `HeatmapSaver`: Visualization storage
  - Original images
  - Heatmaps
  - Overlays
  - Organized directory structure

#### 3. **src/preprocessing.py** - Image Preprocessing
**Classes:**
- `ImagePreprocessor`: Main preprocessing pipeline
  
**Features:**
- Image loading and format conversion
- Resizing to target dimensions
- Normalization (0-1 range)
- Optional CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Optional denoising (fastNlMeansDenoising)
- Optional Gaussian blur
- Batch processing with progress bars
- Preserves folder structure

**Functions:**
- `preprocess_training_data()`: Batch preprocess entire dataset
- `preprocess_single_image()`: Single image preprocessing

#### 4. **src/roi.py** - Region of Interest Selection
**Classes:**
- `ROISelector`: ROI extraction and management
  
**Features:**
- Fixed rectangular ROI cropping
- Configurable coordinates (x, y, w, h)
- Debug visualization
- Interactive ROI selection (mouse-based)
- Validation of ROI bounds

**Functions:**
- `crop_roi()`: Convenience function for ROI extraction
- `process_image_with_roi()`: Complete pipeline (load → preprocess → crop)

#### 5. **src/train_models.py** - Model Training
**Classes:**
- `AnomalibModelTrainer`: Individual model trainer
  
**Features:**
- 19 Anomalib model support
- Model-specific hyperparameters
- Automatic checkpoint saving
- Training and validation loops
- Metrics logging
- Early stopping support
- GPU/CPU acceleration

**Functions:**
- `train_single_model_folder()`: Train one model on one folder
- `train_model_all_folders()`: Train one model on all folders
- `train_all_models()`: Train all models on all folders
- Parallel training support (folder-level and model-level)

**Supported Models (19 total):**
1. CFA - Coupled-hypersphere Feature Adaptation
2. C-Flow - Conditional Normalizing Flow
3. CS-Flow - Cross-Scale Flows
4. DFKDE - Deep Feature KDE
5. DFM - Deep Feature Modeling
6. DRAEM - Discriminative Reconstruction
7. DSR - Dual Subspace Re-Projection
8. EfficientAD - Efficient Anomaly Detection
9. FastFlow - Fast Normalizing Flow
10. FRE - Feature Reconstruction Error
11. GANomaly - GAN-based Anomaly Detection
12. PaDiM - Patch Distribution Modeling
13. PatchCore - Memory Bank Approach
14. Reverse Distillation - Student-Teacher
15. STFPM - Feature Pyramid Matching
16. SuperSimpleNet - Lightweight Detection
17. U-Flow - U-shaped Flow
18. VLM-AD - Vision-Language Model
19. AI-VAD - Adaptive Image VAD

#### 6. **src/validate.py** - Validation and Visualization
**Classes:**
- `ModelValidator`: Single model validation
  - Checkpoint loading
  - Image prediction
  - Heatmap generation
  - Overlay creation
  - Grid visualization
  
- `InteractiveValidator`: Terminal-based interface
  - Glove selection
  - Model selection
  - Real-time visualization
  - Batch processing

**Features:**
- Anomaly score calculation
- Heatmap generation with colormaps
- Overlay transparency control
- Grid layout (4x4 for 16 images)
- Side-by-side comparisons
- Interactive terminal mode

#### 7. **src/utils.py** - Utility Functions
**Functions:**
- `get_folder_structure()`: Analyze directory structure
- `verify_data_structure()`: Validate data organization
- `collect_all_metrics()`: Aggregate metrics from all models
- `plot_training_curves()`: Visualize training progress
- `compare_models()`: Cross-model performance comparison
- `create_summary_report()`: Generate comprehensive report
- `cleanup_temp_files()`: Clean temporary files

##  Data Flow

### Training Pipeline

```
Raw Images → Preprocessing → ROI Extraction → Training → Model Checkpoints
     │            │               │              │            │
     │            │               │              │            └─→ Metrics
     │            │               │              └─→ Validation
     │            │               └─→ Augmentation
     │            └─→ Normalization
     └─→ Quality Check
```

### Validation Pipeline

```
Test Images → Preprocessing → Model Inference → Anomaly Maps
     │            │                  │               │
     │            │                  │               └─→ Heatmaps
     │            │                  └─→ Scores      └─→ Overlays
     │            └─→ ROI Extraction                 └─→ Visualization
     └─→ Load
```

##  Configuration System

### Hierarchical Configuration

1. **Global Config** (`config.py`)
   - Project-wide settings
   - Default values
   
2. **Model-Specific Config** (`MODEL_CONFIGS`)
   - Per-model hyperparameters
   - Architecture settings
   
3. **Runtime Arguments**
   - Command-line overrides
   - Dynamic parameters

### Key Configuration Sections

```python
# ROI Configuration
ROI_CONFIG = {
    'x': 50, 'y': 50, 'w': 150, 'h': 150,
    'debug': False
}

# Training Configuration
TRAINING_CONFIG = {
    'num_epochs': 50,
    'batch_size': 32,
    'learning_rate': 0.001,
    'early_stopping_patience': 10,
}

# Model Configurations
MODEL_CONFIGS = {
    'patchcore': {
        'backbone': 'wide_resnet50_2',
        'coreset_sampling_ratio': 0.1,
    },
    # ... 18 more models
}
```

##  Metrics and Logging

### Tracked Metrics

**Training Metrics:**
- Loss (train/validation)
- Learning rate
- Epoch time
- Gradient norms

**Image-Level Metrics:**
- AUROC (Area Under ROC Curve)
- F1-Score
- Accuracy
- Precision
- Recall

**Pixel-Level Metrics:**
- Pixel AUROC
- Pixel F1-Score
- IoU (Intersection over Union)

**Per-Image Outputs:**
- Anomaly score
- Prediction (normal/anomalous)
- Threshold used
- Confidence score

### Logging Hierarchy

```
logs/
├── main.log                    # Main pipeline
├── Train_{model}_{folder}.log  # Per-model training
├── Validate_{model}_{folder}.log
└── experiments/
    └── {experiment_name}/
        ├── config.json
        └── {experiment_name}.log
```

### Metrics Storage

```
results/metrics/{model}/folder_{idx}/
├── training_metrics.csv        # Epoch-by-epoch training
├── validation_metrics.csv      # Validation results
├── image_results.csv           # Per-image predictions
└── summary.json                # Final summary
```

##  Visualization System

### Heatmap Generation

**Process:**
1. Model generates anomaly map (spatial anomaly scores)
2. Normalize to [0, 1] range
3. Resize to match original image
4. Apply colormap (default: JET)
5. Create overlay with original image

**Output Formats:**
- Original image (RGB)
- Heatmap (colormap applied)
- Overlay (alpha-blended)
- Grid view (all 16 images)

### Interactive Visualization

**Terminal Interface:**
```
Select glove: [0-N]
Select model: [1-19]
→ Displays grid with all 16 lighting conditions
→ Shows original | heatmap | overlay
→ Includes anomaly scores
```

##  Performance Optimization

### Parallel Processing

**Folder-Level Parallelism:**
```python
# Train 16 folders in parallel
train_model_all_folders(
    'patchcore',
    parallel=True,
    max_workers=4
)
```

**Model-Level Parallelism:**
```python
# Train multiple models simultaneously
train_all_models(
    parallel_models=True,
    max_workers=4
)
```

### Memory Management

- Batch processing for large datasets
- Automatic garbage collection
- Checkpoint-based training (resume capability)
- Gradient accumulation support

### GPU Optimization

- Automatic GPU detection
- Mixed precision training support
- Distributed training ready
- CPU fallback

##  Error Handling

### Exception Management

**Levels:**
1. **Try-Catch Blocks**: All critical operations
2. **Logging**: All exceptions logged with context
3. **Graceful Degradation**: Continue on non-critical errors
4. **User Feedback**: Clear error messages

**Example:**
```python
try:
    result = train_model()
except Exception as e:
    logger.log_exception(e, "train_model")
    return {'status': 'failed', 'error': str(e)}
```

##  File Organization

### Model Checkpoints

```
models/{model_name}/folder_{idx}/
├── last.ckpt           # Latest checkpoint
├── best.ckpt           # Best validation score
└── checkpoints/
    ├── epoch_10.ckpt
    └── ...
```

### Results Structure

```
results/
├── heatmaps/{model}/{folder}/
│   ├── original/       # Original images
│   ├── heatmap/        # Heatmaps
│   └── overlay/        # Overlays
├── metrics/{model}/{folder}/
│   ├── training_metrics.csv
│   ├── validation_metrics.csv
│   └── image_results.csv
└── summary_report.json
```

##  Testing and Validation

### Unit Tests

Each module includes standalone tests:
```bash
python src/preprocessing.py  # Test preprocessing
python src/roi.py            # Test ROI selection
python src/logger.py         # Test logging
```

### Integration Tests

```bash
python examples/quick_start.py  # Run all examples
python main.py --verify        # Verify data structure
```

### Validation Metrics

- Data structure validation
- Configuration validation
- Model checkpoint verification
- Output quality checks

##  Workflow Integration

### Command-Line Interface

**Basic Commands:**
```bash
python main.py --preprocess
python main.py --train --model patchcore
python main.py --validate --interactive
python main.py --report
```

**Advanced Commands:**
```bash
python main.py --full-pipeline
python main.py --train --all --parallel-folders --max-workers 8
python main.py --compare-models --metric image_AUROC
```

### Python API

```python
from src import (
    preprocess_training_data,
    train_model_all_folders,
    InteractiveValidator
)

# Preprocess
preprocess_training_data()

# Train
train_model_all_folders('patchcore')

# Validate
validator = InteractiveValidator()
validator.run_interactive_mode()
```

##  Dependencies

### Core Dependencies

- **anomalib** >= 1.0.0: Anomaly detection framework
- **torch** >= 2.0.0: Deep learning
- **torchvision** >= 0.15.0: Vision models
- **opencv-python** >= 4.8.0: Image processing
- **numpy** >= 1.24.0: Numerical computing
- **pandas** >= 2.0.0: Data analysis
- **matplotlib** >= 3.7.0: Visualization

### Optional Dependencies

- **tensorboard**: Training visualization
- **wandb**: Experiment tracking
- **albumentations**: Advanced augmentation

##  Future Enhancements

### Potential Additions

1. **Real-time Processing**: Live camera feed support
2. **Model Ensemble**: Combine multiple models
3. **Auto-tuning**: Hyperparameter optimization
4. **Web Interface**: Browser-based visualization
5. **Mobile Deployment**: Edge device support
6. **Active Learning**: Iterative improvement
7. **Explainability**: SHAP/LIME integration
8. **Multi-GPU**: Distributed training

##  Usage Scenarios

### Scenario 1: Single Model Evaluation
```bash
# Train one model
python main.py --train --model patchcore

# Validate
python main.py --validate --model patchcore --glove 0
```

### Scenario 2: Model Comparison
```bash
# Train multiple models
python main.py --train --all

# Compare
python main.py --compare-models --metric image_AUROC
```

### Scenario 3: Production Deployment
```bash
# Train best model
python main.py --train --model patchcore --parallel-folders

# Generate report
python main.py --report

# Deploy checkpoint from models/patchcore/folder_XX/best.ckpt
```

##  Learning Resources

### Understanding the Code

1. Start with `config.py` - understand all settings
2. Read `src/preprocessing.py` - data pipeline
3. Study `src/train_models.py` - training loop
4. Explore `src/validate.py` - inference and visualization

### Extending the System

1. Add new model: Update `MODEL_CLASS_MAP` in `train_models.py`
2. Add preprocessing step: Extend `ImagePreprocessor` class
3. Add metric: Update `METRICS_TO_TRACK` in config
4. Add visualization: Extend `ModelValidator` class

---

**Version:** 1.0.0  
**Last Updated:** 2025  
**Maintainer:** [Your Name]