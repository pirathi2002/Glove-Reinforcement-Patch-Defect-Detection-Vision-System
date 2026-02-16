"""
Configuration file for glove defect detection project.
Contains all paths, parameters, and model configurations.
"""

# ==================== CRITICAL: SSL/HTTPS CONFIGURATION ====================
# This MUST be at the very top before any imports
import os
import sys

# Comprehensive SSL fix - patches at all levels
os.environ['PYTHONHTTPSVERIFY'] = '0'
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['CURL_CA_BUNDLE'] = ''

# Disable warnings before any imports
import warnings
warnings.filterwarnings('ignore')

try:
    # Patch SSL module before anything uses it
    import ssl
    _orig_create = ssl.create_default_context
    
    def patched_create_context(*args, **kwargs):
        try:
            ctx = _orig_create(*args, **kwargs)
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            return ctx
        except:
            return None
    
    ssl.create_default_context = patched_create_context
    ssl._create_default_https_context = patched_create_context
except Exception as e:
    pass

# Disable urllib3 SSL warnings
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except:
    pass

from pathlib import Path

# ==================== PROJECT PATHS ====================
PROJECT_ROOT = Path(__file__).parent
DATA_ROOT = PROJECT_ROOT / "data"
SRC_ROOT = PROJECT_ROOT / "src"

# Data directories
TRAIN_DIR = DATA_ROOT / "train" / "acceptable"
VALIDATION_DIR = DATA_ROOT / "test"
TRAIN_IMAGES_DIR = DATA_ROOT / "train_images"

# Output directories
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"
HEATMAPS_DIR = RESULTS_DIR / "heatmaps"
METRICS_DIR = RESULTS_DIR / "metrics"

# Create directories if they don't exist
for dir_path in [TRAIN_IMAGES_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR, HEATMAPS_DIR, METRICS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ==================== ROI CONFIGURATION ====================
ROI_CONFIG = {
    'x': 30,
    'y': 60,
    'w': 180,
    'h': 120,
    'debug': False
}

# ==================== PREPROCESSING CONFIGURATION ====================
PREPROCESSING_CONFIG = {
    'target_size': (256, 256),  # Target size for all images
    'normalize': True,
    'clahe': False,  # Contrast Limited Adaptive Histogram Equalization
    'denoise': False,
    'gaussian_blur': False,
    'blur_kernel': (5, 5),
    'save_preprocessed': True
}

# ==================== MODEL CONFIGURATIONS ====================
# List of all 19 Anomalib models to train
ANOMALIB_MODELS = [
    'cfa',
    'cflow',
    'csflow',
    'dfkde',
    'dfm',
    'draem',
    'dsr',
    'efficient_ad',
    'fastflow',
    'fre',
    'ganomaly',
    'padim',
    'patchcore',
    'reverse_distillation',
    'stfpm',
    'supersimplenet',
    'uflow',
    'vlm_ad',
    'ai_vad'  # Additional model
]

# Model-specific configurations
MODEL_CONFIGS = {
    'cfa': {
        'backbone': 'wide_resnet50_2',
        'gamma_c': 1,
        'gamma_d': 1,
    },
    'cflow': {
        'backbone': 'wide_resnet50_2',
        'layers': ['layer2', 'layer3', 'layer4'],
        'decoder': 'freia-cflow',
    },
    'csflow': {
        'cross_conv_hidden_channels': 1024,
        'n_coupling_blocks': 4,
    },
    'dfkde': {
        'backbone': 'resnet18',
        'max_training_points': 40000,
    },
    'dfm': {
        'backbone': 'resnet18',
        'layer': 'layer3',
        'pca_level': 0.97,
    },
    'draem': {
        'enable_sspcab': True,
    },
    'dsr': {
        'latent_anomaly_strength': 0.2,
    },
    'efficient_ad': {
        'teacher_out_channels': 384,
        'model_size': 'medium',
        'padding': False,
    },
    'fastflow': {
        'backbone': 'resnet18',
        'flow_steps': 8,
    },
    'fre': {
        'backbone': 'resnet18',
        'layer': 'layer3',
    },
    'ganomaly': {
        'latent_vec_size': 100,
        'n_features': 64,
    },
    'padim': {
        'backbone': 'resnet18',
        'layers': ['layer1', 'layer2', 'layer3'],
    },
    'patchcore': {
        'backbone': 'wide_resnet50_2',
        'layers': ['layer2', 'layer3'],
        'coreset_sampling_ratio': 0.1,
        'num_neighbors': 9,
    },
    'reverse_distillation': {
        'backbone': 'wide_resnet50_2',
        'layers': ['layer1', 'layer2', 'layer3'],
    },
    'stfpm': {
        'backbone': 'resnet18',
        'layers': ['layer1', 'layer2', 'layer3'],
    },
    'supersimplenet': {
        'backbone': 'resnet18',
        'layers': ['layer1', 'layer2', 'layer3'],
    },
    'uflow': {
        'backbone': 'mcait',
        'flow_steps': 4,
    },
    'vlm_ad': {
        'model_name': 'openai/clip-vit-base-patch32',
        'few_shot_k': 0,
    },
    'ai_vad': {
        'n_scales': 3,
    }
}

# ==================== TRAINING CONFIGURATION ====================
TRAINING_CONFIG = {
    'image_size': (256, 256),
    'num_epochs': 1,
    'batch_size': 2,
    'num_workers': 4,
    'learning_rate': 0.001,
    'early_stopping_patience': 10,
    'early_stopping_delta': 0.001,
    'seed': 42,
    'accelerator': 'auto',  # 'cpu', 'gpu', or 'auto'
    'devices': 1,
    'precision': 32,
    'log_every_n_steps': 10,
}

# ==================== VALIDATION CONFIGURATION ====================
VALIDATION_CONFIG = {
    'threshold_method': 'adaptive',  # 'adaptive' or 'manual'
    'manual_threshold': 0.5,
    'visualization_mode': 'overlay',  # 'overlay', 'side_by_side', or 'both'
    'save_visualizations': True,
    'interactive_mode': True,
}

# ==================== LOGGING CONFIGURATION ====================
LOGGING_CONFIG = {
    'log_level': 'INFO',  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    'log_to_file': True,
    'log_to_console': True,
    'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S',
    'save_images': True,
    'save_metrics': True,
    'save_heatmaps': True,
    'use_tensorboard': True,
    'use_wandb': False,  # Set to True if using Weights & Biases
    'wandb_project': 'glove-defect-detection',
}

# ==================== METRICS CONFIGURATION ====================
METRICS_TO_TRACK = [
    'train_loss',
    'val_loss',
    'image_AUROC',
    'pixel_AUROC',
    'image_F1Score',
    'pixel_F1Score',
    'accuracy',
    'precision',
    'recall',
]

# ==================== DATASET CONFIGURATION ====================
DATASET_CONFIG = {
    'task': 'segmentation',  # 'classification' or 'segmentation'
    'normal_dir': 'acceptable',
    'abnormal_dir': None,  # One-class learning
    'mask_dir': None,
    'extensions': ['.jpg', '.jpeg', '.png', '.bmp'],
    'split_ratio': 0.2,  # For validation split if needed
}

# ==================== VISUALIZATION CONFIGURATION ====================
VISUALIZATION_CONFIG = {
    'figsize': (20, 12),
    'dpi': 100,
    'cmap': 'jet',
    'alpha': 0.5,
    'show_colorbar': True,
    'fontsize': 10,
    'title_fontsize': 12,
}

# ==================== FOLDER STRUCTURE ====================
# Expected number of lighting conditions (folders per glove)
NUM_LIGHTING_CONDITIONS = 16

# ==================== HELPER FUNCTIONS ====================
def get_model_save_path(model_name: str, folder_idx: int) -> Path:
    """Get the save path for a specific model and folder."""
    model_dir = MODELS_DIR / model_name / f"folder_{folder_idx:02d}"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir

def get_results_save_path(model_name: str, folder_idx: int) -> Path:
    """Get the results path for a specific model and folder."""
    results_dir = RESULTS_DIR / model_name / f"folder_{folder_idx:02d}"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir

def get_log_file_path(model_name: str = None) -> Path:
    """Get log file path."""
    if model_name:
        log_file = LOGS_DIR / f"{model_name}.log"
    else:
        log_file = LOGS_DIR / "main.log"
    return log_file

# ==================== GPU CONFIGURATION ====================
GPU_CONFIG = {
    'use_gpu': True,
    'gpu_id': 0,
    'mixed_precision': False,
}

# Print configuration summary
if __name__ == "__main__":
    print("=" * 1)
    print("GLOVE DEFECT DETECTION - CONFIGURATION")
    print("=" * 1)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Root: {DATA_ROOT}")
    print(f"Models Directory: {MODELS_DIR}")
    print(f"Results Directory: {RESULTS_DIR}")
    print(f"Number of Models: {len(ANOMALIB_MODELS)}")
    print(f"Lighting Conditions: {NUM_LIGHTING_CONDITIONS}")
    print(f"Image Size: {TRAINING_CONFIG['image_size']}")
    print(f"Batch Size: {TRAINING_CONFIG['batch_size']}")
    print(f"Epochs: {TRAINING_CONFIG['num_epochs']}")
    print("=" * 1)