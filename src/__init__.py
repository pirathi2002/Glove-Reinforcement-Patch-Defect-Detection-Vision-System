"""
Glove Defect Detection Package
A comprehensive anomaly detection system for glove reinforcement patch defects.
"""

__version__ = "1.0.0"
__author__ = "Pirathishanth"
__description__ = "Anomaly detection for glove defect detection using Anomalib"

from src.logger import ProjectLogger, MetricsTracker, HeatmapSaver
from src.preprocessing import ImagePreprocessor, preprocess_training_data
from src.roi import ROISelector, crop_roi
from src.train_models import (
    AnomalibModelTrainer,
    train_single_model_folder,
    train_model_all_folders,
    train_all_models,
)
from src.validate import ModelValidator, InteractiveValidator
from src.utils import (
    get_folder_structure,
    verify_data_structure,
    collect_all_metrics,
    plot_training_curves,
    compare_models,
    create_summary_report,
)

__all__ = [
    # Logger
    'ProjectLogger',
    'MetricsTracker',
    'HeatmapSaver',
    
    # Preprocessing
    'ImagePreprocessor',
    'preprocess_training_data',
    
    # ROI
    'ROISelector',
    'crop_roi',
    
    # Training
    'AnomalibModelTrainer',
    'train_single_model_folder',
    'train_model_all_folders',
    'train_all_models',
    
    # Validation
    'ModelValidator',
    'InteractiveValidator',
    
    # Utils
    'get_folder_structure',
    'verify_data_structure',
    'collect_all_metrics',
    'plot_training_curves',
    'compare_models',
    'create_summary_report',
]