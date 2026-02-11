"""
Comprehensive logging utility for glove defect detection project.
Handles console and file logging, metrics tracking, and visualization saving.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import csv
from datetime import datetime
import pandas as pd
import numpy as np
import cv2

from config import LOGGING_CONFIG, LOGS_DIR, METRICS_DIR, HEATMAPS_DIR


class ProjectLogger:
    """
    Custom logger for the glove defect detection project.
    Supports multiple logging levels, file/console output, and metrics tracking.
    """
    
    def __init__(self, name: str = "GloveDefectDetection", log_file: Optional[Path] = None):
        """
        Initialize the project logger.
        
        Args:
            name: Logger name
            log_file: Path to log file. If None, uses default from config.
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, LOGGING_CONFIG['log_level']))
        
        # Remove existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # Formatter
        formatter = logging.Formatter(
            fmt=LOGGING_CONFIG['log_format'],
            datefmt=LOGGING_CONFIG['date_format']
        )
        
        # Console handler
        if LOGGING_CONFIG['log_to_console']:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, LOGGING_CONFIG['log_level']))
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if LOGGING_CONFIG['log_to_file']:
            if log_file is None:
                log_file = LOGS_DIR / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(getattr(logging, LOGGING_CONFIG['log_level']))
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
            self.log_file = log_file
            self.info(f"Logging to file: {log_file}")
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Log critical message."""
        self.logger.critical(message)
    
    def log_exception(self, exception: Exception, context: str = ""):
        """
        Log exception with full traceback.
        
        Args:
            exception: Exception object
            context: Additional context string
        """
        if context:
            self.error(f"Exception in {context}: {str(exception)}")
        else:
            self.error(f"Exception: {str(exception)}")
        
        self.logger.exception(exception)
    
    def log_config(self, config: Dict[str, Any], config_name: str = "Configuration"):
        """
        Log configuration dictionary.
        
        Args:
            config: Configuration dictionary
            config_name: Name of the configuration
        """
        self.info(f"{config_name}:")
        for key, value in config.items():
            self.info(f"  {key}: {value}")
    
    def log_metrics(self, metrics: Dict[str, float], epoch: Optional[int] = None, 
                    model_name: Optional[str] = None, folder_idx: Optional[int] = None):
        """
        Log metrics dictionary.
        
        Args:
            metrics: Dictionary of metric names and values
            epoch: Optional epoch number
            model_name: Optional model name
            folder_idx: Optional folder index
        """
        prefix = ""
        if model_name:
            prefix += f"[{model_name}]"
        if folder_idx is not None:
            prefix += f"[Folder {folder_idx:02d}]"
        if epoch is not None:
            prefix += f"[Epoch {epoch}]"
        
        self.info(f"{prefix} Metrics:")
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, float):
                self.info(f"  {metric_name}: {metric_value:.4f}")
            else:
                self.info(f"  {metric_name}: {metric_value}")


class MetricsTracker:
    """
    Track and save metrics across training and validation.
    """
    
    def __init__(self, model_name: str, folder_idx: int, save_dir: Optional[Path] = None):
        """
        Initialize metrics tracker.
        
        Args:
            model_name: Name of the model
            folder_idx: Folder index (lighting condition)
            save_dir: Directory to save metrics
        """
        self.model_name = model_name
        self.folder_idx = folder_idx
        
        if save_dir is None:
            save_dir = METRICS_DIR / model_name / f"folder_{folder_idx:02d}"
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics storage
        self.training_metrics: List[Dict[str, Any]] = []
        self.validation_metrics: List[Dict[str, Any]] = []
        self.image_level_results: List[Dict[str, Any]] = []
        
        # CSV files
        self.train_csv = self.save_dir / "training_metrics.csv"
        self.val_csv = self.save_dir / "validation_metrics.csv"
        self.results_csv = self.save_dir / "image_results.csv"
        
        self.logger = ProjectLogger(f"MetricsTracker_{model_name}_{folder_idx}")
    
    def add_training_metrics(self, epoch: int, metrics: Dict[str, float]):
        """
        Add training metrics for an epoch.
        
        Args:
            epoch: Epoch number
            metrics: Dictionary of metrics
        """
        try:
            record = {'epoch': epoch, 'timestamp': datetime.now().isoformat()}
            record.update(metrics)
            self.training_metrics.append(record)
            
            # Save to CSV
            self._save_to_csv(self.train_csv, [record])
            
        except Exception as e:
            self.logger.log_exception(e, "add_training_metrics")
            raise
    
    def add_validation_metrics(self, epoch: int, metrics: Dict[str, float]):
        """
        Add validation metrics for an epoch.
        
        Args:
            epoch: Epoch number
            metrics: Dictionary of metrics
        """
        try:
            record = {'epoch': epoch, 'timestamp': datetime.now().isoformat()}
            record.update(metrics)
            self.validation_metrics.append(record)
            
            # Save to CSV
            self._save_to_csv(self.val_csv, [record])
            
        except Exception as e:
            self.logger.log_exception(e, "add_validation_metrics")
            raise
    
    def add_image_result(self, image_path: str, anomaly_score: float, 
                        prediction: str, threshold: float, **kwargs):
        """
        Add result for a single image.
        
        Args:
            image_path: Path to the image
            anomaly_score: Anomaly score
            prediction: 'normal' or 'anomalous'
            threshold: Detection threshold used
            **kwargs: Additional fields (e.g., ground_truth, pixel_scores)
        """
        try:
            record = {
                'image_path': str(image_path),
                'anomaly_score': anomaly_score,
                'prediction': prediction,
                'threshold': threshold,
                'timestamp': datetime.now().isoformat()
            }
            record.update(kwargs)
            self.image_level_results.append(record)
            
            # Save to CSV
            self._save_to_csv(self.results_csv, [record])
            
        except Exception as e:
            self.logger.log_exception(e, "add_image_result")
            raise
    
    def _save_to_csv(self, csv_path: Path, records: List[Dict[str, Any]]):
        """
        Save records to CSV file.
        
        Args:
            csv_path: Path to CSV file
            records: List of record dictionaries
        """
        try:
            if not records:
                return
            
            # Check if file exists
            file_exists = csv_path.exists()
            
            # Get all unique keys from records
            fieldnames = set()
            for record in records:
                fieldnames.update(record.keys())
            fieldnames = sorted(list(fieldnames))
            
            # Append to CSV
            with open(csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                
                writer.writerows(records)
                
        except Exception as e:
            self.logger.log_exception(e, f"_save_to_csv: {csv_path}")
            raise
    
    def save_summary(self):
        """Save a summary of all metrics to JSON."""
        try:
            summary = {
                'model_name': self.model_name,
                'folder_idx': self.folder_idx,
                'timestamp': datetime.now().isoformat(),
                'training_metrics': self.training_metrics,
                'validation_metrics': self.validation_metrics,
                'num_images_processed': len(self.image_level_results),
            }
            
            # Calculate summary statistics
            if self.validation_metrics:
                last_metrics = self.validation_metrics[-1]
                summary['final_metrics'] = last_metrics
            
            if self.image_level_results:
                scores = [r['anomaly_score'] for r in self.image_level_results]
                summary['score_statistics'] = {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'min': float(np.min(scores)),
                    'max': float(np.max(scores)),
                }
            
            # Save to JSON
            summary_path = self.save_dir / "summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            self.logger.info(f"Summary saved to {summary_path}")
            
        except Exception as e:
            self.logger.log_exception(e, "save_summary")
            raise
    
    def get_dataframe(self, metric_type: str = 'training') -> pd.DataFrame:
        """
        Get metrics as a pandas DataFrame.
        
        Args:
            metric_type: 'training', 'validation', or 'results'
            
        Returns:
            DataFrame with metrics
        """
        try:
            if metric_type == 'training':
                return pd.DataFrame(self.training_metrics)
            elif metric_type == 'validation':
                return pd.DataFrame(self.validation_metrics)
            elif metric_type == 'results':
                return pd.DataFrame(self.image_level_results)
            else:
                raise ValueError(f"Unknown metric type: {metric_type}")
                
        except Exception as e:
            self.logger.log_exception(e, "get_dataframe")
            raise


class HeatmapSaver:
    """
    Save heatmaps and visualization images.
    """
    
    def __init__(self, model_name: str, folder_idx: int, save_dir: Optional[Path] = None):
        """
        Initialize heatmap saver.
        
        Args:
            model_name: Name of the model
            folder_idx: Folder index
            save_dir: Directory to save heatmaps
        """
        self.model_name = model_name
        self.folder_idx = folder_idx
        
        if save_dir is None:
            save_dir = HEATMAPS_DIR / model_name / f"folder_{folder_idx:02d}"
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.original_dir = self.save_dir / "original"
        self.heatmap_dir = self.save_dir / "heatmap"
        self.overlay_dir = self.save_dir / "overlay"
        
        for dir_path in [self.original_dir, self.heatmap_dir, self.overlay_dir]:
            dir_path.mkdir(exist_ok=True)
        
        self.logger = ProjectLogger(f"HeatmapSaver_{model_name}_{folder_idx}")
    
    def save_visualization(self, image_name: str, original: np.ndarray, 
                          heatmap: np.ndarray, overlay: np.ndarray):
        """
        Save original image, heatmap, and overlay.
        
        Args:
            image_name: Name of the image (without extension)
            original: Original image (RGB)
            heatmap: Heatmap image
            overlay: Overlay image
        """
        try:
            # Save original
            original_path = self.original_dir / f"{image_name}.png"
            cv2.imwrite(str(original_path), cv2.cvtColor(original, cv2.COLOR_RGB2BGR))
            
            # Save heatmap
            heatmap_path = self.heatmap_dir / f"{image_name}_heatmap.png"
            cv2.imwrite(str(heatmap_path), heatmap)
            
            # Save overlay
            overlay_path = self.overlay_dir / f"{image_name}_overlay.png"
            cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            
            self.logger.debug(f"Saved visualizations for {image_name}")
            
        except Exception as e:
            self.logger.log_exception(e, f"save_visualization: {image_name}")
            raise


def setup_experiment_logging(experiment_name: str, config: Dict[str, Any]) -> ProjectLogger:
    """
    Set up logging for an experiment.
    
    Args:
        experiment_name: Name of the experiment
        config: Configuration dictionary
        
    Returns:
        ProjectLogger instance
    """
    try:
        # Create experiment directory
        exp_dir = LOGS_DIR / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        log_file = exp_dir / f"{experiment_name}.log"
        logger = ProjectLogger(experiment_name, log_file)
        
        # Log configuration
        logger.info("=" * 80)
        logger.info(f"Experiment: {experiment_name}")
        logger.info("=" * 80)
        logger.log_config(config, "Experiment Configuration")
        
        # Save configuration to JSON
        config_path = exp_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        logger.info(f"Configuration saved to {config_path}")
        
        return logger
        
    except Exception as e:
        print(f"Error setting up experiment logging: {e}")
        raise


if __name__ == "__main__":
    # Test logger
    logger = ProjectLogger("Test")
    logger.info("Test info message")
    logger.warning("Test warning message")
    logger.error("Test error message")
    
    # Test metrics tracker
    tracker = MetricsTracker("test_model", 0)
    tracker.add_training_metrics(1, {'loss': 0.5, 'auroc': 0.85})
    tracker.add_validation_metrics(1, {'val_loss': 0.45, 'val_auroc': 0.88})
    tracker.add_image_result('test.jpg', 0.75, 'anomalous', 0.5)
    tracker.save_summary()
    
    print("Logger test completed successfully!")