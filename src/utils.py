"""
Utility functions for glove defect detection project.
Contains helper functions for data management, visualization, and analysis.
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
import shutil

from config import (
    TRAIN_IMAGES_DIR,
    VALIDATION_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    METRICS_DIR,
    NUM_LIGHTING_CONDITIONS,
)
from src.logger import ProjectLogger


logger = ProjectLogger("Utils")


def get_folder_structure(root_dir: Path) -> Dict:
    """
    Get the folder structure of a directory.
    
    Args:
        root_dir: Root directory path
        
    Returns:
        Dictionary with folder structure information
    """
    try:
        structure = {
            'root': str(root_dir),
            'folders': [],
            'total_images': 0,
        }
        
        if not root_dir.exists():
            logger.warning(f"Directory does not exist: {root_dir}")
            return structure
        
        # Get all subdirectories
        folders = sorted([d for d in root_dir.iterdir() if d.is_dir()])
        
        for folder in folders:
            # Count images in folder
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            image_count = 0
            for ext in image_extensions:
                image_count += len(list(folder.glob(f'*{ext}')))
                image_count += len(list(folder.glob(f'*{ext.upper()}')))
            
            structure['folders'].append({
                'name': folder.name,
                'path': str(folder),
                'image_count': image_count,
            })
            
            structure['total_images'] += image_count
        
        return structure
        
    except Exception as e:
        logger.log_exception(e, "get_folder_structure")
        raise


def verify_data_structure(expected_folders: int = NUM_LIGHTING_CONDITIONS) -> bool:
    """
    Verify that the data structure is correct.
    
    Args:
        expected_folders: Expected number of folders
        
    Returns:
        True if structure is valid, False otherwise
    """
    try:
        logger.info("Verifying data structure...")
        
        # Check train_images directory
        if not TRAIN_IMAGES_DIR.exists():
            logger.error(f"Train images directory does not exist: {TRAIN_IMAGES_DIR}")
            return False
        
        # Get folder structure
        structure = get_folder_structure(TRAIN_IMAGES_DIR)
        
        # Check number of folders
        if len(structure['folders']) != expected_folders:
            logger.error(f"Expected {expected_folders} folders, found {len(structure['folders'])}")
            return False
        
        # Check that each folder has images
        for folder_info in structure['folders']:
            if folder_info['image_count'] == 0:
                logger.error(f"Folder {folder_info['name']} has no images")
                return False
        
        logger.info(f"✓ Data structure valid: {len(structure['folders'])} folders, {structure['total_images']} images")
        
        return True
        
    except Exception as e:
        logger.log_exception(e, "verify_data_structure")
        return False


def collect_all_metrics(model_name: Optional[str] = None) -> pd.DataFrame:
    """
    Collect all metrics from CSV files.
    
    Args:
        model_name: If provided, collect only for this model
        
    Returns:
        DataFrame with all metrics
    """
    try:
        all_metrics = []
        
        # Get model directories
        if model_name:
            model_dirs = [METRICS_DIR / model_name]
        else:
            model_dirs = [d for d in METRICS_DIR.iterdir() if d.is_dir()]
        
        for model_dir in model_dirs:
            model_name_current = model_dir.name
            
            # Get folder directories
            folder_dirs = sorted([d for d in model_dir.iterdir() if d.is_dir()])
            
            for folder_dir in folder_dirs:
                # Extract folder index
                folder_idx = int(folder_dir.name.split('_')[-1])
                
                # Read metrics CSVs
                train_csv = folder_dir / "training_metrics.csv"
                val_csv = folder_dir / "validation_metrics.csv"
                
                if train_csv.exists():
                    df_train = pd.read_csv(train_csv)
                    df_train['model'] = model_name_current
                    df_train['folder_idx'] = folder_idx
                    df_train['metric_type'] = 'training'
                    all_metrics.append(df_train)
                
                if val_csv.exists():
                    df_val = pd.read_csv(val_csv)
                    df_val['model'] = model_name_current
                    df_val['folder_idx'] = folder_idx
                    df_val['metric_type'] = 'validation'
                    all_metrics.append(df_val)
        
        if all_metrics:
            return pd.concat(all_metrics, ignore_index=True)
        else:
            logger.warning("No metrics found")
            return pd.DataFrame()
        
    except Exception as e:
        logger.log_exception(e, "collect_all_metrics")
        raise


def plot_training_curves(model_name: str, folder_idx: int, save_path: Optional[Path] = None):
    """
    Plot training curves for a specific model and folder.
    
    Args:
        model_name: Model name
        folder_idx: Folder index
        save_path: Path to save the plot
    """
    try:
        # Load metrics
        metrics_dir = METRICS_DIR / model_name / f"folder_{folder_idx:02d}"
        train_csv = metrics_dir / "training_metrics.csv"
        val_csv = metrics_dir / "validation_metrics.csv"
        
        if not train_csv.exists():
            logger.error(f"Training metrics not found: {train_csv}")
            return
        
        df_train = pd.read_csv(train_csv)
        df_val = pd.read_csv(val_csv) if val_csv.exists() else None
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"{model_name.upper()} - Folder {folder_idx:02d} - Training Curves", 
                    fontsize=14, fontweight='bold')
        
        # Plot loss
        if 'train_loss' in df_train.columns:
            axes[0, 0].plot(df_train['epoch'], df_train['train_loss'], label='Train Loss')
        if df_val is not None and 'val_loss' in df_val.columns:
            axes[0, 0].plot(df_val['epoch'], df_val['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot AUROC
        if df_val is not None:
            if 'image_AUROC' in df_val.columns:
                axes[0, 1].plot(df_val['epoch'], df_val['image_AUROC'], label='Image AUROC')
            if 'pixel_AUROC' in df_val.columns:
                axes[0, 1].plot(df_val['epoch'], df_val['pixel_AUROC'], label='Pixel AUROC')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('AUROC')
        axes[0, 1].set_title('AUROC')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot F1 Score
        if df_val is not None:
            if 'image_F1Score' in df_val.columns:
                axes[1, 0].plot(df_val['epoch'], df_val['image_F1Score'], label='Image F1')
            if 'pixel_F1Score' in df_val.columns:
                axes[1, 0].plot(df_val['epoch'], df_val['pixel_F1Score'], label='Pixel F1')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_title('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot accuracy
        if df_val is not None and 'accuracy' in df_val.columns:
            axes[1, 1].plot(df_val['epoch'], df_val['accuracy'], label='Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title('Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
    except Exception as e:
        logger.log_exception(e, "plot_training_curves")
        raise


def compare_models(folder_idx: int, metric: str = 'image_AUROC', save_path: Optional[Path] = None):
    """
    Compare all models for a specific folder on a given metric.
    
    Args:
        folder_idx: Folder index
        metric: Metric to compare
        save_path: Path to save the plot
    """
    try:
        # Collect metrics for all models
        all_scores = []
        
        for model_dir in METRICS_DIR.iterdir():
            if not model_dir.is_dir():
                continue
            
            model_name = model_dir.name
            folder_dir = model_dir / f"folder_{folder_idx:02d}"
            
            if not folder_dir.exists():
                continue
            
            val_csv = folder_dir / "validation_metrics.csv"
            
            if val_csv.exists():
                df = pd.read_csv(val_csv)
                if metric in df.columns:
                    # Get last epoch value
                    last_value = df[metric].iloc[-1]
                    all_scores.append({
                        'model': model_name,
                        'score': last_value
                    })
        
        if not all_scores:
            logger.warning(f"No scores found for folder {folder_idx}")
            return
        
        # Create DataFrame and sort
        df_scores = pd.DataFrame(all_scores).sort_values('score', ascending=False)
        
        # Create bar plot
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(df_scores)), df_scores['score'])
        plt.xticks(range(len(df_scores)), df_scores['model'], rotation=45, ha='right')
        plt.ylabel(metric)
        plt.title(f"Model Comparison - Folder {folder_idx:02d} - {metric}")
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        # Print ranking
        logger.info(f"\nModel Ranking for Folder {folder_idx:02d} ({metric}):")
        for idx, row in df_scores.iterrows():
            logger.info(f"  {row['model']}: {row['score']:.4f}")
        
    except Exception as e:
        logger.log_exception(e, "compare_models")
        raise


def create_summary_report(output_path: Optional[Path] = None):
    """
    Create a summary report of all training results.
    
    Args:
        output_path: Path to save the report
    """
    try:
        logger.info("Creating summary report...")
        
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'models': {},
        }
        
        # Collect information for each model
        for model_dir in METRICS_DIR.iterdir():
            if not model_dir.is_dir():
                continue
            
            model_name = model_dir.name
            report['models'][model_name] = {
                'folders': {},
                'avg_performance': {},
            }
            
            # Collect metrics for each folder
            folder_metrics = []
            
            for folder_dir in sorted(model_dir.iterdir()):
                if not folder_dir.is_dir():
                    continue
                
                folder_idx = int(folder_dir.name.split('_')[-1])
                
                # Load summary
                summary_path = folder_dir / "summary.json"
                if summary_path.exists():
                    with open(summary_path, 'r') as f:
                        summary = json.load(f)
                    
                    report['models'][model_name]['folders'][folder_idx] = summary.get('final_metrics', {})
                    folder_metrics.append(summary.get('final_metrics', {}))
            
            # Calculate average performance
            if folder_metrics:
                avg_metrics = {}
                for key in folder_metrics[0].keys():
                    if isinstance(folder_metrics[0][key], (int, float)):
                        values = [m[key] for m in folder_metrics if key in m]
                        avg_metrics[key] = np.mean(values)
                
                report['models'][model_name]['avg_performance'] = avg_metrics
        
        # Save report
        if output_path is None:
            output_path = RESULTS_DIR / "summary_report.json"
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Summary report saved to {output_path}")
        
        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 80)
        
        for model_name, model_data in report['models'].items():
            logger.info(f"\n{model_name.upper()}:")
            if model_data['avg_performance']:
                for metric, value in model_data['avg_performance'].items():
                    logger.info(f"  {metric}: {value:.4f}")
        
        logger.info("=" * 80)
        
        return report
        
    except Exception as e:
        logger.log_exception(e, "create_summary_report")
        raise


def cleanup_temp_files(directories: List[Path]):
    """
    Clean up temporary files and directories.
    
    Args:
        directories: List of directories to clean
    """
    try:
        logger.info("Cleaning up temporary files...")
        
        for directory in directories:
            if directory.exists():
                logger.info(f"Removing {directory}")
                shutil.rmtree(directory)
        
        logger.info("Cleanup completed")
        
    except Exception as e:
        logger.log_exception(e, "cleanup_temp_files")
        raise


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Test folder structure
    print("\nChecking folder structure...")
    structure = get_folder_structure(TRAIN_IMAGES_DIR)
    print(f"Folders: {len(structure['folders'])}")
    print(f"Total images: {structure['total_images']}")
    
    # Test data verification
    print("\nVerifying data structure...")
    is_valid = verify_data_structure()
    print(f"Data structure valid: {is_valid}")
    
    print("\nUtility tests completed!")