"""
Model training module for glove defect detection using Anomalib.
Trains 19 different anomaly detection models across 16 lighting condition folders.
"""

import torch
from pathlib import Path
from typing import Dict, List, Optional
import yaml
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from anomalib.engine import Engine
from anomalib.data import Folder
from anomalib.models import (
    Cfa,
    Cflow,
    Csflow,
    Dfkde,
    Dfm,
    Draem,
    Dsr,
    EfficientAd,
    Fastflow,
    Fre,
    Ganomaly,
    Padim,
    Patchcore,
    ReverseDistillation,
    Stfpm,
)

from config import (
    ANOMALIB_MODELS,
    MODEL_CONFIGS,
    TRAINING_CONFIG,
    TRAIN_IMAGES_DIR,
    MODELS_DIR,
    NUM_LIGHTING_CONDITIONS,
    get_model_save_path,
    get_results_save_path,
)
from src.logger import ProjectLogger, MetricsTracker


# Model class mapping
MODEL_CLASS_MAP = {
    'cfa': Cfa,
    'cflow': Cflow,
    'csflow': Csflow,
    'dfkde': Dfkde,
    'dfm': Dfm,
    'draem': Draem,
    'dsr': Dsr,
    'efficient_ad': EfficientAd,
    'fastflow': Fastflow,
    'fre': Fre,
    'ganomaly': Ganomaly,
    'padim': Padim,
    'patchcore': Patchcore,
    'reverse_distillation': ReverseDistillation,
    'stfpm': Stfpm,
}


class AnomalibModelTrainer:
    """
    Trainer for Anomalib models on glove defect detection task.
    """
    
    def __init__(self, model_name: str, folder_idx: int, config: Dict = None):
        """
        Initialize model trainer.
        
        Args:
            model_name: Name of the Anomalib model
            folder_idx: Folder index (lighting condition)
            config: Training configuration
        """
        self.model_name = model_name.lower()
        self.folder_idx = folder_idx
        self.config = config if config is not None else TRAINING_CONFIG
        
        # Setup logger and metrics tracker
        self.logger = ProjectLogger(f"Train_{model_name}_{folder_idx:02d}")
        self.metrics_tracker = MetricsTracker(model_name, folder_idx)
        
        # Get paths
        self.model_save_dir = get_model_save_path(model_name, folder_idx)
        self.results_save_dir = get_results_save_path(model_name, folder_idx)
        
        # Get folder path
        self.data_dir = self._get_folder_path(folder_idx)
        
        self.logger.info(f"Initializing trainer for {model_name} on folder {folder_idx}")
        self.logger.info(f"Data directory: {self.data_dir}")
        self.logger.info(f"Model save directory: {self.model_save_dir}")
    
    def _get_folder_path(self, folder_idx: int) -> Path:
        """
        Get the path to a specific folder.
        
        Args:
            folder_idx: Folder index
            
        Returns:
            Path to folder
        """
        try:
            # Get all folders in train_images directory
            folders = sorted([d for d in TRAIN_IMAGES_DIR.iterdir() if d.is_dir()])
            
            if folder_idx >= len(folders):
                raise ValueError(f"Folder index {folder_idx} out of range (0-{len(folders)-1})")
            
            return folders[folder_idx]
            
        except Exception as e:
            self.logger.log_exception(e, "_get_folder_path")
            raise
    
    def _create_model(self):
        """
        Create Anomalib model instance.
        
        Returns:
            Model instance
        """
        try:
            # Get model class
            if self.model_name not in MODEL_CLASS_MAP:
                # Try loading custom models or newer additions
                self.logger.warning(f"Model {self.model_name} not in default map, attempting dynamic load")
                return None
            
            model_class = MODEL_CLASS_MAP[self.model_name]
            
            # Get model-specific config
            model_config = MODEL_CONFIGS.get(self.model_name, {})
            
            # Create model instance
            self.logger.info(f"Creating {self.model_name} model with config: {model_config}")
            model = model_class(**model_config)
            
            return model
            
        except Exception as e:
            self.logger.log_exception(e, "_create_model")
            raise
    
    def _create_datamodule(self):
        """
        Create Anomalib data module for the specific folder.
        
        Returns:
            DataModule instance
        """
        try:
            # Create data module for single folder (one-class learning)
            datamodule = Folder(
                name=f"glove_folder_{self.folder_idx:02d}",
                root=self.data_dir.parent,  # Parent directory
                normal_dir=self.data_dir.name,  # Current folder name
                abnormal_dir=None,  # No abnormal examples (one-class)
                normal_test_dir=None,  # Will use split
                task="segmentation",
                image_size=self.config['image_size'],
                train_batch_size=self.config['batch_size'],
                eval_batch_size=self.config['batch_size'],
                num_workers=self.config['num_workers'],
                seed=self.config['seed'],
            )
            
            self.logger.info(f"Created datamodule for folder: {self.data_dir.name}")
            
            return datamodule
            
        except Exception as e:
            self.logger.log_exception(e, "_create_datamodule")
            raise
    
    def train(self):
        """
        Train the model on the specified folder.
        """
        try:
            self.logger.info("=" * 80)
            self.logger.info(f"Starting training: {self.model_name} on folder {self.folder_idx}")
            self.logger.info("=" * 80)
            
            # Create model
            model = self._create_model()
            if model is None:
                self.logger.error(f"Failed to create model: {self.model_name}")
                return
            
            # Create datamodule
            datamodule = self._create_datamodule()
            
            # Create trainer/engine
            engine = Engine(
                task="segmentation",
                image_metrics=["AUROC", "F1Score"],
                pixel_metrics=["AUROC", "F1Score"],
                accelerator=self.config['accelerator'],
                devices=self.config['devices'],
                max_epochs=self.config['num_epochs'],
                default_root_dir=str(self.model_save_dir),
                log_every_n_steps=self.config['log_every_n_steps'],
            )
            
            self.logger.info("Training started...")
            
            # Train model
            engine.fit(
                model=model,
                datamodule=datamodule,
            )
            
            self.logger.info("Training completed!")
            
            # Test model
            self.logger.info("Running test evaluation...")
            test_results = engine.test(
                model=model,
                datamodule=datamodule,
            )
            
            self.logger.info("Test results:")
            self.logger.log_metrics(test_results)
            
            # Save final metrics
            self.metrics_tracker.add_validation_metrics(
                epoch=self.config['num_epochs'],
                metrics=test_results
            )
            
            # Save summary
            self.metrics_tracker.save_summary()
            
            self.logger.info("=" * 80)
            self.logger.info(f"Training completed: {self.model_name} on folder {self.folder_idx}")
            self.logger.info("=" * 80)
            
            return test_results
            
        except Exception as e:
            self.logger.log_exception(e, "train")
            raise


def train_single_model_folder(model_name: str, folder_idx: int) -> Dict:
    """
    Train a single model on a single folder.
    
    Args:
        model_name: Name of the model
        folder_idx: Folder index
        
    Returns:
        Dictionary with training results
    """
    try:
        trainer = AnomalibModelTrainer(model_name, folder_idx)
        results = trainer.train()
        
        return {
            'model_name': model_name,
            'folder_idx': folder_idx,
            'status': 'success',
            'results': results
        }
        
    except Exception as e:
        logger = ProjectLogger("train_single_model_folder")
        logger.log_exception(e, f"{model_name}_folder_{folder_idx}")
        
        return {
            'model_name': model_name,
            'folder_idx': folder_idx,
            'status': 'failed',
            'error': str(e)
        }


def train_model_all_folders(model_name: str, 
                           num_folders: int = NUM_LIGHTING_CONDITIONS,
                           parallel: bool = False,
                           max_workers: int = 4) -> List[Dict]:
    """
    Train a single model on all folders.
    
    Args:
        model_name: Name of the model
        num_folders: Number of folders to train on
        parallel: If True, train folders in parallel
        max_workers: Maximum number of parallel workers
        
    Returns:
        List of training results
    """
    logger = ProjectLogger(f"TrainAll_{model_name}")
    logger.info(f"Training {model_name} on {num_folders} folders")
    
    results = []
    
    try:
        if parallel:
            # Parallel training
            logger.info(f"Using parallel training with {max_workers} workers")
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(train_single_model_folder, model_name, idx): idx 
                    for idx in range(num_folders)
                }
                
                for future in tqdm(as_completed(futures), total=num_folders, 
                                 desc=f"Training {model_name}"):
                    result = future.result()
                    results.append(result)
                    
                    if result['status'] == 'success':
                        logger.info(f"✓ Completed folder {result['folder_idx']}")
                    else:
                        logger.error(f"✗ Failed folder {result['folder_idx']}: {result['error']}")
        else:
            # Sequential training
            logger.info("Using sequential training")
            
            for folder_idx in tqdm(range(num_folders), desc=f"Training {model_name}"):
                result = train_single_model_folder(model_name, folder_idx)
                results.append(result)
                
                if result['status'] == 'success':
                    logger.info(f"✓ Completed folder {folder_idx}")
                else:
                    logger.error(f"✗ Failed folder {folder_idx}: {result['error']}")
        
        # Summary
        successful = sum(1 for r in results if r['status'] == 'success')
        failed = sum(1 for r in results if r['status'] == 'failed')
        
        logger.info("=" * 80)
        logger.info(f"Training summary for {model_name}:")
        logger.info(f"  Successful: {successful}/{num_folders}")
        logger.info(f"  Failed: {failed}/{num_folders}")
        logger.info("=" * 80)
        
        return results
        
    except Exception as e:
        logger.log_exception(e, "train_model_all_folders")
        raise


def train_all_models(model_list: Optional[List[str]] = None,
                     num_folders: int = NUM_LIGHTING_CONDITIONS,
                     parallel_folders: bool = False,
                     parallel_models: bool = False,
                     max_workers: int = 4) -> Dict[str, List[Dict]]:
    """
    Train all models on all folders.
    
    Args:
        model_list: List of model names. If None, uses all models from config.
        num_folders: Number of folders to train on
        parallel_folders: If True, train folders in parallel
        parallel_models: If True, train models in parallel
        max_workers: Maximum number of parallel workers
        
    Returns:
        Dictionary mapping model names to their results
    """
    logger = ProjectLogger("TrainAllModels")
    
    if model_list is None:
        model_list = ANOMALIB_MODELS
    
    logger.info("=" * 80)
    logger.info("TRAINING ALL MODELS")
    logger.info("=" * 80)
    logger.info(f"Models to train: {len(model_list)}")
    logger.info(f"Folders per model: {num_folders}")
    logger.info(f"Total training runs: {len(model_list) * num_folders}")
    logger.info(f"Parallel folders: {parallel_folders}")
    logger.info(f"Parallel models: {parallel_models}")
    logger.info("=" * 80)
    
    all_results = {}
    
    try:
        if parallel_models:
            # Train models in parallel
            logger.info(f"Training models in parallel with {max_workers} workers")
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(train_model_all_folders, model_name, num_folders, 
                                  parallel_folders, max_workers): model_name
                    for model_name in model_list
                }
                
                for future in as_completed(futures):
                    model_name = futures[future]
                    try:
                        results = future.result()
                        all_results[model_name] = results
                        logger.info(f"✓ Completed all folders for {model_name}")
                    except Exception as e:
                        logger.error(f"✗ Failed {model_name}: {e}")
                        all_results[model_name] = []
        else:
            # Train models sequentially
            logger.info("Training models sequentially")
            
            for model_name in tqdm(model_list, desc="Training models"):
                try:
                    results = train_model_all_folders(model_name, num_folders, 
                                                    parallel_folders, max_workers)
                    all_results[model_name] = results
                    logger.info(f"✓ Completed all folders for {model_name}")
                except Exception as e:
                    logger.error(f"✗ Failed {model_name}: {e}")
                    all_results[model_name] = []
        
        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE - FINAL SUMMARY")
        logger.info("=" * 80)
        
        for model_name, results in all_results.items():
            successful = sum(1 for r in results if r['status'] == 'success')
            total = len(results)
            logger.info(f"{model_name}: {successful}/{total} successful")
        
        logger.info("=" * 80)
        
        return all_results
        
    except Exception as e:
        logger.log_exception(e, "train_all_models")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Anomalib models for glove defect detection")
    parser.add_argument('--model', type=str, help='Specific model to train (if not provided, trains all)')
    parser.add_argument('--folder', type=int, help='Specific folder to train on (if not provided, trains all)')
    parser.add_argument('--parallel-folders', action='store_true', help='Train folders in parallel')
    parser.add_argument('--parallel-models', action='store_true', help='Train models in parallel')
    parser.add_argument('--max-workers', type=int, default=4, help='Maximum number of parallel workers')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("GLOVE DEFECT DETECTION - MODEL TRAINING")
    print("=" * 80)
    
    if args.model and args.folder is not None:
        # Train specific model on specific folder
        print(f"Training: {args.model} on folder {args.folder}")
        result = train_single_model_folder(args.model, args.folder)
        print(f"\nResult: {result['status']}")
        
    elif args.model:
        # Train specific model on all folders
        print(f"Training: {args.model} on all folders")
        results = train_model_all_folders(args.model, parallel=args.parallel_folders, 
                                         max_workers=args.max_workers)
        
    else:
        # Train all models
        print("Training: ALL MODELS on all folders")
        results = train_all_models(parallel_folders=args.parallel_folders,
                                   parallel_models=args.parallel_models,
                                   max_workers=args.max_workers)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)