"""
Validation and visualization module for glove defect detection.
Generates heatmaps, overlays, and provides interactive visualization.
"""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
from tqdm import tqdm

from anomalib.engine import Engine
from anomalib.data import PredictDataset
from anomalib import models

from config import (
    ANOMALIB_MODELS,
    VALIDATION_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    HEATMAPS_DIR,
    VISUALIZATION_CONFIG,
    get_model_save_path,
)
from src.logger import ProjectLogger, MetricsTracker, HeatmapSaver
from src.preprocessing import ImagePreprocessor


class ModelValidator:
    """
    Validator for trained Anomalib models with heatmap visualization.
    """
    
    def __init__(self, model_name: str, folder_idx: int):
        """
        Initialize validator.
        
        Args:
            model_name: Name of the trained model
            folder_idx: Folder index the model was trained on
        """
        self.model_name = model_name
        self.folder_idx = folder_idx
        
        # Setup logger
        self.logger = ProjectLogger(f"Validate_{model_name}_{folder_idx:02d}")
        
        # Get model path
        self.model_dir = get_model_save_path(model_name, folder_idx)
        self.checkpoint_path = self._find_checkpoint()
        
        # Setup heatmap saver (no Visualizer needed in v2.x)
        self.heatmap_saver = HeatmapSaver(model_name, folder_idx)
        
        # Load model
        self.model = None
        self.engine = None
        
        self.logger.info(f"Initialized validator for {model_name} folder {folder_idx}")
    
    def _find_checkpoint(self) -> Path:
        """
        Find the best model checkpoint.
        
        Returns:
            Path to checkpoint file
        """
        try:
            # Look for checkpoint files
            checkpoint_patterns = [
                'last.ckpt',
                'best.ckpt',
                '*.ckpt'
            ]
            
            for pattern in checkpoint_patterns:
                checkpoints = list(self.model_dir.rglob(pattern))
                if checkpoints:
                    checkpoint = checkpoints[0]
                    self.logger.info(f"Found checkpoint: {checkpoint}")
                    return checkpoint
            
            raise FileNotFoundError(f"No checkpoint found in {self.model_dir}")
            
        except Exception as e:
            self.logger.log_exception(e, "_find_checkpoint")
            raise
    
    def load_model(self):
        """Load the trained model from checkpoint."""
        try:
            self.logger.info(f"Loading model from {self.checkpoint_path}")
            
            # Get model class dynamically (Anomalib v2.x)
            model_class = getattr(models, self.model_name.capitalize(), None)
            
            if model_class is None:
                # Try alternative naming
                model_variants = [
                    self.model_name.upper(),
                    self.model_name.title(),
                    ''.join(word.capitalize() for word in self.model_name.split('_')),
                ]
                for variant in model_variants:
                    model_class = getattr(models, variant, None)
                    if model_class:
                        break
            
            if model_class is None:
                raise ValueError(f"Model {self.model_name} not found in anomalib.models")
            
            # Create model instance
            self.model = model_class()
            
            # Load checkpoint weights
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            # Set to eval mode
            self.model.eval()
            
            # Create engine for prediction
            self.engine = Engine()
            
            self.logger.info("Model loaded successfully")
            
        except Exception as e:
            self.logger.log_exception(e, "load_model")
            raise
    
    def predict_single_image(self, image_path: Path) -> Dict:
        """
        Run prediction on a single image.
        
        Args:
            image_path: Path to image
            
        Returns:
            Dictionary with prediction results
        """
        try:
            if self.model is None:
                self.load_model()
            
            # Preprocess image
            preprocessor = ImagePreprocessor()
            image = preprocessor.load_image(image_path)
            preprocessed = preprocessor.preprocess(image, return_normalized=False)
            
            # Create dataset
            dataset = PredictDataset(
                path=image_path.parent,
                image_size=preprocessed.shape[:2],
            )
            
            # Run prediction
            with torch.no_grad():
                predictions = self.engine.predict(
                    model=self.model,
                    dataloaders=dataset
                )
            
            # Extract results
            result = predictions[0]
            
            return {
                'image_path': str(image_path),
                'pred_score': float(result.pred_score),
                'pred_label': str(result.pred_label),
                'anomaly_map': result.anomaly_map.cpu().numpy(),
                'pred_mask': result.pred_mask.cpu().numpy() if hasattr(result, 'pred_mask') else None,
            }
            
        except Exception as e:
            self.logger.log_exception(e, f"predict_single_image: {image_path}")
            raise
    
    def generate_heatmap(self, image: np.ndarray, anomaly_map: np.ndarray) -> np.ndarray:
        """
        Generate heatmap from anomaly map.
        
        Args:
            image: Original image (RGB)
            anomaly_map: Anomaly map from model
            
        Returns:
            Heatmap image
        """
        try:
            # Normalize anomaly map to [0, 1]
            anomaly_map_norm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
            
            # Resize to match image size
            anomaly_map_resized = cv2.resize(anomaly_map_norm, (image.shape[1], image.shape[0]))
            
            # Apply colormap
            heatmap = cv2.applyColorMap(
                (anomaly_map_resized * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            
            return heatmap
            
        except Exception as e:
            self.logger.log_exception(e, "generate_heatmap")
            raise
    
    def create_overlay(self, image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        Create overlay of heatmap on original image.
        
        Args:
            image: Original image (RGB)
            heatmap: Heatmap image (BGR from cv2.applyColorMap)
            alpha: Transparency factor
            
        Returns:
            Overlay image (RGB)
        """
        try:
            # Convert heatmap from BGR to RGB
            heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # Ensure same size
            if image.shape[:2] != heatmap_rgb.shape[:2]:
                heatmap_rgb = cv2.resize(heatmap_rgb, (image.shape[1], image.shape[0]))
            
            # Create overlay
            overlay = cv2.addWeighted(image, 1 - alpha, heatmap_rgb, alpha, 0)
            
            return overlay
            
        except Exception as e:
            self.logger.log_exception(e, "create_overlay")
            raise
    
    def visualize_predictions(self, image_paths: List[Path], save_dir: Optional[Path] = None):
        """
        Visualize predictions for multiple images.
        
        Args:
            image_paths: List of image paths
            save_dir: Directory to save visualizations
        """
        try:
            if save_dir is None:
                save_dir = HEATMAPS_DIR / self.model_name / f"folder_{self.folder_idx:02d}"
            
            save_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Visualizing {len(image_paths)} images")
            
            preprocessor = ImagePreprocessor()
            
            for img_path in tqdm(image_paths, desc="Generating visualizations"):
                try:
                    # Load original image
                    original = preprocessor.load_image(img_path)
                    
                    # Predict
                    result = self.predict_single_image(img_path)
                    
                    # Generate heatmap
                    heatmap = self.generate_heatmap(original, result['anomaly_map'])
                    
                    # Create overlay
                    overlay = self.create_overlay(original, heatmap)
                    
                    # Save visualizations
                    image_name = img_path.stem
                    self.heatmap_saver.save_visualization(image_name, original, heatmap, overlay)
                    
                    self.logger.debug(f"Processed {image_name}: score={result['pred_score']:.4f}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to process {img_path.name}: {e}")
                    continue
            
            self.logger.info(f"Visualizations saved to {save_dir}")
            
        except Exception as e:
            self.logger.log_exception(e, "visualize_predictions")
            raise
    
    def display_results_grid(self, image_paths: List[Path], max_images: int = 16):
        """
        Display results in a grid layout.
        
        Args:
            image_paths: List of image paths (up to 16 for one glove)
            max_images: Maximum number of images to display
        """
        try:
            image_paths = image_paths[:max_images]
            n_images = len(image_paths)
            
            # Calculate grid size
            n_cols = 4
            n_rows = (n_images + n_cols - 1) // n_cols
            
            # Create figure
            fig, axes = plt.subplots(n_rows, n_cols * 3, 
                                    figsize=VISUALIZATION_CONFIG['figsize'],
                                    dpi=VISUALIZATION_CONFIG['dpi'])
            
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            
            preprocessor = ImagePreprocessor()
            
            for idx, img_path in enumerate(tqdm(image_paths, desc="Generating grid")):
                row = idx // n_cols
                col_base = (idx % n_cols) * 3
                
                try:
                    # Load original
                    original = preprocessor.load_image(img_path)
                    
                    # Predict
                    result = self.predict_single_image(img_path)
                    
                    # Generate visualizations
                    heatmap = self.generate_heatmap(original, result['anomaly_map'])
                    overlay = self.create_overlay(original, heatmap)
                    
                    # Display original
                    axes[row, col_base].imshow(original)
                    axes[row, col_base].set_title(f"{img_path.stem}\nScore: {result['pred_score']:.3f}",
                                                 fontsize=VISUALIZATION_CONFIG['fontsize'])
                    axes[row, col_base].axis('off')
                    
                    # Display heatmap
                    axes[row, col_base + 1].imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
                    axes[row, col_base + 1].set_title("Heatmap",
                                                      fontsize=VISUALIZATION_CONFIG['fontsize'])
                    axes[row, col_base + 1].axis('off')
                    
                    # Display overlay
                    axes[row, col_base + 2].imshow(overlay)
                    axes[row, col_base + 2].set_title("Overlay",
                                                      fontsize=VISUALIZATION_CONFIG['fontsize'])
                    axes[row, col_base + 2].axis('off')
                    
                except Exception as e:
                    self.logger.error(f"Failed to display {img_path.name}: {e}")
                    continue
            
            # Hide unused subplots
            for idx in range(n_images, n_rows * n_cols):
                row = idx // n_cols
                col_base = (idx % n_cols) * 3
                for offset in range(3):
                    axes[row, col_base + offset].axis('off')
            
            plt.suptitle(f"{self.model_name.upper()} - Folder {self.folder_idx:02d}",
                        fontsize=VISUALIZATION_CONFIG['title_fontsize'],
                        fontweight='bold')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            self.logger.log_exception(e, "display_results_grid")
            raise


class InteractiveValidator:
    """
    Interactive validation interface for glove defect detection.
    """
    
    def __init__(self):
        """Initialize interactive validator."""
        self.logger = ProjectLogger("InteractiveValidator")
        self.validators = {}  # Cache validators
        
        # Get available models and folders
        self.available_models = self._get_available_models()
        self.available_gloves = self._get_available_gloves()
        
        self.logger.info("Interactive validator initialized")
        self.logger.info(f"Available models: {len(self.available_models)}")
        self.logger.info(f"Available gloves: {len(self.available_gloves)}")
    
    def _get_available_models(self) -> List[str]:
        """Get list of available trained models."""
        try:
            models = []
            if MODELS_DIR.exists():
                models = [d.name for d in MODELS_DIR.iterdir() if d.is_dir()]
            return sorted(models)
        except Exception as e:
            self.logger.error(f"Error getting available models: {e}")
            return []
    
    def _get_available_gloves(self) -> List[Path]:
        """Get list of available glove directories in validation folder."""
        try:
            gloves = []
            if VALIDATION_DIR.exists():
                gloves = sorted([d for d in VALIDATION_DIR.iterdir() if d.is_dir()])
            return gloves
        except Exception as e:
            self.logger.error(f"Error getting available gloves: {e}")
            return []
    
    def get_validator(self, model_name: str, folder_idx: int) -> ModelValidator:
        """
        Get or create validator for a model-folder combination.
        
        Args:
            model_name: Model name
            folder_idx: Folder index
            
        Returns:
            ModelValidator instance
        """
        key = f"{model_name}_{folder_idx}"
        
        if key not in self.validators:
            self.validators[key] = ModelValidator(model_name, folder_idx)
        
        return self.validators[key]
    
    def validate_glove(self, glove_idx: int, model_idx: int):
        """
        Validate a glove with a specific model.
        
        Args:
            glove_idx: Index of glove directory
            model_idx: Index of model (1-19)
        """
        try:
            # Get glove directory
            if glove_idx >= len(self.available_gloves):
                raise ValueError(f"Glove index {glove_idx} out of range")
            
            glove_dir = self.available_gloves[glove_idx]
            
            # Get model name
            if model_idx < 1 or model_idx > len(ANOMALIB_MODELS):
                raise ValueError(f"Model index {model_idx} out of range (1-{len(ANOMALIB_MODELS)})")
            
            model_name = ANOMALIB_MODELS[model_idx - 1]
            
            self.logger.info(f"Validating glove '{glove_dir.name}' with model '{model_name}'")
            
            # Get image files from glove directory
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            image_files = []
            for ext in image_extensions:
                image_files.extend(list(glove_dir.glob(f'*{ext}')))
                image_files.extend(list(glove_dir.glob(f'*{ext.upper()}')))
            
            image_files = sorted(image_files)[:16]  # Limit to 16 images
            
            self.logger.info(f"Found {len(image_files)} images")
            
            # Validate with each folder model
            self.logger.info("Generating visualizations for all 16 lighting conditions...")
            
            for folder_idx in range(16):
                self.logger.info(f"\nFolder {folder_idx + 1}/16")
                
                try:
                    # Get validator
                    validator = self.get_validator(model_name, folder_idx)
                    
                    # Generate visualizations
                    validator.visualize_predictions(image_files)
                    
                    # Display grid
                    validator.display_results_grid(image_files)
                    
                except Exception as e:
                    self.logger.error(f"Failed folder {folder_idx}: {e}")
                    continue
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("Validation completed!")
            self.logger.info("=" * 80)
            
        except Exception as e:
            self.logger.log_exception(e, "validate_glove")
            raise
    
    def run_interactive_mode(self):
        """Run interactive validation mode in terminal."""
        try:
            self.logger.info("\n" + "=" * 80)
            self.logger.info("INTERACTIVE VALIDATION MODE")
            self.logger.info("=" * 80)
            
            while True:
                # Display available gloves
                print("\n" + "=" * 80)
                print("AVAILABLE GLOVES:")
                print("=" * 80)
                for idx, glove_dir in enumerate(self.available_gloves):
                    print(f"  [{idx}] {glove_dir.name}")
                
                # Get glove selection
                glove_input = input("\nSelect glove index (or 'q' to quit): ").strip()
                
                if glove_input.lower() == 'q':
                    break
                
                try:
                    glove_idx = int(glove_input)
                except ValueError:
                    print("Invalid input. Please enter a number.")
                    continue
                
                # Display available models
                print("\n" + "=" * 80)
                print("AVAILABLE MODELS:")
                print("=" * 80)
                for idx, model_name in enumerate(ANOMALIB_MODELS, 1):
                    print(f"  [{idx}] {model_name}")
                
                # Get model selection
                model_input = input("\nSelect model index (1-19): ").strip()
                
                try:
                    model_idx = int(model_input)
                except ValueError:
                    print("Invalid input. Please enter a number.")
                    continue
                
                # Run validation
                try:
                    self.validate_glove(glove_idx, model_idx)
                except Exception as e:
                    print(f"Error during validation: {e}")
                    continue
                
                # Ask to continue
                cont = input("\nValidate another glove? (y/n): ").strip().lower()
                if cont != 'y':
                    break
            
            self.logger.info("\nExiting interactive mode")
            
        except Exception as e:
            self.logger.log_exception(e, "run_interactive_mode")
            raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate trained models for glove defect detection")
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--model', type=str, help='Model name')
    parser.add_argument('--folder', type=int, help='Folder index')
    parser.add_argument('--glove', type=int, help='Glove index')
    
    args = parser.parse_args()
    
    if args.interactive:
        # Run interactive mode
        validator = InteractiveValidator()
        validator.run_interactive_mode()
    elif args.model and args.glove is not None:
        # Run specific validation
        validator = InteractiveValidator()
        model_idx = ANOMALIB_MODELS.index(args.model) + 1 if args.model in ANOMALIB_MODELS else 1
        validator.validate_glove(args.glove, model_idx)
    else:
        print("Please specify --interactive or provide --model and --glove")