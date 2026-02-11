"""
Image preprocessing module for glove defect detection.
Handles image loading, resizing, normalization, and preprocessing pipeline.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import shutil
from tqdm import tqdm

from config import (
    PREPROCESSING_CONFIG, 
    TRAIN_DIR, 
    TRAIN_IMAGES_DIR,
    NUM_LIGHTING_CONDITIONS
)
from src.logger import ProjectLogger


class ImagePreprocessor:
    """
    Image preprocessing pipeline for glove images.
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize preprocessor.
        
        Args:
            config: Preprocessing configuration dictionary
        """
        self.config = config if config is not None else PREPROCESSING_CONFIG
        self.logger = ProjectLogger("ImagePreprocessor")
        
        self.target_size = tuple(self.config['target_size'])
        self.normalize = self.config['normalize']
        self.clahe = self.config['clahe']
        self.denoise = self.config['denoise']
        self.gaussian_blur = self.config['gaussian_blur']
        self.blur_kernel = tuple(self.config['blur_kernel'])
        
        # Initialize CLAHE if needed
        if self.clahe:
            self.clahe_processor = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    def load_image(self, image_path: Path) -> np.ndarray:
        """
        Load image from file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image as numpy array (RGB)
        """
        try:
            # Read image
            img = cv2.imread(str(image_path))
            
            if img is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            return img
            
        except Exception as e:
            self.logger.log_exception(e, f"load_image: {image_path}")
            raise
    
    def resize_image(self, image: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            image: Input image
            target_size: Target (width, height). If None, uses config.
            
        Returns:
            Resized image
        """
        try:
            if target_size is None:
                target_size = self.target_size
            
            resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
            return resized
            
        except Exception as e:
            self.logger.log_exception(e, "resize_image")
            raise
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image to [0, 1] range.
        
        Args:
            image: Input image
            
        Returns:
            Normalized image
        """
        try:
            # Convert to float32
            normalized = image.astype(np.float32) / 255.0
            return normalized
            
        except Exception as e:
            self.logger.log_exception(e, "normalize_image")
            raise
    
    def denormalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Denormalize image from [0, 1] to [0, 255].
        
        Args:
            image: Normalized image
            
        Returns:
            Denormalized image
        """
        try:
            denormalized = (image * 255.0).astype(np.uint8)
            return denormalized
            
        except Exception as e:
            self.logger.log_exception(e, "denormalize_image")
            raise
    
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        Args:
            image: Input image (RGB)
            
        Returns:
            Image with CLAHE applied
        """
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            
            # Apply CLAHE to L channel
            lab[:, :, 0] = self.clahe_processor.apply(lab[:, :, 0])
            
            # Convert back to RGB
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            return enhanced
            
        except Exception as e:
            self.logger.log_exception(e, "apply_clahe")
            raise
    
    def apply_denoising(self, image: np.ndarray) -> np.ndarray:
        """
        Apply denoising filter.
        
        Args:
            image: Input image
            
        Returns:
            Denoised image
        """
        try:
            denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
            return denoised
            
        except Exception as e:
            self.logger.log_exception(e, "apply_denoising")
            raise
    
    def apply_gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian blur.
        
        Args:
            image: Input image
            
        Returns:
            Blurred image
        """
        try:
            blurred = cv2.GaussianBlur(image, self.blur_kernel, 0)
            return blurred
            
        except Exception as e:
            self.logger.log_exception(e, "apply_gaussian_blur")
            raise
    
    def preprocess(self, image: np.ndarray, return_normalized: bool = False) -> np.ndarray:
        """
        Apply full preprocessing pipeline.
        
        Args:
            image: Input image (RGB)
            return_normalized: If True, return normalized image [0, 1]
            
        Returns:
            Preprocessed image
        """
        try:
            # Resize
            processed = self.resize_image(image)
            
            # Apply CLAHE if enabled
            if self.clahe:
                processed = self.apply_clahe(processed)
            
            # Apply denoising if enabled
            if self.denoise:
                processed = self.apply_denoising(processed)
            
            # Apply Gaussian blur if enabled
            if self.gaussian_blur:
                processed = self.apply_gaussian_blur(processed)
            
            # Normalize if needed
            if return_normalized and self.normalize:
                processed = self.normalize_image(processed)
            
            return processed
            
        except Exception as e:
            self.logger.log_exception(e, "preprocess")
            raise
    
    def save_image(self, image: np.ndarray, save_path: Path):
        """
        Save image to file.
        
        Args:
            image: Image to save (RGB)
            save_path: Path to save the image
        """
        try:
            # Ensure directory exists
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert RGB to BGR for OpenCV
            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Save
            cv2.imwrite(str(save_path), bgr_image)
            
        except Exception as e:
            self.logger.log_exception(e, f"save_image: {save_path}")
            raise


def preprocess_training_data(source_dir: Path = TRAIN_DIR, 
                            target_dir: Path = TRAIN_IMAGES_DIR,
                            num_folders: int = NUM_LIGHTING_CONDITIONS) -> None:
    """
    Preprocess all training images and save to train_images directory.
    Preserves the folder structure (16 folders for 16 lighting conditions).
    
    Args:
        source_dir: Source directory containing training data
        target_dir: Target directory for preprocessed images
        num_folders: Expected number of folders (lighting conditions)
    """
    logger = ProjectLogger("PreprocessTrainingData")
    preprocessor = ImagePreprocessor()
    
    try:
        logger.info("=" * 80)
        logger.info("Starting training data preprocessing")
        logger.info("=" * 80)
        logger.info(f"Source directory: {source_dir}")
        logger.info(f"Target directory: {target_dir}")
        logger.info(f"Expected folders: {num_folders}")
        
        # Check if source directory exists
        if not source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {source_dir}")
        
        # Get all subdirectories (lighting condition folders)
        folder_paths = sorted([d for d in source_dir.iterdir() if d.is_dir()])
        
        if len(folder_paths) == 0:
            raise ValueError(f"No folders found in {source_dir}")
        
        logger.info(f"Found {len(folder_paths)} folders")
        
        # Process each folder
        total_images = 0
        for folder_idx, folder_path in enumerate(folder_paths):
            logger.info(f"\nProcessing folder {folder_idx + 1}/{len(folder_paths)}: {folder_path.name}")
            
            # Create corresponding folder in target directory
            target_folder = target_dir / folder_path.name
            target_folder.mkdir(parents=True, exist_ok=True)
            
            # Get all image files in folder
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            image_files = []
            for ext in image_extensions:
                image_files.extend(list(folder_path.glob(f'*{ext}')))
                image_files.extend(list(folder_path.glob(f'*{ext.upper()}')))
            
            logger.info(f"  Found {len(image_files)} images")
            
            # Process each image
            for img_path in tqdm(image_files, desc=f"  Processing {folder_path.name}"):
                try:
                    # Load image
                    image = preprocessor.load_image(img_path)
                    
                    # Preprocess (returns uint8 RGB image)
                    preprocessed = preprocessor.preprocess(image, return_normalized=False)
                    
                    # Save preprocessed image
                    save_path = target_folder / img_path.name
                    preprocessor.save_image(preprocessed, save_path)
                    
                    total_images += 1
                    
                except Exception as e:
                    logger.error(f"  Failed to process {img_path.name}: {e}")
                    continue
        
        logger.info("\n" + "=" * 80)
        logger.info(f"Preprocessing completed successfully!")
        logger.info(f"Total images processed: {total_images}")
        logger.info(f"Total folders: {len(folder_paths)}")
        logger.info(f"Preprocessed images saved to: {target_dir}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.log_exception(e, "preprocess_training_data")
        raise


def preprocess_single_image(image_path: Path, 
                           target_size: Tuple[int, int] = (256, 256),
                           return_normalized: bool = False) -> np.ndarray:
    """
    Preprocess a single image.
    
    Args:
        image_path: Path to image
        target_size: Target size (width, height)
        return_normalized: If True, return normalized image
        
    Returns:
        Preprocessed image
    """
    preprocessor = ImagePreprocessor()
    preprocessor.target_size = target_size
    
    # Load image
    image = preprocessor.load_image(image_path)
    
    # Preprocess
    preprocessed = preprocessor.preprocess(image, return_normalized=return_normalized)
    
    return preprocessed


if __name__ == "__main__":
    # Test preprocessing
    print("Testing image preprocessing...")
    
    # Test preprocessor initialization
    preprocessor = ImagePreprocessor()
    print(f"✓ Preprocessor initialized")
    print(f"  Target size: {preprocessor.target_size}")
    print(f"  Normalize: {preprocessor.normalize}")
    print(f"  CLAHE: {preprocessor.clahe}")
    print(f"  Denoise: {preprocessor.denoise}")
    print(f"  Gaussian blur: {preprocessor.gaussian_blur}")
    
    # Test preprocessing pipeline on training data
    print("\nStarting full training data preprocessing...")
    try:
        preprocess_training_data()
        print("✓ Training data preprocessing completed successfully!")
    except FileNotFoundError as e:
        print(f"⚠ Note: {e}")
        print("  This is expected if running tests without actual data.")
    except Exception as e:
        print(f"✗ Error: {e}")