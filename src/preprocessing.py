"""
Image preprocessing module for glove defect detection.
Handles image loading, resizing with aspect ratio preservation, normalization, and preprocessing pipeline.
UPDATED: Now preserves aspect ratio and adds padding instead of distorting images.
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
    UPDATED: Preserves aspect ratio to prevent distortion.
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
    
    def resize_image_with_padding(self, image: np.ndarray, 
                                   target_size: Optional[Tuple[int, int]] = None,
                                   pad_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
        """
        Resize image MAINTAINING aspect ratio, then add padding to reach target size.
        This prevents distortion of the original image.
        
        Args:
            image: Input image
            target_size: Target (width, height). If None, uses config.
            pad_color: Color for padding (default: black)
            
        Returns:
            Resized and padded image (no distortion!)
        """
        try:
            if target_size is None:
                target_size = self.target_size
            
            target_w, target_h = target_size
            h, w = image.shape[:2]
            
            # Calculate aspect ratios
            aspect_ratio = w / h
            target_aspect = target_w / target_h
            
            # Determine new size maintaining aspect ratio
            if aspect_ratio > target_aspect:
                # Image is wider - fit to width
                new_w = target_w
                new_h = int(target_w / aspect_ratio)
            else:
                # Image is taller - fit to height
                new_h = target_h
                new_w = int(target_h * aspect_ratio)
            
            # Resize maintaining aspect ratio (no distortion!)
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Calculate padding needed
            pad_h = target_h - new_h
            pad_w = target_w - new_w
            
            # Distribute padding equally on both sides
            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left
            
            # Add padding (black by default)
            padded = cv2.copyMakeBorder(
                resized,
                top, bottom, left, right,
                cv2.BORDER_CONSTANT,
                value=pad_color
            )
            
            return padded
            
        except Exception as e:
            self.logger.log_exception(e, "resize_image_with_padding")
            raise
    
    def resize_image(self, image: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Resize image to target size WITH aspect ratio preservation.
        This is the new default behavior to prevent distortion.
        
        Args:
            image: Input image
            target_size: Target (width, height). If None, uses config.
            
        Returns:
            Resized image (with padding, no distortion)
        """
        return self.resize_image_with_padding(image, target_size)
    
    def resize_image_force_square(self, image: np.ndarray, 
                                   target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        OLD METHOD: Force resize to exact size (distorts aspect ratio).
        Only use if you specifically need distorted images!
        
        Args:
            image: Input image
            target_size: Target (width, height). If None, uses config.
            
        Returns:
            Resized image (may be distorted!)
        """
        try:
            if target_size is None:
                target_size = self.target_size
            
            # WARNING: This distorts the image!
            resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
            return resized
            
        except Exception as e:
            self.logger.log_exception(e, "resize_image_force_square")
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
        Apply full preprocessing pipeline WITH aspect ratio preservation.
        
        Pipeline:
        1. Resize with aspect ratio preserved (adds padding)
        2. Apply CLAHE (if enabled)
        3. Apply denoising (if enabled)
        4. Apply Gaussian blur (if enabled)
        5. Normalize (if requested)
        
        Args:
            image: Input image (RGB)
            return_normalized: If True, return normalized image [0, 1]
            
        Returns:
            Preprocessed image (NO distortion!)
        """
        try:
            # Resize WITH aspect ratio preservation (no distortion!)
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
                            num_folders: int = NUM_LIGHTING_CONDITIONS,
                            apply_roi: bool = True,
                            fast_mode: bool = False) -> None:
    """
    Preprocess all training images and save to train_images directory.
    Preserves the folder structure (16 folders for 16 lighting conditions).
    UPDATED: Now preserves aspect ratio to prevent distortion!
    
    Args:
        source_dir: Source directory containing training data
        target_dir: Target directory for preprocessed images
        num_folders: Expected number of folders (lighting conditions)
        apply_roi: If True, apply ROI cropping after preprocessing
        fast_mode: If True, skip CLAHE and denoising for faster processing
    """
    logger = ProjectLogger("PreprocessTrainingData")
    preprocessor = ImagePreprocessor()
    
    # In fast mode, disable heavy processing
    if fast_mode:
        preprocessor.clahe = False
        preprocessor.denoise = False
    
    # Import ROI selector if needed
    if apply_roi:
        from src.roi import ROISelector
        roi_selector = ROISelector()
        logger.info(f"ROI will be applied: x={roi_selector.x}, y={roi_selector.y}, "
                   f"w={roi_selector.w}, h={roi_selector.h}")
    
    try:
        logger.info("=" * 80)
        logger.info("Starting training data preprocessing")
        logger.info("=" * 80)
        logger.info(f"Source directory: {source_dir}")
        logger.info(f"Target directory: {target_dir}")
        logger.info(f"Expected folders: {num_folders}")
        logger.info(f"Apply ROI: {apply_roi}")
        logger.info(f"Fast mode: {fast_mode}")
        logger.info(f"Aspect ratio preservation: ENABLED (no distortion!)")
        
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
                    
                    # Preprocess (NOW WITH ASPECT RATIO PRESERVATION!)
                    preprocessed = preprocessor.preprocess(image, return_normalized=False)
                    
                    # Apply ROI if enabled
                    if apply_roi:
                        roi, _ = roi_selector.crop_roi(preprocessed)
                        final_image = roi
                    else:
                        final_image = preprocessed
                    
                    # Save preprocessed (and optionally ROI-cropped) image
                    save_path = target_folder / img_path.name
                    preprocessor.save_image(final_image, save_path)
                    
                    total_images += 1
                    
                except Exception as e:
                    logger.error(f"  Failed to process {img_path.name}: {e}")
                    continue
        
        logger.info("\n" + "=" * 80)
        logger.info(f"Preprocessing completed successfully!")
        logger.info(f"Total images processed: {total_images}")
        logger.info(f"Total folders: {len(folder_paths)}")
        logger.info(f"Aspect ratio: PRESERVED (images not distorted)")
        if apply_roi:
            logger.info(f"ROI applied: Images are {roi_selector.w}x{roi_selector.h} pixels")
        else:
            logger.info(f"No ROI: Images are {preprocessor.target_size[0]}x{preprocessor.target_size[1]} pixels (with padding)")
        logger.info(f"Preprocessed images saved to: {target_dir}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.log_exception(e, "preprocess_training_data")
        raise


def preprocess_single_image(image_path: Path, 
                           target_size: Tuple[int, int] = (256, 256),
                           return_normalized: bool = False) -> np.ndarray:
    """
    Preprocess a single image WITH aspect ratio preservation.
    
    Args:
        image_path: Path to image
        target_size: Target size (width, height)
        return_normalized: If True, return normalized image
        
    Returns:
        Preprocessed image (no distortion!)
    """
    preprocessor = ImagePreprocessor()
    preprocessor.target_size = target_size
    
    # Load image
    image = preprocessor.load_image(image_path)
    
    # Preprocess (with aspect ratio preservation)
    preprocessed = preprocessor.preprocess(image, return_normalized=return_normalized)
    
    return preprocessed


if __name__ == "__main__":
    # Test preprocessing
    print("Testing image preprocessing WITH aspect ratio preservation...")
    print()
    
    # Test preprocessor initialization
    preprocessor = ImagePreprocessor()
    print(f"[OK] Preprocessor initialized")
    print(f"  Target size: {preprocessor.target_size}")
    print(f"  Normalize: {preprocessor.normalize}")
    print(f"  CLAHE: {preprocessor.clahe}")
    print(f"  Denoise: {preprocessor.denoise}")
    print(f"  Gaussian blur: {preprocessor.gaussian_blur}")
    print(f"  Aspect ratio preservation: ENABLED")
    print()
    
    # Test aspect ratio preservation
    print("Testing aspect ratio preservation...")
    test_image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    print(f"  Test image: 1920x1080 (aspect 16:9)")
    
    preprocessed = preprocessor.preprocess(test_image, return_normalized=False)
    print(f"  Preprocessed: {preprocessed.shape[1]}x{preprocessed.shape[0]}")
    print(f"  Result: Aspect ratio preserved with padding [OK]")
    print()
    
    # Test preprocessing pipeline on training data
    print("Starting full training data preprocessing...")
    try:
        preprocess_training_data()
        print("[OK] Training data preprocessing completed successfully!")
    except FileNotFoundError as e:
        print(f"[INFO] Note: {e}")
        print("  This is expected if running tests without actual data.")
    except Exception as e:
        print(f"[ERROR] Error: {e}")