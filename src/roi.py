"""
ROI (Region of Interest) selection module for glove defect detection.
Crops a fixed rectangular ROI from each image focusing on the reinforcement patch.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from src.preprocessing import ImagePreprocessor
from config import ROI_CONFIG
from src.logger import ProjectLogger


class ROISelector:
    """
    ROI selection and cropping for glove reinforcement patch detection.
    """
    
    def __init__(self, x: int = None, y: int = None, w: int = None, h: int = None, debug: bool = None):
        """
        Initialize ROI selector.
        
        Args:
            x: Top-left x coordinate of ROI
            y: Top-left y coordinate of ROI
            w: Width of ROI
            h: Height of ROI
            debug: Enable debug visualization
        """
        # Load from config if not provided
        self.x = x if x is not None else ROI_CONFIG['x']
        self.y = y if y is not None else ROI_CONFIG['y']
        self.w = w if w is not None else ROI_CONFIG['w']
        self.h = h if h is not None else ROI_CONFIG['h']
        self.debug = debug if debug is not None else ROI_CONFIG['debug']
        
        self.logger = ProjectLogger("ROISelector")
        
        self.logger.info(f"ROI Configuration: x={self.x}, y={self.y}, w={self.w}, h={self.h}")
    
    def crop_roi(self, resized_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crop a fixed ROI from a resized image.
        
        Args:
            resized_img: Preprocessed/resized image (RGB)
            
        Returns:
            roi: Cropped rectangular ROI
            debug_img: Resized image with ROI rectangle drawn (for debugging)
        """
        try:
            # Validate image dimensions
            img_height, img_width = resized_img.shape[:2]
            
            if self.x + self.w > img_width or self.y + self.h > img_height:
                raise ValueError(
                    f"ROI ({self.x}, {self.y}, {self.w}, {self.h}) "
                    f"exceeds image dimensions ({img_width}, {img_height})"
                )
            
            # Crop ROI
            roi = resized_img[self.y:self.y + self.h, self.x:self.x + self.w].copy()
            
            # Create debug image with rectangle
            debug_img = resized_img.copy()
            cv2.rectangle(debug_img, (self.x, self.y), (self.x + self.w, self.y + self.h), 
                         (0, 255, 0), 2)
            
            # Display debug images if enabled
            if self.debug:
                self._show_debug(debug_img, roi)
            
            return roi, debug_img
            
        except Exception as e:
            self.logger.log_exception(e, "crop_roi")
            raise
    
    def _show_debug(self, debug_img: np.ndarray, roi: np.ndarray):
        """
        Show debug visualization windows.
        
        Args:
            debug_img: Image with ROI rectangle
            roi: Cropped ROI
        """
        try:
            # Convert RGB to BGR for display
            debug_img_bgr = cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR)
            roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
            
            cv2.imshow("Resized Image with ROI", debug_img_bgr)
            cv2.imshow("Cropped ROI", roi_bgr)
            
            self.logger.info("Press any key to close debug windows...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        except Exception as e:
            self.logger.log_exception(e, "_show_debug")
            raise
    
    def set_roi(self, x: int, y: int, w: int, h: int):
        """
        Update ROI coordinates.
        
        Args:
            x: Top-left x coordinate
            y: Top-left y coordinate
            w: Width
            h: Height
        """
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        
        self.logger.info(f"ROI updated: x={self.x}, y={self.y}, w={self.w}, h={self.h}")
    
    def get_roi_coords(self) -> Tuple[int, int, int, int]:
        """
        Get current ROI coordinates.
        
        Returns:
            Tuple of (x, y, w, h)
        """
        return (self.x, self.y, self.w, self.h)
    
    def interactive_roi_selection(self, image_path: Path) -> Tuple[int, int, int, int]:
        """
        Interactive ROI selection using mouse.
        
        Args:
            image_path: Path to image for ROI selection
            
        Returns:
            Selected ROI coordinates (x, y, w, h)
        """
        try:
            # Load and preprocess image
            preprocessor = ImagePreprocessor()
            image = preprocessor.load_image(image_path)
            resized = preprocessor.resize_image(image)
            
            # Convert to BGR for OpenCV
            display_img = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
            
            # Clone for drawing
            clone = display_img.copy()
            
            # Initialize variables
            ref_pt = []
            cropping = False
            
            def click_and_crop(event, x, y, flags, param):
                nonlocal ref_pt, cropping, clone
                
                # If left mouse button clicked, record starting (x, y) coordinates
                if event == cv2.EVENT_LBUTTONDOWN:
                    ref_pt = [(x, y)]
                    cropping = True
                
                # Check to see if the left mouse button was released
                elif event == cv2.EVENT_LBUTTONUP:
                    ref_pt.append((x, y))
                    cropping = False
                    
                    # Draw rectangle
                    cv2.rectangle(clone, ref_pt[0], ref_pt[1], (0, 255, 0), 2)
                    cv2.imshow("Select ROI", clone)
            
            # Create window and set mouse callback
            cv2.namedWindow("Select ROI")
            cv2.setMouseCallback("Select ROI", click_and_crop)
            
            self.logger.info("Click and drag to select ROI. Press 'c' to confirm, 'r' to reset.")
            
            while True:
                cv2.imshow("Select ROI", clone)
                key = cv2.waitKey(1) & 0xFF
                
                # If 'r' pressed, reset
                if key == ord("r"):
                    clone = display_img.copy()
                
                # If 'c' pressed, break
                elif key == ord("c"):
                    break
            
            # Close windows
            cv2.destroyAllWindows()
            
            # Calculate ROI coordinates
            if len(ref_pt) == 2:
                x1, y1 = ref_pt[0]
                x2, y2 = ref_pt[1]
                
                x = min(x1, x2)
                y = min(y1, y2)
                w = abs(x2 - x1)
                h = abs(y2 - y1)
                
                # Update ROI
                self.set_roi(x, y, w, h)
                
                self.logger.info(f"ROI selected: x={x}, y={y}, w={w}, h={h}")
                
                return (x, y, w, h)
            else:
                self.logger.warning("No ROI selected. Using current ROI.")
                return self.get_roi_coords()
                
        except Exception as e:
            self.logger.log_exception(e, "interactive_roi_selection")
            raise


def crop_roi(resized_img: np.ndarray, 
             x: int = None, y: int = None, 
             w: int = None, h: int = None,
             debug: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to crop ROI from a resized image.
    
    Args:
        resized_img: Preprocessed/resized image (RGB)
        x: Top-left x coordinate (uses config if None)
        y: Top-left y coordinate (uses config if None)
        w: Width (uses config if None)
        h: Height (uses config if None)
        debug: Enable debug visualization
        
    Returns:
        roi: Cropped ROI
        debug_img: Image with ROI rectangle
    """
    selector = ROISelector(x, y, w, h, debug)
    return selector.crop_roi(resized_img)


def process_image_with_roi(image_path: Path, 
                           roi_coords: Optional[Tuple[int, int, int, int]] = None,
                           return_all: bool = False) -> np.ndarray:
    """
    Load, preprocess, and crop ROI from an image.
    
    Args:
        image_path: Path to image
        roi_coords: ROI coordinates (x, y, w, h). If None, uses config.
        return_all: If True, return (roi, preprocessed, debug_img)
        
    Returns:
        roi: Cropped ROI (or tuple if return_all=True)
    """
    try:
        # Preprocess image
        preprocessor = ImagePreprocessor()
        image = preprocessor.load_image(image_path)
        preprocessed = preprocessor.preprocess(image)
        
        # Crop ROI
        if roi_coords is not None:
            x, y, w, h = roi_coords
            selector = ROISelector(x, y, w, h)
        else:
            selector = ROISelector()
        
        roi, debug_img = selector.crop_roi(preprocessed)
        
        if return_all:
            return roi, preprocessed, debug_img
        else:
            return roi
            
    except Exception as e:
        logger = ProjectLogger("process_image_with_roi")
        logger.log_exception(e, f"process_image_with_roi: {image_path}")
        raise


if __name__ == "__main__":
    # Test ROI selector
    print("Testing ROI selector...")
    
    # Test initialization
    selector = ROISelector()
    print(f"✓ ROI Selector initialized")
    print(f"  ROI: x={selector.x}, y={selector.y}, w={selector.w}, h={selector.h}")
    
    # Test ROI update
    selector.set_roi(100, 100, 200, 200)
    coords = selector.get_roi_coords()
    print(f"✓ ROI updated: {coords}")
    
    # Test with dummy image
    print("\nTesting with dummy image...")
    dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    try:
        roi, debug_img = selector.crop_roi(dummy_image)
        print(f"✓ ROI cropped successfully")
        print(f"  ROI shape: {roi.shape}")
        print(f"  Debug image shape: {debug_img.shape}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\nROI selector tests completed!")