"""
Quick Start Example Script (Simplified)
Demonstrates basic usage without requiring trained models.
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.logger import ProjectLogger
from src.preprocessing import ImagePreprocessor
from src.roi import ROISelector
from config import TRAIN_DIR, TRAIN_IMAGES_DIR


def example_1_preprocessing():
    """Example: Preprocess training data."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: PREPROCESSING")
    print("=" * 80)
    
    # Initialize preprocessor
    preprocessor = ImagePreprocessor()
    print(f"[OK] Preprocessor initialized")
    print(f"  Target size: {preprocessor.target_size}")
    print(f"  Normalize: {preprocessor.normalize}")
    print(f"  CLAHE: {preprocessor.clahe}")
    print(f"  Denoise: {preprocessor.denoise}")
    
    # Note about actual preprocessing
    print("\nTo preprocess your actual data:")
    print("  python main.py --preprocess")


def example_2_roi_selection():
    """Example: ROI selection and cropping."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: ROI SELECTION")
    print("=" * 80)
    
    # Create a dummy image
    dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # Initialize ROI selector
    selector = ROISelector(x=50, y=50, w=150, h=150)
    print(f"[OK] ROI Selector initialized")
    print(f"  ROI coordinates: x={selector.x}, y={selector.y}, w={selector.w}, h={selector.h}")
    
    # Crop ROI
    roi, debug_img = selector.crop_roi(dummy_image)
    print(f"[OK] ROI cropped successfully")
    print(f"  Original shape: {dummy_image.shape}")
    print(f"  ROI shape: {roi.shape}")
    print(f"  Debug image shape: {debug_img.shape}")


def example_3_logger():
    """Example: Using the logger."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: LOGGING SYSTEM")
    print("=" * 80)
    
    # Create a logger
    logger = ProjectLogger("ExampleLogger")
    
    print("[OK] Logger created")
    print("  Logging to console and file")
    
    # Log some messages
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.debug("This is a debug message (might not show depending on log level)")
    
    print("\n[OK] Check logs/ directory for log files")


def example_4_data_structure():
    """Example: Checking data structure."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: DATA STRUCTURE")
    print("=" * 80)
    
    print("Your data should be organized like this:")
    print("")
    print("data/")
    print("└── train/")
    print("    └── acceptable/")
    print("        ├── folder_01/  # Lighting condition 1")
    print("        │   ├── image1.jpg")
    print("        │   ├── image2.jpg")
    print("        │   └── ...")
    print("        ├── folder_02/  # Lighting condition 2")
    print("        └── ...")
    print("        └── folder_16/  # Lighting condition 16")
    print("")
    
    # Check if data directories exist
    if TRAIN_DIR.exists():
        print(f"[OK] Train directory exists: {TRAIN_DIR}")
        
        # Count folders
        folders = [d for d in TRAIN_DIR.iterdir() if d.is_dir()]
        print(f"  Found {len(folders)} folder(s)")
        
        if len(folders) > 0:
            print(f"  First folder: {folders[0].name}")
    else:
        print(f"[INFO] Train directory not found: {TRAIN_DIR}")
        print(f"  This is normal if you haven't added data yet")
    
    print("\nAfter adding your data:")
    print("  python main.py --preprocess")


def example_5_complete_workflow():
    """Example: Complete workflow from preprocessing to validation."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: COMPLETE WORKFLOW")
    print("=" * 80)
    
    print("\nComplete workflow steps:")
    print("\n1. SETUP (One Time)")
    print("   python setup.py")
    
    print("\n2. PREPARE DATA")
    print("   Place images in: data/train/acceptable/folder_XX/")
    
    print("\n3. PREPROCESS DATA")
    print("   python main.py --preprocess")
    
    print("\n4. TRAIN A MODEL (Start with one)")
    print("   python main.py --train --model patchcore")
    
    print("\n5. TRAIN ALL MODELS (Optional)")
    print("   python main.py --train --all")
    
    print("\n6. VALIDATE MODEL")
    print("   python main.py --validate --interactive")
    
    print("\n7. GENERATE REPORT")
    print("   python main.py --report")
    
    print("\nOr run everything at once:")
    print("   python main.py --full-pipeline")


def example_6_available_models():
    """Example: Show available models."""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: AVAILABLE MODELS")
    print("=" * 80)
    
    from config import ANOMALIB_MODELS
    
    print(f"\nThis system supports {len(ANOMALIB_MODELS)} anomaly detection models:")
    print("")
    
    for i, model_name in enumerate(ANOMALIB_MODELS, 1):
        print(f"  {i:2d}. {model_name}")
    
    print("\nRecommended models to start with:")
    print("  • patchcore     - Best overall performance")
    print("  • padim         - Good for texture defects")
    print("  • efficient_ad  - Fast and accurate")
    print("  • stfpm         - Good generalization")
    
    print("\nTo train a specific model:")
    print("  python main.py --train --model patchcore")


def main():
    """Run all examples."""
    logger = ProjectLogger("QuickStartExamples")
    
    print("=" * 80)
    print("GLOVE DEFECT DETECTION - QUICK START EXAMPLES")
    print("=" * 80)
    print("")
    print("This script demonstrates the basic components of the system.")
    print("No trained models or data required for these examples.")
    print("")
    
    # Run examples
    try:
        example_1_preprocessing()
        example_2_roi_selection()
        example_3_logger()
        example_4_data_structure()
        example_5_complete_workflow()
        example_6_available_models()
        
        print("\n" + "=" * 80)
        print("EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("\nNext steps:")
        print("1. Prepare your data in the correct folder structure")
        print("2. Run: python main.py --preprocess")
        print("3. Run: python main.py --train --model patchcore")
        print("4. Run: python main.py --validate --interactive")
        print("\nFor more information, see:")
        print("  • README.md - Complete documentation")
        print("  • GETTING_STARTED.md - Quick start guide")
        print("  • HOW_TO_RUN_EXAMPLES.md - Running examples")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n[ERROR] An error occurred: {e}")
        print("This is likely due to missing dependencies.")
        print("\nTry running:")
        print("  pip install -r requirements.txt")
    
    input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()