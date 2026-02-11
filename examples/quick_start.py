"""
Quick Start Example Script
Demonstrates basic usage of the glove defect detection system.
"""

from pathlib import Path
from src import (
    ProjectLogger,
    ImagePreprocessor,
    ROISelector,
    preprocess_training_data,
    train_single_model_folder,
    ModelValidator,
    verify_data_structure,
)
from config import TRAIN_DIR, TRAIN_IMAGES_DIR


def example_1_preprocessing():
    """Example: Preprocess training data."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: PREPROCESSING")
    print("=" * 80)
    
    # Initialize preprocessor
    preprocessor = ImagePreprocessor()
    print(f"✓ Preprocessor initialized")
    print(f"  Target size: {preprocessor.target_size}")
    
    # Preprocess all training data
    print("\nPreprocessing training data...")
    try:
        preprocess_training_data(
            source_dir=TRAIN_DIR,
            target_dir=TRAIN_IMAGES_DIR,
        )
        print("✓ Preprocessing completed successfully!")
    except FileNotFoundError as e:
        print(f"⚠ {e}")
        print("  (This is expected if running without actual data)")


def example_2_roi_selection():
    """Example: ROI selection and cropping."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: ROI SELECTION")
    print("=" * 80)
    
    import numpy as np
    
    # Create a dummy image
    dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # Initialize ROI selector
    selector = ROISelector(x=50, y=50, w=150, h=150)
    print(f"✓ ROI Selector initialized")
    print(f"  ROI coordinates: x={selector.x}, y={selector.y}, w={selector.w}, h={selector.h}")
    
    # Crop ROI
    roi, debug_img = selector.crop_roi(dummy_image)
    print(f"✓ ROI cropped successfully")
    print(f"  Original shape: {dummy_image.shape}")
    print(f"  ROI shape: {roi.shape}")


def example_3_single_model_training():
    """Example: Train a single model on a single folder."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: SINGLE MODEL TRAINING")
    print("=" * 80)
    
    # Verify data structure first
    print("Verifying data structure...")
    is_valid = verify_data_structure()
    
    if not is_valid:
        print("⚠ Data structure not valid. Please preprocess data first.")
        return
    
    # Train PatchCore on folder 0
    print("\nTraining PatchCore model on folder 0...")
    try:
        result = train_single_model_folder('patchcore', 0)
        
        if result['status'] == 'success':
            print("✓ Training completed successfully!")
        else:
            print(f"✗ Training failed: {result['error']}")
    except Exception as e:
        print(f"⚠ Error: {e}")
        print("  (This is expected if data is not available)")


def example_4_validation():
    """Example: Validate a trained model."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: VALIDATION")
    print("=" * 80)
    
    try:
        # Initialize validator
        validator = ModelValidator('patchcore', 0)
        print("✓ Validator initialized")
        
        # Note: Actual validation requires trained model and test images
        print("\nTo run validation:")
        print("  1. Ensure model is trained")
        print("  2. Place test images in data/test/")
        print("  3. Run: python main.py --validate --interactive")
        
    except Exception as e:
        print(f"⚠ {e}")
        print("  (This is expected if model is not trained yet)")


def example_5_complete_workflow():
    """Example: Complete workflow from preprocessing to validation."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: COMPLETE WORKFLOW")
    print("=" * 80)
    
    print("\nComplete workflow steps:")
    print("\n1. PREPROCESS DATA")
    print("   python main.py --preprocess")
    
    print("\n2. TRAIN A MODEL")
    print("   python main.py --train --model patchcore")
    
    print("\n3. VALIDATE MODEL")
    print("   python main.py --validate --interactive")
    
    print("\n4. GENERATE REPORT")
    print("   python main.py --report")
    
    print("\nOr run everything at once:")
    print("   python main.py --full-pipeline")


def main():
    """Run all examples."""
    logger = ProjectLogger("QuickStartExamples")
    
    logger.info("=" * 80)
    logger.info("GLOVE DEFECT DETECTION - QUICK START EXAMPLES")
    logger.info("=" * 80)
    
    # Run examples
    example_1_preprocessing()
    example_2_roi_selection()
    example_3_single_model_training()
    example_4_validation()
    example_5_complete_workflow()
    
    print("\n" + "=" * 80)
    print("EXAMPLES COMPLETED")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Prepare your data in the correct folder structure")
    print("2. Run preprocessing: python main.py --preprocess")
    print("3. Train models: python main.py --train --all")
    print("4. Validate results: python main.py --validate --interactive")
    print("\nFor more information, see README.md")
    print("=" * 80)


if __name__ == "__main__":
    main()