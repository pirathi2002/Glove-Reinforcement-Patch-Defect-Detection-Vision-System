"""
Main pipeline for glove defect detection project.
Orchestrates preprocessing, training, and validation.
"""

import argparse
import sys
from pathlib import Path

from config import (
    PROJECT_ROOT,
    TRAIN_DIR,
    TRAIN_IMAGES_DIR,
    VALIDATION_DIR,
    ANOMALIB_MODELS,
    NUM_LIGHTING_CONDITIONS,
)
from src.logger import ProjectLogger, setup_experiment_logging
from src.preprocessing import preprocess_training_data
from src.train_models import (
    train_single_model_folder,
    train_model_all_folders,
    train_all_models,
)
from src.validate import InteractiveValidator
from src.utils import (
    verify_data_structure,
    create_summary_report,
    compare_models,
)


def main():
    """Main pipeline execution."""
    
    parser = argparse.ArgumentParser(
        description="Glove Defect Detection Pipeline using Anomalib",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preprocess training data
  python main.py --preprocess
  
  # Train a specific model on a specific folder
  python main.py --train --model patchcore --folder 0
  
  # Train a specific model on all folders
  python main.py --train --model patchcore
  
  # Train all models on all folders (sequential)
  python main.py --train --all
  
  # Train all models with parallel folder processing
  python main.py --train --all --parallel-folders --max-workers 4
  
  # Run validation in interactive mode
  python main.py --validate --interactive
  
  # Generate summary report
  python main.py --report
  
  # Full pipeline (preprocess + train all + report)
  python main.py --full-pipeline
        """
    )
    
    # Main operations
    parser.add_argument('--preprocess', action='store_true', 
                       help='Preprocess training data')
    parser.add_argument('--train', action='store_true',
                       help='Train models')
    parser.add_argument('--validate', action='store_true',
                       help='Run validation')
    parser.add_argument('--report', action='store_true',
                       help='Generate summary report')
    parser.add_argument('--full-pipeline', action='store_true',
                       help='Run full pipeline (preprocess + train + report)')
    
    # Training options
    parser.add_argument('--model', type=str, choices=ANOMALIB_MODELS,
                       help='Specific model to train')
    parser.add_argument('--folder', type=int,
                       help='Specific folder index to train on')
    parser.add_argument('--all', action='store_true',
                       help='Train all models')
    parser.add_argument('--parallel-folders', action='store_true',
                       help='Train folders in parallel')
    parser.add_argument('--parallel-models', action='store_true',
                       help='Train models in parallel')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Maximum number of parallel workers')
    
    # Preprocessing options
    parser.add_argument('--no-roi', action='store_true',
                       help='Skip ROI cropping during preprocessing (default: apply ROI)')
    parser.add_argument('--fast', action='store_true',
                       help='Fast preprocessing (skip CLAHE, denoise)')
    parser.add_argument('--interactive', action='store_true',
                       help='Run validation in interactive mode')
    parser.add_argument('--glove', type=int,
                       help='Glove index for validation')
    
    # Utility options
    parser.add_argument('--verify', action='store_true',
                       help='Verify data structure')
    parser.add_argument('--compare-models', action='store_true',
                       help='Compare models performance')
    parser.add_argument('--metric', type=str, default='image_AUROC',
                       help='Metric for model comparison')
    
    args = parser.parse_args()
    
    # Setup main logger
    logger = ProjectLogger("MainPipeline")
    
    try:
        logger.info("=" * 80)
        logger.info("GLOVE DEFECT DETECTION PIPELINE")
        logger.info("=" * 80)
        logger.info(f"Project Root: {PROJECT_ROOT}")
        
        # Verify data structure if requested
        if args.verify:
            logger.info("\nVerifying data structure...")
            is_valid = verify_data_structure()
            if not is_valid:
                logger.error("Data structure verification failed!")
                sys.exit(1)
            logger.info(" Data structure verified successfully")
        
        # Preprocessing
        if args.preprocess or args.full_pipeline:
            logger.info("\n" + "=" * 80)
            logger.info("PREPROCESSING TRAINING DATA")
            logger.info("=" * 80)
            
            # Determine if ROI should be applied (default: True)
            apply_roi = not args.no_roi
            
            logger.info(f"Apply ROI: {apply_roi}")
            logger.info(f"Fast mode: {args.fast}")
            
            if args.fast:
                logger.info("Running FAST preprocessing (skip CLAHE, denoise)")
            else:
                logger.info("Running FULL preprocessing with all filters")
            
            preprocess_training_data(
                source_dir=TRAIN_DIR,
                target_dir=TRAIN_IMAGES_DIR,
                num_folders=NUM_LIGHTING_CONDITIONS,
                apply_roi=apply_roi,
                fast_mode=args.fast
            )
            
            logger.info(" Preprocessing completed")
        
        # Training
        if args.train or args.full_pipeline:
            logger.info("\n" + "=" * 80)
            logger.info("TRAINING MODELS")
            logger.info("=" * 80)
            
            if args.model and args.folder is not None:
                # Train specific model on specific folder
                logger.info(f"Training {args.model} on folder {args.folder}")
                result = train_single_model_folder(args.model, args.folder)
                
                if result['status'] == 'success':
                    logger.info(f" Training completed successfully")
                else:
                    logger.error(f"Training failed: {result['error']}")
            
            elif args.model:
                # Train specific model on all folders
                logger.info(f"Training {args.model} on all folders")
                results = train_model_all_folders(
                    args.model,
                    parallel=args.parallel_folders,
                    max_workers=args.max_workers
                )
                
                successful = sum(1 for r in results if r['status'] == 'success')
                logger.info(f" Completed: {successful}/{len(results)} successful")
            
            elif args.all or args.full_pipeline:
                # Train all models
                logger.info("Training all models on all folders")
                results = train_all_models(
                    parallel_folders=args.parallel_folders,
                    parallel_models=args.parallel_models,
                    max_workers=args.max_workers
                )
                
                logger.info(" All training completed")
            
            else:
                logger.error("Please specify --model or --all for training")
                sys.exit(1)
        
        # Validation
        if args.validate:
            logger.info("\n" + "=" * 80)
            logger.info("VALIDATION")
            logger.info("=" * 80)
            
            validator = InteractiveValidator()
            
            if args.interactive:
                # Interactive mode
                validator.run_interactive_mode()
            elif args.model and args.glove is not None:
                # Validate specific glove with specific model
                model_idx = ANOMALIB_MODELS.index(args.model) + 1
                validator.validate_glove(args.glove, model_idx)
            else:
                logger.error("Please specify --interactive or provide --model and --glove")
                sys.exit(1)
        
        # Model comparison
        if args.compare_models:
            logger.info("\n" + "=" * 80)
            logger.info("MODEL COMPARISON")
            logger.info("=" * 80)
            
            if args.folder is not None:
                compare_models(args.folder, metric=args.metric)
            else:
                # Compare for all folders
                for folder_idx in range(NUM_LIGHTING_CONDITIONS):
                    logger.info(f"\nComparing models for folder {folder_idx}")
                    compare_models(folder_idx, metric=args.metric)
        
        # Generate report
        if args.report or args.full_pipeline:
            logger.info("\n" + "=" * 80)
            logger.info("GENERATING SUMMARY REPORT")
            logger.info("=" * 80)
            
            report = create_summary_report()
            logger.info(" Summary report generated")
        
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
    
    except KeyboardInterrupt:
        logger.warning("\nPipeline interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        logger.log_exception(e, "main")
        sys.exit(1)


if __name__ == "__main__":
    main()