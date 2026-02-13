"""
Reset Script for Glove Defect Detection Project
Allows selective or complete cleanup of training data, models, and results.
"""

import shutil
import argparse
from pathlib import Path
from datetime import datetime

from config import (
    TRAIN_IMAGES_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    LOGS_DIR,
    HEATMAPS_DIR,
    METRICS_DIR,
)


class ProjectReset:
    """
    Handle project reset and cleanup operations.
    """
    
    def __init__(self, backup: bool = True):
        """
        Initialize reset handler.
        
        Args:
            backup: If True, create backup before deleting
        """
        self.backup = backup
        self.backup_dir = Path("backups") / datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("=" * 80)
        print("GLOVE DEFECT DETECTION - RESET TOOL")
        print("=" * 80)
        print()
    
    def _backup_directory(self, directory: Path, name: str):
        """
        Create backup of a directory.
        
        Args:
            directory: Directory to backup
            name: Name for the backup
        """
        if not directory.exists():
            return
        
        if self.backup:
            backup_path = self.backup_dir / name
            print(f"  Backing up {name} to {backup_path}...")
            shutil.copytree(directory, backup_path, dirs_exist_ok=True)
            print(f"    [OK] Backup created")
    
    def _delete_directory(self, directory: Path, name: str):
        """
        Delete a directory and its contents.
        
        Args:
            directory: Directory to delete
            name: Name for display
        """
        if not directory.exists():
            print(f"  [SKIP] {name} not found: {directory}")
            return
        
        try:
            # Count items
            items = list(directory.rglob("*"))
            file_count = sum(1 for item in items if item.is_file())
            
            print(f"  Deleting {name}...")
            print(f"    Path: {directory}")
            print(f"    Files: {file_count}")
            
            shutil.rmtree(directory)
            print(f"    [OK] Deleted successfully")
            
        except Exception as e:
            print(f"    [ERROR] Failed to delete: {e}")
    
    def reset_preprocessed_images(self):
        """Delete preprocessed training images."""
        print("\n" + "=" * 80)
        print("RESET: PREPROCESSED IMAGES")
        print("=" * 80)
        
        if self.backup:
            self._backup_directory(TRAIN_IMAGES_DIR, "train_images")
        
        self._delete_directory(TRAIN_IMAGES_DIR, "Preprocessed Images")
        
        # Recreate empty directory
        TRAIN_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        print(f"  [OK] Recreated empty directory: {TRAIN_IMAGES_DIR}")
    
    def reset_models(self, specific_model: str = None):
        """
        Delete trained models.
        
        Args:
            specific_model: If provided, only delete this model. Otherwise delete all.
        """
        print("\n" + "=" * 80)
        if specific_model:
            print(f"RESET: MODEL - {specific_model}")
        else:
            print("RESET: ALL MODELS")
        print("=" * 80)
        
        if specific_model:
            # Delete specific model
            model_dir = MODELS_DIR / specific_model
            if self.backup:
                self._backup_directory(model_dir, f"models/{specific_model}")
            self._delete_directory(model_dir, f"Model: {specific_model}")
        else:
            # Delete all models
            if self.backup:
                self._backup_directory(MODELS_DIR, "models")
            self._delete_directory(MODELS_DIR, "All Models")
            
            # Recreate empty directory
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            print(f"  [OK] Recreated empty directory: {MODELS_DIR}")
    
    def reset_results(self, specific_model: str = None):
        """
        Delete results (heatmaps and metrics).
        
        Args:
            specific_model: If provided, only delete results for this model.
        """
        print("\n" + "=" * 80)
        if specific_model:
            print(f"RESET: RESULTS - {specific_model}")
        else:
            print("RESET: ALL RESULTS")
        print("=" * 80)
        
        if specific_model:
            # Delete specific model results
            heatmap_dir = HEATMAPS_DIR / specific_model
            metrics_dir = METRICS_DIR / specific_model
            
            if self.backup:
                self._backup_directory(heatmap_dir, f"heatmaps/{specific_model}")
                self._backup_directory(metrics_dir, f"metrics/{specific_model}")
            
            self._delete_directory(heatmap_dir, f"Heatmaps: {specific_model}")
            self._delete_directory(metrics_dir, f"Metrics: {specific_model}")
        else:
            # Delete all results
            if self.backup:
                self._backup_directory(RESULTS_DIR, "results")
            
            self._delete_directory(RESULTS_DIR, "All Results")
            
            # Recreate directory structure
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            HEATMAPS_DIR.mkdir(parents=True, exist_ok=True)
            METRICS_DIR.mkdir(parents=True, exist_ok=True)
            print(f"  [OK] Recreated empty directories")
    
    def reset_logs(self):
        """Delete log files."""
        print("\n" + "=" * 80)
        print("RESET: LOGS")
        print("=" * 80)
        
        if self.backup:
            self._backup_directory(LOGS_DIR, "logs")
        
        self._delete_directory(LOGS_DIR, "Logs")
        
        # Recreate empty directory
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"  [OK] Recreated empty directory: {LOGS_DIR}")
    
    def reset_all(self):
        """Complete reset - delete everything except original data."""
        print("\n" + "=" * 80)
        print("COMPLETE RESET")
        print("=" * 80)
        print()
        print("This will delete:")
        print("  • Preprocessed images")
        print("  • All trained models")
        print("  • All results and metrics")
        print("  • All log files")
        print()
        print("Original training data will NOT be deleted.")
        print()
        
        if self.backup:
            print(f"Backups will be created in: {self.backup_dir}")
        
        # Confirm
        response = input("Are you sure you want to continue? (yes/no): ").strip().lower()
        if response != 'yes':
            print("\n[CANCELLED] Reset aborted.")
            return
        
        print("\nProceeding with complete reset...\n")
        
        # Reset everything
        self.reset_preprocessed_images()
        self.reset_models()
        self.reset_results()
        self.reset_logs()
        
        print("\n" + "=" * 80)
        print("COMPLETE RESET FINISHED")
        print("=" * 80)
        print()
        print("Your project has been reset to a clean state.")
        print("Original training data in data/train/ is preserved.")
        
        if self.backup:
            print(f"\nBackups saved to: {self.backup_dir}")
    
    def reset_specific_model_complete(self, model_name: str):
        """
        Reset everything related to a specific model.
        
        Args:
            model_name: Name of the model to reset
        """
        print("\n" + "=" * 80)
        print(f"RESET: COMPLETE MODEL - {model_name}")
        print("=" * 80)
        print()
        print(f"This will delete all data for model: {model_name}")
        print("  • Trained model checkpoints")
        print("  • Results and heatmaps")
        print("  • Metrics and logs")
        print()
        
        response = input("Are you sure? (yes/no): ").strip().lower()
        if response != 'yes':
            print("\n[CANCELLED] Reset aborted.")
            return
        
        print(f"\nResetting {model_name}...\n")
        
        # Reset model-specific items
        self.reset_models(specific_model=model_name)
        self.reset_results(specific_model=model_name)
        
        print("\n" + "=" * 80)
        print(f"RESET COMPLETE: {model_name}")
        print("=" * 80)
        print()
        print(f"All data for {model_name} has been deleted.")
        print("You can now train this model again from scratch.")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Reset/cleanup tool for Glove Defect Detection project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Complete reset (everything except original data)
  python reset.py --all
  
  # Reset only preprocessed images
  python reset.py --preprocessed
  
  # Reset only models
  python reset.py --models
  
  # Reset only results
  python reset.py --results
  
  # Reset only logs
  python reset.py --logs
  
  # Reset specific model (all data for that model)
  python reset.py --model patchcore
  
  # Reset without creating backups (faster)
  python reset.py --all --no-backup
  
  # Interactive mode (choose what to reset)
  python reset.py --interactive
        """
    )
    
    parser.add_argument('--all', action='store_true',
                       help='Complete reset (preprocessed, models, results, logs)')
    parser.add_argument('--preprocessed', action='store_true',
                       help='Delete preprocessed training images')
    parser.add_argument('--models', action='store_true',
                       help='Delete all trained models')
    parser.add_argument('--results', action='store_true',
                       help='Delete all results and metrics')
    parser.add_argument('--logs', action='store_true',
                       help='Delete all log files')
    parser.add_argument('--model', type=str,
                       help='Reset specific model (model name)')
    parser.add_argument('--no-backup', action='store_true',
                       help='Skip creating backups (faster)')
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive mode - choose what to reset')
    
    args = parser.parse_args()
    
    # Check if no arguments provided
    if not any([args.all, args.preprocessed, args.models, args.results, 
                args.logs, args.model, args.interactive]):
        parser.print_help()
        return
    
    # Create reset handler
    reset = ProjectReset(backup=not args.no_backup)
    
    # Interactive mode
    if args.interactive:
        print("\nWhat would you like to reset?")
        print("  1. Everything (complete reset)")
        print("  2. Preprocessed images only")
        print("  3. Models only")
        print("  4. Results only")
        print("  5. Logs only")
        print("  6. Specific model (all data)")
        print("  0. Cancel")
        
        choice = input("\nEnter choice (0-6): ").strip()
        
        if choice == '1':
            reset.reset_all()
        elif choice == '2':
            reset.reset_preprocessed_images()
        elif choice == '3':
            reset.reset_models()
        elif choice == '4':
            reset.reset_results()
        elif choice == '5':
            reset.reset_logs()
        elif choice == '6':
            model_name = input("Enter model name: ").strip()
            reset.reset_specific_model_complete(model_name)
        else:
            print("\n[CANCELLED]")
        return
    
    # Execute based on arguments
    if args.all:
        reset.reset_all()
    else:
        if args.preprocessed:
            reset.reset_preprocessed_images()
        
        if args.models:
            reset.reset_models()
        
        if args.results:
            reset.reset_results()
        
        if args.logs:
            reset.reset_logs()
        
        if args.model:
            reset.reset_specific_model_complete(args.model)
    
    print("\n" + "=" * 80)
    print("RESET OPERATIONS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
