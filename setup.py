#!/usr/bin/env python3
"""
Project setup script
Initializes the project structure and verifies dependencies.
Windows-compatible version without special characters.
"""

import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"ERROR: Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    print(f"OK: Python {version.major}.{version.minor}.{version.micro}")
    return True


def create_directories():
    """Create necessary directories."""
    print("\nCreating project directories...")
    
    directories = [
        'data/train/acceptable',
        'data/train_images',
        'data/test/acceptable',
        'data/test/marginal',
        'data/test/unacceptable',
        'models',
        'results/heatmaps',
        'results/metrics',
        'logs',
        'examples',
    ]
    
    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        print(f"  [OK] {directory}")
    
    print("All directories created successfully")


def install_dependencies():
    """Install Python dependencies."""
    print("\nInstalling dependencies...")
    print("This may take several minutes...")
    
    try:
        subprocess.check_call([
            sys.executable, 
            '-m', 
            'pip', 
            'install', 
            '-r', 
            'requirements.txt'
        ])
        print("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("ERROR: Failed to install dependencies")
        return False


def verify_installation():
    """Verify that key packages are installed."""
    print("\nVerifying installation...")
    
    packages = [
        'torch',
        'cv2',
        'numpy',
        'pandas',
        'matplotlib',
    ]
    
    all_installed = True
    for package in packages:
        try:
            __import__(package)
            print(f"  [OK] {package}")
        except ImportError:
            print(f"  [MISSING] {package}")
            all_installed = False
    
    # Check anomalib separately (different import name)
    try:
        import anomalib
        print(f"  [OK] anomalib")
    except ImportError:
        print(f"  [MISSING] anomalib")
        all_installed = False
    
    return all_installed


def create_sample_config():
    """Create sample configuration in data directory."""
    print("\nCreating sample data structure guide...")
    
    guide = """# Data Structure Guide

Place your training data in the following structure:

data/
+-- train/
    +-- acceptable/
        +-- folder_01/  # Lighting condition 1
        |   +-- glove1_light1.jpg
        |   +-- glove2_light1.jpg
        |   +-- ...
        +-- folder_02/  # Lighting condition 2
        |   +-- ...
        +-- ...
        +-- folder_16/  # Lighting condition 16

+-- test/
    +-- acceptable/
    |   +-- glove_images/
    +-- marginal/
    |   +-- glove_images/
    +-- unacceptable/
        +-- glove_images/

Each folder should contain glove images captured under a specific lighting condition.
The system will train separate models for each lighting condition.

After placing your data:
1. Run preprocessing: python main.py --preprocess
2. Verify structure: python main.py --verify
3. Start training: python main.py --train --all
"""
    
    guide_path = Path('data/DATA_STRUCTURE.md')
    try:
        guide_path.write_text(guide, encoding='utf-8')
        print(f"Guide created: {guide_path}")
    except Exception as e:
        print(f"Warning: Could not create guide file: {e}")


def main():
    """Main setup function."""
    print("=" * 80)
    print("GLOVE DEFECT DETECTION - PROJECT SETUP")
    print("=" * 80)
    
    # Check Python version
    if not check_python_version():
        print("\nPlease install Python 3.8 or higher and try again.")
        input("Press Enter to exit...")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Ask user if they want to install dependencies
    print("\n" + "=" * 80)
    response = input("Install Python dependencies? (y/n): ").strip().lower()
    if response == 'y':
        if not install_dependencies():
            print("\nWARNING: Some dependencies failed to install")
            print("You may need to install them manually:")
            print("  pip install -r requirements.txt")
        else:
            # Verify installation
            if verify_installation():
                print("\nAll packages verified successfully")
            else:
                print("\nWARNING: Some packages could not be verified")
                print("You may need to install them manually")
    else:
        print("\nSkipping dependency installation.")
        print("Remember to install them later with:")
        print("  pip install -r requirements.txt")
    
    # Create sample config
    create_sample_config()
    
    print("\n" + "=" * 80)
    print("SETUP COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Read data/DATA_STRUCTURE.md for data organization")
    print("2. Place your training images in data/train/acceptable/")
    print("3. Run: python main.py --preprocess")
    print("4. Run: python main.py --train --all")
    print("5. Run: python main.py --validate --interactive")
    print("\nFor detailed instructions, see README.md")
    print("For quick examples, run: python examples/quick_start.py")
    print("=" * 80)
    
    input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()