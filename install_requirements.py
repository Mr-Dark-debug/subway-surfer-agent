"""
Installation script for Subway Surfers AI
This script installs all required dependencies for the project.
"""

import os
import sys
import subprocess
import time

def check_tesseract():
    """Check if Tesseract OCR is installed"""
    print("Checking for Tesseract OCR installation...")
    
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        print("✓ Tesseract is already installed and configured!")
        return True
    except:
        print("× Tesseract not found or not properly configured.")
        return False

def install_tesseract():
    """Guide user to install Tesseract OCR"""
    print("\nTesseract OCR is required for this project.")
    print("Please download and install Tesseract from:")
    print("https://github.com/UB-Mannheim/tesseract/wiki")
    print("\nMake sure to:")
    print("1. Install to the default location (C:\\Program Files\\Tesseract-OCR)")
    print("2. Add Tesseract to your PATH during installation")
    
    user_input = input("\nPress Enter when you've installed Tesseract (or type 'skip' to continue anyway): ")
    
    if user_input.lower() == 'skip':
        print("Skipping Tesseract check. Note that score and coin detection may not work correctly.")
        return
    
    # Update main.py with the correct Tesseract path
    try:
        main_path = "main.py"
        with open(main_path, 'r') as file:
            content = file.read()
        
        # Replace the Tesseract path line
        if "pytesseract.pytesseract.tesseract_cmd =" in content:
            content = content.replace(
                "pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'",
                f"pytesseract.pytesseract.tesseract_cmd = r'{os.path.join('C:', 'Program Files', 'Tesseract-OCR', 'tesseract.exe')}'"
            )
            
            with open(main_path, 'w') as file:
                file.write(content)
            
            print("✓ Updated Tesseract path in main.py")
    except Exception as e:
        print(f"× Error updating Tesseract path: {e}")

def install_requirements():
    """Install Python package requirements"""
    print("\nInstalling required Python packages...")
    
    requirements = [
        "numpy",
        "opencv-python",
        "torch",
        "torchvision",
        "pyautogui",
        "matplotlib",
        "pytesseract",
        "pillow"
    ]
    
    # Check for CUDA support
    try:
        import torch
        if torch.cuda.is_available():
            print("✓ CUDA is available! Using GPU acceleration.")
        else:
            print("× CUDA not available. Using CPU only (this will be slower).")
            print("  If you have an NVIDIA GPU, consider installing CUDA:")
            print("  https://developer.nvidia.com/cuda-downloads")
    except:
        print("? Couldn't check CUDA status. Will install PyTorch anyway.")
    
    # Install packages
    for package in requirements:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} installed successfully!")
        except Exception as e:
            print(f"× Error installing {package}: {e}")
            print("  Please install this package manually.")
    
    print("\nAll required packages installed!")

def create_directories():
    """Create required directories"""
    print("\nCreating project directories...")
    
    directories = [
        "screenshots",
        "models",
        "logs",
        "templates",
        "templates/obstacles",
        "gameplay"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created {directory}/")

def main():
    """Main installation function"""
    print("=" * 60)
    print("Subway Surfers AI - Installation")
    print("=" * 60)
    
    # Install Python packages
    install_requirements()
    
    # Check and guide for Tesseract installation
    if not check_tesseract():
        install_tesseract()
    
    # Create project directories
    create_directories()
    
    print("\n" + "=" * 60)
    print("Installation complete!")
    print("You can now run main.py to start training the AI")
    print("=" * 60)
    
    time.sleep(2)

if __name__ == "__main__":
    main()