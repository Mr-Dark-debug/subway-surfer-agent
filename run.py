#!/usr/bin/env python3
# Run script for Subway Surfers AI - provides a simple interface for all functions
import os
import sys
import argparse
import subprocess
import time
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import configuration
from subway_ai.config import *

def print_section(title):
    """Print a section title"""
    print("\n" + "="*70)
    print(f" {title} ".center(70, "="))
    print("="*70 + "\n")

def check_dependencies():
    """Check if all required dependencies are installed"""
    print_section("CHECKING DEPENDENCIES")
    
    dependencies = [
        "torch", "numpy", "cv2", "pyautogui", "pytesseract", "pillow", "pandas", "matplotlib"
    ]
    
    missing = []
    for dep in dependencies:
        try:
            if dep == "cv2":
                __import__("cv2")
            else:
                __import__(dep)
            print(f"✓ {dep}")
        except ImportError:
            missing.append(dep)
            print(f"✗ {dep}")
    
    if missing:
        print("\nMissing dependencies:")
        for dep in missing:
            if dep == "cv2":
                print("  pip install opencv-python")
            elif dep == "pillow":
                print("  pip install pillow")
            else:
                print(f"  pip install {dep}")
        print("\nPlease install the missing dependencies and try again.")
        return False
    
    return True

def check_tesseract():
    """Check if Tesseract OCR is installed"""
    print_section("CHECKING TESSERACT OCR")
    
    if not os.path.exists(TESSERACT_PATH):
        print(f"Tesseract OCR not found at {TESSERACT_PATH}")
        print("Please install Tesseract OCR and update the path in subway_ai/config.py")
        return False
    
    print(f"Tesseract OCR found at {TESSERACT_PATH}")
    
    # Try to use Tesseract
    try:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
        version = pytesseract.get_tesseract_version()
        print(f"Tesseract version: {version}")
        return True
    except Exception as e:
        print(f"Error checking Tesseract: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print_section("CREATING DIRECTORIES")
    
    directories = [
        SCREENSHOTS_DIR,
        os.path.join(SCREENSHOTS_DIR, "gameplay"),
        os.path.join(SCREENSHOTS_DIR, "score"),
        os.path.join(SCREENSHOTS_DIR, "coins"),
        os.path.join(SCREENSHOTS_DIR, "game_over"),
        TEMPLATES_DIR,
        MODELS_DIR,
        LOGS_DIR,
        os.path.join(BASE_DIR, "results"),
        os.path.join(BASE_DIR, "results", "plots")
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory already exists: {directory}")
    
    return True

def calibrate_regions():
    """Run the calibration tool"""
    print_section("CALIBRATING SCREEN REGIONS")
    
    try:
        from subway_ai.calibrate import main as calibrate_main
        calibrate_main()
        return True
    except Exception as e:
        print(f"Error running calibration: {e}")
        return False

def capture_templates():
    """Capture templates for game over and play button"""
    print_section("CAPTURING TEMPLATES")
    
    try:
        from subway_ai.save_templates import main as save_templates_main
        save_templates_main()
        return True
    except Exception as e:
        print(f"Error capturing templates: {e}")
        return False

def train_model(episodes=100, model=None):
    """Train the AI model"""
    print_section(f"TRAINING MODEL FOR {episodes} EPISODES")
    
    cmd = ["python", "main_ai.py", "--mode", "train", "--episodes", str(episodes)]
    
    if model:
        cmd.extend(["--model", model])
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\nTraining completed successfully!")
        return True
    else:
        print(f"\nTraining failed with return code {result.returncode}")
        return False

def play_game(games=5, model=None):
    """Play the game using a trained model"""
    print_section(f"PLAYING {games} GAMES")
    
    cmd = ["python", "main_ai.py", "--mode", "play", "--games", str(games)]
    
    if model:
        cmd.extend(["--model", model])
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\nPlay session completed successfully!")
        return True
    else:
        print(f"\nPlay session failed with return code {result.returncode}")
        return False

def visualize_results(log=None):
    """Visualize training results"""
    print_section("VISUALIZING RESULTS")
    
    try:
        from subway_ai.visualize import main as visualize_main
        
        # Set up sys.argv for the visualize script
        sys.argv = ["visualize.py"]
        if log:
            sys.argv.extend(["--log", log])
        
        visualize_main()
        return True
    except Exception as e:
        print(f"Error visualizing results: {e}")
        return False

def test_detector():
    """Test the object detector"""
    print_section("TESTING OBJECT DETECTOR")
    
    try:
        from subway_ai.detector import ObjectDetector
        import pyautogui
        import cv2
        
        # Create detector
        detector = ObjectDetector()
        
        # Take a screenshot
        print("Taking screenshot in 3 seconds...")
        time.sleep(3)
        screenshot = pyautogui.screenshot(region=GAME_REGION)
        
        # Analyze the screenshot
        print("Analyzing screenshot...")
        analysis = detector.analyze_game_screen(screenshot)
        
        # Print analysis results
        for key, value in analysis.items():
            print(f"{key}: {value}")
        
        # Save visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(SCREENSHOTS_DIR, f"detector_test_{timestamp}.png")
        detector.visualize_detection(screenshot, analysis, save_path)
        
        print(f"Visualization saved to {save_path}")
        return True
    except Exception as e:
        print(f"Error testing detector: {e}")
        return False

def print_help():
    """Print help message"""
    print_section("SUBWAY SURFERS AI HELP")
    
    print("Available commands:")
    print("  setup       - Check dependencies and create directories")
    print("  calibrate   - Run the calibration tool to set screen regions")
    print("  templates   - Capture templates for game over and play button")
    print("  train       - Train the AI model")
    print("  play        - Play the game using a trained model")
    print("  visualize   - Visualize training results")
    print("  test        - Test the object detector")
    print("  all         - Run all steps in sequence")
    print("\nOptions:")
    print("  --episodes N  - Number of episodes for training (default: 100)")
    print("  --games N     - Number of games to play (default: 5)")
    print("  --model PATH  - Path to model file for training or playing")
    print("  --log PATH    - Path to training log file for visualization")
    print("\nExamples:")
    print("  python run.py setup")
    print("  python run.py calibrate")
    print("  python run.py train --episodes 200")
    print("  python run.py play --model subway_dqn_best.pt")
    print("  python run.py visualize")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Subway Surfers AI Run Script')
    parser.add_argument('command', nargs='?', default='help',
                      help='Command to run (setup, calibrate, templates, train, play, visualize, test, all, help)')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes for training')
    parser.add_argument('--games', type=int, default=5, help='Number of games to play')
    parser.add_argument('--model', type=str, help='Path to model file for training or playing')
    parser.add_argument('--log', type=str, help='Path to training log file for visualization')
    
    args = parser.parse_args()
    
    if args.command == 'help':
        print_help()
    elif args.command == 'setup':
        if check_dependencies() and check_tesseract() and create_directories():
            print("\nSetup completed successfully!")
    elif args.command == 'calibrate':
        calibrate_regions()
    elif args.command == 'templates':
        capture_templates()
    elif args.command == 'train':
        train_model(args.episodes, args.model)
    elif args.command == 'play':
        play_game(args.games, args.model)
    elif args.command == 'visualize':
        visualize_results(args.log)
    elif args.command == 'test':
        test_detector()
    elif args.command == 'all':
        print_section("RUNNING ALL STEPS")
        if check_dependencies() and check_tesseract() and create_directories():
            if calibrate_regions() and capture_templates():
                if train_model(args.episodes, args.model):
                    if play_game(args.games, args.model):
                        visualize_results(args.log)
    else:
        print(f"Unknown command: {args.command}")
        print("Run 'python run.py help' for usage information")

if __name__ == "__main__":
    main()