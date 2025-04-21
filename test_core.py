# Test core functionality of Subway Surfers AI
import os
import sys
import time
import pyautogui
import cv2
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from subway_ai.screen_capture import ScreenCapture
from subway_ai.game_control import GameControl
from subway_ai.detector import ObjectDetector
from subway_ai.config import *

def test_screen_capture():
    """Test screen capture and OCR functionality"""
    print("\n=== Testing Screen Capture ===")
    
    # Initialize screen capture
    screen_capture = ScreenCapture()
    
    # Take screenshots
    print("Taking game screenshot...")
    game_screen = screen_capture.capture_game_screen()
    game_screen.save("test_game_screen.png")
    print(f"Game screenshot saved to test_game_screen.png")
    
    print("Taking score screenshot...")
    score_screen = screen_capture.capture_score_screen()
    score_screen.save("test_score_screen.png")
    print(f"Score screenshot saved to test_score_screen.png")
    
    print("Taking coin screenshot...")
    coin_screen = screen_capture.capture_coin_screen()
    coin_screen.save("test_coin_screen.png")
    print(f"Coin screenshot saved to test_coin_screen.png")
    
    # Test OCR
    print("\nTesting OCR...")
    score = screen_capture.extract_score_ocr()
    print(f"Detected score: {score}")
    
    coins = screen_capture.extract_coins_ocr()
    print(f"Detected coins: {coins}")
    
    # Test game over detection
    print("\nTesting game over detection...")
    game_over = screen_capture.detect_game_over()
    print(f"Game over detected: {game_over}")
    
    if game_over:
        print("Testing play button detection...")
        play_button = screen_capture.locate_play_button()
        print(f"Play button found at: {play_button}")
    
    return True

def test_game_control():
    """Test game control functionality"""
    print("\n=== Testing Game Control ===")
    
    # Initialize game control
    game_control = GameControl()
    
    # Check if game is running
    print("Checking if game is running...")
    game_running = game_control.check_game_running()
    print(f"Game running: {game_running}")
    
    if not game_running:
        print("Starting game...")
        game_control.start_game()
        time.sleep(5)  # Wait for game to start
    
    # Focus game window
    print("Focusing game window...")
    game_control.focus_game_window()
    time.sleep(1)
    
    # Test swipe actions
    print("\nTesting swipe actions...")
    print("Wait 5 seconds before starting swipe tests")
    time.sleep(5)
    
    actions = list(ACTIONS.keys())
    
    for action in actions:
        action_name = game_control.perform_action(action)
        print(f"Performed action: {action_name}")
        time.sleep(1)  # Wait between actions
    
    # Test game over detection
    print("\nTesting game over detection...")
    game_over = game_control.detect_game_over()
    print(f"Game over detected: {game_over}")
    
    # Test restart if game is over
    if game_over:
        print("Testing game restart...")
        restart_success = game_control.restart_game()
        print(f"Restart success: {restart_success}")
    
    return True

def test_detector():
    """Test object detector functionality"""
    print("\n=== Testing Object Detector ===")
    
    # Initialize detector
    detector = ObjectDetector()
    
    # Take a screenshot
    print("Taking screenshot...")
    screenshot = pyautogui.screenshot(region=GAME_REGION)
    
    # Test game over detection
    print("\nTesting game over detection...")
    game_over = detector.detect_game_over(screenshot)
    print(f"Game over detected: {game_over}")
    
    # Test play button detection
    print("\nTesting play button detection...")
    play_button = detector.locate_play_button(screenshot)
    print(f"Play button found at: {play_button}")
    
    # Test full screen analysis
    print("\nTesting full screen analysis...")
    analysis = detector.analyze_game_screen(screenshot)
    print(f"Analysis results: {analysis}")
    
    # Save visualization
    print("\nSaving visualization...")
    vis_img = detector.visualize_detection(screenshot, analysis, "test_detection.png")
    print(f"Visualization saved to test_detection.png")
    
    return True

def test_swipe_recording():
    """Test swipe recording functionality"""
    print("\n=== Testing Swipe Recording ===")
    
    # Initialize game control
    game_control = GameControl()
    
    # Start recording
    print("Starting recording...")
    recording_started = game_control.start_recording("test_recording.mp4")
    print(f"Recording started: {recording_started}")
    
    if recording_started:
        # Perform a series of swipe actions
        print("\nPerforming swipe actions...")
        
        # Repeat some swipes to create a meaningful recording
        for _ in range(5):
            for action in range(4):  # LEFT, RIGHT, UP, DOWN
                game_control.perform_action(action)
                game_control.record_frame()
                time.sleep(0.5)
        
        # Stop recording
        print("\nStopping recording...")
        recording_stopped = game_control.stop_recording()
        print(f"Recording stopped: {recording_stopped}")
        print(f"Recording saved to test_recording.mp4")
    
    return True

def main():
    """Main test function"""
    print("=== Subway Surfers AI Core Functionality Test ===")
    print("This script will test the core functionality of the Subway Surfers AI system.")
    print("Make sure the game is running before proceeding.")
    print("\nPress Enter to continue or Ctrl+C to cancel...")
    
    try:
        input()
    except KeyboardInterrupt:
        print("\nTest cancelled")
        return
    
    try:
        test_screen_capture()
        test_game_control()
        test_detector()
        test_swipe_recording()
        
        print("\n=== All Tests Completed Successfully ===")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()