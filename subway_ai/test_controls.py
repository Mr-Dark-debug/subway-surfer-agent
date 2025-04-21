# test_controls.py
import os
import time
import pyautogui
import cv2
import numpy as np
from PIL import Image
from config import *
from game_control import GameControl
from screen_capture import ScreenCapture

def test_keyboard_controls():
    """Test keyboard controls for Subway Surfers"""
    print("Testing keyboard controls...")
    print("Make sure the game window is focused")
    
    # Initialize game control
    game_control = GameControl()
    
    # Focus the game window
    game_control.focus_game_window()
    time.sleep(1)
    
    # Test each direction
    print("\nTesting LEFT arrow key...")
    game_control.perform_action(0)  # LEFT
    time.sleep(1)
    
    print("Testing RIGHT arrow key...")
    game_control.perform_action(1)  # RIGHT
    time.sleep(1)
    
    print("Testing UP arrow key (jump)...")
    game_control.perform_action(2)  # UP
    time.sleep(1)
    
    print("Testing DOWN arrow key (roll)...")
    game_control.perform_action(3)  # DOWN
    time.sleep(1)
    
    print("\nKeyboard control test complete")

def test_ocr():
    """Test OCR for score and coins"""
    print("Testing OCR for score and coins...")
    
    # Initialize screen capture
    screen_capture = ScreenCapture()
    
    # Capture and process score
    print("\nCapturing score region...")
    score_img = screen_capture.capture_score_screen()
    score_img.save("score_test.png")
    print(f"Score image saved to score_test.png")
    
    # Process and detect score
    score = screen_capture.extract_score_ocr()
    print(f"Detected score: {score}")
    
    # Capture and process coins
    print("\nCapturing coin region...")
    coin_img = screen_capture.capture_coin_screen()
    coin_img.save("coin_test.png")
    print(f"Coin image saved to coin_test.png")
    
    # Process and detect coins
    coins = screen_capture.extract_coins_ocr()
    print(f"Detected coins: {coins}")
    
    print("\nOCR test complete")

def test_play_button_detection():
    """Test play button detection"""
    print("Testing play button detection...")
    
    # Initialize screen capture
    screen_capture = ScreenCapture()
    
    # Initialize game control
    game_control = GameControl()
    
    # Capture full game screen
    game_img = screen_capture.capture_game_screen()
    game_img.save("game_screen_test.png")
    print(f"Game screen saved to game_screen_test.png")
    
    # Try to find play button
    play_button_loc = screen_capture.locate_play_button()
    
    if play_button_loc:
        print(f"Play button found at {play_button_loc}")
        
        # Draw a circle on the detected location
        game_img_np = np.array(game_img)
        cv_img = cv2.cvtColor(game_img_np, cv2.COLOR_RGB2BGR)
        cv2.circle(cv_img, (play_button_loc[0] - GAME_REGION[0], play_button_loc[1] - GAME_REGION[1]), 
                  20, (0, 255, 0), -1)
        
        # Save the marked image
        cv2.imwrite("play_button_detected.png", cv_img)
        print(f"Marked image saved to play_button_detected.png")
        
        # Test clicking the play button
        input("Press Enter to test clicking the play button...")
        game_control.restart_game(play_button_loc)
    else:
        print("Play button not detected")
    
    print("\nPlay button detection test complete")

def main():
    """Main test function"""
    print("Subway Surfers AI - Control & OCR Test Utility")
    print("=============================================")
    
    # Create necessary directories
    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
    
    while True:
        print("\nTest Options:")
        print("1. Test Keyboard Controls")
        print("2. Test OCR (Score & Coins)")
        print("3. Test Play Button Detection")
        print("4. Run All Tests")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == '1':
            test_keyboard_controls()
        elif choice == '2':
            test_ocr()
        elif choice == '3':
            test_play_button_detection()
        elif choice == '4':
            test_keyboard_controls()
            time.sleep(1)
            test_ocr()
            time.sleep(1)
            test_play_button_detection()
        elif choice == '5':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()