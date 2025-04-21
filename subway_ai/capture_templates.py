# capture_templates.py
import os
import pyautogui
import cv2
import numpy as np
import time
from config import *

def ensure_directories():
    """Ensure necessary directories exist"""
    os.makedirs(TEMPLATES_DIR, exist_ok=True)
    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

def capture_play_button():
    """Capture the green play button template"""
    print("\n=== Capturing Play Button Template ===")
    print("The script will take a screenshot of the game over screen")
    print("The green PLAY button should be visible")
    
    input("Press Enter when the play button is visible on screen...")
    
    # Take a screenshot of the entire game region
    screenshot = pyautogui.screenshot(region=GAME_REGION)
    img_array = np.array(screenshot)
    
    # Convert to HSV for better color detection
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # Define green color range
    lower_green = np.array([40, 100, 100])
    upper_green = np.array([80, 255, 255])
    
    # Create a mask for green areas
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Find contours of green areas
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest green area (likely the play button)
    if contours:
        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Min area threshold
                # Get bounding box for the contour
                x, y, w, h = cv2.boundingRect(contour)
                
                # Add some padding
                padding = 5
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(img_array.shape[1] - x, w + 2*padding)
                h = min(img_array.shape[0] - y, h + 2*padding)
                
                # Extract the play button
                play_button = img_bgr[y:y+h, x:x+w]
                
                # Save the play button template
                template_path = os.path.join(TEMPLATES_DIR, "play_button.png")
                cv2.imwrite(template_path, play_button)
                
                # Show the extracted template
                cv2.imshow("Play Button Template", play_button)
                cv2.waitKey(2000)
                cv2.destroyAllWindows()
                
                print(f"Play button template saved to {template_path}")
                return template_path
    
    print("No suitable green area found for play button. Taking a screenshot of bottom area as fallback.")
    
    # Take a screenshot of the bottom area where play button is usually located
    height = GAME_REGION[3]
    width = GAME_REGION[2]
    bottom_area = img_bgr[height-100:height, width//2-100:width//2+100]
    
    # Save as template
    template_path = os.path.join(TEMPLATES_DIR, "play_button.png")
    cv2.imwrite(template_path, bottom_area)
    
    # Show the extracted template
    cv2.imshow("Play Button Template (Fallback)", bottom_area)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
    
    print(f"Fallback play button template saved to {template_path}")
    return template_path

def capture_game_over():
    """Capture the game over screen template"""
    print("\n=== Capturing Game Over Template ===")
    print("The script will take a screenshot of the game over screen")
    print("Make sure 'GAME OVER' text or similar is visible")
    
    input("Press Enter when the game over screen is visible...")
    
    # Take a screenshot of the game region
    screenshot = pyautogui.screenshot(region=GAME_REGION)
    img_array = np.array(screenshot)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Save the entire game over screen
    template_path = os.path.join(TEMPLATES_DIR, "gameover.png")
    cv2.imwrite(template_path, img_bgr)
    
    # Show the captured template
    cv2.imshow("Game Over Template", img_bgr)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
    
    print(f"Game over template saved to {template_path}")
    return template_path

def main():
    """Main function to capture all templates"""
    print("Subway Surfers AI - Template Capture Utility")
    print("===========================================")
    
    ensure_directories()
    
    # Capture game over template
    capture_game_over()
    
    # Capture play button template
    capture_play_button()
    
    print("\nAll templates captured successfully!")
    print("You can now run the AI training or play modes.")

if __name__ == "__main__":
    main()