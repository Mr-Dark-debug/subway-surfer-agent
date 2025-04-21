# save_templates.py
import os
import pyautogui
import cv2
import numpy as np
import time
from subway_ai.config import *

def ensure_dirs():
    """Ensure necessary directories exist"""
    os.makedirs(TEMPLATES_DIR, exist_ok=True)
    
def capture_template(region, filename, description):
    """Capture a template image from screen region"""
    print(f"Position your mouse over the center of the {description}")
    print("You have 5 seconds to position...")
    for i in range(5, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    
    # Capture the region
    screenshot = pyautogui.screenshot(region=region)
    
    # Save the template
    filepath = os.path.join(TEMPLATES_DIR, filename)
    screenshot.save(filepath)
    print(f"Template saved to {filepath}")
    
    # Display the template
    img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    cv2.imshow(f"{description} Template", img)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
    
    return filepath

def main():
    ensure_dirs()
    
    # Capture game over template
    print("\n=== Capturing Game Over Template ===")
    print("Please get to a game over screen")
    input("Press Enter when ready...")
    # Approximate region for game over - will be adjusted with calibration
    game_over_region = (GAME_REGION[0] + 200, GAME_REGION[1] + 150, 400, 200)
    game_over_path = capture_template(game_over_region, "gameover.png", "game over screen")
    
    # Capture play button template
    print("\n=== Capturing Play Button Template ===")
    print("Make sure the play button is visible")
    input("Press Enter when ready...")
    # Approximate region for play button - will be adjusted with calibration
    play_button_region = (GAME_REGION[0] + 300, GAME_REGION[1] + 400, 200, 100)
    play_button_path = capture_template(play_button_region, "play_button.png", "play button")
    
    print("\nTemplates captured successfully!")
    print(f"Game Over template: {game_over_path}")
    print(f"Play Button template: {play_button_path}")

if __name__ == "__main__":
    main()