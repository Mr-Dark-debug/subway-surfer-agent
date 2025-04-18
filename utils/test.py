# #!/usr/bin/env python
# # simple_test.py - Simple test script for Subway Surfers region detection

# import cv2
# import numpy as np
# import pyautogui
# import time
# import os
# import logging

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger("SimpleRegionDetector")

# def capture_screen():
#     """Capture the current screen as a numpy array"""
#     # Take screenshot using pyautogui
#     screenshot = pyautogui.screenshot()
#     screenshot = np.array(screenshot)
#     screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
#     return screenshot

# def detect_regions(screenshot):
#     """
#     Detect game regions in the screenshot
    
#     Args:
#         screenshot: Screenshot as numpy array
        
#     Returns:
#         Screenshot with bounding boxes drawn
#     """
#     # Get screen dimensions
#     height, width = screenshot.shape[:2]
    
#     # Make a copy for drawing
#     debug_img = screenshot.copy()
    
#     # Define improved game region (center portion of screen)
#     x_offset = int(width * 0.40)  # 40% from left - adjust for your setup
#     y_offset = int(height * 0.10)  # 10% from top
#     game_width = int(width * 0.55)  # 55% of width
#     game_height = int(height * 0.80)  # 80% of height
    
#     game_region = (x_offset, y_offset, game_width, game_height)
    
#     # Score region (typically top right of game area)
#     score_x = x_offset + int(game_width * 0.65)
#     score_y = y_offset + int(game_height * 0.05)
#     score_width = int(game_width * 0.30)
#     score_height = int(game_height * 0.06)
#     score_region = (score_x, score_y, score_width, score_height)
    
#     # Coin region (typically below score)
#     coin_x = score_x
#     coin_y = score_y + score_height + 5  # Just below score
#     coin_width = score_width
#     coin_height = score_height
#     coin_region = (coin_x, coin_y, coin_width, coin_height)
    
#     # Draw regions on debug image
#     # Game region (green)
#     cv2.rectangle(
#         debug_img,
#         (game_region[0], game_region[1]),
#         (game_region[0] + game_region[2], game_region[1] + game_region[3]),
#         (0, 255, 0),
#         2
#     )
    
#     # Score region (blue)
#     cv2.rectangle(
#         debug_img,
#         (score_region[0], score_region[1]),
#         (score_region[0] + score_region[2], score_region[1] + score_region[3]),
#         (255, 0, 0),
#         2
#     )
    
#     # Coin region (yellow)
#     cv2.rectangle(
#         debug_img,
#         (coin_region[0], coin_region[1]),
#         (coin_region[0] + coin_region[2], coin_region[1] + coin_region[3]),
#         (0, 255, 255),
#         2
#     )
    
#     # Add text labels
#     cv2.putText(debug_img, "Game Region", (game_region[0], game_region[1] - 10), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#     cv2.putText(debug_img, "Score", (score_region[0], score_region[1] - 10), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
#     cv2.putText(debug_img, "Coins", (coin_region[0], coin_region[1] - 10), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
#     # Add the region coordinates as text
#     info_text = f"Game: {game_region}"
#     cv2.putText(debug_img, info_text, (10, height - 90), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
#     info_text = f"Score: {score_region}"
#     cv2.putText(debug_img, info_text, (10, height - 60), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
#     info_text = f"Coins: {coin_region}"
#     cv2.putText(debug_img, info_text, (10, height - 30), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
#     return debug_img, game_region, score_region, coin_region

# def main():
#     """Main function"""
#     try:
#         # Create debug directory if it doesn't exist
#         os.makedirs("debug_images", exist_ok=True)
        
#         logger.info("Taking screenshot and detecting regions...")
        
#         # Capture screen
#         screenshot = capture_screen()
        
#         # Detect regions
#         debug_img, game_region, score_region, coin_region = detect_regions(screenshot)
        
#         # Save debug image
#         timestamp = int(time.time())
#         filepath = f"debug_images/regions_{timestamp}.png"
#         cv2.imwrite(filepath, debug_img)
#         logger.info(f"Debug image saved to {filepath}")
        
#         # Show the image
#         # Resize if too large
#         height, width = debug_img.shape[:2]
#         if height > 900 or width > 1600:
#             scale = min(900 / height, 1600 / width)
#             new_width = int(width * scale)
#             new_height = int(height * scale)
#             debug_img = cv2.resize(debug_img, (new_width, new_height))
        
#         cv2.imshow("Detected Regions", debug_img)
#         logger.info("Press any key to close the window")
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
        
#         # Print region values for copying
#         print("\nRegion values to copy to game_interaction.py:")
#         print(f"self.game_region = {game_region}")
#         print(f"self.score_region = {score_region}")
#         print(f"self.coin_region = {coin_region}")
        
#     except Exception as e:
#         logger.error(f"Error in main: {str(e)}")

# if __name__ == "__main__":
#     main()

import pyautogui
res = pyautogui.locateOnScreen("./utils/image.png")
print(res)