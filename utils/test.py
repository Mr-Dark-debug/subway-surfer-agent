#!/usr/bin/env python
# improved_test.py - Subway Surfers region detection with interactive GUI

import cv2
import numpy as np
import pyautogui
import time
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ImprovedRegionDetector")

# Button parameters
BUTTON_WIDTH = 80
BUTTON_HEIGHT = 40
BUTTON_SPACING = 20
PARAM_LIST_Y = 100
PARAM_ITEM_HEIGHT = 40

def capture_screen():
    """Capture the current screen as a numpy array"""
    screenshot = pyautogui.screenshot()
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

class InteractiveAdjuster:
    def __init__(self):
        self.screenshot = capture_screen()
        self.height, self.width = self.screenshot.shape[:2]
        self.params = {
            'game_x': 0.52, 'game_y': 0.175, 
            'game_w': 0.42, 'game_h': 0.75,
            'score_xf': 0.65, 'score_y': 0.15,
            'score_w': 0.33, 'score_h': 0.05,
            'coin_gap': 8
        }
        self.current_param = 'game_x'
        self.step_size = 0.01
        self.param_list = [
            ('game_x', 'Game X%', 'percentage'),
            ('game_y', 'Game Y%', 'percentage'),
            ('game_w', 'Game Width%', 'percentage'),
            ('game_h', 'Game Height%', 'percentage'),
            ('score_xf', 'Score X Factor', 'factor'),
            ('score_y', 'Score Y%', 'percentage'),
            ('score_w', 'Score Width%', 'factor'),
            ('score_h', 'Score Height%', 'percentage'),
            ('coin_gap', 'Coin Gap', 'pixels')
        ]
        self.mouse_pos = (-1, -1)
        self.last_click = None
        cv2.namedWindow("Region Adjustment")
        cv2.setMouseCallback("Region Adjustment", self.mouse_handler)

    def mouse_handler(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_pos = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.last_click = (x, y)

    def calculate_regions(self):
        x = int(self.width * self.params['game_x'])
        y = int(self.height * self.params['game_y'])
        w = int(self.width * self.params['game_w'])
        h = int(self.height * self.params['game_h'])
        game = (x, y, w, h)

        score_x = x + int(w * self.params['score_xf'])
        score_y = int(self.height * self.params['score_y'])
        score_w = int(w * self.params['score_w'])
        score_h = int(self.height * self.params['score_h'])
        score = (score_x, score_y, score_w, score_h)

        coin = (score_x, score_y + score_h + self.params['coin_gap'], score_w, score_h)
        return game, score, coin

    def draw_interface(self, debug_img):
        # Draw regions
        game, score, coin = self.calculate_regions()
        cv2.rectangle(debug_img, game[:2], (game[0]+game[2], game[1]+game[3]), (0,255,0), 2)
        cv2.rectangle(debug_img, score[:2], (score[0]+score[2], score[1]+score[3]), (255,0,0), 2)
        cv2.rectangle(debug_img, coin[:2], (coin[0]+coin[2], coin[1]+coin[3]), (0,255,255), 2)

        # Draw parameter list
        for i, (key, label, typ) in enumerate(self.param_list):
            y = PARAM_LIST_Y + i * PARAM_ITEM_HEIGHT
            color = (0, 255, 0) if key == self.current_param else (255, 255, 255)
            
            # Parameter label
            cv2.putText(debug_img, label, (20, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            
            # Parameter value
            value = self.params[key]
            if typ == 'percentage':
                val_text = f"{value:.2f}"
            else:
                val_text = str(value)
            cv2.putText(debug_img, val_text, (200, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            
            # Buttons
            if self.is_mouse_over_param(i):
                cv2.rectangle(debug_img, (300, y-25), (380, y+5), (100,100,100), -1)
            else:
                cv2.rectangle(debug_img, (300, y-25), (380, y+5), (50,50,50), -1)
            
            cv2.putText(debug_img, "-", (310, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.putText(debug_img, "+", (350, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        # Draw control buttons
        accept_btn = (self.width-200, self.height-50, 180, 40)
        cv2.rectangle(debug_img, 
                      (accept_btn[0], accept_btn[1]),
                      (accept_btn[0]+accept_btn[2], accept_btn[1]+accept_btn[3]),
                      (0, 200, 0) if self.is_mouse_over(accept_btn) else (0, 150, 0), -1)
        cv2.putText(debug_img, "Accept Regions", (accept_btn[0]+10, accept_btn[1]+25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        return debug_img

    def is_mouse_over_param(self, param_index):
        y = PARAM_LIST_Y + param_index * PARAM_ITEM_HEIGHT
        return 300 <= self.mouse_pos[0] <= 380 and y-25 <= self.mouse_pos[1] <= y+5

    def is_mouse_over(self, rect):
        x, y, w, h = rect
        return (x <= self.mouse_pos[0] <= x + w and 
                y <= self.mouse_pos[1] <= y + h)

    def handle_clicks(self):
        if not self.last_click:
            return False

        # Check parameter buttons
        for i, (key, _, _) in enumerate(self.param_list):
            y = PARAM_LIST_Y + i * PARAM_ITEM_HEIGHT
            if 300 <= self.last_click[0] <= 380 and y-25 <= self.last_click[1] <= y+5:
                if 300 <= self.last_click[0] < 340:  # Minus button
                    self.adjust_param(key, -1)
                else:  # Plus button
                    self.adjust_param(key, 1)
                self.last_click = None
                return True

        # Check accept button
        accept_btn = (self.width-200, self.height-50, 180, 40)
        if self.is_mouse_over(accept_btn):
            self.last_click = None
            return True  # Signal to exit

        self.last_click = None
        return False

    def adjust_param(self, param, direction):
        if param == 'coin_gap':
            self.params[param] = max(0, self.params[param] + direction)
        else:
            step = self.step_size * direction
            self.params[param] = max(0.01, min(0.99, self.params[param] + step))

    def run(self):
        while True:
            debug_img = self.screenshot.copy()
            debug_img = self.draw_interface(debug_img)
            
            # Handle keyboard
            key = cv2.waitKey(25) & 0xFF
            if key == 27:  # ESC
                return None, None, None
            elif key == 13:  # Enter
                break
            
            # Handle mouse
            if self.handle_clicks():
                break

            cv2.imshow("Region Adjustment", debug_img)

        cv2.destroyAllWindows()
        return self.calculate_regions()

def main():
    try:
        os.makedirs("debug_images", exist_ok=True)
        
        while True:
            print("\n1. Auto detection\n2. Interactive adjust\n3. Exit")
            choice = input("Choice: ").strip()
            
            if choice == "1":
                screenshot = capture_screen()
                debug_img, *regions = detect_regions(screenshot)
            elif choice == "2":
                adjuster = InteractiveAdjuster()
                regions = adjuster.run()
                if not regions: continue
                debug_img = adjuster.screenshot.copy()
                debug_img = adjuster.draw_interface(debug_img)
            elif choice == "3":
                return
            else:
                continue
            
            # Save and show results
            timestamp = int(time.time())
            cv2.imwrite(f"debug_images/regions_{timestamp}.png", debug_img)
            cv2.imshow("Results", debug_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()