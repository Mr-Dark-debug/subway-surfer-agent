import cv2
from datetime import datetime
import pyautogui
import re
from PIL import Image
import json
import sys
import pytesseract
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from subway_ai.config import *

# Set pytesseract path
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
else:
    print(f"Warning: Tesseract not found at {TESSERACT_PATH}")
    print("Please install Tesseract OCR and update the path in config.py")

class ScreenCapture:
    """Handles screen capture and processing for Subway Surfers AI"""
    
    def __init__(self):
        # Create necessary directories
        os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
        os.makedirs(os.path.join(SCREENSHOTS_DIR, "gameplay"), exist_ok=True)
        os.makedirs(os.path.join(SCREENSHOTS_DIR, "score"), exist_ok=True)
        os.makedirs(os.path.join(SCREENSHOTS_DIR, "coins"), exist_ok=True)
        os.makedirs(os.path.join(SCREENSHOTS_DIR, "game_over"), exist_ok=True)
        os.makedirs(TEMPLATES_DIR, exist_ok=True)
        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(LOGS_DIR, exist_ok=True)
        
        # Load regions from regions.json if it exists
        self.load_regions()
        
        # Initialize log file for OCR results
        self.log_file_path = os.path.join(LOGS_DIR, f"ocr_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        self.log_file = open(self.log_file_path, 'w')
        self.log_file.write("Timestamp,Step,Type,Raw_Text,Parsed_Value,Filtered_Value\n")
        
        # Initialize step counter
        self.steps = 0
        
        # Initialize last score and coins
        self.last_score = 0
        self.last_coins = 0
        
        # Load templates
        self.templates = {}
        self.load_templates()
        
        print(f"Screen capture initialized with regions:")
        print(f"Game region: {GAME_REGION}")
        print(f"Score region: {SCORE_REGION}")
        print(f"Coin region: {COIN_REGION}")
    
    def load_regions(self):
        """Load regions from regions.json if available"""
        json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "regions.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    regions = json.load(f)
                
                # Update global variables
                global GAME_REGION, SCORE_REGION, COIN_REGION
                if 'game' in regions and len(regions['game']) == 4:
                    GAME_REGION = tuple(regions['game'])
                if 'score' in regions and len(regions['score']) == 4:
                    SCORE_REGION = tuple(regions['score'])
                if 'coins' in regions and len(regions['coins']) == 4:
                    COIN_REGION = tuple(regions['coins'])
                
                print(f"Loaded regions from {json_path}")
            except Exception as e:
                print(f"Error loading regions from {json_path}: {e}")
    
    def load_templates(self):
        """Load template images for game state detection"""
        if os.path.exists(GAME_OVER_TEMPLATE):
            self.templates['game_over'] = cv2.imread(GAME_OVER_TEMPLATE, cv2.IMREAD_COLOR)
            print(f"Loaded 'game_over' template for detection")
        
        if os.path.exists(PLAY_BUTTON_TEMPLATE):
            self.templates['play_button'] = cv2.imread(PLAY_BUTTON_TEMPLATE, cv2.IMREAD_COLOR)
            print(f"Loaded 'play_button' template for detection")
    
    def capture_game_screen(self):
        """Capture the game screen"""
        return pyautogui.screenshot(region=GAME_REGION)
    
    def capture_score_screen(self):
        """Capture the score region"""
        return pyautogui.screenshot(region=SCORE_REGION)
    
    def capture_coin_screen(self):
        """Capture the coin region"""
        return pyautogui.screenshot(region=COIN_REGION)
    
    def save_screenshot(self, img, category, prefix=""):
        """Save a screenshot to the appropriate directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{self.steps}_{timestamp}.png"
        filepath = os.path.join(SCREENSHOTS_DIR, category, filename)
        
        if isinstance(img, Image.Image):
            img.save(filepath)
        else:
            cv2.imwrite(filepath, cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
        
        return filepath
    
    def preprocess_for_ocr(self, img, is_score=True):
        """Improved preprocessing for Subway Surfers score display"""
        # Convert to numpy array if needed
        if not isinstance(img, np.ndarray):
            img_array = np.array(img)
        else:
            img_array = img.copy()
        
        # Save original for debugging
        category = "score" if is_score else "coins"
        self.save_screenshot(img, category, "raw")
        
        # Apply preprocessing to enhance digit visibility
        
        # 1. Convert to grayscale
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array.copy()
        
        # 2. Apply thresholding for better OCR
        if is_score:
            # For score - white text on blue background
            _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        else:
            # For coins - usually yellow/gold on dark background
            _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        
        # 3. Apply morphological operations to clean up the image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        dilated = cv2.dilate(opening, kernel, iterations=1)
        
        # 4. Save processed image for debugging
        self.save_screenshot(dilated, category, "processed")
        
        return dilated
    
    def extract_score_ocr(self):
        """Extract score from screenshot using improved OCR"""
        try:
            score_img = self.capture_score_screen()
            
            # Enhanced preprocessing for Subway Surfers score display
            processed_img = self.preprocess_for_ocr(score_img, is_score=True)
            
            # Apply OCR with specific configuration for digits
            custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
            ocr_result = pytesseract.image_to_string(processed_img, config=custom_config).strip()
            
            # Log OCR results
            self.log_ocr_result("score", ocr_result)
            
            # Process OCR result - extract only digits
            digits_only = re.sub(r'\D', '', ocr_result)
            
            # Convert to integer, default to last score if conversion fails
            try:
                score = int(digits_only) if digits_only else self.last_score
            except ValueError:
                score = self.last_score
                
            # Apply consistency check
            if self.last_score > 0:
                # Score typically increases gradually in Subway Surfers
                max_increase = 50  # Maximum reasonable score increase between frames
                
                if score > self.last_score + max_increase:
                    # Probably a misread, use a reasonable increment
                    score = self.last_score + 10
                elif score < self.last_score:
                    # Scores shouldn't decrease during gameplay
                    score = self.last_score + 1  # Slight increment to show progress
            
            # Update last score
            self.last_score = score
            return score
            
        except Exception as e:
            print(f"Error extracting score: {e}")
            # Return last known score on error
            return self.last_score
    
    def extract_coins_ocr(self):
        """Extract coin count from screenshot using improved OCR"""
        try:
            coin_img = self.capture_coin_screen()
            
            # Enhanced preprocessing for Subway Surfers coin display
            processed_img = self.preprocess_for_ocr(coin_img, is_score=False)
            
            # Apply OCR with digit-specific configuration
            custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
            ocr_result = pytesseract.image_to_string(processed_img, config=custom_config).strip()
            
            # Log OCR results
            self.log_ocr_result("coins", ocr_result)
            
            # Process OCR result - extract only digits
            digits_only = re.sub(r'\D', '', ocr_result)
            
            # Convert to integer, default to last coins if conversion fails
            try:
                coins = int(digits_only) if digits_only else self.last_coins
            except ValueError:
                coins = self.last_coins
                
            # Apply consistency check
            if self.last_coins > 0:
                # Coins typically increase by small amounts in Subway Surfers
                max_increase = 5  # Maximum reasonable coin increase between frames
                
                if coins > self.last_coins + max_increase:
                    # Probably a misread, use a small increment
                    coins = self.last_coins + 1
                elif coins < self.last_coins:
                    # Coins shouldn't decrease during normal gameplay
                    coins = self.last_coins
            
            # Update last coins
            self.last_coins = coins
            return coins
            
        except Exception as e:
            print(f"Error extracting coins: {e}")
            # Return last known coins on error
            return self.last_coins
    
    def log_ocr_result(self, data_type, raw_text):
        """Log OCR results to file for debugging"""
        if self.log_file:
            try:
                # Extract digits only
                digits_only = re.sub(r'\D', '', raw_text)
                
                # Parse value
                try:
                    parsed_value = int(digits_only) if digits_only else -1
                except ValueError:
                    parsed_value = -1
                
                # Get filtered value
                if data_type == "score":
                    filtered_value = self.last_score
                else:
                    filtered_value = self.last_coins
                
                # Write to log
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                self.log_file.write(f"{timestamp},{self.steps},{data_type},{raw_text},{parsed_value},{filtered_value}\n")
                self.log_file.flush()  # Ensure it's written immediately
                
            except Exception as e:
                print(f"Error logging OCR result: {e}")
    
    def detect_game_over(self):
        """Detect if the game is over using multiple methods"""
        try:
            game_img = self.capture_game_screen()
            
            # Convert to numpy array if needed
            if isinstance(game_img, Image.Image):
                img_array = np.array(game_img)
            else:
                img_array = game_img.copy()
            
            # Game over detection method 1: Find the play button
            if self.locate_play_button() is not None:
                print("Play button found - game is likely over")
                self.save_screenshot(img_array, "game_over", f"game_over_play_button")
                return True
            
            # Game over detection method 2: Check for specific green button at bottom
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
                lower_green = np.array([40, 100, 100])
                upper_green = np.array([80, 255, 255])
                green_mask = cv2.inRange(hsv, lower_green, upper_green)
                
                # Focus on bottom portion of the screen where play button usually is
                bottom_portion = green_mask[int(green_mask.shape[0]*0.7):, :]
                if np.sum(bottom_portion) > 5000:  # Significant green area at bottom
                    print("Green area detected at bottom - likely play button")
                    self.save_screenshot(img_array, "game_over", f"game_over_green_button")
                    return True
            
            return False
            
        except Exception as e:
            print(f"Error in game over detection: {e}")
            return False
    
    def locate_play_button(self):
        """Locate the play button on the game over screen"""
        try:
            game_img = self.capture_game_screen()
            
            # Convert to numpy array if needed
            if isinstance(game_img, Image.Image):
                img_array = np.array(game_img)
            else:
                img_array = game_img.copy()
            
            # Method 1: Template matching
            if 'play_button' in self.templates and self.templates['play_button'] is not None:
                if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                else:
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
                    
                result = cv2.matchTemplate(img_bgr, self.templates['play_button'], cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                
                if max_val > 0.5:  # Lower threshold for better detection
                    # Get the center of the play button
                    h, w = self.templates['play_button'].shape[:2]
                    center_x = max_loc[0] + w // 2 + GAME_REGION[0]
                    center_y = max_loc[1] + h // 2 + GAME_REGION[1]
                    
                    print(f"Play button found at ({center_x}, {center_y}) with confidence {max_val:.2f}")
                    return (center_x, center_y)
            
            # Method 2: Color detection for green play button
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
                lower_green = np.array([40, 100, 100])
                upper_green = np.array([80, 255, 255])
                green_mask = cv2.inRange(hsv, lower_green, upper_green)
                
                # Find contours of green areas
                contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Sort by area (largest first)
                    contours = sorted(contours, key=cv2.contourArea, reverse=True)
                    
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if area > 1000:  # Min area threshold
                            # Get bounding box
                            x, y, w, h = cv2.boundingRect(contour)
                            
                            # Check if in bottom half of screen (play button location)
                            if y > img_array.shape[0] * 0.5:
                                center_x = x + w // 2 + GAME_REGION[0]
                                center_y = y + h // 2 + GAME_REGION[1]
                                print(f"Play button found using color detection at ({center_x}, {center_y})")
                                return (center_x, center_y)
            
            # Method 3: Use hardcoded play button position
            play_button_x = GAME_REGION[0] + GAME_REGION[2] // 2
            play_button_y = GAME_REGION[1] + GAME_REGION[3] - 60
            
            # Check if the position has a green color (play button)
            pixel_color = pyautogui.pixel(play_button_x, play_button_y)
            
            # If pixel has green component, it might be the play button
            if pixel_color[1] > 100 and pixel_color[1] > pixel_color[0] and pixel_color[1] > pixel_color[2]:
                print(f"Play button found at hardcoded position ({play_button_x}, {play_button_y})")
                return (play_button_x, play_button_y)
            
            return None
            
        except Exception as e:
            print(f"Error locating play button: {e}")
            return None
    
    def increment_step(self):
        """Increment the step counter"""
        self.steps += 1
        
    def close(self):
        """Close resources"""
        if self.log_file:
            self.log_file.close()