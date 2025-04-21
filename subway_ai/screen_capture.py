# Improved screen_capture.py with better game over detection
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
import time

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
        
        # Game state tracking for stability
        self.consecutive_play_button_detections = 0
        self.last_known_play_button = None
        self.consecutive_identical_scores = 0
        self.game_active = False
        self.last_game_over_check_time = time.time()
        
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
        
        # 2. Apply adaptive thresholding for better OCR (improved method)
        if is_score:
            # For score - typically white text on blue/dark background
            # Try adaptive thresholding which works better with varying lighting
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 21, 10)
        else:
            # For coins - usually yellow/gold on dark background
            # First enhance yellow/gold colors in original image
            if len(img_array.shape) == 3:
                # Convert to HSV
                hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
                # Yellow mask
                lower_yellow = np.array([20, 100, 100])
                upper_yellow = np.array([40, 255, 255])
                yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
                # Combine with grayscale for better results
                binary = cv2.bitwise_or(gray, yellow_mask)
                _, binary = cv2.threshold(binary, 100, 255, cv2.THRESH_BINARY)
            else:
                # If already grayscale
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
                max_increase = 100  # Maximum reasonable score increase between frames
                
                if score > self.last_score + max_increase:
                    # Large jumps are suspicious - check if it's a clean read
                    if len(digits_only) != len(str(self.last_score)):
                        # Different number of digits - likely misread
                        score = self.last_score + 10  # Use modest increment
                    elif score > self.last_score * 2:
                        # More than doubled - highly suspicious
                        score = self.last_score + 10
                elif score < self.last_score:
                    # Scores shouldn't decrease during gameplay
                    # Check if the decrease is significant
                    if self.last_score - score > 50:
                        # Significant decrease - likely a misread
                        score = self.last_score  # Maintain previous score
                    else:
                        # Small decrease - might be correct (e.g., after crash)
                        pass
                        
                # Check for consecutive identical scores
                if score == self.last_score:
                    self.consecutive_identical_scores += 1
                    if self.consecutive_identical_scores > 20:  # If score hasn't changed for many frames
                        # Introduce small increment to show progress
                        score = self.last_score + 1
                        self.consecutive_identical_scores = 0
                else:
                    self.consecutive_identical_scores = 0
            
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
                    # But allow for small variations due to OCR errors
                    if self.last_coins - coins > 2:
                        coins = self.last_coins
                    # else accept the small decrease
            
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
    
    def _is_likely_play_button(self, contour, img_shape, min_area=1000):
        """Determine if a contour is likely to be a play button"""
        area = cv2.contourArea(contour)
        
        # Check area size
        if area < min_area:
            return False
        
        # Get bounding box to analyze shape
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate aspect ratio and relative position
        aspect_ratio = float(w) / h if h > 0 else 0
        relative_y_pos = y / img_shape[0] if img_shape[0] > 0 else 0
        
        # Play button criteria:
        # 1. Should be somewhat wider than tall (typical button shape)
        # 2. Should be in the bottom half of the screen
        # 3. Should have a reasonable size relative to the image
        is_button_shape = 1.5 < aspect_ratio < 5
        is_in_bottom_half = relative_y_pos > 0.6
        relative_size = area / (img_shape[0] * img_shape[1])
        is_reasonable_size = 0.005 < relative_size < 0.1  # Between 0.5% and 10% of screen
        
        return is_button_shape and is_in_bottom_half and is_reasonable_size
    
    def _calculate_symmetry(self, mask, x, y, w, h):
        """Calculate horizontal symmetry of a potential button region"""
        if w < 10 or h < 10:  # Too small to check symmetry
            return 0
            
        # Extract region of interest
        roi = mask[y:y+h, x:x+w]
        
        # Split into left and right halves
        mid = w // 2
        left_half = roi[:, :mid]
        right_half = roi[:, mid:2*mid] if 2*mid <= w else roi[:, mid:]
        
        # If right half is wider, crop it
        if right_half.shape[1] > left_half.shape[1]:
            right_half = right_half[:, :left_half.shape[1]]
            
        # If right half is narrower, pad it
        elif right_half.shape[1] < left_half.shape[1]:
            pad_width = left_half.shape[1] - right_half.shape[1]
            right_half = np.pad(right_half, ((0, 0), (0, pad_width)), 'constant')
        
        # Flip right half horizontally
        right_half_flipped = cv2.flip(right_half, 1)
        
        # Calculate similarity (intersection over union)
        intersection = np.logical_and(left_half, right_half_flipped).sum()
        union = np.logical_or(left_half, right_half_flipped).sum()
        
        if union == 0:
            return 0
            
        return intersection / union
    
    def detect_game_over(self):
        """Improved game over detection with stability checks"""
        # Throttle game over checks to avoid overloading
        current_time = time.time()
        if current_time - self.last_game_over_check_time < 0.5:  # Only check every 0.5 seconds
            return False
        
        self.last_game_over_check_time = current_time
        
        try:
            game_img = self.capture_game_screen()
            
            # Try template matching for game over text/screen
            if 'game_over' in self.templates and self.templates['game_over'] is not None:
                # Convert to numpy array if needed
                if isinstance(game_img, Image.Image):
                    img_array = np.array(game_img)
                    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    else:
                        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
                else:
                    img_bgr = game_img
                
                # Perform template matching
                try:
                    result = cv2.matchTemplate(img_bgr, self.templates['game_over'], cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(result)
                    
                    # High confidence game over detection
                    if max_val > 0.7:
                        self.save_screenshot(img_array, "game_over", f"game_over_template")
                        return True
                except Exception as e:
                    print(f"Error in template matching: {e}")
            
            # Check for play button
            play_button_loc = self.locate_play_button()
            if play_button_loc is not None:
                # If we've detected the play button in the same location multiple times
                if (self.last_known_play_button is not None and 
                    abs(play_button_loc[0] - self.last_known_play_button[0]) < 10 and
                    abs(play_button_loc[1] - self.last_known_play_button[1]) < 10):
                    
                    self.consecutive_play_button_detections += 1
                    
                    # Only consider game over if we've seen the play button multiple times
                    if self.consecutive_play_button_detections >= 3:
                        self.save_screenshot(game_img, "game_over", f"game_over_play_button")
                        return True
                else:
                    # Reset counter for new position
                    self.consecutive_play_button_detections = 1
                    
                self.last_known_play_button = play_button_loc
            else:
                # Reset counter if no play button detected
                self.consecutive_play_button_detections = 0
                self.last_known_play_button = None
            
            # Check for static score as a sign of game over
            # If score hasn't changed for a long time, it might indicate game over
            if self.consecutive_identical_scores > 30:  # If score hasn't changed for 30 frames
                self.save_screenshot(game_img, "game_over", f"game_over_static_score")
                return True
                
            return False
            
        except Exception as e:
            print(f"Error in game over detection: {e}")
            return False
    
    def locate_play_button(self):
        """Improved locate play button function with more robust detection"""
        try:
            game_img = self.capture_game_screen()
            
            # Convert to numpy array if needed
            if isinstance(game_img, Image.Image):
                img_array = np.array(game_img)
            else:
                img_array = game_img.copy()
            
            # METHOD 1: Template matching
            if 'play_button' in self.templates and self.templates['play_button'] is not None:
                try:
                    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    else:
                        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
                        
                    result = cv2.matchTemplate(img_bgr, self.templates['play_button'], cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(result)
                    
                    if max_val > 0.7:  # Higher threshold for better reliability
                        # Get the center of the play button
                        h, w = self.templates['play_button'].shape[:2]
                        center_x = max_loc[0] + w // 2 + GAME_REGION[0]
                        center_y = max_loc[1] + h // 2 + GAME_REGION[1]
                        
                        return (center_x, center_y)
                except Exception as e:
                    print(f"Error in template matching: {e}")
            
            # METHOD 2: Color detection for green play button
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                # Convert to HSV for better color detection
                hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
                
                # More specific green color range for the play button
                lower_green = np.array([45, 120, 120])  # More specific green
                upper_green = np.array([75, 255, 255])  # More specific green
                
                # Create a mask for green areas
                green_mask = cv2.inRange(hsv, lower_green, upper_green)
                
                # Apply morphology to clean the mask
                kernel = np.ones((5, 5), np.uint8)
                green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
                green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
                
                # Check if we have sufficient green pixels
                if np.sum(green_mask) < 5000:  # Minimum number of green pixels
                    return None
                
                # Find contours of green areas
                contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Sort by area (largest first)
                    contours = sorted(contours, key=cv2.contourArea, reverse=True)
                    
                    for contour in contours:
                        # Check if contour is likely a play button using multiple criteria
                        if self._is_likely_play_button(contour, img_array.shape[:2]):
                            # Get bounding box and center point
                            x, y, w, h = cv2.boundingRect(contour)
                            
                            # Additional validation: check for symmetry typical of buttons
                            symmetry_score = self._calculate_symmetry(green_mask, x, y, w, h)
                            if symmetry_score > 0.7:  # High symmetry requirement
                                center_x = x + w//2 + GAME_REGION[0]
                                center_y = y + h//2 + GAME_REGION[1]
                                return (center_x, center_y)
            
            # METHOD 3: Only use fallback if needed for user-initiated restart
            # Avoid using the fallback for automatic game over detection
            # as it can lead to false positives
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