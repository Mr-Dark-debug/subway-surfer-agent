# Improved game_control.py with better swipe mechanics and game state detection
import os
import time
import subprocess
import pyautogui
import numpy as np
import random
import cv2
from datetime import datetime
import json
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from subway_ai.config import *

class GameControl:
    """Handles game control for Subway Surfers AI with improved stability"""
    
    def __init__(self):
        # Initialize pyautogui settings
        pyautogui.PAUSE = 0.1  # Increased pause time for more reliable actions
        pyautogui.FAILSAFE = True
        
        # Get screen size
        self.screen_width, self.screen_height = pyautogui.size()
        print(f"Screen size: {self.screen_width}x{self.screen_height}")
        
        # Load regions from regions.json if it exists
        self.load_regions()
        
        # Print game regions
        print("Game region:", GAME_REGION)
        print("Score region:", SCORE_REGION)
        print("Coin region:", COIN_REGION)
        
        # Load templates
        self.game_over_template = None
        self.play_button_template = None
        self.load_templates()
        
        # Game state tracking
        self.game_running = False
        self.last_action_time = time.time()
        self.action_cooldown = 0.25  # Increased cooldown between actions for stability
        self.last_restart_time = 0
        self.restart_cooldown = 5.0  # Seconds to wait between restart attempts
        
        # Consecutive detection counters for stability
        self.consecutive_play_button_detections = 0
        self.last_known_play_button = None
        
        # Recording
        self.recording = False
        self.video_writer = None
        self.recording_start_time = None
        
        # Stats for real-time display
        self.stats = {
            'total_games': 0,
            'total_steps': 0,
            'highest_score': 0,
            'highest_coins': 0,
            'total_rewards': 0,
            'current_epsilon': 1.0
        }
    
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
            self.game_over_template = cv2.imread(GAME_OVER_TEMPLATE, cv2.IMREAD_COLOR)
            print(f"Loaded 'game_over' template for detection")
        else:
            print(f"Warning: Game over template not found at {GAME_OVER_TEMPLATE}")
        
        if os.path.exists(PLAY_BUTTON_TEMPLATE):
            self.play_button_template = cv2.imread(PLAY_BUTTON_TEMPLATE, cv2.IMREAD_COLOR)
            print(f"Loaded 'play_button' template for detection")
        else:
            print(f"Warning: Play button template not found at {PLAY_BUTTON_TEMPLATE}")
    
    def check_game_running(self):
        """Check if the Subway Surfers game is running"""
        try:
            # Try to take a screenshot of the game region
            game_img = pyautogui.screenshot(region=GAME_REGION)
            # Convert to numpy array and check if it's not all black
            img_array = np.array(game_img)
            
            # More sophisticated check - look for non-black areas and color variation
            if img_array.mean() > 20 and img_array.std() > 30:  # Higher thresholds
                # Check for content variation (not just a solid color)
                channels_std = [img_array[:,:,i].std() for i in range(3)]
                if all(std > 20 for std in channels_std):  # Ensure variation in all color channels
                    self.game_running = True
                    return True
            
            self.game_running = False
            return False
        except Exception as e:
            print(f"Error checking if game is running: {e}")
            self.game_running = False
            return False
    
    def start_game(self):
        """Start the Subway Surfers game if it's not already running"""
        if not self.check_game_running():
            try:
                # Start the game executable
                subprocess.Popen([GAME_PATH])
                print(f"Starting game from {GAME_PATH}")
                
                # Wait for the game to load - increased wait time
                for i in range(10):
                    print(f"Waiting for game to load... {i+1}/10")
                    time.sleep(1)
                    if self.check_game_running():
                        break
                
                # Make sure the window is in focus
                self.focus_game_window()
                
                # Wait a bit more for the game to stabilize
                time.sleep(3)
                
                # Start recording gameplay
                self.start_recording()
                
                self.game_running = True
                return True
            except Exception as e:
                print(f"Error starting game: {e}")
                return False
        else:
            print("Game is already running")
            # Make sure the window is focused
            self.focus_game_window()
            self.game_running = True
            return True
    
    def focus_game_window(self):
        """Focus on the game window by clicking in the center of the game region"""
        center_x = GAME_REGION[0] + GAME_REGION[2] // 2
        center_y = GAME_REGION[1] + GAME_REGION[3] // 2
        
        # Move mouse first to avoid accidental drags
        pyautogui.moveTo(center_x, center_y)
        time.sleep(0.2)  # Short pause
        pyautogui.click(center_x, center_y)
        time.sleep(0.5)
        print(f"Focusing game window by clicking at ({center_x}, {center_y})")
    
    def _is_likely_play_button(self, contour, img_shape, min_area=1000):
        """Determine if a contour is likely to be a play button based on shape and position"""
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
        """Improved detect game over function with multiple detection methods"""
        # Take a screenshot of the game region
        game_img = pyautogui.screenshot(region=GAME_REGION)
        game_img_np = np.array(game_img)
        game_img_cv = cv2.cvtColor(game_img_np, cv2.COLOR_RGB2BGR)
        
        # METHOD 1: Template matching for game over screen
        if self.game_over_template is not None:
            try:
                result = cv2.matchTemplate(game_img_cv, self.game_over_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                
                if max_val > 0.7:  # Higher threshold for more confidence
                    print(f"Game over screen detected with confidence {max_val:.2f}")
                    return True
            except Exception as e:
                print(f"Error in template matching: {e}")
        
        # METHOD 2: Find the play button using improved detection
        play_button_loc = self.find_play_button(game_img_np)
        if play_button_loc is not None:
            # If play button detected in the same location multiple times
            if (self.last_known_play_button is not None and 
                abs(play_button_loc[0] - self.last_known_play_button[0]) < 10 and
                abs(play_button_loc[1] - self.last_known_play_button[1]) < 10):
                
                self.consecutive_play_button_detections += 1
                
                # Only consider game over if we've seen the play button multiple times
                if self.consecutive_play_button_detections >= 3:
                    print("Play button detected consistently - game is likely over")
                    return True
            else:
                # Reset counter for new position
                self.consecutive_play_button_detections = 1
                
            self.last_known_play_button = play_button_loc
        else:
            # Reset counter if no play button detected
            self.consecutive_play_button_detections = 0
            self.last_known_play_button = None
        
        # METHOD 3: Check for specific game over indicators
        # 1. Look for text areas that might contain "GAME OVER"
        # Convert to grayscale
        gray = cv2.cvtColor(game_img_cv, cv2.COLOR_BGR2GRAY)
        # Apply threshold to find text
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        # Look for horizontal text-like structures in the upper half of the screen
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        upper_half = detected_lines[:detected_lines.shape[0]//2, :]
        if np.sum(upper_half) > 5000:  # Significant text detected in upper half
            # Additional check - look for vertical alignment typical of "GAME OVER" text
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
            vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
            if np.sum(vertical_lines[:vertical_lines.shape[0]//2, :]) > 3000:
                print("Text indicators suggest game over screen")
                return True
        
        return False
    
    def find_play_button(self, img=None):
        """Find the play button using multiple detection methods"""
        if img is None:
            # Take a screenshot of the game region
            game_img = pyautogui.screenshot(region=GAME_REGION)
            img = np.array(game_img)
        
        # Convert to BGR for OpenCV if it's RGB
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            # If grayscale or already BGR
            img_bgr = img
        
        # METHOD 1: Template matching
        if self.play_button_template is not None:
            try:
                result = cv2.matchTemplate(img_bgr, self.play_button_template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                
                if max_val > 0.7:  # Higher threshold for better reliability
                    # Get the center of the template
                    h, w = self.play_button_template.shape[:2]
                    center_x = max_loc[0] + w // 2 + GAME_REGION[0]
                    center_y = max_loc[1] + h // 2 + GAME_REGION[1]
                    print(f"Play button found with high confidence {max_val:.2f} at ({center_x}, {center_y})")
                    return (center_x, center_y)
            except Exception as e:
                print(f"Error in template matching: {e}")
        
        # METHOD 2: Enhanced color detection for green play button
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            
            # More specific green color range for play button
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
                    if self._is_likely_play_button(contour, img.shape[:2]):
                        # Get bounding box and center point
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # Additional validation: check for symmetry typical of buttons
                        symmetry_score = self._calculate_symmetry(green_mask, x, y, w, h)
                        if symmetry_score > 0.65:  # High symmetry requirement
                            center_x = x + w//2 + GAME_REGION[0]
                            center_y = y + h//2 + GAME_REGION[1]
                            print(f"Play button found using enhanced color detection at ({center_x}, {center_y})")
                            
                            # Save button image for debugging (if needed)
                            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            # debug_dir = os.path.join(SCREENSHOTS_DIR, "debug")
                            # os.makedirs(debug_dir, exist_ok=True)
                            # cv2.imwrite(os.path.join(debug_dir, f"play_button_{timestamp}.png"), 
                            #             img_bgr[y:y+h, x:x+w])
                            
                            return (center_x, center_y)
        except Exception as e:
            print(f"Error in color detection: {e}")
        
        # METHOD 3: Use hardcoded central play button position only if necessary
        # for user-initiated restart, not for automatic detection
        return None
    
    def restart_game(self, play_button_location=None):
        """Restart the game after game over with improved reliability"""
        current_time = time.time()
        
        # Enforce cooldown period between restart attempts
        if current_time - self.last_restart_time < self.restart_cooldown:
            print(f"Waiting for restart cooldown ({self.restart_cooldown - (current_time - self.last_restart_time):.1f}s left)")
            return False
        
        self.last_restart_time = current_time
        
        # First ensure the game window is focused
        self.focus_game_window()
        time.sleep(0.5)  # Short pause after focusing
        
        # Try different methods to find and click the play button
        if not play_button_location:
            # Try to find the play button
            play_button_location = self.find_play_button()
        
        # Click the play button if found
        if play_button_location:
            # Move mouse first, then click
            pyautogui.moveTo(play_button_location[0], play_button_location[1])
            time.sleep(0.3)  # Short pause
            pyautogui.click()
            print(f"Clicked play button at {play_button_location}")
            time.sleep(1.5)  # Longer wait for the click to register
            
            # Increment total games counter
            self.stats['total_games'] += 1
            print(f"Total games played: {self.stats['total_games']}")
            
            # Clear any cached button positions
            self.last_known_play_button = None
            self.consecutive_play_button_detections = 0
            
            self.game_running = True
            return True
        
        # If no play button found, try clicking in common locations
        common_positions = [
            (GAME_REGION[0] + GAME_REGION[2] // 2, GAME_REGION[1] + GAME_REGION[3] - 100),  # Bottom center
            (GAME_REGION[0] + GAME_REGION[2] // 2, GAME_REGION[1] + GAME_REGION[3] - 150),  # Slightly above bottom center
            (GAME_REGION[0] + GAME_REGION[2] // 2, GAME_REGION[1] + GAME_REGION[3] // 2 + 100)  # Below center
        ]
        
        for pos in common_positions:
            # Check if the position has green color (possible play button)
            try:
                pixel_color = pyautogui.pixel(pos[0], pos[1])
                # If pixel has strong green component
                if pixel_color[1] > 120 and pixel_color[1] > pixel_color[0] * 1.5 and pixel_color[1] > pixel_color[2] * 1.5:
                    print(f"Clicking likely play button at {pos}")
                    pyautogui.click(pos[0], pos[1])
                    time.sleep(1.5)
                    
                    # Increment total games counter
                    self.stats['total_games'] += 1
                    print(f"Total games played: {self.stats['total_games']}")
                    
                    self.game_running = True
                    return True
            except Exception as e:
                print(f"Error checking pixel at {pos}: {e}")
        
        print("Failed to find play button")
        return False
    
    def perform_action(self, action):
        """Improved action performance with more controlled swipe gestures"""
        # Check cooldown to prevent too many actions too quickly
        current_time = time.time()
        if current_time - self.last_action_time < self.action_cooldown:
            # Still in cooldown
            time.sleep(self.action_cooldown - (current_time - self.last_action_time))
            current_time = time.time()  # Update after wait
        
        # Get action name
        action_name = ACTIONS.get(action, "NONE")
        
        # Get the center of the game region as the starting point for swipes
        center_x = GAME_REGION[0] + GAME_REGION[2] // 2
        center_y = GAME_REGION[1] + GAME_REGION[3] // 2
        
        # Calculate swipe distance based on game region size
        swipe_distance_x = GAME_REGION[2] // 4
        swipe_distance_y = GAME_REGION[3] // 4
        
        # Swipe duration (slower for more reliable detection)
        swipe_duration = 0.15
        
        try:
            # Perform swipe action based on the action name
            if action_name == "LEFT":
                # Swipe from center to left
                pyautogui.moveTo(center_x, center_y)
                time.sleep(0.05)  # Short pause before mouseDown
                pyautogui.mouseDown(button='left')
                pyautogui.moveTo(center_x - swipe_distance_x, center_y, duration=swipe_duration)
                pyautogui.mouseUp(button='left')
                # Return to center
                pyautogui.moveTo(center_x, center_y)
            elif action_name == "RIGHT":
                # Swipe from center to right
                pyautogui.moveTo(center_x, center_y)
                time.sleep(0.05)
                pyautogui.mouseDown(button='left')
                pyautogui.moveTo(center_x + swipe_distance_x, center_y, duration=swipe_duration)
                pyautogui.mouseUp(button='left')
                # Return to center
                pyautogui.moveTo(center_x, center_y)
            elif action_name == "UP":
                # Swipe from center to up (jump)
                pyautogui.moveTo(center_x, center_y)
                time.sleep(0.05)
                pyautogui.mouseDown(button='left')
                pyautogui.moveTo(center_x, center_y - swipe_distance_y, duration=swipe_duration)
                pyautogui.mouseUp(button='left')
                # Return to center
                pyautogui.moveTo(center_x, center_y)
            elif action_name == "DOWN":
                # Swipe from center to down (roll)
                pyautogui.moveTo(center_x, center_y)
                time.sleep(0.05)
                pyautogui.mouseDown(button='left')
                pyautogui.moveTo(center_x, center_y + swipe_distance_y, duration=swipe_duration)
                pyautogui.mouseUp(button='left')
                # Return to center
                pyautogui.moveTo(center_x, center_y)
            # For NONE action, do nothing but return to center
            else:
                # Just move to center, no click or drag
                pyautogui.moveTo(center_x, center_y)
                time.sleep(swipe_duration)  # Still pause to maintain timing
        except Exception as e:
            print(f"Error performing action {action_name}: {e}")
        
        # Print the action for debugging
        if action_name != "NONE":
            print(f"Performed swipe action: {action_name}")
        
        self.last_action_time = time.time()
        
        # Update stats
        self.stats['total_steps'] += 1
        
        return action_name
    
    def random_action(self):
        """Return a random action for exploration"""
        return random.randint(0, NUM_ACTIONS - 1)
    
    def start_recording(self, output_path=None):
        """Start recording the game screen"""
        if self.recording:
            print("Already recording")
            return False
        
        # If no output path provided, create one
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs(os.path.join(BASE_DIR, "recordings"), exist_ok=True)
            output_path = os.path.join(BASE_DIR, "recordings", f"gameplay_{timestamp}.mp4")
        
        # Get the game region dimensions
        width, height = GAME_REGION[2], GAME_REGION[3]
        
        # Initialize the video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
        
        if not self.video_writer.isOpened():
            print(f"Error: Could not open video writer for {output_path}")
            return False
        
        self.recording = True
        self.recording_start_time = time.time()
        print(f"Started recording to {output_path}")
        return True
    
    def stop_recording(self):
        """Stop recording the game screen"""
        if not self.recording:
            print("Not recording")
            return False
        
        # Release the video writer
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        
        self.recording = False
        duration = time.time() - self.recording_start_time
        print(f"Stopped recording after {duration:.2f} seconds")
        return True
    
    def record_frame(self):
        """Record a single frame of the game"""
        if not self.recording or self.video_writer is None:
            return False
        
        try:
            # Capture the game screen
            game_img = pyautogui.screenshot(region=GAME_REGION)
            frame = np.array(game_img)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Write the frame to the video
            self.video_writer.write(frame)
            return True
        except Exception as e:
            print(f"Error recording frame: {e}")
            return False
    
    def update_stats(self, score, coins, reward, epsilon):
        """Update and display real-time training statistics"""
        # Update highest stats
        if score > self.stats['highest_score']:
            self.stats['highest_score'] = score
        if coins > self.stats['highest_coins']:
            self.stats['highest_coins'] = coins
        
        # Update current stats
        self.stats['total_rewards'] += reward
        self.stats['current_epsilon'] = epsilon

        # Display stats
        print(f"Score: {score}, Coins: {coins}, Reward: {reward:.2f}, Epsilon: {epsilon:.2f}")
        print(f"Highest Score: {self.stats['highest_score']}, Highest Coins: {self.stats['highest_coins']}")