# Improved game_control.py with swipe mechanics
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
    """Handles game control for Subway Surfers AI"""
    
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
        self.action_cooldown = 0.2  # Cooldown between actions
        
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
            if img_array.mean() > 10:  # If average pixel value is greater than 10, game is likely running
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
                
                # Wait for the game to load
                time.sleep(5)
                
                # Make sure the window is in focus
                self.focus_game_window()
                
                # Wait a bit more for the game to stabilize
                time.sleep(2)
                
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
        pyautogui.click(center_x, center_y)
        time.sleep(0.5)
        print(f"Focusing game window by clicking at ({center_x}, {center_y})")
    
    def detect_game_over(self):
        """Detect if the game is over using template matching or by finding the play button"""
        # Take a screenshot of the game region
        game_img = pyautogui.screenshot(region=GAME_REGION)
        game_img_np = np.array(game_img)
        game_img_cv = cv2.cvtColor(game_img_np, cv2.COLOR_RGB2BGR)
        
        # Method 1: Check for play button (usually visible on game over screen)
        play_button_loc = self.find_play_button()
        if play_button_loc is not None:
            print("Play button found - game is likely over")
            return True
            
        # Method 2: Check for specific colors or patterns on game over screen
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(game_img_cv, cv2.COLOR_BGR2HSV)
        
        # Check for green play button (common on game over screens)
        lower_green = np.array([40, 100, 100])
        upper_green = np.array([80, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # If we have a significant green area at the bottom of the screen (play button)
        bottom_half = green_mask[green_mask.shape[0]//2:, :]
        if np.sum(bottom_half) > 10000:  # Threshold can be adjusted
            print("Green play button detected - game is likely over")
            return True
        
        return False
    
    def find_play_button(self):
        """Find the play button using multiple detection methods"""
        # Take a screenshot of the game region
        game_img = pyautogui.screenshot(region=GAME_REGION)
        game_img_np = np.array(game_img)
        game_img_cv = cv2.cvtColor(game_img_np, cv2.COLOR_RGB2BGR)
        
        # Method 1: Template matching
        if self.play_button_template is not None:
            try:
                result = cv2.matchTemplate(game_img_cv, self.play_button_template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                
                if max_val > 0.5:  # Lower threshold for better detection
                    # Get the center of the template
                    h, w = self.play_button_template.shape[:2]
                    center_x = max_loc[0] + w // 2 + GAME_REGION[0]
                    center_y = max_loc[1] + h // 2 + GAME_REGION[1]
                    print(f"Play button found at ({center_x}, {center_y}) with confidence {max_val:.2f}")
                    return (center_x, center_y)
            except Exception as e:
                print(f"Error in template matching: {e}")
        
        # Method 2: Find green button using color detection
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(game_img_cv, cv2.COLOR_BGR2HSV)
            
            # Green color range for the play button
            lower_green = np.array([40, 100, 100])
            upper_green = np.array([80, 255, 255])
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
            
            # Find contours of green areas
            contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Analyze contours to find the play button
            if contours:
                # Sort contours by area (largest first)
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 1000:  # Min area threshold
                        # Get bounding box for the contour
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # Check if the aspect ratio is reasonable for a button
                        aspect_ratio = float(w)/h
                        if 1.5 < aspect_ratio < 5:  # Play button is usually wider than tall
                            # Get the center of the contour
                            center_x = x + w//2 + GAME_REGION[0]
                            center_y = y + h//2 + GAME_REGION[1]
                            
                            # If in the bottom part of the screen, it's likely the play button
                            if y > game_img_cv.shape[0] * 0.6:
                                print(f"Play button found using color detection at ({center_x}, {center_y})")
                                return (center_x, center_y)
        except Exception as e:
            print(f"Error in color detection: {e}")
        
        # Method 3: Use the specific Play button position from the screenshot
        try:
            # Check for the green play button in the bottom center
            center_x = GAME_REGION[0] + GAME_REGION[2] // 2
            center_y = GAME_REGION[1] + GAME_REGION[3] - 60  # Near bottom
            return (center_x, center_y)
        except Exception as e:
            print(f"Error in fallback detection: {e}")
            return None
    
    def restart_game(self, play_button_location=None):
        """Restart the game after game over"""
        if not play_button_location:
            # Try to find the play button
            play_button_location = self.find_play_button()
        
        # Click the play button
        if play_button_location:
            pyautogui.click(play_button_location[0], play_button_location[1])
            print(f"Clicked play button at {play_button_location}")
            time.sleep(1)  # Wait for the click to register
            
            # Increment total games counter
            self.stats['total_games'] += 1
            print(f"Total games played: {self.stats['total_games']}")
            
            self.game_running = True
            return True
        else:
            print("Failed to find play button")
            return False
    
    def perform_action(self, action):
        """Perform action using swipe gestures instead of keyboard controls"""
        # Check cooldown to prevent too many actions too quickly
        current_time = time.time()
        if current_time - self.last_action_time < self.action_cooldown:
            return "COOLDOWN"
        
        # Get action name
        action_name = ACTIONS.get(action, "NONE")
        
        # Get the center of the game region as the starting point for swipes
        center_x = GAME_REGION[0] + GAME_REGION[2] // 2
        center_y = GAME_REGION[1] + GAME_REGION[3] // 2
        
        # Calculate swipe distance based on game region size
        swipe_distance_x = GAME_REGION[2] // 4
        swipe_distance_y = GAME_REGION[3] // 4
        
        # Perform swipe action based on the action name
        if action_name == "LEFT":
            # Swipe from center to left
            pyautogui.moveTo(center_x, center_y)
            pyautogui.mouseDown(button='left')
            pyautogui.moveTo(center_x - swipe_distance_x, center_y, duration=0.1)
            pyautogui.mouseUp(button='left')
        elif action_name == "RIGHT":
            # Swipe from center to right
            pyautogui.moveTo(center_x, center_y)
            pyautogui.mouseDown(button='left')
            pyautogui.moveTo(center_x + swipe_distance_x, center_y, duration=0.1)
            pyautogui.mouseUp(button='left')
        elif action_name == "UP":
            # Swipe from center to up (jump)
            pyautogui.moveTo(center_x, center_y)
            pyautogui.mouseDown(button='left')
            pyautogui.moveTo(center_x, center_y - swipe_distance_y, duration=0.1)
            pyautogui.mouseUp(button='left')
        elif action_name == "DOWN":
            # Swipe from center to down (roll)
            pyautogui.moveTo(center_x, center_y)
            pyautogui.mouseDown(button='left')
            pyautogui.moveTo(center_x, center_y + swipe_distance_y, duration=0.1)
            pyautogui.mouseUp(button='left')
        # For NONE action, do nothing
        
        # Print the action for debugging
        if action_name != "NONE":
            print(f"Performed swipe action: {action_name}")
        
        self.last_action_time = current_time
        
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
        
        # Capture the game screen
        game_img = pyautogui.screenshot(region=GAME_REGION)
        frame = np.array(game_img)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Write the frame to the video
        self.video_writer.write(frame)
        return True
    
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
        
        # Display real-time stats
        print(f"\n{'=' * 40}")
        print(f"TRAINING STATS:")
        print(f"{'=' * 40}")
        print(f"Total Games: {self.stats['total_games']}")
        print(f"Total Steps: {self.stats['total_steps']}")
        print(f"Highest Score: {self.stats['highest_score']}")
        print(f"Highest Coins: {self.stats['highest_coins']}")
        print(f"Total Rewards: {self.stats['total_rewards']:.2f}")
        print(f"Current Epsilon: {self.stats['current_epsilon']:.4f}")
        print(f"Current Score: {score}")
        print(f"Current Coins: {coins}")
        print(f"Last Reward: {reward:.2f}")
        print(f"{'=' * 40}\n")