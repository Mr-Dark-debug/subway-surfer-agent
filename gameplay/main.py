import pyautogui
import time
import subprocess
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
import pytesseract
from PIL import Image
from collections import deque
from datetime import datetime
import re

# Set pytesseract path if not in system PATH (modify this to your Tesseract installation path)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust this path

# Hyperparameters
BATCH_SIZE = 32
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 20000
MEMORY_SIZE = 10000
LEARNING_RATE = 0.0001
TARGET_UPDATE = 10
CHECKPOINT_INTERVAL = 10
SCREENSHOT_INTERVAL = 50  # Save screenshot every 50 steps
MANUAL_RESTART_WAIT = 8  # Seconds to wait for manual restart

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def setup_game():
    """Set up the game environment and browser"""
    # Get screen size
    screen_width, screen_height = pyautogui.size()
    print(f"Screen size: {screen_width}x{screen_height}")
    
    # Define game regions (x, y, width, height)
    game_region = (1094, 178, 806, 529)
    score_region = (1682, 159, 225, 48)
    coin_region = (1682, 217, 225, 48)
    
    print("Game region:", game_region)
    print("Score region:", score_region)
    print("Coin region:", coin_region)
    
    # Create directories if they don't exist
    os.makedirs("screenshots", exist_ok=True)
    os.makedirs("screenshots/gameplay", exist_ok=True)
    os.makedirs("screenshots/score", exist_ok=True)
    os.makedirs("screenshots/coins", exist_ok=True)
    os.makedirs("screenshots/game_over", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Copy template assets if provided
    copy_template_assets()
    
    # Open Subway Surfers on Poki in Chrome incognito mode
    url = "https://poki.com/en/g/subway-surfers"
    open_browser_incognito(url)
    
    # Wait for browser to load
    print("Waiting for browser to load...")
    time.sleep(5)
    
    # Move browser to right side of screen (split-screen)
    position_window_right_side(screen_width, screen_height)
    
    # Take initial screenshots
    print("Taking initial screenshots...")
    game_img = pyautogui.screenshot(region=game_region)
    game_img.save("screenshots/gameplay/initial_game.png")
    
    score_img = pyautogui.screenshot(region=score_region)
    score_img.save("screenshots/score/initial_score.png")
    
    coin_img = pyautogui.screenshot(region=coin_region)
    coin_img.save("screenshots/coins/initial_coins.png")
    
    return game_region, score_region, coin_region

def copy_template_assets():
    """Copy template assets to templates directory if they exist"""
    assets_dir = "./assets"
    templates_dir = "./templates"
    
    if os.path.exists(assets_dir):
        from shutil import copy2
        
        # Create templates directory if it doesn't exist
        os.makedirs(templates_dir, exist_ok=True)
        
        # List of assets to copy
        assets = [
            "save_me.png",
            "play_button.png",
            "score_screen.png",
            "free_button.png"
        ]
        
        for asset in assets:
            src_path = os.path.join(assets_dir, asset)
            dst_path = os.path.join(templates_dir, asset)
            
            if os.path.exists(src_path) and not os.path.exists(dst_path):
                try:
                    copy2(src_path, dst_path)
                    print(f"Copied template asset: {asset}")
                except Exception as e:
                    print(f"Failed to copy {asset}: {e}")

def open_browser_incognito(url):
    """Open the URL in Chrome's incognito mode"""
    # Try to use Chrome first (most common)
    try:
        chrome_path = 'C:/Program Files/Google/Chrome/Application/chrome.exe'
        if not os.path.exists(chrome_path):
            chrome_path = 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe'
        
        # Launch Chrome in incognito mode with the URL
        subprocess.Popen([chrome_path, '--incognito', url])
        print(f"Opening {url} in Chrome incognito mode")
        return
    except Exception as e:
        print(f"Couldn't open Chrome: {e}")
    
    # Fallback to Edge if Chrome isn't available
    try:
        edge_path = 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'
        subprocess.Popen([edge_path, '-inprivate', url])
        print(f"Opening {url} in Edge InPrivate mode")
        return
    except Exception as e:
        print(f"Couldn't open Edge: {e}")
        
    # Final fallback to system default browser (may not be incognito)
    print("Using default browser (incognito mode not guaranteed)")
    import webbrowser
    webbrowser.open(url)

def position_window_right_side(screen_width, screen_height):
    """Position the active window to the right side of the screen"""
    # First make sure we're focused on the browser window
    pyautogui.click(screen_width // 2, screen_height // 2)
    time.sleep(0.5)
    
    # Windows snap feature (Win+Right arrow)
    pyautogui.hotkey('winleft', 'right')
    
    # Give the window a moment to reposition
    time.sleep(1)
    print("Browser positioned on the right side of the screen")

# DQN Model Definition
class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        # Convolutional layers to process game frames
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Calculate the size of feature maps after convolutions
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        
        # Fully connected layer for action values
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

# Experience Replay Memory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        
    def push(self, state, action, next_state, reward, done):
        self.memory.append((state, action, next_state, reward, done))
        
    def sample(self, batch_size):
        if len(self.memory) < batch_size:
            return None
        transitions = random.sample(self.memory, batch_size)
        state, action, next_state, reward, done = zip(*transitions)
        return state, action, next_state, reward, done
        
    def __len__(self):
        return len(self.memory)

# Preprocess frame for model input
def preprocess_frame(frame):
    """Process game frame for input to neural network"""
    # Convert to numpy array if it's a PIL Image
    if not isinstance(frame, np.ndarray):
        frame_array = np.array(frame)
    else:
        frame_array = frame
    
    # Convert to grayscale
    if len(frame_array.shape) == 3 and frame_array.shape[2] == 3:
        gray = cv2.cvtColor(frame_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = frame_array
    
    # Apply threshold to highlight obstacles and character
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    
    # Resize to a smaller dimension for processing efficiency
    resized = cv2.resize(thresh, (84, 84), interpolation=cv2.INTER_AREA)
    
    # Normalize pixel values
    normalized = resized / 255.0
    
    # Add channel dimension and batch dimension for PyTorch (batch_size, channels, height, width)
    tensor = torch.FloatTensor(normalized).unsqueeze(0).unsqueeze(0)
    
    return tensor.to(device)

# Improved Game State Tracker with OCR
class GameStateTracker:
    def __init__(self, log_file_path=None):
        self.last_score = 0
        self.last_coins = 0
        self.total_score = 0
        self.total_coins = 0
        self.frames_without_change = 0
        self.max_frames_without_change = 15
        self.steps = 0
        
        # Create log file for OCR results
        self.log_file = None
        if log_file_path:
            try:
                self.log_file = open(log_file_path, 'w')
                self.log_file.write("Timestamp,Step,Type,Raw_Text,Parsed_Value,Filtered_Value\n")
            except Exception as e:
                print(f"Error creating OCR log file: {e}")
        
        # Store template images
        self.templates = {
            'save_me': None,
            'score_screen': None,
            'play_button': None,
            'free_button': None
        }
        
        self.try_load_templates()
        
    def try_load_templates(self):
        """Load template images for game state detection"""
        templates_dir = "templates"
        for template_name in self.templates.keys():
            template_path = os.path.join(templates_dir, f"{template_name}.png")
            if os.path.exists(template_path):
                self.templates[template_name] = cv2.imread(template_path, cv2.IMREAD_COLOR)
                print(f"Loaded '{template_name}' template for detection")
    
    def save_template(self, img, name):
        """Save a detected screen as template for future detection"""
        if not isinstance(img, np.ndarray):
            img_array = np.array(img)
        else:
            img_array = img
            
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            # Already in correct format
            pass
        else:
            # Convert grayscale to RGB for template storage
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            
        template_path = os.path.join("templates", f"{name}.png")
        cv2.imwrite(template_path, img_array)
        print(f"Saved new template: {name}.png")
        
        # Update the loaded template
        self.templates[name] = img_array
    
    def preprocess_for_ocr(self, img, is_score=True):
        """Preprocess image for OCR to improve text recognition"""
        # Convert to numpy array if needed
        if not isinstance(img, np.ndarray):
            img_array = np.array(img)
        else:
            img_array = img.copy()
        
        # Save original for debugging
        if is_score:
            category = "score"
        else:
            category = "coins"
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        img_path = f"screenshots/{category}/raw_{self.steps}_{timestamp}.png"
        
        if isinstance(img, Image.Image):
            img.save(img_path)
        else:
            cv2.imwrite(img_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
        
        # Apply preprocessing techniques to enhance text visibility
        
        # 1. Convert to grayscale
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array.copy()
        
        # 2. Apply thresholding to enhance text
        # Try different thresholds for better OCR performance
        if is_score:
            # For score which is often white text
            _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        else:
            # For coins which might be yellow/golden
            # First try to isolate yellow using HSV if it's a color image
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
                # Yellow-gold range in HSV
                lower_gold = np.array([20, 100, 100])
                upper_gold = np.array([40, 255, 255])
                mask = cv2.inRange(hsv, lower_gold, upper_gold)
                # Combine with grayscale threshold
                _, binary_gray = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
                binary = cv2.bitwise_or(mask, binary_gray)
            else:
                # If already grayscale
                _, binary = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
        
        # 3. Apply noise reduction
        binary = cv2.medianBlur(binary, 3)
        
        # 4. Apply dilation to make text more prominent
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        
        # 5. Invert if necessary for better OCR (white text on black background)
        if np.mean(dilated) > 127:
            dilated = cv2.bitwise_not(dilated)
        
        # Save processed image for debugging
        processed_img_path = f"screenshots/{category}/processed_{self.steps}_{timestamp}.png"
        cv2.imwrite(processed_img_path, dilated)
        
        return dilated
    
    def extract_score_ocr(self, score_img):
        """Extract score from screenshot using OCR"""
        # Preprocess image for better OCR performance
        processed_img = self.preprocess_for_ocr(score_img, is_score=True)
        
        # Apply OCR
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Configure tesseract for digits only, with specific configuration
        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
        ocr_result = pytesseract.image_to_string(processed_img, config=custom_config).strip()
        
        # Log OCR results
        self.log_ocr_result("score", ocr_result)
        
        # Process OCR result
        # Extract only digits
        digits_only = re.sub(r'\D', '', ocr_result)
        
        # Convert to integer, default to last score if conversion fails
        try:
            score = int(digits_only) if digits_only else self.last_score
        except ValueError:
            score = self.last_score
            
        # Apply consistency check - prevent unrealistic jumps
        if self.last_score > 0:
            # Score typically increases gradually, big jumps are likely errors
            max_increase = 500  # Maximum reasonable score increase between frames
            
            if score > self.last_score + max_increase:
                # Probably a misread, use a reasonable increment
                score = self.last_score + 10
            elif score < self.last_score:
                # Scores shouldn't decrease during gameplay
                score = self.last_score
        
        return score
    
    def extract_coins_ocr(self, coin_img):
        """Extract coin count from screenshot using OCR"""
        # Preprocess image for better OCR performance
        processed_img = self.preprocess_for_ocr(coin_img, is_score=False)
        
        # Apply OCR
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Configure tesseract for digits only
        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
        ocr_result = pytesseract.image_to_string(processed_img, config=custom_config).strip()
        
        # Log OCR results
        self.log_ocr_result("coins", ocr_result)
        
        # Process OCR result
        # Extract only digits
        digits_only = re.sub(r'\D', '', ocr_result)
        
        # Convert to integer, default to last coins if conversion fails
        try:
            coins = int(digits_only) if digits_only else self.last_coins
        except ValueError:
            coins = self.last_coins
            
        # Apply consistency check - prevent unrealistic jumps
        if self.last_coins > 0:
            # Coins typically increase in small amounts
            max_increase = 50  # Maximum reasonable coin increase between frames
            
            if coins > self.last_coins + max_increase:
                # Probably a misread, use a small increment
                coins = self.last_coins + 5
            elif coins < self.last_coins:
                # Coins shouldn't decrease during normal gameplay
                coins = self.last_coins
        
        return coins
    
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
    
    def detect_save_me_screen(self, game_img):
        """
        Detect the 'Save me!' screen shown after dying
        Uses a combination of template matching and color analysis
        """
        # Save the current game image to check later
        if isinstance(game_img, Image.Image):
            img_array = np.array(game_img)
        else:
            img_array = game_img.copy()
        
        # Save timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Method 1: Template matching if template is available
        if self.templates['save_me'] is not None:
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
                
            result = cv2.matchTemplate(img_bgr, self.templates['save_me'], cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if max_val > 0.6:  # Lower threshold for better detection
                print("'Save me!' screen detected (template matching)")
                
                # Save the image for debugging
                cv2.imwrite(f"screenshots/game_over/save_me_template_{timestamp}_{self.steps}.png", img_bgr)
                return True
        
        # Method 2: Look for the blue "Save me!" text and green "Free!" button
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            
            # Blue color range (for "Save me!" text)
            lower_blue = np.array([100, 100, 100])
            upper_blue = np.array([130, 255, 255])
            blue_mask = cv2.inRange(img_hsv, lower_blue, upper_blue)
            
            # Green color range (for "Free!" button)
            lower_green = np.array([40, 100, 100])
            upper_green = np.array([80, 255, 255])
            green_mask = cv2.inRange(img_hsv, lower_green, upper_green)
            
            # Check for blue and green elements in expected regions
            center_region_blue = blue_mask[150:250, 300:500]
            center_region_green = green_mask[250:350, 300:500]
            
            blue_pixels = np.sum(center_region_blue) / 255
            green_pixels = np.sum(center_region_green) / 255
            
            if blue_pixels > 200 and green_pixels > 200:
                print("'Save me!' screen detected (color analysis)")
                
                # Save the image and masks for debugging
                cv2.imwrite(f"screenshots/game_over/save_me_color_{timestamp}_{self.steps}.png", 
                            cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
                cv2.imwrite(f"screenshots/game_over/save_me_blue_mask_{timestamp}_{self.steps}.png", blue_mask)
                cv2.imwrite(f"screenshots/game_over/save_me_green_mask_{timestamp}_{self.steps}.png", green_mask)
                
                # Save this as a template if we don't have one
                if self.templates['save_me'] is None:
                    self.save_template(img_array, 'save_me')
                return True
        
        return False
    
    def detect_score_screen(self, game_img):
        """
        Detect the score/leaderboard screen
        Uses template matching and feature detection
        """
        # Save the current game image to check later
        if isinstance(game_img, Image.Image):
            img_array = np.array(game_img)
        else:
            img_array = game_img.copy()
        
        # Save timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Method 1: Template matching if template is available
        if self.templates['score_screen'] is not None:
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
                
            result = cv2.matchTemplate(img_bgr, self.templates['score_screen'], cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if max_val > 0.6:
                print("Score screen detected (template matching)")
                
                # Save the image for debugging
                cv2.imwrite(f"screenshots/game_over/score_screen_template_{timestamp}_{self.steps}.png", img_bgr)
                return True
        
        # Method 2: Look for visual features of score screen
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            
            # Look for green "PLAY" button
            lower_green = np.array([40, 100, 100])
            upper_green = np.array([80, 255, 255])
            green_mask = cv2.inRange(img_hsv, lower_green, upper_green)
            
            # Look for blue "Score" text
            lower_blue = np.array([100, 100, 100])
            upper_blue = np.array([130, 255, 255])
            blue_mask = cv2.inRange(img_hsv, lower_blue, upper_blue)
            
            # Check bottom area for green button
            bottom_green = green_mask[400:500, 400:600]
            green_pixels = np.sum(bottom_green) / 255
            
            # Check top area for blue score text
            top_blue = blue_mask[50:150, 300:500]
            blue_pixels = np.sum(top_blue) / 255
            
            if green_pixels > 500 and blue_pixels > 200:
                print("Score screen detected (color analysis)")
                
                # Save the image and masks for debugging
                cv2.imwrite(f"screenshots/game_over/score_screen_color_{timestamp}_{self.steps}.png", 
                            cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
                cv2.imwrite(f"screenshots/game_over/score_screen_blue_mask_{timestamp}_{self.steps}.png", blue_mask)
                cv2.imwrite(f"screenshots/game_over/score_screen_green_mask_{timestamp}_{self.steps}.png", green_mask)
                
                # Save as template if we don't have one
                if self.templates['score_screen'] is None:
                    self.save_template(img_array, 'score_screen')
                return True
        
        return False
    
    def wait_for_manual_restart(self, game_region):
        """
        Instead of automatically handling game over screens, 
        wait for the user to manually restart the game
        """
        # Click in the game area to make sure it has focus
        region_center_x = game_region[0] + game_region[2] // 2
        region_center_y = game_region[1] + game_region[3] // 2
        
        print("Waiting for manual restart...")
        print("Please restart the game manually now!")
        
        # Focus on the game window
        pyautogui.click(region_center_x, region_center_y)
        
        # Wait for user to handle the restart
        time.sleep(MANUAL_RESTART_WAIT)
        
        # Wait until we detect that we're back in the playing state
        max_wait_time = 30  # Maximum seconds to wait
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            # Take screenshot of game region
            game_img = pyautogui.screenshot(region=game_region)
            
            # Check if we're back in the game
            if not self.detect_save_me_screen(game_img) and not self.detect_score_screen(game_img):
                print("Game seems to be restarted. Continuing training...")
                time.sleep(1)  # Brief pause to ensure the game is fully loaded
                return True
            
            # Wait a bit before checking again
            time.sleep(1)
            
        print("Warning: Game might not be properly restarted. Continuing anyway...")
        return False
    
    def detect_game_state(self, game_img):
        """
        Detect the current game state: playing, save_me, or score_screen
        Now modified to always return 'playing' to ignore game over screens.
        """
        # Save gameplay screenshot periodically
        # if self.steps % 10 == 0 or self.steps < 5:  # More frequent at start
        #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #     if isinstance(game_img, Image.Image):
        #         game_img.save(f"screenshots/gameplay/game_state_{self.steps}_{timestamp}.png")
        #     else:
        #         cv2.imwrite(f"screenshots/gameplay/game_state_{self.steps}_{timestamp}.png", 
        #                    cv2.cvtColor(np.array(game_img), cv2.COLOR_RGB2BGR))
        
        # Always return playing to ignore game over states for continuous training
        return "playing"
    
        # Original logic (commented out):
        # Check for specific game states in order of priority
        # if self.detect_save_me_screen(game_img):
        #     return "save_me"
        # 
        # if self.detect_score_screen(game_img):
        #     return "score_screen"
        # 
        # # If no special screen is detected, assume we're playing
        # return "playing"
    
    def handle_game_state(self, state, game_region):
        """
        Modified to wait for manual restart instead of handling automatically
        """
        if state != "playing":
            # Take screenshot for debugging
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            game_img = pyautogui.screenshot(region=game_region)
            game_img.save(f"screenshots/game_over/game_over_{state}_{self.steps}_{timestamp}.png")
            
            # Wait for manual restart
            return self.wait_for_manual_restart(game_region)
            
        return False  # No handling needed for "playing" state
    
    def calculate_reward(self, score_img, coin_img, game_img):
        """
        Calculate reward based on score, coins, and game state
        Returns: reward, is_done, game_state
        """
        self.steps += 1
        
        # Save screenshots periodically
        if self.steps % SCREENSHOT_INTERVAL == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Save full gameplay image
            if isinstance(game_img, Image.Image):
                game_img.save(f"screenshots/gameplay/frame_{self.steps}_{timestamp}.png")
            else:
                cv2.imwrite(f"screenshots/gameplay/frame_{self.steps}_{timestamp}.png", 
                           cv2.cvtColor(np.array(game_img), cv2.COLOR_RGB2BGR))
        
        # Extract game information using OCR
        current_score = self.extract_score_ocr(score_img)
        current_coins = self.extract_coins_ocr(coin_img)
        
        # Check for game state
        game_state = self.detect_game_state(game_img)
        is_done = (game_state != "playing")
        
        # Calculate reward
        reward = 0
        
        # Check if state has changed significantly
        if current_score == self.last_score and current_coins == self.last_coins:
            self.frames_without_change += 1
        else:
            self.frames_without_change = 0
        
        if game_state != "playing":
            # Small penalty for dying (not too large to avoid discouraging exploration)
            reward = -5.0
            print(f"Game over detected! Type: {game_state}")
        else:
            # Small reward for surviving
            reward += 0.1
            
            # Reward for score increase
            if current_score > self.last_score:
                score_diff = current_score - self.last_score
                reward += score_diff * 0.05
                print(f"Score increased by {score_diff}! New score: {current_score}")
                self.total_score += score_diff
            
            # Reward for collecting coins
            if current_coins > self.last_coins:
                coin_diff = current_coins - self.last_coins
                reward += coin_diff * 0.5
                print(f"Collected {coin_diff} coins! Total coins: {current_coins}")
                self.total_coins += coin_diff
        
        # Update state tracking
        self.last_score = current_score
        self.last_coins = current_coins
        
        return reward, is_done, game_state
    
    def reset(self):
        """Reset the game state tracker"""
        self.last_score = 0
        self.last_coins = 0
        self.frames_without_change = 0
    
    def close(self):
        """Close resources like log files"""
        if self.log_file:
            self.log_file.close()
            self.log_file = None

# Agent Implementation
class SubwaySurfersAgent:
    def __init__(self, h, w, n_actions, load_checkpoint=None):
        self.epsilon = EPSILON_START
        self.n_actions = n_actions
        self.steps_done = 0
        self.episode_count = 0
        
        # Initialize Q networks (policy and target)
        self.policy_net = DQN(h, w, n_actions).to(device)
        self.target_net = DQN(h, w, n_actions).to(device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        
        # Load checkpoint if specified
        if load_checkpoint and os.path.exists(load_checkpoint):
            self.load_checkpoint(load_checkpoint)
        else:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print("Initialized new model (no checkpoint loaded)")
        
        self.target_net.eval()
        
        # Initialize replay memory
        self.memory = ReplayMemory(MEMORY_SIZE)
        
        # Action mapping
        self.actions = {
            0: 'left',   # Left arrow key
            1: 'up',     # Up arrow key (jump)
            2: 'right',  # Right arrow key
            3: 'down'    # Down arrow key (roll)
        }
        
        # Training stats
        self.total_rewards = []
        self.episode_steps = []
        self.average_losses = []
        
        print(f"Agent initialized with {n_actions} possible actions: {list(self.actions.values())}")
    
    def check_for_checkpoints(self):
        """Look for and return the most recent checkpoint file"""
        checkpoints = []
        
        # Check for interrupted checkpoint
        if os.path.exists("models/subway_surfers_interrupted.pth"):
            return "models/subway_surfers_interrupted.pth"
        
        # Look for episode checkpoints
        for filename in os.listdir("models"):
            if filename.startswith("subway_surfers_dqn_episode_") and filename.endswith(".pth"):
                try:
                    episode_num = int(filename.split("_")[-1].split(".")[0])
                    checkpoints.append((episode_num, os.path.join("models", filename)))
                except:
                    continue
        
        if checkpoints:
            # Sort by episode number and get the latest
            latest = sorted(checkpoints, key=lambda x: x[0], reverse=True)[0]
            return latest[1]
        
        return None
    
    def load_checkpoint(self, checkpoint_path):
        """Load a saved model checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.policy_net.load_state_dict(checkpoint['model_state_dict'])
                self.target_net.load_state_dict(checkpoint['model_state_dict'])
                
                if 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                self.epsilon = checkpoint.get('epsilon', EPSILON_START)
                self.steps_done = checkpoint.get('steps_done', 0)
                self.episode_count = checkpoint.get('episode', 0)
                
                print(f"Loaded checkpoint from {checkpoint_path}")
                print(f"Resuming from episode {self.episode_count}, steps {self.steps_done}, epsilon {self.epsilon:.4f}")
                
                return self.episode_count
            else:
                # Simple state dict format
                self.policy_net.load_state_dict(checkpoint)
                self.target_net.load_state_dict(checkpoint)
                print(f"Loaded model state from {checkpoint_path}")
                return 0
                
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting with new model")
            return 0
        
    def select_action(self, state):
        """Select action using epsilon-greedy policy"""
        # Random number for exploration decision
        sample = random.random()
        
        # Update epsilon with decay
        eps_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                        math.exp(-1. * self.steps_done / EPSILON_DECAY)
        self.epsilon = eps_threshold
        self.steps_done += 1
        
        if sample > eps_threshold:
            # Exploit: use policy network
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            # Explore: random action
            return torch.tensor([[random.randrange(self.n_actions)]], 
                                device=device, dtype=torch.long)
    
    def execute_action(self, action_idx):
        """Execute the selected action using pyautogui"""
        action = self.actions[action_idx.item()]
        
        # Press the corresponding key
        pyautogui.press(action)
        
        # Short delay for the action to take effect
        time.sleep(0.05)
    
    def save_checkpoint(self, episode, total_reward, avg_loss=None, path=None):
        """Save the current model as a checkpoint"""
        if path is None:
            # Save as episode checkpoint
            path = f"models/subway_surfers_dqn_episode_{episode}.pth"
            
        os.makedirs("models", exist_ok=True)
        
        torch.save({
            'episode': episode,
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'reward': total_reward,
            'loss': avg_loss,
            'epsilon': self.epsilon
        }, path)
        
        print(f"Model checkpoint saved to {path}")
    
    def optimize_model(self):
        """Perform one step of optimization on the model"""
        batch = self.memory.sample(BATCH_SIZE)
        if batch is None:
            return None
        
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = batch
        
        # Convert to tensors
        state_batch = torch.cat(state_batch)
        action_batch = torch.tensor(action_batch, device=device).unsqueeze(1)
        reward_batch = torch.tensor(reward_batch, device=device, dtype=torch.float32)
        
        # Create mask for non-final states
        non_final_mask = torch.tensor(tuple(map(lambda d: not d, done_batch)), 
                                    device=device, dtype=torch.bool)
        
        # Prepare next states, filtering out terminal states
        non_final_next_states = [s for s, d in zip(next_state_batch, done_batch) if not d]
        if non_final_next_states:
            non_final_next_states = torch.cat(non_final_next_states)
        
        # Get current Q values
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Get next state values using target network
        next_state_values = torch.zeros(len(batch[0]), device=device)
        if len(non_final_next_states) > 0:
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        
        # Compute expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        
        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        return loss.item()

def train_agent(agent, game_region, score_region, coin_region, num_episodes=100, start_episode=1):
    """
    Train the agent for a specified number of episodes
    This is an improved training loop focused on continuous play
    """
    # Initialize game state tracker with logging
    log_path = f"logs/ocr_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    game_tracker = GameStateTracker(log_path)
    
    # Set up logging
    os.makedirs("logs", exist_ok=True)
    log_file = open(f"logs/training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "w")
    
    print(f"\nStarting training from episode {start_episode} to {start_episode + num_episodes - 1}...\n")
    
    try:
        for episode in range(start_episode, start_episode + num_episodes):
            print(f"\n--- Episode {episode}/{start_episode + num_episodes - 1} ---")
            
            # Reset variables for this episode
            game_tracker.reset()
            total_reward = 0
            total_loss = 0
            loss_count = 0
            steps = 0
            
            # Restart the game
            print("Restarting game...")
            restart_game(game_region)
            
            # Take screenshot at beginning of episode
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            episode_start_img = pyautogui.screenshot(region=game_region)
            episode_start_img.save(f"screenshots/gameplay/episode_{episode}_start_{timestamp}.png")
            
            # Initial state
            print("Capturing initial state...")
            game_img = pyautogui.screenshot(region=game_region)
            score_img = pyautogui.screenshot(region=score_region)
            coin_img = pyautogui.screenshot(region=coin_region)
            
            # Check if we start in a non-playing state
            game_state = game_tracker.detect_game_state(game_img)
            if game_state != "playing":
                print(f"Game over screen detected at start of episode! Type: {game_state}")
                game_tracker.handle_game_state(game_state, game_region)
                # Re-capture initial state
                game_img = pyautogui.screenshot(region=game_region)
                score_img = pyautogui.screenshot(region=score_region)
                coin_img = pyautogui.screenshot(region=coin_region)
                # Check again
                game_state = game_tracker.detect_game_state(game_img)
                if game_state != "playing":
                    # Skip this episode if we can't get to a playing state
                    print(f"Unable to start game properly. Skipping episode {episode}.")
                    continue
            
            # Process the initial state
            state = preprocess_frame(game_img)
            
            # Main episode loop
            done = False
            while not done:
                # Select and perform an action
                action = agent.select_action(state)
                agent.execute_action(action)
                
                # Short delay for game to respond
                time.sleep(0.1)
                
                # Observe new state
                next_game_img = pyautogui.screenshot(region=game_region)
                next_score_img = pyautogui.screenshot(region=score_region)
                next_coin_img = pyautogui.screenshot(region=coin_region)
                
                # Process reward and next state
                reward, done, game_state = game_tracker.calculate_reward(
                    next_score_img, next_coin_img, next_game_img)
                
                # Add to total reward
                total_reward += reward
                
                # Process next state
                next_state = preprocess_frame(next_game_img)
                
                # Store the transition in memory
                agent.memory.push(state, action.item(), next_state, reward, done)
                
                # Move to the next state
                state = next_state
                
                # Perform optimization step
                loss = agent.optimize_model()
                if loss is not None:
                    total_loss += loss
                    loss_count += 1
                
                # Update step counter
                steps += 1
                
                # Print status occasionally
                if steps % 10 == 0:
                    print(f"Step: {steps}, Epsilon: {agent.epsilon:.4f}, Reward: {total_reward:.2f}")
                
                # Handle game over
                if done:
                    # Take screenshot at game over
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    episode_end_img = pyautogui.screenshot(region=game_region)
                    episode_end_img.save(f"screenshots/gameplay/episode_{episode}_end_{timestamp}.png")
                    
                    print(f"Game over detected (type: {game_state}), waiting for manual restart...")
                    game_tracker.handle_game_state(game_state, game_region)
            
            # Update the target network
            if episode % TARGET_UPDATE == 0:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())
                print(f"Target network updated at episode {episode}")
            
            # Calculate average loss
            avg_loss = total_loss / loss_count if loss_count > 0 else 0.0
            
            # Log episode results
            log_message = f"Episode {episode} complete - Steps: {steps}, Total Reward: {total_reward:.2f}, Avg Loss: {avg_loss:.4f}"
            print(log_message)
            log_file.write(log_message + "\n")
            
            # Save episode stats
            agent.total_rewards.append(total_reward)
            agent.episode_steps.append(steps)
            agent.average_losses.append(avg_loss)
            agent.episode_count = episode
            
            # Save model checkpoint periodically
            if episode % CHECKPOINT_INTERVAL == 0:
                agent.save_checkpoint(episode, total_reward, avg_loss)
                # Generate and save progress plots
                plot_training_progress(agent.total_rewards, agent.episode_steps, agent.average_losses, episode)
    
    except KeyboardInterrupt:
        print("\nTraining stopped by user")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Save final model state
        print("Saving current model state...")
        agent.save_checkpoint(agent.episode_count, total_reward if 'total_reward' in locals() else 0, 
                            avg_loss if 'avg_loss' in locals() else 0,
                            path="models/subway_surfers_interrupted.pth")
        
        # Close log files
        log_file.close()
        game_tracker.close()
        
        return agent.episode_count

def continuous_play_loop(agent, game_region, score_region, coin_region):
    """
    Run the game continuously, ignoring episodes, 
    focusing on maximizing score and learning
    """
    # Initialize game state tracker with logging
    log_path = f"logs/ocr_log_continuous_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    game_tracker = GameStateTracker(log_path)
    
    # Set up logging
    os.makedirs("logs", exist_ok=True)
    log_file = open(f"logs/continuous_play_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "w")
    
    print("\nStarting continuous gameplay mode...\n")
    
    # Variables for tracking performance
    total_steps = 0
    total_reward = 0
    total_loss = 0
    loss_count = 0
    last_save_time = time.time()
    last_update_time = time.time()
    save_interval = 300  # Save every 5 minutes
    update_interval = 300  # Update target network every 5 minutes
    
    try:
        # Make sure the game is started
        restart_game(game_region)
        time.sleep(1)
        
        # Initial state capture
        game_img = pyautogui.screenshot(region=game_region)
        score_img = pyautogui.screenshot(region=score_region)
        coin_img = pyautogui.screenshot(region=coin_region)
        
        # Save initial state
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        game_img.save(f"screenshots/gameplay/continuous_start_{timestamp}.png")
        
        # Handle if game is not in playing state
        game_state = game_tracker.detect_game_state(game_img)
        if game_state != "playing":
            game_tracker.handle_game_state(game_state, game_region)
            time.sleep(1)
            # Re-capture initial state
            game_img = pyautogui.screenshot(region=game_region)
            score_img = pyautogui.screenshot(region=score_region)
            coin_img = pyautogui.screenshot(region=coin_region)
        
        # Process initial state
        state = preprocess_frame(game_img)
        
        print("Continuous play started! Press Ctrl+C to stop.")
        
        # Main continuous loop
        while True:
            # Select and perform action
            action = agent.select_action(state)
            agent.execute_action(action)
            
            # Short delay for game to respond
            time.sleep(0.1)
            
            # Observe new state
            next_game_img = pyautogui.screenshot(region=game_region)
            next_score_img = pyautogui.screenshot(region=score_region)
            next_coin_img = pyautogui.screenshot(region=coin_region)
            
            # Process reward and check game state
            reward, done, game_state = game_tracker.calculate_reward(
                next_score_img, next_coin_img, next_game_img)
            
            # Add to total reward
            total_reward += reward
            
            # Process next state
            next_state = preprocess_frame(next_game_img)
            
            # Store the transition in memory
            agent.memory.push(state, action.item(), next_state, reward, done)
            
            # Move to the next state
            state = next_state
            
            # Perform optimization step
            loss = agent.optimize_model()
            if loss is not None:
                total_loss += loss
                loss_count += 1
            
            # Update step counter
            total_steps += 1
            agent.steps_done += 1
            
            # Print status occasionally
            if total_steps % 100 == 0:
                avg_loss = total_loss / loss_count if loss_count > 0 else 0
                print(f"Steps: {total_steps}, Epsilon: {agent.epsilon:.4f}, Score: {game_tracker.total_score}, Coins: {game_tracker.total_coins}, Avg Loss: {avg_loss:.4f}")
                log_file.write(f"Steps: {total_steps}, Reward: {total_reward:.2f}, Avg Loss: {avg_loss:.4f}\n")
                
                # Reset counters
                total_loss = 0
                loss_count = 0
            
            # Handle game over by waiting for manual restart
            if done:
                # Save screenshot at game over
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                if isinstance(next_game_img, Image.Image):
                    next_game_img.save(f"screenshots/game_over/game_over_{total_steps}_{timestamp}.png")
                else:
                    cv2.imwrite(f"screenshots/game_over/game_over_{total_steps}_{timestamp}.png", 
                               cv2.cvtColor(np.array(next_game_img), cv2.COLOR_RGB2BGR))
                
                print(f"Game over detected (type: {game_state}), waiting for manual restart...")
                game_tracker.handle_game_state(game_state, game_region)
                
                # Wait a moment and re-capture state
                time.sleep(1)
                game_img = pyautogui.screenshot(region=game_region)
                score_img = pyautogui.screenshot(region=score_region)
                coin_img = pyautogui.screenshot(region=coin_region)
                
                # Reset state
                state = preprocess_frame(game_img)
            
            # Save model periodically
            current_time = time.time()
            if current_time - last_save_time > save_interval:
                print("Saving periodic checkpoint...")
                agent.save_checkpoint(agent.episode_count, total_reward, 
                                    path=f"models/subway_surfers_continuous_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
                last_save_time = current_time
            
            # Update target network periodically
            if current_time - last_update_time > update_interval:
                print("Updating target network...")
                agent.target_net.load_state_dict(agent.policy_net.state_dict())
                last_update_time = current_time
    
    except KeyboardInterrupt:
        print("\nContinuous play stopped by user")
        
    except Exception as e:
        print(f"Error during continuous play: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Save final model state
        print("Saving final model state...")
        agent.save_checkpoint(agent.episode_count, total_reward,
                            path="models/subway_surfers_continuous_final.pth")
        
        # Close log files
        log_file.close()
        game_tracker.close()

def restart_game(game_region):
    """Focus on the game area for manual restart"""
    # Click in the game region to ensure focus
    region_center_x = game_region[0] + game_region[2] // 2
    region_center_y = game_region[1] + game_region[3] // 2
    pyautogui.click(region_center_x, region_center_y)
    time.sleep(0.5)
    
    # No automatic space press - wait for manual restart
    print("Game window focused - please start/restart the game manually if needed")
    time.sleep(2)

def plot_training_progress(rewards, lengths, losses, episode):
    """Plot and save training progress metrics"""
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(15, 5))
        
        # Plot rewards
        plt.subplot(1, 3, 1)
        plt.plot(rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        
        # Plot episode lengths
        plt.subplot(1, 3, 2)
        plt.plot(lengths)
        plt.title('Episode Lengths (Steps)')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        
        # Plot losses
        plt.subplot(1, 3, 3)
        plt.plot(losses)
        plt.title('Average Loss per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        
        plt.tight_layout()
        
        # Create directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # Save plot
        plt.savefig(f"logs/training_progress_episode_{episode}.png")
        plt.close()
        
        print(f"Training progress plot saved to logs/training_progress_episode_{episode}.png")
    except Exception as e:
        print(f"Error creating training progress plot: {e}")

def cleanup():
    """Clean up resources before exiting"""
    # Close any open browser windows (only if needed)
    # This is optional and depends on your preferences
    try:
        import webbrowser
        webbrowser.close()
    except:
        pass
    
    print("Cleanup completed.")

def main():
    # Give user time to prepare
    print("Starting Subway Surfers automation in 3 seconds...")
    time.sleep(3)
    
    # Setup the game and get regions
    game_region, score_region, coin_region = setup_game()
    
    print("\nGame is ready! Setting up DQN training...")
    print("NOTE: You will need to manually restart the game when it shows the 'Save me!' screen")
    
    # Create agent (4 actions: left, up, right, down)
    h, w = 84, 84  # Resize frames to this size for processing
    n_actions = 4
    
    # Create agent
    agent = SubwaySurfersAgent(h, w, n_actions)
    
    # Check for existing checkpoints
    checkpoint_path = agent.check_for_checkpoints()
    start_episode = 1
    
    if checkpoint_path:
        print(f"Found checkpoint: {checkpoint_path}")
        use_checkpoint = input("Do you want to continue training from this checkpoint? (y/n): ").strip().lower()
        
        if use_checkpoint in ['y', 'yes']:
            start_episode = agent.load_checkpoint(checkpoint_path) + 1
        else:
            print("Starting fresh training with new model")
    else:
        print("No existing checkpoints found. Starting fresh training.")
    
    # Ask user for training mode
    print("\nTraining modes:")
    print("1. Episode-based training (trains for a fixed number of episodes)")
    print("2. Continuous play (keeps playing and learning indefinitely)")
    
    mode = input("Select training mode (1/2): ").strip()
    
    try:
        if mode == "1":
            # Episode-based training
            num_episodes = int(input("Enter number of episodes to train (default: 100): ") or "100")
            train_agent(agent, game_region, score_region, coin_region, num_episodes, start_episode)
        else:
            # Continuous play mode
            continuous_play_loop(agent, game_region, score_region, coin_region)
    
    except KeyboardInterrupt:
        print("\nTraining stopped by user")
        print("Saving current model state...")
        agent.save_checkpoint(agent.episode_count, 0, path="models/subway_surfers_interrupted.pth")
        print("Model state saved to models/subway_surfers_interrupted.pth")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up resources
        pyautogui.hotkey('alt', 'f4')
        time.sleep(1)
        print("Program completed, browser closed.")

if __name__ == "__main__":
    main()