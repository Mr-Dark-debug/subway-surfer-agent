"""
Subway Surfers AI - Model Inference/Gameplay
This script loads a trained model and plays Subway Surfers with it.
"""

import pyautogui
import time
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import subprocess
from datetime import datetime
import argparse

# Check if CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# DQN Model Definition (same as in training)
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

# Game state detection and handling
class GameStateDetector:
    def __init__(self):
        self.save_me_template = None
        self.score_screen_template = None
        self.try_load_templates()
        
    def try_load_templates(self):
        """Try to load template images for game state detection"""
        try:
            if os.path.exists("templates/save_me.png"):
                self.save_me_template = cv2.imread("templates/save_me.png", cv2.IMREAD_COLOR)
                print("Loaded 'Save me!' template for detection")
                
            if os.path.exists("templates/score_screen.png"):
                self.score_screen_template = cv2.imread("templates/score_screen.png", cv2.IMREAD_COLOR)
                print("Loaded score screen template for detection")
        except Exception as e:
            print(f"Warning: Could not load template images: {e}")
    
    def detect_game_over(self, game_img):
        """Detect if the game is over based on image analysis"""
        img_array = np.array(game_img)
        
        # Check for 'Save me!' screen
        if self.save_me_template is not None:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            result = cv2.matchTemplate(img_bgr, self.save_me_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if max_val > 0.7:
                return True, "save_me"
        
        # Check for score screen
        if self.score_screen_template is not None:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            result = cv2.matchTemplate(img_bgr, self.score_screen_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if max_val > 0.7:
                return True, "score_screen"
        
        # Fallback detection
        # Look for "PLAY" button (green)
        img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        lower_green = np.array([40, 100, 100])
        upper_green = np.array([80, 255, 255])
        green_mask = cv2.inRange(img_hsv, lower_green, upper_green)
        
        # Look for blue "Score" text
        lower_blue = np.array([100, 150, 150])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(img_hsv, lower_blue, upper_blue)
        
        # Check if both colors exist in expected regions
        if (np.sum(green_mask[450:550, 450:650]) > 5000 and 
            np.sum(blue_mask[100:200, 400:600]) > 5000):
            return True, "score_screen"
        
        # Look for blue "Save me!" text
        center_region = blue_mask[150:250, 400:600]
        if np.sum(center_region) > 10000:
            return True, "save_me"
            
        return False, None
    
    def handle_game_over(self, game_over_type, game_region):
        """Handle the game over state based on the type detected"""
        region_center_x = game_region[0] + game_region[2] // 2
        region_center_y = game_region[1] + game_region[3] // 2
        
        if game_over_type == "save_me":
            # Try clicking the "Free!" button
            free_button_x = region_center_x - 50
            free_button_y = region_center_y - 50
            
            pyautogui.click(free_button_x, free_button_y)
            time.sleep(1)
            
        elif game_over_type == "score_screen":
            # Click the "PLAY" button
            play_button_x = region_center_x + 100
            play_button_y = region_center_y + 150
            
            pyautogui.click(play_button_x, play_button_y)
            time.sleep(2)
        
        else:
            # Generic restart
            pyautogui.click(region_center_x, region_center_y)
            time.sleep(0.5)
            pyautogui.press('space')
            time.sleep(2)

# Setup game and browser
def setup_game():
    """Set up the Subway Surfers game in browser"""
    # Get screen size
    screen_width, screen_height = pyautogui.size()
    print(f"Screen size: {screen_width}x{screen_height}")
    
    game_region = (1120, 178, 806, 529)
    score_region = (1682, 170, 225, 48)
    coin_region = (1682, 217, 225, 48)
    
    print("Game region:", game_region)
    print("Score region:", score_region)
    print("Coin region:", coin_region)
    
    # Create screenshots directory if it doesn't exist
    os.makedirs("gameplay", exist_ok=True)
    
    # Open Subway Surfers on Poki in Chrome incognito mode
    url = "https://poki.com/en/g/subway-surfers"
    open_browser_incognito(url)
    
    # Wait for browser to load
    print("Waiting for browser to load...")
    time.sleep(5)
    
    # Move browser to right side of screen (split-screen)
    position_window_right_side(screen_width, screen_height)
    
    return game_region, score_region, coin_region

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

# Frame preprocessing function
def preprocess_frame(frame):
    """Preprocess a frame for model input"""
    # Convert to grayscale
    gray = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2GRAY)
    
    # Resize to a smaller dimension for processing efficiency
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    
    # Normalize pixel values
    normalized = resized / 255.0
    
    # Add channel dimension and convert to tensor
    tensor = torch.FloatTensor(normalized).unsqueeze(0).unsqueeze(0)
    
    return tensor.to(device)

# AI Agent for gameplay
class SubwaySurfersPlayer:
    def __init__(self, model_path):
        # Define model input size
        self.h, self.w = 84, 84
        self.n_actions = 4
        
        # Action mapping
        self.actions = {
            0: 'left',   # Left arrow key
            1: 'up',     # Up arrow key (jump)
            2: 'right',  # Right arrow key
            3: 'down'    # Down arrow key (roll)
        }
        
        # Initialize the model
        self.model = DQN(self.h, self.w, self.n_actions).to(device)
        
        # Load trained model weights
        try:
            checkpoint = torch.load(model_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {model_path}")
            
            # Print model info from checkpoint
            episode = checkpoint.get('episode', 0)
            reward = checkpoint.get('reward', 0)
            epsilon = checkpoint.get('epsilon', 0)
            print(f"Model trained for {episode} episodes")
            print(f"Last reward: {reward:.2f}, Final epsilon: {epsilon:.4f}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        # Set model to evaluation mode
        self.model.eval()
    
    def select_action(self, state):
        """Select best action for the given state"""
        with torch.no_grad():
            # Get Q-values for each action
            q_values = self.model(state)
            
            # Select action with highest Q-value
            action_idx = q_values.max(1)[1].view(1, 1)
            
            # Get the action name for display
            action_name = self.actions[action_idx.item()]
            
            return action_idx, action_name
    
    def execute_action(self, action_idx):
        """Execute the selected action using pyautogui"""
        action = self.actions[action_idx.item()]
        pyautogui.press(action)
        time.sleep(0.05)  # Short delay for the action to take effect

# Main gameplay function
def play_game(model_path):
    """Play Subway Surfers using the trained model"""
    # Setup game
    game_region, score_region, coin_region = setup_game()
    
    # Initialize the player with the model
    player = SubwaySurfersPlayer(model_path)
    
    # Initialize game state detector
    state_detector = GameStateDetector()
    
    print("\nGame is ready! Press Space to start playing...")
    time.sleep(1)
    pyautogui.press('space')
    time.sleep(2)
    
    # Game stats
    frames = 0
    game_count = 0
    start_time = time.time()
    
    try:
        while True:
            # Capture game state
            game_img = pyautogui.screenshot(region=game_region)
            
            # Check if game is over
            game_over, game_over_type = state_detector.detect_game_over(game_img)
            
            if game_over:
                game_count += 1
                print(f"\nGame {game_count} over! Type: {game_over_type}")
                
                # Save final frame
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                game_img.save(f"gameplay/game_{game_count}_end_{timestamp}.png")
                
                # Handle game over
                state_detector.handle_game_over(game_over_type, game_region)
                time.sleep(2)  # Wait for restart
                
                # Reset frame counter for this game
                print(f"Starting new game {game_count + 1}...")
                frames = 0
                continue
            
            # Preprocess frame
            state = preprocess_frame(game_img)
            
            # Select action
            action, action_name = player.select_action(state)
            
            # Execute action
            player.execute_action(action)
            
            # Increment frame counter
            frames += 1
            
            # Periodic status update
            if frames % 10 == 0:
                elapsed = time.time() - start_time
                fps = frames / elapsed if elapsed > 0 else 0
                print(f"Frame: {frames}, Action: {action_name}, FPS: {fps:.1f}", end="\r")
            
            # Save occasional gameplay screenshots
            if frames % 100 == 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                game_img.save(f"gameplay/game_{game_count}_frame_{frames}_{timestamp}.png")
            
            # Short delay to prevent high CPU usage
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\n\nGameplay stopped by user")
        elapsed = time.time() - start_time
        
        # Print gameplay statistics
        print(f"\nGameplay Statistics:")
        print(f"Total time: {elapsed:.1f} seconds")
        print(f"Games played: {game_count}")
        print(f"Frames processed: {frames}")
        print(f"Average FPS: {frames / elapsed if elapsed > 0 else 0:.1f}")
    finally:
        # Clean up resources
        pyautogui.hotkey('alt', 'f4')  # Close browser
        print("Browser closed. Exiting program.")

# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Subway Surfers AI Player')
    parser.add_argument('--model', type=str, default="models/latest_checkpoint.pth",
                        help='Path to the trained model file')
    args = parser.parse_args()
    
    # Print banner
    print("=" * 60)
    print("Subway Surfers AI Player")
    print("=" * 60)
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Model file not found: {args.model}")
        print("Looking for other models...")
        
        # Try to find any model
        models = sorted(glob.glob("models/*.pth"))
        if models:
            print(f"Found {len(models)} model(s):")
            for i, model in enumerate(models):
                print(f"{i+1}. {model}")
            
            # Ask user to select
            selection = input(f"Select a model (1-{len(models)}) or press Enter for the latest: ")
            if selection.strip():
                try:
                    index = int(selection) - 1
                    if 0 <= index < len(models):
                        model_path = models[index]
                    else:
                        model_path = models[-1]  # Default to latest
                except ValueError:
                    model_path = models[-1]  # Default to latest
            else:
                model_path = models[-1]  # Default to latest
        else:
            print("No models found. Please train a model first.")
            exit()
    else:
        model_path = args.model
    
    print(f"Using model: {model_path}")
    print("Starting game in 3 seconds...")
    time.sleep(3)
    
    # Start gameplay
    play_game(model_path)