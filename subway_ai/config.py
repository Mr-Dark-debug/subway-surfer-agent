# Improved configuration file for Subway Surfers AI
import os
import torch

# Game settings
GAME_PATH = "Subway_Surfers.exe"  # Path to game executable (modify if needed)
GAME_WINDOW_SIZE = (800, 600)

# Screen regions - these are updated by calibrate.py
# These are example values, use calibration tool to set proper values
GAME_REGION = (1089, 229, 801, 642)  # Main game window region (x, y, width, height)
SCORE_REGION = (1643, 262, 244, 45)  # Score display region
COIN_REGION = (1755, 311, 131, 43)   # Coin counter region

# Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "screenshots")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Template paths
GAME_OVER_TEMPLATE = os.path.join(TEMPLATES_DIR, "gameover.png")
PLAY_BUTTON_TEMPLATE = os.path.join(TEMPLATES_DIR, "play_button.png")

# OCR settings
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this to your Tesseract installation path

# Model hyperparameters
BATCH_SIZE = 32
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 10000
MEMORY_SIZE = 10000
LEARNING_RATE = 0.0002
TARGET_UPDATE = 10
CHECKPOINT_INTERVAL = 5
SCREENSHOT_INTERVAL = 50

# Device settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Detection thresholds
GAME_OVER_DETECTION_THRESHOLD = 0.7  # Threshold for game over template matching
PLAY_BUTTON_DETECTION_THRESHOLD = 0.7  # Threshold for play button template matching
MIN_GREEN_PIXELS = 5000  # Minimum number of green pixels for play button detection
CONSECUTIVE_DETECTIONS_REQUIRED = 3  # Number of consecutive detections required for stability

# Game control settings
ACTION_COOLDOWN = 0.25  # Seconds between actions
RESTART_COOLDOWN = 5.0  # Seconds between restart attempts
SWIPE_DURATION = 0.15  # Duration of swipe gestures

# Reward settings
REWARD_SURVIVAL = 0.1     # Small reward for surviving each step
REWARD_COIN = 1.0         # Reward for collecting a coin
REWARD_SCORE = 0.02       # Reward multiplier for score increase
PENALTY_CRASH = -10.0     # Penalty for crashing

# Action space - For swipe controls
ACTIONS = {
    0: "LEFT",
    1: "RIGHT",
    2: "UP",
    3: "DOWN",
    4: "NONE"
}
NUM_ACTIONS = len(ACTIONS)

# Debug settings
DEBUG_MODE = False  # Set to True to enable additional debug output and visualizations

# Function to load custom settings from a JSON file (if it exists)
def load_custom_settings():
    """Load custom settings from settings.json if available"""
    import json
    settings_path = os.path.join(BASE_DIR, "settings.json")
    if os.path.exists(settings_path):
        try:
            with open(settings_path, 'r') as f:
                custom_settings = json.load(f)
            
            # Update globals with custom settings
            globals().update(custom_settings)
            print(f"Loaded custom settings from {settings_path}")
        except Exception as e:
            print(f"Error loading custom settings: {e}")

# Try to load custom settings
try:
    load_custom_settings()
except:
    pass