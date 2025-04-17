# env/game_interaction.py
import os
import time
import logging
import numpy as np
import cv2
import pyautogui
import pytesseract
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException, ElementNotInteractableException
import tkinter as tk

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SubwaySurfersEnv")

class SubwaySurfersEnv:
    """
    Environment for interacting with Subway Surfers game in a browser
    """
    def __init__(self, game_url="https://poki.com/en/g/subway-surfers", render_mode=None, position="right"):
        """
        Initialize the Subway Surfers environment
        
        Args:
            game_url: URL to the game
            render_mode: Rendering mode (None or "human")
            position: Position of the browser window ("left" or "right")
        """
        self.game_url = game_url
        self.render_mode = render_mode
        self.position = position
        
        # Game state
        self.score = 0
        self.coins = 0
        self.game_over = False
        self.last_frame = None
        self.step_count = 0
        self.episode_count = 0
        self.last_action = None
        self.last_reward = 0
        
        # Action space (0: no-op, 1: jump, 2: down, 3: left, 4: right)
        self.actions = ['noop', 'up', 'down', 'left', 'right']
        
        # Initialize browser
        logger.info("Initializing browser...")
        self.browser = self._init_browser()
        
        # Position browser window based on preference
        self._position_browser()
        
        # Wait for game to load
        logger.info("Waiting for game to load...")
        time.sleep(8)  # Wait for initial page load
        
        # Initialize game
        self.initialize_game()
        
        # Game regions (will be set during detection)
        self.game_region = None
        self.score_region = None
        self.coin_region = None
        
        # Detect game region
        self.detect_game_region()
        
        # Initialize GUI for visual feedback if render_mode is "human"
        if self.render_mode == "human":
            self._init_gui()
        
        # Debug: create debug_images directory if it doesn't exist
        os.makedirs("debug_images", exist_ok=True)
        os.makedirs("debug_images/states", exist_ok=True)
        
        logger.info("Game initialized successfully")
        
    def _init_browser(self):
        """Initialize the browser with appropriate settings"""
        chrome_options = Options()
        chrome_options.add_argument("--start-maximized")
        chrome_options.add_argument("--disable-infobars")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-notifications")
        chrome_options.add_argument("--disable-popup-blocking")
        
        # Initialize browser
        browser = webdriver.Chrome(options=chrome_options)
        browser.get(self.game_url)
        
        return browser
    
    def _position_browser(self):
        """Position the browser window based on preference"""
        screen_width = pyautogui.size().width
        screen_height = pyautogui.size().height
        
        if self.position == "right":
            # Position on right half of screen
            browser_x = screen_width // 2
            browser_width = screen_width // 2
            self.browser.set_window_rect(browser_x, 0, browser_width, screen_height)
            logger.info(f"Browser positioned on right side of screen at x={browser_x}, width={browser_width}")
        else:
            # Position on left half of screen
            browser_width = screen_width // 2
            self.browser.set_window_rect(0, 0, browser_width, screen_height)
            logger.info(f"Browser positioned on left side of screen with width={browser_width}")
    
    def _init_gui(self):
        """Initialize GUI for visual feedback"""
        self.root = tk.Tk()
        self.root.title("Subway Surfers RL")
        self.root.geometry("400x600")
        
        # Labels for displaying information
        self.info_frame = tk.Frame(self.root)
        self.info_frame.pack(pady=10, fill=tk.X)
        
        # Game info section
        tk.Label(self.info_frame, text="Game Information", font=('Arial', 12, 'bold')).pack()
        
        # Score and coins
        self.score_frame = tk.Frame(self.info_frame)
        self.score_frame.pack(pady=5, fill=tk.X)
        
        tk.Label(self.score_frame, text="Score:", width=10, anchor='w').grid(row=0, column=0, sticky='w')
        self.score_label = tk.Label(self.score_frame, text="0", width=10, anchor='e')
        self.score_label.grid(row=0, column=1, sticky='e')
        
        tk.Label(self.score_frame, text="Coins:", width=10, anchor='w').grid(row=1, column=0, sticky='w')
        self.coins_label = tk.Label(self.score_frame, text="0", width=10, anchor='e')
        self.coins_label.grid(row=1, column=1, sticky='e')
        
        # Episode info
        self.episode_frame = tk.Frame(self.info_frame)
        self.episode_frame.pack(pady=5, fill=tk.X)
        
        tk.Label(self.episode_frame, text="Episode:", width=10, anchor='w').grid(row=0, column=0, sticky='w')
        self.episode_label = tk.Label(self.episode_frame, text="0", width=10, anchor='e')
        self.episode_label.grid(row=0, column=1, sticky='e')
        
        tk.Label(self.episode_frame, text="Step:", width=10, anchor='w').grid(row=1, column=0, sticky='w')
        self.step_label = tk.Label(self.episode_frame, text="0", width=10, anchor='e')
        self.step_label.grid(row=1, column=1, sticky='e')
        
        # Action and reward
        self.action_frame = tk.Frame(self.info_frame)
        self.action_frame.pack(pady=5, fill=tk.X)
        
        tk.Label(self.action_frame, text="Action:", width=10, anchor='w').grid(row=0, column=0, sticky='w')
        self.action_label = tk.Label(self.action_frame, text="noop", width=10, anchor='e')
        self.action_label.grid(row=0, column=1, sticky='e')
        
        tk.Label(self.action_frame, text="Reward:", width=10, anchor='w').grid(row=1, column=0, sticky='w')
        self.reward_label = tk.Label(self.action_frame, text="0.0", width=10, anchor='e')
        self.reward_label.grid(row=1, column=1, sticky='e')
        
        # Game over status
        tk.Label(self.info_frame, text="Game Status:", width=10, anchor='w').pack(anchor='w')
        self.status_label = tk.Label(self.info_frame, text="Playing", fg="green", font=('Arial', 10, 'bold'))
        self.status_label.pack()
        
        # Stats section
        self.stats_frame = tk.Frame(self.root)
        self.stats_frame.pack(pady=10, fill=tk.X)
        
        tk.Label(self.stats_frame, text="Training Statistics", font=('Arial', 12, 'bold')).pack()
        
        # Training stats
        self.training_frame = tk.Frame(self.stats_frame)
        self.training_frame.pack(pady=5, fill=tk.X)
        
        tk.Label(self.training_frame, text="Avg Reward:", width=15, anchor='w').grid(row=0, column=0, sticky='w')
        self.avg_reward_label = tk.Label(self.training_frame, text="0.0", width=10, anchor='e')
        self.avg_reward_label.grid(row=0, column=1, sticky='e')
        
        tk.Label(self.training_frame, text="Epsilon:", width=15, anchor='w').grid(row=1, column=0, sticky='w')
        self.epsilon_label = tk.Label(self.training_frame, text="0.0", width=10, anchor='e')
        self.epsilon_label.grid(row=1, column=1, sticky='e')
        
        tk.Label(self.training_frame, text="Loss:", width=15, anchor='w').grid(row=2, column=0, sticky='w')
        self.loss_label = tk.Label(self.training_frame, text="0.0", width=10, anchor='e')
        self.loss_label.grid(row=2, column=1, sticky='e')
        
        # Canvas for displaying the game state
        self.state_canvas_label = tk.Label(self.root, text="Current State", font=('Arial', 10, 'bold'))
        self.state_canvas_label.pack(pady=(20, 5))
        self.state_canvas = tk.Canvas(self.root, width=200, height=200, bg='black')
        self.state_canvas.pack(pady=5)
        
        # Schedule the first update
        self.root.after(100, self._update_gui)
    
    def _update_gui(self):
        """Update GUI with current game state"""
        if hasattr(self, 'root') and self.root.winfo_exists():
            # Update game information
            self.score_label.config(text=f"{self.score}")
            self.coins_label.config(text=f"{self.coins}")
            self.episode_label.config(text=f"{self.episode_count}")
            self.step_label.config(text=f"{self.step_count}")
            self.action_label.config(text=f"{self.last_action if self.last_action else 'noop'}")
            self.reward_label.config(text=f"{self.last_reward:.2f}")
            
            # Update game status
            if self.game_over:
                self.status_label.config(text="Game Over", fg="red")
            else:
                self.status_label.config(text="Playing", fg="green")
            
            # Update epsilon and loss if available
            if hasattr(self, 'epsilon'):
                self.epsilon_label.config(text=f"{self.epsilon:.4f}")
            
            if hasattr(self, 'loss') and self.loss is not None:
                self.loss_label.config(text=f"{self.loss:.4f}")
            
            if hasattr(self, 'avg_reward') and self.avg_reward is not None:
                self.avg_reward_label.config(text=f"{self.avg_reward:.2f}")
            
            # Update state visualization if available
            if hasattr(self, 'current_state_img') and self.current_state_img is not None:
                self.state_canvas.delete("all")
                self.state_canvas.create_image(100, 100, image=self.current_state_img)
            
            # Schedule the next update
            self.root.after(100, self._update_gui)
    
    def initialize_game(self):
        """Initialize the game by clicking the play button or restarting"""
        logger.info("Initializing game...")
        
        try:
            # Wait for and click the play button (may be different depending on the site)
            WebDriverWait(self.browser, 5).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, ".pokiSdkStartButton"))
            ).click()
        except (TimeoutException, ElementNotInteractableException):
            # If can't find the play button, try clicking in the center of the page
            logger.info("Clicking center of screen as fallback")
            
            # Get the body element as a reference point
            body = self.browser.find_element(By.TAG_NAME, "body")
            
            # Calculate center position relative to the body element
            browser_width = self.browser.execute_script("return document.body.clientWidth")
            browser_height = self.browser.execute_script("return document.body.clientHeight")
            
            # Create action chain and move to body element first, then to center
            action = ActionChains(self.browser)
            action.move_to_element(body)
            action.move_by_offset(browser_width//2, browser_height//2)
            action.click()
            action.perform()
            
            # Wait for click to register
            time.sleep(3)
        
        # Make sure the game is ready
        time.sleep(1)
    
    def detect_game_region(self):
        """Detect the game region, score region, and coin region"""
        logger.info("Detecting game region...")
        # Take a screenshot
        screenshot = self.capture_screen()
        
        # Get screen dimensions from the browser
        screen_width = self.browser.get_window_size()['width']
        screen_height = self.browser.get_window_size()['height']
        
        # Define game region to focus on the main gameplay area (x, y, width, height)
        # These values are based on the screenshots and should match the actual game layout
        x_offset = int(screen_width * 0.4)  # Start a bit to the right of center
        y_offset = int(screen_height * 0.1)  # Start below the top UI elements
        game_width = int(screen_width * 0.55)  # Game width is about half the screen
        game_height = int(screen_height * 0.65)  # Game height is about 2/3 of the screen
        
        self.game_region = (x_offset, y_offset, game_width, game_height)
        logger.info(f"Game region detected: {self.game_region}")
        
        # Define score region at the top right corner
        score_x = int(screen_width * 0.85)
        score_y = int(screen_height * 0.09)
        score_width = int(screen_width * 0.10)
        score_height = int(screen_height * 0.04)
        self.score_region = (score_x, score_y, score_width, score_height)
        logger.info(f"Score region set to: {self.score_region}")
        
        # Define coin region just below the score region
        coin_x = score_x
        coin_y = int(screen_height * 0.135)
        coin_width = score_width
        coin_height = score_height
        self.coin_region = (coin_x, coin_y, coin_width, coin_height)
        logger.info(f"Coin region set to: {self.coin_region}")
        
        # Save a debug screenshot with regions drawn
        debug_img = screenshot.copy()
        
        # Draw game region (green)
        cv2.rectangle(
            debug_img,
            (self.game_region[0], self.game_region[1]),
            (self.game_region[0] + self.game_region[2], self.game_region[1] + self.game_region[3]),
            (0, 255, 0),
            2
        )
        
        # Draw score region (blue)
        cv2.rectangle(
            debug_img,
            (self.score_region[0], self.score_region[1]),
            (self.score_region[0] + self.score_region[2], self.score_region[1] + self.score_region[3]),
            (255, 0, 0),
            2
        )
        
        # Draw coin region (yellow)
        cv2.rectangle(
            debug_img,
            (self.coin_region[0], self.coin_region[1]),
            (self.coin_region[0] + self.coin_region[2], self.coin_region[1] + self.coin_region[3]),
            (0, 255, 255),
            2
        )
        
        # Add labels for clarity
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(debug_img, "Game Region", (self.game_region[0], self.game_region[1] - 10), 
                   font, 0.5, (0, 255, 0), 2)
        cv2.putText(debug_img, "Score", (self.score_region[0], self.score_region[1] - 10), 
                   font, 0.5, (255, 0, 0), 2)
        cv2.putText(debug_img, "Coins", (self.coin_region[0], self.coin_region[1] - 10), 
                   font, 0.5, (0, 255, 255), 2)
        
        # Save debug image
        debug_path = "debug_images/game_region.png"
        cv2.imwrite(debug_path, debug_img)
        logger.info(f"Debug screenshot with regions saved as {debug_path}")
    
    def capture_screen(self):
        """Capture the current screen as a numpy array"""
        # Take screenshot using selenium
        screenshot = self.browser.get_screenshot_as_png()
        
        # Convert to numpy array
        screenshot = np.frombuffer(screenshot, np.uint8)
        screenshot = cv2.imdecode(screenshot, cv2.IMREAD_COLOR)
        
        return screenshot
    
    def preprocess_frame(self, frame):
        """
        Preprocess a frame for the agent
        
        Args:
            frame: RGB image as numpy array
            
        Returns:
            Preprocessed frame (grayscale, resized to 84x84)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Resize to 84x84 (standard for DQN)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        
        # Normalize pixel values to [0, 1]
        normalized = resized / 255.0
        
        return normalized
    
    def get_game_frame(self):
        """Get the current game frame (cropped to game region)"""
        screenshot = self.capture_screen()
        
        # Crop to game region
        if self.game_region:
            x, y, width, height = self.game_region
            game_frame = screenshot[y:y+height, x:x+width]
            return game_frame
        else:
            logger.warning("Game region not detected, using full screenshot")
            return screenshot
    
    def get_score(self):
        """Extract score from the score region using OCR"""
        screenshot = self.capture_screen()
        
        if self.score_region:
            x, y, width, height = self.score_region
            score_image = screenshot[y:y+height, x:x+width]
            
            # Preprocess for OCR
            gray = cv2.cvtColor(score_image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            
            # Apply OCR
            try:
                text = pytesseract.image_to_string(binary, config='--psm 7 -c tessedit_char_whitelist=0123456789')
                text = text.strip()
                if text.isdigit():
                    return int(text)
                else:
                    # If OCR fails, increment the score by a small amount (simulation)
                    return self.score + 1
            except Exception as e:
                logger.warning(f"OCR error for score: {str(e)}")
                return self.score + 1
        else:
            # If region not set, simulate score increase
            return self.score + 1
    
    def get_coins(self):
        """Extract coins from the coin region using OCR"""
        screenshot = self.capture_screen()
        
        if self.coin_region:
            x, y, width, height = self.coin_region
            coin_image = screenshot[y:y+height, x:x+width]
            
            # Preprocess for OCR
            gray = cv2.cvtColor(coin_image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            
            # Apply OCR
            try:
                text = pytesseract.image_to_string(binary, config='--psm 7 -c tessedit_char_whitelist=0123456789')
                text = text.strip()
                if text.isdigit():
                    return int(text)
                else:
                    # If OCR fails, return current coin count
                    return self.coins
            except Exception as e:
                logger.warning(f"OCR error for coins: {str(e)}")
                return self.coins
        else:
            # If region not set, keep current coin count
            return self.coins
    
    def reset(self):
        """
        Reset the environment for a new episode
        
        Returns:
            Initial state (preprocessed frame)
        """
        logger.info("Resetting game...")
        
        # Increment episode counter
        self.episode_count += 1
        
        # Reset step counter
        self.step_count = 0
        
        # Close any game over dialogs by clicking
        self.browser.refresh()
        time.sleep(5)  # Wait for page to reload
        
        # Re-initialize the game
        self.initialize_game()
        
        # Reset game state
        self.score = 0
        self.coins = 0
        self.game_over = False
        self.last_frame = None
        self.last_action = None
        self.last_reward = 0
        
        # Get initial frame
        game_frame = self.get_game_frame()
        
        # Preprocess frame
        state = self.preprocess_frame(game_frame)
        
        # Store initial frame for comparison
        self.last_frame = game_frame
        
        # Return initial state
        return state
    
    def step(self, action):
        """
        Take an action in the environment
        
        Args:
            action: Action to take (integer index)
            
        Returns:
            tuple of (next_state, reward, done, info)
        """
        # Increment step counter
        self.step_count += 1
        
        # Store current score and coins for reward calculation
        prev_score = self.score
        prev_coins = self.coins
        
        # Convert action index to action name
        action_name = self.actions[action]
        self.last_action = action_name
        
        # Perform the action
        self._perform_action(action_name)
        
        # Small delay to allow action to take effect
        time.sleep(0.05)  # Reduced from higher values to allow faster gameplay
        
        # Get new frame
        game_frame = self.get_game_frame()
        
        # Check if game is over
        self.game_over = self._is_game_over(game_frame)
        
        # Update score and coins
        self.score = self.get_score()
        self.coins = self.get_coins()
        
        # Preprocess frame for agent
        next_state = self.preprocess_frame(game_frame)
        
        # Calculate reward components
        survival_reward = 0.1  # Small reward for surviving
        score_reward = (self.score - prev_score) * 0.5  # Reward for increasing score
        coin_reward = (self.coins - prev_coins) * 1.0  # Reward for collecting coins
        
        # Penalty for game over
        game_over_penalty = -10.0 if self.game_over else 0.0
        
        # Total reward
        reward = survival_reward + score_reward + coin_reward + game_over_penalty
        self.last_reward = reward
        
        # Save current frame for next comparison
        self.last_frame = game_frame
        
        # Debug: Save intermediate states periodically
        if self.step_count % 10 == 0:  # Save every 10th step
            debug_path = f"debug_images/screen_ep{self.episode_count}_step{self.step_count}.png"
            cv2.imwrite(debug_path, game_frame)
            logger.info(f"Debug screenshot with regions saved as {debug_path}")
            
            # Also save state visualization
            state_path = f"debug_images/states/state_ep{self.episode_count}_step{self.step_count}.png"
            state_img = (next_state * 255).astype(np.uint8)
            cv2.imwrite(state_path, state_img)
            logger.info(f"State visualization saved to {state_path}")
        
        # Return next_state, reward, done, info
        info = {
            'score': self.score,
            'coins': self.coins,
            'survival_reward': survival_reward,
            'score_reward': score_reward,
            'coin_reward': coin_reward,
            'game_over_penalty': game_over_penalty
        }
        
        return next_state, reward, self.game_over, info
    
    def _perform_action(self, action_name):
        """
        Perform the given action in the game
        
        Args:
            action_name: Name of the action to perform
        """
        # Map action name to keyboard key
        key_map = {
            'noop': None,
            'up': 'up',
            'down': 'down',
            'left': 'left',
            'right': 'right'
        }
        
        key = key_map[action_name]
        
        if key:
            # Click in the game area to ensure focus
            x, y, width, height = self.game_region
            center_x, center_y = x + width // 2, y + height // 2
            
            # Use ActionChains for more reliable interaction
            action = webdriver.ActionChains(self.browser)
            action.move_to_element_with_offset(self.browser.find_element(By.TAG_NAME, 'body'), center_x, center_y).click().perform()
            
            # Press the key
            pyautogui.press(key)
    
    def _is_game_over(self, current_frame):
        """
        Check if the game is over by comparing frames
        
        Args:
            current_frame: Current game frame
            
        Returns:
            Boolean indicating if the game is over
        """
        if self.last_frame is None:
            return False
        
        # Convert frames to grayscale for comparison
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
        last_gray = cv2.cvtColor(self.last_frame, cv2.COLOR_RGB2GRAY)
        
        # Calculate MSE (Mean Squared Error) between frames
        # If frames are nearly identical, the game might be over
        mse = np.mean((current_gray - last_gray) ** 2) / 255.0
        
        # Adjusted threshold to be less sensitive - this prevents premature game over detection
        threshold = 0.005  # Increased from what might have been a lower value

        # Check if the game is static (very little change between frames)
        is_static = mse < threshold
        
        if is_static:
            logger.info(f"Game over detected: screen is static ({mse:.4f})")
            
        return is_static
    
    def visualize_agent_state(self, state, step_num, episode_num):
        """Save a visualization of the current agent state for debugging"""
        if not isinstance(state, np.ndarray):
            logger.warning(f"Cannot visualize state: expected numpy array, got {type(state)}")
            return
            
        # If the state is a stack of frames
        if len(state.shape) == 3 and state.shape[0] > 1:  # (frames, height, width)
            # Create a grid of frames
            num_frames = state.shape[0]
            frame_height = state.shape[1]
            frame_width = state.shape[2]
            
            # Create a grid for visualization
            grid_size = int(np.ceil(np.sqrt(num_frames)))
            grid_height = grid_size * frame_height
            grid_width = grid_size * frame_width
            
            # Create blank grid
            grid = np.zeros((grid_height, grid_width))
            
            # Place frames in grid
            for i in range(num_frames):
                row = i // grid_size
                col = i % grid_size
                grid[row*frame_height:(row+1)*frame_height, col*frame_width:(col+1)*frame_width] = state[i]
            
            # Scale to 0-255 and convert to uint8
            grid = (grid * 255).astype(np.uint8)
            
            # Save visualization
            save_path = f"debug_images/states/state_ep{episode_num}_step{step_num}.png"
            cv2.imwrite(save_path, grid)
            logger.info(f"State visualization saved to {save_path}")
            
            # Update GUI if in human mode
            if self.render_mode == "human" and hasattr(self, 'root') and self.root.winfo_exists():
                # Convert to RGB for tkinter
                grid_rgb = cv2.cvtColor(grid, cv2.COLOR_GRAY2RGB)
                height, width = grid_rgb.shape[:2]
                
                # Resize if too large
                max_size = 200
                if height > max_size or width > max_size:
                    scale = max_size / max(height, width)
                    grid_rgb = cv2.resize(grid_rgb, (int(width * scale), int(height * scale)))
                
                # Convert to PhotoImage format for tkinter
                from PIL import Image, ImageTk
                img = Image.fromarray(grid_rgb)
                self.current_state_img = ImageTk.PhotoImage(image=img)
        else:
            # If it's a single frame, just save it directly
            frame = (state * 255).astype(np.uint8)
            save_path = f"debug_images/states/state_ep{episode_num}_step{step_num}.png"
            cv2.imwrite(save_path, frame)
            logger.info(f"State visualization saved to {save_path}")
            
            # Update GUI if in human mode
            if self.render_mode == "human" and hasattr(self, 'root') and self.root.winfo_exists():
                # Convert to RGB for tkinter
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                
                # Resize if needed
                max_size = a200
                height, width = frame_rgb.shape[:2]
                if height > max_size or width > max_size:
                    scale = max_size / max(height, width)
                    frame_rgb = cv2.resize(frame_rgb, (int(width * scale), int(height * scale)))
                
                # Convert to PhotoImage format for tkinter
                from PIL import Image, ImageTk
                img = Image.fromarray(frame_rgb)
                self.current_state_img = ImageTk.PhotoImage(image=img)
    
    def update_training_stats(self, epsilon=None, loss=None, avg_reward=None):
        """Update training statistics for display in GUI"""
        self.epsilon = epsilon
        self.loss = loss
        self.avg_reward = avg_reward
    
    def close(self):
        """Close the environment and cleanup resources"""
        if hasattr(self, 'browser') and self.browser:
            self.browser.quit()
        
        if hasattr(self, 'root') and self.root:
            self.root.destroy()
        
        logger.info("Environment closed")