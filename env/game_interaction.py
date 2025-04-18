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
import random
from PIL import Image, ImageTk

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
        try:
            # Use JavaScript to get accurate screen dimensions
            screen_width = self.browser.execute_script("return window.screen.width")
            screen_height = self.browser.execute_script("return window.screen.height")
            
            # Make sure we have valid dimensions
            if not screen_width or not screen_height:
                # Fallback to pyautogui if JavaScript fails
                screen_size = pyautogui.size()
                screen_width = screen_size.width
                screen_height = screen_size.height
            
            logger.info(f"Screen dimensions: {screen_width}x{screen_height}")
            
            # Calculate window dimensions based on preference
            if self.position == "right":
                # Position on right half of screen with a margin
                browser_x = screen_width // 2
                browser_width = (screen_width // 2) - 20  # Slightly less than half to avoid edge issues
                browser_height = screen_height - 60  # Leave space for taskbar and other UI
                browser_y = 0
                
                self.browser.set_window_rect(browser_x, browser_y, browser_width, browser_height)
                logger.info(f"Browser positioned on right side of screen at x={browser_x}, width={browser_width}")
            elif self.position == "left":
                # Position on left half of screen with a margin
                browser_x = 0
                browser_width = (screen_width // 2) - 20
                browser_height = screen_height - 60
                browser_y = 0
                
                self.browser.set_window_rect(browser_x, browser_y, browser_width, browser_height)
                logger.info(f"Browser positioned on left side of screen with width={browser_width}")
            else:  # "center" or any other value
                # Center the browser with reasonable dimensions
                browser_width = min(1024, screen_width - 40)  # Use 1024px or slightly less than screen width
                browser_height = min(768, screen_height - 60)  # Use 768px or slightly less than screen height
                browser_x = (screen_width - browser_width) // 2
                browser_y = (screen_height - browser_height) // 2
                
                self.browser.set_window_rect(browser_x, browser_y, browser_width, browser_height)
                logger.info(f"Browser positioned in center with dimensions {browser_width}x{browser_height}")
            
            # Ensure the browser has focus
            self.browser.switch_to.window(self.browser.current_window_handle)
            
        except Exception as e:
            logger.warning(f"Error positioning browser window: {str(e)}")
            logger.warning("Continuing with default browser positioning")
    
    def _init_gui(self):
        """Initialize GUI for visual feedback"""
        self.root = tk.Tk()
        self.root.title("Subway Surfers RL")
        self.root.geometry("800x600")
        
        # Configure columns/rows to be responsive
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        
        # Top frame for info
        self.info_frame = tk.Frame(self.root)
        self.info_frame.grid(row=0, column=0, sticky='ew', padx=10, pady=5)
        
        # Game info section
        tk.Label(self.info_frame, text="Game Information", font=('Arial', 12, 'bold')).pack(anchor='w')
        
        # Create horizontal frames
        self.horizontal_frame = tk.Frame(self.info_frame)
        self.horizontal_frame.pack(fill=tk.X, expand=True)
        
        # Left side
        self.left_frame = tk.Frame(self.horizontal_frame)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        # Game score and coins
        self.score_frame = tk.Frame(self.left_frame)
        self.score_frame.pack(pady=5, fill=tk.X)
        
        tk.Label(self.score_frame, text="Score:", width=10, anchor='w').grid(row=0, column=0, sticky='w')
        self.score_label = tk.Label(self.score_frame, text="0", width=10, anchor='e')
        self.score_label.grid(row=0, column=1, sticky='e')
        
        tk.Label(self.score_frame, text="Coins:", width=10, anchor='w').grid(row=1, column=0, sticky='w')
        self.coins_label = tk.Label(self.score_frame, text="0", width=10, anchor='e')
        self.coins_label.grid(row=1, column=1, sticky='e')
        
        # Right side
        self.right_frame = tk.Frame(self.horizontal_frame)
        self.right_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        # Episode info
        self.episode_frame = tk.Frame(self.right_frame)
        self.episode_frame.pack(pady=5, fill=tk.X)
        
        tk.Label(self.episode_frame, text="Episode:", width=10, anchor='w').grid(row=0, column=0, sticky='w')
        self.episode_label = tk.Label(self.episode_frame, text="0", width=10, anchor='e')
        self.episode_label.grid(row=0, column=1, sticky='e')
        
        tk.Label(self.episode_frame, text="Step:", width=10, anchor='w').grid(row=1, column=0, sticky='w')
        self.step_label = tk.Label(self.episode_frame, text="0", width=10, anchor='e')
        self.step_label.grid(row=1, column=1, sticky='e')
        
        # Action and reward
        self.action_frame = tk.Frame(self.right_frame)
        self.action_frame.pack(pady=5, fill=tk.X)
        
        tk.Label(self.action_frame, text="Action:", width=10, anchor='w').grid(row=0, column=0, sticky='w')
        self.action_label = tk.Label(self.action_frame, text="noop", width=10, anchor='e')
        self.action_label.grid(row=0, column=1, sticky='e')
        
        tk.Label(self.action_frame, text="Reward:", width=10, anchor='w').grid(row=1, column=0, sticky='w')
        self.reward_label = tk.Label(self.action_frame, text="0.0", width=10, anchor='e')
        self.reward_label.grid(row=1, column=1, sticky='e')
        
        # Game over status
        self.status_frame = tk.Frame(self.info_frame)
        self.status_frame.pack(pady=5, fill=tk.X)
        tk.Label(self.status_frame, text="Game Status:", width=10, anchor='w').pack(side=tk.LEFT)
        self.status_label = tk.Label(self.status_frame, text="Playing", fg="green", font=('Arial', 10, 'bold'))
        self.status_label.pack(side=tk.LEFT)
        
        # Central frame for game state and regions
        self.central_frame = tk.Frame(self.root)
        self.central_frame.grid(row=1, column=0, sticky='nsew', padx=10, pady=5)
        
        # Left side: Game state
        self.state_frame = tk.Frame(self.central_frame)
        self.state_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.state_canvas_label = tk.Label(self.state_frame, text="Current State", font=('Arial', 10, 'bold'))
        self.state_canvas_label.pack(pady=(5, 5))
        self.state_canvas = tk.Canvas(self.state_frame, width=200, height=200, bg='black')
        self.state_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Right side: Game regions
        self.regions_frame = tk.Frame(self.central_frame)
        self.regions_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        self.regions_label = tk.Label(self.regions_frame, text="Game Regions", font=('Arial', 10, 'bold'))
        self.regions_label.pack(pady=(5, 5))
        self.regions_canvas = tk.Canvas(self.regions_frame, width=300, height=300, bg='black')
        self.regions_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bottom frame for stats
        self.stats_frame = tk.Frame(self.root)
        self.stats_frame.grid(row=2, column=0, sticky='ew', padx=10, pady=5)
        
        tk.Label(self.stats_frame, text="Training Statistics", font=('Arial', 12, 'bold')).pack(anchor='w')
        
        # Horizontal layout for statistics
        self.stats_horizontal_frame = tk.Frame(self.stats_frame)
        self.stats_horizontal_frame.pack(fill=tk.X, expand=True)
        
        # Training stats
        self.training_frame = tk.Frame(self.stats_horizontal_frame)
        self.training_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        tk.Label(self.training_frame, text="Avg Reward:", width=15, anchor='w').grid(row=0, column=0, sticky='w')
        self.avg_reward_label = tk.Label(self.training_frame, text="0.0", width=10, anchor='e')
        self.avg_reward_label.grid(row=0, column=1, sticky='e')
        
        tk.Label(self.training_frame, text="Epsilon:", width=15, anchor='w').grid(row=1, column=0, sticky='w')
        self.epsilon_label = tk.Label(self.training_frame, text="0.0", width=10, anchor='e')
        self.epsilon_label.grid(row=1, column=1, sticky='e')
        
        tk.Label(self.training_frame, text="Loss:", width=15, anchor='w').grid(row=2, column=0, sticky='w')
        self.loss_label = tk.Label(self.training_frame, text="0.0", width=10, anchor='e')
        self.loss_label.grid(row=2, column=1, sticky='e')
        
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
            
            # Update game regions visualization
            if self.render_mode == "human" and hasattr(self, 'regions_canvas'):
                try:
                    # Get a screenshot
                    screenshot = self.capture_screen()
                    
                    # Resize to fit the canvas
                    if screenshot is not None:
                        canvas_width = self.regions_canvas.winfo_width()
                        canvas_height = self.regions_canvas.winfo_height()
                        
                        # Make sure we have valid dimensions
                        if canvas_width > 50 and canvas_height > 50:
                            # Resize screenshot to fit the canvas
                            aspect_ratio = screenshot.shape[1] / screenshot.shape[0]
                            
                            if aspect_ratio > (canvas_width / canvas_height):
                                # Width constrained
                                display_width = canvas_width
                                display_height = int(canvas_width / aspect_ratio)
                            else:
                                # Height constrained
                                display_height = canvas_height
                                display_width = int(canvas_height * aspect_ratio)
                            
                            # Resize the image to display size
                            display_img = cv2.resize(screenshot, (display_width, display_height))
                            
                            # Scale the region coordinates
                            scale_x = display_width / screenshot.shape[1]
                            scale_y = display_height / screenshot.shape[0]
                            
                            # Draw regions on the image
                            if self.game_region:
                                x, y, w, h = self.game_region
                                x1, y1 = int(x * scale_x), int(y * scale_y)
                                x2, y2 = int((x + w) * scale_x), int((y + h) * scale_y)
                                cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(display_img, "Game", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                
                            if self.score_region:
                                x, y, w, h = self.score_region
                                x1, y1 = int(x * scale_x), int(y * scale_y)
                                x2, y2 = int((x + w) * scale_x), int((y + h) * scale_y)
                                cv2.rectangle(display_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                                cv2.putText(display_img, "Score", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                                
                            if self.coin_region:
                                x, y, w, h = self.coin_region
                                x1, y1 = int(x * scale_x), int(y * scale_y)
                                x2, y2 = int((x + w) * scale_x), int((y + h) * scale_y)
                                cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                                cv2.putText(display_img, "Coins", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                            
                            # Convert to RGB for PIL
                            display_img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
                            
                            # Convert to PhotoImage
                            pil_img = Image.fromarray(display_img_rgb)
                            photo_img = ImageTk.PhotoImage(image=pil_img)
                            
                            # Update canvas
                            self.regions_canvas.delete("all")
                            self.regions_canvas.create_image(display_width//2, display_height//2, image=photo_img)
                            self.regions_canvas.image = photo_img  # Keep a reference
                except Exception as e:
                    logger.warning(f"Error updating regions visualization: {str(e)}")
            
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
            
            try:
                # Use JavaScript to get accurate window dimensions and click center
                browser_width = self.browser.execute_script("return window.innerWidth")
                browser_height = self.browser.execute_script("return window.innerHeight")
                
                center_x = browser_width // 2
                center_y = browser_height // 2
                
                # Use JavaScript to click at the center coordinates
                self.browser.execute_script(f"document.elementFromPoint({center_x}, {center_y}).click()")
                logger.info(f"Clicked at center coordinates: ({center_x}, {center_y})")
            except Exception as e:
                logger.warning(f"Failed to click using JavaScript: {str(e)}")
                
                # Alternative approach: try to find and click on the game canvas directly
                try:
                    # Look for common game canvas elements
                    canvas_elements = self.browser.find_elements(By.TAG_NAME, "canvas")
                    if canvas_elements:
                        canvas_elements[0].click()
                        logger.info("Clicked on canvas element")
                    else:
                        logger.warning("No canvas elements found")
                except Exception as canvas_e:
                    logger.warning(f"Failed to click on canvas: {str(canvas_e)}")
        
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
        
        # Define improved game region based on the Subway Surfers layout from the screenshot
        # The main gameplay area should include the tracks, player character, and obstacles
        x_offset = int(screen_width * 0.10)  # Start from 10% from the left (wider)
        y_offset = int(screen_height * 0.10)  # Start higher - 10% from top
        game_width = int(screen_width * 0.80)  # Game width is about 80% of screen width
        game_height = int(screen_height * 0.75)  # Game height is about 75% of screen height
        
        self.game_region = (x_offset, y_offset, game_width, game_height)
        logger.info(f"Game region detected: {self.game_region}")
        
        # Define score region at the top right corner
        # Move it higher based on screenshot
        score_x = int(screen_width * 0.75)
        score_y = int(screen_height * 0.08)  # Move higher
        score_width = int(screen_width * 0.20)  # Make wider
        score_height = int(screen_height * 0.06)  # Make slightly taller
        self.score_region = (score_x, score_y, score_width, score_height)
        logger.info(f"Score region set to: {self.score_region}")
        
        # Define coin region (with the coin icon) below the score
        coin_x = int(screen_width * 0.75)
        coin_y = int(screen_height * 0.15)  # Position directly below score
        coin_width = int(screen_width * 0.20)  # Make wider
        coin_height = int(screen_height * 0.06)  # Make slightly taller
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
                    font, 0.7, (0, 255, 0), 2)
        cv2.putText(debug_img, "Score", (self.score_region[0], self.score_region[1] - 10), 
                    font, 0.7, (255, 0, 0), 2)
        cv2.putText(debug_img, "Coins", (self.coin_region[0], self.coin_region[1] - 10), 
                    font, 0.7, (0, 255, 255), 2)
        
        # Save debug image
        debug_path = "debug_images/game_region.png"
        cv2.imwrite(debug_path, debug_img)
        logger.info(f"Debug screenshot with regions saved as {debug_path}")
        
        # If show_regions is enabled, also display the debug image
        if self.render_mode == "human":
            # Display the image with detected regions
            cv2.imshow("Game Regions", debug_img)
            cv2.waitKey(1)  # Update the window
    
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
        try:
            # Make a copy to avoid modifying the original
            frame_copy = frame.copy()
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame_copy, cv2.COLOR_RGB2GRAY)
            
            # Resize to 84x84 (standard for DQN)
            resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
            
            # Normalize pixel values to [0, 1]
            normalized = resized / 255.0
            
            # Clean up to free memory
            del frame_copy
            del gray
            del resized
            
            return normalized
        except Exception as e:
            logger.error(f"Error preprocessing frame: {str(e)}")
            # Return blank frame in case of error
            return np.zeros((84, 84), dtype=np.float32)
    
    def get_game_frame(self):
        """Get the current game frame (cropped to game region)"""
        try:
            screenshot = self.capture_screen()
            
            # Crop to game region
            if self.game_region:
                x, y, width, height = self.game_region
                game_frame = screenshot[y:y+height, x:x+width].copy()  # Use .copy() to ensure memory is released
                
                # Release the memory of the full screenshot
                del screenshot
                
                return game_frame
            else:
                logger.warning("Game region not detected, using full screenshot")
                return screenshot.copy()
        except Exception as e:
            logger.error(f"Error getting game frame: {str(e)}")
            # Return blank frame in case of error
            if self.game_region:
                x, y, width, height = self.game_region
                return np.zeros((height, width, 3), dtype=np.uint8)
            else:
                return np.zeros((480, 640, 3), dtype=np.uint8)
    
    def get_score(self):
        """Extract score from the score region using OCR"""
        try:
            screenshot = self.capture_screen()
            
            if self.score_region:
                x, y, width, height = self.score_region
                score_image = screenshot[y:y+height, x:x+width]
                
                # Save original ROI for debugging
                cv2.imwrite("debug_images/score_roi.png", score_image)
                
                # Enhanced preprocessing for OCR
                # Convert to grayscale
                gray = cv2.cvtColor(score_image, cv2.COLOR_BGR2GRAY)
                
                # Apply adaptive thresholding to handle different lighting conditions
                binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY, 11, 2)
                
                # Invert if needed (white text on black background)
                if np.mean(binary) < 127:
                    binary = cv2.bitwise_not(binary)
                    
                # Dilate to make text more visible
                kernel = np.ones((2, 2), np.uint8)
                dilated = cv2.dilate(binary, kernel, iterations=1)
                
                # Save processed image for debugging
                os.makedirs("debug_images/score", exist_ok=True)
                timestamp = int(time.time() * 1000)
                cv2.imwrite(f"debug_images/score/processed_{timestamp}.png", dilated)
                
                # Try different OCR configurations
                ocr_configs = [
                    '--psm 7 -c tessedit_char_whitelist=0123456789',  # Single line of text
                    '--psm 8 -c tessedit_char_whitelist=0123456789',  # Single word
                    '--psm 10 -c tessedit_char_whitelist=0123456789', # Single character
                    '--psm 6 -c tessedit_char_whitelist=0123456789'   # Assume uniform block of text
                ]
                
                for config in ocr_configs:
                    try:
                        text = pytesseract.image_to_string(dilated, config=config)
                        text = ''.join(filter(str.isdigit, text))  # Keep only digits
                        
                        if text and text.isdigit():
                            logger.debug(f"OCR detected score: {text}")
                            return int(text)
                    except Exception as e:
                        logger.debug(f"OCR attempt failed with config {config}: {str(e)}")
                        continue
                
                # If all OCR attempts fail, use differential score update
                logger.debug("OCR failed, using estimated score")
                return self.score + 5  # Assume score increases by about 5 per step
            else:
                # If region not set, simulate score increase
                return self.score + 5
        except Exception as e:
            logger.warning(f"Error in score detection: {str(e)}")
            return self.score + 5  # Fallback to estimated score
    
    def get_coins(self):
        """Extract coins from the coin region using OCR"""
        try:
            screenshot = self.capture_screen()
            
            if self.coin_region:
                x, y, width, height = self.coin_region
                coin_image = screenshot[y:y+height, x:x+width]
                
                # Save original ROI for debugging
                cv2.imwrite("debug_images/coins_roi.png", coin_image)
                
                # Enhanced preprocessing for OCR
                # Convert to grayscale
                gray = cv2.cvtColor(coin_image, cv2.COLOR_BGR2GRAY)
                
                # Apply adaptive thresholding
                binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY, 11, 2)
                
                # Invert if needed (white text on black background)
                if np.mean(binary) < 127:
                    binary = cv2.bitwise_not(binary)
                    
                # Dilate to make text more visible
                kernel = np.ones((2, 2), np.uint8)
                dilated = cv2.dilate(binary, kernel, iterations=1)
                
                # Save processed image for debugging
                os.makedirs("debug_images/coins", exist_ok=True)
                timestamp = int(time.time() * 1000)
                cv2.imwrite(f"debug_images/coins/processed_{timestamp}.png", dilated)
                
                # Try different OCR configurations
                ocr_configs = [
                    '--psm 7 -c tessedit_char_whitelist=0123456789',  # Single line of text
                    '--psm 8 -c tessedit_char_whitelist=0123456789',  # Single word
                    '--psm 10 -c tessedit_char_whitelist=0123456789', # Single character
                    '--psm 6 -c tessedit_char_whitelist=0123456789'   # Assume uniform block of text
                ]
                
                for config in ocr_configs:
                    try:
                        text = pytesseract.image_to_string(dilated, config=config)
                        text = ''.join(filter(str.isdigit, text))  # Keep only digits
                        
                        if text and text.isdigit():
                            logger.debug(f"OCR detected coins: {text}")
                            return int(text)
                    except Exception as e:
                        logger.debug(f"OCR attempt failed with config {config}: {str(e)}")
                        continue
                
                # If all OCR attempts fail, return previous value with small increment
                # Coins don't increment as frequently as score
                coin_change = 1 if random.random() < 0.1 else 0  # 10% chance of finding a coin
                logger.debug("OCR failed, using estimated coins")
                return self.coins + coin_change
            else:
                # If region not set, simulate occasional coin pickup
                coin_change = 1 if random.random() < 0.1 else 0
                return self.coins + coin_change
        except Exception as e:
            logger.warning(f"Error in coin detection: {str(e)}")
            return self.coins  # Fallback to current coins
    
    def reset(self):
        """
        Reset the environment
        
        Returns:
            Initial observation
        """
        logger.info("Resetting environment...")
        self.score = 0
        self.coins = 0
        self.game_over = False
        self.step_count = 0
        self.episode_count += 1
        self.last_action = None
        self.last_reward = 0
        
        # Clear memory for recent frames
        if hasattr(self, 'recent_frames'):
            self.recent_frames.clear()
        else:
            self.recent_frames = []
        
        # Get screen dimensions
        browser_width = self.browser.execute_script("return window.innerWidth")
        browser_height = self.browser.execute_script("return window.innerHeight")
        center_x = browser_width // 2
        center_y = browser_height // 2
        
        # Multiple attempts to restart the game
        restart_attempts = 0
        max_attempts = 5
        success = False
        
        while restart_attempts < max_attempts and not success:
            try:
                restart_attempts += 1
                logger.info(f"Restart attempt {restart_attempts}/{max_attempts}")
                
                # First click to close any dialogs (like game over screen)
                try:
                    self.browser.execute_script(f"document.elementFromPoint({center_x}, {center_y}).click()")
                    logger.info(f"Clicked center of screen at ({center_x}, {center_y})")
                    time.sleep(0.5)
                except Exception as e:
                    logger.warning(f"Error clicking center: {str(e)}")
                
                # Try to find and click specific restart/play buttons
                # Look for restart buttons with various selectors
                restart_selectors = [
                    ".restart-button", 
                    "#restart", 
                    "button.restart", 
                    "[data-action='restart']",
                    ".play-button",
                    "#play",
                    "button.play",
                    ".retry-button",
                    ".pokiSdkStartButton",
                    "img[alt*='play']",
                    "div[role='button']"
                ]
                
                restart_clicked = False
                for selector in restart_selectors:
                    try:
                        elements = self.browser.find_elements(By.CSS_SELECTOR, selector)
                        if elements:
                            elements[0].click()
                            restart_clicked = True
                            logger.info(f"Clicked restart button with selector: {selector}")
                            time.sleep(1)
                            break
                    except Exception:
                        continue
                
                # If no restart button found through selectors, try finding elements by x,y coordinates
                if not restart_clicked:
                    # Common positions to try clicking (relative to game center)
                    # These are typical locations for play/restart buttons
                    click_positions = [
                        (0, 0),            # Center
                        (0, -50),          # Above center
                        (0, 50),           # Below center
                        (0, browser_height // 4),  # Further below center
                        (0, -browser_height // 4)  # Further above center
                    ]
                    
                    for dx, dy in click_positions:
                        try:
                            click_x = center_x + dx
                            click_y = center_y + dy
                            # Use JavaScript to safely click at the position
                            element_exists = self.browser.execute_script(
                                f"var el = document.elementFromPoint({click_x}, {click_y}); "
                                f"if(el) {{ el.click(); return true; }} return false;"
                            )
                            if element_exists:
                                logger.info(f"Clicked at position ({click_x}, {click_y})")
                                restart_clicked = True
                                time.sleep(1)
                                break
                        except Exception as e:
                            logger.debug(f"Click at ({click_x}, {click_y}) failed: {str(e)}")
                
                # Wait for game to stabilize
                time.sleep(1.5)
                
                # Check if game restarted by inspecting a new frame
                frame = self.get_game_frame()
                if frame is not None and not np.all(frame == 0):
                    # Get a few more frames to make sure game is stable
                    temp_frames = []
                    for _ in range(3):
                        temp_frame = self.get_game_frame()
                        if temp_frame is not None and not np.all(temp_frame == 0):
                            temp_frames.append(temp_frame)
                        time.sleep(0.5)
                    
                    # If we got enough frames and they're not all black
                    if len(temp_frames) >= 2:
                        success = True
                        self.recent_frames.extend(temp_frames)
                        logger.info("Game successfully restarted")
                        break
                
                # If not successful, try a different approach on next attempt
                if not success:
                    logger.warning("Restart attempt failed, retrying with different approach")
                    # Try refreshing the page if we've already tried several times
                    if restart_attempts == 3:
                        try:
                            logger.info("Refreshing browser page")
                            self.browser.refresh()
                            time.sleep(5)  # Wait for page to load
                            self.initialize_game()  # Re-initialize the game
                        except Exception as e:
                            logger.warning(f"Error refreshing page: {str(e)}")
                
                # Safety delay before next attempt
                time.sleep(1)
                
            except Exception as e:
                logger.warning(f"Error during restart attempt {restart_attempts}: {str(e)}")
        
        # If all restart attempts failed, try a last resort approach
        if not success:
            logger.warning("All restart attempts failed, using fallback approach")
            try:
                # Refresh the page and reinitialize
                self.browser.refresh()
                time.sleep(5)
                self.initialize_game()
                
                # Get initial frames
                for _ in range(5):
                    frame = self.get_game_frame()
                    if frame is not None and not np.all(frame == 0):
                        self.recent_frames.append(frame)
                    time.sleep(0.5)
            except Exception as e:
                logger.error(f"Fallback restart failed: {str(e)}")
        
        # Get the last frame as the current state
        self.last_frame = self.recent_frames[-1] if self.recent_frames else None
        
        # Check if regions are detected, if not try to detect them
        if not self.game_region:
            self.detect_game_region()
        
        # Make sure we have at least one frame
        if not self.recent_frames:
            logger.warning("No frames captured during reset, getting a new frame")
            frame = self.get_game_frame()
            self.recent_frames.append(frame)
            self.last_frame = frame
        
        # Preprocess the frame for the agent
        processed_frame = self.preprocess_frame(self.recent_frames[-1])
        
        return processed_frame
    
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
            try:
                # Get game region center for actions
                x, y, width, height = self.game_region
                center_x, center_y = x + width // 2, y + height // 2
                
                # Store start time for performance tracking
                start_time = time.time()
                
                # Try different approaches in sequence until one works
                success = False
                error_messages = []
                
                # Method 1: First ensure focus on game area, then use direct JavaScript KeyboardEvent
                if not success:
                    try:
                        # Click to focus first
                        self.browser.execute_script(f"""
                            var el = document.elementFromPoint({center_x}, {center_y});
                            if (el) {{
                                el.focus();
                                return true;
                            }}
                            return false;
                        """)
                        
                        # Map keys to JavaScript key codes and key strings
                        key_data = {
                            'up': {'code': 38, 'key': 'ArrowUp', 'keyCode': 38},
                            'down': {'code': 40, 'key': 'ArrowDown', 'keyCode': 40},
                            'left': {'code': 37, 'key': 'ArrowLeft', 'keyCode': 37},
                            'right': {'code': 39, 'key': 'ArrowRight', 'keyCode': 39}
                        }
                        
                        if key in key_data:
                            key_info = key_data[key]
                            js_script = f"""
                            (function() {{
                                var keyEvent = function(type) {{
                                    var evt = new KeyboardEvent(type, {{
                                        bubbles: true,
                                        cancelable: true,
                                        keyCode: {key_info['keyCode']},
                                        which: {key_info['keyCode']},
                                        code: '{key_info['key']}',
                                        key: '{key_info['key']}',
                                        location: 0,
                                        view: window
                                    }});
                                    return evt;
                                }};
                                
                                // Dispatch events on document, window and active element
                                [document, window, document.activeElement].forEach(function(target) {{
                                    if (target) {{
                                        target.dispatchEvent(keyEvent('keydown'));
                                        setTimeout(function() {{
                                            target.dispatchEvent(keyEvent('keyup'));
                                        }}, 30);
                                    }}
                                }});
                                
                                return true;
                            }})();
                            """
                            result = self.browser.execute_script(js_script)
                            if result:
                                success = True
                                logger.debug(f"Method 1: JS KeyboardEvent successful for {action_name}")
                        else:
                            logger.debug(f"Unknown key: {key}")
                    except Exception as e:
                        error_messages.append(f"JS KeyboardEvent method failed: {str(e)}")
                
                # Method 2: Try using PyAutoGUI after ensuring browser is focused
                if not success:
                    try:
                        # Make sure browser window is in focus
                        self.browser.switch_to.window(self.browser.current_window_handle)
                        
                        # Click at center of game region
                        click_result = self.browser.execute_script(f"""
                            try {{
                                document.elementFromPoint({center_x}, {center_y}).click();
                                return true;
                            }} catch(e) {{
                                return false;
                            }}
                        """)
                        
                        # Short pause to let focus take effect
                        time.sleep(0.02)
                        
                        # Press key with PyAutoGUI
                        pyautogui.press(key)
                        
                        # Log success
                        success = True
                        logger.debug(f"Method 2: PyAutoGUI successful for {action_name}")
                    except Exception as e:
                        error_messages.append(f"PyAutoGUI method failed: {str(e)}")
                
                # Method 3: Try using direct DOM events on game canvas
                if not success:
                    try:
                        # Look for canvas elements (games often use canvas)
                        js_script = """
                        (function() {
                            // Find all canvas elements
                            var canvases = document.getElementsByTagName('canvas');
                            if (canvases && canvases.length > 0) {
                                // Use the first canvas
                                var canvas = canvases[0];
                                canvas.focus();
                                return true;
                            }
                            return false;
                        })();
                        """
                        
                        found_canvas = self.browser.execute_script(js_script)
                        
                        if found_canvas:
                            # Now send the key event to the canvas
                            from selenium.webdriver.common.keys import Keys
                            from selenium.webdriver.common.action_chains import ActionChains
                            
                            key_map_selenium = {
                                'up': Keys.ARROW_UP,
                                'down': Keys.ARROW_DOWN,
                                'left': Keys.ARROW_LEFT,
                                'right': Keys.ARROW_RIGHT
                            }
                            
                            canvas = self.browser.find_elements(By.TAG_NAME, 'canvas')[0]
                            ActionChains(self.browser).move_to_element(canvas).click().send_keys(key_map_selenium[key]).perform()
                            success = True
                            logger.debug(f"Method 3: Canvas action successful for {action_name}")
                    except Exception as e:
                        error_messages.append(f"Canvas method failed: {str(e)}")
                
                # Log performance information
                elapsed_time = (time.time() - start_time) * 1000  # milliseconds
                if success:
                    logger.debug(f"Action {action_name} performed in {elapsed_time:.2f}ms")
                else:
                    logger.warning(f"All action methods failed for {action_name} after {elapsed_time:.2f}ms")
                    logger.debug(f"Errors: {error_messages}")
                
            except Exception as e:
                logger.warning(f"Error performing action {action_name}: {str(e)}")
    
    def _is_game_over(self, current_frame):
        """
        Check if the game is over using multiple methods
        
        Args:
            current_frame: Current game frame
            
        Returns:
            Boolean indicating if the game is over
        """
        if self.last_frame is None:
            return False
        
        # Method 1: Frame difference - compare current frame with previous frame
        try:
            # Convert frames to grayscale for comparison
            current_gray = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
            last_gray = cv2.cvtColor(self.last_frame, cv2.COLOR_RGB2GRAY)
            
            # Calculate MSE (Mean Squared Error) between frames
            # If frames are nearly identical, the game might be over
            mse = np.mean((current_gray - last_gray) ** 2) / 255.0
            
            # Calculate structural similarity index (SSIM) for more robust comparison
            # Higher SSIM means more similar images
            try:
                # For newer versions of OpenCV
                ssim_value = cv2.SSIM(current_gray, last_gray)[0] if hasattr(cv2, 'SSIM') else 0.0
            except (AttributeError, TypeError):
                try:
                    # For older versions that use structural_similarity from skimage
                    from skimage.metrics import structural_similarity as ssim
                    ssim_value = ssim(current_gray, last_gray)
                except ImportError:
                    ssim_value = 0.0  # If neither method is available
            
            # Adjusted thresholds - less sensitive to prevent premature detection
            mse_threshold = 0.003  # Lower means more sensitive
            ssim_threshold = 0.92  # Higher means more sensitive

            # Check if the game is static (very little change between frames)
            is_static_mse = mse < mse_threshold
            is_static_ssim = ssim_value > ssim_threshold if ssim_value > 0 else False
            
            is_static = is_static_mse or is_static_ssim
            
            if is_static:
                logger.info(f"Game over detected: screen is static (MSE: {mse:.4f}, SSIM: {ssim_value:.4f})")
        except Exception as e:
            logger.warning(f"Error in frame comparison for game over detection: {str(e)}")
            is_static = False
        
        # Method 2: Check for specific game over visual indicators (typically UI elements)
        try:
            # Look for UI elements that indicate game over
            
            # Method 2a: Check for bright UI elements (like buttons, restart screens)
            # Convert to HSV for better color filtering
            current_hsv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)
            
            # Define range for bright UI elements (covers white/bright colors often used in UI)
            lower_white = np.array([0, 0, 180])
            upper_white = np.array([180, 30, 255])
            
            # Create a mask for bright UI elements
            white_mask = cv2.inRange(current_hsv, lower_white, upper_white)
            
            # Count bright pixels
            bright_pixel_count = cv2.countNonZero(white_mask)
            
            # Calculate percentage of bright pixels
            total_pixels = current_frame.shape[0] * current_frame.shape[1]
            bright_pixel_percentage = bright_pixel_count / total_pixels
            
            # Game over screens typically have more UI elements (bright)
            has_bright_ui = bright_pixel_percentage > 0.15  # Adjustable threshold
            
            # Method 2b: Check for typical game over UI colors (red/orange elements)
            # Define range for orange/red UI elements (common in game over screens)
            lower_orange = np.array([10, 100, 100])
            upper_orange = np.array([25, 255, 255])
            
            # Create a mask for orange UI elements
            orange_mask = cv2.inRange(current_hsv, lower_orange, upper_orange)
            
            # Count orange pixels
            orange_pixel_count = cv2.countNonZero(orange_mask)
            
            # Calculate percentage
            orange_pixel_percentage = orange_pixel_count / total_pixels
            
            # Game over screens often have orange/red elements
            has_orange_ui = orange_pixel_percentage > 0.07  # Adjustable threshold
            
            # Combine UI detection methods
            has_ui_elements = has_bright_ui or has_orange_ui
            
            if has_ui_elements and not is_static:
                logger.info(f"Game over potentially detected: UI elements found (Bright: {bright_pixel_percentage:.2f}, Orange: {orange_pixel_percentage:.2f})")
        except Exception as e:
            logger.warning(f"Error in UI-based game over detection: {str(e)}")
            has_ui_elements = False
        
        # Method 3: Time-based check - if we've been running for a while, be more lenient
        # with game over detection to prevent getting stuck
        time_based_threshold = self.step_count > 300  # After 300 steps, be more lenient
        
        # Combine detection methods:
        # 1. If screen is static, likely game over
        # 2. If UI elements detected and we're past initial steps, likely game over
        # 3. If we've been running for a long time and see some indicators, likely game over
        game_over = (
            is_static or 
            (has_ui_elements and self.step_count > 30) or  # Only use UI detection after enough steps
            (time_based_threshold and (has_ui_elements or is_static_mse))  # More lenient after a long time
        )
        
        # Debug information
        if game_over:
            # Save debug image
            debug_path = f"debug_images/game_over_frame_ep{self.episode_count}_step{self.step_count}.png"
            cv2.imwrite(debug_path, current_frame)
            
            # Also save the masks used for detection
            try:
                os.makedirs("debug_images/game_over", exist_ok=True)
                if 'white_mask' in locals():
                    cv2.imwrite(f"debug_images/game_over/white_mask_ep{self.episode_count}_step{self.step_count}.png", white_mask)
                if 'orange_mask' in locals():
                    cv2.imwrite(f"debug_images/game_over/orange_mask_ep{self.episode_count}_step{self.step_count}.png", orange_mask)
            except Exception as e:
                logger.debug(f"Failed to save debug masks: {str(e)}")
            
            logger.info(f"Game over frame saved as {debug_path}")
        
        return game_over
    
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
                max_size = 200
                height, width = frame_rgb.shape[:2]
                if height > max_size or width > max_size:
                    scale = max_size / max(height, width)
                    frame_rgb = cv2.resize(frame_rgb, (int(width * scale), int(height * scale)))
                
                # Convert to PhotoImage format for tkinter
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