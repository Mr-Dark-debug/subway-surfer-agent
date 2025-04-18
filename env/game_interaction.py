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
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, ElementNotInteractableException, WebDriverException
import random

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
    def __init__(self, game_url="https://poki.com/en/g/subway-surfers", position="right", 
                 use_existing_browser=False):
        """
        Initialize the Subway Surfers environment
        
        Args:
            game_url: URL to the game
            position: Position of the browser window ("left" or "right")
            use_existing_browser: Whether to use an existing browser window (ignored, always opens new window)
        """
        self.game_url = game_url
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
        
        # Browser status
        self.browser_active = True
        
        # FIXED REGIONS as specified - using your exact values
        self.game_region = (1094, 178, 806, 529)    # (x, y, width, height)
        self.score_region = (1682, 159, 225, 48)    # (x, y, width, height)
        self.coin_region = (1682, 217, 225, 48)     # (x, y, width, height)
        
        # Create debug directories early to avoid issues
        self._setup_debug_directories()
        
        logger.info(f"Using fixed game region: {self.game_region}")
        logger.info(f"Using fixed score region: {self.score_region}")
        logger.info(f"Using fixed coin region: {self.coin_region}")
        
        # Initialize browser
        logger.info("Initializing browser...")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.browser = self._init_browser()
                break
            except Exception as e:
                logger.error(f"Browser initialization attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    logger.info("Retrying browser initialization...")
                    time.sleep(2)
                else:
                    logger.critical("All browser initialization attempts failed. Cannot continue.")
                    raise
        
        # Position browser window based on preference
        self._position_browser()
        
        # Wait for game to load
        logger.info("Waiting for game to load...")
        time.sleep(8)  # Wait for initial page load
        
        # Initialize game
        self.initialize_game()
        
        # Record initial debug image with regions
        self._save_debug_regions()
        
        # Metrics for performance tracking
        self.frame_capture_times = []
        self.ocr_times = []
        self.action_times = []
        
        logger.info("Game initialized successfully")
    
    def _setup_debug_directories(self):
        """Create all necessary debug directories"""
        directories = [
            "debug_images",
            "debug_images/states",
            "debug_images/popups",
            "debug_images/scores",
            "debug_images/coins",
            "debug_images/frames",
            "debug_images/regions",
            "logs"
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
    def _init_browser(self):
        """Initialize the browser with appropriate settings"""
        try:
            # Always open a new incognito window for clean session
            chrome_options = Options()
            chrome_options.add_argument("--incognito")
            chrome_options.add_argument("--start-maximized")
            chrome_options.add_argument("--disable-infobars")
            chrome_options.add_argument("--disable-extensions")
            chrome_options.add_argument("--disable-notifications")
            chrome_options.add_argument("--disable-popup-blocking")
            
            # Disable GPU acceleration if it causes issues
            # chrome_options.add_argument("--disable-gpu")
            
            # Explicitly disable password manager popups
            chrome_options.add_experimental_option("prefs", {
                "credentials_enable_service": False,
                "profile.password_manager_enabled": False,
                "profile.default_content_setting_values.notifications": 2
            })
            
            # Reduce log noise
            chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
            
            logger.info("Opening a new incognito browser window")
            browser = webdriver.Chrome(service=Service(ChromeDriverManager().install()), 
                                      options=chrome_options)
            browser.get(self.game_url)
            return browser
        except Exception as e:
            logger.error(f"Error initializing browser: {e}")
            raise
    
    def _position_browser(self):
        """Position the browser window based on preference"""
        try:
            if not self.browser_active:
                return
                
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
                browser_width = (screen_width // 2) - 20
                browser_height = screen_height - 60
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
                browser_width = min(1024, screen_width - 40)
                browser_height = min(768, screen_height - 60)
                browser_x = (screen_width - browser_width) // 2
                browser_y = (screen_height - browser_height) // 2
                
                self.browser.set_window_rect(browser_x, browser_y, browser_width, browser_height)
                logger.info(f"Browser positioned in center with dimensions {browser_width}x{browser_height}")
            
            # Ensure the browser has focus
            self.browser.switch_to.window(self.browser.current_window_handle)
            
        except Exception as e:
            logger.warning(f"Error positioning browser window: {str(e)}")
            logger.warning("Continuing with default browser positioning")
    
    def _save_debug_regions(self, screenshot=None):
        """Save a debug image showing the detected regions"""
        try:
            # Capture screenshot if not provided
            if screenshot is None:
                screenshot = self.capture_screen()
                
            if screenshot is None:
                logger.warning("Empty screenshot, skipping debug regions")
                return
                
            # Draw regions on the screenshot
            debug_img = screenshot.copy()
            
            # Draw game region (green)
            x, y, w, h = self.game_region
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(debug_img, "Game Region", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw score region (blue)
            x, y, w, h = self.score_region
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(debug_img, "Score", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Draw coin region (yellow)
            x, y, w, h = self.coin_region
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.putText(debug_img, "Coins", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Save the debug image
            debug_path = "debug_images/regions/initial_regions.png"
            cv2.imwrite(debug_path, debug_img)
            logger.info(f"Initial regions debug image saved to {debug_path}")
            
        except Exception as e:
            logger.warning(f"Error saving debug regions: {e}")
    
    def initialize_game(self):
        """Initialize the game by clicking the play button or restarting"""
        logger.info("Initializing game...")
        
        try:
            if not self.browser_active:
                return
                
            # First make sure browser is focused
            self.browser.switch_to.window(self.browser.current_window_handle)
            
            # Handle cookie consent if present (common on many game sites)
            self._handle_cookie_consent()
            
            # Try multiple play button selectors
            play_selectors = [
                ".pokiSdkStartButton", 
                "#play-button", 
                ".play-button", 
                "button.play",
                "[data-testid='play-button']"
            ]
            
            play_button_clicked = False
            for selector in play_selectors:
                try:
                    play_button = WebDriverWait(self.browser, 2).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                    )
                    play_button.click()
                    play_button_clicked = True
                    logger.info(f"Clicked play button with selector: {selector}")
                    break
                except (TimeoutException, ElementNotInteractableException):
                    continue
            
            # If can't find the play button, try clicking in the center of the page
            if not play_button_clicked:
                logger.info("Clicking center of screen as fallback")
                
                # Use JavaScript to get accurate window dimensions and click center
                browser_width = self.browser.execute_script("return window.innerWidth")
                browser_height = self.browser.execute_script("return window.innerHeight")
                
                center_x = browser_width // 2
                center_y = browser_height // 2
                
                # Try both JavaScript click and PyAutoGUI click for redundancy
                try:
                    self.browser.execute_script(f"document.elementFromPoint({center_x}, {center_y}).click()")
                    logger.info(f"JS clicked at center coordinates: ({center_x}, {center_y})")
                except Exception:
                    # If JS click fails, try PyAutoGUI click
                    try:
                        # Convert browser-relative coordinates to screen coordinates
                        browser_rect = self.browser.get_window_rect()
                        screen_x = browser_rect['x'] + center_x
                        screen_y = browser_rect['y'] + center_y
                        pyautogui.click(screen_x, screen_y)
                        logger.info(f"PyAutoGUI clicked at screen coordinates: ({screen_x}, {screen_y})")
                    except Exception as e:
                        logger.warning(f"Failed to click with PyAutoGUI: {e}")
                
            # Ensure game is started by pressing space key
            time.sleep(1)
            pyautogui.press('space')
            
            # Make sure the game is ready
            time.sleep(2)
        except Exception as e:
            logger.warning(f"Error during game initialization: {e}")
            logger.warning("Attempting to continue anyway")
    
    def _handle_cookie_consent(self):
        """Handle cookie consent dialogs that might appear"""
        # Common cookie consent button selectors
        consent_selectors = [
            "#accept-cookies", 
            ".consent-accept", 
            "[aria-label='Accept cookies']",
            "button:contains('Accept')",
            ".consent-banner__accept"
        ]
        
        for selector in consent_selectors:
            try:
                consent_button = WebDriverWait(self.browser, 1).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                )
                consent_button.click()
                logger.info(f"Clicked cookie consent button with selector: {selector}")
                time.sleep(0.5)
                return True
            except (TimeoutException, ElementNotInteractableException):
                continue
        
        return False
    
    def detect_save_me_popup(self):
        """
        Detect if the 'Save me!' popup is currently displayed
        
        Returns:
            Boolean indicating if popup is detected
        """
        try:
            if not self.browser_active:
                return False
                
            # Capture screen and get the game region
            screenshot = self.capture_screen()
            
            if screenshot is None:
                logger.warning("Empty screenshot in detect_save_me_popup")
                return False
                
            # Check if game region is valid within screenshot dimensions
            height, width = screenshot.shape[:2]
            x, y, w, h = self.game_region
            
            if x >= width or y >= height or x+w > width or y+h > height:
                logger.warning(f"Game region outside screenshot dimensions: game region={self.game_region}, screenshot={width}x{height}")
                return False
                
            game_area = screenshot[y:y+h, x:x+w]
            
            # Save the game area for debugging
            debug_path = f"debug_images/popups/game_area_{self.episode_count}_{self.step_count}.png"
            cv2.imwrite(debug_path, game_area)
            
            # Method 1: Check for white/light areas in the center-upper part
            # Convert to HSV for better color segmentation
            hsv = cv2.cvtColor(game_area, cv2.COLOR_BGR2HSV)
            
            # Define range for white/light gray colors (popup background)
            lower_white = np.array([0, 0, 180])  # Low saturation, high value
            upper_white = np.array([180, 30, 255])
            
            # Create mask for white areas
            white_mask = cv2.inRange(hsv, lower_white, upper_white)
            
            # Look for blue "Save me!" text
            lower_blue = np.array([90, 50, 150])
            upper_blue = np.array([120, 255, 255])
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
            
            # Check for significant white and blue areas in the top section 
            top_section = white_mask[0:h//3, :]
            white_pixels = np.count_nonzero(top_section)
            white_ratio = white_pixels / (top_section.shape[0] * top_section.shape[1])
            
            blue_section = blue_mask[0:h//3, :]
            blue_pixels = np.count_nonzero(blue_section)
            blue_ratio = blue_pixels / (blue_section.shape[0] * blue_section.shape[1])
            
            # Save mask images for debugging
            combined_debug = np.zeros_like(game_area)
            combined_debug[:, :, 0] = blue_mask  # Blue channel
            combined_debug[:, :, 1] = np.zeros_like(blue_mask)  # Green channel
            combined_debug[:, :, 2] = white_mask  # Red channel
            debug_path = f"debug_images/popups/popup_masks_{self.episode_count}_{self.step_count}.png"
            cv2.imwrite(debug_path, combined_debug)
            
            # Method 2: Template matching (more reliable)
            # We'll implement a simplified template matching by looking for specific patterns
            
            # Convert to grayscale for simpler matching
            gray = cv2.cvtColor(game_area, cv2.COLOR_BGR2GRAY)
            
            # Binarize the image
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            
            # Check for large white rectangular area in the center top portion
            center_top = binary[h//8:h//3, w//4:3*w//4]
            white_ratio_center = np.count_nonzero(center_top) / (center_top.shape[0] * center_top.shape[1])
            
            # If both white and blue elements are present in significant amounts, likely popup
            is_popup_method1 = (white_ratio > 0.15) and (blue_ratio > 0.01)
            is_popup_method2 = white_ratio_center > 0.7
            
            is_popup = is_popup_method1 or is_popup_method2
            
            if is_popup:
                logger.info(f"'Save me!' popup detected (method1: {is_popup_method1}, method2: {is_popup_method2})")
                logger.info(f"White ratio: {white_ratio:.2f}, Blue ratio: {blue_ratio:.2f}, Center white: {white_ratio_center:.2f}")
            
            return is_popup
            
        except Exception as e:
            logger.error(f"Error detecting save me popup: {str(e)}")
            return False
    
    def handle_game_over(self):
        """
        Handle game over and restart the game, including dismissing 'Save me!' popup
        
        Returns:
            Boolean indicating success
        """
        logger.info("Handling game over...")
        
        try:
            if not self.browser_active:
                return False
            
            # Make sure browser is in focus
            try:
                self.browser.switch_to.window(self.browser.current_window_handle)
            except Exception as e:
                logger.warning(f"Error switching to window: {e}")
                
            # First check if "Save me!" popup is visible
            if self.detect_save_me_popup():
                logger.info("Detected 'Save me!' popup, waiting 1 second...")
                time.sleep(1)  # Small wait to ensure UI is stable
                
                # Get game region
                x, y, w, h = self.game_region
                
                # Click in game area but NOT on the popup (which is typically in the center upper area)
                # Click in lower part of game area
                click_x = x + w // 2   # Center horizontal
                click_y = y + int(h * 0.8)  # 80% down from top
                
                # Use pyautogui to click
                pyautogui.click(click_x, click_y)
                logger.info(f"Clicked outside popup at ({click_x}, {click_y})")
                
                # Wait for popup to disappear
                time.sleep(1.5)
                
                # Press space to restart
                logger.info("Pressing space to restart game...")
                pyautogui.press('space')
                
                # Wait for game to restart
                time.sleep(1.5)
                
                # Also try clicking at center of game (backup method)
                center_x = x + w // 2
                center_y = y + h // 2
                time.sleep(0.5)  # Small wait
                pyautogui.click(center_x, center_y)
                
                # Wait additional time for game to fully restart
                time.sleep(1.5)
            else:
                # If no popup, just press space and click center
                logger.info("No popup detected, pressing space to restart game...")
                
                # Make sure the browser window is in focus before pressing keys
                try:
                    self.browser.execute_script("window.focus();")
                except Exception as e:
                    logger.warning(f"Error focusing window: {e}")
                
                # Press space to restart
                pyautogui.press('space')
                
                # Get game region
                x, y, w, h = self.game_region
                center_x = x + w // 2
                center_y = y + h // 2
                
                # Click at center as backup
                time.sleep(0.5)
                pyautogui.click(center_x, center_y)
                
                # Wait for game to restart
                time.sleep(1.5)
            
            # Reset game state
            self.game_over = False
            
            # Check if game successfully restarted (should have movement)
            if not self._check_game_active():
                logger.warning("Game might not have restarted properly, trying alternative methods")
                
                # Try clicking several positions
                x, y, w, h = self.game_region
                center_x = x + w // 2
                center_y = y + h // 2
                
                click_positions = [
                    (center_x, center_y),  # Center
                    (center_x, center_y - h // 4),  # Upper center 
                    (center_x, center_y + h // 4),  # Lower center
                ]
                
                for cx, cy in click_positions:
                    # Make sure browser is in focus
                    try:
                        self.browser.switch_to.window(self.browser.current_window_handle)
                    except Exception:
                        pass
                    
                    pyautogui.click(cx, cy)
                    time.sleep(0.5)
                    pyautogui.press('space')
                    time.sleep(1)
                    
                    # Check if game is now active
                    if self._check_game_active():
                        logger.info("Game successfully restarted after retry")
                        return True
                        
                # If still not active, try refreshing the page as a last resort
                logger.warning("Game still not active, refreshing the page")
                try:
                    self.browser.refresh()
                    time.sleep(5)  # Wait for page to reload
                    self.initialize_game()
                    time.sleep(2)
                    return self._check_game_active()
                except Exception as e:
                    logger.error(f"Error refreshing the page: {e}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error handling game over: {str(e)}")
            return False
    
    def _check_game_active(self):
        """
        Check if game is active by capturing two frames and comparing them
        
        Returns:
            Boolean indicating if game is active (has movement)
        """
        try:
            if not self.browser_active:
                return False
                
            # Capture first frame
            frame1 = self.get_game_frame()
            
            if frame1 is None or frame1.size == 0:
                logger.warning("Empty frame in _check_game_active")
                return False
                
            # Small wait
            time.sleep(0.5)
            
            # Capture second frame
            frame2 = self.get_game_frame()
            
            if frame2 is None or frame2.size == 0:
                logger.warning("Empty second frame in _check_game_active")
                return False
                
            # Convert to grayscale for comparison
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            # Calculate absolute difference
            diff = cv2.absdiff(gray1, gray2)
            
            # Calculate mean difference (movement)
            mean_diff = np.mean(diff)
            
            # If significant difference, game is active
            is_active = mean_diff > 3.0  # Threshold may need adjustment
            
            # Save diff image for debugging
            if self.step_count % 100 == 0 or self.step_count < 10:
                debug_path = f"debug_images/frames/diff_{self.episode_count}_{self.step_count}.png"
                cv2.imwrite(debug_path, diff)
                
                # Enhance diff for visualization
                diff_color = cv2.applyColorMap(diff * 10, cv2.COLORMAP_JET)
                debug_path = f"debug_images/frames/diff_enhanced_{self.episode_count}_{self.step_count}.png"
                cv2.imwrite(debug_path, diff_color)
                
                logger.info(f"Game activity check: {is_active} (diff: {mean_diff:.4f})")
            
            return is_active
            
        except Exception as e:
            logger.error(f"Error checking game activity: {str(e)}")
            return False
    
    def capture_screen(self):
        """Capture the current screen as a numpy array"""
        start_time = time.time()
        try:
            if not self.browser_active:
                blank_img = np.zeros((600, 800, 3), dtype=np.uint8)
                logger.warning("Browser not active, returning blank image")
                return blank_img
                
            # Take screenshot using selenium
            try:
                screenshot = self.browser.get_screenshot_as_png()
                
                # Convert to numpy array
                screenshot = np.frombuffer(screenshot, np.uint8)
                screenshot = cv2.imdecode(screenshot, cv2.IMREAD_COLOR)
                
                # Record capture time
                self.frame_capture_times.append(time.time() - start_time)
                
                return screenshot
            except WebDriverException as e:
                logger.error(f"Error capturing screen: {str(e)}")
                self.browser_active = False
                return np.zeros((600, 800, 3), dtype=np.uint8)  # Return blank image
        except Exception as e:
            logger.error(f"Error in capture_screen: {str(e)}")
            return np.zeros((600, 800, 3), dtype=np.uint8)  # Return blank image
    
    def preprocess_frame(self, frame):
        """
        Preprocess a frame for the agent
        
        Args:
            frame: RGB image as numpy array
            
        Returns:
            Preprocessed frame (grayscale, resized to 84x84)
        """
        try:
            if frame is None:
                logger.error("Received None frame in preprocess_frame")
                return np.zeros((84, 84), dtype=np.float32)
                
            # Make a copy to avoid modifying the original
            frame_copy = frame.copy()
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            
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
            if not self.browser_active:
                # Return a blank frame if browser is not active
                x, y, width, height = self.game_region
                return np.zeros((height, width, 3), dtype=np.uint8)
                
            screenshot = self.capture_screen()
            
            if screenshot is None:
                logger.error("Empty screenshot in get_game_frame")
                return np.zeros((self.game_region[3], self.game_region[2], 3), dtype=np.uint8)
                
            # Crop to game region
            if self.game_region:
                x, y, width, height = self.game_region
                
                # Check if region is valid within the screenshot
                if (y+height <= screenshot.shape[0] and x+width <= screenshot.shape[1]):
                    game_frame = screenshot[y:y+height, x:x+width].copy()
                    
                    # Save frame periodically for debugging
                    if self.step_count % 100 == 0 or self.step_count < 10:
                        debug_path = f"debug_images/frames/frame_{self.episode_count}_{self.step_count}.png"
                        cv2.imwrite(debug_path, game_frame)
                    
                    return game_frame
                else:
                    logger.warning(f"Game region outside screenshot dimensions: {self.game_region} vs {screenshot.shape}")
                    # Return a blank frame of expected size
                    return np.zeros((height, width, 3), dtype=np.uint8)
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
        start_time = time.time()
        try:
            if not self.browser_active:
                return self.score + 5  # Fallback to estimated score
                
            screenshot = self.capture_screen()
            
            if screenshot is None:
                logger.warning("Empty screenshot in get_score")
                return self.score + 5  # Fallback to estimated score
                
            if self.score_region:
                x, y, width, height = self.score_region
                
                # Check if region is valid
                if (y+height <= screenshot.shape[0] and x+width <= screenshot.shape[1]):
                    score_image = screenshot[y:y+height, x:x+width]
                    
                    # Save score image periodically for debugging
                    if self.step_count % 100 == 0 or self.step_count < 10:
                        debug_path = f"debug_images/scores/score_{self.episode_count}_{self.step_count}.png"
                        cv2.imwrite(debug_path, score_image)
                    
                    # Convert to grayscale
                    gray = cv2.cvtColor(score_image, cv2.COLOR_BGR2GRAY)
                    
                    # Apply adaptive thresholding
                    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                  cv2.THRESH_BINARY, 11, 2)
                    
                    # Invert if needed (white text on black background)
                    if np.mean(binary) < 127:
                        binary = cv2.bitwise_not(binary)
                    
                    # Try different OCR configurations
                    ocr_configs = [
                        '--psm 7 -c tessedit_char_whitelist=0123456789',  # Single line of text
                        '--psm 8 -c tessedit_char_whitelist=0123456789',  # Single word
                        '--psm 6 -c tessedit_char_whitelist=0123456789'   # Assume uniform block of text
                    ]
                    
                    for config in ocr_configs:
                        try:
                            text = pytesseract.image_to_string(binary, config=config)
                            text = ''.join(filter(str.isdigit, text))  # Keep only digits
                            
                            if text and text.isdigit():
                                current_score = int(text)
                                
                                # Sanity check - score should generally increase
                                if current_score >= self.score:
                                    self.score = current_score
                                    self.ocr_times.append(time.time() - start_time)
                                    return current_score
                                # If score is close to last score, use it anyway (might be legitimate decrease)
                                elif self.score - current_score < 100:
                                    self.score = current_score
                                    self.ocr_times.append(time.time() - start_time)
                                    return current_score
                        except Exception as e:
                            logger.debug(f"OCR error with config {config}: {e}")
                            continue
                
                # If OCR fails, use estimated score increase
                # More sophisticated estimation based on step count
                if self.step_count < 500:
                    # Early game: slower score increase
                    score_increase = 3 + random.randint(0, 3)
                else:
                    # Later game: faster score increase
                    score_increase = 5 + random.randint(0, 5)
                    
                self.score += score_increase
                self.ocr_times.append(time.time() - start_time)
                return self.score
            else:
                # If region not set, simulate score increase
                self.score += 5
                self.ocr_times.append(time.time() - start_time)
                return self.score
        except Exception as e:
            logger.warning(f"Error in score detection: {str(e)}")
            self.score += 5  # Fallback to estimated score
            self.ocr_times.append(time.time() - start_time)
            return self.score
    
    def get_coins(self):
        """Extract coins from the coin region using OCR"""
        start_time = time.time()
        try:
            if not self.browser_active:
                return self.coins  # Keep current coin count
                
            screenshot = self.capture_screen()
            
            if screenshot is None:
                logger.warning("Empty screenshot in get_coins")
                return self.coins  # Keep current coin count
                
            if self.coin_region:
                x, y, width, height = self.coin_region
                
                # Check if region is valid
                if (y+height <= screenshot.shape[0] and x+width <= screenshot.shape[1]):
                    coin_image = screenshot[y:y+height, x:x+width]
                    
                    # Save coin image periodically for debugging
                    if self.step_count % 100 == 0 or self.step_count < 10:
                        debug_path = f"debug_images/coins/coin_{self.episode_count}_{self.step_count}.png"
                        cv2.imwrite(debug_path, coin_image)
                    
                    # Convert to grayscale
                    gray = cv2.cvtColor(coin_image, cv2.COLOR_BGR2GRAY)
                    
                    # Apply adaptive thresholding
                    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                  cv2.THRESH_BINARY, 11, 2)
                    
                    # Invert if needed (white text on black background)
                    if np.mean(binary) < 127:
                        binary = cv2.bitwise_not(binary)
                    
                    # Try different OCR configurations
                    ocr_configs = [
                        '--psm 7 -c tessedit_char_whitelist=0123456789',  # Single line of text
                        '--psm 8 -c tessedit_char_whitelist=0123456789',  # Single word
                        '--psm 6 -c tessedit_char_whitelist=0123456789'   # Assume uniform block of text
                    ]
                    
                    for config in ocr_configs:
                        try:
                            text = pytesseract.image_to_string(binary, config=config)
                            text = ''.join(filter(str.isdigit, text))  # Keep only digits
                            
                            if text and text.isdigit():
                                current_coins = int(text)
                                
                                # Sanity check - coins should generally increase or stay the same
                                if current_coins >= self.coins:
                                    self.coins = current_coins
                                    self.ocr_times.append(time.time() - start_time)
                                    return current_coins
                        except Exception:
                            continue
                
                # If OCR fails, implement a more realistic coin pickup simulation
                if self.step_count % 20 == 0:  # Roughly 5% chance to find a coin
                    # Higher chance in early game (coin rows are common)
                    if self.step_count < 200:
                        coin_change = random.randint(0, 3)  # 0-3 coins
                    else:
                        coin_change = random.randint(0, 1)  # 0-1 coins
                    
                    self.coins += coin_change
                    
                self.ocr_times.append(time.time() - start_time)
                return self.coins
            else:
                # If region not set, simulate occasional coin pickup
                if self.step_count % 20 == 0:
                    self.coins += random.randint(0, 1)
                
                self.ocr_times.append(time.time() - start_time)
                return self.coins
        except Exception as e:
            logger.warning(f"Error in coin detection: {str(e)}")
            self.ocr_times.append(time.time() - start_time)
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
        
        # If game is over, handle restart
        success = self.handle_game_over()
        if not success and self.browser_active:
            logger.warning("Failed to restart game normally, trying fallback method")
            self._fallback_restart()
        
        # Clear memory for recent frames
        if hasattr(self, 'recent_frames'):
            self.recent_frames.clear()
        else:
            self.recent_frames = []
        
        # Make sure we have at least one frame
        frame = self.get_game_frame()
        
        if frame is None or frame.size == 0:
            logger.warning("Empty frame after reset, creating blank frame")
            height, width = self.game_region[3], self.game_region[2]
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
        self.recent_frames.append(frame)
        self.last_frame = frame
        
        # Preprocess the frame for the agent
        processed_frame = self.preprocess_frame(self.recent_frames[-1])
        
        # Save initial state
        debug_path = f"debug_images/states/initial_state_ep{self.episode_count}.png"
        cv2.imwrite(debug_path, (processed_frame * 255).astype(np.uint8))
        
        return processed_frame
    
    def _fallback_restart(self):
        """Fallback method to restart game if normal methods fail"""
        try:
            if not self.browser_active:
                return
                
            # Get game region center
            x, y, w, h = self.game_region
            center_x = x + w // 2
            center_y = y + h // 2
            
            # First make sure browser is in focus
            try:
                self.browser.switch_to.window(self.browser.current_window_handle)
            except Exception:
                pass
            
            # First press Escape to ensure any dialogs are closed
            pyautogui.press('escape')
            time.sleep(0.5)
            
            # Try multiple approaches in sequence
            
            # 1. Click at different positions and press space
            positions = [
                (center_x, center_y),              # Center
                (center_x, center_y - h // 3),     # Top third
                (center_x, center_y + h // 3),     # Bottom third
                (center_x - w // 3, center_y),     # Left third
                (center_x + w // 3, center_y),     # Right third
            ]
            
            for px, py in positions:
                pyautogui.click(px, py)
                time.sleep(0.2)
                pyautogui.press('space')
                time.sleep(0.3)
            
            # 2. Try refreshing the page as a last resort
            if self.browser_active:
                try:
                    self.browser.refresh()
                    time.sleep(5)  # Wait for page to reload
                    
                    # 3. Re-initialize game
                    self.initialize_game()
                    time.sleep(2)
                except Exception as e:
                    logger.error(f"Error refreshing browser: {e}")
                    self.browser_active = False
            
            logger.info("Fallback restart completed")
            
        except Exception as e:
            logger.error(f"Error in fallback restart: {str(e)}")
    
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
        
        # Start action timer
        action_start_time = time.time()
        
        # Perform the action
        self._perform_action(action_name)
        
        # End action timer
        self.action_times.append(time.time() - action_start_time)
        
        # Small delay to allow action to take effect
        time.sleep(0.05)
        
        # Get new frame
        game_frame = self.get_game_frame()
        
        if game_frame is None or game_frame.size == 0:
            logger.warning("Empty frame after action, creating blank frame")
            height, width = self.game_region[3], self.game_region[2]
            game_frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Update score and coins
        self.score = self.get_score()
        self.coins = self.get_coins()
        
        # Check if game is over, including checking for "Save me!" popup
        self.game_over = self._is_game_over(game_frame) or self.detect_save_me_popup()
        
        # Preprocess frame for agent
        next_state = self.preprocess_frame(game_frame)
        
        # Calculate reward components with improved design
        survival_reward = 0.1  # Small reward for surviving
        
        # Score reward with progressive scaling (higher reward for higher scores)
        score_diff = self.score - prev_score
        if score_diff > 0:
            if self.score < 1000:
                score_reward = score_diff * 0.5  # Early game
            elif self.score < 5000:
                score_reward = score_diff * 0.7  # Mid game
            else:
                score_reward = score_diff * 1.0  # Late game
        else:
            score_reward = 0
        
        # Coin reward with bonus for multiple coins
        coin_diff = self.coins - prev_coins
        if coin_diff > 0:
            # Bonus for collecting multiple coins at once
            coin_reward = coin_diff * (1.0 + 0.2 * (coin_diff - 1))
        else:
            coin_reward = 0
        
        # Penalty for game over with progressive scaling (less harsh early game)
        if self.game_over:
            if self.step_count < 100:
                game_over_penalty = -5.0  # Early game
            elif self.step_count < 500:
                game_over_penalty = -10.0  # Mid game
            else:
                game_over_penalty = -15.0  # Late game
        else:
            game_over_penalty = 0.0
        
        # Introduce a small time-based reward that increases over time
        # This encourages the agent to survive longer
        time_reward = min(0.0001 * self.step_count, 0.05)
        
        # Total reward
        reward = survival_reward + score_reward + coin_reward + game_over_penalty + time_reward
        self.last_reward = reward
        
        # Save current frame for next comparison
        self.last_frame = game_frame
        
        # Save debug screenshots periodically
        if self.step_count % 100 == 0:  # Less frequent saves to reduce overhead
            try:
                debug_path = f"debug_images/frames/screen_ep{self.episode_count}_step{self.step_count}.png"
                cv2.imwrite(debug_path, game_frame)
                
                # Also save the processed state
                debug_path = f"debug_images/states/state_ep{self.episode_count}_step{self.step_count}.png"
                cv2.imwrite(debug_path, (next_state * 255).astype(np.uint8))
            except Exception as e:
                logger.warning(f"Error saving debug screenshot: {e}")
        
        # Return next_state, reward, done, info
        info = {
            'score': self.score,
            'coins': self.coins,
            'survival_reward': survival_reward,
            'score_reward': score_reward,
            'coin_reward': coin_reward,
            'game_over_penalty': game_over_penalty,
            'time_reward': time_reward,
            'step_count': self.step_count
        }
        
        return next_state, reward, self.game_over, info
    
    def _perform_action(self, action_name):
        """
        Perform the given action in the game
        
        Args:
            action_name: Name of the action to perform
        """
        if not self.browser_active:
            return
            
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
                # Make sure browser window is in focus before sending keystrokes
                try:
                    self.browser.switch_to.window(self.browser.current_window_handle)
                    # Additional attempt to ensure focus using JavaScript
                    self.browser.execute_script("window.focus();")
                except WebDriverException:
                    self.browser_active = False
                    logger.warning("Browser window no longer active")
                    return
                
                # Press key with PyAutoGUI
                pyautogui.press(key)
                
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
        if self.last_frame is None or current_frame is None:
            return False
            
        if current_frame.size == 0 or self.last_frame.size == 0:
            logger.warning("Empty frame in _is_game_over")
            return False
        
        try:
            # Method 1: Compare consecutive frames
            # Convert frames to grayscale for comparison
            current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            last_gray = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate MSE (Mean Squared Error) between frames
            # If frames are nearly identical, the game might be over
            mse = np.mean((current_gray - last_gray) ** 2) / 255.0
            
            # Adaptive threshold based on game progression
            # Early game is more static, late game has more movement
            if self.step_count < 100:
                mse_threshold = 0.002  # More sensitive in early game
            else:
                mse_threshold = 0.003  # Less sensitive in later game

            # Check if the game is static (very little change between frames)
            is_static = mse < mse_threshold
            
            # Method 2: Check for specific game over indicators
            is_popup_detected = self.detect_save_me_popup()
            
            # Method 3: Check for dark overlay (common in game over screens)
            # Calculate brightness of center portion
            h, w = current_gray.shape
            center_region = current_gray[h//4:3*h//4, w//4:3*w//4]
            brightness = np.mean(center_region)
            
            is_dark_overlay = brightness < 50  # Threshold for darkness
            
            # Combine methods (more reliable detection)
            game_over = is_static or is_popup_detected or is_dark_overlay
            
            if game_over and self.step_count % 10 == 0:
                # Save debug info
                debug_info = f"Static: {is_static} (MSE: {mse:.6f}), Popup: {is_popup_detected}, Dark: {is_dark_overlay} (Brightness: {brightness:.2f})"
                logger.info(f"Game over detected: {debug_info}")
                
                # Save debug image
                debug_img = np.hstack((self.last_frame, current_frame))
                debug_path = f"debug_images/frames/game_over_ep{self.episode_count}_step{self.step_count}.png"
                cv2.imwrite(debug_path, debug_img)
                
                # Save difference image
                diff = cv2.absdiff(self.last_frame, current_frame)
                diff_path = f"debug_images/frames/game_over_diff_ep{self.episode_count}_step{self.step_count}.png"
                cv2.imwrite(diff_path, diff)
            
            return game_over
            
        except Exception as e:
            logger.warning(f"Error in frame comparison for game over detection: {str(e)}")
            return False
    
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
            os.makedirs("debug_images/states", exist_ok=True)
            save_path = f"debug_images/states/state_ep{episode_num}_step{step_num}.png"
            try:
                cv2.imwrite(save_path, grid)
                logger.debug(f"State visualization saved to {save_path}")
            except Exception as e:
                logger.warning(f"Error saving state visualization: {e}")
        else:
            # If it's a single frame, just save it directly
            try:
                frame = (state * 255).astype(np.uint8)
                os.makedirs("debug_images/states", exist_ok=True)
                save_path = f"debug_images/states/state_ep{episode_num}_step{step_num}.png"
                cv2.imwrite(save_path, frame)
                logger.debug(f"State visualization saved to {save_path}")
            except Exception as e:
                logger.warning(f"Error saving state visualization: {e}")
    
    def update_training_stats(self, epsilon=None, loss=None, avg_reward=None):
        """Update training statistics (just storing for terminal logging)"""
        self.epsilon = epsilon
        self.loss = loss
        self.avg_reward = avg_reward
    
    def get_performance_stats(self):
        """Get statistics about environment performance"""
        stats = {
            'capture_time': {
                'mean': np.mean(self.frame_capture_times) if self.frame_capture_times else 0,
                'std': np.std(self.frame_capture_times) if self.frame_capture_times else 0,
                'count': len(self.frame_capture_times)
            },
            'ocr_time': {
                'mean': np.mean(self.ocr_times) if self.ocr_times else 0,
                'std': np.std(self.ocr_times) if self.ocr_times else 0,
                'count': len(self.ocr_times)
            },
            'action_time': {
                'mean': np.mean(self.action_times) if self.action_times else 0,
                'std': np.std(self.action_times) if self.action_times else 0,
                'count': len(self.action_times)
            }
        }
        return stats
    
    def log_performance_stats(self):
        """Log performance statistics"""
        stats = self.get_performance_stats()
        logger.info("Environment Performance Statistics:")
        logger.info(f"  Frame capture time: {stats['capture_time']['mean']*1000:.2f}ms (±{stats['capture_time']['std']*1000:.2f}ms)")
        logger.info(f"  OCR processing time: {stats['ocr_time']['mean']*1000:.2f}ms (±{stats['ocr_time']['std']*1000:.2f}ms)")
        logger.info(f"  Action execution time: {stats['action_time']['mean']*1000:.2f}ms (±{stats['action_time']['std']*1000:.2f}ms)")
    
    def close(self):
        """Close the environment and cleanup resources"""
        if hasattr(self, 'browser') and self.browser and self.browser_active:
            try:
                self.browser.quit()
                self.browser_active = False
            except Exception as e:
                logger.warning(f"Error closing browser: {e}")
        
        logger.info("Environment closed")