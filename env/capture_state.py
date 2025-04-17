# env/capture_state.py
import cv2
import numpy as np
import os
import logging
from collections import deque
import pytesseract
import time
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("StateCapture")

class StateCapture:
    """
    Class to capture and preprocess game state for the agent
    """
    def __init__(self, game_region, target_size=(84, 84), stack_frames=4, debug=False):
        """
        Initialize the state capture
        
        Args:
            game_region: Region of the screen containing the game (x, y, width, height)
            target_size: Size to resize frames to (height, width)
            stack_frames: Number of frames to stack for state representation
            debug: Whether to save debug images
        """
        self.game_region = game_region
        self.target_size = target_size
        self.stack_frames = stack_frames
        self.debug = debug
        
        # Initialize frame stack
        self.frame_stack = deque(maxlen=stack_frames)
        
        # Initialize previous frames for motion detection
        self.prev_frames = deque(maxlen=5)
        
        # Regions of Interest (ROI)
        # These will be adjusted based on the game region
        self.score_roi = None
        self.coins_roi = None
        self.set_roi(game_region)
        
        # Create debug directories
        if debug:
            os.makedirs("debug_images/raw", exist_ok=True)
            os.makedirs("debug_images/processed", exist_ok=True)
            os.makedirs("debug_images/states", exist_ok=True)
            os.makedirs("debug_images/ocr", exist_ok=True)
        
        # Last captured scores and coins
        self.last_score = 0
        self.last_coins = 0
        
        # Metrics for evaluating state quality
        self.frame_capture_times = []
        self.processing_times = []
        
        logger.info(f"StateCapture initialized with target size {target_size} and {stack_frames} stacked frames")
    
    def set_roi(self, game_region):
        """
        Set regions of interest based on game region
        
        Args:
            game_region: Region of the screen containing the game (x, y, width, height)
        """
        # Extract game region coordinates
        x, y, width, height = game_region
        
        # Set score ROI at the top right of screen, above the game region
        # These values are calculated relative to the game region but positioned correctly
        score_x = x + width * 0.9  # Positioned near the right edge of the game
        score_y = y - height * 0.05  # Slightly above the game area
        score_width = width * 0.25
        score_height = height * 0.05
        self.score_roi = (int(score_x), int(score_y), int(score_width), int(score_height))
        
        # Set coins ROI below score ROI
        coins_x = score_x
        coins_y = score_y + score_height * 1.5  # Gap between score and coins
        coins_width = score_width
        coins_height = score_height
        self.coins_roi = (int(coins_x), int(coins_y), int(coins_width), int(coins_height))
        
        logger.info(f"Score ROI set to {self.score_roi}")
        logger.info(f"Coins ROI set to {self.coins_roi}")
    
    def preprocess_frame(self, frame):
        """
        Preprocess a frame for the agent
        
        Args:
            frame: RGB image as numpy array
            
        Returns:
            Preprocessed frame (grayscale, resized)
        """
        start_time = time.time()
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Apply contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Resize to target size
        resized = cv2.resize(enhanced, self.target_size, interpolation=cv2.INTER_AREA)
        
        # Normalize pixel values to [0, 1]
        normalized = resized / 255.0
        
        # Record processing time
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        return normalized
    
    def reset(self):
        """Reset frame stack for a new episode"""
        # Clear frame stack
        self.frame_stack.clear()
        self.prev_frames.clear()
        
        # Reset metrics
        self.frame_capture_times = []
        self.processing_times = []
        
        # Reset score and coins
        self.last_score = 0
        self.last_coins = 0
    
    def capture_game_state(self, screenshot, step_num=None, episode_num=None):
        """
        Capture and preprocess the current game state
        
        Args:
            screenshot: Full screenshot as numpy array
            step_num: Current step number (for debugging)
            episode_num: Current episode number (for debugging)
            
        Returns:
            Stacked state of shape (stack_frames, height, width)
        """
        start_time = time.time()
        
        # Extract game region from screenshot
        x, y, width, height = self.game_region
        game_frame = screenshot[y:y+height, x:x+width]
        
        # Save raw frame if debugging
        if self.debug and step_num is not None and episode_num is not None:
            debug_path = f"debug_images/raw/frame_ep{episode_num}_step{step_num}.png"
            cv2.imwrite(debug_path, game_frame)
        
        # Preprocess frame
        processed_frame = self.preprocess_frame(game_frame)
        
        # Store frame for motion detection
        self.prev_frames.append(processed_frame)
        
        # Record frame capture time
        capture_time = time.time() - start_time
        self.frame_capture_times.append(capture_time)
        
        # Save processed frame if debugging
        if self.debug and step_num is not None and episode_num is not None:
            debug_path = f"debug_images/processed/frame_ep{episode_num}_step{step_num}.png"
            # Scale back to 0-255 for saving
            save_frame = (processed_frame * 255).astype(np.uint8)
            cv2.imwrite(debug_path, save_frame)
        
        # If frame stack is empty, fill it with copies of this frame
        if len(self.frame_stack) == 0:
            for _ in range(self.stack_frames):
                self.frame_stack.append(processed_frame)
        else:
            # Add new frame to stack
            self.frame_stack.append(processed_frame)
        
        # Create stacked state (stack_frames, height, width)
        stacked_state = np.array(self.frame_stack)
        
        # Save stacked state if debugging
        if self.debug and step_num is not None and episode_num is not None:
            debug_path = f"debug_images/states/state_ep{episode_num}_step{step_num}.png"
            # Create a grid for visualization
            grid_size = int(np.ceil(np.sqrt(self.stack_frames)))
            grid_height = grid_size * self.target_size[0]
            grid_width = grid_size * self.target_size[1]
            
            # Create blank grid
            grid = np.zeros((grid_height, grid_width))
            
            # Place frames in grid
            for i in range(self.stack_frames):
                row = i // grid_size
                col = i % grid_size
                grid[row*self.target_size[0]:(row+1)*self.target_size[0], 
                     col*self.target_size[1]:(col+1)*self.target_size[1]] = stacked_state[i]
            
            # Scale to 0-255 and convert to uint8
            grid = (grid * 255).astype(np.uint8)
            cv2.imwrite(debug_path, grid)
        
        return stacked_state
    
    def extract_score(self, screenshot, step_num=None, episode_num=None):
        """
        Extract score from the score region using OCR
        
        Args:
            screenshot: Full screenshot as numpy array
            step_num: Current step number (for debugging)
            episode_num: Current episode number (for debugging)
            
        Returns:
            Extracted score as integer
        """
        # Extract score region from screenshot
        x, y, width, height = self.score_roi
        score_region = screenshot[y:y+height, x:x+width]
        
        # Preprocess for OCR (high contrast for light text on dark background)
        gray = cv2.cvtColor(score_region, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to isolate text (adapt for light text on dark background)
        _, thresh1 = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        # Also try inverted threshold in case it's dark text on light background
        _, thresh2 = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        
        # Save OCR debug images if debugging
        if self.debug and step_num is not None and episode_num is not None:
            debug_path1 = f"debug_images/ocr/score1_ep{episode_num}_step{step_num}.png"
            debug_path2 = f"debug_images/ocr/score2_ep{episode_num}_step{step_num}.png"
            cv2.imwrite(debug_path1, thresh1)
            cv2.imwrite(debug_path2, thresh2)
        
        # Try OCR on both thresholded images
        try:
            # Config for single line of digits
            config = '--psm 7 -c tessedit_char_whitelist=0123456789'
            
            text1 = pytesseract.image_to_string(thresh1, config=config)
            text2 = pytesseract.image_to_string(thresh2, config=config)
            
            # Clean up text and convert to integer
            clean_text1 = ''.join(filter(str.isdigit, text1))
            clean_text2 = ''.join(filter(str.isdigit, text2))
            
            # Use the text with more digits, or default to the first one
            if len(clean_text1) > len(clean_text2):
                clean_text = clean_text1
            else:
                clean_text = clean_text2
            
            if clean_text:
                score = int(clean_text)
                # Validate score (should generally increase or stay the same)
                if score < self.last_score and score < self.last_score * 0.8:
                    # If score decreased significantly, it might be a misread
                    logger.warning(f"Score decreased significantly from {self.last_score} to {score}, using previous value")
                    return self.last_score
                else:
                    self.last_score = score
                    return score
            else:
                # If no digits found, increment the last score
                self.last_score += 1
                return self.last_score
                
        except Exception as e:
            logger.warning(f"Error extracting score: {str(e)}")
            # If error, increment the last score
            self.last_score += 1
            return self.last_score
    
    def extract_coins(self, screenshot, step_num=None, episode_num=None):
        """
        Extract coins from the coin region using OCR
        
        Args:
            screenshot: Full screenshot as numpy array
            step_num: Current step number (for debugging)
            episode_num: Current episode number (for debugging)
            
        Returns:
            Extracted coins as integer
        """
        # Extract coin region from screenshot
        x, y, width, height = self.coins_roi
        coin_region = screenshot[y:y+height, x:x+width]
        
        # Preprocess for OCR
        gray = cv2.cvtColor(coin_region, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to isolate text (try both regular and inverted)
        _, thresh1 = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        _, thresh2 = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        
        # Save OCR debug images if debugging
        if self.debug and step_num is not None and episode_num is not None:
            debug_path1 = f"debug_images/ocr/coins1_ep{episode_num}_step{step_num}.png"
            debug_path2 = f"debug_images/ocr/coins2_ep{episode_num}_step{step_num}.png"
            cv2.imwrite(debug_path1, thresh1)
            cv2.imwrite(debug_path2, thresh2)
        
        # Try OCR on both thresholded images
        try:
            # Config for single line of digits
            config = '--psm 7 -c tessedit_char_whitelist=0123456789'
            
            text1 = pytesseract.image_to_string(thresh1, config=config)
            text2 = pytesseract.image_to_string(thresh2, config=config)
            
            # Clean up text and convert to integer
            clean_text1 = ''.join(filter(str.isdigit, text1))
            clean_text2 = ''.join(filter(str.isdigit, text2))
            
            # Use the text with more digits, or default to the first one
            if len(clean_text1) > len(clean_text2):
                clean_text = clean_text1
            else:
                clean_text = clean_text2
            
            if clean_text:
                coins = int(clean_text)
                # Validate coins (should not decrease)
                if coins < self.last_coins:
                    # If coins decreased, it might be a misread
                    logger.warning(f"Coins decreased from {self.last_coins} to {coins}, using previous value")
                    return self.last_coins
                else:
                    self.last_coins = coins
                    return coins
            else:
                # If no digits found, use the last coin count
                return self.last_coins
                
        except Exception as e:
            logger.warning(f"Error extracting coins: {str(e)}")
            return self.last_coins
    
    def detect_motion(self):
        """
        Detect motion between frames to help determine if game is over
        
        Returns:
            Motion score between 0 and 1 (higher means more motion)
        """
        if len(self.prev_frames) < 2:
            return 1.0  # Assume motion if not enough frames
        
        # Get last two frames
        curr_frame = self.prev_frames[-1]
        prev_frame = self.prev_frames[-2]
        
        # Calculate absolute difference
        diff = np.abs(curr_frame - prev_frame)
        
        # Calculate motion score (mean of differences)
        motion_score = np.mean(diff)
        
        return motion_score
    
    def detect_game_over(self, screenshot=None, step_num=None, episode_num=None):
        """
        Detect if the game is over based on motion and potentially UI elements
        
        Args:
            screenshot: Current screenshot (optional)
            step_num: Current step number (for debugging)
            episode_num: Current episode number (for debugging)
            
        Returns:
            Boolean indicating if game is over
        """
        # Check for lack of motion (primary method)
        motion_score = self.detect_motion()
        
        # Increased threshold to avoid false positives
        motion_threshold = 0.005  # Higher threshold means less sensitive
        
        # Debug motion score
        if self.debug and step_num is not None and episode_num is not None and step_num % 10 == 0:
            logger.debug(f"Motion score at step {step_num}: {motion_score:.6f}")
        
        game_over = motion_score < motion_threshold
        
        # If screenshot provided, look for game over UI elements
        if screenshot is not None and game_over:
            # Could implement template matching for "game over" text here
            # For now, rely on motion detection
            pass
        
        return game_over
    
    def get_performance_stats(self):
        """
        Get statistics about state capture performance
        
        Returns:
            Dictionary of performance statistics
        """
        stats = {
            'capture_time': {
                'mean': np.mean(self.frame_capture_times) if self.frame_capture_times else 0,
                'std': np.std(self.frame_capture_times) if self.frame_capture_times else 0,
                'min': np.min(self.frame_capture_times) if self.frame_capture_times else 0,
                'max': np.max(self.frame_capture_times) if self.frame_capture_times else 0
            },
            'processing_time': {
                'mean': np.mean(self.processing_times) if self.processing_times else 0,
                'std': np.std(self.processing_times) if self.processing_times else 0,
                'min': np.min(self.processing_times) if self.processing_times else 0,
                'max': np.max(self.processing_times) if self.processing_times else 0
            }
        }
        
        return stats
    
    def log_performance_stats(self):
        """Log performance statistics to the log file"""
        stats = self.get_performance_stats()
        
        logger.info("State Capture Performance Statistics:")
        logger.info(f"  Frame Capture Time: {stats['capture_time']['mean']*1000:.2f}ms (±{stats['capture_time']['std']*1000:.2f}ms)")
        logger.info(f"  Frame Processing Time: {stats['processing_time']['mean']*1000:.2f}ms (±{stats['processing_time']['std']*1000:.2f}ms)")
        logger.info(f"  Total Mean Processing Time: {(stats['capture_time']['mean'] + stats['processing_time']['mean'])*1000:.2f}ms")