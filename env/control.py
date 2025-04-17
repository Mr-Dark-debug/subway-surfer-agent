# env/control.py
import pyautogui
import time
import random
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("GameControls")

class GameControls:
    def __init__(self, delay=0.05):
        """
        Initialize the game controls
        
        Args:
            delay: Delay after each action in seconds (to allow game to process input)
        """
        self.delay = delay
        self.logger = logger
        
        # Define the action space
        self.actions = {
            0: 'left',
            1: 'right',
            2: 'up',
            3: 'down'
        }
        
        # Key mapping (can be customized based on game)
        self.key_map = {
            'left': 'left',
            'right': 'right',
            'up': 'up',
            'down': 'down'
        }
        
        self.logger.info(f"GameControls initialized with delay {delay}s")
    
    def take_action(self, action_id):
        """
        Execute the specified action
        
        Args:
            action_id: Integer ID of the action to perform
            
        Returns:
            Success status (bool)
        """
        if action_id not in self.actions:
            self.logger.warning(f"Invalid action ID: {action_id}")
            return False
        
        action = self.actions[action_id]
        self.logger.debug(f"Taking action: {action}")
        
        try:
            if action in self.key_map:
                key = self.key_map[action]
                pyautogui.press(key)
                time.sleep(self.delay)  # Wait for action to take effect
                return True
            else:
                self.logger.warning(f"Action '{action}' not mapped to a key")
                return False
        except Exception as e:
            self.logger.error(f"Error taking action {action}: {str(e)}")
            return False
    
    def press_key(self, key):
        """
        Press a specific key
        
        Args:
            key: Key to press
            
        Returns:
            Success status (bool)
        """
        try:
            pyautogui.press(key)
            time.sleep(self.delay)
            return True
        except Exception as e:
            self.logger.error(f"Error pressing key {key}: {str(e)}")
            return False
    
    def hold_key(self, key, duration=0.5):
        """
        Hold a key for a specified duration
        
        Args:
            key: Key to hold
            duration: Duration to hold the key (seconds)
            
        Returns:
            Success status (bool)
        """
        try:
            pyautogui.keyDown(key)
            time.sleep(duration)
            pyautogui.keyUp(key)
            time.sleep(self.delay)
            return True
        except Exception as e:
            self.logger.error(f"Error holding key {key}: {str(e)}")
            # Make sure to release the key in case of exception
            try:
                pyautogui.keyUp(key)
            except:
                pass
            return False
    
    def click_at(self, x, y):
        """
        Click at a specific position on screen
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            Success status (bool)
        """
        try:
            pyautogui.click(x, y)
            time.sleep(self.delay)
            return True
        except Exception as e:
            self.logger.error(f"Error clicking at ({x}, {y}): {str(e)}")
            return False
    
    def random_action(self):
        """
        Take a random action
        
        Returns:
            Action ID that was performed
        """
        action_id = random.randint(0, len(self.actions) - 1)
        self.take_action(action_id)
        return action_id
    
    def swipe(self, direction, duration=0.2):
        """
        Perform a swipe action (for mobile-style controls)
        
        Args:
            direction: Direction to swipe ('left', 'right', 'up', 'down')
            duration: Duration of swipe (seconds)
            
        Returns:
            Success status (bool)
        """
        try:
            # Get screen size
            screen_width, screen_height = pyautogui.size()
            
            # Calculate center point
            center_x = screen_width // 2
            center_y = screen_height // 2
            
            # Calculate swipe distance (25% of screen dimension)
            distance_x = screen_width * 0.25
            distance_y = screen_height * 0.25
            
            # Set start and end points based on direction
            if direction == 'left':
                start_x, start_y = center_x + distance_x, center_y
                end_x, end_y = center_x - distance_x, center_y
            elif direction == 'right':
                start_x, start_y = center_x - distance_x, center_y
                end_x, end_y = center_x + distance_x, center_y
            elif direction == 'up':
                start_x, start_y = center_x, center_y + distance_y
                end_x, end_y = center_x, center_y - distance_y
            elif direction == 'down':
                start_x, start_y = center_x, center_y - distance_y
                end_x, end_y = center_x, center_y + distance_y
            else:
                self.logger.warning(f"Invalid swipe direction: {direction}")
                return False
            
            # Perform the swipe
            pyautogui.moveTo(start_x, start_y)
            pyautogui.dragTo(end_x, end_y, duration=duration)
            
            time.sleep(self.delay)
            return True
            
        except Exception as e:
            self.logger.error(f"Error performing swipe {direction}: {str(e)}")
            return False
    
    def restart_game(self):
        """
        Perform actions to restart the game after game over
        
        Returns:
            Success status (bool)
        """
        try:
            # For Subway Surfers, typically pressing space or clicking works
            # This may need customization based on the specific game version
            
            # Try pressing space
            pyautogui.press('space')
            time.sleep(0.5)
            
            # Also try clicking in the center (as backup)
            screen_width, screen_height = pyautogui.size()
            pyautogui.click(screen_width // 2, screen_height // 2)
            
            time.sleep(1)  # Wait for game to restart
            
            return True
        except Exception as e:
            self.logger.error(f"Error restarting game: {str(e)}")
            return False