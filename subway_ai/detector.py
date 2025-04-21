# Improved detector.py with better play button detection
import os
import cv2
import numpy as np
from PIL import Image
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from subway_ai.config import *

class ObjectDetector:
    """Improved object detector for Subway Surfers with more reliable play button detection"""
    
    def __init__(self, template_dir=None):
        """Initialize the detector with templates"""
        self.templates = {}
        template_dir = template_dir or TEMPLATES_DIR
        
        # Load templates if they exist
        self.load_templates(template_dir)
        
        # Initialize counters for detection stability
        self.consecutive_play_button_detections = 0
        self.consecutive_game_over_detections = 0
        
        print(f"ObjectDetector initialized with {len(self.templates)} templates")
    
    def load_templates(self, template_dir):
        """Load all template images from the template directory"""
        if not os.path.exists(template_dir):
            os.makedirs(template_dir, exist_ok=True)
            print(f"Created template directory: {template_dir}")
            return
        
        # Load standard templates
        standard_templates = {
            'game_over': GAME_OVER_TEMPLATE,
            'play_button': PLAY_BUTTON_TEMPLATE
        }
        
        for name, path in standard_templates.items():
            if os.path.exists(path):
                try:
                    self.templates[name] = cv2.imread(path, cv2.IMREAD_COLOR)
                    print(f"Loaded template: {name}")
                except Exception as e:
                    print(f"Error loading template {name}: {e}")
        
        # Load all other .png files in the template directory
        for file in os.listdir(template_dir):
            if file.lower().endswith('.png') and not file.startswith('.'):
                name = os.path.splitext(file)[0]
                if name not in self.templates:  # Don't overwrite existing templates
                    path = os.path.join(template_dir, file)
                    try:
                        self.templates[name] = cv2.imread(path, cv2.IMREAD_COLOR)
                        print(f"Loaded additional template: {name}")
                    except Exception as e:
                        print(f"Error loading template {name}: {e}")
    
    def detect_template(self, img, template_name, threshold=0.7):
        """Detect a template in the image"""
        if template_name not in self.templates:
            print(f"Template '{template_name}' not found")
            return None
        
        # Convert image to correct format
        if isinstance(img, Image.Image):
            img_np = np.array(img)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif isinstance(img, np.ndarray):
            if len(img.shape) == 2:  # Grayscale
                img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:  # Color
                img_bgr = img.copy()
        else:
            print(f"Unsupported image type: {type(img)}")
            return None
        
        # Get template
        template = self.templates[template_name]
        
        # Perform template matching
        result = cv2.matchTemplate(img_bgr, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # If good match, return the location and size
        if max_val > threshold:
            h, w = template.shape[:2]
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            center = (top_left[0] + w//2, top_left[1] + h//2)
            
            return {
                'confidence': max_val,
                'top_left': top_left,
                'bottom_right': bottom_right,
                'center': center,
                'width': w,
                'height': h
            }
        
        return None
    
    def _is_likely_play_button(self, contour, img_shape, min_area=1000):
        """Determine if a contour is likely to be a play button based on multiple criteria"""
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
    
    def detect_game_over(self, img, threshold=0.6):
        """Improved game over detection with stability counter"""
        # Convert to proper format
        if isinstance(img, Image.Image):
            img_np = np.array(img)
        else:
            img_np = img.copy()
        
        # Try template matching first for "Game Over" text
        result = self.detect_template(img_np, 'game_over', threshold)
        if result and result['confidence'] > 0.75:  # Higher confidence for game over
            self.consecutive_game_over_detections += 1
            if self.consecutive_game_over_detections >= 2:  # Require multiple detections
                print(f"Game over detected with high confidence {result['confidence']:.2f}")
                return True
        
        # Also check for play button as an indicator of game over, requiring multiple consecutive detections
        play_button = self.locate_play_button(img_np, threshold=0.65)  # Increased threshold
        if play_button:
            self.consecutive_play_button_detections += 1
            if self.consecutive_play_button_detections >= 3:  # Require 3 consecutive detections
                print(f"Play button detected consistently - game is likely over")
                return True
        else:
            # Reset counter if no play button detected
            self.consecutive_play_button_detections = 0
        
        # If neither game over text nor play button detected consistently, reset game over counter
        if not result or result['confidence'] <= 0.75:
            self.consecutive_game_over_detections = 0
        
        return False
    
    def locate_play_button(self, img, threshold=0.65):
        """Improved locate play button function with more robust color detection"""
        # Try template matching first (higher threshold for more confidence)
        result = self.detect_template(img, 'play_button', threshold)
        if result and result['confidence'] > 0.7:  # Higher confidence threshold
            # Return center coordinates
            center_x = result['center'][0] + GAME_REGION[0]  # Add game region offset
            center_y = result['center'][1] + GAME_REGION[1]
            print(f"Play button found with high confidence {result['confidence']:.2f} at ({center_x}, {center_y})")
            return (center_x, center_y)
        
        # Convert to proper format for color analysis
        if isinstance(img, Image.Image):
            img_np = np.array(img)
        else:
            img_np = img.copy()
        
        # Try to find green button using improved color detection
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            # Convert to HSV color space
            img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
            
            # Define more specific range for green play button color
            # This is a more targeted green range to reduce false positives
            lower_green = np.array([45, 120, 120])  # More specific green
            upper_green = np.array([75, 255, 255])  # More specific green
            
            # Create a mask for green areas
            green_mask = cv2.inRange(img_hsv, lower_green, upper_green)
            
            # Apply morphology to clean the mask (remove noise)
            kernel = np.ones((5, 5), np.uint8)
            green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
            green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours of green areas
            contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Check if we have significant green areas
            if np.sum(green_mask) < 5000:  # Minimum number of green pixels
                return None
            
            if contours:
                # Sort contours by area (largest first)
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                
                for contour in contours:
                    # Check if contour is likely a play button using multiple criteria
                    if self._is_likely_play_button(contour, img_np.shape[:2]):
                        # Get bounding box and center point
                        x, y, w, h = cv2.boundingRect(contour)
                        center_x = x + w//2 + GAME_REGION[0]
                        center_y = y + h//2 + GAME_REGION[1]
                        
                        # Additional validation: check for symmetry typical of buttons
                        symmetry_score = self._calculate_symmetry(green_mask, x, y, w, h)
                        if symmetry_score > 0.7:  # High symmetry requirement
                            print(f"Play button found using enhanced color detection at ({center_x}, {center_y})")
                            return (center_x, center_y)
        
        # If no play button detected by reliable methods, don't use fallback position
        # This avoids false positives for play button detection
        return None
    
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
    
    def analyze_game_screen(self, img):
        """Analyze the game screen for obstacles and coins"""
        # This is a placeholder for more advanced game analysis
        # In a real implementation, you would use object detection to find:
        # - Character position
        # - Obstacles ahead
        # - Coins
        # - Power-ups
        
        # For now, just return a simple analysis
        return {
            'game_over': self.detect_game_over(img),
            'play_button': self.locate_play_button(img)
        }
    
    def visualize_detection(self, img, detections, save_path=None):
        """Visualize detections on the image"""
        # Convert image to correct format
        if isinstance(img, Image.Image):
            img_np = np.array(img)
            img_vis = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif isinstance(img, np.ndarray):
            img_vis = img.copy()
        else:
            print(f"Unsupported image type: {type(img)}")
            return None
        
        # Draw each detection
        if 'game_over' in detections and detections['game_over']:
            # Draw a "GAME OVER" text
            cv2.putText(img_vis, "GAME OVER", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if 'play_button' in detections and detections['play_button']:
            # Draw a circle at the play button location
            play_x, play_y = detections['play_button']
            # Adjust coordinates to be relative to the image
            rel_x = play_x - GAME_REGION[0]
            rel_y = play_y - GAME_REGION[1]
            if 0 <= rel_x < img_vis.shape[1] and 0 <= rel_y < img_vis.shape[0]:
                cv2.circle(img_vis, (rel_x, rel_y), 20, (0, 255, 0), 2)
                cv2.putText(img_vis, "PLAY", (rel_x-20, rel_y-25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Save the visualization if a path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            cv2.imwrite(save_path, img_vis)
            print(f"Visualization saved to {save_path}")
        
        return img_vis