# Object detector module for Subway Surfers AI
import os
import cv2
import numpy as np
from PIL import Image
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from subway_ai.config import *

class ObjectDetector:
    """Simple object detector for Subway Surfers using template matching and color detection"""
    
    def __init__(self, template_dir=None):
        """Initialize the detector with templates"""
        self.templates = {}
        template_dir = template_dir or TEMPLATES_DIR
        
        # Load templates if they exist
        self.load_templates(template_dir)
        
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
    
    def detect_game_over(self, img, threshold=0.6):
        """Detect game over screen"""
        # Try template matching first
        result = self.detect_template(img, 'game_over', threshold)
        if result:
            print(f"Game over detected with confidence {result['confidence']:.2f}")
            return True
        
        # Also check for play button as an indicator of game over
        play_button = self.detect_template(img, 'play_button', threshold)
        if play_button:
            print(f"Play button detected with confidence {play_button['confidence']:.2f}")
            return True
        
        # Convert to proper format for color analysis
        if isinstance(img, Image.Image):
            img_np = np.array(img)
        else:
            img_np = img
        
        # Check for green play button using color detection
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            # Convert to HSV color space
            img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
            
            # Define range for green color
            lower_green = np.array([40, 100, 100])
            upper_green = np.array([80, 255, 255])
            
            # Create a mask for green areas
            green_mask = cv2.inRange(img_hsv, lower_green, upper_green)
            
            # Check bottom half of the screen for large green areas (play button)
            h = green_mask.shape[0]
            bottom_half = green_mask[h//2:, :]
            if np.sum(bottom_half) > 10000:  # Significant green area
                print("Green play button detected in bottom half")
                return True
        
        return False
    
    def locate_play_button(self, img, threshold=0.6):
        """Locate the play button on the game over screen"""
        # Try template matching first
        result = self.detect_template(img, 'play_button', threshold)
        if result:
            # Return center coordinates
            center_x = result['center'][0] + GAME_REGION[0]  # Add game region offset
            center_y = result['center'][1] + GAME_REGION[1]
            print(f"Play button found at ({center_x}, {center_y}) with confidence {result['confidence']:.2f}")
            return (center_x, center_y)
        
        # Convert to proper format for color analysis
        if isinstance(img, Image.Image):
            img_np = np.array(img)
        else:
            img_np = img
        
        # Try to find green button using color detection
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            # Convert to HSV color space
            img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
            
            # Define range for green color
            lower_green = np.array([40, 100, 100])
            upper_green = np.array([80, 255, 255])
            
            # Create a mask for green areas
            green_mask = cv2.inRange(img_hsv, lower_green, upper_green)
            
            # Find contours of green areas
            contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Sort contours by area (largest first)
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 1000:  # Min area threshold
                        # Get bounding box
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # Check if in bottom half of screen (play button location)
                        if y > img_np.shape[0] * 0.5:
                            center_x = x + w//2 + GAME_REGION[0]
                            center_y = y + h//2 + GAME_REGION[1]
                            print(f"Play button found using color detection at ({center_x}, {center_y})")
                            return (center_x, center_y)
        
        # Fall back to a predefined position
        center_x = GAME_REGION[0] + GAME_REGION[2] // 2
        center_y = GAME_REGION[1] + GAME_REGION[3] - 60  # Near bottom
        print(f"Using fallback play button position at ({center_x}, {center_y})")
        return (center_x, center_y)
    
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

# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = ObjectDetector()
    
    # Take a screenshot
    import pyautogui
    screenshot = pyautogui.screenshot(region=GAME_REGION)
    
    # Analyze the screenshot
    analysis = detector.analyze_game_screen(screenshot)
    
    # Visualize the analysis
    vis_img = detector.visualize_detection(screenshot, analysis, "detection_result.png")
    
    # Display the result
    if vis_img is not None:
        cv2.imshow("Detection Result", vis_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()