# calibrate.py
import os
import tkinter as tk
from tkinter import messagebox, Label, Button, Frame, Radiobutton, StringVar
import pyautogui
import time
import json
import numpy as np
import cv2
from PIL import Image, ImageTk
import sys
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from subway_ai.config import *

# Create calibration window
class CalibrationTool:
    def __init__(self, master):
        self.master = master
        master.title("Subway Surfers AI - Calibration Tool")
        
        # Set window size
        screen_width, screen_height = pyautogui.size()
        window_width = min(1000, screen_width - 100)
        window_height = min(800, screen_height - 100)
        master.geometry(f"{window_width}x{window_height}")
        
        # Create frames
        self.top_frame = Frame(master)
        self.top_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.canvas_frame = Frame(master)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.bottom_frame = Frame(master)
        self.bottom_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Create widgets
        self.label = Label(self.top_frame, text="Subway Surfers AI Calibration")
        self.label.pack(side=tk.LEFT, padx=5)
        
        self.screenshot_button = Button(self.top_frame, text="Take Screenshot", command=self.take_screenshot)
        self.screenshot_button.pack(side=tk.RIGHT, padx=5)
        
        # Canvas for displaying screenshot
        self.canvas = tk.Canvas(self.canvas_frame, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Canvas for selection
        self.start_x = None
        self.start_y = None
        self.rect = None
        
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        
        # Region selection
        self.regions = {
            "game": None,
            "score": None,
            "coins": None
        }
        
        self.current_region = StringVar(value="game")
        
        # Region selection radio buttons
        region_frame = Frame(self.bottom_frame)
        region_frame.pack(side=tk.LEFT, padx=5)
        
        Label(region_frame, text="Select Region:").pack(side=tk.LEFT, padx=5)
        
        for region in self.regions.keys():
            rb = Radiobutton(region_frame, text=region.capitalize(), variable=self.current_region, value=region)
            rb.pack(side=tk.LEFT, padx=5)
        
        # Save button
        self.save_button = Button(self.bottom_frame, text="Save Regions", command=self.save_regions)
        self.save_button.pack(side=tk.RIGHT, padx=5)
        
        # Test button
        self.test_button = Button(self.bottom_frame, text="Test Regions", command=self.test_regions)
        self.test_button.pack(side=tk.RIGHT, padx=5)
        
        # Status label
        self.status = StringVar(value="Ready")
        self.status_label = Label(self.bottom_frame, textvariable=self.status)
        self.status_label.pack(side=tk.BOTTOM, padx=5, pady=5)
        
        # Screenshot image
        self.screenshot = None
        self.photo_image = None
        
        # Setup
        self.load_existing_config()
        self.update_status()
    
    def load_existing_config(self):
        """Load existing configuration if available"""
        try:
            # Try to find existing config in regions.json
            json_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "regions.json")
            
            if os.path.exists(json_file):
                with open(json_file, 'r') as f:
                    regions_data = json.load(f)
                
                for region_name, region_values in regions_data.items():
                    if region_name in self.regions and len(region_values) == 4:
                        self.regions[region_name] = region_values
                
                self.update_status("Loaded existing configuration from regions.json")
            else:
                # Try to find values in config.py
                config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.py")
                
                if os.path.exists(config_file):
                    with open(config_file, 'r') as f:
                        content = f.read()
                    
                    # Parse GAME_REGION value
                    import re
                    
                    game_match = re.search(r'GAME_REGION\s*=\s*\(([^)]+)\)', content)
                    if game_match:
                        values = [int(x.strip()) for x in game_match.group(1).split(',')]
                        if len(values) == 4:
                            self.regions["game"] = values
                    
                    score_match = re.search(r'SCORE_REGION\s*=\s*\(([^)]+)\)', content)
                    if score_match:
                        values = [int(x.strip()) for x in score_match.group(1).split(',')]
                        if len(values) == 4:
                            self.regions["score"] = values
                    
                    coin_match = re.search(r'COIN_REGION\s*=\s*\(([^)]+)\)', content)
                    if coin_match:
                        values = [int(x.strip()) for x in coin_match.group(1).split(',')]
                        if len(values) == 4:
                            self.regions["coins"] = values
                    
                    self.update_status("Loaded existing configuration from config.py")
        except Exception as e:
            print(f"Error loading existing config: {e}")
    
    def take_screenshot(self):
        """Take a screenshot of the entire screen"""
        # Give user time to switch to the game window
        self.status.set("Taking screenshot in 3 seconds...")
        self.master.update()
        time.sleep(3)
        
        # Take the screenshot
        self.screenshot = pyautogui.screenshot()
        
        # Convert to PhotoImage and display on canvas
        self.display_screenshot()
        
        self.status.set("Screenshot taken. Select regions...")
    
    def display_screenshot(self):
        """Display the screenshot on the canvas"""
        if self.screenshot:
            # Resize the screenshot to fit the canvas
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            # Get screenshot size
            img_width, img_height = self.screenshot.size
            
            # Calculate scaling factor
            scale = min(canvas_width / img_width, canvas_height / img_height)
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            # Resize image
            resized_img = self.screenshot.resize((new_width, new_height), Image.LANCZOS)
            
            # Convert to PhotoImage
            self.photo_image = ImageTk.PhotoImage(resized_img)
            
            # Display on canvas
            self.canvas.config(width=new_width, height=new_height)
            self.canvas.create_image(0, 0, image=self.photo_image, anchor=tk.NW)
            
            # Draw existing regions
            self.draw_regions()
    
    def draw_regions(self):
        """Draw existing regions on the canvas"""
        if not self.screenshot:
            return
        
        # Get scale factor
        img_width, img_height = self.screenshot.size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        scale = min(canvas_width / img_width, canvas_height / img_height)
        
        # Clear previous region markings
        self.canvas.delete("region")
        
        # Draw each region
        colors = {
            "game": "blue",
            "score": "green",
            "coins": "orange"
        }
        
        for region_name, region in self.regions.items():
            if region:
                x, y, w, h = region
                # Scale coordinates
                x1 = int(x * scale)
                y1 = int(y * scale)
                x2 = int((x + w) * scale)
                y2 = int((y + h) * scale)
                
                # Draw rectangle
                self.canvas.create_rectangle(x1, y1, x2, y2, outline=colors[region_name], width=2, 
                                            tags=(f"region_{region_name}", "region"))
                
                # Add label
                self.canvas.create_text(x1 + 10, y1 + 10, text=region_name.upper(), fill=colors[region_name],
                                       anchor=tk.NW, tags=(f"label_{region_name}", "region"))
    
    def on_press(self, event):
        """Handle mouse press event"""
        self.start_x = event.x
        self.start_y = event.y
        
        # Create new rectangle
        if self.rect:
            self.canvas.delete(self.rect)
        
        self.rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline="red", width=2, tags="selection")
    
    def on_drag(self, event):
        """Handle mouse drag event"""
        cur_x, cur_y = event.x, event.y
        
        # Update rectangle
        self.canvas.coords(self.rect, self.start_x, self.start_y, cur_x, cur_y)
    
    def on_release(self, event):
        """Handle mouse release event"""
        end_x, end_y = event.x, event.y
        
        # Calculate region in original image coordinates
        if self.screenshot:
            img_width, img_height = self.screenshot.size
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            scale = min(canvas_width / img_width, canvas_height / img_height)
            
            # Convert canvas coordinates to image coordinates
            x1 = min(self.start_x, end_x) / scale
            y1 = min(self.start_y, end_y) / scale
            x2 = max(self.start_x, end_x) / scale
            y2 = max(self.start_y, end_y) / scale
            
            # Calculate width and height
            width = x2 - x1
            height = y2 - y1
            
            # Store region
            region_name = self.current_region.get()
            self.regions[region_name] = [int(x1), int(y1), int(width), int(height)]
            
            # Update canvas
            self.canvas.delete(f"region_{region_name}")
            self.canvas.delete(f"label_{region_name}")
            self.draw_regions()
            
            self.update_status(f"Region '{region_name}' selected")
    
    def update_status(self, message=None):
        """Update status message"""
        if message:
            self.status.set(message)
        
        # Count selected regions
        selected = sum(1 for r in self.regions.values() if r is not None)
        self.status.set(f"{selected}/{len(self.regions)} regions selected - {self.status.get()}")
    
    def test_regions(self):
        """Test the selected regions by taking screenshots of each"""
        if not all(self.regions.values()):
            missing = [name for name, region in self.regions.items() if region is None]
            messagebox.showwarning("Missing Regions", f"Please select all regions first. Missing: {', '.join(missing)}")
            return
        
        try:
            # Create test directory
            test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "screenshots", "test")
            os.makedirs(test_dir, exist_ok=True)
            
            # Take screenshots of each region
            for region_name, region in self.regions.items():
                # Take screenshot
                region_screenshot = pyautogui.screenshot(region=tuple(region))
                
                # Save screenshot
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"{region_name}_{timestamp}.png"
                filepath = os.path.join(test_dir, filename)
                region_screenshot.save(filepath)
                
                print(f"Saved test screenshot for {region_name} region to {filepath}")
            
            messagebox.showinfo("Test Complete", f"Test screenshots saved to {test_dir}")
            self.update_status("Test screenshots saved")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to test regions: {str(e)}")
    
    def save_regions(self):
        """Save the regions to config.py and regions.json"""
        # Check if all regions are selected
        if not all(self.regions.values()):
            missing = [name for name, region in self.regions.items() if region is None]
            messagebox.showwarning("Missing Regions", f"Please select all regions first. Missing: {', '.join(missing)}")
            return
        
        try:
            # First, save to regions.json for easy loading
            json_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "regions.json")
            with open(json_file, 'w') as f:
                json.dump(self.regions, f, indent=4)
            
            # Then, update config.py
            config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.py")
            
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    content = f.read()
                
                # Update regions in the content
                for region_name, region in self.regions.items():
                    pattern = rf'{region_name.upper()}_REGION\s*=\s*\([^)]+\)'
                    replacement = f'{region_name.upper()}_REGION = ({region[0]}, {region[1]}, {region[2]}, {region[3]})'
                    
                    if re.search(pattern, content):
                        content = re.sub(pattern, replacement, content)
                    else:
                        # If not found, add it
                        content += f"\n{replacement}"
                
                # Write back
                with open(config_file, 'w') as f:
                    f.write(content)
                
                messagebox.showinfo("Success", "Regions saved to config.py and regions.json")
                self.update_status("Configuration saved successfully")
            else:
                messagebox.showerror("Error", f"Config file not found: {config_file}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration: {str(e)}")
            print(f"Error details: {e}")

def main():
    root = tk.Tk()
    app = CalibrationTool(root)
    root.mainloop()

if __name__ == "__main__":
    main()