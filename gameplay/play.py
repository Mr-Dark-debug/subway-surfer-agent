"""
Subway Surfers AI - Playing with trained model
This script uses a trained DQN model to play Subway Surfers automatically
"""

import pyautogui
import time
import subprocess
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import traceback
from datetime import datetime

# Import DQN model definition
from main import DQN, setup_game, preprocess_frame, GameStateTracker, GameDigitReader

# Set device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class SubwaySurfersPlayer:
    def __init__(self, model_path, n_actions=4):
        self.model_path = model_path
        self.n_actions = n_actions
        
        # Load model
        self.model = self.load_model()
        
        # Action mapping
        self.actions = {
            0: 'left',
            1: 'up',
            2: 'right',
            3: 'down'
        }
        
        # State tracking
        self.game_tracker = GameStateTracker()
        self.total_steps = 0
        self.total_reward = 0
        self.start_time = None
        self.game_count = 0
        
    def load_model(self):
        """Load the trained model"""
        try:
            # Network dimensions
            h, w = 84, 84
            
            # Create and load model
            model = DQN(h, w, self.n_actions).to(device)
            
            checkpoint = torch.load(self.model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()  # Set to evaluation mode
            
            # Print model info
            epsilon = checkpoint.get('epsilon', 'N/A')
            steps = checkpoint.get('steps_done', 'N/A')
            episode = checkpoint.get('episode', 'N/A')
            reward = checkpoint.get('reward', 'N/A')
            
            print(f"Loaded model from {self.model_path}")
            print(f"Model info - Episode: {episode}, Steps: {steps}, Epsilon: {epsilon}, Reward: {reward}")
            
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            traceback.print_exc()
            raise
    
    def select_action(self, state):
        """Select the best action for current state"""
        with torch.no_grad():
            # Forward pass through the model
            q_values = self.model(state)
            action_idx = q_values.max(1)[1].view(1, 1)
            action_name = self.actions[action_idx.item()]
            return action_idx, action_name
    
    def execute_action(self, action_idx):
        """Execute the action in the game"""
        action = self.actions[action_idx.item()]
        pyautogui.press(action)
        time.sleep(0.05)
    
    def play(self, game_region, score_region, coin_region, max_games=10, max_steps=5000):
        """Play the game using the trained model"""
        self.start_time = time.time()
        self.game_count = 0
        self.total_steps = 0
        self.total_reward = 0
        
        # Create directories for recordings
        os.makedirs("gameplay", exist_ok=True)
        
        # Start playing
        print(f"\nStarting gameplay with trained model - max {max_games} games")
        print("Press Ctrl+C at any time to stop")
        
        # Start gameplay loop
        try:
            while self.game_count < max_games:
                # Start a new game
                self.game_count += 1
                print(f"\n--- Game {self.game_count}/{max_games} ---")
                
                # Reset game state tracker
                self.game_tracker.reset()
                
                # Start game
                pyautogui.press('space')
                time.sleep(2)
                
                # Get initial state
                game_img = pyautogui.screenshot(region=game_region)
                score_img = pyautogui.screenshot(region=score_region)
                coin_img = pyautogui.screenshot(region=coin_region)
                
                # Check if we're already in a game over state
                initial_done, initial_type = self.game_tracker.detect_game_over(game_img)
                if initial_done:
                    print(f"Game over screen detected at start! Type: {initial_type}")
                    self.game_tracker.handle_game_over(initial_type, game_region)
                    time.sleep(2)
                    continue
                
                # Process initial state
                current_state = preprocess_frame(game_img)
                prev_game_img = game_img
                
                # Save initial game state
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                game_img.save(f"gameplay/game_{self.game_count}_start_{timestamp}.png")
                
                # Game loop
                steps = 0
                episode_reward = 0
                done = False
                
                while not done and steps < max_steps:
                    # Select best action
                    action, action_name = self.select_action(current_state)
                    
                    # Execute action
                    self.execute_action(action)
                    
                    # Wait for game to respond
                    time.sleep(0.1)
                    
                    # Capture next state
                    next_game_img = pyautogui.screenshot(region=game_region)
                    next_score_img = pyautogui.screenshot(region=score_region)
                    next_coin_img = pyautogui.screenshot(region=coin_region)
                    
                    # Process next state
                    next_state = preprocess_frame(next_game_img)
                    
                    # Calculate reward and check if game over
                    reward, done, game_over_type = self.game_tracker.calculate_reward(
                        next_score_img, next_coin_img, next_game_img, prev_game_img
                    )
                    
                    # Update current state
                    current_state = next_state
                    prev_game_img = next_game_img
                    
                    # Update stats
                    episode_reward += reward
                    steps += 1
                    self.total_steps += 1
                    
                    # Periodic status update
                    if steps % 10 == 0:
                        # Calculate time played and fps
                        elapsed = time.time() - self.start_time
                        fps = self.total_steps / elapsed if elapsed > 0 else 0
                        
                        # Calculate score and coins
                        score = self.game_tracker.last_score
                        coins = self.game_tracker.last_coins
                        
                        print(f"Step: {steps}, Action: {action_name}, Score: {score}, Coins: {coins}, FPS: {fps:.1f}", end="\r")
                    
                    # Save occasional screenshots
                    if steps % 100 == 0:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        next_game_img.save(f"gameplay/game_{self.game_count}_step_{steps}_{timestamp}.png")
                
                # Game ended
                self.total_reward += episode_reward
                
                # Log game results
                print(f"\nGame {self.game_count} ended after {steps} steps - Reward: {episode_reward:.2f}")
                print(f"Final Score: {self.game_tracker.last_score}, Coins: {self.game_tracker.last_coins}")
                
                # Save final game state
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                next_game_img.save(f"gameplay/game_{self.game_count}_end_{timestamp}.png")
                
                # Handle game over
                if game_over_type:
                    print(f"Game over (type: {game_over_type})")
                    self.game_tracker.handle_game_over(game_over_type, game_region)
                    time.sleep(2)
        
        except KeyboardInterrupt:
            print("\nGameplay stopped by user")
        
        except Exception as e:
            print(f"\nError during gameplay: {e}")
            traceback.print_exc()
        
        finally:
            # Print gameplay statistics
            elapsed = time.time() - self.start_time
            
            print("\n" + "=" * 50)
            print("Gameplay Statistics:")
            print(f"Total games played: {self.game_count}")
            print(f"Total steps: {self.total_steps}")
            print(f"Average steps per game: {self.total_steps / self.game_count if self.game_count > 0 else 0:.1f}")
            print(f"Total reward: {self.total_reward:.2f}")
            print(f"Time played: {elapsed:.1f} seconds")
            print(f"Average FPS: {self.total_steps / elapsed if elapsed > 0 else 0:.1f}")
            print("=" * 50)

def cleanup():
    """Close browser and cleanup resources"""
    try:
        # Attempt to close the browser using Alt+F4
        pyautogui.hotkey('alt', 'f4')
        time.sleep(1)
        
        # Press Enter in case a "confirm close" dialog appears
        pyautogui.press('enter')
        time.sleep(0.5)
    except Exception as e:
        print(f"Error during cleanup: {e}")
    
    print("Browser closed. Exiting program.")

def find_best_model():
    """Find the best trained model to use"""
    # Look for final model first
    if os.path.exists("models/subway_surfers_final_model.pth"):
        return "models/subway_surfers_final_model.pth"
    
    # Then look for latest checkpoint
    if os.path.exists("models/latest_checkpoint.pth"):
        return "models/latest_checkpoint.pth"
    
    # Finally look for any episode checkpoint, get the latest
    checkpoints = sorted(glob.glob("models/subway_surfers_dqn_episode_*.pth"))
    if checkpoints:
        return checkpoints[-1]
    
    return None

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Subway Surfers AI Player")
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model file (default: best available model)')
    parser.add_argument('--games', type=int, default=10,
                        help='Number of games to play (default: 10)')
    parser.add_argument('--steps', type=int, default=5000,
                        help='Maximum steps per game (default: 5000)')
    args = parser.parse_args()
    
    # If no model specified, find the best one
    model_path = args.model
    if model_path is None:
        model_path = find_best_model()
        if model_path is None:
            print("No trained model found. Please train a model first or specify a model path.")
            return
    
    # Set up game
    game_region, score_region, coin_region = setup_game()
    
    try:
        # Create player with trained model
        player = SubwaySurfersPlayer(model_path)
        
        # Start playing
        player.play(game_region, score_region, coin_region, args.games, args.steps)
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
    
    finally:
        # Clean up
        cleanup()

if __name__ == "__main__":
    # Print banner
    print("=" * 60)
    print("Subway Surfers AI - Playing with trained model")
    print("=" * 60)
    
    print("Starting in 3 seconds...")
    time.sleep(3)
    
    # Run main function
    main()