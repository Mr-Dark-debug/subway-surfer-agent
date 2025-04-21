# main.py - Enhanced with real-time training data display
import os
import time
import argparse
import numpy as np
from datetime import datetime
from subway_ai.screen_capture import ScreenCapture
from subway_ai.game_control import GameControl
import torch
from subway_ai.model import DQN, ReplayMemory, preprocess_frame, calculate_reward, save_model, load_model
try:
    from subway_ai.detector import ObjectDetector
except ImportError:
    print("Object detector not available, using template matching instead")
    ObjectDetector = None

from subway_ai.config import *

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def get_epsilon(steps_done):
    """Calculate epsilon for epsilon-greedy policy"""
    return EPSILON_END + (EPSILON_START - EPSILON_END) * \
        np.exp(-1. * steps_done / EPSILON_DECAY)

def train(screen_capture, game_control, yolo_detector=None, num_episodes=1000, load_checkpoint=None):
    """Train the AI model with enhanced real-time stats display"""
    print("Starting training mode...")
    print(f"Game region: {GAME_REGION}")
    print(f"Score region: {SCORE_REGION}")
    print(f"Coin region: {COIN_REGION}")
    
    # Initialize DQN model
    init_screen = screen_capture.capture_game_screen()
    screen_tensor = preprocess_frame(init_screen)
    _, screen_height, screen_width = screen_tensor.shape
    n_actions = NUM_ACTIONS
    
    policy_net = DQN(screen_height, screen_width, n_actions).to(device)
    target_net = DQN(screen_height, screen_width, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    
    # Initialize replay memory
    memory = ReplayMemory(MEMORY_SIZE)
    
    # Load checkpoint if specified
    start_episode = 0
    if load_checkpoint:
        model_path = os.path.join(MODELS_DIR, load_checkpoint) if not os.path.isabs(load_checkpoint) else load_checkpoint
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            policy_net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_episode = checkpoint.get('episode', 0)
            print(f"Loaded model from {model_path} (episode {start_episode})")
            target_net.load_state_dict(policy_net.state_dict())
        else:
            print(f"Checkpoint {model_path} not found, starting from scratch")
    
    # Create results directory
    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Open results log file
    results_log = os.path.join(results_dir, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    with open(results_log, 'w') as f:
        f.write("Episode,Steps,Score,Coins,TotalReward,Epsilon,DurationSecs\n")
    
    # Training loop
    steps_done = 0
    best_score = 0
    consecutive_failures = 0
    
    for episode in range(start_episode, num_episodes):
        # Reset game if needed
        if game_control.detect_game_over() or not game_control.game_running:
            play_button_loc = screen_capture.locate_play_button()
            if not game_control.restart_game(play_button_loc):
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    print("Failed to restart game multiple times, restarting from scratch")
                    game_control.start_game()
                    consecutive_failures = 0
                else:
                    time.sleep(3)  # Wait a bit before trying again
                    continue
            else:
                consecutive_failures = 0
            
            time.sleep(2)  # Wait for game to restart
        
        # Get initial state
        screen = screen_capture.capture_game_screen()
        state = preprocess_frame(screen).unsqueeze(0).to(device)
        
        # Get initial score and coins
        score = screen_capture.extract_score_ocr()
        coins = screen_capture.extract_coins_ocr()
        
        # Episode loop
        total_reward = 0
        game_over = False
        episode_steps = 0
        start_time = time.time()
        
        print(f"\n{'=' * 60}")
        print(f"STARTING EPISODE {episode} - Initial score: {score}, coins: {coins}")
        print(f"{'=' * 60}")
        
        while not game_over and episode_steps < 2000:  # Max steps per episode to prevent infinite loops
            # Record frame if recording
            game_control.record_frame()
            
            # Select action using epsilon-greedy policy
            epsilon = get_epsilon(steps_done)
            
            if np.random.random() < epsilon:
                action = game_control.random_action()
            else:
                with torch.no_grad():
                    action = policy_net(state).max(1)[1].item()
            
            # Perform action
            action_name = game_control.perform_action(action)
            
            # Wait for action to take effect
            time.sleep(0.2)
            
            # Get next state
            next_screen = screen_capture.capture_game_screen()
            next_state = preprocess_frame(next_screen).unsqueeze(0).to(device)
            
            # Get new score and coins
            next_score = screen_capture.extract_score_ocr()
            next_coins = screen_capture.extract_coins_ocr()
            
            # Check if game is over
            game_over = game_control.detect_game_over()
            
            # Calculate reward
            reward = calculate_reward(score, next_score, coins, next_coins, game_over)
            total_reward += reward
            
            # Store transition in replay memory
            memory.push(
                state.cpu(), 
                action, 
                next_state.cpu(), 
                reward, 
                game_over
            )
            
            # Move to the next state
            state = next_state
            score = next_score
            coins = next_coins
            
            # Increment step counters
            steps_done += 1
            episode_steps += 1
            screen_capture.increment_step()
            
            # Perform one step of optimization
            if len(memory) >= BATCH_SIZE:
                try:
                    # Sample batch from replay memory
                    batch = memory.sample(BATCH_SIZE)
                    if batch:
                        state_batch, action_batch, next_state_batch, reward_batch, done_batch = batch
                        
                        # Convert to tensors
                        state_batch = torch.cat(state_batch).to(device)
                        action_batch = torch.tensor(action_batch, device=device).unsqueeze(1)
                        reward_batch = torch.tensor(reward_batch, device=device).unsqueeze(1)
                        next_state_batch = torch.cat(next_state_batch).to(device)
                        done_batch = torch.tensor(done_batch, device=device).unsqueeze(1).float()
                        
                        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
                        state_action_values = policy_net(state_batch).gather(1, action_batch)
                        
                        # Compute V(s_{t+1}) for all next states
                        next_state_values = torch.zeros(BATCH_SIZE, 1, device=device)
                        with torch.no_grad():
                            next_state_values = target_net(next_state_batch).max(1)[0].unsqueeze(1)
                        
                        # Compute the expected Q values
                        expected_state_action_values = reward_batch + (GAMMA * next_state_values * (1 - done_batch))
                        
                        # Compute Huber loss
                        loss = torch.nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values)
                        
                        # Optimize the model
                        optimizer.zero_grad()
                        loss.backward()
                        for param in policy_net.parameters():
                            param.grad.data.clamp_(-1, 1)
                        optimizer.step()
                except Exception as e:
                    print(f"Error during optimization: {e}")
            
            # Save screenshot periodically
            if steps_done % SCREENSHOT_INTERVAL == 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                screen_capture.save_screenshot(next_screen, "gameplay", f"step_{steps_done}_{timestamp}")
            
            # Update and display real-time training stats
            if episode_steps % 10 == 0:
                # Calculate elapsed time
                elapsed = time.time() - start_time
                
                # Update game control stats
                game_control.update_stats(score, coins, reward, epsilon)
                
                # Print progress in a compact format
                print(f"\rEpisode: {episode}/{num_episodes} | Steps: {episode_steps} | Score: {score} | Coins: {coins} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.4f} | Time: {elapsed:.1f}s", end="")
        
        # Update target network periodically
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # Log episode results
        episode_duration = time.time() - start_time
        with open(results_log, 'a') as f:
            f.write(f"{episode},{episode_steps},{score},{coins},{total_reward:.2f},{epsilon:.4f},{episode_duration:.1f}\n")
        
        # Save model checkpoint
        if episode % CHECKPOINT_INTERVAL == 0 or score > best_score:
            if score > best_score:
                best_score = score
                save_model(policy_net, optimizer, episode, f"subway_dqn_best.pt")
            save_model(policy_net, optimizer, episode, f"subway_dqn_episode_{episode}.pt")
        
        # Print episode summary
        elapsed = time.time() - start_time
        print(f"\n\n{'=' * 60}")
        print(f"EPISODE {episode} FINISHED:")
        print(f"Steps: {episode_steps}")
        print(f"Score: {score}")
        print(f"Coins: {coins}")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Duration: {elapsed:.1f}s")
        print(f"{'=' * 60}\n")
        
        # Short delay between episodes
        time.sleep(1)
    
    print("Training complete!")
    return policy_net

def play(screen_capture, game_control, model_path=None, num_games=5):
    """Play using a trained model"""
    print("Starting play mode...")
    
    # Initialize DQN model
    init_screen = screen_capture.capture_game_screen()
    screen_tensor = preprocess_frame(init_screen)
    _, screen_height, screen_width = screen_tensor.shape
    n_actions = NUM_ACTIONS
    
    # Create model
    model = DQN(screen_height, screen_width, n_actions).to(device)
    
    # Load model if specified
    use_model = False
    if model_path:
        model_path = os.path.join(MODELS_DIR, model_path) if not os.path.isabs(model_path) else model_path
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {model_path}")
            model.eval()  # Set to evaluation mode
            use_model = True
        else:
            print(f"Model file {model_path} not found, using random actions")
    else:
        print("No model specified, using random actions")
    
    # Create results directory
    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Open results log file
    results_log = os.path.join(results_dir, f"play_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    with open(results_log, 'w') as f:
        f.write("Game,Steps,Score,Coins,Duration\n")
    
    # Play loop
    games_played = 0
    best_score = 0
    consecutive_failures = 0
    
    # Start recording
    game_control.start_recording()
    
    while games_played < num_games:
        # Reset game if needed
        if game_control.detect_game_over() or not game_control.game_running:
            play_button_loc = screen_capture.locate_play_button()
            if not game_control.restart_game(play_button_loc):
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    print("Failed to restart game multiple times, restarting from scratch")
                    game_control.start_game()
                    consecutive_failures = 0
                else:
                    time.sleep(3)  # Wait a bit before trying again
                    continue
            else:
                consecutive_failures = 0
            
            games_played += 1
            print(f"Game {games_played}/{num_games} starting...")
            time.sleep(2)  # Wait for game to restart
        
        # Get initial state
        screen = screen_capture.capture_game_screen()
        state = preprocess_frame(screen).unsqueeze(0).to(device)
        
        # Get initial score and coins
        score = screen_capture.extract_score_ocr()
        coins = screen_capture.extract_coins_ocr()
        
        # Game loop
        game_over = False
        steps = 0
        start_time = time.time()
        
        print(f"Starting game {games_played} - Initial score: {score}, coins: {coins}")
        
        while not game_over and steps < 2000 and games_played <= num_games:  # Max steps per game
            # Record frame
            game_control.record_frame()
            
            # Select action
            if use_model:
                with torch.no_grad():
                    action = model(state).max(1)[1].item()
            else:
                action = game_control.random_action()
            
            # Perform action
            action_name = game_control.perform_action(action)
            
            # Wait for action to take effect
            time.sleep(0.2)
            
            # Get next state
            next_screen = screen_capture.capture_game_screen()
            next_state = preprocess_frame(next_screen).unsqueeze(0).to(device)
            
            # Get new score and coins
            next_score = screen_capture.extract_score_ocr()
            next_coins = screen_capture.extract_coins_ocr()
            
            # Check if game is over
            game_over = game_control.detect_game_over()
            
            # Move to the next state
            state = next_state
            score = next_score
            coins = next_coins
            
            # Increment step counter
            steps += 1
            screen_capture.increment_step()
            
            # Print progress periodically
            if steps % 10 == 0:
                elapsed = time.time() - start_time
                print(f"\rGame {games_played}/{num_games} | Steps: {steps} | Score: {score} | Coins: {coins} | Time: {elapsed:.1f}s", end="")
            
            # Update best score
            if score > best_score:
                best_score = score
        
        # Log game results
        elapsed = time.time() - start_time
        with open(results_log, 'a') as f:
            f.write(f"{games_played},{steps},{score},{coins},{elapsed:.1f}\n")
        
        # Print game summary
        print(f"\n\nGame {games_played}/{num_games} finished - Steps: {steps}, Score: {score}, Coins: {coins}, Time: {elapsed:.1f}s")
        time.sleep(1)  # Short delay between games
    
    # Stop recording
    game_control.stop_recording()
    
    print(f"Played {games_played} games. Best score: {best_score}")

def main():
    """Main function for Subway Surfers AI"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Subway Surfers AI')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'play', 'calibrate'], help='Mode: train, play, or calibrate')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes to train')
    parser.add_argument('--games', type=int, default=5, help='Number of games to play')
    parser.add_argument('--model', type=str, help='Model checkpoint to load')
    parser.add_argument('--yolo', type=str, help='YOLOv5 model to load')
    args = parser.parse_args()
    
    # Handle calibration mode separately
    if args.mode == 'calibrate':
        print("Starting calibration mode...")
        try:
            import tkinter as tk
            from subway_ai.calibrate import CalibrationTool
            root = tk.Tk()
            app = CalibrationTool(root)
            root.mainloop()
        except ImportError:
            print("Error: tkinter not available. Please install tkinter to use calibration mode.")
        return
    
    # Create necessary directories
    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
    os.makedirs(TEMPLATES_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Initialize screen capture
    screen_capture = ScreenCapture()
    
    # Initialize game control
    game_control = GameControl()
    
    # Check if game is running, start it if not
    if not game_control.check_game_running():
        game_control.start_game()
    
    # Initialize YOLOv5 detector if specified
    yolo_detector = None
    if args.yolo and ObjectDetector is not None:
        yolo_path = os.path.join(MODELS_DIR, args.yolo) if not os.path.isabs(args.yolo) else args.yolo
        yolo_detector = ObjectDetector(yolo_path)
    
    try:
        # Run in specified mode
        if args.mode == 'train':
            train(screen_capture, game_control, yolo_detector, num_episodes=args.episodes, load_checkpoint=args.model)
        else:  # play mode
            play(screen_capture, game_control, model_path=args.model, num_games=args.games)
    except KeyboardInterrupt:
        print("\nInterrupted by user. Cleaning up...")
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        screen_capture.close()
        if game_control.recording:
            game_control.stop_recording()
        print("Done!")

if __name__ == "__main__":
    main()