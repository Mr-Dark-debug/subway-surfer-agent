# Main training script for Subway Surfers AI
import os
import time
import torch
import torch.optim as optim
import numpy as np
from datetime import datetime
import cv2
import argparse

# Import our modules
from subway_ai.screen_capture import ScreenCapture
from subway_ai.game_control import GameControl
from subway_ai.model import DQN, ReplayMemory, YOLODetector, preprocess_frame, calculate_reward, save_model, load_model
from subway_ai.config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def get_epsilon(steps_done):
    """Calculate epsilon for epsilon-greedy policy"""
    return EPSILON_END + (EPSILON_START - EPSILON_END) * \
        np.exp(-1. * steps_done / EPSILON_DECAY)

def train(num_episodes=1000, load_checkpoint=None):
    """Main training loop"""
    # Initialize screen capture
    screen_capture = ScreenCapture()
    
    # Initialize game control
    game_control = GameControl()
    
    # Check if game is running, start it if not
    if not game_control.check_game_running():
        game_control.start_game()
    
    # Initialize YOLOv5 detector
    yolo_detector = YOLODetector(os.path.join(MODELS_DIR, "yolov5_subway.pt"))
    
    # Initialize DQN model
    init_screen = screen_capture.capture_game_screen()
    _, screen_height, screen_width = preprocess_frame(init_screen).shape
    n_actions = NUM_ACTIONS
    
    policy_net = DQN(screen_height, screen_width, n_actions).to(device)
    target_net = DQN(screen_height, screen_width, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    # Initialize optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    
    # Initialize replay memory
    memory = ReplayMemory(MEMORY_SIZE)
    
    # Load checkpoint if specified
    start_episode = 0
    if load_checkpoint:
        start_episode = load_model(policy_net, optimizer, load_checkpoint)
        target_net.load_state_dict(policy_net.state_dict())
    
    # Training loop
    steps_done = 0
    best_score = 0
    
    for episode in range(start_episode, num_episodes):
        # Reset game if needed
        if screen_capture.detect_game_over():
            play_button_loc = screen_capture.locate_play_button()
            game_control.restart_game(play_button_loc)
            time.sleep(2)  # Wait for game to restart
        
        # Get initial state
        screen = screen_capture.capture_game_screen()
        state = preprocess_frame(screen)
        
        # Get initial score and coins
        score = screen_capture.extract_score_ocr()
        coins = screen_capture.extract_coins_ocr()
        
        # Episode loop
        total_reward = 0
        game_over = False
        
        while not game_over:
            # Select action
            epsilon = get_epsilon(steps_done)
            if torch.cuda.is_available():
                state_tensor = state.cuda()
            else:
                state_tensor = state
                
            # Get action using epsilon-greedy policy
            if np.random.random() < epsilon:
                action = game_control.random_action()
            else:
                with torch.no_grad():
                    action = policy_net(state_tensor).max(1)[1].item()
            
            # Perform action
            action_name = game_control.perform_action(action)
            
            # Wait for action to take effect
            time.sleep(0.1)
            
            # Get next state
            next_screen = screen_capture.capture_game_screen()
            next_state = preprocess_frame(next_screen)
            
            # Get new score and coins
            next_score = screen_capture.extract_score_ocr()
            next_coins = screen_capture.extract_coins_ocr()
            
            # Check if game is over
            game_over = screen_capture.detect_game_over()
            
            # Calculate reward
            reward = calculate_reward(score, next_score, coins, next_coins, game_over)
            total_reward += reward
            
            # Store transition in replay memory
            memory.push(state, action, next_state, reward, game_over)
            
            # Move to the next state
            state = next_state
            score = next_score
            coins = next_coins
            
            # Increment step counter
            steps_done += 1
            screen_capture.increment_step()
            
            # Perform one step of optimization
            if len(memory) >= BATCH_SIZE:
                # Sample batch from replay memory
                batch = memory.sample(BATCH_SIZE)
                if batch:
                    state_batch, action_batch, next_state_batch, reward_batch, done_batch = batch
                    
                    # Convert to tensors
                    state_batch = torch.cat(state_batch)
                    action_batch = torch.tensor(action_batch, device=device).unsqueeze(1)
                    reward_batch = torch.tensor(reward_batch, device=device).unsqueeze(1)
                    next_state_batch = torch.cat(next_state_batch)
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
            
            # Save screenshot periodically
            if steps_done % SCREENSHOT_INTERVAL == 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                screen_capture.save_screenshot(next_screen, "gameplay", f"step_{steps_done}_{timestamp}")
            
            # Print progress
            if steps_done % 100 == 0:
                print(f"Episode: {episode}, Steps: {steps_done}, Score: {score}, Coins: {coins}, Epsilon: {epsilon:.2f}")
        
        # Update target network
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # Save model checkpoint
        if episode % CHECKPOINT_INTERVAL == 0 or score > best_score:
            if score > best_score:
                best_score = score
                save_model(policy_net, optimizer, episode, f"subway_dqn_best.pt")
            save_model(policy_net, optimizer, episode, f"subway_dqn_episode_{episode}.pt")
        
        # Print episode summary
        print(f"Episode {episode} finished - Score: {score}, Coins: {coins}, Total Reward: {total_reward:.2f}")
    
    # Clean up
    screen_capture.close()
    print("Training complete!")

def main():
    parser = argparse.ArgumentParser(description='Train Subway Surfers AI')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes to train')
    parser.add_argument('--load', type=str, help='Load model checkpoint')
    args = parser.parse_args()
    
    train(num_episodes=args.episodes, load_checkpoint=args.load)

if __name__ == "__main__":
    main()