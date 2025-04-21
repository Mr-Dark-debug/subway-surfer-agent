# Model module for Subway Surfers AI
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
import cv2
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from subway_ai.config import *

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class DQN(nn.Module):
    """Deep Q-Network for Subway Surfers AI"""
    
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        # Convolutional layers to process game frames
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Calculate the size of feature maps after convolutions
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64
        
        # Fully connected layers
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.fc2 = nn.Linear(512, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        return self.fc2(x)

class ReplayMemory:
    """Experience Replay Memory for DQN"""
    
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        
    def push(self, state, action, next_state, reward, done):
        """Save a transition"""
        self.memory.append((state, action, next_state, reward, done))
        
    def sample(self, batch_size):
        """Sample a batch of transitions"""
        if len(self.memory) < batch_size:
            return None
        transitions = random.sample(self.memory, batch_size)
        state, action, next_state, reward, done = zip(*transitions)
        return state, action, next_state, reward, done
        
    def __len__(self):
        return len(self.memory)

def preprocess_frame(frame):
    """Process game frame for input to neural network"""
    # Convert to numpy array if it's a PIL Image
    if not isinstance(frame, np.ndarray):
        frame_array = np.array(frame)
    else:
        frame_array = frame.copy()
    
    # Convert to grayscale
    if len(frame_array.shape) == 3 and frame_array.shape[2] == 3:
        gray = cv2.cvtColor(frame_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = frame_array
    
    # Resize to a smaller dimension for processing efficiency
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    
    # Normalize pixel values
    normalized = resized / 255.0
    
    # Add channel dimension for PyTorch (channels, height, width)
    tensor = torch.FloatTensor(normalized).unsqueeze(0)
    
    return tensor

def calculate_reward(prev_score, curr_score, prev_coins, curr_coins, game_over):
    """Calculate reward based on score, coins, and game state"""
    reward = 0
    
    # Reward for surviving
    reward += REWARD_SURVIVAL
    
    # Reward for score increase
    score_diff = curr_score - prev_score
    if score_diff > 0:
        reward += score_diff * REWARD_SCORE
    
    # Reward for collecting coins
    coin_diff = curr_coins - prev_coins
    if coin_diff > 0:
        reward += coin_diff * REWARD_COIN
    
    # Penalty for game over
    if game_over:
        reward += PENALTY_CRASH
    
    return reward

def save_model(model, optimizer, episode, filename):
    """Save model checkpoint"""
    os.makedirs(MODELS_DIR, exist_ok=True)
    checkpoint_path = os.path.join(MODELS_DIR, filename)
    torch.save({
        'episode': episode,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Model saved to {checkpoint_path}")

def load_model(model, optimizer, filename):
    """Load model checkpoint"""
    checkpoint_path = os.path.join(MODELS_DIR, filename)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        episode = checkpoint.get('episode', 0)
        print(f"Loaded model from {checkpoint_path} (episode {episode})")
        return episode
    else:
        print(f"No checkpoint found at {checkpoint_path}, starting from scratch")
        return 0