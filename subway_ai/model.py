# Improved model.py with better GPU support and error handling
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
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from subway_ai.config import *

# Check if CUDA is available and properly configured
try:
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        # Get GPU info
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
        print(f"CUDA is available: {gpu_count} device(s)")
        print(f"Using GPU: {gpu_name}")
        device = torch.device("cuda")
    else:
        print("CUDA is not available, using CPU")
        device = torch.device("cpu")
except Exception as e:
    print(f"Error checking CUDA: {e}")
    print("Defaulting to CPU")
    device = torch.device("cpu")

print(f"Using device: {device}")

class DQN(nn.Module):
    """Deep Q-Network for Subway Surfers AI with enhanced architecture"""
    
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
        self.dropout = nn.Dropout(0.2)  # Add dropout for regularization
        self.fc2 = nn.Linear(512, outputs)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Move model to appropriate device
        self.to(device)
    
    def _init_weights(self, module):
        """Initialize weights with Xavier initialization"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """Forward pass with input on the correct device"""
        # Ensure input is on the correct device
        x = x.to(device)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x = self.dropout(x)  # Apply dropout
        return self.fc2(x)

class ReplayMemory:
    """Experience Replay Memory for DQN with improved implementation"""
    
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.max_priority = 1.0  # For prioritized experience replay (optional)
        
    def push(self, state, action, next_state, reward, done):
        """Save a transition"""
        # Convert tensors to CPU before storing to save GPU memory
        if isinstance(state, torch.Tensor):
            state = state.cpu()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu()
            
        self.memory.append((state, action, next_state, reward, done))
        
    def sample(self, batch_size):
        """Sample a batch of transitions"""
        if len(self.memory) < batch_size:
            return None
            
        # Randomly sample transitions
        transitions = random.sample(self.memory, batch_size)
        
        # Unpack the transitions
        states, actions, next_states, rewards, dones = zip(*transitions)
        
        # Convert states to tensor batch
        state_batch = torch.cat([state.unsqueeze(0) if state.dim() == 3 else state for state in states])
        
        # Convert next_states to tensor batch
        next_state_batch = torch.cat([state.unsqueeze(0) if state.dim() == 3 else state for state in next_states])
        
        return state_batch, actions, next_state_batch, rewards, dones
    
    def get_prioritized_sample(self, batch_size, alpha=0.6):
        """Sample transitions based on priority (optional advanced feature)"""
        if len(self.memory) < batch_size:
            return None
            
        # Calculate priorities (using age-based priority as simple approach)
        priorities = np.array([i**(-alpha) for i in range(1, len(self.memory) + 1)])
        probabilities = priorities / sum(priorities)
        
        # Sample based on priorities
        indices = np.random.choice(len(self.memory), batch_size, p=probabilities, replace=False)
        transitions = [self.memory[i] for i in indices]
        
        # Unpack the transitions
        states, actions, next_states, rewards, dones = zip(*transitions)
        
        # Convert states to tensor batch
        state_batch = torch.cat([state.unsqueeze(0) if state.dim() == 3 else state for state in states])
        
        # Convert next_states to tensor batch
        next_state_batch = torch.cat([state.unsqueeze(0) if state.dim() == 3 else state for state in next_states])
        
        return state_batch, actions, next_state_batch, rewards, dones, indices
        
    def __len__(self):
        return len(self.memory)

def preprocess_frame(frame):
    """Process game frame for input to neural network with better preprocessing"""
    # Convert to numpy array if it's a PIL Image
    if not isinstance(frame, np.ndarray):
        frame_array = np.array(frame)
    else:
        frame_array = frame.copy()
    
    # Convert to grayscale
    if len(frame_array.shape) == 3 and frame_array.shape[2] == 3:
        # Use weighted grayscale conversion for better feature distinction
        gray = cv2.cvtColor(frame_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = frame_array
    
    # Apply contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Resize to a smaller dimension for processing efficiency
    resized = cv2.resize(enhanced, (84, 84), interpolation=cv2.INTER_AREA)
    
    # Normalize pixel values
    normalized = resized / 255.0
    
    # Add channel dimension for PyTorch (channels, height, width)
    tensor = torch.FloatTensor(normalized).unsqueeze(0)
    
    return tensor

def calculate_reward(prev_score, curr_score, prev_coins, curr_coins, game_over):
    """Calculate reward based on score, coins, and game state with improved weighting"""
    reward = 0
    
    # Base reward for surviving
    reward += REWARD_SURVIVAL
    
    # Dynamic reward for score increase - make it more meaningful
    score_diff = curr_score - prev_score
    if score_diff > 0:
        # Log-based reward to prevent excessive impact of large score jumps
        # which are often OCR errors
        log_score = np.log1p(score_diff) * REWARD_SCORE
        reward += min(log_score, 5.0)  # Cap to prevent extreme values
    
    # Reward for collecting coins
    coin_diff = curr_coins - prev_coins
    if coin_diff > 0:
        reward += coin_diff * REWARD_COIN
    
    # Penalty for game over
    if game_over:
        reward += PENALTY_CRASH
    
    return reward

def save_model(model, optimizer, episode, filename):
    """Save model checkpoint with error handling"""
    try:
        # Ensure directory exists
        os.makedirs(MODELS_DIR, exist_ok=True)
        checkpoint_path = os.path.join(MODELS_DIR, filename)
        
        # Move model to CPU before saving to prevent CUDA errors
        model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        
        # Save the checkpoint
        torch.save({
            'episode': episode,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        
        print(f"Model saved to {checkpoint_path}")
        return True
    except Exception as e:
        print(f"Error saving model: {e}")
        # Try an emergency save with just the state dict
        try:
            emergency_path = os.path.join(MODELS_DIR, f"emergency_{int(time.time())}.pt")
            torch.save(model.state_dict(), emergency_path)
            print(f"Emergency model save to {emergency_path}")
        except:
            print("Emergency save also failed")
        return False

def load_model(model, optimizer, filename):
    """Load model checkpoint with better error handling and device management"""
    checkpoint_path = os.path.join(MODELS_DIR, filename)
    try:
        if os.path.exists(checkpoint_path):
            # Load the checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Load model state dict
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state dict
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                # Move optimizer parameters to the correct device
                for param_group in optimizer.param_groups:
                    for param in param_group['params']:
                        if param.grad is not None:
                            param.grad.data = param.grad.data.to(device)
            
            episode = checkpoint.get('episode', 0)
            print(f"Loaded model from {checkpoint_path} (episode {episode})")
            
            # Ensure model is on the correct device
            model.to(device)
            
            return episode
        else:
            print(f"No checkpoint found at {checkpoint_path}, starting from scratch")
            return 0
    except Exception as e:
        print(f"Error loading model from {checkpoint_path}: {e}")
        print("Starting from scratch")
        return 0

class YOLODetector:
    """YOLO object detector for Subway Surfers (placeholder - optional feature)"""
    
    def __init__(self, model_path=None):
        self.model = None
        
        if model_path and os.path.exists(model_path):
            try:
                # Placeholder for YOLOv5 (or other object detection) integration
                # For integration with YOLOv5 or similar, uncomment and modify:
                # import torch.hub
                # self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
                print(f"Loaded YOLO model from {model_path}")
            except Exception as e:
                print(f"Error loading YOLO model: {e}")
    
    def detect_objects(self, image):
        """Detect objects in the image"""
        if self.model is None:
            return []
            
        # Placeholder for object detection implementation
        # For integration with YOLOv5 or similar, uncomment and modify:
        # results = self.model(image)
        # return results.pandas().xyxy[0].to_dict('records')
        
        return []