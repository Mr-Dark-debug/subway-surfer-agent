# Project Plan: Training RL Model to Play Subway Surfers

This project aims to train a reinforcement learning model to play Subway Surfers using **PyTorch** on a local machine with **RTX 3050ti GPU** (4GB VRAM) and **16GB RAM**. The model will interact with the game, continuously learn, and improve its ability to play autonomously.

## Project Overview

- **Goal**: Build a model that plays Subway Surfers using RL.
- **Framework**: **PyTorch** for model building and training.
- **Techniques**: Reinforcement Learning (RL), Deep Q-Learning (DQN), Experience Replay, Self-Play.
- **Hardware**: RTX 3050 Ti GPU, 16GB RAM, Local system.
- **Game Interface**: Browser-based game (Poki - Subway Surfers).
  
---

## Project Breakdown

### 1. **Environment Setup**

#### 1.1 **Game Interaction Setup**
- [ ] **Use PyAutoGUI** to simulate key presses (swipes, jumps, etc.) and restart actions.
- [ ] **Use Selenium** or **PyAutoGUI** to capture game screen frames.
- [ ] Create a Python script to interact with the browser window and start the game.
- [ ] Implement **Game Over detection** (using YOLOv5 or image matching).
- [ ] Implement **automatic game restart** logic.

#### 1.2 **Game State Representation**
- [ ] Capture the game screen and convert it into a usable state for the model (e.g., grayscale image).
- [ ] Ensure proper frame resizing and normalization for the model’s input.
- [ ] **Use YOLOv5** to detect locations of score, coins, and potentially obstacles/game over elements.
- [ ] **Use OCR** (e.g., Tesseract) to extract numerical score and coin count from detected regions.

#### 1.3 **Game Control Interface**
- [ ] Map game controls (up, down, left, right, jump) to actions in the model.

---

### 2. **Model Architecture**

#### 2.1 **Model Selection**
- [ ] Implement **Convolutional Neural Networks (CNN)** to process game frames.
- [ ] Use **Deep Q-Network (DQN)** for reinforcement learning (RL) setup.
  - [ ] Use **Experience Replay** to store past states, actions, rewards.
  - [ ] Implement **Target Networks** to stabilize training.

#### 2.2 **Training Setup**
- [ ] Design **reward function** based on the game’s objectives (time survived, coins collected, etc.).
  - [ ] Reward for surviving longer and collecting coins.
  - [ ] Penalty for crashing (game-over).
  
#### 2.3 **Learning Algorithm**
- [ ] Use **Epsilon-Greedy Strategy** for balancing exploration vs. exploitation.
- [ ] Implement **Decaying Epsilon** to reduce randomness as the model learns.

---

### 3. **Training Loop and Model Learning**

#### 3.1 **Reinforcement Learning Setup**
- [ ] Create training loop for **model training**.
- [ ] Use **Q-Learning** to estimate future rewards based on actions taken.
- [ ] Implement **Experience Replay Buffer** to store state transitions and improve sample efficiency.

#### 3.2 **Model Update & Continuous Learning**
- [ ] Set up online learning where the model updates after each game (iteration).
- [ ] Regularly save the model after each epoch for incremental learning.
- [ ] Implement **model checkpoints** to avoid losing progress in case of interruptions.

#### 3.3 **Reward Function Adjustment**
- [ ] Monitor model's progress and adjust the reward function for better training.

---

### 4. **Performance Monitoring & Evaluation**

#### 4.1 **Model Evaluation**
- [ ] Set up automatic evaluation based on:
  - Survival time.
  - Number of coins collected.
  - Distance traveled.
  
#### 4.2 **Logging**
- [ ] Implement logging for tracking:
  - Training performance (reward per episode, survival time).
  - Model performance (accuracy, loss).
  
#### 4.3 **Performance Metrics**
- [ ] Track the average performance across multiple runs.
- [ ] Compare the model performance to a baseline (e.g., random actions).
  
#### 4.4 **Hyperparameter Tuning**
- [ ] Adjust hyperparameters like learning rate, batch size, etc., for better performance.
  
---

### 5. **Folder Structure & Code Organization**

```
/subwaysurferai
│
├── /env                    # Game environment setup
│   ├── game_interaction.py  # PyAutoGUI, Selenium-based game interaction
│   ├── capture_state.py     # Frame capturing and pre-processing
│   └── control.py           # Simulating key presses
│
├── /model                   # Model definition and training code
│   ├── dqn.py               # DQN model class
│   ├── experience_replay.py # Experience replay buffer
│   ├── agent.py             # Agent class for interacting with the environment
│   └── training.py          # Training loop and setup
│
├── /logs                    # Training logs and checkpoints
│   ├── performance_logs.txt # Logs for performance metrics
│   └── checkpoints/         # Model checkpoints
│
├── /utils                   # Helper functions
│   ├── utils.py             # Utility functions (e.g., frame resizing, data augmentation)
│   └── plot.py              # Plotting performance graphs
│
└── requirements.txt         # Dependencies for the project
```

---

### 6. **Utilities & Tools**

#### 6.1 **Libraries & Dependencies**
- [ ] **PyTorch** for deep learning and model training.
- [ ] **NumPy** for numerical operations and data handling.
- [ ] **PyAutoGUI** or **Selenium** for browser-based game interaction.
- [ ] **Matplotlib** for performance plotting.
- [ ] **Gym** for creating RL environments (optional, can build custom env).
- [ ] **OpenCV (cv2)** for image processing (resizing, grayscale, cropping, template matching).
- [ ] **YOLOv5** (via Ultralytics package) for object detection.
- [ ] **Pytesseract** (and Tesseract OCR engine) for reading text from images.

#### 6.2 **Utilities**
- [ ] **Frame Preprocessing**: Implement preprocessing steps like resizing, grayscale conversion, and normalization of game frames.
- [ ] **Logging Utility**: Create logging functions to track the model's performance, rewards, and training progress.

---

### 7. **Techniques and Methods**

#### 7.1 **Reinforcement Learning Algorithms**
- [ ] **Deep Q-Learning (DQN)**:
  - Q-value approximation with CNN.
  - Use target networks to reduce instability during training.
- [ ] **Experience Replay** to store state-action transitions.
  
#### 7.2 **Model Training Techniques**
- [ ] **Epsilon-Greedy Strategy** for action selection during training.
- [ ] **Decaying Epsilon** to slowly shift focus from exploration to exploitation.

#### 7.3 **Optimization Methods**
- [ ] **Adam Optimizer** for training the model.
- [ ] **Learning Rate Scheduling** for stable and gradual training.

---

### 8. **Milestones & Checkpoints**

#### 8.1 **Initial Setup**
- [ ] Set up the environment and game interaction.
- [ ] Test basic frame capture and control simulation.

#### 8.2 **Model Architecture**
- [ ] Define CNN-based DQN model.
- [ ] Implement Experience Replay buffer.
- [ ] Implement Q-Learning algorithm.

#### 8.3 **Training Phase 1**
- [ ] Train model for basic actions.
- [ ] Evaluate model's performance.

#### 8.4 **Refinement**
- [ ] Fine-tune hyperparameters and reward function.
- [ ] Increase training duration and evaluate the model after each run.

#### 8.5 **Model Evaluation**
- [ ] Set benchmarks for the model’s performance.
- [ ] Monitor model’s ability to improve over time.

#### 8.6 **Final Testing & Integration**
- [ ] Ensure continuous learning functionality.
- [ ] Save trained models at regular intervals.

---

### 9. **To-Do List**

- [ ] **Set up the environment** and interaction scripts.
- [ ] **Capture and label data** for YOLOv5 (score area, coin area, game over elements).
- [ ] **Train YOLOv5 model** on custom data.
- [ ] **Implement YOLOv5 inference** in the game loop.
- [ ] **Implement OCR** to read score/coins from detected areas.
- [ ] **Implement Game Over detection and restart logic**.
- [ ] **Implement DQN model** with CNN layers.
- [ ] **Build Experience Replay buffer**.
- [ ] **Define reward function** incorporating score, coins, survival time, and game over penalty.
- [ ] **Train the agent** and log performance.
- [ ] **Evaluate performance** and adjust the reward function.
- [ ] **Tune hyperparameters** and continuously improve the model.

---

### 10. **Challenges & Considerations**

- **Training Time**: This could take a significant amount of time depending on how long the agent takes to learn the game mechanics.
- **GPU Memory**: Since your GPU has 4GB of VRAM, make sure the model is optimized for memory usage (e.g., reducing batch size or resolution if necessary).
- **Real-time Control**: Real-time control and precise actions in the game can be tricky, so careful fine-tuning of the reward function is required.

---

### 11. **Future Improvements**

- [ ] **Transfer Learning**: Leverage pre-trained models for feature extraction.
- [ ] **Advanced RL Techniques**: Explore PPO or A3C for better training stability.
- [ ] **Multi-Agent Training**: Implement multi-agent setups if you want more complex learning scenarios.

---

This plan serves as a guide to implement your reinforcement learning agent for Subway Surfers. You can track progress with checkboxes and milestones and ensure that each aspect of the project is covered.