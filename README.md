# Subway Surfers Reinforcement Learning Agent

A deep reinforcement learning agent that learns to play Subway Surfers directly from screen captures. This project uses a combination of computer vision, deep Q-learning, and browser automation to train an AI to master the game.

![Subway Surfers RL](https://raw.githubusercontent.com/username/subway-surfers-rl/main/docs/subway_surfers_rl.png)

## Features

- **Deep Q-Learning (DQN)** with advanced variants:
  - Dueling DQN architecture
  - Double DQN algorithm
  - Prioritized Experience Replay
- **Frame stacking** for temporal information
- **Real-time game interaction** using Selenium and PyAutoGUI
- **Computer vision** for game state analysis, including score and coin tracking
- **Comprehensive visualization** of training progress
- **Split-screen mode** to observe both terminal output and gameplay simultaneously
- **Automatic checkpointing** for resuming training
- **Detailed reward component analysis**

## Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA-compatible GPU recommended (min. 4GB VRAM)
- Chrome browser

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/subway-surfers-rl.git
   cd subway-surfers-rl
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install Tesseract OCR for score detection:
   - **Windows**: Download and install from [here](https://github.com/UB-Mannheim/tesseract/wiki)
   - **macOS**: `brew install tesseract`
   - **Linux**: `sudo apt install tesseract-ocr`

## Configuration

The code is designed to work with any browser-based version of Subway Surfers. The default URL is set to the Poki version, but you can change it in the command line arguments.

## Usage

### Basic Training

```bash
python main.py --browser_position right
```

This will:
1. Open Chrome browser on the right side of your screen
2. Navigate to Subway Surfers
3. Train the agent with default parameters

### Training with Advanced Features

```bash
python main.py --use_dueling --use_double --use_per --browser_position right
```

This enables all advanced DQN features for better training performance.

### Full Command Line Options

```
usage: main.py [-h] [--game_url GAME_URL] [--browser_position {left,right}]
              [--max_episodes MAX_EPISODES] [--max_steps MAX_STEPS]
              [--save_interval SAVE_INTERVAL] [--eval_interval EVAL_INTERVAL]
              [--visual_feedback] [--use_dueling] [--use_double] [--use_per]
              [--frame_stack FRAME_STACK] [--load_checkpoint LOAD_CHECKPOINT]
              [--learning_rate LEARNING_RATE] [--gamma GAMMA]
              [--batch_size BATCH_SIZE] [--target_update TARGET_UPDATE]
              [--memory_capacity MEMORY_CAPACITY] [--epsilon_start EPSILON_START]
              [--epsilon_end EPSILON_END] [--epsilon_decay EPSILON_DECAY]
              [--debug] [--record_video]
```

### Common Options

- `--browser_position {left,right}`: Position the browser window for split-screen viewing
- `--max_episodes`: Maximum number of episodes for training (default: 1000)
- `--visual_feedback`: Show visual feedback during training
- `--debug`: Enable debug mode with extra logging and visualizations
- `--record_video`: Record a video of gameplay after training
- `--load_checkpoint PATH`: Path to a checkpoint file to resume training

### Resume Training

To resume training from a checkpoint:

```bash
python main.py --load_checkpoint logs/checkpoints/dqn_episode_100_20240417_123456.pt
```

## Project Structure

```
subwaysurferai/
├── env/                      # Game environment code
│   ├── game_interaction.py   # Browser & game interaction
│   ├── capture_state.py      # Frame capture and processing
│   └── control.py            # Keyboard controls
│
├── model/                    # Deep learning models
│   ├── dqn.py                # DQN architectures
│   ├── experience_replay.py  # Replay buffer implementations
│   ├── agent.py              # DQN agent
│   └── training.py           # Training loop
│
├── utils/                    # Utility functions
│   ├── utils.py              # General utilities
│   ├── plot_utils.py         # Visualization functions
|   |── plot.py
│   └── reward_tracker.py     # Reward analysis tools
│
├── logs/                     # Training logs and checkpoints
│   ├── checkpoints/          # Saved models
│   └── videos/               # Gameplay recordings
│
├── debug_images/             # Debug screenshots and visualizations
├── main.py                   # Main training script
└── README.md                 # This file
```

## Training Process

1. **Environment Setup**: The system opens Chrome, positions it according to your preference, and navigates to Subway Surfers.

2. **Game Interaction**: The agent interacts with the game using keyboard controls, simulating player actions.

3. **State Representation**: Game screens are captured, converted to grayscale, resized to 84x84 pixels, and stacked (default: 4 frames) to provide temporal information.

4. **Reward System**: The agent receives rewards for:
   - Surviving (small positive reward per step)
   - Collecting coins (larger positive reward)
   - Increasing score (scaled positive reward)
   - Game over (negative reward penalty)

5. **Learning**: The DQN algorithm with experience replay learns to maximize these rewards through trial and error.

6. **Evaluation**: Periodically, the agent is evaluated without exploration to assess its performance.

## Debugging and Troubleshooting

If you encounter issues with game detection or interaction:

1. Enable debug mode with `--debug` to generate detailed logs and debug images
2. Check the `debug_images/` directory for visual diagnostics
3. Adjust the browser position if the game is not detected properly
4. Ensure your screen resolution is sufficient for split-screen viewing

## Common Issues

- **Game not starting**: Try running with `--debug` to see why the game isn't loading correctly
- **Poor performance**: Ensure your GPU is being utilized by checking the logs
- **Browser positioning**: If split-screen doesn't work well, try adjusting your display settings

## Future Improvements

- Support for other versions of Subway Surfers
- Implementation of additional RL algorithms (A3C, PPO)
- Transfer learning from pre-trained models
- More sophisticated computer vision for better game state detection

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI Gym for inspiration on environment design
- DeepMind for their pioneering work on DQN
- Poki for providing a browser-based version of Subway Surfers