# Subway Surfers Reinforcement Learning Agent

A deep reinforcement learning agent that learns to play Subway Surfers directly from screen captures. This advanced implementation combines computer vision, deep Q-learning, and browser automation to train an AI to master the game.

![Subway Surfers RL](https://raw.githubusercontent.com/username/subway-surfers-rl/main/docs/subway_surfers_rl.png)

## Overview

This project implements a reinforcement learning agent that learns to play Subway Surfers by:
- Automatically controlling the browser-based game
- Capturing and processing screen frames in real-time
- Adapting hyperparameters based on performance
- Using memory-efficient algorithms for limited hardware
- Providing detailed visualizations and evaluations

## Features

### Advanced RL Algorithms
- **Deep Q-Learning (DQN)** with state-of-the-art variants:
  - Dueling DQN architecture for better value estimation
  - Double DQN algorithm to reduce overestimation bias
  - Prioritized Experience Replay for efficient learning
  - Memory-efficient implementations optimized for 4GB VRAM
  - Adaptive hyperparameter tuning based on performance

### Computer Vision Integration
- **Frame stacking** for temporal information
- **CLAHE preprocessing** for improved state representation
- **OCR and computer vision** for score and coin tracking
- **Multi-method game over detection** for reliable episodic training

### User Experience
- **Comprehensive visualization** of training progress
- **Split-screen mode** to observe both terminal output and gameplay
- **Real-time decision visualization** during testing
- **Detailed evaluation reports** with performance metrics
- **Simple command-line interface** for training, testing, and evaluation

## Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA-compatible GPU recommended (works with 4GB VRAM)
- Chrome browser
- The following Python packages (see requirements.txt):
  - opencv-python
  - numpy 
  - matplotlib
  - selenium
  - pytesseract
  - pyautogui
  - tqdm
  - pandas
  - seaborn

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

## Quick Start

### Training

Start training with the simplified launcher script:
```bash
python run.py
```

For advanced options:
```bash
python run.py --episodes 500 --performance high --browser-position right
```

Or use the main script directly:
```bash
python main.py --max_episodes 500 --browser_position right --use_dueling --use_double --use_per
```

### Testing

Visualize a trained agent:
```bash
python run.py --mode test --checkpoint models/latest.pt --visualize
```

Record a video of the agent playing:
```bash
python test.py --checkpoint models/latest.pt --record --show_q_values
```

### Evaluation

Evaluate agent performance with detailed metrics:
```bash
python run.py --mode evaluate --checkpoint models/latest.pt --episodes 10
```

Compare with a previous evaluation:
```bash
python evaluate.py --checkpoint models/latest.pt --episodes 10 --compare evaluation_results/previous_eval
```

## Project Structure

The project is organized into several modules:

```
SubwaySurfersRL/
├── env/                      # Game environment code
│   ├── game_interaction.py   # Browser & game interaction
│   ├── capture_state.py      # Frame capture and processing
│   └── control.py            # Keyboard controls
│
├── model/                    # Deep learning models
│   ├── dqn.py                # DQN architectures (Standard, Dueling, Efficient)
│   ├── experience_replay.py  # Replay buffer implementations
│   ├── agent.py              # DQN and Adaptive agents
│   └── training.py           # Training loop with monitoring
│
├── utils/                    # Utility functions
│   ├── utils.py              # General utilities
│   ├── plot_utils.py         # Visualization functions
│   ├── plot.py               # Additional plotting utilities
│   ├── reward_tracker.py     # Reward analysis tools
│   └── test.py               # Test utilities
│
├── logs/                     # Training logs and checkpoints
├── debug_images/             # Debug images for visualization
├── models/                   # Trained models and results
├── evaluation_results/       # Detailed evaluation results
├── videos/                   # Recorded gameplay videos
│
├── main.py                   # Main training script
├── run.py                    # Simple launcher script
├── test.py                   # Test/visualization script
├── evaluate.py               # Evaluation script
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

## Command-Line Options

### Run Script Options

```bash
python run.py --help
```

Key options:
- `--mode`: Choose between `train`, `test`, or `evaluate`
- `--episodes`: Number of episodes to run
- `--checkpoint`: Path to a saved model checkpoint
- `--performance`: Performance profile (`low`, `medium`, `high`, or `auto`)
- `--browser-position`: Position of browser window (`left` or `right`)
- `--debug`: Enable debug mode with extra logging
- `--visualize`: Enable detailed state visualization

### Main Script Options

```bash
python main.py --help
```

Key options:
- `--max_episodes`: Maximum number of episodes to train for
- `--max_steps`: Maximum steps per episode
- `--save_interval`: Save checkpoint every N episodes
- `--eval_interval`: Evaluate agent every N episodes
- `--use_dueling`: Use Dueling DQN architecture
- `--use_double`: Use Double DQN algorithm
- `--use_per`: Use Prioritized Experience Replay
- `--frame_stack`: Number of frames to stack
- `--memory_efficient`: Use memory-efficient implementations
- `--adaptive`: Use adaptive hyperparameter tuning

## Performance Tips

1. **Hardware Recommendations**:
   - GPU with 4GB+ VRAM provides best performance
   - At least 8GB system RAM recommended
   - Modern multi-core CPU for game rendering and state processing

2. **Optimizing Training**:
   - Start with `--performance auto` to automatically detect optimal settings
   - Use `--memory_efficient` on systems with limited VRAM
   - Reduce `--frame_stack` to 3 on very limited systems

3. **Browser Settings**:
   - Ensure Chrome browser is up-to-date
   - Close other browser windows to reduce resource usage
   - Position the game window where it has clear visibility (`--browser-position right`)

## Understanding Results

The agent produces several types of outputs:

1. **Training Logs**: Found in `logs/` directory, showing rewards, losses, and other metrics

2. **Checkpoints**: Saved in `logs/checkpoints/` with a copy in `models/` for the latest version

3. **Debug Images**: Stored in `debug_images/` showing game frames, states, and OCR detection

4. **Evaluation Reports**: Comprehensive HTML reports in `evaluation_results/` with:
   - Performance metrics (rewards, scores, episode lengths)
   - Action distribution analysis
   - Reward component breakdown
   - Comparison with previous evaluations
   - Visualizations and plots

## Customization

You can customize the screen regions for game detection by modifying the constants in `env/game_interaction.py`:

```python
self.game_region = (1094, 178, 806, 529)    # (x, y, width, height)
self.score_region = (1682, 159, 225, 48)    # (x, y, width, height)
self.coin_region = (1682, 217, 225, 48)     # (x, y, width, height)
```

For testing different region settings without modifying the code, use:
```bash
python utils/test.py
```

## Troubleshooting

### Common Issues

1. **Game not detected**:
   - Verify screen regions in `env/game_interaction.py`
   - Use `utils/test.py` to visualize and adjust regions

2. **OCR failures**:
   - Ensure Tesseract OCR is properly installed
   - Check debug images in `debug_images/scores/` and `debug_images/coins/`

3. **Browser issues**:
   - Update Chrome to the latest version
   - Try `--browser-position left` if right doesn't work
   - Check if `selenium` and `webdriver-manager` are up-to-date

4. **Memory errors**:
   - Use `--memory_efficient` flag
   - Reduce `--batch_size` to 16 or lower
   - Decrease `--memory_capacity` to 5000

### Debug Mode

For detailed debugging information:
```bash
python run.py --debug
```

This enables:
- Verbose logging
- Saving of debug images
- Detailed error messages
- Performance statistics

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI Gym for inspiration on environment design
- DeepMind for their pioneering work on DQN
- Poki for providing a browser-based version of Subway Surfers
- Game: (1094, 178, 806. 529) Score: (1682, 159.,225, 48) Coins: (1682, 217, 225, 48)