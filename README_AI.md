# Subway Surfers AI

This project implements an AI agent that can play Subway Surfers using reinforcement learning with a Deep Q-Network (DQN) and computer vision techniques.

## Project Structure

The project follows a modular approach with separate components for different functionalities:

```
subwaysurferai/
├── config.py                 # Configuration settings
├── main_ai.py               # Main script to run the AI
├── train.py                 # Training script
├── play.py                  # Play script (existing)
├── subway_ai/               # AI modules
│   ├── __init__.py          # Package initialization
│   ├── screen_capture.py    # Screen capture and OCR
│   ├── game_control.py      # Game control and actions
│   ├── model.py             # DQN model and replay memory
│   └── detector.py          # YOLOv5 object detection
├── models/                  # Saved model checkpoints
├── screenshots/             # Captured screenshots
├── templates/               # Template images
└── logs/                    # Log files
```

## Features

- **Modular Design**: Separate modules for screen capture, game control, model, and detection
- **YOLOv5 Integration**: Object detection for coins, obstacles, and game over screen
- **OCR**: Text recognition for score and coin count
- **Reinforcement Learning**: DQN with experience replay for learning optimal gameplay
- **Swipe Controls**: Uses swipe gestures instead of arrow keys
- **Automatic Restart**: Detects game over and automatically restarts

## Requirements

- Python 3.7+
- PyTorch
- OpenCV
- PyAutoGUI
- Pytesseract (with Tesseract OCR installed)
- YOLOv5 (installed via torch.hub)

## Usage

### Training Mode

To train the AI agent:

```bash
python main_ai.py --mode train --episodes 1000
```

Options:
- `--episodes`: Number of episodes to train (default: 1000)
- `--model`: Load a previously trained model checkpoint
- `--yolo`: Specify a YOLOv5 model for object detection

### Play Mode

To play using a trained model:

```bash
python main_ai.py --mode play --model subway_dqn_best.pt --games 5
```

Options:
- `--games`: Number of games to play (default: 5)
- `--model`: Model checkpoint to use
- `--yolo`: Specify a YOLOv5 model for object detection

## Game Configuration

The AI is configured to work with the Subway Surfers game running at 600x800 resolution. The game regions are defined in `config.py` and can be adjusted if needed.

## Reward System

The AI uses a reward system that incentivizes:
- Survival time (small continuous reward)
- Collecting coins (larger reward)
- Score increases (proportional reward)
- Avoiding crashes (large negative penalty)

## Training Process

1. The AI captures the game screen and processes it for input to the neural network
2. It selects actions using an epsilon-greedy policy (balancing exploration and exploitation)
3. It observes the results of its actions (new state, score, coins, game over)
4. It calculates rewards and stores experiences in replay memory
5. It periodically trains the neural network using batches of experiences
6. It saves checkpoints of the model during training

## Notes

- The game should be running at `D:\Projects\Subway-Surfers-AI\Subway_Surfers.exe`
- The AI handles game over detection and automatically clicks the play button to restart
- For best results, ensure the game window is positioned correctly on screen

# Subway Surfers AI

A reinforcement learning AI that learns to play Subway Surfers using swipe controls.

## Features

- Captures game screen and extracts score and coin information using OCR
- Uses deep Q-learning to train an AI model to play the game
- Implements proper swipe gestures instead of keyboard controls
- Displays real-time training data
- Includes a calibration tool to easily set up screen regions
- Automatically handles game restarts when the player crashes

## Requirements

- Python 3.7+
- PyTorch
- OpenCV
- NumPy
- PyAutoGUI
- Pytesseract
- Tkinter (for calibration tool)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/subway-surfers-ai.git
   cd subway-surfers-ai
   ```

2. Install required packages:
   ```
   pip install torch torchvision opencv-python numpy pyautogui pytesseract pillow
   ```

3. Install Tesseract OCR:
   - Windows: Download and install from https://github.com/UB-Mannheim/tesseract/wiki
   - Linux: `sudo apt install tesseract-ocr`
   - macOS: `brew install tesseract`

4. Update the Tesseract path in `subway_ai/config.py` to match your installation.

## Setup

1. Update the game path in `subway_ai/config.py` if needed.

2. Create necessary directories:
   ```
   mkdir -p screenshots/gameplay screenshots/score screenshots/coins screenshots/game_over
   mkdir -p models templates logs
   ```

3. Calibrate the screen regions:
   ```
   python -m subway_ai.calibrate
   ```
   - Take a screenshot of the game
   - Define the game region, score region, and coin region by selecting them on the screenshot
   - Save the regions

4. Capture templates for game over and play button detection:
   ```
   python -m subway_ai.save_templates
   ```

## Usage

### Training the AI

```
python main_ai.py --mode train --episodes 100
```

Optional arguments:
- `--model <path>`: Load a previously trained model
- `--episodes <number>`: Number of episodes to train (default: 1000)

### Playing with a trained model

```
python main_ai.py --mode play --model subway_dqn_best.pt --games 5
```

Optional arguments:
- `--model <path>`: Path to the model file
- `--games <number>`: Number of games to play (default: 5)

### Calibrating

```
python main_ai.py --mode calibrate
```

## Project Structure

```
/subway-surfers-ai
├── subway_ai/
│   ├── __init__.py
│   ├── calibrate.py          # Screen region calibration tool
│   ├── config.py             # Configuration settings
│   ├── game_control.py       # Game control using swipe gestures
│   ├── model.py              # DQN model and related functions
│   ├── regions.json          # Saved screen regions
│   ├── save_templates.py     # Tool to save detection templates
│   └── screen_capture.py     # Screen capture and OCR
├── main_ai.py                # Main entry point
├── models/                   # Saved models
├── screenshots/              # Captured screenshots
├── templates/                # Detection templates
└── logs/                     # Log files
```

## Troubleshooting

1. **OCR not working properly**:
   - Make sure Tesseract is installed correctly and the path is set in config.py
   - Recalibrate the score and coin regions to ensure they're capturing the correct areas

2. **Swipe controls not working**:
   - Make sure the game window is in focus
   - Calibrate the game region correctly
   - Try adjusting the swipe distance in game_control.py

3. **Game not restarting automatically**:
   - Capture new templates for the play button detection
   - Ensure the game window is in focus

4. **CUDA/GPU issues**:
   - If you encounter CUDA errors, try running with CPU only by modifying the device setting

## Tips for Better Results

1. Calibrate the regions carefully for accurate OCR readings
2. Run the game in windowed mode
3. Use a consistent game window size
4. Let the AI train for at least 100 episodes for meaningful results
5. Save templates for game over and play button when the game is in the correct state

## Customization

- Adjust reward parameters in `config.py` to emphasize different aspects of gameplay
- Modify the model architecture in `model.py` for better performance
- Change epsilon decay rate to control exploration vs. exploitation balance

## License

This project is licensed under the MIT License.