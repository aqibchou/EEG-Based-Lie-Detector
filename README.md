# Neuropawn Lie Detector

A lie detection system using the Neuropawn Knight Board with:
- **Hybrid CNN-RNN-Transformer** model for advanced temporal pattern recognition
- **Synchronized speech detection** for context-aware analysis

## Overview

This project implements a lie detector that:
- Connects to the Neuropawn Knight Board via BrainFlow
- **Synchronizes speech detection with EEG data collection** using Silero VAD
- Collects EEG signals during truth and lie conditions
- Maps speech segments to corresponding brain signals
- Uses **Hybrid CNN-RNN-Transformer** deep learning model that combines:
  - CNNs for spatial feature extraction
  - RNNs (LSTM/GRU) for temporal sequence modeling
  - Transformer-style attention for temporal weighting
- Trains classifiers to predict truth vs. lie based on EEG patterns

### Key Feature: Speech-EEG Synchronization

The system can simultaneously record speech and EEG signals, mapping speech segments to corresponding brain activity. This allows you to:
- Analyze the relationship between speech context and brain signals
- Compare EEG patterns when the same situation is mentioned truthfully vs. when lied about
- Study how brain signals differ based on the content being spoken

**Thesis**: There should be a relationship in the difference in signals when the context of the situation is mentioned to when the user lies vs when the user tells the truth.

### Hybrid CNN-RNN-Transformer Model

The system uses a **Hybrid Deep Learning Architecture** that combines:
- **Spatial CNN**: Extracts spatial features from EEG channels (treats channels as spatial dimensions)
- **Channel Attention**: Weights important EEG channels
- **Temporal RNN**: Models temporal sequences using LSTM or GRU (bidirectional)
- **Temporal Attention**: Transformer-style attention to weight important time points
- **Feature Fusion**: Combines spatial and temporal features for classification

**Key Features**:
- Explicit spatial and temporal feature extraction
- Modular architecture (similar to modern speech recognition systems)
- Handles full-length sequences (250+ timepoints)
- Better for longer temporal dependencies

## Requirements

- Python 3.7+
- Neuropawn Knight Board device
- USB connection to your computer

## Installation

1. Install the required Python packages:

```bash
pip install -r requirements.txt
```

2. Ensure you have the correct serial port permissions:
   - **Windows**: Usually no special permissions needed
   - **Linux**: You may need to add your user to the `dialout` group:
     ```bash
     sudo usermod -a -G dialout $USER
     ```
   - **macOS**: Usually no special permissions needed, but use `/dev/cu.*` ports, not `/dev/tty.*`

## Hardware Setup

### Neuropawn Knight Board Configuration

1. **Connect your Neuropawn Knight Board to your computer via USB**

2. **Electrode Placement**: The following electrode positions were used to measure brainwave activity:
   - **F3, F5**: Left frontal regions
   - **F4, F6**: Right frontal regions
   - **P3, CP3**: Left parietal/central-parietal regions
   - **P4, CP4**: Right parietal/central-parietal regions
   
   These 8 channels provide coverage of frontal and parietal brain regions, which are important for cognitive processing and decision-making tasks.

3. **Identify the serial port**:
   - **Windows**: Check Device Manager or use `COM3`, `COM4`, etc.
   - **macOS**: Use `/dev/cu.usbserial-*` or `/dev/cu.usbmodem*` (NOT `/dev/tty.*`)
   - **Linux**: Usually `/dev/ttyUSB0` or `/dev/ttyACM0`

## Usage

### Basic Board Initialization

```python
from utils.board_initializer import NeuropawnKnightBoard

# Initialize board (replace with your serial port)
board = NeuropawnKnightBoard(serial_port="/dev/cu.usbserial-*")  # macOS
# board = NeuropawnKnightBoard(serial_port="COM3")  # Windows
# board = NeuropawnKnightBoard(serial_port="/dev/ttyUSB0")  # Linux

# Connect
board.connect()

# Start streaming
board.start_streaming()

# Collect data for 10 seconds
import time
time.sleep(10)

# Get data
data = board.get_data()
print(f"Data shape: {data.shape}")

# Stop and disconnect
board.stop_streaming()
board.disconnect()
```

### Using Context Manager

```python
from utils.board_initializer import NeuropawnKnightBoard
import time

with NeuropawnKnightBoard(serial_port="/dev/cu.usbserial-*") as board:
    board.get_board_info()
    board.start_streaming()
    time.sleep(10)
    data = board.get_data()
    print(f"Collected {data.shape[1]} samples")
```

### Two-Phase Training Data Collection

The system uses a **question/answer phase** approach for training:

```python
from detectors.lie_detector_hybrid import HybridLieDetectorSystem

detector = HybridLieDetectorSystem(
    serial_port="/dev/cu.usbserial-*",
    enable_speech=True,
    rnn_type='LSTM'  # or 'GRU'
)

detector.connect()

# Collect training data (same two-phase approach)
X, y, phase_data = detector.collect_training_data(
    n_questions=10,
    answer_duration=20.0
)

# Train the hybrid model
detector.train(X, y, epochs=30, batch_size=16, learning_rate=0.001)

# Save and use
detector.save_model('hybrid_lie_detector_model.pth')
result = detector.predict(duration=5.0)

detector.disconnect()
```

### Running the Example Script

```bash
# Test board connection first
python tests/test_connection.py

# Run two-phase training example
python examples/example_two_phase_training.py /dev/cu.usbserial-*

# Or on Windows
python examples/example_two_phase_training.py COM3
```

**Hybrid Model Advantages**:
- Explicit spatial (CNN) and temporal (RNN) feature extraction
- Transformer-style attention for temporal weighting
- Modular architecture for interpretability
- Handles full-length sequences with long temporal dependencies

## Saving and Loading Training Data

### Saving Training Data

Training data is **NOT automatically saved** by `collect_training_data()`. You need to manually save it:

```python
from detectors.lie_detector_hybrid import HybridLieDetectorSystem
import numpy as np

detector = HybridLieDetectorSystem(serial_port="/dev/cu.usbserial-*", enable_speech=True)
detector.connect()

# Collect data
X, y, phase_data = detector.collect_training_data(
    n_questions=5,
    answer_duration=10.0,
    use_speech=True
)

# Save to file
save_data = {
    'qa_pairs': X,
    'labels': y,
    'sampling_rate': phase_data.get('sampling_rate', detector.sampling_rate),
    'n_channels': phase_data.get('n_channels', detector.n_channels),
    'structure': 'unified_qa_pairs'
}

np.savez_compressed('my_training_data.npz', **save_data, allow_pickle=True)
print(f"✓ Saved training data")
```

### Loading Saved Data

```python
import numpy as np

# Load data
data = np.load('my_training_data.npz', allow_pickle=True)

X = data['qa_pairs']              # Q-A pairs
y = data['labels']                 # Labels
sampling_rate = data.get('sampling_rate', 125)
n_channels = data.get('n_channels', 8)

# Access first pair
first_pair = X[0]
eeg = first_pair['eeg_continuous']  # Combined Q-A EEG
speech_segments = first_pair['speech_segments']  # Speech segments
metadata = first_pair['metadata']  # Transition info, durations, labels
```

### Data Structure

Each training sample (Q-A pair) contains:
- **`eeg_continuous`**: `(n_channels, total_samples)` - Concatenated [Question Phase EEG | Answer Phase EEG]
- **`speech_segments`**: List of speech segments with mapped EEG, each containing:
  - `speech_segment`: Audio data
  - `eeg_segment`: Corresponding EEG data
  - `phase`: 'question' or 'answer'
  - `start_time`, `end_time`: Timestamps
- **`metadata`**: Dictionary with:
  - `question_num`: Question number
  - `label`: 'truth' or 'lie'
  - `transition_time`: Time when phase transition occurred
  - `transition_sample`: Sample index of transition
  - `question_samples`, `answer_samples`: Number of samples in each phase
  - `question_duration`, `answer_duration`: Durations in seconds

## Project Structure

```
Neuropawn/
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── models/                       # Deep learning models
│   ├── __init__.py
│   └── hybrid_eeg_model.py      # Hybrid CNN-RNN-Transformer model
├── detectors/                    # Lie detection systems
│   ├── __init__.py
│   └── lie_detector_hybrid.py   # Hybrid model lie detector
├── utils/                        # Utility modules
│   ├── __init__.py
│   ├── board_initializer.py     # Board connection and management
│   ├── speech_eeg_synchronizer.py # Speech-EEG synchronization
│   └── list_audio_devices.py    # Audio device selection helper
├── examples/                     # Example scripts
│   └── example_two_phase_training.py # Main training example
├── tests/                        # Test scripts
│   ├── test_connection.py       # Board connection test
│   └── test_hybrid_workflow.py  # ML pipeline test (simulated data)
└── data/                         # Data files
    └── test_speech_eeg_sync_data.npz
```

## Important Notes

### Serial Port Selection

- **macOS**: You MUST use `/dev/cu.*` ports, NOT `/dev/tty.*` ports
- **Linux**: Use `/dev/tty.*` ports, but you may need sudo or proper permissions
- **Windows**: Use `COM3`, `COM4`, etc.

### Data Collection Tips

1. **Training Data Quality**: The accuracy of your lie detector depends heavily on the quality of training data
   - Collect multiple sessions of truth/lie data
   - Ensure consistent conditions (same environment, same person)
   - Longer collection times generally improve accuracy

2. **EEG Signal Quality**:
   - Ensure good electrode contact
   - Minimize movement during data collection
   - Avoid electrical interference

## Results

### Model Performance

The Hybrid CNN-RNN-Transformer model achieved the following results:
- **Training Accuracy**: 96%
- **Validation Accuracy**: 91%

These results demonstrate strong performance on the training dataset, with the model successfully learning to distinguish between truth and lie conditions.

## Important Disclaimers

### Subject-Specific Training Required

** CRITICAL**: This model performs **poorly when the subject is changed**. Testing showed that accuracy dropped to as low as **71%** when applied to a different subject than the one used for training.

**Why this happens**: Everyone has different brainwave responses to the same actions. The neural patterns associated with truth-telling and lying are highly individualized, making cross-subject generalization challenging.

**Recommendations**:
- **Train with your own data**: Each user should collect and train on their own EEG data
- **Recommended dataset size**: **250 pairs of truths and lies** (125 truth pairs + 125 lie pairs) for robust model performance
- **Subject-specific models**: Do not use a model trained on one person to predict for another person

### Limitations

**1. Unrealistic Testing Conditions**
- The testing environment is not representative of real-world high-stakes situations
- Participants may not experience the same level of nervousness, stress, or emotional response as they would in actual high-pressure scenarios
- This limitation may skew results and affect real-world applicability

**2. General Limitations**
- Lie detection using EEG is not 100% accurate
- Results may vary significantly between individuals
- Requires proper training data collection
- Environmental factors can affect signal quality
- This is for research/educational purposes only
- Not suitable for legal or forensic applications without extensive validation

## Troubleshooting

### Connection Issues

1. **"Error connecting to board"**:
   - Check USB connection
   - Verify serial port is correct
   - On macOS, ensure you're using `/dev/cu.*` not `/dev/tty.*`
   - On Linux, try running with `sudo` or check permissions

2. **"Permission denied"** (Linux):
   ```bash
   sudo usermod -a -G dialout $USER
   # Then log out and log back in
   ```

3. **"Port not found"**:
   - List available ports:
     - macOS: `ls /dev/cu.*`
     - Linux: `ls /dev/tty*`
     - Windows: Check Device Manager

### Data Collection Issues

1. **No data collected**:
   - Ensure `start_streaming()` is called before collecting
   - Check that the board is properly connected
   - Verify the sampling rate is correct

2. **Poor classification accuracy**:
   - Collect more training data
   - Ensure consistent conditions during training
   - Try adjusting model hyperparameters (fragment_size, batch_size, learning_rate)
   - Check signal quality (noise, artifacts)

## Speech-EEG Synchronization

The project uses **Silero VAD** ([GitHub](https://github.com/snakers4/silero-vad)) for voice activity detection:
- Pre-trained neural network for speech detection
- Real-time processing with low latency
- Works offline, no internet required
- Maps speech segments to EEG data with precise timing

### How It Works

The `SpeechEEGSynchronizer` class handles:
- Simultaneous audio and EEG recording
- Real-time speech detection (processes audio in 512-sample chunks)
- Temporal synchronization between speech and brain signals
- Data structure mapping speech segments to corresponding EEG segments

**Synchronization Process**:
1. Start Recording: Both audio and EEG streams start simultaneously
2. Real-time Processing: Audio processed in chunks, VAD detects speech, EEG collected continuously
3. Temporal Mapping: Each speech segment mapped to EEG data using timestamps
4. Data Structure: Creates synchronized pairs of (speech_segment, eeg_segment)

### Two-Phase Training Workflow

The system uses a **question/answer phase** approach:

**For Each Training Question**:
1. **QUESTION PHASE**: Recording starts automatically when context/question is presented
   - Focus: Detect brain activity when context is mentioned
   - Press **SPACEBAR** to transition to answer phase
2. **ANSWER PHASE**: User responds (truth or lie)
   - Recording continues
   - Press **ANY KEY** to end (or auto-ends after `answer_duration` if specified)
   - Question + Answer phases stored together as one training sample

**Key Features**:
- **Interactive Phase Transition**: No hardcoded timestamps - transition happens when you press SPACEBAR
- **Flexible Timing**: Question phase can be any duration
- **User-Controlled**: You decide when to start answering
- **Unified Storage**: Each training sample contains both question and answer phases concatenated

### Audio Device Selection

**List Available Devices**:
```bash
python utils/list_audio_devices.py
```

**Or in Python**:
```python
from detectors.lie_detector_hybrid import HybridLieDetectorSystem

devices = HybridLieDetectorSystem.list_audio_devices()
for device in devices:
    print(f"[{device['index']}] {device['name']}")
```

**Select Specific Device**:
```python
detector = HybridLieDetectorSystem(
    serial_port="/dev/cu.usbserial-*",
    enable_speech=True,
    speech_input_device_index=2  # Use device at index 2
)
```

**Default**: If `speech_input_device_index` is not specified, uses system default microphone.

## Hybrid Model Architecture Details

### Complete Data Flow

```
Raw EEG: (batch, n_channels, time)
    ↓
[Spatial CNN] → Extracts spatial features from channels
    ↓
[Channel Attention] → Weights important channels
    ↓
[Temporal RNN] → Models temporal sequences (LSTM/GRU, bidirectional)
    ↓
[Temporal Attention] → Transformer-style attention for time points
    ↓
[Feature Fusion] → Combines spatial + temporal features
    ↓
[Classification] → Truth/Lie prediction
```

### Architecture Components

**1. Spatial CNN**
- Treats EEG channels as spatial dimensions (like topographic maps)
- Uses 1D convolutions to capture channel relationships
- Architecture: `Conv1d → BatchNorm → ReLU → Dropout` (2 layers)
- Output: Spatial feature maps

**2. Channel Attention**
- Identifies which channels are most discriminative
- Global average and max pooling across time
- Learns channel importance weights
- Applies weights to feature maps

**3. Temporal RNN (LSTM/GRU)**
- Models temporal sequences in EEG signals
- Captures time-varying patterns
- Handles long-term dependencies
- Bidirectional for forward and backward context
- Configurable: `rnn_type='LSTM'` or `'GRU'`

**4. Temporal Attention (Transformer-style)**
- Uses multi-head self-attention
- Identifies critical moments in the signal
- Learns which time points matter most
- Similar to attention in Transformers

**5. Feature Fusion**
- Concatenates spatial (CNN) and temporal (RNN+Attention) features
- Applies fusion layers with LayerNorm
- Creates unified representation

**6. Classification**
- Global pooling over time
- Fully connected layers
- Output: Truth/Lie probabilities

### Why This Architecture?

- **CNNs**: Extract spatial features from EEG channels (treats channels as spatial dimensions)
- **RNNs**: Capture temporal sequences crucial for understanding time-varying signals
- **Transformers**: Weight important channels and time points, similar to modern speech recognition systems
- **Modular Design**: Each component has a clear purpose, making the model interpretable

## References

- [BrainFlow Documentation](https://brainflow.readthedocs.io/)
- [BrainFlow GitHub](https://github.com/brainflow-dev/brainflow)
- [Neuropawn Getting Started](https://www.neuropawn.tech/getting-started/)
- [Silero VAD GitHub](https://github.com/snakers4/silero-vad)
- Hybrid Model: Combines CNN, RNN, and Transformer architectures for EEG lie detection

## License

This project is for educational and research purposes.

## Contributing

Feel free to submit issues or pull requests for improvements!

