"""
Test Hybrid Model Workflow with Simulated EEG and Speech Data
Tests the complete ML pipeline without requiring hardware
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from models.hybrid_eeg_model import HybridEEGLieDetector
from sklearn.model_selection import train_test_split
import time

def generate_simulated_eeg_data(
    n_samples: int,
    n_channels: int = 8,
    n_timepoints: int = 250,
    sampling_rate: int = 125,
    label: str = 'truth',
    noise_level: float = 0.1
) -> np.ndarray:
    """
    Generate simulated EEG data

    Args:
        n_samples: Number of samples to generate
        n_channels: Number of EEG channels
        n_timepoints: Number of time points per sample
        sampling_rate: Sampling rate (Hz)
        label: 'truth' or 'lie' - affects signal characteristics
        noise_level: Amount of noise to add

    Returns:
        Simulated EEG data (n_samples, n_channels, n_timepoints)
    """
    data = []

    for i in range(n_samples):

        t = np.linspace(0, n_timepoints / sampling_rate, n_timepoints)

        signal = np.zeros((n_channels, n_timepoints))

        for ch in range(n_channels):

            alpha_freq = 10 + np.random.uniform(-2, 2)
            alpha_amp = np.random.uniform(0.5, 1.5)

            beta_freq = 20 + np.random.uniform(-5, 5)
            beta_amp = np.random.uniform(0.3, 1.0)

            theta_freq = 6 + np.random.uniform(-1, 1)
            theta_amp = np.random.uniform(0.2, 0.8)

            signal[ch, :] = (
                alpha_amp * np.sin(2 * np.pi * alpha_freq * t) +
                beta_amp * np.sin(2 * np.pi * beta_freq * t) +
                theta_amp * np.sin(2 * np.pi * theta_freq * t)
            )

            if label == 'lie':

                stress_signal = 1.5 * np.sin(2 * np.pi * 25 * t) * np.exp(-t/2)
                signal[ch, :] += stress_signal

                if ch < 3:
                    signal[ch, :] *= 1.2

            noise = np.random.normal(0, noise_level, n_timepoints)
            signal[ch, :] += noise

            signal[ch, :] = (signal[ch, :] - signal[ch, :].mean()) / (signal[ch, :].std() + 1e-8)

        data.append(signal)

    return np.array(data)

def generate_simulated_qa_pairs(
    n_questions: int = 5,
    n_channels: int = 8,
    question_duration: float = 3.0,
    answer_duration: float = 5.0,
    sampling_rate: int = 125
) -> tuple:
    """
    Generate simulated Question-Answer pairs in the unified format

    Args:
        n_questions: Number of Q-A pairs
        n_channels: Number of EEG channels
        question_duration: Duration of question phase (seconds)
        answer_duration: Duration of answer phase (seconds)
        sampling_rate: Sampling rate (Hz)

    Returns:
        Tuple of (X, y, phase_data) matching the real data structure
    """
    X = []
    y = []
    phase_data_list = []

    question_samples = int(question_duration * sampling_rate)
    answer_samples = int(answer_duration * sampling_rate)

    for q_idx in range(n_questions):

        label = 'truth' if q_idx % 2 == 0 else 'lie'

        question_eeg = generate_simulated_eeg_data(
            n_samples=1,
            n_channels=n_channels,
            n_timepoints=question_samples,
            sampling_rate=sampling_rate,
            label='neutral',
            noise_level=0.1
        )[0]

        answer_eeg = generate_simulated_eeg_data(
            n_samples=1,
            n_channels=n_channels,
            n_timepoints=answer_samples,
            sampling_rate=sampling_rate,
            label=label,
            noise_level=0.1
        )[0]

        qa_combined = np.hstack([question_eeg, answer_eeg])

        qa_pair = {
            'eeg_continuous': qa_combined,
            'speech_segments': [
                {
                    'speech_segment': {
                        'start_time': 0.0,
                        'end_time': question_duration,
                        'duration': question_duration,
                        'audio_data': np.random.randn(int(16000 * question_duration)).astype(np.float32),
                        'speech_probability': 0.8
                    },
                    'eeg_segment': {
                        'start_time': 0.0,
                        'end_time': question_duration,
                        'start_sample': 0,
                        'end_sample': question_samples,
                        'eeg_data': question_eeg,
                        'sampling_rate': sampling_rate
                    },
                    'phase': 'question',
                    'label': label,
                    'question_num': q_idx + 1
                },
                {
                    'speech_segment': {
                        'start_time': question_duration,
                        'end_time': question_duration + answer_duration,
                        'duration': answer_duration,
                        'audio_data': np.random.randn(int(16000 * answer_duration)).astype(np.float32),
                        'speech_probability': 0.9
                    },
                    'eeg_segment': {
                        'start_time': question_duration,
                        'end_time': question_duration + answer_duration,
                        'start_sample': question_samples,
                        'end_sample': question_samples + answer_samples,
                        'eeg_data': answer_eeg,
                        'sampling_rate': sampling_rate
                    },
                    'phase': 'answer',
                    'label': label,
                    'question_num': q_idx + 1
                }
            ],
            'metadata': {
                'question_num': q_idx + 1,
                'label': label,
                'transition_time': question_duration,
                'transition_sample': question_samples,
                'question_samples': question_samples,
                'answer_samples': answer_samples,
                'total_samples': qa_combined.shape[1],
                'question_duration': question_duration,
                'answer_duration': answer_duration,
                'n_question_segments': 1,
                'n_answer_segments': 1
            }
        }

        X.append(qa_pair)
        y.append(label)

        phase_data_list.append({
            'speech_segments': qa_pair['speech_segments'],
            'metadata': qa_pair['metadata']
        })

    X_array = np.array(X, dtype=object)
    y_array = np.array(y)

    phase_data = {
        'speech_segments': [seg for pair in phase_data_list for seg in pair['speech_segments']],
        'metadata': [pair['metadata'] for pair in phase_data_list],
        'sampling_rate': sampling_rate,
        'n_channels': n_channels,
        'structure': 'unified_qa_pairs'
    }

    return X_array, y_array, phase_data

def test_hybrid_model_workflow():
    """
    Test the complete hybrid model workflow with simulated data
    """
    print("="*70)
    print("Testing Hybrid Model Workflow with Simulated Data")
    print("="*70)

    n_questions = 10
    n_channels = 8
    sampling_rate = 125

    print(f"\n1. Generating Simulated Q-A Pairs")
    print(f"   - Questions: {n_questions}")
    print(f"   - Channels: {n_channels}")
    print(f"   - Sampling rate: {sampling_rate} Hz")

    X, y, phase_data = generate_simulated_qa_pairs(
        n_questions=n_questions,
        n_channels=n_channels,
        question_duration=3.0,
        answer_duration=5.0,
        sampling_rate=sampling_rate
    )

    print(f"\n   ✓ Generated {len(X)} Q-A pairs")
    print(f"   - Truth pairs: {sum(y == 'truth')}")
    print(f"   - Lie pairs: {sum(y == 'lie')}")

    first_pair = X[0]
    print(f"\n   Data structure:")
    print(f"   - EEG shape: {first_pair['eeg_continuous'].shape}")
    print(f"   - Speech segments: {len(first_pair['speech_segments'])}")
    print(f"   - Transition at: {first_pair['metadata']['transition_time']:.2f}s")

    print(f"\n2. Extracting EEG Data for Training")
    eeg_data = []
    for qa_pair in X:
        eeg_data.append(qa_pair['eeg_continuous'])
    X_eeg = np.array(eeg_data)

    print(f"   ✓ Extracted EEG data: {X_eeg.shape}")

    print(f"\n3. Segmenting into Trials")
    window_size = 2.0
    overlap = 0.5
    samples_per_trial = int(window_size * sampling_rate)
    step_size = int(samples_per_trial * (1 - overlap))

    all_trials = []
    all_labels = []

    for i, eeg_pair in enumerate(X_eeg):
        n_samples = eeg_pair.shape[1]
        for start in range(0, n_samples - samples_per_trial + 1, step_size):
            end = start + samples_per_trial
            trial = eeg_pair[:, start:end]
            all_trials.append(trial)
            all_labels.append(y[i])

    X_trials = np.array(all_trials)
    y_trials = np.array(all_labels)

    print(f"   ✓ Segmented into {len(X_trials)} trials")
    print(f"   - Trial shape: {X_trials[0].shape}")
    print(f"   - Truth trials: {sum(y_trials == 'truth')}")
    print(f"   - Lie trials: {sum(y_trials == 'lie')}")

    print(f"\n4. Initializing Hybrid Model")
    n_timepoints = X_trials[0].shape[1]

    detector = HybridEEGLieDetector(
        n_channels=n_channels,
        n_timepoints=n_timepoints,
        device=torch.device('cpu'),
        rnn_type='LSTM',
        cnn_out_channels=32,
        rnn_hidden_size=64,
        rnn_num_layers=1,
        attention_heads=4
    )

    print(f"   ✓ Model initialized")
    print(f"   - Input shape: (batch, {n_channels}, {n_timepoints})")
    print(f"   - Device: {detector.device}")

    print(f"\n5. Splitting Data")
    X_train, X_val, y_train, y_val = train_test_split(
        X_trials, y_trials, test_size=0.2, random_state=42, stratify=y_trials
    )

    print(f"   ✓ Data split")
    print(f"   - Training: {len(X_train)} trials")
    print(f"   - Validation: {len(X_val)} trials")

    print(f"\n6. Testing Forward Pass")
    detector.model.eval()
    with torch.no_grad():
        test_input = torch.FloatTensor(X_train[:2]).to(detector.device)
        output = detector.model(test_input)
        print(f"   ✓ Forward pass successful")
        print(f"   - Input shape: {test_input.shape}")
        print(f"   - Output shape: {output.shape}")
        print(f"   - Output (logits): {output[0].cpu().numpy()}")

    print(f"\n7. Training Model (Short Training for Testing)")
    print(f"   - Epochs: 5 (reduced for testing)")
    print(f"   - Batch size: 8")
    print(f"   - Learning rate: 0.001")

    start_time = time.time()
    detector.train_model(
        X_train, y_train,
        X_val, y_val,
        epochs=5,
        batch_size=8,
        learning_rate=0.001
    )
    training_time = time.time() - start_time

    print(f"\n   ✓ Training completed in {training_time:.2f} seconds")

    print(f"\n8. Testing Prediction")
    test_sample = X_trials[0:1]
    result = detector.predict(test_sample)

    print(f"   ✓ Prediction successful")
    print(f"   - Prediction: {result['prediction']}")
    print(f"   - Confidence: {result['confidence'] * 100:.2f}%")
    print(f"   - Probabilities:")
    for label, prob in result['probabilities'].items():
        print(f"     {label}: {prob * 100:.2f}%")

    print(f"\n9. Testing Batch Prediction")
    test_batch = X_trials[:5]
    batch_result = detector.predict(test_batch)

    print(f"   ✓ Batch prediction successful")
    if 'predictions' in batch_result:
        print(f"   - Predictions: {batch_result['predictions']}")
        print(f"   - Number of samples: {len(batch_result['predictions'])}")

    print(f"\n10. Testing Feature Extraction")
    test_input = torch.FloatTensor(X_trials[:1]).to(detector.device)
    features = detector.model.extract_features(test_input)

    print(f"   ✓ Feature extraction successful")
    print(f"   - Extracted features:")
    for key, value in features.items():
        if isinstance(value, torch.Tensor):
            print(f"     {key}: {value.shape}")
        elif isinstance(value, tuple):
            print(f"     {key}: {[v.shape if isinstance(v, torch.Tensor) else type(v).__name__ for v in value]}")

    print(f"\n" + "="*70)
    print("WORKFLOW TEST SUMMARY")
    print("="*70)
    print(f"✓ Data generation: SUCCESS")
    print(f"✓ Data segmentation: SUCCESS")
    print(f"✓ Model initialization: SUCCESS")
    print(f"✓ Forward pass: SUCCESS")
    print(f"✓ Training: SUCCESS")
    print(f"✓ Prediction: SUCCESS")
    print(f"✓ Feature extraction: SUCCESS")
    print(f"\nAll components working correctly!")
    print("="*70)

if __name__ == "__main__":

    test_hybrid_model_workflow()

    print("\n" + "="*70)
    print("HYBRID MODEL TEST COMPLETE")
    print("="*70)
    print("\n✓ Hybrid model workflow tested successfully with simulated data!")
    print("You can now use the hybrid model with real hardware data.")
