"""
Lie Detector with Hybrid CNN-RNN-Transformer Model
Integrates HybridEEGModel for improved lie detection using spatial, temporal, and attention features
"""

import numpy as np
import time
import threading
from typing import Optional, Tuple, List, Dict
from utils.board_initializer import NeuropawnKnightBoard
from utils.speech_eeg_synchronizer import SpeechEEGSynchronizer
from models.hybrid_eeg_model import HybridEEGLieDetector
from sklearn.model_selection import train_test_split
import torch

class HybridLieDetectorSystem:
    """
    Complete lie detection system using Hybrid CNN-RNN-Transformer model
    Combines board initialization, data collection, and hybrid model
    """

    def __init__(
        self,
        serial_port: Optional[str] = None,
        enable_speech: bool = True,
        n_channels: int = 8,
        device: Optional[str] = None,
        rnn_type: str = 'LSTM',
        **model_kwargs
    ):
        """
        Initialize Hybrid lie detector system

        Args:
            serial_port: Serial port for Neuropawn Knight board
            enable_speech: Enable speech-EEG synchronization
            n_channels: Number of EEG channels
            device: Device for training ('cuda' or 'cpu')
            rnn_type: 'LSTM' or 'GRU' for RNN component
            **model_kwargs: Additional arguments for HybridEEGModel
        """
        self.board = NeuropawnKnightBoard(serial_port=serial_port)
        self.enable_speech = enable_speech
        self.synchronizer = None
        self.n_channels = n_channels

        torch_device = torch.device(device) if device else None
        self.hybrid_detector = HybridEEGLieDetector(
            n_channels=n_channels,
            n_timepoints=250,
            device=torch_device,
            rnn_type=rnn_type,
            **model_kwargs
        )

        self.sampling_rate = None
        self.eeg_channels = None
        self.is_trained = False

    def connect(self) -> bool:
        """
        Connect to the board and initialize speech synchronizer if enabled

        Returns:
            bool: True if connection successful
        """
        success = self.board.connect()
        if success:
            self.sampling_rate = self.board.get_sampling_rate()
            self.eeg_channels = self.board.get_eeg_channels()
            self.n_channels = len(self.eeg_channels)

            self.hybrid_detector.n_channels = self.n_channels

            from hybrid_eeg_model import HybridEEGModel
            model_kwargs = {
                'n_timepoints': 250,
                'n_classes': 2,
                'rnn_type': 'LSTM'
            }
            self.hybrid_detector.model = HybridEEGModel(
                n_channels=self.n_channels,
                **model_kwargs
            ).to(self.hybrid_detector.device)

            print(f"Sampling rate: {self.sampling_rate} Hz")
            print(f"EEG channels: {self.eeg_channels}")
            print(f"Number of channels: {self.n_channels}")
            print(f"Model: Hybrid CNN-RNN-Transformer")
            print(f"  - CNN: Spatial feature extraction")
            print(f"  - RNN: {model_kwargs.get('rnn_type', 'LSTM')} for temporal sequences")
            print(f"  - Attention: Channel + Temporal attention")

            if self.enable_speech:
                try:
                    self.synchronizer = SpeechEEGSynchronizer(self.board)
                    print("✓ Speech-EEG synchronization enabled")
                except Exception as e:
                    print(f"Warning: Could not initialize speech synchronizer: {e}")
                    print("Continuing without speech detection...")
                    self.enable_speech = False
        return success

    def collect_training_data(
        self,
        n_questions: int = 5,
        answer_duration: Optional[float] = None,
        truth_label: str = "truth",
        lie_label: str = "lie",
        use_speech: Optional[bool] = None
    ) -> Tuple[np.ndarray, np.ndarray, Optional[Dict]]:
        """
        Collect training data with question/answer phases stored together

        Workflow:
        - Question phase starts automatically
        - Press SPACEBAR to transition to answer phase
        - Press ANY KEY to end answer phase (or auto-ends after answer_duration)
        - Stores unified Q-A pairs: [Question EEG | Answer EEG]

        Args:
            n_questions: Number of questions to collect (each has question + answer phase)
            answer_duration: Maximum duration for answer phase (seconds, None = no limit, ends on key press)
            truth_label: Label for truth condition
            lie_label: Label for lie condition
            use_speech: Override enable_speech setting

        Returns:
            Tuple of (data, labels, phase_data) where:
                - data: numpy array of EEG data (n_questions, n_channels, total_samples)
                - labels: numpy array shape (n_questions,) - truth/lie labels
                - phase_data: Dict with metadata about phase transitions and speech segments
        """
        if not self.board.board:
            raise RuntimeError("Board not connected. Call connect() first.")

        use_speech_detection = use_speech if use_speech is not None else self.enable_speech

        all_qa_pairs = []
        all_labels = []

        print(f"\n{'='*70}")
        print("TWO-PHASE TRAINING DATA COLLECTION")
        print(f"{'='*70}")
        print(f"\nYou will answer {n_questions} questions.")
        print("\nFor each question:")
        print("  1. QUESTION PHASE: The question/context will be presented")
        print("     - Recording starts automatically")
        print("     - Focus on brain activity when context is mentioned")
        print("  2. Press SPACEBAR when ready to answer")
        print("  3. ANSWER PHASE: Respond truthfully or with a lie")
        print("     - Recording continues for answer phase")
        if answer_duration:
            print(f"     - Maximum duration: {answer_duration} seconds (or press any key to end)")
        else:
            print("     - Press any key when finished answering")
        print(f"\n{'='*70}\n")

        for q_idx in range(n_questions):
            print(f"\n{'='*70}")
            print(f"QUESTION {q_idx + 1} / {n_questions}")
            print(f"{'='*70}")

            if q_idx % 2 == 0:
                expected_label = truth_label
                instruction = "TRUTH"
            else:
                expected_label = lie_label
                instruction = "LIE"

            print(f"\nThis question should be answered with: {instruction.upper()}")
            print("\nInstructions:")
            print("  1. The question/context will be presented (QUESTION PHASE)")
            print("  2. Press SPACEBAR when ready to answer")
            print(f"  3. Answer with {instruction.lower()} (ANSWER PHASE)")
            print("\nPress Enter when ready to start this question...")
            try:
                input()
            except EOFError:
                pass

            if use_speech_detection and self.synchronizer:
                print(f"\n{'─'*70}")
                print("QUESTION PHASE - Recording started")
                print("Context/question is being presented...")
                print("Press SPACEBAR when ready to answer")
                print(f"{'─'*70}\n")

                self.synchronizer.start_phase_recording(initial_phase='question')

                print("Waiting for SPACEBAR press to start answer phase...")
                transition_occurred = self.synchronizer.wait_for_phase_transition(timeout=None)

                if transition_occurred:
                    print(f"\n{'─'*70}")
                    print(f"ANSWER PHASE - Recording answer")
                    print(f"Please answer with {instruction.lower()}")
                    if answer_duration:
                        print(f"Press any key to end (or will auto-end after {answer_duration}s)")
                    else:
                        print("Press any key when finished answering")
                    print(f"{'─'*70}\n")

                    if answer_duration:
                        answer_ended = self.synchronizer.wait_for_answer_end(timeout=answer_duration)
                        if not answer_ended:
                            print(f"\n{'─'*70}")
                            print(f"Answer phase duration ({answer_duration}s) reached. Ending recording...")
                            print(f"{'─'*70}\n")
                    else:
                        print("Waiting for key press to end answer phase...")
                        self.synchronizer.wait_for_answer_end(timeout=None)

                    self.synchronizer.stop_synchronized_recording()

                    phase_data_dict = self.synchronizer.get_phase_data()
                    question_segments = phase_data_dict['question']
                    answer_segments = phase_data_dict['answer']

                    all_eeg = self.board.get_data()[self.eeg_channels, :]
                    transition_sample = int(
                        self.synchronizer.phase_transition_time * self.sampling_rate
                    )
                    question_eeg = all_eeg[:, :transition_sample]
                    answer_eeg = all_eeg[:, transition_sample:]

                    print(f"  Question phase EEG: {question_eeg.shape}")
                    print(f"  Answer phase EEG: {answer_eeg.shape}")

                    if question_eeg.shape[1] > 0 and answer_eeg.shape[1] > 0:
                        qa_combined_eeg = np.hstack([question_eeg, answer_eeg])

                        all_segments_for_pair = []
                        for entry in question_segments + answer_segments:
                            segment_entry = {
                                **entry,
                                'label': expected_label,
                                'question_num': q_idx + 1
                            }
                            all_segments_for_pair.append(segment_entry)

                        qa_pair = {
                            'eeg_continuous': qa_combined_eeg,
                            'speech_segments': all_segments_for_pair,
                            'metadata': {
                                'question_num': q_idx + 1,
                                'label': expected_label,
                                'transition_time': self.synchronizer.phase_transition_time,
                                'transition_sample': question_eeg.shape[1],
                                'question_samples': question_eeg.shape[1],
                                'answer_samples': answer_eeg.shape[1],
                                'total_samples': qa_combined_eeg.shape[1],
                                'question_duration': question_eeg.shape[1] / self.sampling_rate,
                                'answer_duration': answer_eeg.shape[1] / self.sampling_rate,
                                'n_question_segments': len(question_segments),
                                'n_answer_segments': len(answer_segments)
                            }
                        }

                        all_qa_pairs.append(qa_pair)
                        all_labels.append(expected_label)

                        print(f"  Combined Q-A EEG: {qa_combined_eeg.shape} (Q: {question_eeg.shape[1]} samples, A: {answer_eeg.shape[1]} samples)")
                        print(f"  Speech segments: {len(all_segments_for_pair)} (Q: {len(question_segments)}, A: {len(answer_segments)})")
                else:
                    print("No phase transition detected. Stopping recording...")
                    self.synchronizer.stop_synchronized_recording()
                    continue

            else:

                print(f"\n{'─'*70}")
                print("QUESTION PHASE - Recording started")
                print("Context/question is being presented...")
                print("Press SPACEBAR when ready to answer")
                print(f"{'─'*70}\n")

                self.board.start_streaming()
                start_time = time.time()
                transition_flag = threading.Event()

                from pynput import keyboard

                def on_press(key):
                    try:
                        if key == keyboard.Key.space:
                            transition_flag.set()
                            return False
                    except AttributeError:
                        pass

                listener = keyboard.Listener(on_press=on_press)
                listener.start()

                print("Waiting for SPACEBAR press...")
                transition_flag.wait()
                listener.stop()

                transition_time = time.time() - start_time
                print(f"\n{'─'*70}")
                print(f"ANSWER PHASE - Recording answer (transitioned at {transition_time:.2f}s)")
                print(f"Please answer with {instruction.lower()}")
                if answer_duration:
                    print(f"Press any key to end (or will auto-end after {answer_duration}s)")
                else:
                    print("Press any key when finished answering")
                print(f"{'─'*70}\n")

                answer_end_flag = threading.Event()

                def on_press_answer(key):
                    try:
                        answer_end_flag.set()
                        return False
                    except AttributeError:
                        pass

                listener_answer = keyboard.Listener(on_press=on_press_answer)
                listener_answer.start()

                if answer_duration:
                    answer_ended = answer_end_flag.wait(timeout=answer_duration)
                    if not answer_ended:
                        print(f"\n{'─'*70}")
                        print(f"Answer phase duration ({answer_duration}s) reached. Ending recording...")
                        print(f"{'─'*70}\n")
                else:
                    print("Waiting for key press to end answer phase...")
                    answer_end_flag.wait()

                listener_answer.stop()

                all_data = self.board.get_data()
                self.board.stop_streaming()

                transition_sample = int(transition_time * self.sampling_rate)
                question_eeg = all_data[self.eeg_channels, :transition_sample]
                answer_eeg = all_data[self.eeg_channels, transition_sample:]

                print(f"  Question phase EEG: {question_eeg.shape}")
                print(f"  Answer phase EEG: {answer_eeg.shape}")

                if question_eeg.shape[1] > 0 and answer_eeg.shape[1] > 0:
                    qa_combined_eeg = np.hstack([question_eeg, answer_eeg])

                    qa_pair = {
                        'eeg_continuous': qa_combined_eeg,
                        'speech_segments': [],
                        'metadata': {
                            'question_num': q_idx + 1,
                            'label': expected_label,
                            'transition_time': transition_time,
                            'transition_sample': question_eeg.shape[1],
                            'question_samples': question_eeg.shape[1],
                            'answer_samples': answer_eeg.shape[1],
                            'total_samples': qa_combined_eeg.shape[1],
                            'question_duration': question_eeg.shape[1] / self.sampling_rate,
                            'answer_duration': answer_eeg.shape[1] / self.sampling_rate,
                            'n_question_segments': 0,
                            'n_answer_segments': 0
                        }
                    }

                    all_qa_pairs.append(qa_pair)
                    all_labels.append(expected_label)

                    print(f"  Combined Q-A EEG: {qa_combined_eeg.shape} (Q: {question_eeg.shape[1]} samples, A: {answer_eeg.shape[1]} samples)")
                    print(f"  Speech segments: 0 (speech detection disabled)")

            print(f"  ✓ Question {q_idx + 1} completed")
            print(f"    - Label: {expected_label}")

        if all_qa_pairs:
            all_data = np.array(all_qa_pairs, dtype=object)
            all_labels = np.array(all_labels)
        else:
            all_data = np.array([], dtype=object)
            all_labels = np.array([])

        print(f"\n{'='*70}")
        print("DATA COLLECTION SUMMARY")
        print(f"{'='*70}")
        print(f"Total question-answer pairs: {len(all_qa_pairs)}")
        print(f"  - Truth pairs: {sum(all_labels == truth_label)}")
        print(f"  - Lie pairs: {sum(all_labels == lie_label)}")

        if all_qa_pairs:
            avg_q_duration = np.mean([pair['metadata']['question_duration'] for pair in all_qa_pairs])
            avg_a_duration = np.mean([pair['metadata']['answer_duration'] for pair in all_qa_pairs])
            total_speech_segs = sum(len(pair['speech_segments']) for pair in all_qa_pairs)

            print(f"\nAverage durations:")
            print(f"  - Question phase: {avg_q_duration:.2f} seconds")
            print(f"  - Answer phase: {avg_a_duration:.2f} seconds")
            print(f"  - Total per sample: {avg_q_duration + avg_a_duration:.2f} seconds")

            if use_speech_detection:
                print(f"\nSpeech segments:")
                print(f"  - Total: {total_speech_segs}")
                question_segs = sum(
                    sum(1 for seg in pair['speech_segments'] if seg.get('phase') == 'question')
                    for pair in all_qa_pairs
                )
                answer_segs = sum(
                    sum(1 for seg in pair['speech_segments'] if seg.get('phase') == 'answer')
                    for pair in all_qa_pairs
                )
                print(f"    - Question phase segments: {question_segs}")
                print(f"    - Answer phase segments: {answer_segs}")

        if len(all_qa_pairs) > 0:
            eeg_data = []
            for qa_pair in all_qa_pairs:
                eeg_data.append(qa_pair['eeg_continuous'])
            X_eeg = np.array(eeg_data)
        else:
            X_eeg = np.array([])

        phase_data = {
            'sampling_rate': self.sampling_rate,
            'n_channels': self.n_channels,
            'structure': 'unified_qa_pairs'
        }

        return X_eeg, all_labels, phase_data

        if len(X) > 0 and isinstance(X[0], dict):

            eeg_data = []
            for qa_pair in X:
                eeg_data.append(qa_pair['eeg_continuous'])
            X_eeg = np.array(eeg_data)
        else:
            X_eeg = X

        return X_eeg, y, phase_data

    def _segment_eeg_data(self, eeg_data: np.ndarray, window_size: float = 2.0, overlap: float = 0.5) -> List[np.ndarray]:
        """
        Segment EEG data into trials

        Args:
            eeg_data: EEG data shape (n_channels, n_samples)
            window_size: Window size in seconds
            overlap: Overlap ratio

        Returns:
            List of trial arrays
        """
        n_channels, n_samples = eeg_data.shape
        samples_per_trial = int(window_size * self.sampling_rate)
        step_size = int(samples_per_trial * (1 - overlap))

        trials = []
        for start in range(0, n_samples - samples_per_trial + 1, step_size):
            end = start + samples_per_trial
            trial = eeg_data[:, start:end]
            trials.append(trial)

        return trials

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        window_size: float = 2.0,
        overlap: float = 0.5
    ):
        """
        Train the Hybrid model on question-answer phase data

        Args:
            X: Training data - array of Q-A pairs or EEG trials
            y: Labels shape (n_samples,) - truth/lie labels
            test_size: Proportion of data for validation
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            window_size: Window size in seconds for segmenting Q-A pairs
            overlap: Overlap ratio for windows (0.0 to 1.0)
        """
        print("\n" + "="*50)
        print("Training Hybrid CNN-RNN-Transformer Model")
        print("="*50)
        print("Architecture:")
        print("  - CNN: Spatial feature extraction from EEG channels")
        print("  - RNN: Temporal sequence modeling (LSTM/GRU)")
        print("  - Attention: Channel + Temporal attention mechanisms")
        print("  - Fusion: Combines all features for classification")

        all_trials = []
        all_trial_labels = []

        if X.dtype == object and len(X) > 0:
            first_item = X[0]
            if isinstance(first_item, dict) and 'eeg_continuous' in first_item:

                for i, qa_pair in enumerate(X):
                    eeg_continuous = qa_pair['eeg_continuous']

                    trials = self._segment_eeg_data(eeg_continuous, window_size=window_size, overlap=overlap)
                    all_trials.extend(trials)
                    all_trial_labels.extend([y[i]] * len(trials))
            else:

                for i, qa_pair in enumerate(X):
                    trials = self._segment_eeg_data(qa_pair, window_size=window_size, overlap=overlap)
                    all_trials.extend(trials)
                    all_trial_labels.extend([y[i]] * len(trials))
        else:

            all_trials = list(X)
            all_trial_labels = list(y)

        X_segmented = np.array(all_trials)
        y_segmented = np.array(all_trial_labels)

        print(f"\nSegmentation:")
        print(f"  - Original samples: {len(X)}")
        print(f"  - Segmented trials: {len(X_segmented)}")
        print(f"  - Window size: {window_size}s")
        print(f"  - Overlap: {overlap * 100:.0f}%")
        print(f"  - Trial shape: {X_segmented[0].shape}")

        if len(X_segmented) > 0:
            actual_timepoints = X_segmented[0].shape[1]
            if actual_timepoints != self.hybrid_detector.n_timepoints:
                print(f"  - Updating model timepoints: {self.hybrid_detector.n_timepoints} -> {actual_timepoints}")

                from hybrid_eeg_model import HybridEEGModel
                model_kwargs = {
                    'n_classes': 2,
                    'rnn_type': 'LSTM'
                }
                self.hybrid_detector.model = HybridEEGModel(
                    n_channels=self.n_channels,
                    n_timepoints=actual_timepoints,
                    **model_kwargs
                ).to(self.hybrid_detector.device)
                self.hybrid_detector.n_timepoints = actual_timepoints

        X_train, X_val, y_train, y_val = train_test_split(
            X_segmented, y_segmented, test_size=test_size, random_state=42, stratify=y_segmented
        )

        print(f"\nTraining samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Device: {self.hybrid_detector.device}")

        self.hybrid_detector.train_model(
            X_train, y_train,
            X_val, y_val,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )

        self.is_trained = True
        print("✓ Training complete!")
        print("\nModel components:")
        print("  - CNN extracted spatial features from channels")
        print("  - RNN modeled temporal sequences")
        print("  - Attention weighted important channels and timepoints")
        print("  - All features fused for final classification")

    def predict(self, data: Optional[np.ndarray] = None, duration: float = 5.0) -> dict:
        """
        Predict whether the current signal indicates truth or lie

        Args:
            data: Optional pre-collected data. If None, collects new data
            duration: Duration to collect data if data is None

        Returns:
            dict: Prediction results
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        if data is None:
            print(f"\nCollecting data for {duration} seconds...")
            self.board.start_streaming()
            time.sleep(duration)
            raw_data = self.board.get_data()
            self.board.stop_streaming()

            eeg_data = raw_data[self.eeg_channels, :]
        else:
            eeg_data = data

        trials = self._segment_eeg_data(eeg_data, window_size=2.0)

        if len(trials) == 0:
            raise ValueError("Not enough data collected. Need at least 2 seconds.")

        trials_array = np.array(trials)

        result = self.hybrid_detector.predict(trials_array)

        if 'predictions' in result:
            predictions = result['predictions']
            probabilities = np.array(result['probabilities'])

            unique, counts = np.unique(predictions, return_counts=True)
            majority_pred = unique[np.argmax(counts)]
            avg_probs = np.mean(probabilities, axis=0)
            confidence = np.max(avg_probs)

            result = {
                'prediction': majority_pred,
                'confidence': float(confidence),
                'probabilities': {
                    'truth': float(avg_probs[0]),
                    'lie': float(avg_probs[1])
                },
                'n_trials': len(trials)
            }

        return result

    def save_model(self, filepath: str):
        """Save the trained model"""
        if not self.is_trained:
            raise RuntimeError("No model to save. Train the model first.")

        self.hybrid_detector.save_model(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load a trained model"""
        self.hybrid_detector.load_model(filepath)
        self.is_trained = True
        print(f"Model loaded from {filepath}")

    def disconnect(self):
        """Disconnect from the board and cleanup resources"""
        if self.synchronizer:
            self.synchronizer.cleanup()
        self.board.disconnect()

def main():
    """
    Example usage of Hybrid lie detector
    """
    import sys

    serial_port = sys.argv[1] if len(sys.argv) > 1 else None

    detector = HybridLieDetectorSystem(
        serial_port=serial_port,
        enable_speech=True,
        rnn_type='LSTM'
    )

    try:

        if not detector.connect():
            print("Failed to connect to board. Exiting.")
            return

        print("\n" + "="*50)
        print("TRAINING PHASE - Hybrid CNN-RNN-Transformer Model")
        print("="*50)
        X, y, phase_data = detector.collect_training_data(
            n_questions=5,
            answer_duration=20.0,
            use_speech=True
        )

        detector.train(X, y, epochs=30, batch_size=16, learning_rate=0.001)

        detector.save_model('hybrid_lie_detector_model.pth')

        print("\n" + "="*50)
        print("TESTING PHASE")
        print("="*50)
        print("Now we'll test the detector. Think of something true or false.")
        try:
            input("Press Enter when ready to start test...")
        except EOFError:
            pass

        result = detector.predict(duration=5.0)

        print("\n" + "="*50)
        print("PREDICTION RESULTS")
        print("="*50)
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence'] * 100:.2f}%")
        print(f"Probabilities:")
        for label, prob in result['probabilities'].items():
            print(f"  {label}: {prob * 100:.2f}%")
        print(f"Number of trials analyzed: {result.get('n_trials', 1)}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        detector.disconnect()

if __name__ == "__main__":
    main()