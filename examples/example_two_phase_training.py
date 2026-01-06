"""
Example: Two-Phase Training Data Collection
Demonstrates the question/answer phase workflow for lie detection training
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from detectors.lie_detector_hybrid import HybridLieDetectorSystem

def main():
    """
    Example of collecting training data with question/answer phases
    """

    serial_port = sys.argv[1] if len(sys.argv) > 1 else None

    detector = HybridLieDetectorSystem(serial_port=serial_port, enable_speech=True)

    try:

        print("\nConnecting to board...")
        if not detector.connect():
            print("Failed to connect. Exiting.")
            return

        print("\n" + "=" * 70)
        print("TWO-PHASE TRAINING DATA COLLECTION")
        print("=" * 70)
        print("\nWorkflow:")
        print("  1. QUESTION PHASE: Context/question is presented")
        print("     - Recording starts automatically")
        print("     - Focus on brain activity when context is mentioned")
        print("  2. Press SPACEBAR to transition to answer phase")
        print("  3. ANSWER PHASE: Respond truthfully or with a lie")
        print("     - Recording continues")
        print("     - Press any key when finished answering")
        print("     - Question + Answer phases stored together as one training sample")
        print("     - V-TAM learns relationship between phases")
        print("=" * 70)

        X, y, phase_data = detector.collect_training_data(
            n_questions=5,
            answer_duration=None,
            use_speech=True
        )

        print("\n" + "=" * 70)
        print("COLLECTION RESULTS")
        print("=" * 70)
        print(f"Question-Answer pairs collected: {len(X)}")
        print(f"  - Truth pairs: {sum(y == 'truth')}")
        print(f"  - Lie pairs: {sum(y == 'lie')}")
        print("\nEach pair contains:")
        print("  1. eeg_continuous: [Question Phase EEG | Answer Phase EEG]")
        print("     - Concatenated along time axis")
        print("  2. speech_segments: Speech segments with mapped EEG")
        print("     - Question phase segments + Answer phase segments")
        print("     - Each segment: {speech_audio, eeg_segment, phase, timestamps}")
        print("  3. metadata: Transition info, durations, labels")

        if len(X) > 0:

            first_pair = X[0]
            if isinstance(first_pair, dict):
                print(f"\nExample structure:")
                print(f"  - EEG shape: {first_pair['eeg_continuous'].shape}")
                print(f"  - Speech segments: {len(first_pair['speech_segments'])}")
                print(f"  - Transition at: {first_pair['metadata']['transition_time']:.2f}s")

                question_segs = sum(1 for seg in first_pair['speech_segments'] if seg.get('phase') == 'question')
                answer_segs = sum(1 for seg in first_pair['speech_segments'] if seg.get('phase') == 'answer')
                print(f"    - Question phase: {question_segs} segments")
                print(f"    - Answer phase: {answer_segs} segments")

            import numpy as np
            save_data = {
                'qa_pairs': X,
                'labels': y,
                'sampling_rate': phase_data.get('sampling_rate', detector.sampling_rate),
                'n_channels': phase_data.get('n_channels', detector.n_channels),
                'structure': 'unified_qa_pairs'
            }
            np.savez_compressed('qa_training_data.npz', **save_data, allow_pickle=True)
            print(f"\n✓ Unified Q-A pair data saved to 'qa_training_data.npz'")

        print("\n" + "=" * 70)
        print("TRAINING HYBRID MODEL")
        print("=" * 70)
        print("Training on Unified Question-Answer pairs...")
        print("  - Each sample contains:")
        print("    * Continuous EEG: [Question Phase | Answer Phase]")
        print("    * Speech segments: Speech audio + mapped EEG")
        print("  - Model will learn relationship between phases")
        print("  - Hybrid CNN-RNN-Transformer architecture")
        print("  - Training on continuous EEG data")

        detector.train(X, y, epochs=30, batch_size=16, learning_rate=0.001)

        detector.save_model('hybrid_lie_detector_model.pth')
        print("✓ Model saved to 'hybrid_lie_detector_model.pth'")

        print("\n" + "=" * 70)
        print("TESTING PHASE")
        print("=" * 70)
        print("Now we'll test the detector with a new question.")
        print("The system will collect data in question/answer phases.")
        try:
            input("Press Enter when ready to start test...")
        except EOFError:
            pass

        result = detector.predict(duration=5.0)

        print("\n" + "=" * 70)
        print("PREDICTION RESULTS")
        print("=" * 70)
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence'] * 100:.2f}%")
        print(f"Probabilities:")
        for label, prob in result['probabilities'].items():
            print(f"  {label}: {prob * 100:.2f}%")
        print(f"Number of trials analyzed: {result['n_trials']}")

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
