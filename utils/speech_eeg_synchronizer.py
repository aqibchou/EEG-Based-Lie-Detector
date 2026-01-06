"""
Speech-EEG Synchronization Module
Synchronizes speech detection (using Silero VAD) with EEG data collection
Maps speech segments to corresponding brain signals for analysis
"""

import numpy as np
import time
import threading
import queue
from typing import Optional, List, Dict, Tuple
from datetime import datetime
from collections import deque
import pyaudio
import torch
from .board_initializer import NeuropawnKnightBoard
from pynput import keyboard

class SpeechEEGSynchronizer:
    """
    Synchronizes speech detection with EEG data collection
    Maps speech segments to corresponding brain signals
    """

    def __init__(self,
                 board: NeuropawnKnightBoard,
                 sample_rate: int = 16000,
                 chunk_size: int = 512,
                 vad_threshold: float = 0.5,
                 input_device_index: Optional[int] = None):
        """
        Initialize the speech-EEG synchronizer

        Args:
            board: NeuropawnKnightBoard instance
            sample_rate: Audio sampling rate (8000 or 16000 Hz for Silero VAD)
            chunk_size: Audio chunk size for processing
            vad_threshold: VAD threshold for speech detection (0.0 to 1.0)
            input_device_index: Index of audio input device (None = use default)
        """
        self.board = board
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.vad_threshold = vad_threshold
        self.input_device_index = input_device_index

        print("Loading Silero VAD model from local repository...")
        try:

            torch.set_num_threads(1)

            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            silero_vad_path = os.path.join(current_dir, 'silero-vad')

            if not os.path.exists(silero_vad_path):

                silero_vad_path = os.path.join(os.path.dirname(current_dir), 'silero-vad')

            if not os.path.exists(silero_vad_path):
                raise FileNotFoundError(
                    f"Silero VAD repository not found at {silero_vad_path}. "
                    "Please clone it: git clone https://github.com/snakers4/silero-vad.git"
                )

            self.vad_model, self.vad_utils = torch.hub.load(
                repo_or_dir=silero_vad_path,
                model='silero_vad',
                source='local',
                trust_repo=True
            )

            self.get_speech_timestamps = self.vad_utils[0]
            self.VADIterator = self.vad_utils[3]
            print("✓ VAD model loaded from local repository")
        except Exception as e:
            print(f"Error loading VAD model: {e}")
            print("\nTroubleshooting:")
            print("  - Make sure silero-vad repository is cloned in the project directory")
            print("  - Run: git clone https://github.com/snakers4/silero-vad.git")
            import traceback
            traceback.print_exc()
            raise

        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.audio = pyaudio.PyAudio()

        self.eeg_sampling_rate = None
        self.start_time = None
        self.is_recording = False

        self.audio_buffer = deque(maxlen=int(sample_rate * 60))
        self.eeg_data_queue = queue.Queue()
        self.speech_segments = []
        self.eeg_segments = []
        self.synchronized_data = []

        self.audio_thread = None
        self.eeg_thread = None
        self.processing_thread = None

        self.current_phase = None
        self.phase_transition_time = None
        self.phase_transition_flag = threading.Event()
        self.answer_end_flag = threading.Event()
        self.keyboard_listener = None

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream"""
        if self.is_recording:
            audio_array = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
            timestamp = time.time() - self.start_time
            self.audio_buffer.append((timestamp, audio_array))

            if len(self.audio_buffer) <= 3:
                print(f"  Audio chunk received: {len(audio_array)} samples, timestamp: {timestamp:.2f}s")
        return (None, pyaudio.paContinue)

    def _collect_eeg_data(self):
        """Thread function to continuously collect EEG data"""
        while self.is_recording:
            try:

                if self.board.board:

                    current_time = time.time() - self.start_time
                    num_samples = int(self.eeg_sampling_rate * 0.1)
                    data = self.board.get_data(num_samples=num_samples)

                    if data.shape[1] > 0:
                        self.eeg_data_queue.put((current_time, data.copy()))

                time.sleep(0.05)
            except Exception as e:
                print(f"Error collecting EEG data: {e}")
                break

    def _process_speech_segments(self):
        """Process audio buffer to detect speech segments using VAD iterator"""

        print("  Initializing VAD iterator...")
        try:
            vad_iterator = self.VADIterator(self.vad_model, sampling_rate=self.sample_rate)
            print("  ✓ VAD iterator initialized")
        except Exception as e:
            print(f"  ✗ Error initializing VAD iterator: {e}")
            import traceback
            traceback.print_exc()
            return

        last_processed_time = 0.0
        chunks_processed = 0

        while self.is_recording:
            try:
                if len(self.audio_buffer) < 1:
                    time.sleep(0.05)
                    continue

                buffer_list = list(self.audio_buffer)
                current_time = time.time() - self.start_time

                if chunks_processed == 0 and len(buffer_list) > 0:
                    print(f"  Processing audio buffer: {len(buffer_list)} chunks available")

                processed_any = False
                for ts, audio_chunk in buffer_list:
                    if ts <= last_processed_time:
                        continue

                    if len(audio_chunk) != self.chunk_size:
                        if chunks_processed == 0:
                            print(f"  Warning: chunk size mismatch: {len(audio_chunk)} != {self.chunk_size}")
                        continue

                    processed_any = True

                    try:

                        audio_tensor = torch.tensor(audio_chunk, dtype=torch.float32)

                        if audio_tensor.dim() == 1:
                            audio_tensor = audio_tensor.unsqueeze(0)

                        vad_result = vad_iterator(audio_tensor, return_seconds=False)
                        chunks_processed += 1

                        if chunks_processed <= 10:
                            print(f"  VAD result: {vad_result}")

                        if vad_result is not None:
                            if 'start' in vad_result:

                                chunk_duration = self.chunk_size / self.sample_rate
                                chunk_start_time = ts
                                chunk_end_time = ts + chunk_duration

                                self.speech_segments.append({
                                    'start_time': chunk_start_time,
                                    'end_time': chunk_end_time,
                                    'audio_data': audio_chunk.copy(),
                                    'speech_probability': 0.8
                                })
                                print(f"  ✓ Speech START detected at {chunk_start_time:.2f}s")

                            elif 'end' in vad_result:

                                if self.speech_segments:
                                    chunk_end_time = ts + (self.chunk_size / self.sample_rate)
                                    self.speech_segments[-1]['end_time'] = chunk_end_time
                                    self.speech_segments[-1]['audio_data'] = np.concatenate([
                                        self.speech_segments[-1]['audio_data'],
                                        audio_chunk
                                    ])
                                    print(f"  ✓ Speech END detected at {chunk_end_time:.2f}s")

                                elif len(self.speech_segments) > 0:
                                    chunk_duration = self.chunk_size / self.sample_rate
                                    self.speech_segments[-1]['end_time'] = ts + chunk_duration
                                    self.speech_segments[-1]['audio_data'] = np.concatenate([
                                        self.speech_segments[-1]['audio_data'],
                                        audio_chunk
                                    ])
                    except Exception as e:

                        if chunks_processed <= 10:
                            print(f"  Error processing chunk: {e}")
                            import traceback
                            traceback.print_exc()
                        pass

                    last_processed_time = ts

                if not processed_any and chunks_processed == 0 and len(buffer_list) > 5:
                    print(f"  Warning: No chunks processed. Last processed time: {last_processed_time:.2f}, buffer times: {[b[0] for b in buffer_list[:3]]}")

                time.sleep(0.05)

            except Exception as e:
                print(f"Error processing speech: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)

    def start_synchronized_recording(self, duration: Optional[float] = None):
        """
        Start synchronized recording of speech and EEG

        Args:
            duration: Recording duration in seconds. If None, records until stop() is called
        """
        if not self.board.board:
            raise RuntimeError("Board not connected. Call board.connect() first.")

        if self.is_recording:
            print("Already recording. Call stop() first.")
            return

        self.eeg_sampling_rate = self.board.get_sampling_rate()

        self.start_time = time.time()
        self.is_recording = True
        self.speech_segments = []
        self.eeg_segments = []
        self.synchronized_data = []
        self.audio_buffer.clear()

        print("Starting EEG stream...")
        self.board.start_streaming()

        print("Starting audio stream...")
        try:

            if self.input_device_index is not None:
                device_info = self.audio.get_device_info_by_index(self.input_device_index)
                print(f"  Using specified audio device: {device_info['name']} (index {self.input_device_index})")
                input_device_index = self.input_device_index
            else:
                device_info = self.audio.get_default_input_device_info()
                print(f"  Using default audio device: {device_info['name']}")
                input_device_index = None

            self.audio_stream = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=input_device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            self.audio_stream.start_stream()
            print("  ✓ Audio stream started")
        except Exception as e:
            print(f"  ✗ Error starting audio stream: {e}")
            raise

        self.eeg_thread = threading.Thread(target=self._collect_eeg_data, daemon=True)
        self.processing_thread = threading.Thread(target=self._process_speech_segments, daemon=True)

        self.eeg_thread.start()
        self.processing_thread.start()

        print("✓ Synchronized recording started")
        print(f"  - Audio: {self.sample_rate} Hz")
        print(f"  - EEG: {self.eeg_sampling_rate} Hz")
        print(f"  - VAD threshold: {self.vad_threshold}")
        print("  - Listening for speech...")

        if duration:
            time.sleep(duration)
            self.stop_synchronized_recording()

    def start_phase_recording(self, initial_phase: str = 'question'):
        """
        Start recording with phase tracking (question/answer phases)

        Args:
            initial_phase: Initial phase ('question' or 'answer')
        """
        if not self.board.board:
            raise RuntimeError("Board not connected. Call board.connect() first.")

        if self.is_recording:
            print("Already recording. Call stop() first.")
            return

        self.current_phase = initial_phase
        self.phase_transition_time = None
        self.phase_transition_flag.clear()

        self._start_keyboard_listener()

        self.start_synchronized_recording(duration=None)

        print(f"✓ Phase recording started in '{initial_phase}' phase")
        print("  - Press SPACEBAR to transition to answer phase")

    def _start_keyboard_listener(self):
        """Start keyboard listener for spacebar and answer end detection"""
        def on_press(key):
            try:
                if key == keyboard.Key.space:
                    if self.is_recording and self.current_phase == 'question':
                        self.transition_to_answer_phase()
                else:

                    if self.is_recording and self.current_phase == 'answer':
                        self.end_answer_phase()
            except AttributeError:
                pass

        self.keyboard_listener = keyboard.Listener(on_press=on_press)
        self.keyboard_listener.start()

    def transition_to_answer_phase(self):
        """Transition from question phase to answer phase"""
        if self.current_phase == 'question':
            self.phase_transition_time = time.time() - self.start_time
            self.current_phase = 'answer'
            self.phase_transition_flag.set()
            self.answer_end_flag.clear()
            print(f"\n{'='*50}")
            print(f"✓ TRANSITIONED TO ANSWER PHASE at {self.phase_transition_time:.2f}s")
            print(f"{'='*50}\n")

    def end_answer_phase(self):
        """End answer phase recording"""
        if self.current_phase == 'answer':
            self.answer_end_flag.set()
            print(f"\n{'='*50}")
            print(f"✓ ANSWER PHASE ENDED - Press any key to stop recording")
            print(f"{'='*50}\n")

    def get_phase_data(self) -> Dict[str, List[Dict]]:
        """
        Get synchronized data separated by phase

        Returns:
            Dictionary with 'question' and 'answer' keys containing phase-specific data
        """
        if not self.phase_transition_time:

            return {
                'question': self.synchronized_data.copy(),
                'answer': []
            }

        question_data = []
        answer_data = []

        for entry in self.synchronized_data:
            seg_start = entry['speech_segment']['start_time']
            if seg_start < self.phase_transition_time:
                question_data.append(entry)
            else:
                answer_data.append(entry)

        return {
            'question': question_data,
            'answer': answer_data
        }

    def wait_for_phase_transition(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for phase transition (spacebar press)

        Args:
            timeout: Maximum time to wait in seconds (None for infinite)

        Returns:
            True if transition occurred, False if timeout
        """
        return self.phase_transition_flag.wait(timeout=timeout)

    def wait_for_answer_end(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for answer phase to end (any key press)

        Args:
            timeout: Maximum time to wait in seconds (None for infinite)

        Returns:
            True if answer phase ended, False if timeout
        """
        return self.answer_end_flag.wait(timeout=timeout)

    def stop_synchronized_recording(self):
        """Stop synchronized recording and process data"""
        if not self.is_recording:
            return

        print("\nStopping synchronized recording...")
        self.is_recording = False

        if hasattr(self, 'audio_stream'):
            self.audio_stream.stop_stream()
            self.audio_stream.close()

        self.board.stop_streaming()

        if self.eeg_thread:
            self.eeg_thread.join(timeout=2.0)
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)

        print("Processing and synchronizing data...")
        self._synchronize_data()

        print("✓ Recording stopped")
        print(f"  - Speech segments detected: {len(self.speech_segments)}")
        print(f"  - EEG segments collected: {len(self.eeg_segments)}")
        print(f"  - Synchronized pairs: {len(self.synchronized_data)}")

    def _synchronize_data(self):
        """
        Synchronize speech segments with EEG data
        Maps each speech segment to corresponding EEG data
        """

        eeg_data_list = []
        while not self.eeg_data_queue.empty():
            eeg_data_list.append(self.eeg_data_queue.get())

        eeg_data_list.sort(key=lambda x: x[0])

        all_eeg_data = self.board.get_data()
        eeg_channels = self.board.get_eeg_channels()
        eeg_data = all_eeg_data[eeg_channels, :]

        total_duration = all_eeg_data.shape[1] / self.eeg_sampling_rate
        time_per_sample = 1.0 / self.eeg_sampling_rate

        for speech_seg in self.speech_segments:
            start_time = speech_seg['start_time']
            end_time = speech_seg['end_time']

            start_sample = int(start_time * self.eeg_sampling_rate)
            end_sample = int(end_time * self.eeg_sampling_rate)

            start_sample = max(0, min(start_sample, eeg_data.shape[1] - 1))
            end_sample = max(start_sample + 1, min(end_sample, eeg_data.shape[1]))

            eeg_segment = eeg_data[:, start_sample:end_sample]

            phase = 'question'
            if self.phase_transition_time and start_time >= self.phase_transition_time:
                phase = 'answer'

            sync_entry = {
                'speech_segment': {
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time,
                    'audio_data': speech_seg['audio_data'],
                    'speech_probability': speech_seg.get('speech_probability', 0.0)
                },
                'eeg_segment': {
                    'start_time': start_time,
                    'end_time': end_time,
                    'start_sample': start_sample,
                    'end_sample': end_sample,
                    'eeg_data': eeg_segment,
                    'sampling_rate': self.eeg_sampling_rate
                },
                'phase': phase,
                'timestamp': datetime.now().isoformat(),
                'synchronization_quality': 'good' if (end_sample - start_sample) > 0 else 'poor'
            }

            self.synchronized_data.append(sync_entry)

        self.synchronized_data.sort(key=lambda x: x['speech_segment']['start_time'])

    def get_synchronized_data(self) -> List[Dict]:
        """
        Get synchronized speech-EEG data

        Returns:
            List of dictionaries containing synchronized speech and EEG segments
        """
        return self.synchronized_data

    def get_speech_segments(self) -> List[Dict]:
        """Get detected speech segments"""
        return self.speech_segments

    def save_synchronized_data(self, filepath: str):
        """
        Save synchronized data to file

        Args:
            filepath: Path to save the data (will use .npz format)
        """
        if not self.synchronized_data:
            print("No synchronized data to save")
            return

        save_data = {
            'speech_segments': [],
            'eeg_segments': [],
            'metadata': {
                'sample_rate_audio': self.sample_rate,
                'sample_rate_eeg': self.eeg_sampling_rate,
                'num_segments': len(self.synchronized_data),
                'recording_start': self.start_time
            }
        }

        for entry in self.synchronized_data:
            save_data['speech_segments'].append(entry['speech_segment'])
            save_data['eeg_segments'].append(entry['eeg_segment'])

        np.savez_compressed(
            filepath,
            **save_data,
            allow_pickle=True
        )

        print(f"Synchronized data saved to {filepath}")

    @staticmethod
    def list_audio_devices():
        """
        List all available audio input devices

        Returns:
            List of dictionaries with device information
        """
        audio = pyaudio.PyAudio()
        devices = []

        print("\n" + "="*70)
        print("Available Audio Input Devices")
        print("="*70)

        try:
            default_device = audio.get_default_input_device_info()
            default_index = default_device['index']

            for i in range(audio.get_device_count()):
                device_info = audio.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    is_default = (i == default_index)
                    devices.append({
                        'index': i,
                        'name': device_info['name'],
                        'channels': device_info['maxInputChannels'],
                        'sample_rate': int(device_info['defaultSampleRate']),
                        'is_default': is_default
                    })

                    default_marker = " (DEFAULT)" if is_default else ""
                    print(f"  [{i}] {device_info['name']}{default_marker}")
                    print(f"      Channels: {device_info['maxInputChannels']}, "
                          f"Sample Rate: {int(device_info['defaultSampleRate'])} Hz")
        finally:
            audio.terminate()

        print("="*70 + "\n")
        return devices

    def cleanup(self):
        """Clean up resources"""

        if self.keyboard_listener:
            self.keyboard_listener.stop()
            self.keyboard_listener = None

        if hasattr(self, 'audio'):
            self.audio.terminate()

def test_synchronizer():
    """Test function for the synchronizer"""
    import sys

    serial_port = sys.argv[1] if len(sys.argv) > 1 else None

    board = NeuropawnKnightBoard(serial_port=serial_port)

    try:
        if not board.connect():
            print("Failed to connect to board")
            return

        synchronizer = SpeechEEGSynchronizer(board)

        print("\n" + "="*60)
        print("Starting synchronized recording for 10 seconds...")
        print("Please speak during this time")
        print("="*60 + "\n")

        synchronizer.start_synchronized_recording(duration=10.0)

        synchronized_data = synchronizer.get_synchronized_data()

        print("\n" + "="*60)
        print("Results:")
        print("="*60)
        for i, entry in enumerate(synchronized_data):
            speech = entry['speech_segment']
            eeg = entry['eeg_segment']
            print(f"\nSegment {i+1}:")
            print(f"  Speech: {speech['start_time']:.2f}s - {speech['end_time']:.2f}s "
                  f"(duration: {speech['duration']:.2f}s)")
            print(f"  EEG: samples {eeg['start_sample']} - {eeg['end_sample']} "
                  f"(shape: {eeg['eeg_data'].shape})")
            print(f"  Quality: {entry['synchronization_quality']}")

        synchronizer.save_synchronized_data('test_synchronized_data.npz')

        synchronizer.cleanup()
        board.disconnect()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        synchronizer.cleanup()
        board.disconnect()

if __name__ == "__main__":
    test_synchronizer()
