"""
Board Initialization Module for Neuropawn Knight Board
Handles connection and setup of the EEG device using BrainFlow
"""

import sys
import platform
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, LogLevels
from brainflow.data_filter import DataFilter

class NeuropawnKnightBoard:
    """
    Wrapper class for Neuropawn Knight Board initialization and management
    """

    def __init__(self, serial_port=None):
        """
        Initialize the Neuropawn Knight Board

        Args:
            serial_port (str): Serial port path (e.g., "COM3" for Windows,
                              "/dev/cu.*" for macOS, "/dev/tty.*" for Linux)
        """
        self.board = None
        self.params = BrainFlowInputParams()
        self.board_id = BoardIds.NEUROPAWN_KNIGHT_BOARD

        if serial_port:
            self.params.serial_port = serial_port
        else:

            self.params.serial_port = self._detect_serial_port()

        BoardShim.enable_board_logger()
        DataFilter.enable_data_logger()

    def _detect_serial_port(self):
        """
        Attempt to detect the serial port based on OS
        Returns a default port or prompts user
        """
        system = platform.system()

        if system == "Windows":
            return "COM3"
        elif system == "Darwin":
            return "/dev/cu.usbserial-*"
        elif system == "Linux":
            return "/dev/ttyUSB0"
        else:
            raise ValueError(f"Unsupported operating system: {system}")

    def connect(self):
        """
        Connect to the board and prepare the session

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            print(f"Initializing Neuropawn Knight Board...")
            print(f"Serial port: {self.params.serial_port}")

            self.board = BoardShim(self.board_id, self.params)

            print("Preparing session...")
            self.board.prepare_session()

            print("✓ Board connected successfully!")
            return True

        except Exception as e:
            print(f"✗ Error connecting to board: {e}")
            print("\nTroubleshooting tips:")
            print("1. Ensure the device is connected via USB")
            print("2. Check if the serial port is correct")
            print("3. On macOS, use /dev/cu.* instead of /dev/tty.*")
            print("4. On Linux, you may need to run with sudo or configure permissions")
            return False

    def start_streaming(self):
        """
        Start data streaming from the board
        """
        if not self.board:
            raise RuntimeError("Board not connected. Call connect() first.")

        try:
            print("Starting data stream...")
            self.board.start_stream()
            print("✓ Streaming started!")
        except Exception as e:
            print(f"✗ Error starting stream: {e}")
            raise

    def stop_streaming(self):
        """
        Stop data streaming
        """
        if not self.board:
            return

        try:
            print("Stopping data stream...")
            self.board.stop_stream()
            print("✓ Streaming stopped!")
        except Exception as e:
            print(f"✗ Error stopping stream: {e}")

    def get_data(self, num_samples=None):
        """
        Retrieve data from the board

        Args:
            num_samples (int): Number of samples to retrieve. If None, gets all available data

        Returns:
            numpy.ndarray: EEG data array with shape (channels, samples)
        """
        if not self.board:
            raise RuntimeError("Board not connected. Call connect() first.")

        if num_samples:
            data = self.board.get_current_board_data(num_samples)
        else:
            data = self.board.get_board_data()

        return data

    def get_eeg_channels(self):
        """
        Get the list of EEG channel indices for this board

        Returns:
            list: List of EEG channel indices
        """
        return BoardShim.get_eeg_channels(self.board_id)

    def get_sampling_rate(self):
        """
        Get the sampling rate of the board

        Returns:
            int: Sampling rate in Hz
        """
        return BoardShim.get_sampling_rate(self.board_id)

    def get_board_info(self):
        """
        Print information about the board configuration
        """
        if not self.board:
            print("Board not connected")
            return

        eeg_channels = self.get_eeg_channels()
        sampling_rate = self.get_sampling_rate()

        print("\n=== Board Information ===")
        print(f"Board ID: {self.board_id}")
        print(f"Serial Port: {self.params.serial_port}")
        print(f"Sampling Rate: {sampling_rate} Hz")
        print(f"EEG Channels: {eeg_channels}")
        print(f"Number of EEG Channels: {len(eeg_channels)}")
        print("=" * 25)

    def disconnect(self):
        """
        Release the board session and disconnect
        """
        if not self.board:
            return

        try:
            print("Releasing board session...")
            self.board.release_session()
            print("✓ Board disconnected!")
        except Exception as e:
            print(f"✗ Error disconnecting: {e}")

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_streaming()
        self.disconnect()

if __name__ == "__main__":

    import time

    serial_port = None

    if len(sys.argv) > 1:
        serial_port = sys.argv[1]

    try:
        with NeuropawnKnightBoard(serial_port=serial_port) as board:
            board.get_board_info()

            board.start_streaming()

            print("\nCollecting data for 5 seconds...")
            time.sleep(5)

            data = board.get_data()
            print(f"\nCollected data shape: {data.shape}")
            print(f"Data preview (first 5 samples of first channel): {data[0, :5]}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
