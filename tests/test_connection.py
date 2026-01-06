"""
Test script to verify Neuropawn Knight Board connection
Tests board initialization, connection, and data collection
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import subprocess
from utils.board_initializer import NeuropawnKnightBoard

def find_serial_ports():
    """Find available serial ports on macOS"""
    try:

        result = subprocess.run(['ls', '/dev/cu.*'],
                              capture_output=True,
                              text=True,
                              shell=True,
                              stderr=subprocess.DEVNULL)
        if result.returncode == 0 and result.stdout.strip():
            ports = [p.strip() for p in result.stdout.strip().split('\n') if p.strip()]

            usb_ports = [p for p in ports if 'usb' in p.lower() or 'usbserial' in p.lower() or 'usbmodem' in p.lower()]
            return usb_ports if usb_ports else ports
        return []
    except:
        return []

def test_board_connection(serial_port=None):
    """Test the board connection and data collection"""
    print("=" * 70)
    print("Neuropawn Knight Board Connection Test")
    print("=" * 70)

    if not serial_port:
        print("\nSearching for available serial ports...")
        ports = find_serial_ports()

        if ports:
            print(f"\nFound {len(ports)} potential port(s):")
            for i, port in enumerate(ports, 1):
                print(f"  {i}. {port}")

            if len(ports) == 1:
                serial_port = ports[0]
                print(f"\nUsing: {serial_port}")
            else:

                usb_ports = [p for p in ports if 'usb' in p.lower()]
                if usb_ports:
                    serial_port = usb_ports[0]
                    print(f"\nAuto-selecting first USB port: {serial_port}")
                else:
                    serial_port = ports[0]
                    print(f"\nAuto-selecting first port: {serial_port}")
        else:
            print("\nNo serial ports found automatically.")
            print("Common macOS ports:")
            print("  - /dev/cu.usbserial-*")
            print("  - /dev/cu.usbmodem*")

            import glob
            common_ports = glob.glob('/dev/cu.usb*')
            if common_ports:
                serial_port = common_ports[0]
                print(f"\nFound port: {serial_port}")
            else:
                print("\nPlease provide serial port as command line argument:")
                print("  python3 test_connection.py /dev/cu.usbmodem*")
                return False

    if not serial_port:
        print("No serial port provided. Exiting.")
        return False

    print(f"\nTesting connection to: {serial_port}")
    print("-" * 70)

    board = None
    try:

        print("\n1. Initializing board...")
        board = NeuropawnKnightBoard(serial_port=serial_port)
        print("   ✓ Board object created")

        print("\n2. Connecting to board...")
        if not board.connect():
            print("   ✗ Failed to connect")
            print("\nTroubleshooting:")
            print("  - Check USB connection")
            print("  - Verify serial port is correct")
            print("  - On macOS, use /dev/cu.* not /dev/tty.*")
            print("  - Try unplugging and replugging the device")
            return False

        print("   ✓ Connected successfully!")

        print("\n3. Board Information:")
        board.get_board_info()

        print("\n4. Testing data streaming...")
        print("   Starting stream...")
        board.start_streaming()
        print("   ✓ Stream started")

        print("\n5. Collecting test data (3 seconds)...")
        print("   Please stay still and avoid movement...")
        for i in range(3, 0, -1):
            print(f"   {i}...", end='\r')
            time.sleep(1)
        print("   Done!    ")

        print("\n6. Retrieving collected data...")
        data = board.get_data()
        print(f"   ✓ Data retrieved: shape {data.shape}")

        board.stop_streaming()
        print("   ✓ Stream stopped")

        print("\n7. Data Analysis:")
        eeg_channels = board.get_eeg_channels()
        sampling_rate = board.get_sampling_rate()

        if len(eeg_channels) > 0:
            eeg_data = data[eeg_channels, :]
            print(f"   EEG channels: {eeg_channels}")
            print(f"   EEG data shape: {eeg_data.shape}")
            print(f"   Sampling rate: {sampling_rate} Hz")
            print(f"   Duration: {eeg_data.shape[1] / sampling_rate:.2f} seconds")

            if eeg_data.shape[0] > 0:
                first_channel = eeg_data[0, :]
                print(f"\n   First EEG channel statistics:")
                print(f"     Mean: {first_channel.mean():.4f}")
                print(f"     Std:  {first_channel.std():.4f}")
                print(f"     Min:  {first_channel.min():.4f}")
                print(f"     Max:  {first_channel.max():.4f}")

                if abs(first_channel.mean()) < 1000 and first_channel.std() < 1000:
                    print("     ✓ Values look reasonable for EEG")
                else:
                    print("     ⚠ Values may need scaling/calibration")

        print("\n" + "=" * 70)
        print("✓ CONNECTION TEST SUCCESSFUL!")
        print("=" * 70)
        print("\nYour Neuropawn Knight Board is working correctly!")
        print("You can now use:")
        print("  - example_two_phase_training.py for training data collection")
        print("  - lie_detector_hybrid.py for Hybrid CNN-RNN-Transformer model")
        print("  - example_speech_eeg.py for speech-EEG synchronization")
        print("=" * 70)

        return True

    except KeyboardInterrupt:
        print("\n\n⚠ Test interrupted by user")
        return False
    except Exception as e:
        print(f"\n✗ Error during test: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("  - Check USB connection")
        print("  - Verify device is powered on")
        print("  - Try a different USB port")
        print("  - Check if device appears in System Information")
        return False
    finally:

        if board:
            try:
                board.stop_streaming()
                board.disconnect()
            except:
                pass

def main():
    """Main test function"""
    serial_port = sys.argv[1] if len(sys.argv) > 1 else None

    success = test_board_connection(serial_port)

    if success:
        print("\n✓ All tests passed! Your board is ready to use.")
        sys.exit(0)
    else:
        print("\n✗ Connection test failed. Please check the troubleshooting tips above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
