"""
Utility script to list available audio input devices
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.speech_eeg_synchronizer import SpeechEEGSynchronizer

def main():
    """List all available audio input devices"""
    print("\n" + "="*70)
    print("AUDIO INPUT DEVICE LISTER")
    print("="*70)
    print("\nThis script lists all available audio input devices.")
    print("Use the device index to specify which microphone to use.\n")

    devices = SpeechEEGSynchronizer.list_audio_devices()

    if devices:
        print(f"\nFound {len(devices)} input device(s)")
        print("\nTo use a specific device, pass its index when initializing:")
        print("  detector = HybridLieDetectorSystem(")
        print("      serial_port='/dev/cu.usbmodem...',")
        print("      speech_input_device_index=2  # Use device index 2")
        print("  )")
    else:
        print("\nNo input devices found!")

if __name__ == "__main__":
    main()
