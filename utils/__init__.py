"""
Utils package - Contains utility modules for board connection, speech synchronization, etc.
"""

from .board_initializer import NeuropawnKnightBoard
from .speech_eeg_synchronizer import SpeechEEGSynchronizer

__all__ = ['NeuropawnKnightBoard', 'SpeechEEGSynchronizer']
