"""
__init__.py
Part of the tests module.
Test suite for AI Trading Bot.
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test configuration
TEST_SYMBOL = "BTCUSDT"
TEST_INITIAL_CAPITAL = 100000
TEST_DATA_LENGTH = 100