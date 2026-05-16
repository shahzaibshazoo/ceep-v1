#!/usr/bin/env python3
"""Test runner script to verify test suite installation."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Now run pytest
import pytest

if __name__ == '__main__':
    sys.exit(pytest.main(sys.argv[1:]))
