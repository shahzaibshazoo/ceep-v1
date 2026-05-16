#!/bin/bash
# Quick fix script for known issues

echo "🔧 Quick Fix: CuPy and imports"

# Fix 1: Clean up conflicting CuPy packages
echo "Cleaning conflicting CuPy packages..."
pip uninstall -y cupy-cuda11x cupy-cuda12x cupy 2>/dev/null || true

# Fix 2: Install single CuPy version
echo "Installing CuPy (latest)..."
pip install -q cupy-cuda12x 2>/dev/null || {
    echo "Trying alternative: cupy-cuda11x..."
    pip install -q cupy-cuda11x 2>/dev/null || echo "CuPy install skipped (optional)"
}

# Fix 3: Reinstall CEEP to pick up new imports
echo "Reinstalling CEEP..."
cd "$(dirname "$0")"
pip install -e . -q

echo "✓ Quick fixes applied!"
echo ""
echo "Now run: bash RUN_EVERYTHING.sh"
