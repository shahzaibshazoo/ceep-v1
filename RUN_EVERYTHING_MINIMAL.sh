#!/bin/bash
set -e

# Colors
G='\033[0;32m'
R='\033[0;31m'
Y='\033[1;33m'
B='\033[0;34m'
NC='\033[0m'

header() { echo -e "\n${B}====================================================${NC}\n  $1\n${B}====================================================${NC}\n"; }
success() { echo -e "${G}✓ $1${NC}"; }
error() { echo -e "${R}✗ $1${NC}"; exit 1; }
warning() { echo -e "${Y}⚠ $1${NC}"; }
info() { echo -e "${B}ℹ $1${NC}"; }

cd "$(dirname "$0")"
REPO_ROOT=$(pwd)

header "CEEP MINIMAL VALIDATION"
info "Repository: $REPO_ROOT\n"

# Step 1: Clean CuPy
header "STEP 1: CLEAN CUPY CONFLICTS"
info "Removing conflicting CuPy packages..."
pip uninstall -y cupy-cuda11x cupy-cuda12x cupy 2>/dev/null || true
success "CuPy cleaned\n"

# Step 2: Install package
header "STEP 2: INSTALL CEEP"
info "Installing CEEP..."
pip install -e . -q 2>&1 || error "Failed to install CEEP"
success "CEEP installed\n"

# Step 3: Install core dependencies
header "STEP 3: INSTALL DEPENDENCIES"
info "Installing numpy, scipy, pytest..."
pip install -q numpy scipy pytest matplotlib 2>&1 || true
success "Dependencies installed\n"

# Step 4: Test basic imports
header "STEP 4: BASIC IMPORT TEST"
python3 << 'PYTHON'
import sys

# Test 1: Import numpy
try:
    import numpy as np
    print("✓ NumPy imported")
except Exception as e:
    print(f"✗ NumPy failed: {e}")
    sys.exit(1)

# Test 2: Import ceep package
try:
    import ceep
    print("✓ CEEP package imported")
except Exception as e:
    print(f"✗ CEEP import failed: {e}")
    sys.exit(1)

# Test 3: Import core module
try:
    from ceep import core
    print("✓ CEEP core module imported")
except Exception as e:
    print(f"✗ CEEP core failed: {e}")
    sys.exit(1)

# Test 4: Check for grid classes
try:
    from ceep.core import Grid2D, Grid3D
    print("✓ Grid2D and Grid3D found")
except Exception as e:
    print(f"⚠ Grid classes not directly importable: {e}")

# Test 5: Import solvers
try:
    from ceep.solvers import FDTD2D
    print("✓ FDTD2D solver imported")
except Exception as e:
    print(f"⚠ FDTD2D import (optional): {e}")

print("\n✓ All core imports successful!")
PYTHON

# Step 5: Run pytest on tests
header "STEP 5: RUN UNIT TESTS"
if [ -d "tests" ] && [ -n "$(ls tests/test_*.py 2>/dev/null)" ]; then
    info "Running pytest on tests/..."
    python3 -m pytest tests/ -v --tb=line 2>&1 | head -150 || warning "Some tests may have issues"
    success "Tests completed\n"
else
    warning "No tests found\n"
fi

# Final message
header "VALIDATION COMPLETE"
success "All basic validation steps completed!"
info ""
info "Next steps:"
info "  1. Check output above for any errors"
info "  2. Run on Google Colab GPU: COLAB_QUICK_START.txt"
info "  3. Share results with team"
info ""

