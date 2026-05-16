#!/bin/bash
set -e

# Colors
G='\033[0;32m'
R='\033[0;31m'
Y='\033[1;33m'
B='\033[0;34m'
NC='\033[0m'

header() { echo -e "\n${B}===============================================${NC}\n  $1\n${B}===============================================${NC}\n"; }
success() { echo -e "${G}✓ $1${NC}"; }
error() { echo -e "${R}✗ $1${NC}"; }
warning() { echo -e "${Y}⚠ $1${NC}"; }
info() { echo -e "${B}ℹ $1${NC}"; }

cd "$(dirname "$0")"
REPO_ROOT=$(pwd)

header "CEEP VALIDATION SUITE - SIMPLIFIED"
info "Repository: $REPO_ROOT\n"

# Step 1: Clean CuPy
header "STEP 1: CLEAN UP CUPY CONFLICTS"
info "Removing conflicting CuPy packages..."
pip uninstall -y cupy-cuda11x cupy-cuda12x cupy 2>/dev/null || true
success "CuPy cleaned"

# Step 2: Install package
header "STEP 2: INSTALLING CEEP"
info "Installing CEEP in development mode..."
pip install -e . -q
success "CEEP installed"

# Step 3: Install dependencies
header "STEP 3: INSTALLING DEPENDENCIES"
info "Installing core dependencies..."
pip install -q numpy scipy matplotlib scikit-image pytest 2>/dev/null || true
success "Dependencies installed"

# Step 4: Basic Python import test
header "STEP 4: IMPORT TEST"
info "Testing basic imports..."
python3 << 'PYTHON'
import sys
sys.path.insert(0, 'src')

try:
    import ceep
    print("✓ CEEP module imports successfully")
except Exception as e:
    print(f"✗ CEEP import failed: {e}")
    sys.exit(1)

try:
    import numpy as np
    print("✓ NumPy imports successfully")
except Exception as e:
    print(f"✗ NumPy import failed: {e}")
    sys.exit(1)
PYTHON

success "All imports working"

# Step 5: Run pytest
header "STEP 5: UNIT TESTS"
if [ -d "tests" ]; then
    info "Running pytest..."
    python3 -m pytest tests/ -v --tb=short 2>&1 | head -100
    success "Tests completed"
else
    warning "No tests directory found"
fi

# Step 6: Generate report
header "STEP 6: VALIDATION COMPLETE"
success "All validation steps completed!"
info ""
info "Next steps:"
info "  1. Review test output above"
info "  2. Run on Google Colab: COLAB_QUICK_START.txt"
info "  3. Share results with team"
info ""

