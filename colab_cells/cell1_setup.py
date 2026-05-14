# Cell 1: Setup & Install
# ========================
# Run this first in Google Colab
# Make sure you selected GPU runtime: Runtime > Change runtime type > T4 GPU

# Option A: Public clone (if repo is public)
# !git clone https://github.com/shahzaibshazoo/ceep-v1.git

# Option B: Private clone with token
# Generate token at: https://github.com/settings/tokens (give it 'repo' scope)
TOKEN = "YOUR_GITHUB_TOKEN_HERE"  # <-- REPLACE THIS

import subprocess
import os

# Clone
result = subprocess.run(
    f"git clone https://{TOKEN}@github.com/shahzaibshazoo/ceep-v1.git",
    shell=True, capture_output=True, text=True
)

if result.returncode != 0:
    print("ERROR: Clone failed!")
    print(result.stderr)
    print("\nTroubleshooting:")
    print("1. Make sure you replaced YOUR_GITHUB_TOKEN_HERE with your actual token")
    print("2. Generate token at: https://github.com/settings/tokens")
    print("3. Give it 'repo' scope")
    print("4. Make sure the repo exists: github.com/shahzaibshazoo/ceep-v1")
else:
    os.chdir("ceep-v1")
    print("Cloned successfully!")

    # Install dependencies
    os.system("pip install -e . -q 2>&1 | tail -3")
    os.system("pip install cupy-cuda12x -q 2>&1 | tail -3")
    os.system("pip install meep -q 2>&1 | tail -3")
    print("\nSetup complete! Run the next cells.")
