# Cell 1: Setup & Install
# ========================
# Run this first in Google Colab
# Make sure you selected GPU runtime: Runtime > Change runtime type > T4 GPU

# Clone private repo — replace YOUR_TOKEN with a GitHub personal access token
# Generate token at: https://github.com/settings/tokens (give it 'repo' scope)
TOKEN = "YOUR_GITHUB_TOKEN_HERE"

import os
os.system(f"git clone https://{TOKEN}@github.com/shahzaibshazoo/ceep-v1.git")
os.chdir("ceep-v1")

# Install dependencies
os.system("pip install -e . -q 2>/dev/null || pip install numpy pytest -q")
os.system("pip install cupy-cuda12x -q")
os.system("pip install meep -q")

print("\n✅ Setup complete!")
