# Cell 3: Run GPU Validation Tests
# ==================================

import os
os.system("PYTHONPATH=./src python -m pytest tests/test_gpu.py -v")
