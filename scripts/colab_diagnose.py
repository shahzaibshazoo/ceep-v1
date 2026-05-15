#!/usr/bin/env python3
"""
Diagnose Colab Import Issues
=============================

Run this to figure out why CEEP won't import.
"""

import os
import sys

print("="*70)
print(" CEEP Import Diagnostic")
print("="*70)

# 1. Current directory
print("\n[1/7] Current Directory:")
cwd = os.getcwd()
print(f"  {cwd}")

# 2. Check if we're in the repo
print("\n[2/7] Repository Structure:")
expected_files = ['src', 'examples', 'README.md', 'setup.py']
for f in expected_files:
    exists = os.path.exists(f)
    status = "✓" if exists else "✗"
    print(f"  {status} {f}")

# 3. Check src/ceep exists
print("\n[3/7] CEEP Source:")
ceep_paths = [
    'src/ceep',
    'src/ceep/__init__.py',
    'src/ceep/core',
    'src/ceep/solvers',
]
for path in ceep_paths:
    exists = os.path.exists(path)
    status = "✓" if exists else "✗"
    print(f"  {status} {path}")

# 4. List what's in src/
print("\n[4/7] Contents of src/:")
if os.path.exists('src'):
    contents = os.listdir('src')
    for item in contents[:10]:
        print(f"  - {item}")
else:
    print("  ✗ src/ does not exist!")

# 5. Python path
print("\n[5/7] Python Path:")
for i, path in enumerate(sys.path[:5]):
    print(f"  [{i}] {path}")

# 6. Try adding path and importing
print("\n[6/7] Attempting Import:")

# Try different path configurations
paths_to_try = [
    os.path.join(cwd, 'src'),
    '/content/ceep-v1/src',
    'src',
]

success = False
for path_attempt in paths_to_try:
    if path_attempt not in sys.path:
        sys.path.insert(0, path_attempt)

    print(f"\n  Trying path: {path_attempt}")
    print(f"    Exists: {os.path.exists(path_attempt)}")

    if os.path.exists(path_attempt):
        ceep_init = os.path.join(path_attempt, 'ceep', '__init__.py')
        print(f"    ceep/__init__.py exists: {os.path.exists(ceep_init)}")

    try:
        import ceep
        print(f"    ✓ SUCCESS! Imported from: {ceep.__file__}")
        success = True
        break
    except ImportError as e:
        print(f"    ✗ Failed: {e}")

# 7. Recommendation
print("\n[7/7] Recommendation:")
if success:
    print("  ✓ CEEP can be imported!")
    print(f"  Use this path: {sys.path[0]}")
    print("\n  Add this to your cells:")
    print(f"  import sys")
    print(f"  sys.path.insert(0, '{sys.path[0]}')")
else:
    print("  ✗ CEEP cannot be imported")
    print("\n  Possible issues:")
    print("    1. Not in ceep-v1 directory - run: %cd ceep-v1")
    print("    2. Repository incomplete - re-clone")
    print("    3. src/ceep folder missing - check git clone")

    print("\n  Debug commands to run:")
    print("    !pwd")
    print("    !ls -la")
    print("    !ls -la src/ 2>/dev/null || echo 'src not found'")
    print("    !git status 2>/dev/null || echo 'not a git repo'")

print("\n" + "="*70)
