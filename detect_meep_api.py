#!/usr/bin/env python3
"""Detect MEEP's actual API"""

import meep as mp

print("Checking MEEP API...")
print(f"MEEP module: {mp}")
print()

# Check for coordinate constructors
checks = [
    ('mp.vec', 'vec'),
    ('mp.Vector3', 'Vector3'),
    ('mp.vector3', 'vector3'),
]

for name, attr in checks:
    if hasattr(mp, attr):
        print(f"✓ Found: {name}")
        obj = getattr(mp, attr)
        print(f"  Type: {type(obj)}")
        try:
            result = obj(1, 2, 3)
            print(f"  Test: {name}(1,2,3) = {result}")
            print(f"  SUCCESS!")
            break
        except Exception as e:
            print(f"  Test failed: {e}")
    else:
        print(f"✗ Not found: {name}")

print()
print("All MEEP attributes containing 'vec' or 'Vector':")
for attr in sorted(dir(mp)):
    if 'vec' in attr.lower() or 'vector' in attr.lower():
        print(f"  {attr}")
