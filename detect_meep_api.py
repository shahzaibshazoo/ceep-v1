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

print()
print("Checking if MEEP uses tuples/lists for coordinates...")
print("All MEEP classes (first 50):")
for attr in sorted(dir(mp))[:50]:
    if not attr.startswith('_'):
        obj = getattr(mp, attr)
        if type(obj).__name__ in ['type', 'class']:
            print(f"  {attr}: {type(obj)}")

print()
print("Testing MEEP simulation with tuple coordinates...")
try:
    # Try using plain tuples
    cell = (10, 10, 0)
    sources = [mp.Source(
        mp.GaussianSource(frequency=1.0, fwidth=0.2),
        component=mp.Ez,
        center=(0, 0, 0)
    )]
    sim = mp.Simulation(
        cell_size=cell,
        sources=sources,
        resolution=10
    )
    print("✓ Tuples work for coordinates!")
except Exception as e:
    print(f"✗ Tuples don't work: {e}")

print()
print("Checking mp.Source signature...")
import inspect
if hasattr(mp, 'Source'):
    print(inspect.signature(mp.Source.__init__))
else:
    print("mp.Source doesn't exist!")

print()
print("="*60)
print("ALL MEEP ATTRIBUTES (showing everything):")
print("="*60)
for attr in sorted(dir(mp)):
    if not attr.startswith('_'):
        obj = getattr(mp, attr)
        print(f"{attr:<30} {type(obj).__name__}")

print()
print("="*60)
print("CHECKING FOR SUBMODULES:")
print("="*60)
for attr in ['meep_swig', 'simulation', 'source']:
    if hasattr(mp, attr):
        submod = getattr(mp, attr)
        print(f"\nmp.{attr} has:")
        for item in dir(submod)[:10]:
            if not item.startswith('_'):
                print(f"  {item}")
