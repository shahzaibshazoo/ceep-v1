#!/usr/bin/env python3
"""
Compare CEEP vs MEEP Results
============================

Loads ceep_results.json and meep_results.json and compares:
- S-parameters for each scenario
- Signal waveforms
- Accuracy metrics
- Performance comparison

Usage:
  python compare_ceep_meep.py
"""

import json
import numpy as np
import sys

print("="*80)
print(" CEEP vs MEEP COMPARISON")
print("="*80)

# Load results
try:
    with open('ceep_results.json', 'r') as f:
        ceep = json.load(f)
    print("\n✓ Loaded CEEP results")
except FileNotFoundError:
    print("\n❌ ceep_results.json not found!")
    print("   Run ceep_reference_simulation.py on Colab first")
    sys.exit(1)

try:
    with open('meep_results.json', 'r') as f:
        meep = json.load(f)
    print("✓ Loaded MEEP results")
except FileNotFoundError:
    print("\n❌ meep_results.json not found!")
    print("   Run meep_reference_simulation.py locally first")
    sys.exit(1)

# Verify parameters match
print("\n" + "="*80)
print(" PARAMETER VERIFICATION")
print("="*80)

params_match = True
for key in ['nx', 'ny', 'dx', 'frequency', 'total_steps']:
    ceep_val = ceep['parameters'][key]
    meep_val = meep['parameters'][key]
    match = "✓" if ceep_val == meep_val else "✗"
    print(f"{match} {key}: CEEP={ceep_val}, MEEP={meep_val}")
    if ceep_val != meep_val:
        params_match = False

if not params_match:
    print("\n⚠️  WARNING: Parameters don't match! Results may not be comparable.")
else:
    print("\n✓ All parameters match")

# Compare scenarios
print("\n" + "="*80)
print(" SCENARIO COMPARISON")
print("="*80)

scenarios = ['empty', 'brain', 'hemorrhage']
results = []

for scenario in scenarios:
    ceep_s = ceep['scenarios'][scenario]['s_parameter']
    meep_s = meep['scenarios'][scenario]['s_parameter']

    error = abs(ceep_s - meep_s)
    error_pct = (error / meep_s * 100) if meep_s != 0 else 0

    # Load signals
    ceep_real = np.array(ceep['scenarios'][scenario]['signal_real'])
    ceep_imag = np.array(ceep['scenarios'][scenario]['signal_imag'])
    meep_real = np.array(meep['scenarios'][scenario]['signal_real'])
    meep_imag = np.array(meep['scenarios'][scenario]['signal_imag'])

    ceep_signal = ceep_real + 1j * ceep_imag
    meep_signal = meep_real + 1j * meep_imag

    # Align signal lengths (may differ by 1-2 samples)
    min_len = min(len(ceep_signal), len(meep_signal))
    ceep_signal = ceep_signal[:min_len]
    meep_signal = meep_signal[:min_len]

    # Correlation
    correlation = np.corrcoef(np.abs(ceep_signal), np.abs(meep_signal))[0, 1]

    # RMS error
    rms_error = np.sqrt(np.mean(np.abs(ceep_signal - meep_signal)**2))

    results.append({
        'scenario': scenario,
        'ceep_s': ceep_s,
        'meep_s': meep_s,
        'error': error,
        'error_pct': error_pct,
        'correlation': correlation,
        'rms_error': rms_error,
        'passed': error_pct < 10.0
    })

# Print results table
print(f"\n{'Scenario':<15} {'CEEP':<12} {'MEEP':<12} {'Error':<10} {'Error %':<10} {'Status':<10}")
print("-"*80)

for r in results:
    scenario = r['scenario'].capitalize()
    ceep_s = f"{r['ceep_s']:.3f}"
    meep_s = f"{r['meep_s']:.3f}"
    error = f"{r['error']:.3f}"
    error_pct = f"{r['error_pct']:.1f}%"
    status = "✅ PASS" if r['passed'] else "❌ FAIL"

    print(f"{scenario:<15} {ceep_s:<12} {meep_s:<12} {error:<10} {error_pct:<10} {status:<10}")

print("-"*80)

# Signal correlation
print("\n" + "="*80)
print(" SIGNAL CORRELATION")
print("="*80)

for r in results:
    print(f"{r['scenario'].capitalize():<15} Correlation: {r['correlation']:.4f}  RMS Error: {r['rms_error']:.3f}")

# Performance comparison
print("\n" + "="*80)
print(" PERFORMANCE")
print("="*80)

ceep_total = sum(ceep['scenarios'][s]['runtime'] for s in scenarios)
meep_total = sum(meep['scenarios'][s]['runtime'] for s in scenarios)
speedup = meep_total / ceep_total if ceep_total > 0 else 0

print(f"CEEP ({ceep['backend']}):")
print(f"  Total runtime: {ceep_total:.2f}s")
print(f"  Average per scenario: {ceep_total/len(scenarios):.2f}s")

print(f"\nMEEP ({meep['backend']}):")
print(f"  Total runtime: {meep_total:.2f}s")
print(f"  Average per scenario: {meep_total/len(scenarios):.2f}s")

print(f"\nSpeedup: {speedup:.1f}x")

# Accuracy summary
print("\n" + "="*80)
print(" ACCURACY SUMMARY")
print("="*80)

errors = [r['error_pct'] for r in results]
passed = sum(1 for r in results if r['passed'])
total = len(results)

print(f"Scenarios passed: {passed}/{total}")
print(f"Average error: {np.mean(errors):.1f}%")
print(f"Max error: {np.max(errors):.1f}%")
print(f"Min error: {np.min(errors):.1f}%")

correlations = [r['correlation'] for r in results]
print(f"\nAverage correlation: {np.mean(correlations):.4f}")

# Final verdict
print("\n" + "="*80)
if passed == total and np.mean(errors) < 5.0:
    print("✅ EXCELLENT AGREEMENT!")
    print("\nCEEP accurately reproduces MEEP results.")
    print("Ready for production use and dataset generation.")
elif passed == total:
    print("✅ GOOD AGREEMENT")
    print("\nCEEP shows acceptable accuracy vs MEEP.")
    print(f"Average error: {np.mean(errors):.1f}%")
elif passed >= total // 2:
    print("⚠️  PARTIAL AGREEMENT")
    print(f"\n{passed}/{total} scenarios passed.")
    print("Review failed scenarios.")
else:
    print("❌ POOR AGREEMENT")
    print(f"\nOnly {passed}/{total} scenarios passed.")
    print("Significant differences detected - investigate.")

print("="*80)
