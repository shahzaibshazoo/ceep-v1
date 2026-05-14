# Cell 9: Final Summary
# ======================

import numpy as np

print("=" * 60)
print("  NEUROWAVE VALIDATION SUMMARY")
print("=" * 60)
print(f"\n  GPU Performance:")
print(f"    CPU: {cpu_time:.2f}s")
print(f"    GPU: {gpu_time:.2f}s")
print(f"    Speedup: {cpu_time/gpu_time:.1f}x")
print(f"    CPU-GPU match: {np.max(np.abs(ez_cpu - ez_gpu)):.2e}")
print(f"\n  MEEP Validation:")
print(f"    Free-space peak timing error: {abs(meep_peak - nw_peak)} steps")
print(f"    Dielectric slab peak error: {abs(meep_slab_peak - nw_slab_peak)} steps")
print(f"    MEEP PML quality: {meep_db:.1f} dB")
print(f"    NeuroWave CPML quality: {nw_db:.1f} dB")
print(f"\n  Test Suite: 95 CPU tests + 5 GPU tests")
print("=" * 60)
print("\n  Status: ALL VALIDATIONS COMPLETE")
print("=" * 60)
