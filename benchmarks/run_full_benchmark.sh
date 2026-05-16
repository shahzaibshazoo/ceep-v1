#!/bin/bash
# ============================================================================
# Full Benchmark Suite Runner
# ============================================================================
# Runs complete benchmarking pipeline:
#   1. Execute benchmark measurements
#   2. Analyze results and generate report
#   3. Create publication-quality visualizations
#
# Usage:
#   ./benchmarks/run_full_benchmark.sh [--quick]
#
# Output:
#   - benchmarks/benchmark_raw_data.json (raw measurements)
#   - benchmarks/batched_2d_results.md (comprehensive report)
#   - benchmarks/plots/*.png (visualizations)
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "================================================================================"
echo "BATCHED 2D FDTD BENCHMARK SUITE"
echo "================================================================================"
echo ""
echo "Starting comprehensive benchmarking pipeline..."
echo ""

# Check for --quick flag
QUICK_FLAG=""
if [[ "$1" == "--quick" ]]; then
    QUICK_FLAG="--quick"
    echo "Mode: QUICK (reduced test matrix)"
else
    echo "Mode: FULL (comprehensive test matrix)"
fi
echo ""

# Step 1: Run benchmarks
echo "STEP 1: Running benchmark measurements..."
echo "───────────────────────────────────────────────────────────────────────────────"
cd "$PROJECT_ROOT"
python benchmarks/batched_2d_benchmark.py $QUICK_FLAG
benchmark_status=$?

if [ $benchmark_status -ne 0 ]; then
    echo "ERROR: Benchmark execution failed"
    exit 1
fi

echo ""
echo "✓ Benchmark measurements complete"
echo ""

# Step 2: Analyze results
echo "STEP 2: Analyzing results and generating report..."
echo "───────────────────────────────────────────────────────────────────────────────"
python benchmarks/analyze_results.py
analysis_status=$?

if [ $analysis_status -ne 0 ]; then
    echo "WARNING: Analysis failed (continuing)"
else
    echo ""
    echo "✓ Report generation complete"
fi
echo ""

# Step 3: Generate visualizations
echo "STEP 3: Creating publication-quality visualizations..."
echo "───────────────────────────────────────────────────────────────────────────────"
python benchmarks/generate_plots.py 2>/dev/null || {
    echo "WARNING: Visualization generation failed (matplotlib not available)"
}

echo ""
echo "================================================================================"
echo "BENCHMARK SUITE COMPLETE"
echo "================================================================================"
echo ""
echo "Output files:"
echo "  - benchmarks/benchmark_raw_data.json"
echo "  - benchmarks/batched_2d_results.md"
echo "  - benchmarks/plots/*.png"
echo ""
echo "Next steps:"
echo "  1. Review the report: cat benchmarks/batched_2d_results.md"
echo "  2. Check plots: ls -lh benchmarks/plots/"
echo "  3. Commit results: git add benchmarks/"
echo ""
