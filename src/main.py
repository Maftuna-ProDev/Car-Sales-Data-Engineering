"""
Car Sales Data Analysis — Entry Point
Big Data & Business Analytics (Unit 10)

Usage:
    uv run streamlit run src/ui.py     Launch Streamlit dashboard
    uv run python -m src.main          Launch Streamlit dashboard
    uv run python -m src.main --pipeline   Run full pipeline (CLI batch mode)
    uv run python -m src.main --help       Show help
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis import CarSalesAnalyzer


def run_pipeline():
    """Execute the full analysis pipeline non-interactively."""
    print("=" * 65)
    print("  CAR SALES DATA ANALYSIS — BIG DATA & BUSINESS ANALYTICS")
    print("  Author: Abdumalik-ProDev")
    print("=" * 65)

    print("\n[1/3] Loading and cleaning data...")
    analyzer = CarSalesAnalyzer()
    analyzer.load_and_clean()

    print("\n[2/3] Saving cleaned dataset...")
    path = analyzer.save_cleaned()
    print(f"  -> {path}")

    print("\n[3/3] Running all 15 analyses and generating figures...")
    t0 = time.time()
    results = analyzer.run_all(plot=True)
    elapsed = time.time() - t0

    print(f"\nPipeline complete in {elapsed:.2f}s")
    print(f"Figures saved to: {analyzer.figures_dir}/")
    print(f"Cleaned data:    {analyzer.cleaned_path}")

    for key, result in results.items():
        label = result.title.split(":")[0] if ":" in result.title else result.title
        status = "✓" if result.figure_path else " "
        print(f"  [{status}] {key}: {label}")

    print(f"\nWeb dashboard:  uv run streamlit run src/ui.py")


def run_streamlit():
    ui_path = Path(__file__).resolve().parent / "ui.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(ui_path)]
    subprocess.run(cmd, check=True)


def show_help():
    print("Car Sales Data Analysis — Big Data & Business Analytics (Unit 10)")
    print("Author: Abdumalik-ProDev")
    print()
    print("Usage:")
    print("  uv run streamlit run src/ui.py           Web dashboard")
    print("  uv run python -m src.main                Web dashboard")
    print("  uv run python -m src.main --pipeline     Batch pipeline (CLI)")
    print("  uv run python -m src.main --help         This help")
    print()
    print("Dashboard features:")
    print("  - 15 business questions with answers & interactive charts")
    print("  - Real-time controls (bins, themes, chart types)")
    print("  - Filter & Explore with CSV export")
    print("  - Segment comparison with Welch t-test")


def main():
    args = [a.lower() for a in sys.argv[1:]]

    if "--help" in args or "-h" in args:
        show_help()
        return

    if "--pipeline" in args or "-p" in args:
        run_pipeline()
        return

    run_streamlit()


if __name__ == "__main__":
    main()
