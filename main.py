#!/usr/bin/env python3
"""
Car Sales Data Analysis — Root Entry Point
Usage:
    uv run streamlit run src/ui.py     Launch web dashboard
    uv run python -m src.main          Launch web dashboard
    uv run python -m src.main --pipeline   Run full pipeline
"""

if __name__ == "__main__":
    import sys
    from src.main import main
    sys.exit(main())
