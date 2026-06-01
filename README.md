# Car Sales Analytics Pipeline

A data engineering & analytics project that processes, cleans, and analyzes ~24K car sales records to uncover pricing trends, regional patterns, and feature correlations.

Built with Python's scientific stack and orchestrated via `uv`.

## Tech Stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.14 |
| Package mgmt | `uv` (pip alternative) |
| Data processing | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Statistics | scipy |

## Quick Start

```bash
# Clone & enter
git clone https://github.com/Abdumalik-ProDev/Car-Sales-Data-Engineering.git
cd car-sales-data-engineering

# Create venv & install deps
uv sync

# Run the full pipeline
uv run python src/main.py
```

Outputs land in `outputs/`:
- `cleaned_data.csv` — preprocessed dataset
- `figures/` — 15 auto-generated visualizations

## Project Structure

```
car-sales-analytics/
├── data/
│   └── Car sales.csv              # Raw dataset (23,906 rows)
├── outputs/
│   ├── cleaned_data.csv            # Cleaned & transformed dataset
│   └── figures/                    # 15 analysis charts (PNG)
├── src/
│   ├── main.py                     # Pipeline entry point
│   ├── analysis.py                 # All ETL + analytics logic
│   └── __init__.py
├── pyproject.toml                  # Project config & deps
├── uv.lock                         # Lockfile for reproducible builds
└── .python-version                 # Python version pinning
```

## Pipeline Overview

### 1. Data Ingestion & Cleaning
- Encoding normalization, date parsing, column standardization
- Missing value imputation, duplicate removal
- Outlier capping via IQR

### 2. Exploratory Analysis

15 business questions across 4 categories:

**Sales & Revenue**
- Price distribution (histogram + KDE)
- Monthly sales trends (time series)
- Regional sales volume
- Dealer price comparison

**Customer Demographics**
- Gender split
- Income variation by region
- Income vs. purchase price correlation

**Product Insights**
- Average price by manufacturer
- Price variation by body style
- Automatic vs. manual pricing (t-test)
- Popular car colors
- Body style × transmission combos (heatmap)

**Statistical Modeling**
- Multiple linear regression (price ~ income + transmission + engine + body style)
- Z-score outlier detection
- Normality testing (Shapiro-Wilk + Q-Q plot)

## Key Findings

| Metric | Value |
|--------|-------|
| Total revenue | $655.6M |
| Avg car price | $27,426 (median $23,000) |
| Top body styles | SUV (27%), Hatchback (26%), Sedan (19%) |
| Top region | Austin (17% of sales) |
| Highest avg price brand | Cadillac ($37,557) |
| Gender split | 79% Male / 21% Female |
| Transmission | 53% Automatic / 47% Manual |

## Development

```bash
# Add a dependency
uv add <package>

# Run in an existing venv without re-syncing
uv run src/main.py

# Update lockfile
uv lock
```

## Author: 
Abdumalik-ProDev

## License

MIT