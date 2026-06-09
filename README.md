# Car Sales Analytics Dashboard

A data engineering & analytics project processing 23,906 car sales records with an interactive web dashboard.

**Student:** Baxtiyorov Abdumalik | **Group:** 25-102 | **ID:** 250076

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.10+ |
| Package mgmt | `uv` |
| Web UI | Streamlit |
| Data processing | pandas, numpy |
| Visualization | matplotlib |
| Statistics | scipy |

---

## Quick Start

```bash
# Install dependencies
uv sync

# Launch web dashboard
uv run streamlit run src/ui.py

# Or via entry point
uv run python -m src.main

# Run full pipeline (generates all 15 figures)
uv run python -m src.main --pipeline
```

---

## Dashboard Features

| Feature | Description |
|---------|-------------|
| **📊 Overview** | Column summary, sample data, descriptive statistics |
| **💰 Sales & Revenue** | Price distribution, monthly trends, regional sales, dealer prices |
| **👥 Demographics** | Gender split, income by region, income vs price regression |
| **🔧 Product Insights** | Company prices, body style analysis, auto vs manual, colors, heatmap |
| **📈 Statistical Modeling** | Multiple regression, Z-score outliers, normality test |
| **🔍 Filter & Explore** | Multi-dimensional filtering with CSV export |
| **⚖️ Compare Segments** | Side-by-side comparison with Welch t-test and plots |

---

## Project Structure

```
car-sales-analytics/
├── data/
│   └── Car sales.csv              # Raw dataset (23,906 rows)
├── outputs/
│   ├── cleaned_data.csv            # Cleaned dataset
│   └── figures/                    # 15 analysis charts (PNG)
├── src/
│   ├── __init__.py
│   ├── analysis.py                 # Core engine: ETL, stats, modeling
│   ├── ui.py                       # Streamlit web dashboard
│   └── main.py                     # Entry point
├── main.py                         # Root entry point
├── pyproject.toml
└── README.md
```

---

## Pipeline — 15 Business Questions

**Sales & Revenue:** Q1 Price Distribution · Q2 Monthly Sales Trend · Q3 Sales by Region · Q14 Dealer Prices

**Customer Demographics:** Q4 Gender Split · Q5 Income by Region · Q6 Income vs Price (Regression)

**Product Insights:** Q7 Avg Price by Company · Q8 Price by Body Style · Q9 Auto vs Manual (t-test) · Q10 Popular Colors · Q11 Body Style × Transmission (Heatmap)

**Statistical Modeling:** Q12 Multiple Linear Regression · Q13 Z-score Outliers · Q15 Normality Test (Shapiro-Wilk)

---

## Key Findings

| Metric | Value |
|--------|-------|
| Total records | 23,906 |
| Total revenue | $655.6M |
| Avg car price | $27,426 (median $23,000) |
| Top body style | SUV (27%) |
| Top region | Austin (17%) |
| Highest avg price brand | Cadillac ($37,557) |
| Gender split | 79% Male / 21% Female |
| Transmission | 53% Automatic / 47% Manual |
