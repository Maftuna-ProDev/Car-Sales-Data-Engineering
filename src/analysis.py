"""
Car Sales Data Analysis Engine
Big Data & Business Analytics — Unit 10
Covers: A.D1, B.D2, B.M2, B.M3, B.P2, B.P3, C.D3, C.M4, C.M5, C.M6, C.P4, C.P5, C.P6

Architecture:
  CarSalesAnalyzer   — Core engine: ETL, statistics, modeling, plotting
  AnalysisResult     — Typed structured result from every analysis method
  PlotConfig         — Reusable figure configuration

Design principles:
  - Every analysis method returns structured data (dict/DataFrame),
    not side-effect print statements. Rendering is the caller's responsibility.
  - Plotting is optional and controlled via a `plot` parameter.
  - All paths are configurable; no hardcoded magic strings.
  - Robust error handling with informative messages.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, linregress, shapiro, norm, zscore

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "Car sales.csv")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
DEFAULT_FIGURES_DIR = os.path.join(DEFAULT_OUTPUT_DIR, "figures")
DEFAULT_CLEANED_PATH = os.path.join(DEFAULT_OUTPUT_DIR, "cleaned_data.csv")

NUMERIC_COLS = ["Price", "Annual_Income"]
CATEGORICAL_COLS = ["Gender", "Transmission", "Color", "Body_Style", "Engine", "Dealer_Region"]
DATE_COL = "Date"
ID_COL = "Car_id"


# ---------------------------------------------------------------------------
# Typed result containers
# ---------------------------------------------------------------------------

@dataclass
class AnalysisResult:
    """Generic structured result from any analysis method."""
    title: str
    description: str
    summary: dict[str, Any] = field(default_factory=dict)
    table: pd.DataFrame | None = None
    series: pd.Series | None = None
    figure_path: str | None = None
    raw: Any = None

    def __str__(self) -> str:
        lines = [f"[{self.title}] {self.description}"]
        if self.summary:
            for k, v in self.summary.items():
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------

def _configure_matplotlib():
    plt.rcParams.update({
        "figure.dpi": 150,
        "figure.figsize": (10, 6),
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "#f8f9fa",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def _save_figure(fig: plt.Figure, name: str, figures_dir: str) -> str:
    path = os.path.join(figures_dir, name)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Core Engine
# ---------------------------------------------------------------------------

class CarSalesAnalyzer:
    """End-to-end car sales data analysis engine.

    Usage:
        analyzer = CarSalesAnalyzer()
        analyzer.load_and_clean()
        result = analyzer.price_distribution(plot=True)
        print(result.summary)
    """

    def __init__(
        self,
        data_path: str = DEFAULT_DATA_PATH,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        figures_dir: str = DEFAULT_FIGURES_DIR,
        cleaned_path: str = DEFAULT_CLEANED_PATH,
    ):
        self.data_path = data_path
        self.output_dir = output_dir
        self.figures_dir = figures_dir
        self.cleaned_path = cleaned_path
        self.df: pd.DataFrame | None = None

        os.makedirs(self.figures_dir, exist_ok=True)
        _configure_matplotlib()

    # -----------------------------------------------------------------------
    # ETL Pipeline
    # -----------------------------------------------------------------------

    def load_and_clean(self) -> pd.DataFrame:
        """Load, parse, clean, and return the dataset.

        Steps:
          1. Read CSV with encoding detection
          2. Standardise column names (snake_case, strip special chars)
          3. Parse dates / coerce numeric types
          4. Drop rows missing critical values
          5. Remove duplicate Car_ids
          6. Cap outliers via IQR (preserving data volume)

        Returns:
            Cleaned DataFrame (stored internally as ``self.df``).
        """
        raw = pd.read_csv(self.data_path)

        df = raw.copy()

        # -- Column name normalisation -----------------------------------
        df.columns = (
            df.columns
            .str.strip()
            .str.replace(r"\s+", "_", regex=True)
            .str.replace(r"[($)]", "", regex=True)
            .str.replace("Dealer_No_", "Dealer_No")
        )
        df.rename(columns={"Price_": "Price"}, inplace=True)

        # -- Date parsing -------------------------------------------------
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], format="%m/%d/%Y", errors="coerce")
        pre_drop = len(df)
        df.dropna(subset=[DATE_COL], inplace=True)
        self._log(f"Date parse: dropped {pre_drop - len(df)} invalid-date rows")

        # -- String cleaning ----------------------------------------------
        df["Engine"] = df["Engine"].str.replace("Â", "", regex=False)
        for col in ["Company", "Model"]:
            df[col] = df[col].str.strip()
        df["Gender"] = df["Gender"].str.strip().str.title()

        # -- Numeric coercion ---------------------------------------------
        df["Annual_Income"] = pd.to_numeric(df["Annual_Income"], errors="coerce")
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

        before = len(df)
        df.dropna(subset=["Annual_Income", "Price"], inplace=True)
        self._log(f"Missing values: dropped {before - len(df)} rows without income/price")

        # -- Duplicate removal --------------------------------------------
        before = len(df)
        df.drop_duplicates(subset=[ID_COL], inplace=True)
        self._log(f"Duplicates: removed {before - len(df)} duplicate Car_id rows")

        # -- IQR outlier capping (preserve rows, cap extremes) ------------
        for col in NUMERIC_COLS:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n = ((df[col] < lo) | (df[col] > hi)).sum()
            df[col] = df[col].clip(lo, hi)
            self._log(f"Outlier capping ({col}): capped {n} values outside [{lo:,.0f}, {hi:,.0f}]")

        self._log(f"Final shape: {df.shape[0]} rows × {df.shape[1]} columns")
        self.df = df
        return df

    def save_cleaned(self, path: str | None = None) -> str:
        """Persist cleaned DataFrame to CSV."""
        if self.df is None:
            raise RuntimeError("No data loaded. Call load_and_clean() first.")
        path = path or self.cleaned_path
        self.df.to_csv(path, index=False)
        return path

    # -----------------------------------------------------------------------
    # Data exploration helpers
    # -----------------------------------------------------------------------

    def column_summary(self) -> pd.DataFrame:
        """Return a DataFrame with dtype, non-null count, unique count per column."""
        if self.df is None:
            raise RuntimeError("No data loaded.")
        info = []
        for col in self.df.columns:
            info.append({
                "column": col,
                "dtype": str(self.df[col].dtype),
                "non_null": self.df[col].notna().sum(),
                "null_pct": round(self.df[col].isna().mean() * 100, 2),
                "unique": self.df[col].nunique(),
            })
        return pd.DataFrame(info)

    def descriptive_stats(self, column: str) -> dict[str, float]:
        """Central tendency + dispersion for a numeric column.

        Includes non-routine metrics: skewness, kurtosis.
        """
        if self.df is None:
            raise RuntimeError("No data loaded.")
        data = self.df[column].dropna()
        q1, q3 = data.quantile(0.25), data.quantile(0.75)
        return {
            "mean": round(data.mean(), 2),
            "median": round(data.median(), 2),
            "mode": round(data.mode().iloc[0], 2),
            "min": round(data.min(), 2),
            "max": round(data.max(), 2),
            "range": round(data.max() - data.min(), 2),
            "variance": round(data.var(), 2),
            "std": round(data.std(), 2),
            "skewness": round(data.skew(), 4),
            "kurtosis": round(data.kurtosis(), 4),
            "q1": round(q1, 2),
            "q3": round(q3, 2),
            "iqr": round(q3 - q1, 2),
            "count": int(data.count()),
        }

    def frequency_table(self, column: str) -> pd.DataFrame:
        """Frequency and percentage for a categorical column."""
        if self.df is None:
            raise RuntimeError("No data loaded.")
        freq = self.df[column].value_counts()
        pct = self.df[column].value_counts(normalize=True) * 100
        result = pd.DataFrame({"count": freq, "percentage": pct.round(1)})
        result.index.name = column
        return result

    def filter_dataframe(
        self,
        region: str | None = None,
        company: str | None = None,
        body_style: str | None = None,
        transmission: str | None = None,
        color: str | None = None,
        price_range: tuple[float, float] | None = None,
        income_range: tuple[float, float] | None = None,
    ) -> pd.DataFrame:
        """Return a filtered view of the cleaned dataset."""
        if self.df is None:
            raise RuntimeError("No data loaded.")
        df = self.df.copy()
        if region:
            df = df[df["Dealer_Region"].str.contains(region, case=False, na=False)]
        if company:
            df = df[df["Company"].str.contains(company, case=False, na=False)]
        if body_style:
            df = df[df["Body_Style"].str.contains(body_style, case=False, na=False)]
        if transmission:
            df = df[df["Transmission"].str.contains(transmission, case=False, na=False)]
        if color:
            df = df[df["Color"].str.contains(color, case=False, na=False)]
        if price_range:
            df = df[df["Price"].between(*price_range)]
        if income_range:
            df = df[df["Annual_Income"].between(*income_range)]
        return df

    # -----------------------------------------------------------------------
    # Domain — 15 Business Questions
    # -----------------------------------------------------------------------

    # -- Q1: Price Distribution --------------------------------------------

    def price_distribution(self, plot: bool = True) -> AnalysisResult:
        """Distribution of car prices with histogram + KDE overlay."""
        prices = self.df["Price"].dropna()
        kde = stats.gaussian_kde(prices)
        x_kde = np.linspace(prices.min(), prices.max(), 500)

        result = AnalysisResult(
            title="Q1: Price Distribution",
            description="Histogram with KDE overlay of car prices",
            summary={
                "mean": f"${prices.mean():,.2f}",
                "median": f"${prices.median():,.2f}",
                "std": f"${prices.std():,.2f}",
                "min": f"${prices.min():,.2f}",
                "max": f"${prices.max():,.2f}",
                "skewness": f"{prices.skew():.4f}",
            },
        )

        if plot:
            fig, ax = plt.subplots()
            ax.hist(prices, bins=50, color="steelblue", edgecolor="white", alpha=0.8, density=True)
            ax.plot(x_kde, kde(x_kde), color="crimson", lw=2, label="KDE")
            ax.axvline(prices.mean(), color="darkorange", ls="--", lw=1.5,
                       label=f"Mean = ${prices.mean():,.0f}")
            ax.axvline(prices.median(), color="green", ls=":", lw=1.5,
                       label=f"Median = ${prices.median():,.0f}")
            ax.set(xlabel="Price ($)", ylabel="Density", title="Q1: Distribution of Car Prices")
            ax.legend()
            result.figure_path = _save_figure(fig, "q01_price_distribution.png", self.figures_dir)

        return result

    # -- Q2: Monthly Sales Trend -------------------------------------------

    def monthly_sales_trend(self, plot: bool = True) -> AnalysisResult:
        """Monthly sales volume over time."""
        monthly = self.df.set_index(DATE_COL).resample("ME").size()

        result = AnalysisResult(
            title="Q2: Monthly Sales Trend",
            description="Time series of monthly car sales volume",
            summary={
                "highest_month": f"{monthly.idxmax().strftime('%b %Y')} ({monthly.max()} sales)",
                "lowest_month": f"{monthly.idxmin().strftime('%b %Y')} ({monthly.min()} sales)",
                "avg_monthly": f"{monthly.mean():.0f}",
                "total_months": f"{len(monthly)}",
            },
            series=monthly,
        )

        if plot:
            fig, ax = plt.subplots()
            ax.plot(monthly.index, monthly.values, marker="o", color="steelblue", lw=1.5)
            ax.fill_between(monthly.index, monthly.values, alpha=0.15, color="steelblue")
            ax.set(xlabel="Date", ylabel="Number of Sales", title="Q2: Monthly Car Sales Trend")
            result.figure_path = _save_figure(fig, "q02_monthly_sales_trend.png", self.figures_dir)

        return result

    # -- Q3: Sales by Region -----------------------------------------------

    def sales_by_region(self, plot: bool = True) -> AnalysisResult:
        """Car sales volume by dealer region."""
        region_counts = self.df["Dealer_Region"].value_counts()

        result = AnalysisResult(
            title="Q3: Sales by Dealer Region",
            description="Total cars sold per region",
            summary={
                "top_region": f"{region_counts.index[0]} ({region_counts.iloc[0]} sales)",
                "num_regions": f"{len(region_counts)}",
            },
            series=region_counts,
        )

        if plot:
            fig, ax = plt.subplots()
            colors = plt.cm.Set2(np.linspace(0, 1, len(region_counts)))
            bars = ax.barh(region_counts.index, region_counts.values, color=colors)
            ax.bar_label(bars, labels=[f"{v}" for v in region_counts.values], padding=2)
            ax.set(xlabel="Number of Cars Sold", title="Q3: Sales by Dealer Region")
            ax.invert_yaxis()
            result.figure_path = _save_figure(fig, "q03_sales_by_region.png", self.figures_dir)

        return result

    # -- Q4: Gender Distribution -------------------------------------------

    def gender_distribution(self, plot: bool = True) -> AnalysisResult:
        """Gender split of customers."""
        g = self.df["Gender"].value_counts()
        total = g.sum()

        result = AnalysisResult(
            title="Q4: Customer Gender Distribution",
            description="Gender breakdown of car buyers",
            summary={k: f"{v} ({v / total * 100:.1f}%)" for k, v in g.items()},
        )

        if plot:
            fig, ax = plt.subplots()
            wedges, texts, autotexts = ax.pie(
                g.values, labels=g.index, autopct="%1.1f%%",
                colors=["#ff9999", "#66b3ff"], startangle=90,
                explode=(0.03,) * len(g), textprops={"fontsize": 12},
            )
            ax.set(title="Q4: Customer Gender Distribution")
            result.figure_path = _save_figure(fig, "q04_gender_distribution.png", self.figures_dir)

        return result

    # -- Q5: Income by Region ----------------------------------------------

    def income_by_region(self, plot: bool = True) -> AnalysisResult:
        """Annual income distribution across dealer regions."""
        grouped = self.df.groupby("Dealer_Region")["Annual_Income"].describe()
        order = self.df.groupby("Dealer_Region")["Annual_Income"].median().sort_values().index

        result = AnalysisResult(
            title="Q5: Annual Income by Region",
            description="Income distribution statistics per dealer region",
            table=grouped,
            summary={"regions": f"{len(grouped)}", "highest_median_region": f"{order[-1]}"},
        )

        if plot:
            fig, ax = plt.subplots()
            bp = ax.boxplot(
                [self.df[self.df["Dealer_Region"] == r]["Annual_Income"].dropna() / 1000 for r in order],
                labels=order, patch_artist=True, notch=True,
            )
            for patch, color in zip(bp["boxes"], plt.cm.viridis(np.linspace(0.2, 0.8, len(order)))):
                patch.set_facecolor(color)
            ax.set(title="Q5: Annual Income by Dealer Region", xlabel="Region",
                   ylabel="Annual Income (thousands $)")
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha="right")
            result.figure_path = _save_figure(fig, "q05_income_by_region.png", self.figures_dir)

        return result

    # -- Q6: Income vs Price (Correlation + Regression) --------------------

    def income_vs_price(self, plot: bool = True) -> AnalysisResult:
        """Pearson correlation + OLS linear regression."""
        sub = self.df[["Annual_Income", "Price"]].dropna()
        r, p = pearsonr(sub["Annual_Income"], sub["Price"])
        slope, intercept, r_val, p_val, std_err = linregress(sub["Annual_Income"], sub["Price"])

        result = AnalysisResult(
            title="Q6: Income vs Price",
            description="Correlation and linear regression analysis",
            summary={
                "pearson_r": f"{r:.4f}",
                "p_value": f"{p:.2e}",
                "r_squared": f"{r_val ** 2:.4f}",
                "regression": f"Price = {slope:.6f} × Income + {intercept:.2f}",
                "std_error": f"{std_err:.4f}",
                "sample_size": f"{len(sub)}",
            },
            raw={"sub": sub, "slope": slope, "intercept": intercept,
                 "r_val": r_val, "p_val": p_val, "std_err": std_err},
        )

        if plot:
            fig, ax = plt.subplots()
            ax.scatter(sub["Annual_Income"] / 1000, sub["Price"] / 1000,
                       alpha=0.3, s=5, color="steelblue", label="Data")
            x_line = np.linspace(sub["Annual_Income"].min(), sub["Annual_Income"].max(), 200)
            ax.plot(x_line / 1000, (slope * x_line + intercept) / 1000,
                    color="crimson", lw=2, label=f"OLS fit (R² = {r_val ** 2:.3f})")
            ax.set(xlabel="Annual Income (thousands $)", ylabel="Price (thousands $)",
                   title="Q6: Annual Income vs Car Price")
            ax.legend()
            result.figure_path = _save_figure(fig, "q06_income_vs_price.png", self.figures_dir)

        return result

    # -- Q7: Avg Price by Company ------------------------------------------

    def avg_price_by_company(self, min_count: int = 5, top_n: int = 15,
                             plot: bool = True) -> AnalysisResult:
        """Average car price by manufacturer (filtering by min sales)."""
        grouped = self.df.groupby("Company")["Price"].agg(["mean", "count", "std"])
        grouped["mean"] = grouped["mean"].round(2)
        grouped = grouped[grouped["count"] >= min_count].sort_values("mean", ascending=False)
        top = grouped.head(top_n)

        result = AnalysisResult(
            title="Q7: Average Price by Company",
            description=f"Top {top_n} manufacturers by avg price (min {min_count} sales)",
            table=top,
            summary={
                "top_company": f"{top.index[0]} (${top['mean'].iloc[0]:,.2f})",
                "bottom_company": f"{top.index[-1]} (${top['mean'].iloc[-1]:,.2f})",
            },
        )

        if plot:
            fig, ax = plt.subplots()
            plot_data = top.sort_values("mean")
            colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(plot_data)))
            bars = ax.barh(plot_data.index, plot_data["mean"], color=colors, edgecolor="white")
            ax.bar_label(bars, labels=[f"${v:.0f}" for v in plot_data["mean"]],
                         padding=2, fontsize=8)
            ax.set(xlabel="Average Price ($)", title=f"Q7: Top {top_n} Companies by Avg Price")
            ax.invert_yaxis()
            result.figure_path = _save_figure(fig, "q07_avg_price_by_company.png", self.figures_dir)

        return result

    # -- Q8: Price by Body Style -------------------------------------------

    def price_by_body_style(self, plot: bool = True) -> AnalysisResult:
        """Price distribution per body style (box plots)."""
        grouped = self.df.groupby("Body_Style")["Price"].describe()
        order = self.df.groupby("Body_Style")["Price"].median().sort_values().index

        result = AnalysisResult(
            title="Q8: Price by Body Style",
            description="Descriptive statistics of price per body style",
            table=grouped,
        )

        if plot:
            fig, ax = plt.subplots()
            bp = ax.boxplot(
                [self.df[self.df["Body_Style"] == bs]["Price"].dropna() / 1000 for bs in order],
                labels=order, patch_artist=True, notch=True,
            )
            for patch, color in zip(bp["boxes"], plt.cm.tab10(np.linspace(0, 1, len(order)))):
                patch.set_facecolor(color)
            ax.set(title="Q8: Price Distribution by Body Style", xlabel="Body Style",
                   ylabel="Price (thousands $)")
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
            result.figure_path = _save_figure(fig, "q08_price_by_body_style.png", self.figures_dir)

        return result

    # -- Q9: Auto vs Manual Price (t-test) ---------------------------------

    def auto_vs_manual_price(self, plot: bool = True) -> AnalysisResult:
        """Compare average price between automatic and manual transmissions."""
        auto = self.df[self.df["Transmission"].str.lower() == "auto"]["Price"].dropna()
        manual = self.df[self.df["Transmission"].str.lower() == "manual"]["Price"].dropna()
        t_stat, t_p = stats.ttest_ind(auto, manual, equal_var=False)

        result = AnalysisResult(
            title="Q9: Auto vs Manual Price",
            description="Welch's t-test comparing automatic vs manual transmission prices",
            summary={
                "auto_mean": f"${auto.mean():,.2f}",
                "manual_mean": f"${manual.mean():,.2f}",
                "difference": f"${auto.mean() - manual.mean():,.2f}",
                "t_statistic": f"{t_stat:.4f}",
                "p_value": f"{t_p:.2e}",
                "significant": "Yes" if t_p < 0.05 else "No (p ≥ 0.05)",
                "auto_count": f"{len(auto)}",
                "manual_count": f"{len(manual)}",
            },
        )

        if plot:
            fig, ax = plt.subplots()
            labels = ["Automatic", "Manual"]
            means = [auto.mean(), manual.mean()]
            sems = [auto.std() / np.sqrt(len(auto)), manual.std() / np.sqrt(len(manual))]
            bars = ax.bar(labels, means, yerr=sems, capsize=8, color=["#66b3ff", "#ff9999"],
                          edgecolor="black", width=0.5)
            ax.bar_label(bars, labels=[f"${v:,.0f}" for v in means], padding=4)
            ax.set(ylabel="Average Price ($)", title="Q9: Automatic vs Manual — Average Price")
            result.figure_path = _save_figure(fig, "q09_auto_vs_manual_price.png", self.figures_dir)

        return result

    # -- Q10: Popular Colors -----------------------------------------------

    def popular_colors(self, top_n: int = 8, plot: bool = True) -> AnalysisResult:
        """Most common car colors."""
        colors = self.df["Color"].value_counts()
        total = colors.sum()
        top = colors.head(top_n)

        result = AnalysisResult(
            title="Q10: Most Popular Car Colors",
            description=f"Top {top_n} car colors by frequency",
            summary={k: f"{v} ({v / total * 100:.1f}%)" for k, v in top.items()},
            series=colors,
        )

        if plot:
            fig, ax = plt.subplots()
            other = colors.iloc[top_n:].sum()
            plot_data = top.copy()
            if other > 0:
                plot_data["Other"] = other
            palette = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
                       "#911eb4", "#42d4f4", "#f032e6", "#bfef45"]
            bars = ax.barh(plot_data.index, plot_data.values,
                           color=palette[:len(plot_data)], edgecolor="white")
            ax.bar_label(bars, labels=[f"{v}" for v in plot_data.values], padding=2)
            ax.set(xlabel="Number of Cars", title="Q10: Most Popular Car Colors")
            ax.invert_yaxis()
            result.figure_path = _save_figure(fig, "q10_popular_colors.png", self.figures_dir)

        return result

    # -- Q11: Body Style × Transmission Heatmap ----------------------------

    def body_style_transmission_heatmap(self, plot: bool = True) -> AnalysisResult:
        """Cross-tabulation of body style and transmission type."""
        ct = pd.crosstab(self.df["Body_Style"], self.df["Transmission"],
                         margins=True, margins_name="Total")

        result = AnalysisResult(
            title="Q11: Body Style × Transmission",
            description="Cross-tabulation frequency matrix",
            table=ct,
        )

        if plot:
            fig, ax = plt.subplots()
            ct_plot = pd.crosstab(self.df["Body_Style"], self.df["Transmission"])
            im = ax.imshow(ct_plot.values, cmap="YlOrRd", aspect="auto")
            ax.set_xticks(range(len(ct_plot.columns)))
            ax.set_xticklabels(ct_plot.columns)
            ax.set_yticks(range(len(ct_plot.index)))
            ax.set_yticklabels(ct_plot.index)
            for i in range(len(ct_plot.index)):
                for j in range(len(ct_plot.columns)):
                    ax.text(j, i, ct_plot.values[i, j], ha="center", va="center", fontsize=8)
            ax.set(title="Q11: Body Style vs Transmission Frequency",
                   xlabel="Transmission", ylabel="Body Style")
            plt.colorbar(im, ax=ax, label="Count")
            result.figure_path = _save_figure(fig, "q11_style_transmission_heatmap.png",
                                              self.figures_dir)

        return result

    # -- Q12: Multiple Linear Regression -----------------------------------

    def multiple_regression(self, plot: bool = True) -> AnalysisResult:
        """Predict Price using Income + Transmission + Engine + Body_Style."""
        sub = self.df[["Price", "Annual_Income", "Transmission", "Engine",
                        "Body_Style"]].dropna().copy()
        sub["Transmission"] = (sub["Transmission"].str.lower() == "auto").astype(int)
        sub["Engine_DOHC"] = sub["Engine"].str.lower().str.contains("double").astype(int)
        body_dummies = pd.get_dummies(sub["Body_Style"], prefix="body", drop_first=True).astype(int)

        X = pd.concat([sub[["Annual_Income", "Transmission", "Engine_DOHC"]], body_dummies], axis=1)
        y = sub["Price"]
        X_with_const = np.column_stack([np.ones(len(X)), X.values])
        coeffs = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
        y_pred = X_with_const @ coeffs
        residuals = y - y_pred
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_sq = 1 - ss_res / ss_tot

        coeff_dict = {"Intercept": round(coeffs[0], 2)}
        for name, c in zip(
            ["Annual_Income", "Transmission_Auto", "Engine_DOHC"] + list(X.columns[3:]),
            coeffs[1:]
        ):
            coeff_dict[name] = round(c, 4)

        result = AnalysisResult(
            title="Q12: Multiple Linear Regression",
            description="Price ~ Annual_Income + Transmission + Engine + Body_Style",
            summary={
                "r_squared": f"{r_sq:.4f}",
                "adj_r_squared": f"{1 - (1 - r_sq) * (len(y) - 1) / (len(y) - len(coeffs)):.4f}",
                "sample_size": f"{len(y)}",
                "features": f"{len(coeffs) - 1}",
                **coeff_dict,
            },
            raw={"coefficients": coeffs, "residuals": residuals, "y_pred": y_pred},
        )

        if plot:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            ax1.scatter(y_pred / 1000, y / 1000, alpha=0.3, s=5, color="steelblue")
            ax1.plot([y.min() / 1000, y.max() / 1000],
                     [y.min() / 1000, y.max() / 1000], "r--", lw=2, label="Perfect fit")
            ax1.set(xlabel="Predicted Price (thousands $)", ylabel="Actual Price (thousands $)",
                    title="Predicted vs Actual")
            ax1.legend()
            ax2.hist(residuals / 1000, bins=40, color="steelblue", edgecolor="white")
            ax2.set(xlabel="Residuals (thousands $)", ylabel="Frequency",
                    title="Residual Distribution")
            result.figure_path = _save_figure(fig, "q12_multiple_regression.png", self.figures_dir)

        return result

    # -- Q13: Z-score Outliers ---------------------------------------------

    def outliers_zscore(self, threshold: float = 3.0, plot: bool = True) -> AnalysisResult:
        """Identify price outliers using Z-score method."""
        prices = self.df["Price"].dropna()
        z = np.abs(zscore(prices))
        outliers = prices[z > threshold]

        result = AnalysisResult(
            title="Q13: Price Outliers (Z-score)",
            description=f"Outliers detected with |z| > {threshold}",
            summary={
                "total_points": f"{len(prices)}",
                "outlier_count": f"{len(outliers)}",
                "outlier_pct": f"{len(outliers) / len(prices) * 100:.2f}%",
                "outlier_min": f"${outliers.min():,.2f}" if len(outliers) > 0 else "N/A",
                "outlier_max": f"${outliers.max():,.2f}" if len(outliers) > 0 else "N/A",
            },
            raw={"prices": prices, "z_scores": z, "outliers": outliers},
        )

        if plot:
            fig, ax = plt.subplots()
            ax.scatter(range(len(prices)), prices / 1000, alpha=0.3, s=4,
                       color="steelblue", label="Normal")
            if len(outliers) > 0:
                ax.scatter(np.where(z > threshold)[0], outliers / 1000, alpha=0.7, s=15,
                           color="crimson", label=f"Outliers (n={len(outliers)})")
            ax.axhline(prices.mean() / 1000, color="green", ls="--", lw=1,
                       label=f"Mean = ${prices.mean() / 1000:,.1f}k")
            ax.set(xlabel="Index", ylabel="Price (thousands $)",
                   title="Q13: Car Price Outliers (Z-score)")
            ax.legend()
            result.figure_path = _save_figure(fig, "q13_outliers_zscore.png", self.figures_dir)

        return result

    # -- Q14: Avg Price by Dealer ------------------------------------------

    def avg_price_by_dealer(self, min_sales: int = 3, top_n: int = 15,
                            plot: bool = True) -> AnalysisResult:
        """Top dealers by average car price."""
        grouped = self.df.groupby("Dealer_Name")["Price"].agg(["mean", "count"])
        grouped["mean"] = grouped["mean"].round(2)
        grouped = grouped[grouped["count"] >= min_sales].sort_values("mean", ascending=False)
        top = grouped.head(top_n)

        result = AnalysisResult(
            title="Q14: Top Dealers by Avg Price",
            description=f"Top {top_n} dealers with ≥ {min_sales} sales",
            table=top,
        )

        if plot:
            fig, ax = plt.subplots()
            plot_data = top.sort_values("mean")
            bars = ax.barh(plot_data.index, plot_data["mean"],
                           color=plt.cm.Blues(np.linspace(0.3, 0.9, len(plot_data))),
                           edgecolor="white")
            ax.bar_label(bars, labels=[f"${v:.0f}" for v in plot_data["mean"]],
                         padding=2, fontsize=8)
            ax.set(xlabel="Average Price ($)", title=f"Q14: Top {top_n} Dealers by Avg Price")
            ax.invert_yaxis()
            result.figure_path = _save_figure(fig, "q14_avg_price_by_dealer.png", self.figures_dir)

        return result

    # -- Q15: Normality Test -----------------------------------------------

    def normality_test(self, plot: bool = True) -> AnalysisResult:
        """Shapiro-Wilk test + Q-Q plot + density comparison."""
        prices = self.df["Price"].dropna()
        sample = prices.sample(min(5000, len(prices)), random_state=42)
        stat, p_val = shapiro(sample)
        skew = prices.skew()
        kurt = prices.kurtosis()
        is_normal = p_val > 0.05

        result = AnalysisResult(
            title="Q15: Normality Test for Price",
            description="Shapiro-Wilk test with Q-Q plot and density comparison",
            summary={
                "shapiro_statistic": f"{stat:.6f}",
                "p_value": f"{p_val:.2e}",
                "skewness": f"{skew:.4f}",
                "kurtosis": f"{kurt:.4f}",
                "is_normal": f"{'Yes' if is_normal else 'No'} (α = 0.05)",
                "interpretation": "Data IS normally distributed" if is_normal
                                  else "Data is NOT normally distributed",
                "note": "Large samples often reject normality even for minor deviations",
            },
            raw={"sample": sample},
        )

        if plot:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            stats.probplot(sample, dist="norm", plot=ax1)
            ax1.set(title="Q15: Q-Q Plot for Price", ylabel="Ordered Values ($)")
            ax2.hist(prices / 1000, bins=50, color="steelblue", edgecolor="white",
                     density=True, alpha=0.7)
            mu, sigma = prices.mean(), prices.std()
            x = np.linspace(prices.min(), prices.max(), 500)
            ax2.plot(x / 1000, norm.pdf(x, mu, sigma), color="crimson", lw=2,
                     label="Normal fit")
            ax2.set(xlabel="Price (thousands $)", ylabel="Density",
                    title="Price Density vs Normal")
            ax2.legend()
            result.figure_path = _save_figure(fig, "q15_normality_test.png", self.figures_dir)

        return result

    # -----------------------------------------------------------------------
    # Aggregated pipeline
    # -----------------------------------------------------------------------

    def run_all(self, plot: bool = True) -> dict[str, AnalysisResult]:
        """Execute the full 15-question analysis pipeline.

        Returns:
            Mapping of question keys to their AnalysisResult.
        """
        results = {}

        results["q01"] = self.price_distribution(plot=plot)
        results["q02"] = self.monthly_sales_trend(plot=plot)
        results["q03"] = self.sales_by_region(plot=plot)
        results["q04"] = self.gender_distribution(plot=plot)
        results["q05"] = self.income_by_region(plot=plot)
        results["q06"] = self.income_vs_price(plot=plot)
        results["q07"] = self.avg_price_by_company(plot=plot)
        results["q08"] = self.price_by_body_style(plot=plot)
        results["q09"] = self.auto_vs_manual_price(plot=plot)
        results["q10"] = self.popular_colors(plot=plot)
        results["q11"] = self.body_style_transmission_heatmap(plot=plot)
        results["q12"] = self.multiple_regression(plot=plot)
        results["q13"] = self.outliers_zscore(plot=plot)
        results["q14"] = self.avg_price_by_dealer(plot=plot)
        results["q15"] = self.normality_test(plot=plot)

        return results

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _log(msg: str) -> None:
        print(f"  [ETL] {msg}")
