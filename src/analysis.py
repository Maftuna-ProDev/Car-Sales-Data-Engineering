"""
Car Sales Data Analysis — Big Data & Business Analytics (Unit 10)
=================================================================
Covers criteria: A.D1, B.D2, B.M2, B.M3, B.P2, B.P3, C.D3, C.M4, C.M5, C.M6, C.P4, C.P5, C.P6

Student: Baxtiyorov Abdumalik Baxrom o'g'li
Group: 25-102 | Student ID: 250076
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr, linregress, shapiro, norm, zscore

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH = "data/Car sales.csv"
OUTPUT_DIR = "outputs"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
CLEANED_PATH = os.path.join(OUTPUT_DIR, "cleaned_data.csv")

os.makedirs(FIGURES_DIR, exist_ok=True)

# Global matplotlib settings
plt.rcParams.update({
    "figure.dpi": 150,
    "figure.figsize": (10, 6),
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
})


# ========================= DATA LOADING & CLEANING =========================

def load_and_clean_data() -> pd.DataFrame:
    """Load CSV, fix encoding, clean column names, handle missing & outliers (C.P4, C.M4)."""
    df = pd.read_csv(DATA_PATH)

    # Clean column names
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[($)]", "", regex=True)
        .str.replace("Dealer_No_", "Dealer_No")
    )
    df.rename(columns={"Price_": "Price"}, inplace=True)

    # Parse date
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
    df.dropna(subset=["Date"], inplace=True)

    # Clean encoding artifacts (DoubleÂ -> Double)
    df["Engine"] = df["Engine"].str.replace("Â", "", regex=False)
    df["Company"] = df["Company"].str.strip()
    df["Model"] = df["Model"].str.strip()

    # Convert numeric
    df["Annual_Income"] = pd.to_numeric(df["Annual_Income"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    # Drop rows with missing critical values
    before = len(df)
    df.dropna(subset=["Annual_Income", "Price"], inplace=True)
    print(f"  Dropped {before - len(df)} rows with missing income/price")

    # Fix Gender — map to consistent labels
    df["Gender"] = df["Gender"].str.strip().str.title()

    # Remove duplicates
    dups = df.duplicated(subset=["Car_id"]).sum()
    df.drop_duplicates(subset=["Car_id"], inplace=True)
    print(f"  Removed {dups} duplicate Car_id rows")

    # --- Outlier capping (IQR method) for Price and Income ---
    for col in ["Price", "Annual_Income"]:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_before = (df[col] < lower).sum() + (df[col] > upper).sum()
        # Cap, don't drop: preserve data for analysis
        df[col] = df[col].clip(lower, upper)
        print(f"  Capped {n_before} outliers in {col}")

    print(f"  Final dataset: {len(df)} rows, {len(df.columns)} columns\n")
    return df


# ========================= DESCRIPTIVE STATISTICS =========================

def descriptive_statistics(df: pd.DataFrame):
    """Routine & non-routine central tendency + dispersion (B.P2, B.M2)."""
    print("=" * 65)
    print("SECTION: Descriptive Statistics (Central Tendency & Dispersion)")
    print("=" * 65)

    num_cols = ["Price", "Annual_Income"]

    for col in num_cols:
        data = df[col].dropna()
        print(f"\n  --- {col} ---")
        print(f"    Mean:           {data.mean():>12,.2f}")
        print(f"    Median:         {data.median():>12,.2f}")
        print(f"    Mode:           {data.mode().iloc[0]:>12,.2f}")
        print(f"    Min:            {data.min():>12,.2f}")
        print(f"    Max:            {data.max():>12,.2f}")
        print(f"    Range:          {(data.max() - data.min()):>12,.2f}")
        print(f"    Variance:       {data.var():>12,.2f}")
        print(f"    Std Dev:        {data.std():>12,.2f}")
        print(f"    Skewness:       {data.skew():>12,.4f}  (non-routine)")
        print(f"    Kurtosis:       {data.kurtosis():>12,.4f}  (non-routine)")
        print(f"    P25 (Q1):       {data.quantile(0.25):>12,.2f}")
        print(f"    P50 (Q2):       {data.quantile(0.50):>12,.2f}")
        print(f"    P75 (Q3):       {data.quantile(0.75):>12,.2f}")
        print(f"    IQR:            {(data.quantile(0.75) - data.quantile(0.25)):>12,.2f}")


# ========================= 15 QUESTIONS =========================

def question_01_price_distribution(df: pd.DataFrame):
    """Q1: What is the distribution of car prices?"""
    print("\n--- Q1: Price Distribution ---")
    prices = df["Price"].dropna()
    print(f"  Mean=${prices.mean():.2f}, Median=${prices.median():.2f}, "
          f"Std=${prices.std():.2f}")
    fig, ax = plt.subplots()
    ax.hist(prices, bins=50, color="steelblue", edgecolor="white", alpha=0.8, density=True)
    kde_x = np.linspace(prices.min(), prices.max(), 500)
    kde = stats.gaussian_kde(prices)
    ax.plot(kde_x, kde(kde_x), color="crimson", lw=2, label="KDE")
    ax.axvline(prices.mean(), color="darkorange", ls="--", lw=1.5, label=f"Mean=${prices.mean():.0f}")
    ax.axvline(prices.median(), color="green", ls=":", lw=1.5, label=f"Median=${prices.median():.0f}")
    ax.set(xlabel="Price ($)", ylabel="Density", title="Q1: Distribution of Car Prices")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "q01_price_distribution.png"))
    plt.close(fig)
    print("  -> Saved q01_price_distribution.png")


def question_02_monthly_sales_trend(df: pd.DataFrame):
    """Q2: What is the monthly sales trend over time?"""
    print("\n--- Q2: Monthly Sales Trend ---")
    monthly = df.set_index("Date").resample("ME").size()
    print(f"  Highest month: {monthly.idxmax().strftime('%b %Y')} ({monthly.max()} sales)")
    print(f"  Lowest month:  {monthly.idxmin().strftime('%b %Y')} ({monthly.min()} sales)")
    fig, ax = plt.subplots()
    ax.plot(monthly.index, monthly.values, marker="o", color="steelblue", lw=1.5)
    ax.fill_between(monthly.index, monthly.values, alpha=0.15, color="steelblue")
    ax.set(xlabel="Date", ylabel="Number of Sales", title="Q2: Monthly Car Sales Trend")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "q02_monthly_sales_trend.png"))
    plt.close(fig)
    print("  -> Saved q02_monthly_sales_trend.png")


def question_03_sales_by_region(df: pd.DataFrame):
    """Q3: Which dealer regions sell the most cars?"""
    print("\n--- Q3: Sales by Dealer Region ---")
    region_counts = df["Dealer_Region"].value_counts()
    print(region_counts.to_string())
    fig, ax = plt.subplots()
    colors = plt.cm.Set2(np.linspace(0, 1, len(region_counts)))
    bars = ax.barh(region_counts.index, region_counts.values, color=colors)
    ax.bar_label(bars, labels=[f"{v}" for v in region_counts.values], padding=2)
    ax.set(xlabel="Number of Cars Sold", title="Q3: Sales by Dealer Region")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "q03_sales_by_region.png"))
    plt.close(fig)
    print("  -> Saved q03_sales_by_region.png")


def question_04_gender_distribution(df: pd.DataFrame):
    """Q4: What is the gender split of customers?"""
    print("\n--- Q4: Customer Gender Distribution ---")
    g = df["Gender"].value_counts()
    total = g.sum()
    for k, v in g.items():
        print(f"  {k}: {v} ({v / total * 100:.1f}%)")
    fig, ax = plt.subplots()
    colors = ["#ff9999", "#66b3ff"]
    wedges, texts, autotexts = ax.pie(
        g.values, labels=g.index, autopct="%1.1f%%",
        colors=colors, startangle=90, explode=(0.03,) * len(g),
        textprops={"fontsize": 12},
    )
    ax.set(title="Q4: Customer Gender Distribution")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "q04_gender_distribution.png"))
    plt.close(fig)
    print("  -> Saved q04_gender_distribution.png")


def question_05_income_by_region(df: pd.DataFrame):
    """Q5: How does annual income vary across dealer regions?"""
    print("\n--- Q5: Annual Income by Region ---")
    region_income = df.groupby("Dealer_Region")["Annual_Income"].describe()
    print(region_income.to_string())
    fig, ax = plt.subplots()
    order = df.groupby("Dealer_Region")["Annual_Income"].median().sort_values().index
    bp = ax.boxplot(
        [df[df["Dealer_Region"] == r]["Annual_Income"].dropna() / 1000 for r in order],
        labels=order, patch_artist=True, notch=True,
    )
    for patch, color in zip(bp["boxes"], plt.cm.viridis(np.linspace(0.2, 0.8, len(order)))):
        patch.set_facecolor(color)
    ax.set(title="Q5: Annual Income by Dealer Region", xlabel="Region", ylabel="Annual Income (thousands $)")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "q05_income_by_region.png"))
    plt.close(fig)
    print("  -> Saved q05_income_by_region.png")


def question_06_income_vs_price(df: pd.DataFrame):
    """Q6: Relationship between annual income and car price (correlation + regression)."""
    print("\n--- Q6: Income vs Price (Correlation & Regression) ---")
    sub = df[["Annual_Income", "Price"]].dropna()
    r, p = pearsonr(sub["Annual_Income"], sub["Price"])
    slope, intercept, r_val, p_val, std_err = linregress(sub["Annual_Income"], sub["Price"])
    print(f"  Pearson r = {r:.4f}  (p = {p:.2e})")
    print(f"  R-squared = {r_val ** 2:.4f}")
    print(f"  Regression: Price = {slope:.6f} * Income + {intercept:.2f}")
    fig, ax = plt.subplots()
    ax.scatter(sub["Annual_Income"] / 1000, sub["Price"] / 1000,
               alpha=0.3, s=5, color="steelblue", label="Data")
    x_line = np.linspace(sub["Annual_Income"].min(), sub["Annual_Income"].max(), 200)
    ax.plot(x_line / 1000, (slope * x_line + intercept) / 1000,
            color="crimson", lw=2, label=f"OLS fit (R²={r_val ** 2:.3f})")
    ax.set(xlabel="Annual Income (thousands $)", ylabel="Price (thousands $)",
           title="Q6: Annual Income vs Car Price")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "q06_income_vs_price.png"))
    plt.close(fig)
    print("  -> Saved q06_income_vs_price.png")


def question_07_avg_price_by_company(df: pd.DataFrame):
    """Q7: Which car companies have the highest average prices?"""
    print("\n--- Q7: Average Price by Company (Top 15) ---")
    comp_price = df.groupby("Company")["Price"].agg(["mean", "count", "std"]).sort_values("mean", ascending=False)
    comp_price["mean"] = comp_price["mean"].round(2)
    comp_price = comp_price[comp_price["count"] >= 5]
    top15 = comp_price.head(15)
    print(top15.to_string())
    fig, ax = plt.subplots()
    top15 = top15.sort_values("mean")
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(top15)))
    bars = ax.barh(top15.index, top15["mean"], color=colors, edgecolor="white")
    ax.bar_label(bars, labels=[f"${v:.0f}" for v in top15["mean"]], padding=2, fontsize=8)
    ax.set(xlabel="Average Price ($)", title="Q7: Top 15 Companies by Average Car Price")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "q07_avg_price_by_company.png"))
    plt.close(fig)
    print("  -> Saved q07_avg_price_by_company.png")


def question_08_price_by_body_style(df: pd.DataFrame):
    """Q8: How does price differ by body style?"""
    print("\n--- Q8: Price by Body Style ---")
    bs = df.groupby("Body_Style")["Price"].describe()
    print(bs.to_string())
    fig, ax = plt.subplots()
    order = df.groupby("Body_Style")["Price"].median().sort_values().index
    bp = ax.boxplot(
        [df[df["Body_Style"] == bs]["Price"].dropna() / 1000 for bs in order],
        labels=order, patch_artist=True, notch=True,
    )
    for patch, color in zip(bp["boxes"], plt.cm.tab10(np.linspace(0, 1, len(order)))):
        patch.set_facecolor(color)
    ax.set(title="Q8: Price Distribution by Body Style", xlabel="Body Style", ylabel="Price (thousands $)")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "q08_price_by_body_style.png"))
    plt.close(fig)
    print("  -> Saved q08_price_by_body_style.png")


def question_09_auto_vs_manual_price(df: pd.DataFrame):
    """Q9: Do automatic cars cost more than manual?"""
    print("\n--- Q9: Automatic vs Manual Transmission Price Comparison ---")
    trans = df.groupby("Transmission")["Price"].describe()
    print(trans.to_string())
    auto = df[df["Transmission"].str.lower() == "auto"]["Price"].dropna()
    manual = df[df["Transmission"].str.lower() == "manual"]["Price"].dropna()
    t_stat, t_p = stats.ttest_ind(auto, manual, equal_var=False)
    print(f"\n  Welch t-test: t = {t_stat:.4f}, p = {t_p:.2e}")
    fig, ax = plt.subplots()
    labels = ["Automatic", "Manual"]
    means = [auto.mean(), manual.mean()]
    sems = [auto.std() / np.sqrt(len(auto)), manual.std() / np.sqrt(len(manual))]
    bars = ax.bar(labels, means, yerr=sems, capsize=8, color=["#66b3ff", "#ff9999"],
                  edgecolor="black", width=0.5)
    ax.bar_label(bars, labels=[f"${v:.0f}" for v in means], padding=4)
    ax.set(ylabel="Average Price ($)", title="Q9: Automatic vs Manual — Average Price Comparison")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "q09_auto_vs_manual_price.png"))
    plt.close(fig)
    print("  -> Saved q09_auto_vs_manual_price.png")


def question_10_popular_colors(df: pd.DataFrame):
    """Q10: What are the most popular car colors?"""
    print("\n--- Q10: Most Popular Car Colors ---")
    colors = df["Color"].value_counts()
    total = colors.sum()
    top8 = colors.head(8)
    for k, v in top8.items():
        print(f"  {k:15s}: {v:>6d}  ({v / total * 100:.1f}%)")
    fig, ax = plt.subplots()
    other_count = colors.iloc[8:].sum()
    plot_data = top8.copy()
    if other_count > 0:
        plot_data["Other"] = other_count
    bar_colors = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
                   "#911eb4", "#42d4f4", "#f032e6", "#bfef45"]
    bars = ax.barh(plot_data.index, plot_data.values, color=bar_colors[:len(plot_data)], edgecolor="white")
    ax.bar_label(bars, labels=[f"{v}" for v in plot_data.values], padding=2)
    ax.set(xlabel="Number of Cars", title="Q10: Most Popular Car Colors")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "q10_popular_colors.png"))
    plt.close(fig)
    print("  -> Saved q10_popular_colors.png")


def question_11_body_style_transmission_heatmap(df: pd.DataFrame):
    """Q11: Which body style x transmission combos are most common?"""
    print("\n--- Q11: Body Style vs Transmission Cross-Tabulation ---")
    ct = pd.crosstab(df["Body_Style"], df["Transmission"], margins=True, margins_name="Total")
    print(ct.to_string())
    fig, ax = plt.subplots()
    ct_plot = pd.crosstab(df["Body_Style"], df["Transmission"])
    im = ax.imshow(ct_plot.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(ct_plot.columns)))
    ax.set_xticklabels(ct_plot.columns)
    ax.set_yticks(range(len(ct_plot.index)))
    ax.set_yticklabels(ct_plot.index)
    for i in range(len(ct_plot.index)):
        for j in range(len(ct_plot.columns)):
            ax.text(j, i, ct_plot.values[i, j], ha="center", va="center", fontsize=8)
    ax.set(title="Q11: Body Style vs Transmission Frequency", xlabel="Transmission", ylabel="Body Style")
    plt.colorbar(im, ax=ax, label="Count")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "q11_style_transmission_heatmap.png"))
    plt.close(fig)
    print("  -> Saved q11_style_transmission_heatmap.png")


def question_12_multiple_regression(df: pd.DataFrame):
    """Q12: Predict price from income + transmission + body style (multiple regression)."""
    print("\n--- Q12: Multiple Linear Regression (Price ~ Income + Transmission + Body_Engine) ---")
    sub = df[["Price", "Annual_Income", "Transmission", "Engine", "Body_Style"]].dropna().copy()
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
    print(f"  R-squared: {r_sq:.4f}")
    print(f"  Intercept: {coeffs[0]:.2f}")
    for name, c in zip(["Annual_Income", "Transmission_Auto", "Engine_DOHC"] + list(X.columns[3:]), coeffs[1:]):
        print(f"    {name}: {c:.4f}")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.scatter(y_pred / 1000, y / 1000, alpha=0.3, s=5, color="steelblue")
    ax1.plot([y.min() / 1000, y.max() / 1000], [y.min() / 1000, y.max() / 1000],
             "r--", lw=2, label="Perfect fit")
    ax1.set(xlabel="Predicted Price (thousands $)", ylabel="Actual Price (thousands $)",
            title="Q12: Predicted vs Actual Price")
    ax1.legend()
    ax2.hist(residuals / 1000, bins=40, color="steelblue", edgecolor="white")
    ax2.set(xlabel="Residuals (thousands $)", ylabel="Frequency", title="Q12: Residual Distribution")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "q12_multiple_regression.png"))
    plt.close(fig)
    print("  -> Saved q12_multiple_regression.png")


def question_13_outliers_zscore(df: pd.DataFrame):
    """Q13: Identify price outliers using Z-score method."""
    print("\n--- Q13: Price Outliers (Z-score Method) ---")
    prices = df["Price"].dropna()
    z = np.abs(zscore(prices))
    outliers = prices[z > 3]
    print(f"  Total data points: {len(prices)}")
    print(f"  Outliers (|z| > 3): {len(outliers)} ({len(outliers) / len(prices) * 100:.2f}%)")
    if len(outliers) > 0:
        print(f"  Outlier price range: ${outliers.min():.2f} - ${outliers.max():.2f}")
        print(f"  Top 5 highest outliers: {outliers.nlargest(5).to_list()}")
    fig, ax = plt.subplots()
    ax.scatter(range(len(prices)), prices / 1000, alpha=0.3, s=4, color="steelblue", label="Normal")
    ax.scatter(np.where(z > 3)[0], outliers / 1000, alpha=0.7, s=15, color="crimson", label=f"Outliers (n={len(outliers)})")
    ax.axhline(prices.mean() / 1000, color="green", ls="--", lw=1, label=f"Mean=${prices.mean() / 1000:.1f}k")
    ax.set(xlabel="Index", ylabel="Price (thousands $)", title="Q13: Car Price Outliers (Z-score)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "q13_outliers_zscore.png"))
    plt.close(fig)
    print("  -> Saved q13_outliers_zscore.png")


def question_14_avg_price_by_dealer(df: pd.DataFrame):
    """Q14: Which dealers sell the most expensive cars on average?"""
    print("\n--- Q14: Top 15 Dealers by Average Car Price ---")
    dealer_price = df.groupby("Dealer_Name")["Price"].agg(["mean", "count"]).sort_values("mean", ascending=False)
    dealer_price["mean"] = dealer_price["mean"].round(2)
    dealer_price = dealer_price[dealer_price["count"] >= 3]
    top15 = dealer_price.head(15)
    print(top15.to_string())
    fig, ax = plt.subplots()
    top15_sorted = top15.sort_values("mean")
    bars = ax.barh(top15_sorted.index, top15_sorted["mean"], color=plt.cm.Blues(np.linspace(0.3, 0.9, len(top15_sorted))),
                   edgecolor="white")
    ax.bar_label(bars, labels=[f"${v:.0f}" for v in top15_sorted["mean"]], padding=2, fontsize=8)
    ax.set(xlabel="Average Price ($)", title="Q14: Top 15 Dealers by Average Car Price")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "q14_avg_price_by_dealer.png"))
    plt.close(fig)
    print("  -> Saved q14_avg_price_by_dealer.png")


def question_15_normality_test(df: pd.DataFrame):
    """Q15: Is the price data normally distributed? (Shapiro-Wilk + Q-Q plot)"""
    print("\n--- Q15: Normality Test for Price Distribution ---")
    prices = df["Price"].dropna()
    sample = prices.sample(min(5000, len(prices)), random_state=42)
    stat, p_val = shapiro(sample)
    print(f"  Shapiro-Wilk: statistic = {stat:.6f}, p-value = {p_val:.2e}")
    skew = prices.skew()
    kurt = prices.kurtosis()
    print(f"  Skewness: {skew:.4f}  |  Kurtosis: {kurt:.4f}")
    if p_val > 0.05:
        print("  -> Data IS normally distributed (p > 0.05)")
    else:
        print("  -> Data is NOT normally distributed (p < 0.05)")
    print("  (Large sample sizes often reject normality even for minor deviations)")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    stats.probplot(prices.sample(min(5000, len(prices)), random_state=42), dist="norm", plot=ax1)
    ax1.set(title="Q15: Q-Q Plot for Price", ylabel="Ordered Values ($)")
    ax2.hist(prices / 1000, bins=50, color="steelblue", edgecolor="white", density=True, alpha=0.7)
    mu, sigma = prices.mean(), prices.std()
    x = np.linspace(prices.min(), prices.max(), 500)
    ax2.plot(x / 1000, norm.pdf(x, mu, sigma), color="crimson", lw=2, label="Normal fit")
    ax2.set(xlabel="Price (thousands $)", ylabel="Density", title="Q15: Price Density vs Normal")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "q15_normality_test.png"))
    plt.close(fig)
    print("  -> Saved q15_normality_test.png")


# ========================= FREQUENCY & PERCENTAGE =========================

def frequency_analysis(df: pd.DataFrame):
    """Frequency and percentage tables for categorical data (Task B requirement)."""
    print("\n" + "=" * 65)
    print("SECTION: Frequency & Percentage Analysis")
    print("=" * 65)
    cat_cols = ["Gender", "Transmission", "Color", "Body_Style", "Engine", "Dealer_Region"]
    for col in cat_cols:
        print(f"\n  --- {col} ---")
        freq = df[col].value_counts()
        total = freq.sum()
        for k, v in freq.items():
            print(f"    {str(k):25s}: {v:>6d}  ({v / total * 100:5.1f}%)")


# ========================= CONCLUSIONS & RECOMMENDATIONS =========================

def conclusions_and_recommendations(df: pd.DataFrame):
    """Key findings and business recommendations (C.P6, C.M6, C.D3)."""
    print("\n" + "=" * 65)
    print("CONCLUSIONS & RECOMMENDATIONS")
    print("=" * 65)
    avg_price = df["Price"].mean()
    median_price = df["Price"].median()
    top_company = df.groupby("Company")["Price"].mean().idxmax()
    top_region = df["Dealer_Region"].value_counts().idxmax()
    total_sales = len(df)
    total_revenue = df["Price"].sum()

    print(f"""
  1. Data Overview:
     - Total car sales analyzed: {total_sales:,}
     - Total revenue generated: ${total_revenue:,.2f}
     - Average car price: ${avg_price:,.2f} (Median: ${median_price:,.2f})

  2. Key Findings:
     - The company with highest average price is: {top_company}
     - Top sales region: {top_region}
     - Most popular body styles are SUV, Hatchback, and Sedan
     - Automatic transmissions dominate the market
     - Pale White is the most common car color

  3. Recommendations for the Sales Organization:
     a) INVENTORY: Stock more SUVs and Hatchbacks (highest demand sectors)
     b) PRICING: Use income-based pricing tiers to maximise revenue
     c) MARKETING: Target regions with highest sales (Austin, Scottsdale) for campaigns
     d) BENCHMARKING: Monitor top-performing dealers and replicate their strategies
     e) DATA COLLECTION: Implement real-time data capture for predictive analytics

  4. Performance Improvement:
     - Descriptive analytics shows current sales patterns
     - Diagnostic analytics can identify why certain regions underperform
     - Predictive models (regression) can forecast demand and revenue
     - Prescriptive analytics can recommend optimal inventory allocation

  5. Innovation Trigger:
     - Use customer income data to personalise financing options
     - Implement dynamic pricing based on demand by body style and region
""")


# ========================= MAIN ORCHESTRATOR =========================

def run_analysis():
    """Run the full analysis pipeline and save all outputs (C.D3)."""
    print("=" * 65)
    print("CAR SALES DATA ANALYSIS — BIG DATA & BUSINESS ANALYTICS")
    print("Student: Baxtiyorov Abdumalik | Group: 25-102 | ID: 250076")
    print("=" * 65)

    # 1. Load & Clean
    print("\n[1/5] Loading and cleaning data...")
    df = load_and_clean_data()
    df.to_csv(CLEANED_PATH, index=False)
    print(f"  Cleaned data saved to {CLEANED_PATH}")

    # 2. Descriptive Statistics
    print("\n[2/5] Computing descriptive statistics...")
    descriptive_statistics(df)

    # 3. Frequency Analysis
    print("\n[3/5] Frequency & percentage analysis...")
    frequency_analysis(df)

    # 4. All 15 Questions
    print("\n[4/5] Answering 15 business questions...")
    question_01_price_distribution(df)
    question_02_monthly_sales_trend(df)
    question_03_sales_by_region(df)
    question_04_gender_distribution(df)
    question_05_income_by_region(df)
    question_06_income_vs_price(df)
    question_07_avg_price_by_company(df)
    question_08_price_by_body_style(df)
    question_09_auto_vs_manual_price(df)
    question_10_popular_colors(df)
    question_11_body_style_transmission_heatmap(df)
    question_12_multiple_regression(df)
    question_13_outliers_zscore(df)
    question_14_avg_price_by_dealer(df)
    question_15_normality_test(df)

    # 5. Conclusions
    print("\n[5/5] Conclusions & recommendations...")
    conclusions_and_recommendations(df)

    print("\n" + "=" * 65)
    print(f"Analysis complete. All figures saved to {FIGURES_DIR}/")
    print("=" * 65)


if __name__ == "__main__":
    run_analysis()
