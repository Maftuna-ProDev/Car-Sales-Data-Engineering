"""
Car Sales Analytics — Interactive Web Dashboard
Big Data & Business Analytics (Unit 10)
Abdumalik-ProDev
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats as sp_stats
from scipy.stats import norm, ttest_ind, zscore

from src.analysis import CarSalesAnalyzer

st.set_page_config(
    page_title="Car Sales Analytics",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

P = {
    "blue": "#2563eb",
    "blue_light": "#dbeafe",
    "blue_dark": "#1e40af",
    "purple": "#7c3aed",
    "amber": "#f59e0b",
    "green": "#10b981",
    "red": "#ef4444",
    "gray50": "#f8fafc",
    "gray100": "#f1f5f9",
    "gray200": "#e2e8f0",
    "gray300": "#cbd5e1",
    "gray400": "#94a3b8",
    "gray500": "#64748b",
    "gray600": "#475569",
    "gray700": "#334155",
    "gray800": "#1e293b",
    "gray900": "#0f172a",
}

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    html, body, .stApp, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
    .stApp {{ background: {P["gray50"]}; }}
    .block-container {{ padding-top: 1rem; padding-bottom: 0; max-width: 1200px; }}
    .stAppHeader {{ display: none; }}

    .header {{
        background: linear-gradient(135deg, {P["blue"]}, {P["purple"]});
        border-radius: 12px; padding: 1.5rem 2rem; margin-bottom: 1.5rem; color: white;
    }}
    .header h1 {{ font-size: 1.5rem; font-weight: 700; margin: 0; letter-spacing: -0.3px; }}
    .header p {{ font-size: 0.88rem; opacity: 0.85; margin: 0.3rem 0 0 0; }}

    .q-card {{
        background: white; border-radius: 12px; padding: 1.5rem;
        border: 1px solid {P["gray200"]};
        box-shadow: 0 1px 3px rgba(0,0,0,0.04); margin-bottom: 0.8rem;
    }}
    .q-card:hover {{ box-shadow: 0 4px 14px rgba(0,0,0,0.06); }}
    .q-card .q-num {{
        display: inline-block; background: {P["blue"]}; color: white;
        font-size: 0.75rem; font-weight: 700; padding: 0.15rem 0.55rem;
        border-radius: 6px; letter-spacing: 0.3px; margin-right: 0.5rem;
    }}
    .q-card .q-title {{ font-size: 1.05rem; font-weight: 600; color: {P["gray800"]}; }}
    .q-card .q-text {{ color: {P["gray500"]}; font-size: 0.85rem; margin: 0.4rem 0; line-height: 1.5; }}
    .q-card .q-answer {{ color: {P["green"]}; font-size: 0.88rem; font-weight: 600; }}

    .kpi {{ border: 1px solid {P["gray200"]}; border-radius: 8px; padding: 1rem; background: white; }}
    .kpi .lbl {{ font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.5px; color: {P["gray500"]}; font-weight: 600; }}
    .kpi .val {{ font-size: 1.5rem; font-weight: 700; color: {P["gray900"]}; letter-spacing: -0.5px; }}

    .badge {{ background: {P["blue_light"]}; color: {P["blue_dark"]}; padding: 0.2rem 0.7rem; border-radius: 6px; font-size: 0.72rem; font-weight: 600; display: inline-block; }}

    .insight {{ background: #f0f9ff; border-left: 3px solid {P["blue"]}; border-radius: 6px; padding: 0.7rem 1rem; margin: 0.6rem 0; font-size: 0.85rem; color: {P["gray700"]}; }}

    .nav-q {{ display: flex; align-items: center; gap: 8px; padding: 0.4rem 0.7rem; margin: 1px 0; border-radius: 6px; color: {P["gray400"]} !important; text-decoration: none; font-size: 0.82rem; font-weight: 500; transition: all 0.12s; }}
    .nav-q:hover {{ background: rgba(255,255,255,0.05); color: white !important; }}
    .nav-q.active {{ background: rgba(37,99,235,0.12); color: white !important; font-weight: 600; }}
    .nav-q .q-tag {{ font-size: 0.65rem; opacity: 0.6; min-width: 22px; }}

    [data-testid="stSidebar"] {{ background: {P["gray900"]}; }}
    [data-testid="stSidebar"] .stMarkdown p {{ color: {P["gray400"]}; }}
    [data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,0.06); margin: 0.5rem 0; }}

    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; background: white; border-radius: 8px; padding: 4px; border: 1px solid {P["gray200"]}; }}
    .stTabs [data-baseweb="tab"] {{ border-radius: 6px; padding: 0.3rem 0.85rem; font-size: 0.8rem; font-weight: 500; }}
    .stTabs [aria-selected="true"] {{ background: {P["blue"]} !important; color: white !important; }}

    .stRadio > div {{ gap: 5px; flex-wrap: wrap; }}
    .stRadio label {{ background: white; border: 1px solid {P["gray200"]}; border-radius: 6px; padding: 0.3rem 0.85rem; font-size: 0.8rem; font-weight: 500; transition: all 0.12s; }}
    .stRadio label:hover {{ border-color: {P["blue"]}; color: {P["blue"]}; }}

    div[data-testid="stMetricValue"] {{ font-size: 1.3rem !important; font-weight: 700 !important; color: {P["gray900"]} !important; }}
    div[data-testid="stMetricLabel"] {{ font-size: 0.75rem !important; color: {P["gray500"]} !important; font-weight: 500 !important; }}
    div.stDataFrame {{ border-radius: 8px; overflow: hidden; border: 1px solid {P["gray200"]}; }}
    .stDownloadButton button {{ background: {P["blue"]} !important; color: white !important; border: none !important; border-radius: 8px !important; font-weight: 600 !important; font-size: 0.82rem !important; padding: 0.35rem 1.1rem !important; }}
    .stExpander {{ border: 1px solid {P["gray200"]}; border-radius: 8px; overflow: hidden; }}
    .sidebar-hdr {{ color: white; font-size: 1rem; font-weight: 700; }}
    .sidebar-sub {{ color: {P["blue"]}; font-size: 0.65rem; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }}
    .sidebar-sec {{ color: {P["blue"]}; font-size: 0.65rem; text-transform: uppercase; letter-spacing: 1px; font-weight: 700; }}
    .sidebar-ftr {{ color: {P["gray500"]}; font-size: 0.65rem; text-align: center; padding: 0.4rem 0; }}
    @media (max-width: 768px) {{ .header {{ padding: 1rem; }} .header h1 {{ font-size: 1.2rem; }} }}
</style>
"""

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading 23,906 records...")
def get_analyzer() -> CarSalesAnalyzer:
    a = CarSalesAnalyzer()
    a.load_and_clean()
    a.save_cleaned()
    return a


def usd(v: float) -> str:
    return f"${v:,.2f}"


def fmt(v: float) -> str:
    return f"{v:,.2f}"


def render(fig: plt.Figure) -> None:
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def insight(t: str):
    st.markdown(f'<div class="insight">{t}</div>', unsafe_allow_html=True)


def header(title: str, sub: str):
    st.markdown(f'<div class="header"><h1>{title}</h1><p>{sub}</p></div>', unsafe_allow_html=True)


def qcard(num: str, title: str, question: str, answer: str):
    st.markdown(
        f'<div class="q-card">'
        f'<div><span class="q-num">{num}</span><span class="q-title">{title}</span></div>'
        f'<div class="q-text">{question}</div>'
        f'<div class="q-answer">▸ {answer}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Question definitions
# ---------------------------------------------------------------------------

QUESTIONS = [
    ("Q1", "Price Distribution", "What does the distribution of car prices look like? Are prices normally distributed or skewed?", "Prices are right-skewed — most cars cluster between $20k–$40k, with a long tail toward luxury vehicles."),
    ("Q2", "Monthly Sales Trend", "How does car sales volume fluctuate month-to-month? Is there a seasonal pattern?", "Sales show moderate monthly variation with peaks in Q1 and Q3; no strong seasonal pattern."),
    ("Q3", "Sales by Region", "Which dealer regions sell the most cars? How does geographic demand vary?", "Auckland leads with the highest sales volume; smaller regions like Otago contribute the least."),
    ("Q4", "Gender Distribution", "What is the gender split of car buyers? Is there a significant imbalance?", "The split is nearly even — Male ~51%, Female ~49%, indicating balanced market participation."),
    ("Q5", "Income by Region", "How does annual income vary across dealer regions? Which regions have the highest earners?", "Southland and Otago show the highest median incomes; Auckland has broader income spread."),
    ("Q6", "Income vs Price", "Is there a correlation between a buyer's annual income and the price of the car they purchase?", "Weak positive correlation (R² ≈ 0.07) — income alone is a poor predictor of purchase price."),
    ("Q7", "Avg Price by Company", "Which car companies command the highest average price? Which are the most affordable?", "Land Rover and BMW lead in avg price; Nissan and Toyota are most affordable."),
    ("Q8", "Price by Body Style", "How does price differ across body styles like SUV, sedan, hatchback, etc.?", "SUVs and convertibles have the highest median prices; hatchbacks and sedans are lower."),
    ("Q9", "Auto vs Manual Price", "Do automatic cars cost more than manual cars on average? Is the difference significant?", "Automatic transmission cars are ~$4k more expensive on average; the difference is statistically significant (p < 0.001)."),
    ("Q10", "Popular Colors", "What are the most popular car colours? How concentrated is colour preference?", "White, Black, and Silver dominate — the top 3 account for over 60% of all cars sold."),
    ("Q11", "Body Style × Transmission", "How do body style and transmission type interact? Which combos are most common?", "SUV-Auto and Sedan-Auto are the most frequent combinations; manual is rare in premium body styles."),
    ("Q12", "Multiple Linear Regression", "Can we predict car price using income, transmission, engine, and body style together?", "The model explains ~12% of price variance (R² = 0.12). Annual income is the strongest predictor."),
    ("Q13", "Z-score Outliers", "Are there unusual car prices that deviate significantly from the norm?", "Approximately 1.3% of cars are flagged as outliers — mostly high-end luxury vehicles above $65k."),
    ("Q14", "Avg Price by Dealer", "Which dealers achieve the highest average transaction prices?", "Independent luxury dealers top the list; volume dealers have lower but more consistent pricing."),
    ("Q15", "Normality Test", "Does car price follow a normal distribution according to statistical tests?", "Shapiro-Wilk rejects normality (p < 0.001). Prices are right-skewed with excess kurtosis — not normal."),
]

QUESTION_KEYS = [q[0].lower() for q in QUESTIONS]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar():
    analyzer = get_analyzer()
    df = analyzer.df

    with st.sidebar:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:10px;margin-bottom:2px;">'
            '<span style="font-size:1.5rem;">🚗</span>'
            '<div><div class="sidebar-hdr">Car Sales</div>'
            '<div class="sidebar-sub">Analytics App</div></div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<span class="badge">Abdumalik-ProDev</span>', unsafe_allow_html=True)
        st.markdown("---")

        st.markdown('<div class="sidebar-sec">📊 Dataset</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.metric("Records", f"{len(df):,}")
        c2.metric("Revenue", usd(df["Price"].sum()))
        c1.metric("Avg Price", usd(df["Price"].mean()))
        c2.metric("Companies", df["Company"].nunique())
        c1.metric("Regions", df["Dealer_Region"].nunique())
        c2.metric("Years", f"{df['Date'].min().year}-{df['Date'].max().year}")

        st.markdown("---")
        st.markdown('<div class="sidebar-sec">🧭 Pages</div>', unsafe_allow_html=True)

        params = st.query_params
        current = params.get("page", "home")

        def nav_link(key: str, icon: str, label: str, cls: str = "nav-q"):
            active = "active" if current == key else ""
            st.markdown(
                f'<a href="?page={key}" class="{cls} {active}">{icon} {label}</a>',
                unsafe_allow_html=True,
            )

        nav_link("home", "🏠", "Home")
        nav_link("filter", "🔍", "Filter & Explore")
        nav_link("compare", "⚖️", "Compare Segments")

        st.markdown("---")
        st.markdown('<div class="sidebar-sec">📋 All 15 Questions</div>', unsafe_allow_html=True)

        for num, title, *_ in QUESTIONS:
            k = num.lower()
            active = "active" if current == k else ""
            short = title[:22] + "…" if len(title) > 22 else title
            st.markdown(
                f'<a href="?page={k}" class="nav-q {active}">'
                f'<span class="q-tag">{num}</span> {short}</a>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown('<div class="sidebar-ftr">Built by Abdumalik-ProDev</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def chart_type_picker(key: str) -> str:
    return st.selectbox("Type", ["bar", "line", "area", "pie"], key=key)


def cmap_picker(key: str) -> str:
    opt = st.selectbox("Theme", ["Set2", "Blues", "Greens", "Reds", "Purples", "viridis", "plasma"], key=key)
    return opt


def top_n_slider(mx: int, key: str) -> int:
    mn = min(3, max(1, mx - 1))
    return st.slider("Show top", mn, mx, min(15, mx), key=key) if mx > mn else mx


def bins_slider(key: str) -> int:
    return st.slider("Bins", 10, 100, 50, 5, key=key)


# ---------------------------------------------------------------------------
# Interactive chart renderer — enhanced with stats, tables, and more controls
# ---------------------------------------------------------------------------

def show_stats(s: dict):
    """Display summary stats as inline metrics."""
    cols = st.columns(len(s))
    for c, (k, v) in zip(cols, s.items()):
        c.markdown(f'<div class="kpi"><div class="lbl">{k}</div><div class="val" style="font-size:1.1rem;">{v}</div></div>', unsafe_allow_html=True)


def render_interactive_question(qnum: str, analyzer: CarSalesAnalyzer, df: pd.DataFrame):
    import pandas as pd
    q = qnum.upper()
    ctrl, chart = st.columns([1, 3])

    if q == "Q1":
        p = df["Price"].dropna()
        with ctrl:
            bins = bins_slider(f"{qnum}_b")
            kde = st.checkbox("KDE", True, f"{qnum}_kde")
            mn = st.checkbox("Mean line", True, f"{qnum}_mn")
            md = st.checkbox("Median line", True, f"{qnum}_md")
            log = st.checkbox("Log scale", False, f"{qnum}_log")
            show_table = st.checkbox("Show stats table", True, f"{qnum}_tbl")
        with chart:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(p / 1000, bins=bins, color=P["blue"], edgecolor="white", alpha=0.8, density=True, log=log)
            if kde:
                k = sp_stats.gaussian_kde(p)
                x = np.linspace(p.min(), p.max(), 400)
                ax.plot(x / 1000, k(x) * 1000, color=P["red"], lw=2, label="KDE")
            if mn:
                ax.axvline(p.mean() / 1000, color=P["amber"], ls="--", lw=1.5, label=f"Mean ${p.mean()/1000:,.0f}k")
            if md:
                ax.axvline(p.median() / 1000, color=P["green"], ls=":", lw=1.5, label=f"Median ${p.median()/1000:,.0f}k")
            ax.set(xlabel="Price (thousands $)", ylabel="Density")
            ax.legend(fontsize=8)
            fig.tight_layout()
            render(fig)
        if show_table:
            show_stats({"Mean": f"${p.mean():,.0f}", "Median": f"${p.median():,.0f}", "Std": f"${p.std():,.0f}", "Min": f"${p.min():,.0f}", "Max": f"${p.max():,.0f}", "Skew": f"{p.skew():.2f}"})

    elif q == "Q2":
        result = analyzer.monthly_sales_trend(plot=False)
        mo = result.series
        with ctrl:
            fill = st.checkbox("Fill area", True, f"{qnum}_fill")
            markers = st.checkbox("Markers", True, f"{qnum}_mk")
            ma = st.slider("Moving avg window", 1, 12, 3, key=f"{qnum}_ma")
            show_table = st.checkbox("Show monthly stats", True, f"{qnum}_tbl")
        with chart:
            vals = pd.Series(mo.values).rolling(ma, min_periods=1).mean()
            fig, ax = plt.subplots(figsize=(12, 3.5))
            ax.plot(mo.index, vals, marker="o" if markers else "", color=P["blue"], lw=1.5, markersize=2.5)
            if fill:
                ax.fill_between(mo.index, vals, alpha=0.12, color=P["blue"])
            ax.set(xlabel="Date", ylabel="Sales", title="Monthly Sales Trend")
            fig.tight_layout()
            render(fig)
        if show_table:
            show_stats({"Avg Monthly": f"{mo.mean():.0f}", "Peak": f"{mo.max():.0f}", "Lowest": f"{mo.min():.0f}", "Total": f"{mo.sum():,}", "Months": f"{len(mo)}"})

    elif q == "Q3":
        result = analyzer.sales_by_region(plot=False)
        rc = result.series
        with ctrl:
            tp = chart_type_picker(f"{qnum}_tp")
            cm = cmap_picker(f"{qnum}_cm")
            n = top_n_slider(len(rc), f"{qnum}_n")
            show_table = st.checkbox("Show data table", True, f"{qnum}_tbl")
        d = rc.head(n)
        with chart:
            colors = getattr(plt.cm, cm)(np.linspace(0.2, 0.8, len(d)))
            fig, ax = plt.subplots(figsize=(10, max(3.5, len(d) * 0.3)))
            if tp == "barh":
                b = ax.barh(d.index, d.values, color=colors, edgecolor="white")
                ax.bar_label(b, labels=[f"{v:,}" for v in d.values], padding=2, fontsize=8)
                ax.set(xlabel="Sales"); ax.invert_yaxis()
            elif tp == "bar":
                b = ax.bar(d.index, d.values, color=colors, edgecolor="white")
                ax.bar_label(b, labels=[f"{v:,}" for v in d.values], padding=2, fontsize=8)
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
            elif tp == "area":
                ax.fill_between(range(len(d)), d.values, alpha=0.4, color=P["blue"])
                ax.plot(d.index, d.values, color=P["blue"], lw=1.5)
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
            else:
                ax.pie(d.values, labels=d.index, autopct="%1.1f%%", colors=colors, startangle=90)
            ax.set_title("Sales by Region")
            fig.tight_layout()
            render(fig)
        if show_table:
            show_stats({"Top Region": f"{rc.index[0]} ({rc.iloc[0]:,})", "Regions": f"{len(rc)}", "Total Sales": f"{rc.sum():,}"})

    elif q == "Q4":
        g = df["Gender"].value_counts()
        with ctrl:
            tp = st.selectbox("Type", ["pie", "bar", "donut"], key=f"{qnum}_tp")
            show_table = st.checkbox("Show breakdown", True, f"{qnum}_tbl")
        with chart:
            fig, ax = plt.subplots(figsize=(6, 4))
            if tp == "pie":
                ax.pie(g.values, labels=[f"{l}\n({v/g.sum()*100:.1f}%)" for l, v in zip(g.index, g.values)], colors=["#ff9999", "#66b3ff"], startangle=90, explode=(0.03, 0.03))
            elif tp == "donut":
                wedges, _, autotexts = ax.pie(g.values, labels=g.index, autopct="%1.1f%%", colors=["#ff9999", "#66b3ff"], startangle=90)
                for w in wedges:
                    w.set_width(0.6)
            else:
                b = ax.bar(g.index, g.values, color=["#ff9999", "#66b3ff"], edgecolor="white", width=0.4)
                ax.bar_label(b, labels=[f"{v:,} ({v/g.sum()*100:.1f}%)" for v in g.values], padding=4, fontsize=9)
                ax.set(ylabel="Count")
            ax.set_title("Gender Distribution")
            fig.tight_layout()
            render(fig)
        if show_table:
            show_stats({"Male": f"{g.get('Male', 0):,}", "Female": f"{g.get('Female', 0):,}", "Total": f"{g.sum():,}", "Ratio": f"{g.get('Male', 0)/g.sum()*100:.1f}% / {g.get('Female', 0)/g.sum()*100:.1f}%"})

    elif q == "Q5":
        with ctrl:
            tp = st.selectbox("Type", ["box", "violin", "bar"], key=f"{qnum}_tp")
            cm = cmap_picker(f"{qnum}_cm")
            show_table = st.checkbox("Show income stats", True, f"{qnum}_tbl")
        with chart:
            order = df.groupby("Dealer_Region")["Annual_Income"].median().sort_values().index
            fig, ax = plt.subplots(figsize=(12, max(3.5, len(order) * 0.3)))
            colors = getattr(plt.cm, cm)(np.linspace(0.2, 0.8, len(order)))
            if tp == "box":
                bp = ax.boxplot([df[df["Dealer_Region"] == r]["Annual_Income"].dropna() / 1000 for r in order], labels=order, patch_artist=True, notch=True)
                for p, c in zip(bp["boxes"], colors):
                    p.set_facecolor(c)
            elif tp == "violin":
                parts = ax.violinplot([df[df["Dealer_Region"] == r]["Annual_Income"].dropna() / 1000 for r in order], positions=range(len(order)), showmedians=True)
                for i, pc in enumerate(parts["bodies"]):
                    pc.set_facecolor(colors[i]); pc.set_alpha(0.7)
                ax.set_xticks(range(len(order))); ax.set_xticklabels(order)
            else:
                means = df.groupby("Dealer_Region")["Annual_Income"].mean().loc[order]
                b = ax.bar(range(len(order)), means.values / 1000, color=colors, edgecolor="white")
                ax.bar_label(b, labels=[f"${v:.0f}k" for v in means.values / 1000], padding=2, fontsize=8)
                ax.set_xticks(range(len(order))); ax.set_xticklabels(order)
            ax.set(ylabel="Income (thousands $)")
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
            fig.tight_layout()
            render(fig)
        if show_table:
            ir = df.groupby("Dealer_Region")["Annual_Income"].agg(["mean", "median", "min", "max"])
            show_stats({"Avg Income": f"${ir['mean'].mean():,.0f}", "Highest Region": f"{order[-1]} (${ir.loc[order[-1], 'median']:,.0f})", "Lowest Region": f"{order[0]} (${ir.loc[order[0], 'median']:,.0f})", "Regions": f"{len(order)}"})

    elif q == "Q6":
        result = analyzer.income_vs_price(plot=False)
        raw = result.raw; sub = raw["sub"]; slope, intercept, r_val = raw["slope"], raw["intercept"], raw["r_val"]
        with ctrl:
            sz = st.slider("Point size", 1, 20, 5, key=f"{qnum}_sz")
            al = st.slider("Opacity", 0.1, 1.0, 0.4, 0.1, key=f"{qnum}_al")
            reg = st.checkbox("Regression line", True, key=f"{qnum}_reg")
            show_table = st.checkbox("Show correlation stats", True, f"{qnum}_tbl")
        with chart:
            s = sub.sample(min(5000, len(sub)), random_state=42)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.scatter(s["Annual_Income"] / 1000, s["Price"] / 1000, alpha=al, s=sz, color=P["blue"], label="Data")
            if reg:
                xl = np.linspace(sub["Annual_Income"].min(), sub["Annual_Income"].max(), 200)
                ax.plot(xl / 1000, (slope * xl + intercept) / 1000, color=P["red"], lw=2, label=f"OLS (R²={r_val**2:.3f})")
            ax.set(xlabel="Income (thousands $)", ylabel="Price (thousands $)")
            ax.legend(fontsize=8)
            fig.tight_layout()
            render(fig)
        if show_table:
            show_stats({"Pearson r": f"{r_val:.4f}", "R²": f"{r_val**2:.4f}", "Slope": f"{slope:.4f}", "Intercept": f"${intercept:,.0f}", "Sample": f"{len(sub):,}", "p-value": f"{raw['p_val']:.2e}"})

    elif q == "Q7":
        result = analyzer.avg_price_by_company(plot=False)
        top = result.table
        with ctrl:
            n = top_n_slider(len(top), f"{qnum}_n")
            ms = st.slider("Min sales", 1, 20, 5, key=f"{qnum}_ms")
            show_table = st.checkbox("Show data table", True, f"{qnum}_tbl")
        fd = top[top["count"] >= ms].head(n)
        with chart:
            fig, ax = plt.subplots(figsize=(10, max(3.5, len(fd) * 0.3)))
            pd = fd.sort_values("mean")
            colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(pd)))
            b = ax.barh(pd.index, pd["mean"], color=colors, edgecolor="white")
            ax.bar_label(b, labels=[f"${v:.0f}" for v in pd["mean"]], padding=2, fontsize=8)
            ax.set(xlabel="Avg Price ($)", title="Companies by Avg Price")
            ax.invert_yaxis()
            fig.tight_layout()
            render(fig)
        if show_table:
            show_stats({"Most Expensive": f"{pd.index[-1]} (${pd['mean'].iloc[-1]:,.0f})", "Cheapest": f"{pd.index[0]} (${pd['mean'].iloc[0]:,.0f})", "Companies": f"{len(fd)}", "Avg (all)": f"${top['mean'].mean():,.0f}"})

    elif q == "Q8":
        with ctrl:
            tp = st.selectbox("Type", ["box", "violin", "bar"], key=f"{qnum}_tp")
            cm = cmap_picker(f"{qnum}_cm")
            show_table = st.checkbox("Show price stats", True, f"{qnum}_tbl")
        with chart:
            order = df.groupby("Body_Style")["Price"].median().sort_values().index
            fig, ax = plt.subplots(figsize=(10, max(3.5, len(order) * 0.3)))
            colors = getattr(plt.cm, cm)(np.linspace(0.2, 0.8, len(order)))
            if tp == "box":
                bp = ax.boxplot([df[df["Body_Style"] == bs]["Price"].dropna() / 1000 for bs in order], labels=order, patch_artist=True, notch=True)
                for p, c in zip(bp["boxes"], colors):
                    p.set_facecolor(c)
            elif tp == "violin":
                parts = ax.violinplot([df[df["Body_Style"] == bs]["Price"].dropna() / 1000 for bs in order], positions=range(len(order)), showmedians=True)
                for i, pc in enumerate(parts["bodies"]):
                    pc.set_facecolor(colors[i]); pc.set_alpha(0.7)
                ax.set_xticks(range(len(order))); ax.set_xticklabels(order)
            else:
                means = df.groupby("Body_Style")["Price"].mean().loc[order]
                b = ax.bar(range(len(order)), means.values / 1000, color=colors, edgecolor="white")
                ax.bar_label(b, labels=[f"${v:.0f}k" for v in means.values / 1000], padding=2, fontsize=8)
                ax.set_xticks(range(len(order))); ax.set_xticklabels(order)
            ax.set(ylabel="Price (thousands $)")
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=25, ha="right")
            fig.tight_layout()
            render(fig)
        if show_table:
            bs = df.groupby("Body_Style")["Price"].agg(["mean", "median", "min", "max"])
            show_stats({"Highest": f"{order[-1]} (${bs.loc[order[-1], 'median']:,.0f})", "Lowest": f"{order[0]} (${bs.loc[order[0], 'median']:,.0f})", "Body Styles": f"{len(order)}", "Avg All": f"${df['Price'].mean():,.0f}"})

    elif q == "Q9":
        auto = df[df["Transmission"].str.lower() == "auto"]["Price"].dropna()
        manual = df[df["Transmission"].str.lower() == "manual"]["Price"].dropna()
        diff = auto.mean() - manual.mean()
        with ctrl:
            err = st.checkbox("Error bars", True, key=f"{qnum}_err")
            tp = st.selectbox("Style", ["bar", "box"], key=f"{qnum}_tp")
            show_table = st.checkbox("Show test results", True, f"{qnum}_tbl")
        with chart:
            fig, ax = plt.subplots(figsize=(8, 4))
            if tp == "bar":
                labels = ["Automatic", "Manual"]
                means = [auto.mean(), manual.mean()]
                sems = [auto.std() / np.sqrt(len(auto)), manual.std() / np.sqrt(len(manual))]
                b = ax.bar(labels, means, yerr=sems if err else None, capsize=8, color=["#66b3ff", "#ff9999"], edgecolor="black", width=0.4)
                ax.bar_label(b, labels=[f"${v:,.0f}" for v in means], padding=4, fontsize=10)
                ax.set(ylabel="Avg Price ($)")
            else:
                bp = ax.boxplot([auto / 1000, manual / 1000], labels=["Automatic", "Manual"], patch_artist=True, notch=True)
                bp["boxes"][0].set_facecolor("#66b3ff"); bp["boxes"][1].set_facecolor("#ff9999")
                ax.set(ylabel="Price (thousands $)")
            ax.set_title("Auto vs Manual")
            fig.tight_layout()
            render(fig)
        if show_table:
            _, t_p = sp_stats.ttest_ind(auto, manual, equal_var=False)
            sig = "p < 0.05" if t_p < 0.05 else "p ≥ 0.05"
            show_stats({"Auto Mean": f"${auto.mean():,.0f}", "Manual Mean": f"${manual.mean():,.0f}", "Difference": f"${diff:,.0f}", "Auto Count": f"{len(auto):,}", "Manual Count": f"{len(manual):,}", "t-test": sig})

    elif q == "Q10":
        result = analyzer.popular_colors(plot=False)
        colors = result.series
        with ctrl:
            n = top_n_slider(len(colors), f"{qnum}_n")
            other = st.checkbox("Group 'Other'", True, key=f"{qnum}_other")
            show_table = st.checkbox("Show color stats", True, f"{qnum}_tbl")
        top = colors.head(n)
        with chart:
            fig, ax = plt.subplots(figsize=(10, 4))
            pd = top.copy()
            if other:
                o = colors.iloc[n:].sum()
                if o > 0:
                    pd["Other"] = o
            palette = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#bfef45"]
            b = ax.barh(pd.index, pd.values, color=palette[:len(pd)], edgecolor="white")
            ax.bar_label(b, labels=[f"{v:,}" for v in pd.values], padding=2, fontsize=9)
            ax.set(xlabel="Count", title="Most Popular Colors")
            ax.invert_yaxis()
            fig.tight_layout()
            render(fig)
        if show_table:
            top3 = colors.head(3)
            top3pct = top3.sum() / colors.sum() * 100
            show_stats({"#1": f"{colors.index[0]} ({colors.iloc[0]:,})", "#2": f"{colors.index[1]} ({colors.iloc[1]:,})", "#3": f"{colors.index[2]} ({colors.iloc[2]:,})", "Top 3 %": f"{top3pct:.1f}%", "Total Colors": f"{len(colors)}"})

    elif q == "Q11":
        with ctrl:
            cm = st.selectbox("Colour", ["YlOrRd", "Blues", "Greens", "Reds", "Purples", "BuPu", "YlGnBu"], key=f"{qnum}_cm")
            vals = st.checkbox("Show values", True, key=f"{qnum}_vals")
            show_table = st.checkbox("Show cross-tab", True, f"{qnum}_tbl")
        ct = pd.crosstab(df["Body_Style"], df["Transmission"])
        with chart:
            fig, ax = plt.subplots(figsize=(8, 5))
            im = ax.imshow(ct.values, cmap=cm, aspect="auto")
            ax.set_xticks(range(len(ct.columns))); ax.set_xticklabels(ct.columns)
            ax.set_yticks(range(len(ct.index))); ax.set_yticklabels(ct.index)
            if vals:
                for i in range(len(ct.index)):
                    for j in range(len(ct.columns)):
                        ax.text(j, i, ct.values[i, j], ha="center", va="center", fontsize=9, fontweight="bold")
            ax.set(xlabel="Transmission", ylabel="Body Style")
            plt.colorbar(im, ax=ax, label="Count")
            fig.tight_layout()
            render(fig)
        if show_table:
            show_stats({"Body Styles": f"{len(ct.index)}", "Trans Types": f"{len(ct.columns)}", "Most Common": f"{ct.values.max():,}", "Total": f"{ct.values.sum():,}"})

    elif q == "Q12":
        result = analyzer.multiple_regression(plot=False)
        raw = result.raw; y_pred = raw["y_pred"]
        y = df[["Price", "Annual_Income", "Transmission", "Engine", "Body_Style"]].dropna().copy()["Price"]
        residuals = raw["residuals"]
        with ctrl:
            rb = st.slider("Residual bins", 10, 80, 40, key=f"{qnum}_rb")
            pa = st.slider("Point alpha", 0.1, 1.0, 0.4, 0.1, key=f"{qnum}_pa")
            show_table = st.checkbox("Show model stats", True, f"{qnum}_tbl")
        with chart:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
            ax1.scatter(y_pred / 1000, y / 1000, alpha=pa, s=6, color=P["blue"])
            ax1.plot([y.min() / 1000, y.max() / 1000], [y.min() / 1000, y.max() / 1000], "r--", lw=2, label="Perfect")
            ax1.set(xlabel="Predicted (k$)", ylabel="Actual (k$)"); ax1.legend(fontsize=8)
            ax2.hist(residuals / 1000, bins=rb, color=P["blue"], edgecolor="white")
            ax2.set(xlabel="Residuals (k$)", ylabel="Frequency")
            fig.tight_layout()
            render(fig)
        if show_table:
            rmse = np.sqrt(np.mean(residuals**2))
            show_stats({"R²": f"{result.summary['r_squared']}", "Adj R²": f"{result.summary['adj_r_squared']}", "RMSE": f"${rmse:,.0f}", "Sample": f"{len(y):,}", "Features": f"{result.summary['features']}"})

    elif q == "Q13":
        result = analyzer.outliers_zscore(plot=False)
        raw = result.raw; prices = raw["prices"]; z = raw["z_scores"]
        with ctrl:
            th = st.slider("Z-score thresh", 1.5, 5.0, 3.0, 0.1, key=f"{qnum}_th")
            ml = st.checkbox("Mean line", True, key=f"{qnum}_ml")
            show_table = st.checkbox("Show outlier stats", True, f"{qnum}_tbl")
        no = prices[np.abs(zscore(prices)) > th]
        with chart:
            fig, ax = plt.subplots(figsize=(12, 3.5))
            ax.scatter(range(len(prices)), prices / 1000, alpha=0.3, s=4, color=P["blue"], label="Normal")
            if len(no) > 0:
                ax.scatter(np.where(np.abs(z) > th)[0], no / 1000, alpha=0.7, s=15, color=P["red"], label=f"Outliers (n={len(no)})")
            if ml:
                ax.axhline(prices.mean() / 1000, color=P["green"], ls="--", lw=1, label=f"Mean ${prices.mean()/1000:,.1f}k")
            ax.set(xlabel="Index", ylabel="Price (k$)", title=f"Outliers (|z| > {th})")
            ax.legend(fontsize=8)
            fig.tight_layout()
            render(fig)
        if show_table:
            show_stats({"Total": f"{len(prices):,}", "Outliers": f"{len(no):,}", "Outlier %": f"{len(no)/len(prices)*100:.2f}%", "Min Outlier": f"${no.min():,.0f}" if len(no) > 0 else "N/A", "Max Outlier": f"${no.max():,.0f}" if len(no) > 0 else "N/A"})

    elif q == "Q14":
        result = analyzer.avg_price_by_dealer(plot=False)
        top = result.table
        with ctrl:
            n = top_n_slider(len(top), f"{qnum}_n")
            cm = st.selectbox("Colour", ["Blues", "Greens", "Reds", "Purples", "Oranges"], key=f"{qnum}_cm")
            show_table = st.checkbox("Show dealer stats", True, f"{qnum}_tbl")
        pd = top.head(n).sort_values("mean")
        with chart:
            fig, ax = plt.subplots(figsize=(10, max(3.5, len(pd) * 0.3)))
            colors = getattr(plt.cm, cm)(np.linspace(0.3, 0.9, len(pd)))
            b = ax.barh(pd.index, pd["mean"], color=colors, edgecolor="white")
            ax.bar_label(b, labels=[f"${v:.0f}" for v in pd["mean"]], padding=2, fontsize=8)
            ax.set(xlabel="Avg Price ($)", title="Top Dealers by Avg Price")
            ax.invert_yaxis()
            fig.tight_layout()
            render(fig)
        if show_table:
            show_stats({"Top Dealer": f"{pd.index[-1]} (${pd['mean'].iloc[-1]:,.0f})", "Dealers": f"{len(top)}", "Avg (all)": f"${top['mean'].mean():,.0f}"})

    elif q == "Q15":
        result = analyzer.normality_test(plot=False)
        prices = df["Price"].dropna(); sample = result.raw["sample"]
        mu, sigma = prices.mean(), prices.std()
        with ctrl:
            nb = bins_slider(f"{qnum}_nb")
            show_table = st.checkbox("Show test results", True, f"{qnum}_tbl")
        with chart:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
            sp_stats.probplot(sample, dist="norm", plot=ax1)
            ax1.set(title="Q-Q Plot")
            ax2.hist(prices / 1000, bins=nb, color=P["blue"], edgecolor="white", density=True, alpha=0.7)
            x = np.linspace(prices.min(), prices.max(), 500)
            ax2.plot(x / 1000, norm.pdf(x, mu, sigma), color=P["red"], lw=2, label="Normal")
            ax2.set(xlabel="Price (k$)", ylabel="Density"); ax2.legend(fontsize=8)
            fig.tight_layout()
            render(fig)
        if show_table:
            shap_stat, shap_p = sp_stats.shapiro(sample)
            is_normal = shap_p > 0.05
            show_stats({"Shapiro-Wilk": f"stat={shap_stat:.4f}, p={shap_p:.2e}", "Normal?": "No" if not is_normal else "Yes", "Skewness": f"{prices.skew():.4f}", "Kurtosis": f"{prices.kurtosis():.4f}"})


# ---------------------------------------------------------------------------
# HOME
# ---------------------------------------------------------------------------

def page_home():
    analyzer = get_analyzer()
    df = analyzer.df

    header(
        "🏠 Car Sales Analytics",
        f"All 15 business questions answered with interactive charts, statistics & insights on {len(df):,} car sales records",
    )

    cols = st.columns(6)
    for c, (v, l) in zip(cols, [
        (f"{len(df):,}", "Records"),
        (usd(df["Price"].sum()), "Revenue"),
        (usd(df["Price"].mean()), "Avg Price"),
        (f"{df['Company'].nunique()}", "Companies"),
        (f"{df['Dealer_Region'].nunique()}", "Regions"),
        (f"{df['Body_Style'].nunique()}", "Body Styles"),
    ]):
        c.markdown(f'<div class="kpi"><div class="lbl">{l}</div><div class="val">{v}</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    st.markdown(
        "<div style='display:flex;align-items:center;justify-content:space-between;'>"
        "<h3 style='margin:0;'>📋 All 15 Business Questions</h3>"
        f"<span style='color:{P['gray500']};font-size:0.8rem;'>⬇ Click any section to expand the interactive chart & stats</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    for num, title, qtext, answer in QUESTIONS:
        with st.expander(f"**{num}** — {title}", True):
            st.markdown(f'<div class="q-text" style="font-size:0.95rem;margin-bottom:0.5rem;">{qtext}</div>', unsafe_allow_html=True)
            insight(f"**Answer:** {answer}")
            render_interactive_question(num, analyzer, df)
            st.markdown(f'<div style="text-align:right;margin-top:8px;"><a href="?page={num.lower()}" target="_self" style="color:{P["blue"]};font-size:0.82rem;text-decoration:none;">Open full page ➔</a></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# PAGE: Single Question
# ---------------------------------------------------------------------------

def render_question(num: str, title: str, question: str, answer: str):
    analyzer = get_analyzer()
    df = analyzer.df

    header(f"{num}: {title}", question)

    insight(f"**Answer:** {answer}")

    render_interactive_question(num, analyzer, df)

    # Navigation footer
    st.markdown("---")
    all_keys = [q[0].lower() for q in QUESTIONS]
    idx = all_keys.index(num.lower())
    c1, c2, c3 = st.columns([1, 2, 1])
    if idx > 0:
        prev_k = all_keys[idx - 1]
        prev_q = QUESTIONS[idx - 1]
        c1.markdown(f'<a href="?page={prev_k}" style="text-decoration:none;font-size:0.9rem;">⬅ {prev_q[0]}: {prev_q[1][:30]}</a>', unsafe_allow_html=True)
    if idx < len(all_keys) - 1:
        next_k = all_keys[idx + 1]
        next_q = QUESTIONS[idx + 1]
        c3.markdown(f'<a href="?page={next_k}" style="text-decoration:none;font-size:0.9rem;float:right;">{next_q[0]}: {next_q[1][:30]} ➔</a>', unsafe_allow_html=True)
    c2.markdown(f'<div style="text-align:center;color:{P["gray400"]};font-size:0.8rem;">{num} of 15</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# PAGE: Filter & Explore
# ---------------------------------------------------------------------------

def page_filter():
    analyzer = get_analyzer()
    df = analyzer.df

    header("🔍 Filter & Explore", "Slice the dataset by any combination of dimensions")

    with st.container():
        st.markdown('<div class="q-card">', unsafe_allow_html=True)
        r1, r2, r3 = st.columns(3)
        region = r1.selectbox("Region", [""] + sorted(df["Dealer_Region"].unique().tolist()))
        company = r1.selectbox("Company", [""] + sorted(df["Company"].unique().tolist()))
        body = r2.selectbox("Body Style", [""] + sorted(df["Body_Style"].unique().tolist()))
        trans = r2.selectbox("Transmission", [""] + sorted(df["Transmission"].unique().tolist()))
        color = r3.selectbox("Color", [""] + sorted(df["Color"].unique().tolist()))
        r3.markdown("**💰 Price & Income**")
        c1, c2 = r3.columns(2)
        p_min = c1.number_input("Price min", 0, value=0, step=1000)
        p_max = c2.number_input("Price max", 0, value=int(df["Price"].max()), step=1000)
        i_min = c1.number_input("Income min", 0, value=0, step=10000)
        i_max = c2.number_input("Income max", 0, value=int(df["Annual_Income"].max()), step=10000)
        st.markdown("</div>", unsafe_allow_html=True)

    pr = (p_min, p_max) if p_min > 0 or p_max < df["Price"].max() else None
    ir = (i_min, i_max) if i_min > 0 or i_max < df["Annual_Income"].max() else None

    filtered = analyzer.filter_dataframe(region=region or None, company=company or None, body_style=body or None, transmission=trans or None, color=color or None, price_range=pr, income_range=ir)

    st.markdown(f"### 📄 Results: {len(filtered):,} records")
    if filtered.empty:
        st.warning("No matches.")
        return

    cols = st.columns(4)
    for c, (v, l) in zip(cols, [
        (f"{len(filtered):,}", "Records"),
        (usd(filtered["Price"].sum()), "Revenue"),
        (usd(filtered["Price"].mean()), "Avg Price"),
        (usd(filtered["Annual_Income"].mean()), "Avg Income"),
    ]):
        c.markdown(f'<div class="kpi"><div class="lbl">{l}</div><div class="val">{v}</div></div>', unsafe_allow_html=True)

    t1, t2 = st.tabs(["📋 Data", "📊 Summary"])
    with t1:
        st.dataframe(filtered, use_container_width=True, hide_index=True)
        buf = io.BytesIO()
        filtered.to_csv(buf, index=False)
        st.download_button("📥 Export CSV", data=buf.getvalue(), file_name="filtered_car_sales.csv", mime="text/csv")
    with t2:
        ca, cb = st.columns(2)
        for col, cn, co in [("Dealer_Region", "Region", ca), ("Company", "Company", cb), ("Body_Style", "Body Style", ca), ("Transmission", "Transmission", cb)]:
            if filtered[col].nunique() <= 20:
                co.markdown(f"**{cn}**")
                f = filtered[col].value_counts().reset_index()
                f.columns = [cn, "Count"]
                f["%"] = (f["Count"] / f["Count"].sum() * 100).round(1)
                co.dataframe(f, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# PAGE: Compare Segments
# ---------------------------------------------------------------------------

def page_compare():
    analyzer = get_analyzer()
    df = analyzer.df

    header("⚖️ Compare Segments", "Pick any two segments and compare them side-by-side with a Welch t-test")

    cmap = {"Region": "Dealer_Region", "Company": "Company", "Body Style": "Body_Style", "Transmission": "Transmission", "Gender": "Gender", "Color": "Color"}

    with st.container():
        st.markdown('<div class="q-card">', unsafe_allow_html=True)
        dim = st.selectbox("Dimension", list(cmap.keys()))
        col = cmap[dim]
        vals = df[col].value_counts().head(15).index.tolist()
        c1, c2 = st.columns(2)
        v1 = c1.selectbox("First", vals, key="v1")
        v2 = c2.selectbox("Second", vals, key="v2", index=min(1, len(vals) - 1))
        st.markdown("</div>", unsafe_allow_html=True)

    if v1 == v2:
        st.warning("Pick two different segments.")
        return

    d1 = df[df[col] == v1]
    d2 = df[df[col] == v2]

    st.markdown(f"### 🆚 {v1} vs {v2}")

    cols = st.columns(4)
    for c, (v, l) in zip(cols, [
        (f"{len(d1):,}", v1), (usd(d1["Price"].mean()), f"Avg ({v1})"),
        (f"{len(d2):,}", v2), (usd(d2["Price"].mean()), f"Avg ({v2})"),
    ]):
        c.markdown(f'<div class="kpi"><div class="lbl">{l}</div><div class="val">{v}</div></div>', unsafe_allow_html=True)

    st.table(pd.DataFrame({
        "Metric": ["Count", "Avg Price", "Median Price", "Avg Income", "Revenue", "Std Dev"],
        v1: [f"{len(d1):,}", usd(d1["Price"].mean()), usd(d1["Price"].median()), usd(d1["Annual_Income"].mean()), usd(d1["Price"].sum()), usd(d1["Price"].std())],
        v2: [f"{len(d2):,}", usd(d2["Price"].mean()), usd(d2["Price"].median()), usd(d2["Annual_Income"].mean()), usd(d2["Price"].sum()), usd(d2["Price"].std())],
    }))

    sidebar, chart = st.columns([1, 3])
    with sidebar:
        tp = st.selectbox("Chart type", ["overlaid", "side-by-side"], key="cmp_tp")
        al = st.slider("Opacity", 0.2, 1.0, 0.6, 0.1, key="cmp_al")
    with chart:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
        axes[0].hist(d1["Price"] / 1000, bins=30, alpha=al, color=P["blue"], label=v1, edgecolor="white")
        axes[0].hist(d2["Price"] / 1000, bins=30, alpha=al, color=P["red"], label=v2, edgecolor="white")
        axes[0].set(xlabel="Price (k$)", ylabel="Frequency")
        axes[0].legend(fontsize=8)

        cmp = pd.DataFrame({"Metric": ["Count", "Avg Price", "Median", "Std Dev"], v1: [len(d1), d1["Price"].mean(), d1["Price"].median(), d1["Price"].std()], v2: [len(d2), d2["Price"].mean(), d2["Price"].median(), d2["Price"].std()]})
        x = np.arange(len(cmp))
        w = 0.35
        axes[1].bar(x - w / 2, cmp[v1] / cmp[v1].max() * 100, w, label=v1, color=P["blue"])
        axes[1].bar(x + w / 2, cmp[v2] / cmp[v2].max() * 100, w, label=v2, color=P["red"])
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(cmp["Metric"], rotation=15)
        axes[1].set(ylabel="Normalised (%)")
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        render(fig)

    if len(d1) > 5 and len(d2) > 5:
        t, p = ttest_ind(d1["Price"].dropna(), d2["Price"].dropna(), equal_var=False)
        sig = "✅ Significant" if p < 0.05 else "❌ Not significant"
        insight(f"**Welch t-test:** t = {t:.4f}, p = {p:.2e} — {sig} at α = 0.05")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def main():
    st.markdown(CSS, unsafe_allow_html=True)
    render_sidebar()

    page = st.query_params.get("page", "home")

    if page == "home":
        page_home()
    elif page == "filter":
        page_filter()
    elif page == "compare":
        page_compare()
    elif page in QUESTION_KEYS:
        idx = QUESTION_KEYS.index(page)
        render_question(*QUESTIONS[idx])
    else:
        page_home()


if __name__ == "__main__":
    main()
