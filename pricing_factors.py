import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.decomposition import PCA

# ---- IMPORTS
fitted_zero_SGB = pd.read_excel("zero_yields_SGB.xlsx", sheet_name="zero_yields")
params_SGB = pd.read_excel("zero_yields_SGB.xlsx", sheet_name="fit_params")

fitted_zero_SGBIL = pd.read_excel("zero_yields_SGBIL.xlsx", sheet_name="zero_yields")
params_SGBIL = pd.read_excel("zero_yields_SGBIL.xlsx", sheet_name="fit_params")

kpi_data = pd.read_excel("kpi_data.xlsx", sheet_name="basår 1980")

nominal_bid_ask = pd.read_excel("nominal_bid_ask.xlsx")
linked_bid_ask = pd.read_excel("inflation_linked_bid_ask.xlsx")

turnover = pd.read_excel("turnover_govt_bonds.xlsx")

# ---- FORMAT DATES
fitted_zero_SGB["date"] = pd.to_datetime(fitted_zero_SGB["date"])
params_SGB["date"] = pd.to_datetime(params_SGB["date"])

fitted_zero_SGBIL["date"] = pd.to_datetime(fitted_zero_SGBIL["date"])
params_SGBIL["date"] = pd.to_datetime(params_SGBIL["date"])

kpi_data["date"] = pd.to_datetime(kpi_data["date"], errors="coerce")

nominal_bid_ask["date"] = pd.to_datetime(nominal_bid_ask["date"])
linked_bid_ask["date"] = pd.to_datetime(linked_bid_ask["date"])

turnover["Month"] = pd.to_datetime(turnover["Month"])

# region TERM STRUCTURE PLOTS

start_date = pd.Timestamp("2004-01-01")
end_date   = pd.Timestamp("2025-12-31")

# fixed x-axis ticks for all plots
tick_years = [2005, 2010, 2015, 2020, 2025]
tick_dates = [pd.Timestamp(f"{year}-01-01") for year in tick_years]

# ---- 1) FITTED NOMINAL ZERO-COUPON YIELDS: SELECTED MATURITIES

plot_df = fitted_zero_SGB.loc[
    (fitted_zero_SGB["date"] >= start_date) &
    (fitted_zero_SGB["date"] <= end_date)
].copy()

maturity_cols = [
    "y_1m",
    "y_12m",
    "y_24m",
    "y_48m",
    "y_72m",
    "y_96m",
    "y_120m",
]

labels = {
    "y_1m": "1m",
    "y_12m": "1y",
    "y_24m": "2y",
    "y_48m": "4y",
    "y_72m": "6y",
    "y_96m": "8y",
    "y_120m": "10y",
}

colors = [
    "#0A2342",
    "#123A73",
    "#1A52A3",
    "#2369D3",
    "#0057FF",
    "#4F86FF",
    "#9CBFFF",
]

fig, ax = plt.subplots(figsize=(13, 7))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

for col, color in zip(maturity_cols, colors):
    ax.plot(
        plot_df["date"],
        plot_df[col],
        color=color,
        linewidth=2.2,
        label=labels[col]
    )

ax.set_xlabel("")
ax.set_ylabel("Yield (decimal)", fontsize=12)

for spine in ax.spines.values():
    spine.set_linewidth(1.0)
    spine.set_color("black")

ax.tick_params(
    axis="both",
    which="major",
    direction="in",
    top=True,
    right=True,
    length=5,
    width=1.0
)

ax.set_xlim(start_date, end_date)
ax.set_xticks(tick_dates)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(axis="x", rotation=45)

ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.4)
ax.margins(x=0)

ax.legend(
    loc="upper left",
    ncol=3,
    frameon=True,
    fancybox=False,
    edgecolor="black",
    facecolor="white",
    framealpha=1.0
)

fig.tight_layout()
ymin, ymax = ax.get_ylim()
ax.set_ylim(ymin, ymax + 0.01)
plt.show()

# ---- 2) FITTED INFLATION-LINKED ZERO-COUPON YIELDS: SELECTED MATURITIES

plot_df = fitted_zero_SGBIL.loc[
    (fitted_zero_SGBIL["date"] >= start_date) &
    (fitted_zero_SGBIL["date"] <= end_date)
].copy()

maturity_cols = [
    "y_24m",
    "y_48m",
    "y_72m",
    "y_96m",
    "y_120m",
]

labels = {
    "y_24m": "2y",
    "y_48m": "4y",
    "y_72m": "6y",
    "y_96m": "8y",
    "y_120m": "10y",
}

colors = [
    "#123A73",
    "#1A52A3",
    "#0057FF",
    "#4F86FF",
    "#9CBFFF",
]

fig, ax = plt.subplots(figsize=(13, 7))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

for col, color in zip(maturity_cols, colors):
    ax.plot(
        plot_df["date"],
        plot_df[col],
        color=color,
        linewidth=2.2,
        label=labels[col]
    )

ax.set_title("", fontsize=16, pad=15)
ax.set_xlabel("")
ax.set_ylabel("Yield (decimal)", fontsize=12)

for spine in ax.spines.values():
    spine.set_linewidth(1.0)
    spine.set_color("black")

ax.tick_params(
    axis="both",
    which="major",
    direction="in",
    top=True,
    right=True,
    length=5,
    width=1.0
)

ax.set_xlim(start_date, end_date)
ax.set_xticks(tick_dates)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(axis="x", rotation=45)

ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.4)
ax.margins(x=0)

ax.legend(
    loc="upper left",
    ncol=3,
    frameon=True,
    fancybox=False,
    edgecolor="black",
    facecolor="white",
    framealpha=1.0
)

fig.tight_layout()
ymin, ymax = ax.get_ylim()
ax.set_ylim(ymin, ymax + 0.01)
plt.show()

# ---- 3) BEI 2Y, 5Y, 10Y

nominal_be = fitted_zero_SGB.loc[
    (fitted_zero_SGB["date"] >= start_date) &
    (fitted_zero_SGB["date"] <= end_date),
    ["date", "y_24m", "y_60m", "y_120m"]
].copy()

linker_be = fitted_zero_SGBIL.loc[
    (fitted_zero_SGBIL["date"] >= start_date) &
    (fitted_zero_SGBIL["date"] <= end_date),
    ["date", "y_24m", "y_60m", "y_120m"]
].copy()

be_df = pd.merge(
    nominal_be,
    linker_be,
    on="date",
    how="inner",
    suffixes=("_nom", "_il")
)

be_df["be_2y"] = be_df["y_24m_nom"] - be_df["y_24m_il"]
be_df["be_5y"] = be_df["y_60m_nom"] - be_df["y_60m_il"]
be_df["be_10y"] = be_df["y_120m_nom"] - be_df["y_120m_il"]

fig, ax = plt.subplots(figsize=(13, 7))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

be_cols = ["be_2y", "be_5y", "be_10y"]
be_labels = {
    "be_2y": "2y",
    "be_5y": "5y",
    "be_10y": "10y",
}
be_colors = ["#123A73", "#0057FF", "#9CBFFF"]

for col, color in zip(be_cols, be_colors):
    ax.plot(
        be_df["date"],
        be_df[col],
        color=color,
        linewidth=2.2,
        label=be_labels[col]
    )

ax.set_xlabel("")
ax.set_ylabel("Spread (decimal)", fontsize=12)

for spine in ax.spines.values():
    spine.set_linewidth(1.0)
    spine.set_color("black")

ax.tick_params(
    axis="both",
    which="major",
    direction="in",
    top=True,
    right=True,
    length=5,
    width=1.0
)

ax.set_xlim(start_date, end_date)
ax.set_xticks(tick_dates)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(axis="x", rotation=45)

ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.4)
ax.margins(x=0)

ax.legend(
    loc="upper left",
    ncol=3,
    frameon=True,
    fancybox=False,
    edgecolor="black",
    facecolor="white",
    framealpha=1.0
)

fig.tight_layout()
ymin, ymax = ax.get_ylim()
ax.set_ylim(ymin, ymax + 0.01)
plt.show()


# ---- 4) BEI vs. INFLATION

# Prepare monthly KPI levels
df_kpi_plot = kpi_data.loc[:, ["date", "KPI"]].copy()
df_kpi_plot["date"] = pd.to_datetime(df_kpi_plot["date"], errors="coerce")
df_kpi_plot["KPI"] = pd.to_numeric(df_kpi_plot["KPI"], errors="coerce")
df_kpi_plot["month"] = df_kpi_plot["date"].dt.to_period("M")

df_kpi_plot = (
    df_kpi_plot.loc[df_kpi_plot["date"].between(start_date, end_date)]
    .dropna(subset=["month", "KPI"])
    .sort_values("month")
    .drop_duplicates(subset=["month"], keep="last")
    .reset_index(drop=True)
)

# Realized annualized CPI inflation h years ahead:
# ((KPI_{t+h}/KPI_t)^(1/h) - 1)
df_kpi_plot["realized_cpi_2y_fwd"] = (
    (df_kpi_plot["KPI"].shift(-24) / df_kpi_plot["KPI"]) ** (1 / 2.0) - 1.0
)

df_kpi_plot["realized_cpi_5y_fwd"] = (
    (df_kpi_plot["KPI"].shift(-60) / df_kpi_plot["KPI"]) ** (1 / 5.0) - 1.0
)

df_kpi_plot["realized_cpi_10y_fwd"] = (
    (df_kpi_plot["KPI"].shift(-120) / df_kpi_plot["KPI"]) ** (1 / 10.0) - 1.0
)

# Backward-looking realized annual inflation rate:
# KPI_t / KPI_{t-12} - 1
df_kpi_plot["realized_cpi_1y_back"] = (
    df_kpi_plot["KPI"] / df_kpi_plot["KPI"].shift(12) - 1.0
)

# Keep only the fields needed for plotting
df_kpi_plot = df_kpi_plot.loc[:, [
    "month",
    "realized_cpi_2y_fwd",
    "realized_cpi_5y_fwd",
    "realized_cpi_10y_fwd",
    "realized_cpi_1y_back",
]]

# Map KPI monthly series onto the third-plot date grid
plot4_df = be_df.copy()
plot4_df["month"] = plot4_df["date"].dt.to_period("M")

plot4_df = pd.merge(
    plot4_df,
    df_kpi_plot,
    on="month",
    how="left"
).sort_values("date").reset_index(drop=True)

fig, ax = plt.subplots(figsize=(13, 7))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# ---- Break-even (nominal - IL)
ax.plot(
    plot4_df["date"],
    plot4_df["be_2y"],
    color="#123A73",
    linewidth=2.2,
    linestyle="-",
    label="BEI 2y"
)

ax.plot(
    plot4_df["date"],
    plot4_df["be_5y"],
    color="#0057FF",
    linewidth=2.2,
    linestyle="-",
    label="BEI 5y"
)

# ---- Realized forward CPI (annualized)
ax.plot(
    plot4_df["date"],
    plot4_df["realized_cpi_2y_fwd"],
    color="#1B5E20",
    linewidth=2.2,
    linestyle="-",
    label="Inflation, 2y forward"
)

ax.plot(
    plot4_df["date"],
    plot4_df["realized_cpi_5y_fwd"],
    color="#66BB6A",
    linewidth=2.2,
    linestyle="-",
    label="Inflation, 5y forward"
)

# ---- Backward-looking realized CPI (1y), red dash-dot
ax.plot(
    plot4_df["date"],
    plot4_df["realized_cpi_1y_back"],
    color="#C62828",
    linewidth=2.0,
    linestyle="-.",
    label="Realized inflation",
)

ax.set_xlabel("")
ax.set_ylabel("Annual rate (decimal)", fontsize=12)

for spine in ax.spines.values():
    spine.set_linewidth(1.0)
    spine.set_color("black")

ax.tick_params(
    axis="both",
    which="major",
    direction="in",
    top=True,
    right=True,
    length=5,
    width=1.0
)

ax.set_xlim(start_date, end_date)
ax.set_xticks(tick_dates)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(axis="x", rotation=45)

# ---- Y-axis cap at 10%
ax.set_ylim(bottom=None, top=0.10)

ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.4)
ax.margins(x=0)

ax.legend(
    loc="upper left",
    ncol=2,
    frameon=True,
    fancybox=False,
    edgecolor="black",
    facecolor="white",
    framealpha=1.0
)

fig.tight_layout()
plt.show()

# endregion

# region BID ASK

start_date = pd.Timestamp("2004-01-01")
end_date   = pd.Timestamp("2025-12-31")

tick_years = [2005, 2010, 2015, 2020, 2025]
tick_dates = [pd.Timestamp(f"{year}-01-01") for year in tick_years]

color_median = "#4F86FF"   
color_mean   = "#66BB6A"   


def plot_bid_ask_summary(df, ylabel, median_col, mean_col):
    plot_df = df.loc[
        (df["date"] >= start_date) &
        (df["date"] <= end_date)
    ].copy()

    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(
        plot_df["date"],
        plot_df[median_col],
        color=color_median,
        linewidth=2.2,
        linestyle="-",
        label="Median yield bid-ask spread (bp)"
    )

    ax.plot(
        plot_df["date"],
        plot_df[mean_col],
        color=color_mean,
        linewidth=2.2,
        linestyle="-",
        label="Average yield bid-ask spread (bp)"
    )

    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=12)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("black")

    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        length=5,
        width=1.0
    )

    ax.set_xlim(start_date, end_date)
    ax.set_xticks(tick_dates)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=45)

    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.4)
    ax.margins(x=0)

    ax.legend(
        loc="upper left",
        ncol=2,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        facecolor="white",
        framealpha=1.0
    )

    fig.tight_layout()
    plt.show()


# ---- 5) NOMINAL BID-ASK YIELD SPREAD: MEDIAN AND MEAN

plot_bid_ask_summary(
    df=nominal_bid_ask,
    ylabel="",
    median_col="median_bid_ask_yield_spread_bp",
    mean_col="mean_bid_ask_yield_spread_bp"
)

# ---- 6) INFLATION-LINKED BID-ASK YIELD SPREAD: MEDIAN AND MEAN

plot_bid_ask_summary(
    df=linked_bid_ask,
    ylabel="",
    median_col="median_bid_ask_yield_spread_bp",
    mean_col="mean_bid_ask_yield_spread_bp"
)

# ---- 7) AVERAGE BID-ASK SPREADS: NOMINAL, LINKED, AND LINKED/NOMINAL RATIO

plot_nom = nominal_bid_ask.loc[
    (nominal_bid_ask["date"] >= start_date) &
    (nominal_bid_ask["date"] <= end_date),
    ["date", "mean_bid_ask_yield_spread_bp"]
].copy()

plot_il = linked_bid_ask.loc[
    (linked_bid_ask["date"] >= start_date) &
    (linked_bid_ask["date"] <= end_date),
    ["date", "mean_bid_ask_yield_spread_bp"]
].copy()

plot_nom = plot_nom.rename(columns={"mean_bid_ask_yield_spread_bp": "nominal_mean_bp"})
plot_il = plot_il.rename(columns={"mean_bid_ask_yield_spread_bp": "linked_mean_bp"})

plot_df = pd.merge(plot_nom, plot_il, on="date", how="inner").sort_values("date").reset_index(drop=True)

plot_df["linked_to_nominal_ratio"] = plot_df["linked_mean_bp"] / plot_df["nominal_mean_bp"]

fig, ax1 = plt.subplots(figsize=(13, 7))
fig.patch.set_facecolor("white")
ax1.set_facecolor("white")

# Left axis: bid-ask spreads in bp
ax1.plot(
    plot_df["date"],
    plot_df["nominal_mean_bp"],
    color = "#4F86FF",
    linewidth=2.2,
    linestyle="-.",
    label="SGB average yield bid-ask spread bp (lhs)"
)

ax1.plot(
    plot_df["date"],
    plot_df["linked_mean_bp"],
    color="#66BB6A",
    linewidth=2.2,
    linestyle="-.",
    label="SGBi average yield bid-ask spread bp (lhs)"
)

ax1.set_xlabel("")
ax1.set_ylabel("", fontsize=12)

for spine in ax1.spines.values():
    spine.set_linewidth(1.0)
    spine.set_color("black")

ax1.tick_params(
    axis="both",
    which="major",
    direction="in",
    top=True,
    right=False,
    length=5,
    width=1.0
)

ax1.set_xlim(start_date, end_date)
ax1.set_xticks(tick_dates)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.tick_params(axis="x", rotation=45)

ax1.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.4)
ax1.margins(x=0)

# Right axis: ratio
ax2 = ax1.twinx()

ax2.plot(
    plot_df["date"],
    plot_df["linked_to_nominal_ratio"],
    color="#3A3A3A",
    linewidth=2.0,
    linestyle="-",
    label="Ratio linked/nominal (rhs)"
)

ax2.set_ylabel("", fontsize=12)

for spine in ax2.spines.values():
    spine.set_linewidth(1.0)
    spine.set_color("black")

ax2.tick_params(
    axis="y",
    which="major",
    direction="in",
    left=False,
    right=True,
    length=5,
    width=1.0
)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="upper left",
    ncol=1,
    frameon=True,
    fancybox=False,
    edgecolor="black",
    facecolor="white",
    framealpha=1.0
)

fig.tight_layout()
plt.show()

# --- EXPORT TO EXCEL
liquidity_df = plot_df.loc[:, ["date", "linked_to_nominal_ratio"]].copy()
liquidity_df = liquidity_df.rename(columns={"linked_to_nominal_ratio": "bid_ask_ratio"})

liquidity_df["date"] = liquidity_df["date"].dt.date  # optional, cleaner Excel format

with pd.ExcelWriter("price_factors.xlsx", engine="openpyxl", mode="w") as writer:
    liquidity_df.to_excel(writer, sheet_name="liquidity", index=False)

# endregion

# region TURNOVER (MONTHLY/DAILY? SWEDISH MARKET? UNIT?)

# ---- 1) PREPARE DATA

# Compute from full ILB sample start
turnover_df = turnover.loc[
    (turnover["Month"] >= pd.Timestamp("2003-11-30")) &
    (turnover["Month"] <= pd.Timestamp("2025-12-31")),
    ["Month", "GVB", "ILB"]
].copy()

turnover_df = turnover_df.rename(columns={
    "Month": "date",
    "GVB": "nominal_turnover",
    "ILB": "linked_turnover"
})

turnover_df = turnover_df.sort_values("date").reset_index(drop=True)

# Ratio: nominal / linked
turnover_df["turnover_ratio"] = np.where(
    turnover_df["linked_turnover"] > 0,
    turnover_df["nominal_turnover"] / turnover_df["linked_turnover"],
    np.nan
)

# 3-month moving average (computed BEFORE filtering)
turnover_df["turnover_ratio_ma3"] = turnover_df["turnover_ratio"].rolling(window=3).mean()

# Plot sample starts when MA is available
plot_turnover_df = turnover_df.loc[
    (turnover_df["date"] >= pd.Timestamp("2004-01-31")) &
    (turnover_df["date"] <= pd.Timestamp("2025-12-31"))
].copy()

# ---- 2) PLOT TURNOVER AND RATIO

fig, ax1 = plt.subplots(figsize=(13, 7))
fig.patch.set_facecolor("white")
ax1.set_facecolor("white")

# Left axis: turnover levels
ax1.plot(
    plot_turnover_df["date"],
    plot_turnover_df["nominal_turnover"],
    color="#4F86FF",
    linewidth=2.2,
    linestyle="-.",
    label="SGB turnover (lhs)"
)

ax1.plot(
    plot_turnover_df["date"],
    plot_turnover_df["linked_turnover"],
    color="#66BB6A",
    linewidth=2.2,
    linestyle="-.",
    label="SGBi turnover (lhs)"
)

ax1.set_xlabel("")
ax1.set_ylabel("mSEK", fontsize=12)

for spine in ax1.spines.values():
    spine.set_linewidth(1.0)
    spine.set_color("black")

ax1.tick_params(
    axis="both",
    which="major",
    direction="in",
    top=True,
    right=False,
    length=5,
    width=1.0
)

ax1.set_xlim(pd.Timestamp("2004-01-31"), pd.Timestamp("2025-12-31"))
ax1.set_xticks(tick_dates)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.tick_params(axis="x", rotation=45)

ax1.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.4)
ax1.margins(x=0)

# Right axis: ratio
ax2 = ax1.twinx()

ax2.plot(
    plot_turnover_df["date"],
    plot_turnover_df["turnover_ratio"],
    color="#C62828",
    linewidth=2.0,
    linestyle="--",
    label="Turnover ratio SGB/SGBi (rhs)"
)

ax2.plot(
    plot_turnover_df["date"],
    plot_turnover_df["turnover_ratio_ma3"],
    color="#000000",
    linewidth=2.0,
    linestyle="-",
    label="3m MA turnover ratio (rhs)"
)

ax2.set_ylabel("", fontsize=12)

for spine in ax2.spines.values():
    spine.set_linewidth(1.0)
    spine.set_color("black")

ax2.tick_params(
    axis="y",
    which="major",
    direction="in",
    left=False,
    right=True,
    length=5,
    width=1.0
)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="upper left",
    ncol=1,
    frameon=True,
    fancybox=False,
    edgecolor="black",
    facecolor="white",
    framealpha=1.0
)

fig.tight_layout()
plt.show()

# ---- 3) TURNOVER 

fig, ax1 = plt.subplots(figsize=(13, 7))
fig.patch.set_facecolor("white")
ax1.set_facecolor("white")

# Left axis: nominal turnover
ax1.plot(
    plot_turnover_df["date"],
    plot_turnover_df["nominal_turnover"],
    color="#4F86FF",
    linewidth=2.2,
    linestyle="-",
    label="SGB turnover (lhs)"
)

ax1.set_xlabel("")
ax1.set_ylabel("mSEK", fontsize=12)

for spine in ax1.spines.values():
    spine.set_linewidth(1.0)
    spine.set_color("black")

ax1.tick_params(
    axis="both",
    which="major",
    direction="in",
    top=True,
    right=False,
    length=5,
    width=1.0
)

ax1.set_xlim(pd.Timestamp("2004-01-31"), pd.Timestamp("2025-12-31"))
ax1.set_xticks(tick_dates)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.tick_params(axis="x", rotation=45)

ax1.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.4)
ax1.margins(x=0)

# Right axis: inflation-linked turnover
ax2 = ax1.twinx()

ax2.plot(
    plot_turnover_df["date"],
    plot_turnover_df["linked_turnover"],
    color="#66BB6A",
    linewidth=2.2,
    linestyle="-",
    label="SGBi turnover (rhs)"
)

ax2.set_ylabel("mSEK", fontsize=12)

for spine in ax2.spines.values():
    spine.set_linewidth(1.0)
    spine.set_color("black")

ax2.tick_params(
    axis="y",
    which="major",
    direction="in",
    left=False,
    right=True,
    length=5,
    width=1.0
)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="upper left",
    ncol=1,
    frameon=True,
    fancybox=False,
    edgecolor="black",
    facecolor="white",
    framealpha=1.0
)

fig.tight_layout()
plt.show()

# ---- 4) EXPORT 3M TURNOVER RATIO MA TO EXCEL SHEET "liquidity"

# Prepare monthly turnover ratio series
turnover_export = plot_turnover_df.loc[:, ["date", "turnover_ratio_ma3"]].copy()
turnover_export["year_month"] = turnover_export["date"].dt.to_period("M")

# Read existing liquidity sheet
liquidity_existing = pd.read_excel("price_factors.xlsx", sheet_name="liquidity")
liquidity_existing["date"] = pd.to_datetime(liquidity_existing["date"], errors="coerce")
liquidity_existing["year_month"] = liquidity_existing["date"].dt.to_period("M")

# Merge by year-month
liquidity_updated = pd.merge(
    liquidity_existing,
    turnover_export.loc[:, ["year_month", "turnover_ratio_ma3"]],
    on="year_month",
    how="left"
)

# Put the new column immediately to the right of bid_ask_ratio if that column exists
if "bid_ask_ratio" in liquidity_updated.columns:
    cols = list(liquidity_updated.columns)
    cols.remove("turnover_ratio_ma3")
    insert_pos = cols.index("bid_ask_ratio") + 1
    cols = cols[:insert_pos] + ["turnover_ratio_ma3"] + cols[insert_pos:]
    liquidity_updated = liquidity_updated.loc[:, cols]

# Drop helper key before export
liquidity_updated = liquidity_updated.drop(columns=["year_month"])

# Keep Excel date column clean
liquidity_updated["date"] = liquidity_updated["date"].dt.date

# Write back to the same sheet
with pd.ExcelWriter("price_factors.xlsx", engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    liquidity_updated.to_excel(writer, sheet_name="liquidity", index=False)

# endregion

# region YIELD FITTING ERRORS (RMSE) 

# ---- 1) PREPARE RMSE SERIES

rmse_start = pd.Timestamp("2004-01-30")
rmse_end   = pd.Timestamp("2025-12-31")

plot_nom_rmse = params_SGB.loc[
    (params_SGB["date"] >= rmse_start) &
    (params_SGB["date"] <= rmse_end),
    ["date", "rmse_yield_bp_ext"]
].copy()

plot_il_rmse = params_SGBIL.loc[
    (params_SGBIL["date"] >= rmse_start) &
    (params_SGBIL["date"] <= rmse_end),
    ["date", "rmse_yield_bp_ext"]
].copy()

plot_nom_rmse = plot_nom_rmse.rename(columns={"rmse_yield_bp_ext": "nominal_rmse_bp"})
plot_il_rmse = plot_il_rmse.rename(columns={"rmse_yield_bp_ext": "linked_rmse_bp"})

plot_rmse = pd.merge(
    plot_nom_rmse,
    plot_il_rmse,
    on="date",
    how="inner"
).sort_values("date").reset_index(drop=True)

# ---- 2) PLOT RMSE FOR NOMINAL AND LINKED

# Compute averages over the sample
avg_nominal_rmse = plot_rmse["nominal_rmse_bp"].mean()
avg_linked_rmse  = plot_rmse["linked_rmse_bp"].mean()

fig, ax = plt.subplots(figsize=(13, 7))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# Time series
ax.plot(
    plot_rmse["date"],
    plot_rmse["nominal_rmse_bp"],
    color="#4F86FF",
    linewidth=2.2,
    linestyle="-",
    label=f"SGB yield fitting error (avg: {avg_nominal_rmse:.2f} bp)"
)

ax.plot(
    plot_rmse["date"],
    plot_rmse["linked_rmse_bp"],
    color="#66BB6A",
    linewidth=2.2,
    linestyle="-",
    label=f"SGBi yield fitting error (avg: {avg_linked_rmse:.2f} bp)"
)

# Horizontal average lines
ax.axhline(
    avg_nominal_rmse,
    color="#4F86FF",
    linewidth=1.8,
    linestyle="--"
)

ax.axhline(
    avg_linked_rmse,
    color="#66BB6A",
    linewidth=1.8,
    linestyle="--"
)

ax.set_xlabel("")
ax.set_ylabel("RMSE (bp)", fontsize=12)

for spine in ax.spines.values():
    spine.set_linewidth(1.0)
    spine.set_color("black")

ax.tick_params(
    axis="both",
    which="major",
    direction="in",
    top=True,
    right=True,
    length=5,
    width=1.0
)

ax.set_xlim(rmse_start, rmse_end)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.tick_params(axis="x", rotation=45)

ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.4)
ax.margins(x=0)

ax.legend(
    loc="upper left",
    ncol=1,
    frameon=True,
    fancybox=False,
    edgecolor="black",
    facecolor="white",
    framealpha=1.0
)

fig.tight_layout()
plt.show()

# ---- 3) EXPORT LINKED RMSE TO "liquidity" SHEET

# Prepare monthly inflation-linked RMSE series
linked_rmse_export = params_SGBIL.loc[
    (params_SGBIL["date"] >= pd.Timestamp("2004-01-01")) &
    (params_SGBIL["date"] <= pd.Timestamp("2025-12-31")),
    ["date", "rmse_yield_bp_ext"]
].copy()

linked_rmse_export = linked_rmse_export.rename(
    columns={"rmse_yield_bp_ext": "linked_rmse_yield_bp"}
)

linked_rmse_export["year_month"] = linked_rmse_export["date"].dt.to_period("M")

# Read existing liquidity sheet
liquidity_df = pd.read_excel("price_factors.xlsx", sheet_name="liquidity")
liquidity_df["date"] = pd.to_datetime(liquidity_df["date"])
liquidity_df["year_month"] = liquidity_df["date"].dt.to_period("M")

# Merge by year-month
liquidity_df = pd.merge(
    liquidity_df,
    linked_rmse_export.loc[:, ["year_month", "linked_rmse_yield_bp"]],
    on="year_month",
    how="left"
)

# Drop helper column and write back
liquidity_df = liquidity_df.drop(columns="year_month")
liquidity_df["date"] = liquidity_df["date"].dt.date

with pd.ExcelWriter("price_factors.xlsx", engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    liquidity_df.to_excel(writer, sheet_name="liquidity", index=False)

# endregion

# region COMPOSITE LIQUIDITY

# ---- 1) IMPORT LIQUIDITY SHEET
liq = pd.read_excel("price_factors.xlsx", sheet_name="liquidity")
liq["date"] = pd.to_datetime(liq["date"], errors="coerce")

# Keep only the needed columns and sort
liq = liq.loc[:, ["date", "bid_ask_ratio", "turnover_ratio_ma3", "linked_rmse_yield_bp"]].copy()
liq = liq.sort_values("date").reset_index(drop=True)

# ---- 2) CLEAN / NUMERIC CONVERSION
base_cols = ["bid_ask_ratio", "turnover_ratio_ma3", "linked_rmse_yield_bp"]

for col in base_cols:
    liq[col] = pd.to_numeric(liq[col], errors="coerce")

# ---- 3) STANDARDIZE EACH SERIES
liq_std = liq.copy()

for col in base_cols:
    col_mean = liq_std[col].mean()
    col_std = liq_std[col].std()
    liq_std[col] = (liq_std[col] - col_mean) / col_std

# ---- 4) COMPUTE EQUAL-WEIGHT COMPOSITE
liq_std["composite_liquidity_raw"] = liq_std[base_cols].mean(axis=1)

# Ensure positivity as in the paper
composite_min = liq_std["composite_liquidity_raw"].min()
liq_std["composite_liquidity"] = liq_std["composite_liquidity_raw"] - composite_min

# ---- 5) PLOT STANDARDIZED INPUT SERIES (STACKED)
fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True)
fig.patch.set_facecolor("white")

color_blue   = "#1f3b73"
color_orange = "#d88a34"
color_green  = "#2f6b3f"

series_info = [
    ("bid_ask_ratio",        "Bid-ask ratio (std.)",         color_blue),
    ("linked_rmse_yield_bp", "SGBi yield RMSE (std.)",       color_orange),
    ("turnover_ratio_ma3",   "Turnover ratio, 3m MA (std.)", color_green),
]

for ax, (col, label, color) in zip(axes, series_info):
    ax.set_facecolor("white")

    ax.plot(
        liq_std["date"],
        liq_std[col],
        color=color,
        linewidth=2.4
    )

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_ylabel(label, fontsize=11)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("black")

    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        length=5,
        width=1.0
    )

    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.4)

axes[-1].set_xlim(pd.Timestamp("2004-01-01"), pd.Timestamp("2025-12-31"))
axes[-1].set_xticks([pd.Timestamp(f"{year}-01-01") for year in [2005, 2010, 2015, 2020, 2025]])
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axes[-1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()

# ---- 6) PLOT COMPOSITE LIQUIDITY INDEX
fig, ax = plt.subplots(figsize=(13, 7))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# Bright but still muted green
color_green = "#4c9a6a"

# Line
ax.plot(
    liq_std["date"],
    liq_std["composite_liquidity"],
    color=color_green,
    linewidth=2.5
)

# Filled area (from 0 up to the series)
ax.fill_between(
    liq_std["date"],
    0,
    liq_std["composite_liquidity"],
    color=color_green,
    alpha=0.25
)

ax.set_xlabel("")
ax.set_ylabel("Composite liquidity index", fontsize=12)

for spine in ax.spines.values():
    spine.set_linewidth(1.0)
    spine.set_color("black")

ax.tick_params(
    axis="both",
    which="major",
    direction="in",
    top=True,
    right=True,
    length=5,
    width=1.0
)

ax.set_xlim(pd.Timestamp("2004-01-01"), pd.Timestamp("2025-12-31"))
ax.set_xticks([pd.Timestamp(f"{year}-01-01") for year in [2005, 2010, 2015, 2020, 2025]])
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(axis="x", rotation=45)

ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.4)
ax.margins(x=0)

plt.tight_layout()
plt.show()

# ---- 7) EXPORT COMPOSITE BACK TO THE LIQUIDITY SHEET
# Merge only the final composite into the existing liquidity sheet
liquidity_export = pd.read_excel("price_factors.xlsx", sheet_name="liquidity")
liquidity_export["date"] = pd.to_datetime(liquidity_export["date"], errors="coerce")

composite_export = liq_std.loc[:, ["date", "composite_liquidity"]].copy()

# Remove existing composite_liquidity if already present, then merge fresh version
if "composite_liquidity" in liquidity_export.columns:
    liquidity_export = liquidity_export.drop(columns=["composite_liquidity"])

liquidity_export = pd.merge(
    liquidity_export,
    composite_export,
    on="date",
    how="left"
)

# Put composite_liquidity at the far right
liquidity_export["date"] = liquidity_export["date"].dt.date

with pd.ExcelWriter("price_factors.xlsx", engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    liquidity_export.to_excel(writer, sheet_name="liquidity", index=False)


# endregion

# region NOMINAL PRINCIPAL COMPONENTS

# ---- 1) PREPARE NOMINAL YIELD PANEL

start_date = pd.Timestamp("2004-01-01")
end_date   = pd.Timestamp("2025-12-31")

tick_years = [2005, 2010, 2015, 2020, 2025]
tick_dates = [pd.Timestamp(f"{year}-01-01") for year in tick_years]

nominal_pca_cols = [
    "y_3m",
    "y_6m",
    "y_12m",
    "y_24m",
    "y_36m",
    "y_48m",
    "y_60m",
    "y_72m",
    "y_84m",
    "y_96m",
    "y_108m",
    "y_120m",
]

maturity_months = [3, 6, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120]
maturity_years = [m / 12 for m in maturity_months]
maturity_labels_years = ["0.25", "0.5", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

nominal_pca_df = fitted_zero_SGB.loc[
    (fitted_zero_SGB["date"] >= start_date) &
    (fitted_zero_SGB["date"] <= end_date),
    ["date"] + nominal_pca_cols
].copy()

nominal_pca_df = nominal_pca_df.dropna().sort_values("date").reset_index(drop=True)

# X has one row per date and one column per maturity
X_nominal = nominal_pca_df[nominal_pca_cols].to_numpy()

# ---- 2) PCA

# sklearn PCA centers each maturity series automatically.
# The resulting:
# - components_ are the maturity loadings,
# - transformed scores are the factor realizations over time,
# - explained_variance_ratio_ gives the variance share.
pca_nominal = PCA(n_components=3)
scores_nominal = pca_nominal.fit_transform(X_nominal)

explained_var_ratio = pca_nominal.explained_variance_ratio_
loadings_nominal = pca_nominal.components_

nominal_pca_df["pc1"] = scores_nominal[:, 0]
nominal_pca_df["pc2"] = scores_nominal[:, 1]
nominal_pca_df["pc3"] = scores_nominal[:, 2]

# Flip signs so that PC1 has positive average loading
for i in range(3):
    if loadings_nominal[i, :].mean() < 0:
        loadings_nominal[i, :] = -loadings_nominal[i, :]
        nominal_pca_df[f"pc{i+1}"] = -nominal_pca_df[f"pc{i+1}"]

# Flip PC3 so that the middle maturities load more positively
# relative to the short and long ends.
loadings_nominal[2, :] = -loadings_nominal[2, :]
nominal_pca_df["pc3"] = -nominal_pca_df["pc3"]

print("Nominal yield PCA variance shares")
print(f"PC1: {explained_var_ratio[0]:.2%}")
print(f"PC2: {explained_var_ratio[1]:.2%}")
print(f"PC3: {explained_var_ratio[2]:.2%}")

# ---- 3) PLOT LOADINGS BY MATURITY
# The maturity loading profiles tell us how each principal component
# moves the yield curve across maturities

fig, ax = plt.subplots(figsize=(13, 7))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

ax.plot(
    maturity_years,
    loadings_nominal[0, :],
    color="#0A2342",
    linewidth=2.2,
    linestyle="-",
    marker="o",
    markersize=5,
    label=f"PC1: variance explained {explained_var_ratio[0]:.1%}"
)

ax.plot(
    maturity_years,
    loadings_nominal[1, :],
    color="#1A52A3",
    linewidth=2.2,
    linestyle="-",
    marker="o",
    markersize=5,
    label=f"PC2: variance explained {explained_var_ratio[1]:.1%}"
)

ax.plot(
    maturity_years,
    loadings_nominal[2, :],
    color="#9CBFFF",
    linewidth=2.2,
    linestyle="-",
    marker="o",
    markersize=5,
    label=f"PC3: variance explained {explained_var_ratio[2]:.1%}"
)

ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)

ax.set_xlabel("Maturity (years)", fontsize=12)
ax.set_ylabel("PCA loading", fontsize=12)

for spine in ax.spines.values():
    spine.set_linewidth(1.0)
    spine.set_color("black")

ax.tick_params(
    axis="both",
    which="major",
    direction="in",
    top=True,
    right=True,
    length=5,
    width=1.0
)

ax.set_xticks(maturity_years)
ax.set_xticklabels(maturity_labels_years, rotation=45)

ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.4)
ax.margins(x=0.02)

ax.legend(
    loc="upper right",
    ncol=1,
    frameon=True,
    fancybox=False,
    edgecolor="black",
    facecolor="white",
    framealpha=1.0
)

fig.tight_layout()
plt.show()

# ---- 4) PLOT EVOLUTION OF FIRST THREE PCs OVER TIME 
# These are the time series realizations of the three nominal pricing factors.

fig, ax = plt.subplots(figsize=(13, 7))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

ax.plot(
    nominal_pca_df["date"],
    nominal_pca_df["pc1"],
    color="#0A2342",
    linewidth=2.2,
    linestyle="-",
    label="PC1"
)

ax.plot(
    nominal_pca_df["date"],
    nominal_pca_df["pc2"],
    color="#1A52A3",
    linewidth=2.2,
    linestyle="-",
    label="PC2"
)

ax.plot(
    nominal_pca_df["date"],
    nominal_pca_df["pc3"],
    color="#9CBFFF",
    linewidth=2.2,
    linestyle="-",
    label="PC3"
)

ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)

ax.set_xlabel("")
ax.set_ylabel("Principal component score", fontsize=12)

for spine in ax.spines.values():
    spine.set_linewidth(1.0)
    spine.set_color("black")

ax.tick_params(
    axis="both",
    which="major",
    direction="in",
    top=True,
    right=True,
    length=5,
    width=1.0
)

ax.set_xlim(start_date, end_date)
ax.set_xticks(tick_dates)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(axis="x", rotation=45)

ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.4)
ax.margins(x=0)

ax.legend(
    loc="upper left",
    ncol=1,
    frameon=True,
    fancybox=False,
    edgecolor="black",
    facecolor="white",
    framealpha=1.0
)

fig.tight_layout()
plt.show()

# ---- 5) EXPORT NOMINAL PCA RESULTS TO EXCEL

# Date-by-date PCA scores
nominal_pc_export = nominal_pca_df.loc[:, ["date", "pc1", "pc2", "pc3"]].copy()
nominal_pc_export["date"] = nominal_pc_export["date"].dt.date

# Loadings by maturity plus variance explained
nominal_pca_loadings_export = pd.DataFrame({
    "maturity_years": maturity_years,
    "pc1_loading": loadings_nominal[0, :],
    "pc2_loading": loadings_nominal[1, :],
    "pc3_loading": loadings_nominal[2, :],
    "pc1_variance_explained": explained_var_ratio[0],
    "pc2_variance_explained": explained_var_ratio[1],
    "pc3_variance_explained": explained_var_ratio[2],
})

with pd.ExcelWriter("price_factors.xlsx", engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    nominal_pc_export.to_excel(writer, sheet_name="nominal PC", index=False)
    nominal_pca_loadings_export.to_excel(writer, sheet_name="nominal PCA loadings", index=False)

# endregion

# region LINKED ORHTOGONAL PRINCIPAL COMPONENTS

# ---- 1) PREPARE LINKED YIELD PANEL
# Paper-consistent maturity range for linked yields: 24m to 120m

linked_pca_cols = [
    "y_24m",
    "y_36m",
    "y_48m",
    "y_60m",
    "y_72m",
    "y_84m",
    "y_96m",
    "y_108m",
    "y_120m",
]

linked_maturity_months = [24, 36, 48, 60, 72, 84, 96, 108, 120]
linked_maturity_years = [m / 12 for m in linked_maturity_months]
linked_maturity_labels_years = ["2", "3", "4", "5", "6", "7", "8", "9", "10"]

linked_panel = fitted_zero_SGBIL.loc[
    (fitted_zero_SGBIL["date"] >= start_date) &
    (fitted_zero_SGBIL["date"] <= end_date),
    ["date"] + linked_pca_cols
].copy()

# ---- 2) PREPARE NOMINAL PCS + LIQUIDITY FACTOR
nominal_factors = nominal_pca_df.loc[:, ["date", "pc1", "pc2", "pc3"]].copy()

liquidity_factor = liq_std.loc[:, ["date", "composite_liquidity"]].copy()

# ---- 3) MERGE EVERYTHING ON DATE
linked_reg_df = pd.merge(linked_panel, nominal_factors, on="date", how="inner")
linked_reg_df = pd.merge(linked_reg_df, liquidity_factor, on="date", how="inner")

linked_reg_df = linked_reg_df.dropna().sort_values("date").reset_index(drop=True)

# ---- 4) ORTHOGONALIZE LINKED YIELDS
# Regress each linked yield maturity on:
# constant + nominal PC1 + nominal PC2 + nominal PC3 + liquidity factor
# Then keep residuals.

X = linked_reg_df[["pc1", "pc2", "pc3", "composite_liquidity"]].to_numpy()
X = np.column_stack([np.ones(len(X)), X])   # add intercept

residuals = np.empty((len(linked_reg_df), len(linked_pca_cols)))

for j, col in enumerate(linked_pca_cols):
    y = linked_reg_df[col].to_numpy()
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    residuals[:, j] = y - y_hat

residual_df = pd.DataFrame(residuals, columns=[f"{c}_resid" for c in linked_pca_cols])
residual_df.insert(0, "date", linked_reg_df["date"].values)

# ---- 5) PCA ON RESIDUALS
pca_linked = PCA(n_components=2)
scores_linked = pca_linked.fit_transform(residuals)

explained_var_ratio_linked = pca_linked.explained_variance_ratio_
loadings_linked = pca_linked.components_.copy()

linked_pca_df = linked_reg_df.loc[:, ["date"]].copy()
linked_pca_df["real_pc1"] = scores_linked[:, 0]
linked_pca_df["real_pc2"] = scores_linked[:, 1]

print("Orthogonal linked yield PCA variance shares")
print(f"Real PC1: {explained_var_ratio_linked[0]:.2%}")
print(f"Real PC2: {explained_var_ratio_linked[1]:.2%}")

# ---- 7) PLOT LOADINGS BY MATURITY
fig, ax = plt.subplots(figsize=(13, 7))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

ax.plot(
    linked_maturity_years,
    loadings_linked[0, :],
    color="#1f3b73",
    linewidth=2.2,
    linestyle="-",
    marker="o",
    markersize=5,
    label=f"Real PC1: variance explained {explained_var_ratio_linked[0]:.1%}"
)

ax.plot(
    linked_maturity_years,
    loadings_linked[1, :],
    color="#d88a34",
    linewidth=2.2,
    linestyle="-",
    marker="o",
    markersize=5,
    label=f"Real PC2: variance explained {explained_var_ratio_linked[1]:.1%}"
)

ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)

ax.set_xlabel("Maturity (years)", fontsize=12)
ax.set_ylabel("PCA loading", fontsize=12)

for spine in ax.spines.values():
    spine.set_linewidth(1.0)
    spine.set_color("black")

ax.tick_params(
    axis="both",
    which="major",
    direction="in",
    top=True,
    right=True,
    length=5,
    width=1.0
)

ax.set_xticks(linked_maturity_years)
ax.set_xticklabels(linked_maturity_labels_years, rotation=45)

ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.4)
ax.margins(x=0.02)

ax.legend(
    loc="upper right",
    ncol=1,
    frameon=True,
    fancybox=False,
    edgecolor="black",
    facecolor="white",
    framealpha=1.0
)

fig.tight_layout()
plt.show()

# ---- 8) PLOT EVOLUTION OF THE TWO ORTHOGONAL LINKED PCs
fig, ax = plt.subplots(figsize=(13, 7))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

ax.plot(
    linked_pca_df["date"],
    linked_pca_df["real_pc1"],
    color="#1f3b73",
    linewidth=2.2,
    linestyle="-",
    label="Real PC1"
)

ax.plot(
    linked_pca_df["date"],
    linked_pca_df["real_pc2"],
    color="#d88a34",
    linewidth=2.2,
    linestyle="-",
    label="Real PC2"
)

ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)

ax.set_xlabel("")
ax.set_ylabel("Principal component score", fontsize=12)

for spine in ax.spines.values():
    spine.set_linewidth(1.0)
    spine.set_color("black")

ax.tick_params(
    axis="both",
    which="major",
    direction="in",
    top=True,
    right=True,
    length=5,
    width=1.0
)

ax.set_xlim(start_date, end_date)
ax.set_xticks(tick_dates)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(axis="x", rotation=45)

ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.4)
ax.margins(x=0)

ax.legend(
    loc="upper left",
    ncol=1,
    frameon=True,
    fancybox=False,
    edgecolor="black",
    facecolor="white",
    framealpha=1.0
)

fig.tight_layout()
plt.show()

# ---- 9) EXPORT TO EXCEL
linked_pc_export = linked_pca_df.loc[:, ["date", "real_pc1", "real_pc2"]].copy()
linked_pc_export["date"] = linked_pc_export["date"].dt.date

linked_pca_loadings_export = pd.DataFrame({
    "maturity_years": linked_maturity_years,
    "real_pc1_loading": loadings_linked[0, :],
    "real_pc2_loading": loadings_linked[1, :],
    "real_pc1_variance_explained": explained_var_ratio_linked[0],
    "real_pc2_variance_explained": explained_var_ratio_linked[1],
})

linked_pc_export = linked_pca_df.loc[:, ["date", "real_pc1", "real_pc2"]].copy()
linked_pc_export["date"] = linked_pc_export["date"].dt.date

with pd.ExcelWriter("price_factors.xlsx", engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    linked_pc_export.to_excel(writer, sheet_name="linked PCA", index=False)
    linked_pca_loadings_export.to_excel(writer, sheet_name="linked PCA loadings", index=False)

# endregion

# region LINKED YIELD R2 SUMMARY

# ---- 1) PREPARE LINKED YIELD PANEL
linked_yield_cols = [
    "y_24m",
    "y_36m",
    "y_48m",
    "y_60m",
    "y_72m",
    "y_84m",
    "y_96m",
    "y_108m",
    "y_120m",
]

linked_r2_df = fitted_zero_SGBIL.loc[
    (fitted_zero_SGBIL["date"] >= start_date) &
    (fitted_zero_SGBIL["date"] <= end_date),
    ["date"] + linked_yield_cols
].copy()

linked_r2_df = pd.merge(
    linked_r2_df,
    nominal_pca_df.loc[:, ["date", "pc1", "pc2", "pc3"]],
    on="date",
    how="inner"
)

linked_r2_df = pd.merge(
    linked_r2_df,
    liq_std.loc[:, ["date", "composite_liquidity"]],
    on="date",
    how="inner"
)

linked_r2_df = pd.merge(
    linked_r2_df,
    linked_pca_df.loc[:, ["date", "real_pc1", "real_pc2"]],
    on="date",
    how="inner"
)

linked_r2_df = linked_r2_df.dropna().sort_values("date").reset_index(drop=True)

# ---- 2) HELPER FUNCTION
# Runs one OLS per linked-yield maturity and pools SSR and SST across maturities
def compute_panel_r2(df, y_cols, x_cols):
    X = df[x_cols].to_numpy()
    X = np.column_stack([np.ones(len(X)), X])  # intercept

    ssr_total = 0.0
    sst_total = 0.0

    for col in y_cols:
        y = df[col].to_numpy()
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X @ beta
        resid = y - y_hat

        ssr_total += np.sum(resid ** 2)
        sst_total += np.sum((y - y.mean()) ** 2)

    r2 = 1.0 - ssr_total / sst_total
    return r2

# ---- 3) COMPUTE R² BY SPECIFICATION
r2_spec1 = compute_panel_r2(
    linked_r2_df,
    linked_yield_cols,
    ["pc1", "pc2", "pc3"]
)

r2_spec2 = compute_panel_r2(
    linked_r2_df,
    linked_yield_cols,
    ["pc1", "pc2", "pc3", "composite_liquidity"]
)

r2_spec3 = compute_panel_r2(
    linked_r2_df,
    linked_yield_cols,
    ["pc1", "pc2", "pc3", "composite_liquidity", "real_pc1", "real_pc2"]
)

print("R² for linked yields by specification")
print(f"1) Nominal PCs only:                 {r2_spec1:.2%}")
print(f"2) Nominal PCs + liquidity:          {r2_spec2:.2%}")
print(f"3) Nominal PCs + liquidity + real PCs: {r2_spec3:.2%}")

# ---- 4) EXPORT SUMMARY TO EXCEL
linked_r2_summary = pd.DataFrame({
    "specification": [
        "PC1, PC2, PC3",
        "PC1, PC2, PC3, Liquidity",
        "PC1, PC2, PC3, Liquidity, Orthogonal PC1, Orthogonal PC2"
    ],
    "r_squared": [
        r2_spec1,
        r2_spec2,
        r2_spec3
    ]
})

with pd.ExcelWriter("price_factors.xlsx", engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    linked_r2_summary.to_excel(writer, sheet_name="linked yield R2", index=False)

# endregion
