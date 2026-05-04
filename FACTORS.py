import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.decomposition import PCA as _PCA
 
# region IMPORTS
 
DIR_NOM  = "../Curve estimation/NOMINAL_CURVE_ESTIMATION"
DIR_IL   = "../Curve estimation/LINKER_CURVE_ESTIMATION"
DIR_CURV = "../Curve estimation"
 
# ---- Curve fit parameters
raw_sgb_params = pd.read_excel(f"{DIR_NOM}/zero_yields_SGB.xlsx",   sheet_name="fit_params")
raw_il_params  = pd.read_excel(f"{DIR_IL}/zero_yields_SGBIL.xlsx",  sheet_name="fit_params")
 
# ---- Zero-coupon yield panels
raw_sgb_yields = pd.read_excel(f"{DIR_NOM}/zero_yields_SGB.xlsx",   sheet_name="zero_yields")
raw_il_yields  = pd.read_excel(f"{DIR_IL}/zero_yields_SGBIL.xlsx",  sheet_name="zero_yields")
 
# ---- Bid-ask spread summaries
raw_nom_ba = pd.read_excel(f"{DIR_NOM}/nominal_bid_ask.xlsx", sheet_name="bid_ask_summary")
raw_il_ba  = pd.read_excel(f"{DIR_IL}/real_bid_ask.xlsx",     sheet_name="bid_ask_summary")
 
# ---- KPI
raw_kpi = pd.read_excel(f"{DIR_CURV}/kpi_data.xlsx", sheet_name="basår 1980")
 
# ---- VIX
raw_vix = pd.read_excel("VIX.xlsx", sheet_name="Daily, Close")
 
# ---- Riksgälden turnover (dates stored as "YYYY-MM" strings)
raw_rg_sw = pd.read_excel("turnover_riksgälden.xlsx", sheet_name="svenska investerare")
raw_rg_fo = pd.read_excel("turnover_riksgälden.xlsx", sheet_name="utländska investerare")
 
# ---- Standardise date columns
raw_sgb_params["date"] = pd.to_datetime(raw_sgb_params["date"])
raw_il_params["date"]  = pd.to_datetime(raw_il_params["date"])
raw_sgb_yields["date"] = pd.to_datetime(raw_sgb_yields["date"])
raw_il_yields["date"]  = pd.to_datetime(raw_il_yields["date"])
raw_nom_ba["date"]     = pd.to_datetime(raw_nom_ba["date"])
raw_il_ba["date"]      = pd.to_datetime(raw_il_ba["date"])
raw_kpi["date"]        = pd.to_datetime(raw_kpi["date"], errors="coerce")
raw_vix["date"]        = pd.to_datetime(raw_vix["observation_date"])
 
def _parse_rg_sheet(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = ["date", "instrument", "turnover_mdr_sek"]
    df["date"] = pd.to_datetime(
        df["date"].astype(str).str.strip(), format="%Y-%m", errors="coerce"
    )
    df["turnover_mdr_sek"] = pd.to_numeric(df["turnover_mdr_sek"], errors="coerce")
    return df.dropna(subset=["date"]).reset_index(drop=True)
 
rg_sw = _parse_rg_sheet(raw_rg_sw)
rg_fo = _parse_rg_sheet(raw_rg_fo)
 
# ---- Time periods spanned by each dataset
print(f"  SGB  fit_params  : {raw_sgb_params['date'].min().date()} -> {raw_sgb_params['date'].max().date()}")
print(f"  SGBi fit_params  : {raw_il_params['date'].min().date()} -> {raw_il_params['date'].max().date()}")
print(f"  SGB  zero yields : {raw_sgb_yields['date'].min().date()} -> {raw_sgb_yields['date'].max().date()}")
print(f"  SGBi zero yields : {raw_il_yields['date'].min().date()} -> {raw_il_yields['date'].max().date()}")
print(f"  Nominal bid-ask  : {raw_nom_ba['date'].min().date()} -> {raw_nom_ba['date'].max().date()}")
print(f"  Real    bid-ask  : {raw_il_ba['date'].min().date()} -> {raw_il_ba['date'].max().date()}")
print(f"  KPI              : {raw_kpi['date'].min().date()} -> {raw_kpi['date'].max().date()}")
print(f"  VIX              : {raw_vix['date'].min().date()} -> {raw_vix['date'].max().date()}")
print(f"  Riksgälden (SW)  : {rg_sw['date'].min().date()} -> {rg_sw['date'].max().date()}")
print(f"  Riksgälden (FO)  : {rg_fo['date'].min().date()} -> {rg_fo['date'].max().date()}")
 
# endregion

# region CONFIGURATION
 
# ---- Sample window
SAMPLE_START = pd.Timestamp("2004-01-01")
SAMPLE_END   = pd.Timestamp("2025-12-31")
 
# Pre-sample horizon for moving-average warmup (13 months before sample start)
PRE_SAMPLE = SAMPLE_START - pd.DateOffset(months=13)
 
# ---- Output
PLOT_DIR  = "factor_plots"
FACTOR_XL = "Factors.xlsx"
os.makedirs(PLOT_DIR, exist_ok=True)
 
# ---- Figure settings
FIG_WIDTH  = 5.5
FIG_HEIGHT = 3.4
FIG_DPI    = 300
FIG_FORMAT = "pdf"
 
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":          11,
    "axes.labelsize":     11,
    "axes.titlesize":     11,
    "legend.fontsize":    10,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "axes.linewidth":     0.8,
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "xtick.major.size":   4.0,
    "ytick.major.size":   4.0,
})
 
# ---- Colors 
C_FILL   = "#1A5EA8"   # fill/shading (semi-transparent)
C_ZERO   = "#666666"   # zero line
 
# ---- X-axis tick dates (shared across all time-series plots)
TICK_YEARS = [2004, 2008, 2012, 2016, 2020, 2024]
TICK_DATES = [pd.Timestamp(f"{y}-01-01") for y in TICK_YEARS]
 
# ---- Shared plotting helpers
 
def _style_ax(ax):
    """Uniform axis style: white background, inward ticks on all four sides,
    horizontal grid. Matches the estimation scripts' visual language."""
    ax.set_facecolor("white")
    ax.yaxis.grid(True, color="#CCCCCC", linewidth=0.5, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("black")
    ax.tick_params(axis="both", which="major", direction="in",
                   top=True, right=True, pad=5)
 
 
def _format_date_axis(ax, xlim=None):
    """Apply standard date-axis formatting to ax.
    Uses TICK_DATES by default; pass xlim=(t0, t1) to override the x-limits."""
    ax.set_xticks(TICK_DATES)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=0)
    if xlim is not None:
        ax.set_xlim(*xlim)
    else:
        ax.set_xlim(SAMPLE_START, SAMPLE_END)
    ax.margins(x=0)
 
 
def _save_fig(fig, stem: str):
    """Save fig to PLOT_DIR/<stem>.<FIG_FORMAT> at FIG_DPI."""
    path = os.path.join(PLOT_DIR, f"{stem}.{FIG_FORMAT}")
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {path}")
 
# endregion

# region BEI PLOT
# Source: zero_yields_SGB.xlsx and zero_yields_SGBIL.xlsx, sheet "zero_yields".
# Maturities: 1y (12m), 2y (24m), 5y (60m), 10y (120m).
 
BEI_MATS   = [12, 24, 60, 120]
BEI_LABELS = {12: "1y", 24: "2y", 60: "5y", 120: "10y"}
BEI_COLORS = ["#A8C8E8", "#4D96C0", "#1A5EA8", "#0D2B4E"]   # light -> dark blue
 
# ---- Align on year_month
_nom = raw_sgb_yields.copy()
_il  = raw_il_yields.copy()
 
_nom["year_month"] = _nom["date"].dt.to_period("M")
_il["year_month"]  = _il["date"].dt.to_period("M")
 
_nom_cols = ["year_month", "date"] + [f"y_{m}m" for m in BEI_MATS]
_il_cols  = ["year_month"]         + [f"y_{m}m" for m in BEI_MATS]
 
bei = (
    pd.merge(
        _nom[_nom_cols],
        _il[_il_cols].rename(columns={f"y_{m}m": f"il_y_{m}m" for m in BEI_MATS}),
        on="year_month", how="inner",
    )
    .loc[lambda d: d["date"].between(SAMPLE_START, SAMPLE_END)]
    .sort_values("date")
    .reset_index(drop=True)
)
 
for m in BEI_MATS:
    bei[f"bei_{m}m"] = bei[f"y_{m}m"] - bei[f"il_y_{m}m"]
 
print(f"\n  BEI panel : {bei['date'].min().date()} -> {bei['date'].max().date()}  "
      f"(n={len(bei)})")
for m in BEI_MATS:
    col = f"bei_{m}m"
    print(f"  BEI {BEI_LABELS[m]:>3s}  : mean {bei[col].mean()*100:+.2f}%  "
          f"std {bei[col].std()*100:.2f}%  "
          f"range [{bei[col].min()*100:.2f}%, {bei[col].max()*100:.2f}%]")
 
# ---- Plot
fig, ax = plt.subplots(figsize=(FIG_WIDTH * 1.4, FIG_HEIGHT))
fig.patch.set_facecolor("white")
ax.axhline(0, color=C_ZERO, lw=0.8, ls="-", zorder=1)

for m, color in zip(BEI_MATS, BEI_COLORS):
    ax.plot(bei["date"], bei[f"bei_{m}m"] * 100,
            color=color, lw=1.6, zorder=3, label=BEI_LABELS[m])

ax.set_ylabel("BEI (%)")
ax.set_xlim(bei["date"].min(), bei["date"].max())
ax.set_ylim(-3, 5)
ax.set_yticks(range(-2, 5))
ax.set_yticklabels([str(v) for v in range(-2, 5)])


_style_ax(ax)
ax.legend(loc="upper left", frameon=True, fancybox=False,
          edgecolor="#AAAAAA", facecolor="white", framealpha=1.0,
          borderpad=0.6, handlelength=2.2, ncol=4)
plt.tight_layout(pad=0.6)
_save_fig(fig, "break_even_inflation")
plt.show()
plt.close(fig)
 
# endregion

# ---- LIQUIDITY

    # region BID-ASK SPREADS
# Source: nominal_bid_ask.xlsx and real_bid_ask.xlsx, sheet "bid_ask_summary".
# A 1 bp floor is imposed on the nominal spread before computing the ratio.

C_BA_NOM   = "#1A5EA8" 
C_BA_IL    = "#2E7D54"   
C_BA_RATIO = "#2C3E50"  

# ---- Align on year_month
_nom_ba = raw_nom_ba.copy()
_il_ba  = raw_il_ba.copy()

_nom_ba["year_month"] = _nom_ba["date"].dt.to_period("M")
_il_ba["year_month"]  = _il_ba["date"].dt.to_period("M")

ba = (
    pd.merge(
        _nom_ba[["year_month", "date", "mean_1_10y", "mean_2_10y"]],
        _il_ba[["year_month", "mean_1_10y", "mean_2_10y"]]
            .rename(columns={"mean_1_10y": "il_mean_1_10y",
                              "mean_2_10y": "il_mean_2_10y"}),
        on="year_month", how="inner",
    )
    .loc[lambda d: d["date"].between(SAMPLE_START, SAMPLE_END)]
    .sort_values("date")
    .reset_index(drop=True)
)

ba["ratio_1_10y"] = ba["il_mean_1_10y"] / ba["mean_1_10y"].clip(lower=1.0)
ba["ratio_2_10y"] = ba["il_mean_2_10y"] / ba["mean_2_10y"].clip(lower=1.0)

print(f"\n  Bid-ask panel : {ba['date'].min().date()} -> "
      f"{ba['date'].max().date()}  (n={len(ba)})")

def _plot_bid_ask(nom_col: str, il_col: str, ratio_col: str, stem: str):
    fig, ax1 = plt.subplots(figsize=(FIG_WIDTH * 1.4, FIG_HEIGHT))
    fig.patch.set_facecolor("white")
    ax2 = ax1.twinx()

    ax1.plot(ba["date"], ba[nom_col],   color=C_BA_NOM,   lw=1.8, ls="-", zorder=3,
             label="SGB")
    ax1.plot(ba["date"], ba[il_col],    color=C_BA_IL,    lw=1.8, ls="-", zorder=2,
             label="SGBi")
    ax2.plot(ba["date"], ba[ratio_col], color=C_BA_RATIO, lw=1.8, ls=":", zorder=1,
             label="Ratio")

    ax1.set_ylabel("Bid\u2013ask spread (bp)")
    ax2.set_ylabel("Ratio")
    ax1.set_xlim(ba["date"].min(), ba["date"].max())
    ax1.set_ylim(0, 40)
    ax1.set_yticks([5, 10, 15, 20, 25, 30, 35])
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax2.set_ylim(0, 8)
    ax2.set_yticks([1, 2, 3, 4, 5, 6, 7])
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))

    _style_ax(ax1)
    _style_ax(ax2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="upper left", frameon=True, fancybox=False,
               edgecolor="#AAAAAA", facecolor="white", framealpha=1.0,
               borderpad=0.6, handlelength=2.2)

    plt.tight_layout(pad=0.6)
    _save_fig(fig, stem)
    plt.show()
    plt.close(fig)

_plot_bid_ask("mean_1_10y", "il_mean_1_10y", "ratio_1_10y", "bid_ask_1_10y")
_plot_bid_ask("mean_2_10y", "il_mean_2_10y", "ratio_2_10y", "bid_ask_2_10y")

# endregion

    # region TURNOVER RATIO
# Source: turnover_riksgälden.xlsx, sheets "svenska investerare" and
#         "utländska investerare".
# Construction:
#   1. Pivot long-format data to wide (date | nom_turn | il_turn) per investor group.
#   2. Sum SW + FO turnover.
#   3. Apply 12-month rolling mean to turnover levels and ratio (min_periods=6).
#   4. Data loaded from PRE_SAMPLE so the MA is warmed up by SAMPLE_START.

C_RG_SW   = "#1A5EA8"   
C_RG_FO   = "#2E7D54"    
C_RG_SWFO = "#2C3E50"   

IL_LABEL  = "Reala statsobligationer"
NOM_LABEL = "Statsobligationer"


def _pivot_rg(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot long Riksgälden sheet to wide (date | nom_turn | il_turn)."""
    il  = (df.loc[df["instrument"] == IL_LABEL]
             .rename(columns={"turnover_mdr_sek": "il_turn"})
             [["date", "il_turn"]])
    nom = (df.loc[df["instrument"] == NOM_LABEL]
             .rename(columns={"turnover_mdr_sek": "nom_turn"})
             [["date", "nom_turn"]])
    return (pd.merge(il, nom, on="date", how="outer")
              .sort_values("date").reset_index(drop=True))


def _build_ratio_ma(wide: pd.DataFrame) -> pd.DataFrame:
    """Compute 12-month MA of turnover levels and ratio."""
    df = wide.loc[wide["date"] >= PRE_SAMPLE].copy()
    df["nom_turn_ma12"] = df["nom_turn"].rolling(window=12, min_periods=6).mean()
    df["il_turn_ma12"]  = df["il_turn"].rolling(window=12, min_periods=6).mean()
    df["ratio"] = np.where(
        df["il_turn"] > 0,
        df["nom_turn"] / df["il_turn"],
        np.nan,
    )
    df["ratio_ma12"] = df["ratio"].rolling(window=12, min_periods=6).mean()
    return df


# ---- Build combined SW + FO wide frame
rg_sw_wide = _pivot_rg(rg_sw)
rg_fo_wide = _pivot_rg(rg_fo)

rg_swfo = pd.merge(rg_sw_wide, rg_fo_wide, on="date", how="outer",
                   suffixes=("_sw", "_fo"))
rg_swfo["il_turn"]  = rg_swfo["il_turn_sw"].fillna(0)  + rg_swfo["il_turn_fo"].fillna(0)
rg_swfo["nom_turn"] = rg_swfo["nom_turn_sw"].fillna(0) + rg_swfo["nom_turn_fo"].fillna(0)
rg_swfo = rg_swfo[["date", "nom_turn", "il_turn"]].sort_values("date").reset_index(drop=True)

rg_swfo_ratio = _build_ratio_ma(rg_swfo)

# ---- Restrict to sample window
rg_swfo_s = (rg_swfo_ratio
             .loc[rg_swfo_ratio["date"].between(SAMPLE_START, SAMPLE_END)]
             .reset_index(drop=True))

print(f"\n  Turnover panel : {rg_swfo_s['date'].min().date()} -> "
      f"{rg_swfo_s['date'].max().date()}  (n={len(rg_swfo_s)})")
print(f"  Nominal turnover MA12 : mean {rg_swfo_s['nom_turn_ma12'].mean():.1f} Mdr SEK")
print(f"  IL turnover      MA12 : mean {rg_swfo_s['il_turn_ma12'].mean():.1f} Mdr SEK")
print(f"  Ratio            MA12 : mean {rg_swfo_s['ratio_ma12'].mean():.2f}")

# ---- Plot
fig, ax1 = plt.subplots(figsize=(FIG_WIDTH * 1.4, FIG_HEIGHT))
fig.patch.set_facecolor("white")
ax2 = ax1.twinx()

ax1.plot(rg_swfo_s["date"], rg_swfo_s["nom_turn_ma12"], color=C_RG_SW, ls="-", lw=1.8,
         zorder=3, label="SGB")
ax1.plot(rg_swfo_s["date"], rg_swfo_s["il_turn_ma12"],  color=C_RG_FO, ls="-", lw=1.8,
         zorder=2, label="SGBi")
ax2.plot(rg_swfo_s["date"], rg_swfo_s["ratio_ma12"],    color=C_RG_SWFO, ls="-.", lw=1.8,
         zorder=1, label="Ratio")

ax1.set_ylabel("Turnover (Mdr SEK), 12m MA")
ax2.set_ylabel("Ratio")
ax1.set_xlim(rg_swfo_s["date"].min(), rg_swfo_s["date"].max())
ax1.set_ylim(0, 20)
ax1.set_yticks([2, 4, 6, 8, 10, 12, 14, 16, 18])
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))
ax2.set_ylim(5, 25)
ax2.set_yticks([7, 9, 11, 13, 15, 17, 19, 21, 23])
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))

_style_ax(ax1)
_style_ax(ax2)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
           loc="upper right", frameon=True, fancybox=False,
           edgecolor="#AAAAAA", facecolor="white", framealpha=1.0,
           borderpad=0.6, handlelength=2.2)

plt.tight_layout(pad=0.6)
_save_fig(fig, "turnover_ratio")
plt.show()
plt.close(fig)

# endregion

    # region YIELD FITTING ERRORS (RMSE)
# Source: fit_params sheets of zero_yields_SGB.xlsx and zero_yields_SGBIL.xlsx.
# Extended-fit yield RMSE (bp) for nominal and inflation-linked curves.

C_RMSE_NOM = "#0D2B4E" 
C_RMSE_IL  = "#C0392B"  

nom_rmse = (
    raw_sgb_params
    .loc[raw_sgb_params["date"].between(SAMPLE_START, SAMPLE_END),
         ["date", "rmse_yield_bp_ext"]]
    .rename(columns={"rmse_yield_bp_ext": "nom_rmse_bp"})
    .sort_values("date").reset_index(drop=True)
)

il_rmse = (
    raw_il_params
    .loc[raw_il_params["date"].between(SAMPLE_START, SAMPLE_END),
         ["date", "rmse_yield_bp_ext"]]
    .rename(columns={"rmse_yield_bp_ext": "il_rmse_bp"})
    .sort_values("date").reset_index(drop=True)
)

print(f"\n  Nominal RMSE : mean {nom_rmse['nom_rmse_bp'].mean():.2f} bp  "
      f"n={len(nom_rmse)}")
print(f"  IL RMSE      : mean {il_rmse['il_rmse_bp'].mean():.2f} bp  "
      f"n={len(il_rmse)}")

# ---- Plot
fig, ax = plt.subplots(figsize=(FIG_WIDTH * 1.4, FIG_HEIGHT))
fig.patch.set_facecolor("white")

ax.plot(nom_rmse["date"], nom_rmse["nom_rmse_bp"], color=C_RMSE_NOM, lw=1.8,
        zorder=3, label="Nominal")
ax.plot(il_rmse["date"],  il_rmse["il_rmse_bp"],   color=C_RMSE_IL,  lw=1.8,
        zorder=3, label="Inflation-linked")

ax.set_ylabel("Yield RMSE (bp)")
ax.set_xlim(nom_rmse["date"].min(), nom_rmse["date"].max())
ax.set_ylim(0, 15)
ax.set_yticks([2, 4, 6, 8, 10, 12, 14])
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))
_style_ax(ax)
ax.legend(loc="upper left", frameon=True, fancybox=False,
          edgecolor="#AAAAAA", facecolor="white", framealpha=1.0,
          borderpad=0.6, handlelength=2.2)

plt.tight_layout(pad=0.6)
_save_fig(fig, "yield_rmse")
plt.show()
plt.close(fig)

# endregion

    # region VIX
# Source: VIX.xlsx, sheet "Daily, Close"
# Two series: monthly mean and end-of-month (last observation) VIX.

C_VIX_MEAN = "#C0392B"   
C_VIX_EOM  = "#0D2B4E"   

vix_daily = raw_vix[["date", "VIXCLS"]].copy()
vix_daily["VIXCLS"]     = pd.to_numeric(vix_daily["VIXCLS"], errors="coerce")
vix_daily               = vix_daily.dropna(subset=["VIXCLS"])
vix_daily["year_month"] = vix_daily["date"].dt.to_period("M")

vix_mean = (
    vix_daily
    .loc[vix_daily["date"].between(SAMPLE_START, SAMPLE_END)]
    .groupby("year_month", as_index=False)
    .agg(date=("date", "last"), vix_mean=("VIXCLS", "mean"))
    .sort_values("date").reset_index(drop=True)
)

vix_eom = (
    vix_daily
    .loc[vix_daily["date"].between(SAMPLE_START, SAMPLE_END)]
    .groupby("year_month", as_index=False)
    .last()
    .rename(columns={"VIXCLS": "vix_eom"})
    [["date", "vix_eom"]]
    .sort_values("date").reset_index(drop=True)
)

print(f"\n  VIX mean (monthly avg) : mean {vix_mean['vix_mean'].mean():.2f}  "
      f"n={len(vix_mean)}")
print(f"  VIX EOM                : mean {vix_eom['vix_eom'].mean():.2f}  "
      f"n={len(vix_eom)}")

# ---- Plot
fig, ax = plt.subplots(figsize=(FIG_WIDTH * 1.4, FIG_HEIGHT))
fig.patch.set_facecolor("white")

ax.plot(vix_mean["date"], vix_mean["vix_mean"], color=C_VIX_MEAN, lw=1.8, ls=":",
        zorder=3, label="Monthly mean")
ax.plot(vix_eom["date"],  vix_eom["vix_eom"],  color=C_VIX_EOM,  lw=1.8, ls="-",
        zorder=2, label="End of month")

#ax.set_ylabel("VIX")
ax.set_xlim(vix_mean["date"].min(), vix_mean["date"].max())
ax.set_ylim(0, 70)
ax.set_yticks([0, 10, 20, 30, 40, 50, 60])
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))
_style_ax(ax)
ax.legend(loc="upper right", frameon=True, fancybox=False,
          edgecolor="#AAAAAA", facecolor="white", framealpha=1.0,
          borderpad=0.6, handlelength=2.2)

plt.tight_layout(pad=0.6)
_save_fig(fig, "vix")
plt.show()
plt.close(fig)

# endregion

    # region COMPOSITE LIQUIDITY FACTORS
# Three liquidity indexes built from standardised inputs:
#
#   Liq1 : VIX eom
#   Liq2 : avg(VIX eom, nominal bid-ask 1-10y)
#   Liq3 : avg(VIX eom, nominal bid-ask 1-10y, nominal RMSE)
#
# Construction:
#   1. Standardise each input to mean=0, std=1 over the sample window.
#   2. Average the standardised inputs with equal weights.
#   3. Enforce positivity by adding back the absolute minimum.

C_COMP = "#2C3E50"   
C_STD  = "#777777"   


def _standardise(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()


def _enforce_positivity(s: pd.Series) -> pd.Series:
    return s + abs(s.min()) if s.min() < 0 else s


# ---- Align inputs on year_month (inner join)
_vix_m = vix_eom.copy()
_vix_m["year_month"] = _vix_m["date"].dt.to_period("M")

_ba_m = ba[["date", "mean_1_10y"]].copy()
_ba_m["year_month"] = _ba_m["date"].dt.to_period("M")

_rmse_m = nom_rmse.copy()
_rmse_m["year_month"] = _rmse_m["date"].dt.to_period("M")

liq_base = (
    pd.merge(_vix_m[["year_month", "date", "vix_eom"]],
             _ba_m[["year_month", "mean_1_10y"]],
             on="year_month", how="inner")
    .pipe(pd.merge, _rmse_m[["year_month", "nom_rmse_bp"]],
          on="year_month", how="inner")
    .sort_values("date").reset_index(drop=True)
)

# ---- Standardise
liq_base["vix_std"]  = _standardise(liq_base["vix_eom"])
liq_base["ba_std"]   = _standardise(liq_base["mean_1_10y"])
liq_base["rmse_std"] = _standardise(liq_base["nom_rmse_bp"])

# ---- Composite averages
liq_base["liq1_raw"] = liq_base["vix_std"]
liq_base["liq2_raw"] = liq_base[["vix_std", "ba_std"]].mean(axis=1)
liq_base["liq3_raw"] = liq_base[["vix_std", "ba_std", "rmse_std"]].mean(axis=1)

# ---- Enforce positivity
liq_base["liq1"] = _enforce_positivity(liq_base["liq1_raw"])
liq_base["liq2"] = _enforce_positivity(liq_base["liq2_raw"])
liq_base["liq3"] = _enforce_positivity(liq_base["liq3_raw"])

print(f"\n  Liquidity panel : {liq_base['date'].min().date()} -> "
      f"{liq_base['date'].max().date()}  (n={len(liq_base)})")
print(f"  Liq1 : mean {liq_base['liq1'].mean():.3f}  "
      f"min {liq_base['liq1'].min():.3f}  max {liq_base['liq1'].max():.3f}")
print(f"  Liq2 : mean {liq_base['liq2'].mean():.3f}  "
      f"min {liq_base['liq2'].min():.3f}  max {liq_base['liq2'].max():.3f}")
print(f"  Liq3 : mean {liq_base['liq3'].mean():.3f}  "
      f"min {liq_base['liq3'].min():.3f}  max {liq_base['liq3'].max():.3f}")


# region COMPOSITE LIQUIDITY — STACKED PLOTS

def _stacked_liq_plot(components: list, comp_col: str, stem: str, comp_label: str = "Composite"):
    n = len(components) + 1
    fig, axes = plt.subplots(
        n, 1, figsize=(FIG_WIDTH * 1.4, FIG_HEIGHT * 0.9 * n),
        sharex=True,
        gridspec_kw={"hspace": 0.06},
    )
    fig.patch.set_facecolor("white")

    for ax, (col, label) in zip(axes[:-1], components):
            ax.axhline(0, color=C_ZERO, lw=0.8, ls="-", zorder=1)
            ax.fill_between(liq_base["date"], liq_base[col], y2=-2,
                            color="#AAAAAA", alpha=0.3, zorder=2)
            ax.plot(liq_base["date"], liq_base[col], color=C_STD, lw=1.6, zorder=3)
            ax.set_ylabel(label, fontsize=11)
            ax.set_xlim(liq_base["date"].min(), liq_base["date"].max())
            ax.set_ylim(-2, 6)
            ax.set_yticks([-1, 0, 1, 2, 3, 4, 5])
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))
            _style_ax(ax)

    axes[-1].fill_between(liq_base["date"], liq_base[comp_col],
                          color=C_FILL, alpha=0.35, zorder=2)
    axes[-1].plot(liq_base["date"], liq_base[comp_col], color=C_FILL, lw=1.8,
                  zorder=3)
    axes[-1].set_ylabel(comp_label, fontsize=11)
    axes[-1].set_xlim(liq_base["date"].min(), liq_base["date"].max())
    axes[-1].set_ylim(0, 5)
    axes[-1].set_yticks([1, 2, 3, 4])
    axes[-1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))
    _style_ax(axes[-1])

    plt.tight_layout(pad=0.6)
    _save_fig(fig, stem)
    plt.show()
    plt.close(fig)


_stacked_liq_plot(
    components=[("vix_std", "VIX (std)")],
    comp_col="liq1", stem="liq1_stacked",
    comp_label="Composite",
)
_stacked_liq_plot(
    components=[("vix_std", "VIX (std)"),
                ("ba_std",  "SGB bid-ask (std)")],
    comp_col="liq2", stem="liq2_stacked",
    comp_label="Composite",
)
_stacked_liq_plot(
    components=[("vix_std",  "VIX (std)"),
                ("ba_std",   "SGB bid-ask (std)"),
                ("rmse_std", "SGB rmse (std)")],
    comp_col="liq3", stem="liq3_stacked",
    comp_label="Composite",
)

# endregion


# region COMPOSITE LIQUIDITY — EXPORT

with pd.ExcelWriter(FACTOR_XL, engine="openpyxl", mode="w") as writer:
    (liq_base[["date", "vix_std", "ba_std", "rmse_std", "liq1", "liq2", "liq3"]]
     .to_excel(writer, sheet_name="liquidity", index=False))

print(f"\n  Exported liquidity factors -> {FACTOR_XL}")

# endregion


# endregion

    # region INFLATION
# Source: kpi_data.xlsx, sheet "basår 1980".
# Rolling 12-month CPI inflation: KPI_t / KPI_{t-12} - 1.
# Correlations:
#   corr(liq2, inflation_12m),  corr(liq2, inflation_std)
#   corr(liq3, inflation_12m),  corr(liq3, inflation_std)
# Plots: stacked two-panel figure per index (liq2, liq3).
#   Upper panel : raw 12m inflation (%) vs composite liquidity
#   Lower panel : standardised 12m inflation vs composite liquidity

C_INF = "#CB4335" 

# ---- Compute rolling 12-month inflation
kpi = raw_kpi[["date", "KPI"]].copy()
kpi["KPI"] = pd.to_numeric(kpi["KPI"], errors="coerce")
kpi = kpi.dropna(subset=["date", "KPI"]).sort_values("date").reset_index(drop=True)
kpi["inflation_12m"] = kpi["KPI"] / kpi["KPI"].shift(12) - 1.0

inf_panel = (
    kpi.loc[kpi["date"].between(SAMPLE_START, SAMPLE_END),
            ["date", "inflation_12m"]]
    .dropna(subset=["inflation_12m"])
    .reset_index(drop=True)
)
inf_panel["year_month"]    = inf_panel["date"].dt.to_period("M")
inf_panel["inflation_std"] = _standardise(inf_panel["inflation_12m"])
inf_panel["log_inflation_12m"] = np.log(1 + inf_panel["inflation_12m"])

# ---- Align with liquidity factors on year_month
liq_inf = pd.merge(
    liq_base[["year_month", "date", "liq2", "liq3"]],
    inf_panel[["year_month", "inflation_12m", "inflation_std"]],
    on="year_month", how="inner",
).sort_values("date").reset_index(drop=True)

# ---- Correlations
corr_liq2_raw = liq_inf["liq2"].corr(liq_inf["inflation_12m"])
corr_liq3_raw = liq_inf["liq3"].corr(liq_inf["inflation_12m"])
corr_liq2_std = liq_inf["liq2"].corr(liq_inf["inflation_std"])
corr_liq3_std = liq_inf["liq3"].corr(liq_inf["inflation_std"])

print(f"\n  Inflation panel : {liq_inf['date'].min().date()} -> "
      f"{liq_inf['date'].max().date()}  (n={len(liq_inf)})")
print(f"  Mean inflation  : {liq_inf['inflation_12m'].mean()*100:.2f}%  "
      f"std {liq_inf['inflation_12m'].std()*100:.2f}%")
print(f"\n  Corr(Liq2, inflation_12m) : {corr_liq2_raw:+.3f}")
print(f"  Corr(Liq2, inflation_std) : {corr_liq2_std:+.3f}")
print(f"  Corr(Liq3, inflation_12m) : {corr_liq3_raw:+.3f}")
print(f"  Corr(Liq3, inflation_std) : {corr_liq3_std:+.3f}")


# ---- Plot helper
def _plot_liq_inf(liq_col: str, liq_label: str, stem: str):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(FIG_WIDTH * 1.4, FIG_HEIGHT * 1.8),
        sharex=True,
        gridspec_kw={"hspace": 0.06},
    )
    fig.patch.set_facecolor("white")

    # ---- Upper panel: raw inflation vs composite
    ax1r = ax1.twinx()
    ax1.fill_between(liq_inf["date"], liq_inf[liq_col],
                     color=C_FILL, alpha=0.35, zorder=2)
    ax1.plot(liq_inf["date"], liq_inf[liq_col],
             color=C_FILL, lw=1.8, zorder=3, label=liq_label)
    ax1r.plot(liq_inf["date"], liq_inf["inflation_12m"] * 100,
              color=C_INF, lw=1.8, zorder=3, ls="-.", label="CPI inflation (12m)")
    ax1.set_ylabel(liq_label)
    ax1r.set_ylabel("Inflation (%)")
    ax1.set_ylim(0, 5)
    ax1.set_yticks([1, 2, 3, 4])
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax1r.set_ylim(-3, 12)
    ax1r.set_yticks([0, 3, 6, 9])
    ax1r.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))
    _style_ax(ax1)
    _style_ax(ax1r)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1r.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="upper left", frameon=True, fancybox=False,
               edgecolor="#AAAAAA", facecolor="white", framealpha=1.0,
               borderpad=0.6, handlelength=2.2)

    # ---- Lower panel: standardised inflation vs composite
    ax2r = ax2.twinx()
    ax2.fill_between(liq_inf["date"], liq_inf[liq_col],
                     color=C_FILL, alpha=0.35, zorder=2)
    ax2.plot(liq_inf["date"], liq_inf[liq_col],
             color=C_FILL, lw=1.8, zorder=3, label=liq_label)
    ax2r.plot(liq_inf["date"], liq_inf["inflation_std"],
              color=C_INF, lw=1.8, zorder=3, ls="-.", label="CPI inflation (std)")
    ax2.set_ylabel(liq_label)
    ax2r.set_ylabel("Inflation (std)")
    ax2.set_xlim(liq_inf["date"].min(), liq_inf["date"].max())
    ax2.set_ylim(0, 5)
    ax2.set_yticks([1, 2, 3, 4])
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax2r.set_ylim(-2, 8)
    ax2r.set_yticks([0, 2, 4, 6])
    ax2r.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))
    _style_ax(ax2)
    _style_ax(ax2r)
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2,
               loc="upper left", frameon=True, fancybox=False,
               edgecolor="#AAAAAA", facecolor="white", framealpha=1.0,
               borderpad=0.6, handlelength=2.2)

    plt.tight_layout(pad=0.6)
    _save_fig(fig, stem)
    plt.show()
    plt.close(fig)


_plot_liq_inf("liq2", "Composite", "liq2_inflation")
_plot_liq_inf("liq3", "Composite", "liq3_inflation")

# endregion

# ---- PCA 

# region PRINCIPAL COMPONENTS — SHARED SETUP
# Maturities:
#   Nominal : 6m – 120m (11 maturities)
#   IL      : 24m – 120m (9 maturities)
#
# Active liquidity factor: change ACTIVE_LIQ to switch between liq1/liq2/liq3

ACTIVE_LIQ = "liq2"

NOM_MATS       = ["y_6m",  "y_12m", "y_24m", "y_36m", "y_48m",  "y_60m",
                  "y_72m", "y_84m", "y_96m", "y_108m", "y_120m"]
NOM_MAT_LABELS = ["6m","1y","2y","3y","4y","5y","6y","7y","8y","9y","10y"]
NOM_MAT_YEARS  = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

IL_MATS        = ["y_24m", "y_36m", "y_48m",  "y_60m",  "y_72m",
                  "y_84m", "y_96m", "y_108m", "y_120m"]
IL_MAT_LABELS  = ["2y","3y","4y","5y","6y","7y","8y","9y","10y"]
IL_MAT_YEARS   = [2, 3, 4, 5, 6, 7, 8, 9, 10]
IL_MATS_P      = ["il_" + c for c in IL_MATS]

# ---- Colours
C_NOM = ["#0A1F3A", "#1A6E9E", "#A8C8E8"]
C_IL  = ["#0D4E2B", "#4DC090"] 

# ---- Yield panels
nom_yields = (
    raw_sgb_yields
    .loc[raw_sgb_yields["date"].between(SAMPLE_START, SAMPLE_END),
         ["date"] + NOM_MATS]
    .dropna().sort_values("date").reset_index(drop=True)
)
nom_yields["year_month"] = nom_yields["date"].dt.to_period("M")

il_yields = (
    raw_il_yields
    .loc[raw_il_yields["date"].between(SAMPLE_START, SAMPLE_END),
         ["date"] + IL_MATS]
    .dropna().sort_values("date").reset_index(drop=True)
    .rename(columns={c: "il_" + c for c in IL_MATS})
)
il_yields["year_month"] = il_yields["date"].dt.to_period("M")

# ---- Active liquidity and inflation series
liq_active = (liq_base[["date", "year_month", ACTIVE_LIQ]]
              .copy().rename(columns={ACTIVE_LIQ: "liq"}))

inf_series = inf_panel[["date", "year_month", "log_inflation_12m"]].copy()

print(f"\n  Nominal yields  : {len(nom_yields)} obs")
print(f"  IL yields       : {len(il_yields)} obs")
print(f"  Active liq      : {ACTIVE_LIQ}")


# ---- Alignment helper
def _align(extras=None):
    base = pd.merge(nom_yields,
                    il_yields[["year_month"] + IL_MATS_P],
                    on="year_month", how="inner")
    if extras:
        for df, cols in extras:
            base = pd.merge(base, df[["year_month"] + cols],
                            on="year_month", how="inner")
    return base.dropna().sort_values("date").reset_index(drop=True)


# ---- PCA helpers
def _orthogonalize(Y: np.ndarray, X: np.ndarray) -> np.ndarray:
    Xc   = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.lstsq(Xc, Y, rcond=None)[0]
    return Y - Xc @ beta

def _run_pca(Y: np.ndarray, n: int):
    pca = _PCA(n_components=n)
    sc  = pca.fit_transform(Y)
    return sc, pca.components_.copy(), pca.explained_variance_ratio_.copy()

def _std_scores(sc: np.ndarray) -> np.ndarray:
    return (sc - sc.mean(axis=0)) / sc.std(axis=0)

def _sign_nom(ld: np.ndarray, sc: np.ndarray):
    for i in range(3):
        if ld[i].mean() < 0:
            ld[i] *= -1; sc[:, i] *= -1
    ld[2] *= -1; sc[:, 2] *= -1
    return ld, sc

def _sign_il(ld: np.ndarray, sc: np.ndarray):
    for i in range(ld.shape[0]):
        if ld[i].mean() < 0:
            ld[i] *= -1; sc[:, i] *= -1
    return ld, sc

def _panel_r2(Y: np.ndarray, X: np.ndarray) -> float:
    Xc    = np.column_stack([np.ones(len(X)), X])
    beta  = np.linalg.lstsq(Xc, Y, rcond=None)[0]
    resid = Y - Xc @ beta
    return 1.0 - np.sum(resid**2) / np.sum((Y - Y.mean(axis=0))**2)


# ---- Variance explained and factor summary printer
def _print_var_explained(spec_name, Y_nom, Y_il, nom_sc, il_sc,
                          nom_evr, il_evr, il_orth=None):
    """
    il_orth : list of (array, label) for extra regressors between nom PCs
              and IL PCs, e.g. [(liq_arr, "liq"), (inf_arr, "inf")]
    """
    print(f"\n  {'='*52}")
    print(f"  {spec_name}")
    print(f"  {'='*52}")

    # ---- PCA variance explained
    print(f"\n  Nominal PCA — share of variance explained:")
    print(f"    PC1: {nom_evr[0]:.1%}  "
          f"PC2: {nom_evr[1]:.1%}  "
          f"PC3: {nom_evr[2]:.1%}  "
          f"Sum: {nom_evr[:3].sum():.1%}")

    resid_frac_il = 1.0 - _panel_r2(Y_il, nom_sc)
    if il_orth:
        X_tmp = nom_sc.copy()
        for arr, _ in il_orth:
            X_tmp = np.column_stack([X_tmp, arr])
        resid_frac_il = 1.0 - _panel_r2(Y_il, X_tmp)
    print(f"\n  IL PCA — share of residual variance explained:")
    print(f"    il_PC1: {il_evr[0]:.1%}  "
          f"il_PC2: {il_evr[1]:.1%}  "
          f"Sum: {il_evr[:2].sum():.1%}  "
          f"(residual = {resid_frac_il:.1%} of total IL variance)")

    # ---- Sequential panel R² — nominal yields
    print(f"\n  Nominal yields — sequential panel R²:")
    r2_nom = _panel_r2(Y_nom, nom_sc)
    print(f"    nom PC1+2+3 : {r2_nom:.1%}")

    # ---- Sequential panel R² — IL yields
    print(f"\n  IL yields — sequential panel R²:")
    X_il = nom_sc.copy()
    r2   = _panel_r2(Y_il, X_il)
    print(f"    nom PC1+2+3 : {r2:.1%}")
    prev = r2
    if il_orth:
        for arr, lbl in il_orth:
            X_il = np.column_stack([X_il, arr])
            r2   = _panel_r2(Y_il, X_il)
            print(f"    + {lbl:<10} : {r2:.1%}  (Δ {r2-prev:+.1%})")
            prev = r2
    X_il  = np.column_stack([X_il, il_sc])
    r2_il = _panel_r2(Y_il, X_il)
    print(f"    + il PC1+2  : {r2_il:.1%}  (Δ {r2_il-prev:+.1%})")

    # ---- Factor summary statistics
    print(f"\n  Factor summary (mean / std):")
    for i in range(3):
        print(f"    nom_PC{i+1} : "
              f"mean {nom_sc[:,i].mean():+.4f}  "
              f"std  {nom_sc[:,i].std():.4f}")
    if il_orth:
        for arr, lbl in il_orth:
            print(f"    {lbl:<8} : "
                  f"mean {arr.mean():+.4f}  "
                  f"std  {arr.std():.4f}")
    for i in range(2):
        print(f"    il_PC{i+1}  : "
              f"mean {il_sc[:,i].mean():+.4f}  "
              f"std  {il_sc[:,i].std():.4f}")


# ---- Plot helper 1: loadings (line+markers) and score time series  [1×2 figure]
def _plot_spec(stem, dates, scores, loadings, mat_years, mat_labels,
               pc_colors, pc_labels, extra_ts=None):

    n_pc    = len(pc_labels)
    zorders = list(range(n_pc + 2, 2, -1))   # e.g. 3 PCs -> [5, 4, 3]

    fig, (ax_ld, ax_sc) = plt.subplots(
        1, 2, figsize=(FIG_WIDTH * 2.5, FIG_HEIGHT)
    )
    fig.patch.set_facecolor("white")

    # ---- Loadings
    for i, (col, lbl) in enumerate(zip(pc_colors, pc_labels)):
        ax_ld.plot(mat_years, loadings[i], color=col, lw=1.8,
                   marker="o", markersize=5, zorder=zorders[i], label=lbl)
    ax_ld.axhline(0, color=C_ZERO, lw=0.8, zorder=1)
    ax_ld.set_xticks(mat_years)
    ax_ld.set_xticklabels(mat_labels, rotation=45, fontsize=9)
    ax_ld.set_ylim(-0.75, 0.75)
    ax_ld.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
    ax_ld.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))
    _style_ax(ax_ld)
    ax_ld.legend(loc="upper right", frameon=True, fancybox=False,
                 edgecolor="#AAAAAA", facecolor="white", framealpha=1.0,
                 borderpad=0.6, handlelength=2.2, ncol=n_pc)

    # ---- Scores
    for i, (col, lbl) in enumerate(zip(pc_colors, pc_labels)):
        ax_sc.plot(dates, scores[:, i], color=col, lw=1.8,
                   zorder=zorders[i], label=lbl)
    if extra_ts:
        for arr, lbl, col in extra_ts:
            ax_sc.plot(dates, arr, color=col, lw=1.8,
                       ls="--", zorder=2, label=lbl)
    ax_sc.axhline(0, color=C_ZERO, lw=0.8, zorder=1)
    ax_sc.set_xlim(dates.min(), dates.max())
    ax_sc.set_ylim(-3, 5)
    ax_sc.set_yticks([-2, -1, 0, 1, 2, 3, 4])
    ax_sc.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))
    _style_ax(ax_sc)
    ax_sc.legend(loc="upper right", frameon=True, fancybox=False,
                 edgecolor="#AAAAAA", facecolor="white", framealpha=1.0,
                 borderpad=0.6, handlelength=2.2,
                 ncol=n_pc + (len(extra_ts) if extra_ts else 0))

    plt.tight_layout(pad=0.6)
    _save_fig(fig, stem)
    plt.show(); plt.close(fig)

# ---- Plot helper 2: macro factors
def _plot_extra_factors(stem, dates, series_list):
    """
    Extra factor time series plot matching the style of _plot_liq_inf.
    series_list : list of (array, label, color)
      - 1 series  : single axis with fill_between
      - 2 series  : twinx; first series on lhs (fill), second on rhs (line)
    """
    fig, ax1 = plt.subplots(figsize=(FIG_WIDTH * 1.4, FIG_HEIGHT))
    fig.patch.set_facecolor("white")

    arr1, lbl1, col1 = series_list[0]
    ax1.fill_between(dates, arr1, color=col1, alpha=0.35, zorder=2)
    ax1.plot(dates, arr1, color=col1, lw=1.8, zorder=3, label=lbl1)
    ax1.set_xlim(dates.min(), dates.max())
    ax1.set_ylim(0, 5)
    ax1.set_yticks([1, 2, 3, 4])
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))
    _style_ax(ax1)

    if len(series_list) == 2:
        arr2, lbl2, col2 = series_list[1]
        ax2 = ax1.twinx()
        ax2.plot(dates, arr2, color=col2, lw=1.8, ls="-.", zorder=3, label=lbl2)
        ax2.set_ylim(-2, 8)
        ax2.set_yticks([-0, 2, 4, 6])
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))
        _style_ax(ax2)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2,
                   loc="upper left", frameon=True, fancybox=False,
                   edgecolor="#AAAAAA", facecolor="white", framealpha=1.0,
                   borderpad=0.6, handlelength=2.2, ncol=2)
    else:
        ax1.legend(loc="upper left", frameon=True, fancybox=False,
                   edgecolor="#AAAAAA", facecolor="white", framealpha=1.0,
                   borderpad=0.6, handlelength=2.2)

    plt.tight_layout(pad=0.6)
    _save_fig(fig, stem)
    plt.show(); plt.close(fig)

# ---- Export helper
def _export_spec(sheet, dates, nom_sc, il_sc, extra_cols=None):
    df = pd.DataFrame({
        "date":    dates,
        "nom_PC1": nom_sc[:, 0],
        "nom_PC2": nom_sc[:, 1],
        "nom_PC3": nom_sc[:, 2],
    })
    if extra_cols:
        for col, arr in extra_cols.items():
            df[col] = arr
    df["il_PC1"] = il_sc[:, 0]
    df["il_PC2"] = il_sc[:, 1]
    mode = "a" if os.path.exists(FACTOR_XL) else "w"
    with pd.ExcelWriter(FACTOR_XL, engine="openpyxl", mode=mode,
                        if_sheet_exists="replace") as writer:
        df.to_excel(writer, sheet_name=sheet, index=False)
    print(f"\n  Exported -> {FACTOR_XL} [{sheet}]")

# endregion

# region 5-FACTOR MODEL
# Factors : nom_PC1, nom_PC2, nom_PC3, il_PC1, il_PC2
# Nominal PCA on raw yields (6m–120m).
# IL PCs from PCA on residuals of IL yields (24m–120m) regressed on nom PCs.

f5 = _align()
Y_nom5 = f5[NOM_MATS].to_numpy()
Y_il5  = f5[IL_MATS_P].to_numpy()

nom_sc5, nom_ld5, nom_evr5 = _run_pca(Y_nom5, 3)
nom_ld5, nom_sc5            = _sign_nom(nom_ld5, nom_sc5)
nom_sc5                     = _std_scores(nom_sc5)

il_r5               = _orthogonalize(Y_il5, nom_sc5)
il_sc5, il_ld5, il_evr5 = _run_pca(il_r5, 2)
il_ld5, il_sc5      = _sign_il(il_ld5, il_sc5)
il_sc5              = _std_scores(il_sc5)

_print_var_explained("5-factor model",
                     Y_nom5, Y_il5, nom_sc5, il_sc5, nom_evr5, il_evr5)
_export_spec("5_factor", f5["date"], nom_sc5, il_sc5)

# 5-FACTOR — PLOTS

_plot_spec("f5_nom", f5["date"], nom_sc5, nom_ld5,
           NOM_MAT_YEARS, NOM_MAT_LABELS, C_NOM,
           ["SGB PC1", "SGB PC2", "SGB PC3"])

_plot_spec("f5_il", f5["date"], il_sc5, il_ld5,
           IL_MAT_YEARS, IL_MAT_LABELS, C_IL,
           ["SGBi PC1", "SGBi PC2"])

# endregion

# region 6-FACTOR MODEL
# Factors : nom_PC1, nom_PC2, nom_PC3, liq, il_PC1, il_PC2
# Nominal PCA on raw yields.
# IL PCs from PCA on residuals of IL yields regressed on nom PCs + liq.

f6 = _align(extras=[(liq_active, ["liq"])])
Y_nom6 = f6[NOM_MATS].to_numpy()
Y_il6  = f6[IL_MATS_P].to_numpy()
liq_6  = f6["liq"].to_numpy()

nom_sc6, nom_ld6, nom_evr6 = _run_pca(Y_nom6, 3)
nom_ld6, nom_sc6            = _sign_nom(nom_ld6, nom_sc6)
nom_sc6                     = _std_scores(nom_sc6)

il_r6               = _orthogonalize(Y_il6, np.column_stack([nom_sc6, liq_6]))
il_sc6, il_ld6, il_evr6 = _run_pca(il_r6, 2)
il_ld6, il_sc6      = _sign_il(il_ld6, il_sc6)
il_sc6              = _std_scores(il_sc6)

_print_var_explained("6-factor model",
                     Y_nom6, Y_il6, nom_sc6, il_sc6, nom_evr6, il_evr6,
                     il_orth=[(liq_6, "liq")])
_export_spec("6_factor", f6["date"], nom_sc6, il_sc6,
             extra_cols={"liq": liq_6})

# 6-FACTOR — PLOTS

_plot_spec("f6_nom", f6["date"], nom_sc6, nom_ld6,
           NOM_MAT_YEARS, NOM_MAT_LABELS, C_NOM,
           ["SGB PC1", "SGB PC2", "SGB PC3"])

_plot_spec("f6_il", f6["date"], il_sc6, il_ld6,
           IL_MAT_YEARS, IL_MAT_LABELS, C_IL,
           ["SGBi PC1", "SGBi PC2"])

# endregion

# region 7-FACTOR MODEL
# Factors : nom_PC1, nom_PC2, nom_PC3, liq, inf, il_PC1, il_PC2
# Nominal PCA on raw yields.
# Inflation: log 12m rolling inflation residualised on nom PCs, standardised.
# IL PCs from PCA on residuals of IL yields regressed on nom PCs + liq + inf.

f7 = _align(extras=[(liq_active, ["liq"]),
                    (inf_series,  ["log_inflation_12m"])])
Y_nom7 = f7[NOM_MATS].to_numpy()
Y_il7  = f7[IL_MATS_P].to_numpy()
liq_7  = f7["liq"].to_numpy()
inf_7     = f7["log_inflation_12m"].to_numpy()

nom_sc7, nom_ld7, nom_evr7 = _run_pca(Y_nom7, 3)
nom_ld7, nom_sc7            = _sign_nom(nom_ld7, nom_sc7)
nom_sc7                     = _std_scores(nom_sc7)

inf_orth7 = _orthogonalize(inf_7.reshape(-1, 1), nom_sc7).ravel()
inf_orth7 = (inf_orth7 - inf_orth7.mean()) / inf_orth7.std()

il_r7 = _orthogonalize(Y_il7,
                        np.column_stack([nom_sc7, liq_7, inf_orth7]))
il_sc7, il_ld7, il_evr7 = _run_pca(il_r7, 2)
il_ld7, il_sc7          = _sign_il(il_ld7, il_sc7)
il_sc7                  = _std_scores(il_sc7)

_print_var_explained("7-factor model",
                     Y_nom7, Y_il7, nom_sc7, il_sc7, nom_evr7, il_evr7,
                     il_orth=[(liq_7, "liq"), (inf_orth7, "inf")])
_export_spec("7_factor", f7["date"], nom_sc7, il_sc7,
             extra_cols={"liq": liq_7, "inf": inf_orth7})

# 7-FACTOR — PLOTS

_plot_spec("f7_nom", f7["date"], nom_sc7, nom_ld7,
           NOM_MAT_YEARS, NOM_MAT_LABELS, C_NOM,
           ["SGB PC1", "SGB PC2", "SGB PC3"])

_plot_spec("f7_il", f7["date"], il_sc7, il_ld7,
           IL_MAT_YEARS, IL_MAT_LABELS, C_IL,
           ["SGBi PC1", "SGBi PC2"])

_plot_extra_factors("f7_extra", f7["date"],
                    [(liq_7,     "Liquidity factor (lhs)",  C_FILL),
                     (inf_orth7, "Inflation factor (rhs)",  C_INF)])


# endregion

# region PRINCIPAL COMPONENTS — CARRY-ADJUSTED IL YIELDS
# Identical to the three standard PCA models above, but the IL yield panel
# is replaced with the lag-adjusted yields (ỹ = y_obs + (12/n)·D_t).
# Nominal PCA and extra factors (liq, inf) are unchanged — only the IL
# residuals fed into the IL PCA change.
# Results exported to 5_factor_adj, 6_factor_adj, 7_factor_adj sheets.

# ---- Load carry-adjusted IL yields

raw_il_adj_yields = pd.read_excel(
    f"{DIR_IL}/zero_yields_SGBIL_lag_adj.xlsx",
    sheet_name="zero_yields_lag_adj",
)
raw_il_adj_yields["date"] = pd.to_datetime(raw_il_adj_yields["date"])

il_yields_adj = (
    raw_il_adj_yields
    .loc[raw_il_adj_yields["date"].between(SAMPLE_START, SAMPLE_END),
         ["date"] + IL_MATS]
    .dropna()
    .sort_values("date")
    .reset_index(drop=True)
    .rename(columns={c: "il_" + c for c in IL_MATS})
)
il_yields_adj["year_month"] = il_yields_adj["date"].dt.to_period("M")

print(f"\n  IL yields (adj) : {len(il_yields_adj)} obs")


# ---- Alignment helper (carry-adjusted IL)

def _align_adj(extras=None):
    base = pd.merge(
        nom_yields,
        il_yields_adj[["year_month"] + IL_MATS_P],
        on="year_month", how="inner",
    )
    if extras:
        for df, cols in extras:
            base = pd.merge(base, df[["year_month"] + cols],
                            on="year_month", how="inner")
    return base.dropna().sort_values("date").reset_index(drop=True)


# region 5-FACTOR MODEL (ADJ)
# Factors: nom_PC1, nom_PC2, nom_PC3, il_PC1_adj, il_PC2_adj

f5b = _align_adj()
Y_nom5b = f5b[NOM_MATS].to_numpy()
Y_il5b  = f5b[IL_MATS_P].to_numpy()

nom_sc5b, nom_ld5b, nom_evr5b = _run_pca(Y_nom5b, 3)
nom_ld5b, nom_sc5b             = _sign_nom(nom_ld5b, nom_sc5b)
nom_sc5b                       = _std_scores(nom_sc5b)

il_r5b               = _orthogonalize(Y_il5b, nom_sc5b)
il_sc5b, il_ld5b, il_evr5b = _run_pca(il_r5b, 2)
il_ld5b, il_sc5b     = _sign_il(il_ld5b, il_sc5b)
il_sc5b              = _std_scores(il_sc5b)

_print_var_explained("5-factor model (adj IL)",
                     Y_nom5b, Y_il5b, nom_sc5b, il_sc5b, nom_evr5b, il_evr5b)
_export_spec("5_factor_adj", f5b["date"], nom_sc5b, il_sc5b)

_plot_spec("f5b_nom", f5b["date"], nom_sc5b, nom_ld5b,
           NOM_MAT_YEARS, NOM_MAT_LABELS, C_NOM,
           ["SGB PC1", "SGB PC2", "SGB PC3"])

_plot_spec("f5b_il", f5b["date"], il_sc5b, il_ld5b,
           IL_MAT_YEARS, IL_MAT_LABELS, C_IL,
           ["SGBi PC1 (adj)", "SGBi PC2 (adj)"])

# endregion


# region 6-FACTOR MODEL (ADJ)
# Factors: nom_PC1, nom_PC2, nom_PC3, liq, il_PC1_adj, il_PC2_adj

f6b = _align_adj(extras=[(liq_active, ["liq"])])
Y_nom6b = f6b[NOM_MATS].to_numpy()
Y_il6b  = f6b[IL_MATS_P].to_numpy()
liq_6b  = f6b["liq"].to_numpy()

nom_sc6b, nom_ld6b, nom_evr6b = _run_pca(Y_nom6b, 3)
nom_ld6b, nom_sc6b             = _sign_nom(nom_ld6b, nom_sc6b)
nom_sc6b                       = _std_scores(nom_sc6b)

il_r6b               = _orthogonalize(Y_il6b, np.column_stack([nom_sc6b, liq_6b]))
il_sc6b, il_ld6b, il_evr6b = _run_pca(il_r6b, 2)
il_ld6b, il_sc6b     = _sign_il(il_ld6b, il_sc6b)
il_sc6b              = _std_scores(il_sc6b)

_print_var_explained("6-factor model (adj IL)",
                     Y_nom6b, Y_il6b, nom_sc6b, il_sc6b, nom_evr6b, il_evr6b,
                     il_orth=[(liq_6b, "liq")])
_export_spec("6_factor_adj", f6b["date"], nom_sc6b, il_sc6b,
             extra_cols={"liq": liq_6b})

_plot_spec("f6b_nom", f6b["date"], nom_sc6b, nom_ld6b,
           NOM_MAT_YEARS, NOM_MAT_LABELS, C_NOM,
           ["SGB PC1", "SGB PC2", "SGB PC3"])

_plot_spec("f6b_il", f6b["date"], il_sc6b, il_ld6b,
           IL_MAT_YEARS, IL_MAT_LABELS, C_IL,
           ["SGBi PC1 (adj)", "SGBi PC2 (adj)"])

# endregion


# region 7-FACTOR MODEL (ADJ)
# Factors: nom_PC1, nom_PC2, nom_PC3, liq, inf, il_PC1_adj, il_PC2_adj

f7b = _align_adj(extras=[(liq_active, ["liq"]),
                          (inf_series,  ["log_inflation_12m"])])
Y_nom7b = f7b[NOM_MATS].to_numpy()
Y_il7b  = f7b[IL_MATS_P].to_numpy()
liq_7b  = f7b["liq"].to_numpy()
inf_7b  = f7b["log_inflation_12m"].to_numpy()

nom_sc7b, nom_ld7b, nom_evr7b = _run_pca(Y_nom7b, 3)
nom_ld7b, nom_sc7b             = _sign_nom(nom_ld7b, nom_sc7b)
nom_sc7b                       = _std_scores(nom_sc7b)

inf_orth7b = _orthogonalize(inf_7b.reshape(-1, 1), nom_sc7b).ravel()
inf_orth7b = (inf_orth7b - inf_orth7b.mean()) / inf_orth7b.std()

il_r7b = _orthogonalize(Y_il7b, np.column_stack([nom_sc7b, liq_7b, inf_orth7b]))
il_sc7b, il_ld7b, il_evr7b = _run_pca(il_r7b, 2)
il_ld7b, il_sc7b            = _sign_il(il_ld7b, il_sc7b)
il_sc7b                     = _std_scores(il_sc7b)

_print_var_explained("7-factor model (adj IL)",
                     Y_nom7b, Y_il7b, nom_sc7b, il_sc7b, nom_evr7b, il_evr7b,
                     il_orth=[(liq_7b, "liq"), (inf_orth7b, "inf")])
_export_spec("7_factor_adj", f7b["date"], nom_sc7b, il_sc7b,
             extra_cols={"liq": liq_7b, "inf": inf_orth7b})

_plot_spec("f7b_nom", f7b["date"], nom_sc7b, nom_ld7b,
           NOM_MAT_YEARS, NOM_MAT_LABELS, C_NOM,
           ["SGB PC1", "SGB PC2", "SGB PC3"])

_plot_spec("f7b_il", f7b["date"], il_sc7b, il_ld7b,
           IL_MAT_YEARS, IL_MAT_LABELS, C_IL,
           ["SGBi PC1 (adj)", "SGBi PC2 (adj)"])

_plot_extra_factors("f7b_extra", f7b["date"],
                    [(liq_7b,     "Liquidity factor (lhs)",  C_FILL),
                     (inf_orth7b, "Inflation factor (rhs)",  C_INF)])

# endregion

# endregion


