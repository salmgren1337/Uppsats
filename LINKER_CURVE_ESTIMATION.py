import pandas as pd
import numpy as np
import QuantLib as ql
import matplotlib.pyplot as plt
import multiprocessing as mp
import os

# ------------------------------------------------------------------------------
#   Fits the Nelson-Siegel yield curve to a monthly cross-section of
#   Swedish inflation-linked government bonds (SGBi).
#
#   Quoted Bloomberg prices are nominal clean prices.  They are
#   converted to real clean prices using the Swedish National Debt Office
#   indexation convention: a 3-month lag with linear day-weighted interpolation
#   of the reference KPI at the bond's settlement date.
#
#       Real clean price = Nominal clean price / (KPI_ref(t_s) / KPI_base)
#
#       KPI_ref(t_s) = KPI_{t_s-3m} + (d-1)/30 * (KPI_{t_s-2m} - KPI_{t_s-3m})
#
#   where t_s is the settlement date and d is the settlement day (capped at 30).
#
#   For each date two NS curves are fitted to real prices and real coupons:
#     - Actual:   fitted to observed SGBi bonds only
#     - Extended: fitted to observed bonds + synthetic real ZCBs inserted at
#                 the midpoint between each adjacent pair of actual maturities
#                 (priced from a preliminary real bootstrap, to fill gaps)
#
#   The short-rate anchor (starting value for b0 + b1) is:
#       anchor = log(1 + SSVX_1m - pi_{t,trailing12m})
#   where SSVX_1m is the quoted 1-month T-bill rate and pi_{t,trailing12m}
#   is the 12-month trailing CPI inflation rate KPI_M/KPI_{M-12} - 1.
# ------------------------------------------------------------------------------
#   Run the script once from 0–5 to load data and define all functions.
#   Then use the two RUN regions 6 and 7 independently:
#
#   RUN SINGLE DATE  — set SINGLE_YM at the top of that region, then run
#                      Step 1 (estimation) and Step 2 (diagnostics + plots)
#                      by selecting and executing them individually.
#
#   RUN PANEL        — run Step 1 (full estimation), Step 2 (diagnostics),
#                      and Step 3 (export) individually in sequence.
#                      Steps 2 and 3 are independent of each other and can
#                      be re-run without re-estimating.
# ------------------------------------------------------------------------------
#   All settings that govern the NS estimation (parameter bounds, L2
#   regularisation weights, optimiser tolerances, and the multistart grid)
#   are collected in the "Estimation pipeline" sub-region inside CONFIGURATION.
#   Adjust them there only — they are referenced globally throughout the script.
# ------------------------------------------------------------------------------

# region 0. CONFIGURATION

# ---- Data and output
SGBIL_FILE  = "statsobligationer_data.xlsx"
KPI_FILE    = "kpi_data copy.xlsx"
OUTPUT_FILE = "LINKER_CURVE_ESTIMATION/fit_params_SGBIL.xlsx"

# ---- Sample period
START = pd.Timestamp("2004-01-01")
END   = pd.Timestamp("2025-12-31")

# ---- TTM filter
MIN_TTM_YEARS = 1.0
MAX_TTM_YEARS = 20.0

# ---- QuantLib bond conventions (Swedish government bonds: 30/360, annual)
SETTLEMENT_DAYS = 2
FACE            = 100.0
REDEMPTION      = 100.0
DAY_COUNT       = ql.Thirty360(ql.Thirty360.European)
TENOR           = ql.Period(ql.Annual)
CALENDAR        = ql.Sweden()
BUS_CONV        = ql.ModifiedFollowing
DATE_GEN_RULE   = ql.DateGeneration.Backward
END_OF_MONTH    = False

# ---- Figure settings
# FIG_WIDTH / FIG_HEIGHT in inches. 5.5 x 3.4 fits a standard A4 LaTeX column.
FIG_WIDTH    = 5.5
FIG_HEIGHT   = 3.4
FIG_DPI      = 300
FIG_FORMAT   = "pdf"        # "pdf" or "png"
FIG_SAVE_DIR = "LINKER_CURVE_ESTIMATION"

# ---- Colours
COLOR_ACTUAL   = "#3A3A3A"
COLOR_EXTENDED = "#1A5EA8"
COLOR_BOOT     = "#8B1A1A"
COLOR_OBSERVED = "#888888"

plt.rcParams.update({
    "font.family":        "serif",
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

os.makedirs("LINKER_CURVE_ESTIMATION", exist_ok=True)

    # region 0.1. Estimation pipeline

    # ---- Parameter bounds
    # Beta bounds are in continuously compounded rate space.
    # Tau bounds are in years; internally converted to kappa = 1/tau.
B0_BOUNDS  = (-0.03,  0.06)   # long-run real level
B1_BOUNDS  = (-0.15,  0.15)   # slope
B2_BOUNDS  = (-0.08,  0.08)   # hump magnitude
TAU_BOUNDS = (2.0, 8.0)    # hump location (years)

    # ---- L2 regularisation weights [b0, b1, b2, k1]
    # b2 and k1 carry moderately higher penalties than the nominal script to
    # partially offset the sparse SGBi cross-section (4–8 bonds vs ~10 for SGBs).
    # Numerical investigation confirms fit quality is insensitive to L2 in this
    # range; the primary role of the weights is parameter stability.
L2_WEIGHTS = [0.05, 0.01, 0.20, 0.05]

    # ---- Optimiser (Nelder-Mead simplex)
ACCURACY       = 1e-10
MAX_EVALS      = 15_000
SIMPLEX_LAMBDA = 0.005

    # ---- Multistart grid
    # b0 and b1 are anchored per date from the bootstrap and SSVX/CPI anchor.
    # Every (b2, tau) combination is tried.
    # B2_GRID spans ±0.04 to match the wider B2_BOUNDS.
B2_GRID  = [-0.04, -0.02, -0.01, 0.0, 0.01, 0.02, 0.04]
TAU_GRID = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0]

    # endregion

# endregion

# region 1. DATA PREPARATION
# Loads SGBi bonds, KPI, and the SSVX 1m rate; aligns all sources to the
# last SGBi trading day of each month; builds the short-rate anchor series.
# Run once at script startup — all downstream sections depend on these objects.

    # region 1.1. SGBi bonds

raw_data = pd.read_excel(SGBIL_FILE, sheet_name="SGBi long")

# Series 3002 and 3003 were issued before base-KPI records were kept separately;
# backfill their base KPI from the contemporaneous series 3001 value.
base_kpi_3001 = raw_data.loc[raw_data["Serie"] == 3001, "BasKPI"].iloc[0]
raw_data.loc[raw_data["Serie"].isin([3002, 3003]), "BasKPI"] = base_kpi_3001

df_SGBILs = (
    raw_data.loc[:, [
        "Date", "PX_LAST", "YLD_YTM_MID", "BID_YIELD", "ASK_YIELD",
        "Kupong", "Issue date", "Maturity date", "ISIN", "Serie", "BasKPI",
    ]]
    .rename(columns={
        "Date":          "date",
        "PX_LAST":       "price",
        "YLD_YTM_MID":   "yield",
        "BID_YIELD":     "bid_yield",
        "ASK_YIELD":     "ask_yield",
        "Kupong":        "coupon",
        "Issue date":    "issue_date",
        "Maturity date": "maturity_date",
        "ISIN":          "isin",
        "Serie":         "serie",
        "BasKPI":        "bas_kpi",
    })
    .copy()
)

for col in ["date", "issue_date", "maturity_date"]:
    df_SGBILs[col] = pd.to_datetime(df_SGBILs[col], errors="coerce")
for col in ["price", "yield", "bid_yield", "ask_yield", "coupon", "bas_kpi"]:
    df_SGBILs[col] = pd.to_numeric(df_SGBILs[col], errors="coerce")

df_SGBILs[["yield", "bid_yield", "ask_yield", "coupon"]] /= 100.0

# Remove the illiquid series 3103 (duplicate maturity date with another bond)
df_SGBILs = df_SGBILs.loc[df_SGBILs["serie"] != 3103].copy()

df_SGBILs = df_SGBILs.loc[
    (df_SGBILs["date"] >= START) & (df_SGBILs["date"] <= END)
]

ttm_years = (df_SGBILs["maturity_date"] - df_SGBILs["date"]).dt.days / 365.25
df_SGBILs = df_SGBILs[
    (ttm_years >= MIN_TTM_YEARS) & (ttm_years <= MAX_TTM_YEARS)
]
df_SGBILs = df_SGBILs.sort_values(["date", "maturity_date"]).reset_index(drop=True)

# Build the last-trading-day index; used below to align all monthly data sources
df_SGBILs["_month"] = df_SGBILs["date"].dt.to_period("M")
month_last_trade    = df_SGBILs.groupby("_month")["date"].max()
df_SGBILs           = df_SGBILs.drop(columns="_month")

    # endregion

    # region 1.2. KPI

raw_kpi = pd.read_excel(KPI_FILE, sheet_name="basår 1980")

df_kpi = raw_kpi.loc[:, ["date", "KPI"]].copy()
df_kpi["date"] = pd.to_datetime(df_kpi["date"], errors="coerce")
df_kpi["KPI"]  = pd.to_numeric(df_kpi["KPI"],  errors="coerce")
df_kpi["month"] = df_kpi["date"].dt.to_period("M")
df_kpi = (
    df_kpi[["month", "KPI"]]
    .dropna()
    .drop_duplicates(subset=["month"])
    .sort_values("month")
    .reset_index(drop=True)
)

# Series indexed by Period — used inside estimation functions
kpi_by_month = df_kpi.set_index("month")["KPI"]

# 12-month trailing inflation for month M:
#   trail_infl_12m(M) = KPI_M / KPI_{M-12} - 1
# This is the standard year-over-year CPI rate. It is smooth (no seasonal
# distortion), requires no forward-looking data, and is the appropriate
# companion to the annualized SSVX 1m rate in the anchor calculation.
# Using 1-month annualized forward inflation caused severe seasonal spikes
# (December KPI reliably falls in January, producing annualized "deflation"
# of -8% to -16%) that pushed the anchor to economically impossible values
# and caused NS optimization failures on nearly every December date.
df_kpi["KPI_lag12"]      = df_kpi["KPI"].shift(12)
df_kpi["trail_infl_12m"] = df_kpi["KPI"] / df_kpi["KPI_lag12"] - 1
trail_infl_by_month      = df_kpi.set_index("month")["trail_infl_12m"]

    # endregion

    # region 1.3. SSVX 1m rate and short-rate anchor

raw_short  = pd.read_excel(SGBIL_FILE, sheet_name="korta räntor riksbanken")

df_ssvx = raw_short.loc[:, ["date", "SSVX_1m"]].copy()
df_ssvx["date"]    = pd.to_datetime(df_ssvx["date"], errors="coerce")
df_ssvx["SSVX_1m"] = pd.to_numeric(df_ssvx["SSVX_1m"], errors="coerce") / 100.0

# Align publication month to the SGBi last trading day of that month
df_ssvx["_month"] = df_ssvx["date"].dt.to_period("M")
df_ssvx["date"]   = df_ssvx["_month"].map(month_last_trade)
df_ssvx = (
    df_ssvx.drop(columns="_month")
    .dropna(subset=["date", "SSVX_1m"])
    .sort_values("date")
    .reset_index(drop=True)
)

# Map 12-month trailing inflation to each aligned date
df_ssvx["_month"]      = df_ssvx["date"].dt.to_period("M")
df_ssvx["trail_infl"]  = df_ssvx["_month"].map(trail_infl_by_month)

# Short-rate anchor (continuously compounded):
#   anchor_cc = log(1 + SSVX_1m - trail_infl_12m)
# SSVX_1m is an annualized simple rate; trail_infl_12m is the year-over-year
# CPI rate — both in annual terms, so the subtraction is consistent.
# The result approximates the real cc short rate at t→0 and is used only
# to set the starting value b0 + b1 = anchor_cc.
df_anchor = (
    df_ssvx.drop(columns="_month")
    .dropna(subset=["trail_infl"])
    .copy()
)
df_anchor["anchor_simple"] = df_anchor["SSVX_1m"] - df_anchor["trail_infl"]
df_anchor["anchor_cc"]     = np.log(1.0 + df_anchor["anchor_simple"])
df_anchor = (
    df_anchor
    .loc[(df_anchor["date"] >= START) & (df_anchor["date"] <= END)]
    .sort_values("date")
    .reset_index(drop=True)
)

    # endregion

# endregion

# region 2. ESTIMATION FUNCTIONS

def _to_ql_date(ts: pd.Timestamp) -> ql.Date:
    return ql.Date(ts.day, ts.month, ts.year)


def _ql_settle(ts: pd.Timestamp) -> pd.Timestamp:
    """Advance ts by SETTLEMENT_DAYS good business days (Sweden calendar)."""
    ql_d = _to_ql_date(ts)
    ql_s = CALENDAR.advance(ql_d, SETTLEMENT_DAYS, ql.Days)
    return pd.Timestamp(ql_s.year(), ql_s.month(), ql_s.dayOfMonth())


def swedish_reference_kpi(
    settlement_date: pd.Timestamp,
    kpi_by_month: pd.Series,
    lag_months: int = 3,
) -> float:
    """
    Swedish National Debt Office reference KPI at a given settlement date.

    Uses a 3-month lag with linear day-weighted interpolation:

        KPI_ref(t_s) = KPI_{t_s-3m} + (d-1)/30 * (KPI_{t_s-2m} - KPI_{t_s-3m})

    where d is the day of the settlement month, capped at 30.
    """
    m0 = settlement_date.to_period("M") - lag_months
    m1 = m0 + 1

    if m0 not in kpi_by_month.index or m1 not in kpi_by_month.index:
        raise ValueError(
            f"KPI missing for interpolation: need {m0} and {m1}"
        )

    kpi_0 = float(kpi_by_month.loc[m0])
    kpi_1 = float(kpi_by_month.loc[m1])

    day = min(settlement_date.day, 30)
    return kpi_0 + (day - 1) / 30.0 * (kpi_1 - kpi_0)


def build_il_bond_helpers(
    cross_section: pd.DataFrame,
    settlement_date: pd.Timestamp,
    kpi_by_month: pd.Series,
):
    """
    Convert nominal clean prices to real clean prices, then build QuantLib
    bond objects and price helpers for the NS fitting.

    The index ratio at settlement is KPI_ref(t_s) / KPI_base, computed using
    the Swedish 3-month lag convention.  All helpers receive real coupons and
    a real face value of 100, so QuantLib treats the fitted curve as a real
    yield curve throughout.

    Returns
    -------
    helpers       : list of ql.FixedRateBondHelper
    bonds         : list of ql.FixedRateBond
    real_prices   : np.ndarray of real clean prices
    real_yields   : np.ndarray of real YTMs (compounded annual, for diagnostics)
    maturities    : list of maturity date Timestamps
    reference_kpi : float — the interpolated reference KPI at settlement
    """
    reference_kpi = swedish_reference_kpi(settlement_date, kpi_by_month)
    ql_settle     = _to_ql_date(settlement_date)

    helpers, bonds, real_prices, real_yields, maturities = [], [], [], [], []

    for _, row in cross_section.iterrows():
        index_ratio      = reference_kpi / float(row["bas_kpi"])
        real_clean_price = float(row["price"]) / index_ratio

        ql_issue    = _to_ql_date(row["issue_date"])
        ql_maturity = _to_ql_date(row["maturity_date"])
        coupon_rate = float(row["coupon"])

        schedule = ql.Schedule(
            ql_issue, ql_maturity, TENOR, CALENDAR,
            BUS_CONV, BUS_CONV, DATE_GEN_RULE, END_OF_MONTH,
        )
        bond = ql.FixedRateBond(
            SETTLEMENT_DAYS, FACE, schedule,
            [coupon_rate], DAY_COUNT, BUS_CONV, REDEMPTION, ql_issue,
        )
        helper = ql.FixedRateBondHelper(
            ql.QuoteHandle(ql.SimpleQuote(real_clean_price)),
            SETTLEMENT_DAYS, FACE, schedule,
            [coupon_rate], DAY_COUNT, BUS_CONV, REDEMPTION, ql_issue,
        )

        # Real YTM: back-solve from real clean price (compounded annual).
        # Swedish IL bonds can have deeply negative real yields, so a single
        # starting guess of 0.0 frequently fails to bracket the root.
        # Try a sequence of guesses spanning the plausible real yield range.
        price_obj = ql.BondPrice(real_clean_price, ql.BondPrice.Clean)
        real_ytm  = np.nan
        for _guess in [-0.05, -0.02, 0.0, 0.02, 0.05]:
            try:
                real_ytm = float(ql.BondFunctions.bondYield(
                    bond, price_obj, DAY_COUNT,
                    ql.Compounded, ql.Annual,
                    ql_settle, 1e-10, 1000, _guess,
                ))
                break
            except RuntimeError:
                continue

        bonds.append(bond)
        helpers.append(helper)
        real_prices.append(real_clean_price)
        real_yields.append(float(real_ytm))
        maturities.append(row["maturity_date"])

    return (
        helpers, bonds,
        np.array(real_prices), np.array(real_yields),
        maturities, reference_kpi,
    )


def build_bootstrap_curve(helpers: list) -> ql.PiecewiseLogCubicDiscount:
    """
    Piecewise log-cubic real discount curve used as a non-parametric reference
    and for pricing synthetic midpoint instruments.
    """
    curve = ql.PiecewiseLogCubicDiscount(
        SETTLEMENT_DAYS, CALENDAR, helpers, DAY_COUNT
    )
    curve.enableExtrapolation()
    return curve


def build_synthetic_helpers(
    bonds_actual: list,
    pre_curve,
    ql_eval_date: ql.Date,
) -> list:
    """
    Insert a synthetic real zero-coupon bond at the midpoint between each
    adjacent pair of actual bond maturities, priced from the bootstrapped real
    pre-curve.  Fills maturity gaps and regularises the NS fit (used in the
    extended fit only).
    """
    ref_date       = CALENDAR.advance(ql_eval_date, SETTLEMENT_DAYS, ql.Days)
    maturity_dates = sorted({b.maturityDate() for b in bonds_actual})
    synth_helpers  = []

    for d1, d2 in zip(maturity_dates[:-1], maturity_dates[1:]):
        t1 = DAY_COUNT.yearFraction(ref_date, d1)
        t2 = DAY_COUNT.yearFraction(ref_date, d2)

        if t1 <= 0.0 or t2 <= t1:
            continue

        t_mid      = 0.5 * (t1 + t2)
        months_mid = max(1, int(round(t_mid * 12)))
        mid_date   = CALENDAR.advance(ref_date, ql.Period(months_mid, ql.Months))

        if not (d1 < mid_date < d2):
            continue

        synth_price = FACE * pre_curve.discount(mid_date)
        schedule    = ql.Schedule(
            ref_date, mid_date, ql.Period(ql.Once), CALENDAR,
            BUS_CONV, BUS_CONV, ql.DateGeneration.Forward, False,
        )
        synth_helpers.append(
            ql.FixedRateBondHelper(
                ql.QuoteHandle(ql.SimpleQuote(float(synth_price))),
                SETTLEMENT_DAYS, FACE, schedule,
                [0.0], DAY_COUNT, BUS_CONV, REDEMPTION, ref_date,
            )
        )

    return synth_helpers


def compute_anchors(
    bonds_actual: list,
    pre_curve,
    anchor_cc: float,
) -> tuple:
    """
    Derive data-driven starting values for b0 and b1:
        b0          = long-end continuous real zero rate from the bootstrap
        b0 + b1     = anchor_cc  (= log(1 + SSVX_1m - trail_infl_12m))
                      ≈ real cc short rate at t → 0
    """
    max_maturity = max(b.maturityDate() for b in bonds_actual)
    b0 = pre_curve.zeroRate(max_maturity, DAY_COUNT, ql.Continuous).rate()
    b1 = anchor_cc - b0
    return b0, b1


def _build_fitting_method() -> ql.NelsonSiegelFitting:
    """Assemble the QuantLib NelsonSiegelFitting object from pipeline settings."""
    kappa_min = 1.0 / TAU_BOUNDS[1]
    kappa_max = 1.0 / TAU_BOUNDS[0]

    lower, upper = ql.Array(4), ql.Array(4)
    for i, (lo, hi) in enumerate([
        B0_BOUNDS, B1_BOUNDS, B2_BOUNDS,
        (kappa_min, kappa_max),
    ]):
        lower[i], upper[i] = lo, hi

    l2 = ql.Array(4)
    for i, w in enumerate(L2_WEIGHTS):
        l2[i] = w

    return ql.NelsonSiegelFitting(
        ql.Array(), ql.Simplex(SIMPLEX_LAMBDA),
        l2, 0.0, 50.0,
        ql.NonhomogeneousBoundaryConstraint(lower, upper),
    )


def fit_multistart(helpers: list, b0: float, b1: float):
    """
    Run NS optimisation from every point on the (b2, tau) grid; return the
    globally best result by minimum objective value.

    Returns: curve, params [b0, b1, b2, k1], objective value, winning start (b2, tau)
    """
    fitting = _build_fitting_method()
    best_obj, best_curve, best_params, best_start = float("inf"), None, None, None

    for b2s in B2_GRID:
        for tau in TAU_GRID:
            guess    = ql.Array(4)
            guess[0] = b0
            guess[1] = b1
            guess[2] = b2s
            guess[3] = 1.0 / tau

            try:
                curve = ql.FittedBondDiscountCurve(
                    SETTLEMENT_DAYS, CALENDAR, helpers,
                    DAY_COUNT, fitting, ACCURACY, MAX_EVALS, guess,
                )
                obj = curve.fitResults().minimumCostValue()

                if obj < best_obj:
                    best_obj    = obj
                    best_curve  = curve
                    best_params = list(curve.fitResults().solution())
                    best_start  = (b2s, tau)

            except RuntimeError:
                continue

    return best_curve, best_params, best_obj, best_start


def evaluate_fit(
    bonds_actual: list,
    real_prices:  np.ndarray,
    real_yields:  np.ndarray,
    curve,
) -> dict:
    """
    Price actual bonds from the fitted curve and compute residuals and RMSEs.
    All quantities are in real terms.
    """
    handle = ql.YieldTermStructureHandle(curve)
    engine = ql.DiscountingBondEngine(handle)

    model_prices, model_yields = [], []
    for bond in bonds_actual:
        bond.setPricingEngine(engine)
        model_prices.append(float(bond.cleanPrice()))
        model_yields.append(float(bond.bondYield(DAY_COUNT, ql.Compounded, ql.Annual)))

    model_prices = np.array(model_prices)
    model_yields = np.array(model_yields)

    return {
        "rmse_price":         float(np.sqrt(np.mean((real_prices - model_prices) ** 2))),
        "rmse_yield_bp":      float(np.sqrt(np.mean((real_yields - model_yields) ** 2))) * 1e4,
        "price_residuals":    real_prices  - model_prices,
        "yield_residuals_bp": (real_yields - model_yields) * 1e4,
        "model_prices":       model_prices,
        "model_yields":       model_yields,
    }


def estimate_ns(
    eval_date:    pd.Timestamp,
    df_il:        pd.DataFrame,
    df_anc:       pd.DataFrame,
    kpi_by_month: pd.Series,
) -> dict:
    """
    Full NS estimation pipeline for a single evaluation date.

    Two fits are produced:
        actual   — fitted to observed real SGBi prices only
        extended — fitted to observed prices + bootstrapped synthetic midpoints

    Returns a dict with parameters, diagnostics, and QL objects.
    Keys prefixed with '_' hold QL objects for plotting; they are stripped
    automatically before panel export.
    """
    cross_section = (
        df_il
        .loc[df_il["date"] == eval_date]
        .dropna(subset=["price", "coupon", "issue_date", "maturity_date", "bas_kpi"])
        .sort_values("maturity_date")
        .reset_index(drop=True)
    )

    if len(cross_section) < 2:
        raise ValueError(
            f"{eval_date.date()}: fewer than 2 instruments available for bootstrap"
        )

    ql_eval_date = _to_ql_date(eval_date)
    ql.Settings.instance().evaluationDate = ql_eval_date

    settlement_date = _ql_settle(eval_date)

    # ---- Short-rate anchor
    anc_row = df_anc.loc[df_anc["date"] == eval_date]
    if anc_row.empty:
        raise ValueError(f"No short-rate anchor for {eval_date.date()}")
    anchor_cc = float(anc_row["anchor_cc"].iloc[0])

    # ---- Build actual helpers (nominal → real price conversion inside)
    helpers_actual, bonds_actual, real_prices, real_yields, maturities, reference_kpi = \
        build_il_bond_helpers(cross_section, settlement_date, kpi_by_month)

    # ---- Bootstrap pre-curve and synthetic midpoint helpers
    pre_curve        = build_bootstrap_curve(helpers_actual)
    synth_helpers    = build_synthetic_helpers(bonds_actual, pre_curve, ql_eval_date)
    helpers_extended = helpers_actual + synth_helpers

    # ---- Starting anchors from bootstrap and SSVX/CPI anchor
    b0, b1 = compute_anchors(bonds_actual, pre_curve, anchor_cc)

    # ---- Multistart NS fit — actual
    curve_actual, params_actual, obj_actual, start_actual = \
        fit_multistart(helpers_actual, b0, b1)
    if curve_actual is None:
        raise RuntimeError(f"Multistart failed (actual) on {eval_date.date()}")

    # ---- Multistart NS fit — extended
    curve_ext, params_ext, obj_ext, start_ext = \
        fit_multistart(helpers_extended, b0, b1)
    if curve_ext is None:
        raise RuntimeError(f"Multistart failed (extended) on {eval_date.date()}")

    # ---- Diagnostics
    diag_actual = evaluate_fit(bonds_actual, real_prices, real_yields, curve_actual)
    diag_ext    = evaluate_fit(bonds_actual, real_prices, real_yields, curve_ext)

    return {
        # NS parameters — actual fit
        "b0_actual": params_actual[0], "b1_actual": params_actual[1],
        "b2_actual": params_actual[2], "k1_actual": params_actual[3],
        # NS parameters — extended fit
        "b0_ext":    params_ext[0],    "b1_ext":    params_ext[1],
        "b2_ext":    params_ext[2],    "k1_ext":    params_ext[3],
        # Fit diagnostics
        "rmse_price_actual":    diag_actual["rmse_price"],
        "rmse_price_ext":       diag_ext["rmse_price"],
        "rmse_yield_bp_actual": diag_actual["rmse_yield_bp"],
        "rmse_yield_bp_ext":    diag_ext["rmse_yield_bp"],
        "objective_actual":     obj_actual,
        "objective_ext":        obj_ext,
        # Counts
        "n_bonds": len(helpers_actual),
        "n_synth": len(synth_helpers),
        # Metadata
        "anchor_cc":    anchor_cc,
        "reference_kpi": reference_kpi,
        # Winning starting values
        "start_b2_actual": start_actual[0], "start_tau_actual": start_actual[1],
        "start_b2_ext":    start_ext[0],    "start_tau_ext":    start_ext[1],
        # QL objects — for single-date plots only, stripped before panel export
        "_eval_date":    eval_date,
        "_ql_eval_date": ql_eval_date,
        "_curve_actual": curve_actual,
        "_curve_ext":    curve_ext,
        "_pre_curve":    pre_curve,
        "_bonds_actual": bonds_actual,
        "_real_prices":  real_prices,
        "_real_yields":  real_yields,
        "_maturities":   maturities,
        "_diag_actual":  diag_actual,
        "_diag_ext":     diag_ext,
    }

# endregion

# region 3. DIAGNOSTICS AND PLOTS
# Diagnostic print functions and plot functions for both modes.
# All functions take the result dict (single-date) or results DataFrame (panel)
# as input and are fully independent of the estimation step.

def print_single_diagnostics(result: dict):
    """Print NS parameters and fit quality for a single-date result."""
    d = result["_eval_date"].date()
    print(f"\n{'─'*52}")
    print(f"  Single-date diagnostics — {d}")
    print(f"{'─'*52}")
    print(f"  Instruments:  {result['n_bonds']} bonds,  {result['n_synth']} synthetic")
    print(f"  Anchor (cc):  {result['anchor_cc']:.6f}")
    print(f"  Ref KPI:      {result['reference_kpi']:.4f}")
    print()
    print(f"  {'Parameter':<12}  {'Actual':>12}  {'Extended':>12}")
    print(f"  {'─'*38}")
    for key in ["b0", "b1", "b2", "k1"]:
        print(f"  {key:<12}  {result[key+'_actual']:>12.6f}  {result[key+'_ext']:>12.6f}")
    print()
    print(f"  {'RMSE price':<20}  {result['rmse_price_actual']:>10.6f}  {result['rmse_price_ext']:>10.6f}")
    print(f"  {'RMSE yield (bp)':<20}  {result['rmse_yield_bp_actual']:>10.2f}  {result['rmse_yield_bp_ext']:>10.2f}")
    print(f"  {'Objective':<20}  {result['objective_actual']:>10.6f}  {result['objective_ext']:>10.6f}")
    print(f"{'─'*52}\n")


def print_panel_diagnostics(results: pd.DataFrame, failed_df: pd.DataFrame):
    """Print summary statistics for a completed panel estimation."""
    print(f"\n{'─'*60}")
    print(f"  Panel diagnostics — {len(results)} months fitted, "
          f"{len(failed_df)} failed")
    print(f"{'─'*60}")

    cols = {
        "rmse_yield_bp_actual": "RMSE yield bp (actual)",
        "rmse_yield_bp_ext":    "RMSE yield bp (extended)",
        "objective_actual":     "Objective (actual)",
        "objective_ext":        "Objective (extended)",
        "n_bonds":              "N bonds",
        "n_synth":              "N synthetic",
    }
    print(f"\n  {'Metric':<28}  {'Mean':>10}  {'Min':>10}  {'Max':>10}")
    print(f"  {'─'*62}")
    for col, label in cols.items():
        if col in results.columns:
            print(f"  {label:<28}  "
                  f"{results[col].mean():>10.4f}  "
                  f"{results[col].min():>10.4f}  "
                  f"{results[col].max():>10.4f}")

    if not failed_df.empty:
        print(f"\n  Failed dates:")
        for _, row in failed_df.iterrows():
            print(f"    {row['date'].date()}  —  {row['error']}")

    print(f"{'─'*60}\n")


def _style_ax(ax):
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("black")
    ax.tick_params(axis="both", which="major", direction="in",
                   top=True, right=True, pad=5)


def _save_fig(fig, stem: str):
    """Save figure to FIG_SAVE_DIR if configured."""
    if FIG_SAVE_DIR is None:
        return
    os.makedirs(FIG_SAVE_DIR, exist_ok=True)
    path = os.path.join(FIG_SAVE_DIR, f"{stem}.{FIG_FORMAT}")
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {path}")


def plot_curves(result: dict):
    """
    Plot NS fitted real zero-coupon curves (actual + extended + bootstrap)
    and observed real yields for a single evaluation date.
    """
    eval_date    = result["_eval_date"]
    ql_eval_date = result["_ql_eval_date"]
    ql.Settings.instance().evaluationDate = ql_eval_date

    curve_actual = result["_curve_actual"]
    curve_ext    = result["_curve_ext"]
    pre_curve    = result["_pre_curve"]
    bonds_actual = result["_bonds_actual"]
    real_yields  = result["_real_yields"]
    maturities   = result["_maturities"]

    max_date = max(b.maturityDate() for b in bonds_actual)
    comp     = ql.Continuous

    times, z_actual, z_ext, z_boot = [], [], [], []
    for months in range(1, int(MAX_TTM_YEARS * 12) + 13):
        d = CALENDAR.advance(ql_eval_date, ql.Period(months, ql.Months))
        if d > max_date:
            break
        times.append(months / 12.0)
        z_actual.append(curve_actual.zeroRate(d, DAY_COUNT, comp).rate())
        z_ext.append(curve_ext.zeroRate(d, DAY_COUNT, comp).rate())
        z_boot.append(pre_curve.zeroRate(d, DAY_COUNT, comp).rate())

    mat_years = [(pd.Timestamp(m) - eval_date).days / 365.25 for m in maturities]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    fig.patch.set_facecolor("white")
    ax.yaxis.grid(True, color="#CCCCCC", linewidth=0.5, linestyle="-", zorder=0)
    ax.set_axisbelow(True)

    ax.plot(times, [v * 100 for v in z_boot],
            color=COLOR_BOOT,     lw=1.4, ls="-.", zorder=2, label="Bootstrap")
    ax.plot(times, [v * 100 for v in z_actual],
            color=COLOR_ACTUAL,   lw=1.6, ls="-", zorder=3, label="Actual")
    ax.plot(times, [v * 100 for v in z_ext],
            color=COLOR_EXTENDED, lw=1.6, ls="-",  zorder=4, label="Extended")
    ax.scatter(mat_years, real_yields * 100,
               color=COLOR_OBSERVED, marker="D", s=50, zorder=5, label="Observed")

    ax.set_xlabel("Years to maturity")
    ax.set_ylabel("Yield (%)")
    #ax.set_title(f"Real yield curve — {eval_date.date()}", pad=8)
    ax.set_xlim(0, max(times) + 0.5)
    _style_ax(ax)
    ax.legend(loc="best", frameon=True, fancybox=False,
              edgecolor="#AAAAAA", facecolor="white", framealpha=1.0,
              borderpad=0.6, handlelength=2.2)
    plt.tight_layout(pad=0.6)
    _save_fig(fig, f"curve_{eval_date.strftime('%Y-%m')}_linker")
    plt.show()
    plt.close(fig)


def plot_price_residuals(result: dict):
    """
    Plot real price residuals (observed − model) by maturity for actual and
    extended fits.
    """
    eval_date    = result["_eval_date"]
    ql_eval_date = result["_ql_eval_date"]
    ql.Settings.instance().evaluationDate = ql_eval_date

    maturities  = result["_maturities"]
    diag_actual = result["_diag_actual"]
    diag_ext    = result["_diag_ext"]

    mat_years = [(pd.Timestamp(m) - eval_date).days / 365.25 for m in maturities]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    fig.patch.set_facecolor("white")
    ax.yaxis.grid(True, color="#CCCCCC", linewidth=0.5, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    ax.axhline(0, color="#666666", lw=0.8, ls="--", zorder=1)

    ax.scatter(mat_years, diag_actual["price_residuals"],
               color=COLOR_ACTUAL,   s=35, marker="^", zorder=3, label="Actual")
    ax.scatter(mat_years, diag_ext["price_residuals"],
               color=COLOR_EXTENDED, s=35, marker="o", zorder=4, label="Extended")

    ax.set_xlabel("Years to maturity")
    ax.set_ylabel("Real price residual")
    ax.set_title(f"Price residuals — {eval_date.date()}", pad=8)
    _style_ax(ax)
    ax.legend(loc="best", frameon=True, fancybox=False,
              edgecolor="#AAAAAA", facecolor="white", framealpha=1.0,
              borderpad=0.6, handlelength=2.2)
    plt.tight_layout(pad=0.6)
    _save_fig(fig, f"residuals_{eval_date.strftime('%Y-%m')}_linker")
    plt.show()
    plt.close(fig)

# endregion

# region 4. SINGLE-DATE ESTIMATION
# estimate_ns() fits both the actual and extended NS curves for one date.
# Called directly in RUN SINGLE DATE and inside run_panel() for RUN PANEL.


def resolve_single_date(ym_str: str) -> pd.Timestamp:
    """Resolve a 'yyyy-mm' string to the last SGBi trading day in that month."""
    target = pd.Period(ym_str, freq="M")
    if target not in month_last_trade.index:
        raise ValueError(f"No SGBi data for '{ym_str}'.")
    return pd.Timestamp(month_last_trade[target])

# endregion

# region 5. PANEL ESTIMATION

# ---- Export column order
EXPORT_COLS = [
    "date",
    "b0_actual", "b1_actual", "b2_actual", "k1_actual", "tau_actual",
    "b0_ext",    "b1_ext",    "b2_ext",    "k1_ext",    "tau_ext",
    "rmse_price_actual",    "rmse_price_ext",
    "rmse_yield_bp_actual", "rmse_yield_bp_ext",
    "objective_actual",     "objective_ext",
    "n_bonds", "n_synth",
    "anchor_cc", "reference_kpi",
    "start_b2_actual", "start_tau_actual",
    "start_b2_ext",    "start_tau_ext",
]

# ---- Worker functions (module-level — required for multiprocessing pickling)

_shared_il  = None
_shared_anc = None
_shared_kpi = None


def _init_worker(df_il, df_anc, kpi):
    """Initialise shared DataFrames in each worker process (fork context)."""
    global _shared_il, _shared_anc, _shared_kpi
    _shared_il  = df_il
    _shared_anc = df_anc
    _shared_kpi = kpi


def _panel_worker(eval_date: pd.Timestamp):
    """
    Estimate NS for a single date.  Runs inside a worker process.
    Returns ('ok', row_dict) on success or ('fail', error_dict) on failure.
    QL objects are stripped before returning — they cannot be pickled.
    """
    try:
        result = estimate_ns(eval_date, _shared_il, _shared_anc, _shared_kpi)
        row = {k: v for k, v in result.items() if not k.startswith("_")}
        row["date"]       = eval_date
        row["tau_actual"] = 1.0 / row["k1_actual"]
        row["tau_ext"]    = 1.0 / row["k1_ext"]
        return "ok", row
    except Exception as e:
        return "fail", {"date": eval_date, "error": str(e)}


def run_panel(df_il, df_anc, kpi, n_workers=None):
    """
    Fit NS for every month in the panel using a parallel worker pool.
    Each month is estimated independently in a separate process.

    Parameters
    ----------
    n_workers : int or None
        Number of worker processes.  None uses all available CPU cores.

    Returns
    -------
    results   : pd.DataFrame of successful fits, columns ordered by EXPORT_COLS
    failed_df : pd.DataFrame of failed dates with error messages
    """
    if n_workers is None:
        n_workers = mp.cpu_count()

    dates = [pd.Timestamp(d) for d in month_last_trade]
    n_total = len(dates)
    print(f"Panel: {n_total} months, {n_workers} workers")

    rows, failed = [], []

    ctx = mp.get_context("fork")
    with ctx.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(df_il, df_anc, kpi),
    ) as pool:
        for i, (status, data) in enumerate(
            pool.imap_unordered(_panel_worker, dates), start=1
        ):
            if status == "ok":
                rows.append(data)
            else:
                failed.append(data)
                print(f"  FAILED {data['date'].date()}: {data['error']}")
            if i % 50 == 0 or i == n_total:
                print(f"  {i}/{n_total} complete")

    results   = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    failed_df = pd.DataFrame(failed)
    results   = results.loc[:, [c for c in EXPORT_COLS if c in results.columns]]
    return results, failed_df


def export_panel(results: pd.DataFrame, failed_df: pd.DataFrame):
    """
    Export panel results to OUTPUT_FILE.
    Sheet 'fit_params' holds results; sheet 'failed' is added if any months failed.
    """
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        results.to_excel(writer, sheet_name="fit_params", index=False)
        if not failed_df.empty:
            failed_df.to_excel(writer, sheet_name="failed", index=False)
    print(f"Exported {len(results)} months to {OUTPUT_FILE}")

# endregion

# region 6. RUN SINGLE DATE
# Set SINGLE_YM below, then run Step 1 and Step 2 individually in sequence.
# Step 2 requires Step 1 to have been run first (result must be in memory).

SINGLE_YM = "2023-04"   # yyyy-mm — resolved automatically to last trading day

# Step 1 — Estimate
eval_date     = resolve_single_date(SINGLE_YM)
result        = estimate_ns(eval_date, df_SGBILs, df_anchor, kpi_by_month)

# Step 2 — Diagnostics and plots
print_single_diagnostics(result)
plot_curves(result)
plot_price_residuals(result)

# endregion

# region 7. RUN PANEL
# Run Steps 1, 2, and 3 individually in sequence.  Step 1 must complete before
# running 2 or 3.  Steps 2 and 3 are independent of each other and can be
# re-run without re-running Step 1, as long as results is still in memory.

# Step 1 — Estimate all months  [slow — uses all CPU cores via fork]
results, failed_df = run_panel(df_SGBILs, df_anchor, kpi_by_month)

# Step 2 — Print panel diagnostics
print_panel_diagnostics(results, failed_df)

# Step 3 — Export to Excel
export_panel(results, failed_df)

# endregion

# region 8. RMSE PLOTS
# Run after the panel has been estimated and exported (region 7, Steps 1–3).
# Reads results from OUTPUT_FILE and df_SGBILs from script startup.
#
# Step 1 — Overall yield RMSE time series (actual and extended, all bonds)
# Step 2 — Binned yield RMSE: short 1-2y, medium 2–5y, long 5y+
#
# Bin RMSE method: for each bond, the model real clean price is obtained by
# discounting every real cash flow at the NS zero rate for that cash flow's
# exact time to payment (continuously compounded), summing to a dirty real
# price, and subtracting accrued interest. The model real YTM is then back-
# solved from that price. The observed real YTM is back-solved from the real
# clean price derived via the Swedish 3-month-lag index ratio, identically to
# what is fed to the optimizer during estimation.

    # region 8.1. Functions

def _ns_zero_rate(t: float, b0: float, b1: float, b2: float, k1: float) -> float:
    """NS continuously compounded zero rate at a single maturity t (scalar)."""
    x1 = k1 * t
    l1 = 1.0 if abs(x1) < 1e-10 else (1.0 - np.exp(-x1)) / x1
    l2 = l1 - np.exp(-x1)
    return b0 + b1 * l1 + b2 * l2


def _il_bond_model_yield(
    row,
    eval_date: pd.Timestamp,
    b0: float, b1: float, b2: float, k1: float,
) -> float:
    """
    Compute the model-implied real YTM for a single SGBi bond.

    Each real cash flow is discounted at the NS zero rate for its exact time
    to payment, summed to a dirty real price, accrued interest subtracted,
    then back-solved for the annual compounded real YTM.
    Returns np.nan if pricing fails.
    """
    ql_eval = _to_ql_date(eval_date)
    ql.Settings.instance().evaluationDate = ql_eval

    ql_issue    = _to_ql_date(pd.Timestamp(row["issue_date"]))
    ql_maturity = _to_ql_date(pd.Timestamp(row["maturity_date"]))

    schedule = ql.Schedule(
        ql_issue, ql_maturity, TENOR, CALENDAR,
        BUS_CONV, BUS_CONV, DATE_GEN_RULE, END_OF_MONTH,
    )
    bond = ql.FixedRateBond(
        SETTLEMENT_DAYS, FACE, schedule,
        [float(row["coupon"])], DAY_COUNT, BUS_CONV, REDEMPTION, ql_issue,
    )

    sett = bond.settlementDate()
    dirty = 0.0
    for cf in bond.cashflows():
        if cf.hasOccurred(sett):
            continue
        t = DAY_COUNT.yearFraction(sett, cf.date())
        if t <= 0.0:
            continue
        dirty += cf.amount() * np.exp(-_ns_zero_rate(t, b0, b1, b2, k1) * t)

    real_clean = dirty - bond.accruedAmount(sett)
    price_obj  = ql.BondPrice(real_clean, ql.BondPrice.Clean)

    for _guess in [-0.05, -0.02, 0.0, 0.02, 0.05]:
        try:
            return float(ql.BondFunctions.bondYield(
                bond, price_obj, DAY_COUNT,
                ql.Compounded, ql.Annual,
                sett, 1e-10, 1000, _guess,
            ))
        except RuntimeError:
            continue
    return np.nan


def _build_bin_rmse_panel(
    results:      pd.DataFrame,
    df_il:        pd.DataFrame,
    kpi_by_month: pd.Series,
) -> pd.DataFrame:
    """
    For every date in results, compute the real yield residual (observed minus
    model) for each bond under both the actual and extended NS fits, then
    aggregate into monthly RMSE by maturity bin.

    Observed real YTM: back-solved from the real clean price obtained via the
    Swedish 3-month-lag index ratio — identical to the price fed to the
    optimizer during estimation.

    Model real YTM: back-solved from the NS-implied real clean price, with
    each cash flow discounted at the NS zero rate for its exact time to payment.

    Returns a DataFrame with columns:
        date, bin, rmse_actual (bp), rmse_ext (bp)
    """
    bin_edges  = [1.0, 2.0, 5.0, np.inf]
    bin_labels = ["1\u20132y", "2\u20135y", "5y+"]

    rows = []
    for _, p in results.iterrows():
        ed      = pd.Timestamp(p["date"])
        ql_eval = _to_ql_date(ed)
        ql.Settings.instance().evaluationDate = ql_eval

        cs = (
            df_il
            .loc[df_il["date"] == ed]
            .dropna(subset=["price", "coupon", "issue_date", "maturity_date", "bas_kpi"])
            .sort_values("maturity_date")
            .reset_index(drop=True)
        )
        if cs.empty:
            continue

        # Reference KPI at settlement — same convention as estimation
        sett = _ql_settle(ed)
        try:
            ref_kpi = swedish_reference_kpi(sett, kpi_by_month)
        except ValueError:
            continue

        # Real clean prices and observed real YTMs
        ql_sett    = _to_ql_date(sett)
        obs_yields = []
        for _, row in cs.iterrows():
            index_ratio = ref_kpi / float(row["bas_kpi"])
            real_price  = float(row["price"]) / index_ratio

            ql_issue    = _to_ql_date(row["issue_date"])
            ql_maturity = _to_ql_date(row["maturity_date"])
            sch = ql.Schedule(
                ql_issue, ql_maturity, TENOR, CALENDAR,
                BUS_CONV, BUS_CONV, DATE_GEN_RULE, END_OF_MONTH,
            )
            bnd = ql.FixedRateBond(
                SETTLEMENT_DAYS, FACE, sch,
                [float(row["coupon"])], DAY_COUNT, BUS_CONV, REDEMPTION, ql_issue,
            )
            po  = ql.BondPrice(real_price, ql.BondPrice.Clean)
            ytm = np.nan
            for _g in [-0.05, -0.02, 0.0, 0.02, 0.05]:
                try:
                    ytm = float(ql.BondFunctions.bondYield(
                        bnd, po, DAY_COUNT, ql.Compounded, ql.Annual,
                        ql_sett, 1e-10, 1000, _g,
                    ))
                    break
                except RuntimeError:
                    continue
            obs_yields.append(ytm)

        cs["yield_real_obs"] = obs_yields
        cs["ttm_years"] = (cs["maturity_date"] - cs["date"]).dt.days / 365.25
        cs["bin"] = pd.cut(
            cs["ttm_years"], bins=bin_edges,
            labels=bin_labels, right=False,
        )

        # Model real YTMs for both fits
        for suffix, pcols in (
            ("actual", ("b0_actual", "b1_actual", "b2_actual", "k1_actual")),
            ("ext",    ("b0_ext",    "b1_ext",    "b2_ext",    "k1_ext")),
        ):
            b0, b1, b2, k1 = (float(p[c]) for c in pcols)
            cs[f"yield_model_{suffix}"] = cs.apply(
                lambda r: _il_bond_model_yield(r, ed, b0, b1, b2, k1),
                axis=1,
            )
            cs[f"resid_bp_{suffix}"] = (
                (cs["yield_real_obs"] - cs[f"yield_model_{suffix}"]) * 1e4
            )

        for bin_label, grp in cs.groupby("bin", observed=True):
            row_out = {"date": ed, "bin": bin_label}
            for suffix in ("actual", "ext"):
                resid = grp[f"resid_bp_{suffix}"].dropna()
                row_out[f"rmse_{suffix}"] = (
                    float(np.sqrt((resid ** 2).mean())) if len(resid) > 0 else np.nan
                )
            rows.append(row_out)

    return pd.DataFrame(rows).sort_values(["date", "bin"]).reset_index(drop=True)


def plot_panel_rmse(results: pd.DataFrame):
    """Overall yield RMSE time series for actual and extended fits."""
    dates = pd.to_datetime(results["date"])

    fig, ax = plt.subplots(figsize=(FIG_WIDTH * 1.4, FIG_HEIGHT))
    fig.patch.set_facecolor("white")
    ax.yaxis.grid(True, color="#CCCCCC", linewidth=0.5, linestyle="-", zorder=0)
    ax.set_axisbelow(True)

    ax.plot(dates, results["rmse_yield_bp_actual"],
            color=COLOR_ACTUAL,   lw=1.8, ls="--", zorder=3, label="Actual fit")
    ax.plot(dates, results["rmse_yield_bp_ext"],
            color=COLOR_EXTENDED, lw=1.8, ls="-",  zorder=2, label="Extended fit")

    ax.set_ylabel("Yield RMSE (bp)")
    ax.set_xlim(dates.min(), dates.max())
    ax.set_ylim(0, 16)
    ax.set_yticks([2, 4, 6, 8, 10, 12, 14])
    _style_ax(ax)
    ax.legend(loc="upper right", frameon=True, fancybox=False,
              edgecolor="#AAAAAA", facecolor="white", framealpha=1.0,
              borderpad=0.6, handlelength=2.2)
    plt.tight_layout(pad=0.6)
    _save_fig(fig, "rmse_real")
    plt.show()
    plt.close(fig)


def plot_panel_rmse_bins(bin_df: pd.DataFrame):
    """
    Three subplots (1-2y / 2-5y / 5y+), each showing the monthly real yield
    RMSE time series for the actual (charcoal dashed) and extended (blue solid)
    fits. Subplots share the y-axis.
    Missing months within a bin produce gaps in the line rather than
    interpolation, achieved by reindexing to the full panel date range.
    """
    bin_labels      = ["1\u20132y", "2\u20135y", "5y+"]
    all_panel_dates = (pd.to_datetime(bin_df["date"])
                       .drop_duplicates().sort_values().reset_index(drop=True))

    fig, axes = plt.subplots(
        1, 3,
        figsize=(FIG_WIDTH * 2.5, FIG_HEIGHT),
        sharey=True,
    )
    fig.patch.set_facecolor("white")

    for ax, label in zip(axes, bin_labels):
        sub = bin_df[bin_df["bin"] == label].sort_values("date")

        # Reindex to the full panel date range so missing months become NaN,
        # which matplotlib renders as gaps rather than interpolated segments.
        sub_full = (
            pd.DataFrame({"date": all_panel_dates})
            .merge(sub[["date", "rmse_actual", "rmse_ext"]], on="date", how="left")
        )
        dates = sub_full["date"]

        ax.yaxis.grid(True, color="#CCCCCC", linewidth=0.5, linestyle="-", zorder=0)
        ax.set_axisbelow(True)

        ax.plot(dates, sub_full["rmse_actual"],
                color=COLOR_ACTUAL,   lw=1.6, ls="--", zorder=3, label="Actual fit")
        ax.plot(dates, sub_full["rmse_ext"],
                color=COLOR_EXTENDED, lw=1.6, ls="-",  zorder=2, label="Extended fit")

        ax.set_title(label, pad=8)
        ax.set_xlim(all_panel_dates.min(), all_panel_dates.max())
        ax.set_ylim(0, 22)
        ax.set_yticks([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
        _style_ax(ax)

    axes[0].set_ylabel("Yield RMSE (bp)")
    axes[2].legend(
        loc="upper right", frameon=True, fancybox=False,
        edgecolor="#AAAAAA", facecolor="white", framealpha=1.0,
        borderpad=0.6, handlelength=2.2,
    )
    plt.tight_layout(pad=0.6)
    _save_fig(fig, "rmse_bins_real")
    plt.show()
    plt.close(fig)

    # endregion

    # region 8.2. Execution
# Reads panel results from OUTPUT_FILE.
# df_SGBILs and kpi_by_month are always available from script startup.

# Step 1 — Load panel results
results = pd.read_excel(OUTPUT_FILE, sheet_name="fit_params")
results["date"] = pd.to_datetime(results["date"])

# Step 2 — Overall RMSE plot
plot_panel_rmse(results)

# Step 3 — Build binned RMSE and plot  [slow: prices every bond via QuantLib]
bin_df = _build_bin_rmse_panel(results, df_SGBILs, kpi_by_month)
plot_panel_rmse_bins(bin_df)

    # endregion

# endregion

# region 9. BID-ASK SPREADS
# Computes mean and median real bid-ask yield spreads (in bp) across SGBi bonds
# for four maturity sets, exports to Excel, and plots the mean time series.
#
# Sets:
#   1-5y  — bonds with TTM in [1, 5) years
#   2-10y — bonds with TTM in [2, 10) years
#   1-10y — bonds with TTM in [1, 10) years
#   all   — all bonds with TTM >= 1 year
#
# Pipeline per bond:
#   quoted nominal bid/ask YTM
#       -> nominal clean price     (standard fixed-rate bond pricing at par)
#       -> real clean price        (deflate by KPI_ref(t_s) / KPI_base)
#       -> real YTM                (invert bond pricing, multi-guess solver)
#   spread_bp = |bid_real_YTM - ask_real_YTM| * 10,000
#
# Output: LINKER_CURVE_ESTIMATION/real_bid_ask.xlsx, sheet "bid_ask_summary"

BID_ASK_OUTPUT = "LINKER_CURVE_ESTIMATION/real_bid_ask.xlsx"

IL_SETS = {
    "1_5y":  (1.0,  5.0),
    "2_10y": (2.0, 10.0),
    "1_10y": (1.0, 10.0),
    "all":   (1.0, np.inf),
}


def _bid_ask_nom_to_real_ytm(
    row,
    nom_yield: float,
    ql_sett:   ql.Date,
) -> float:
    """
    Four-step conversion for one side (bid or ask):
        nominal YTM -> nominal clean price -> real clean price -> real YTM.

    Uses the script-level bond convention globals and the robust multi-guess
    YTM solver. row must have '_index_ratio' pre-computed.
    """
    ql_issue    = _to_ql_date(row["issue_date"])
    ql_maturity = _to_ql_date(row["maturity_date"])

    schedule = ql.Schedule(
        ql_issue, ql_maturity, TENOR, CALENDAR,
        BUS_CONV, BUS_CONV, DATE_GEN_RULE, END_OF_MONTH,
    )
    bond = ql.FixedRateBond(
        SETTLEMENT_DAYS, FACE, schedule,
        [float(row["coupon"])], DAY_COUNT, BUS_CONV, REDEMPTION, ql_issue,
    )

    # Step 1-2: nominal YTM -> nominal clean price
    nom_clean = float(ql.BondFunctions.cleanPrice(
        bond, float(nom_yield), DAY_COUNT,
        ql.Compounded, ql.Annual, ql_sett,
    ))

    # Step 3: nominal clean price -> real clean price
    real_clean = nom_clean / float(row["_index_ratio"])

    # Step 4: real clean price -> real YTM (multi-guess solver)
    price_obj = ql.BondPrice(real_clean, ql.BondPrice.Clean)
    for _guess in [-0.05, -0.02, 0.0, 0.02, 0.05]:
        try:
            return float(ql.BondFunctions.bondYield(
                bond, price_obj, DAY_COUNT,
                ql.Compounded, ql.Annual,
                ql_sett, 1e-10, 1000, _guess,
            ))
        except RuntimeError:
            continue
    return np.nan


def compute_il_bid_ask_summary(
    df_il:        pd.DataFrame,
    kpi_by_month: pd.Series,
) -> pd.DataFrame:
    """
    For each date, compute the cross-sectional mean and median real bid-ask
    yield spread (in bp) for each maturity set defined in IL_SETS.
    """
    work = (
        df_il
        .dropna(subset=[
            "bid_yield", "ask_yield", "bas_kpi",
            "issue_date", "maturity_date", "coupon",
        ])
        .copy()
    )
    work["ttm_years"] = (
        (work["maturity_date"] - work["date"]).dt.days / 365.25
    )

    agg_frames = []

    for label, (lo, hi) in IL_SETS.items():
        mask   = (work["ttm_years"] >= lo) & (work["ttm_years"] < hi)
        subset = work.loc[mask].copy()

        rows = []
        for eval_date, g in subset.groupby("date", sort=True):
            ql_eval = _to_ql_date(eval_date)
            ql.Settings.instance().evaluationDate = ql_eval
            sett    = _ql_settle(eval_date)
            ql_sett = _to_ql_date(sett)

            try:
                ref_kpi = swedish_reference_kpi(sett, kpi_by_month)
            except ValueError:
                continue

            g = g.copy()
            g["_index_ratio"] = ref_kpi / g["bas_kpi"].astype(float)

            spreads = []
            for _, row in g.iterrows():
                try:
                    bid_real = _bid_ask_nom_to_real_ytm(
                        row, row["bid_yield"], ql_sett
                    )
                    ask_real = _bid_ask_nom_to_real_ytm(
                        row, row["ask_yield"], ql_sett
                    )
                    if np.isnan(bid_real) or np.isnan(ask_real):
                        continue
                    spreads.append(abs(bid_real - ask_real) * 1e4)
                except Exception:
                    continue

            if spreads:
                rows.append({
                    "date":            eval_date,
                    f"mean_{label}":   float(np.mean(spreads)),
                    f"median_{label}": float(np.median(spreads)),
                    f"n_{label}":      len(spreads),
                })

        if rows:
            agg_frames.append(
                pd.DataFrame(rows).set_index("date")
            )

    if not agg_frames:
        return pd.DataFrame()

    summary = (
        pd.concat(agg_frames, axis=1)
        .reset_index()
        .sort_values("date")
        .reset_index(drop=True)
    )
    return summary


def plot_il_bid_ask_means(summary: pd.DataFrame):
    """
    Time series of the cross-sectional mean real bid-ask yield spread
    for the four maturity sets.
    """
    dates = pd.to_datetime(summary["date"])

    colors = {
        "1_5y":  "#1A5EA8",
        "2_10y": "#2E8B57",
        "1_10y": "#8B1A1A",
        "all":   "#3A3A3A",
    }
    styles = {
        "1_5y":  "-",
        "2_10y": "-.",
        "1_10y": "--",
        "all":   ":",
    }
    labels = {
        "1_5y":  "1\u20135y",
        "2_10y": "2\u201310y",
        "1_10y": "1\u201310y",
        "all":   "All maturities",
    }

    fig, ax = plt.subplots(figsize=(FIG_WIDTH * 1.4, FIG_HEIGHT))
    fig.patch.set_facecolor("white")
    ax.yaxis.grid(True, color="#CCCCCC", linewidth=0.5, linestyle="-", zorder=0)
    ax.set_axisbelow(True)

    for label in IL_SETS:
        col = f"mean_{label}"
        if col not in summary.columns:
            continue
        ax.plot(dates, summary[col],
                color=colors[label], lw=1.8, ls=styles[label],
                zorder=3, label=labels[label])

    ax.set_ylabel("Mean bid\u2013ask spread (bp)")
    ax.set_xlim(dates.min(), dates.max())
    ax.set_ylim(0, 40)
    ax.set_yticks([4, 8, 12, 16, 20, 24, 28, 32, 36])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))
    _style_ax(ax)
    ax.legend(loc="upper left", frameon=True, fancybox=False,
              edgecolor="#AAAAAA", facecolor="white", framealpha=1.0,
              borderpad=0.6, handlelength=2.2)
    plt.tight_layout(pad=0.6)
    _save_fig(fig, "bid_ask_linker")
    plt.show()
    plt.close(fig)

# Execution

# Step 1 — Compute
il_bid_ask_summary = compute_il_bid_ask_summary(df_SGBILs, kpi_by_month)

# Step 2 — Export
os.makedirs(os.path.dirname(BID_ASK_OUTPUT), exist_ok=True)
with pd.ExcelWriter(BID_ASK_OUTPUT, engine="openpyxl") as writer:
    il_bid_ask_summary.to_excel(writer, sheet_name="bid_ask_summary", index=False)
print(f"Bid-ask summary exported to {BID_ASK_OUTPUT} "
      f"({len(il_bid_ask_summary)} dates)")

# Step 3 — Plot
plot_il_bid_ask_means(il_bid_ask_summary)

# endregion

# region 10. BONDS OUTSTANDING
# Two-panel figure: upper panel shows each SGBi series' TTM over time as a
# separate line; lower panel shows the monthly count of distinct bonds.
# Uses raw_data from region 1.1 (full history, no TTM filter applied).
# Run independently — requires only raw_data from script startup.

def plot_il_bonds_outstanding():
    """
    Two-panel bonds outstanding figure for SGBi.

    Upper panel: TTM trajectory of each SGBi series over time.
    Lower panel: number of distinct bonds outstanding per date.
    """
    # ---- Data preparation
    cols = ["Date", "Issue date", "Maturity date", "ISIN", "Serie"]
    df = (raw_data.loc[:, cols]
          .rename(columns={
              "Date":          "date",
              "Issue date":    "issue_date",
              "Maturity date": "maturity_date",
              "ISIN":          "isin",
              "Serie":         "serie",
          })
          .copy())

    for col in ["date", "issue_date", "maturity_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df = df.dropna(subset=["date", "issue_date", "maturity_date", "serie"])
    df = df.loc[df["serie"] != 3103].copy()
    df = df.loc[df["date"] >= pd.Timestamp("2000-01-01")].copy()

    df["ttm_years"] = (df["maturity_date"] - df["date"]).dt.days / 365.25
    df = df.loc[df["ttm_years"] >= 0.0].sort_values(["serie", "date"]).reset_index(drop=True)

    count_df = (df.groupby("date")["serie"]
                .nunique()
                .rename("n_bonds")
                .reset_index())

    # ---- Figure layout: tall upper / short lower, shared x-axis
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(FIG_WIDTH * 1.4, FIG_HEIGHT * 1.6),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.06},
    )
    fig.patch.set_facecolor("white")

    # ---- Upper panel: TTM lines
    for _, grp in df.groupby("serie"):
        ax1.plot(grp["date"], grp["ttm_years"],
                 color="#0D2B4E", lw=1.4, alpha=1.0, zorder=2)

    ax1.yaxis.grid(True, color="#CCCCCC", linewidth=0.5, linestyle="-", zorder=0)
    ax1.set_axisbelow(True)
    ax1.set_ylabel("Years to maturity")
    ax1.set_ylim(0, 50)
    ax1.set_yticks(range(0, 51, 10))    
    ax1.margins(x=0)
    ax1.tick_params(axis="x", which="both", labelbottom=False)

    # ---- Lower panel: bond count
    ax2.plot(count_df["date"], count_df["n_bonds"],
             color="#0D2B4E", lw=1.6, zorder=2)
    ax2.fill_between(count_df["date"], count_df["n_bonds"],
                     alpha=0.12, color="#1A5EA8", zorder=1)

    ax2.yaxis.grid(True, color="#CCCCCC", linewidth=0.5, linestyle="-", zorder=0)
    ax2.set_axisbelow(True)
    ax2.set_ylabel("Count")
    ax2.set_ylim(0, 15)
    ax2.set_yticks([0, 5, 10])   
    ax2.set_yticks([0, 5, 10])
    ax2.margins(x=0)

    # ---- Shared styling
    for ax in (ax1, ax2):
        _style_ax(ax)

    plt.tight_layout(pad=0.6)
    _save_fig(fig, "bonds_outstanding_real")
    plt.show()
    plt.close(fig)


plot_il_bonds_outstanding()

# endregion

# region 11. ZERO-COUPON YIELDS
# Computes monthly real zero-coupon yields at maturities 1m–240m from the
# extended NS fit, exports to zero_yields_SGBIL.xlsx, and produces a term
# structure plot.
#
# Data source: reads results from OUTPUT_FILE (exported panel parameters).
#
# Step 1 — Compute and export zero-coupon yield panel
# Step 2 — Term structure time series: 1y, 2y, 4y, 6y, 8y, 10y

ZERO_YIELDS_FILE = "LINKER_CURVE_ESTIMATION/zero_yields_SGBIL.xlsx"

EXPORT_MATURITIES_IL = list(range(11, 121))

TERM_STRUCTURE_MATURITIES_IL = {
    "1y":  12,
    "2y":  24,
    "4y":  48,
    "6y":  72,
    "8y":  96,
    "10y": 120,
}

TERM_STRUCTURE_COLORS_IL = [
    "#A8C8E8",
    "#7AAFD4",
    "#4D96C0",
    "#2878A8",
    "#0D5A90",
    "#0D2B4E",
]


def _ns_vec(t, b0, b1, b2, k1):
    """Vectorised NS zero yield (continuously compounded)."""
    x1 = k1 * t
    l1 = np.where(np.abs(x1) < 1e-12, 1.0, (1.0 - np.exp(-x1)) / x1)
    l2 = l1 - np.exp(-x1)
    return b0 + b1 * l1 + b2 * l2


# ---- Step 1 — Compute and export

results_il = pd.read_excel(OUTPUT_FILE, sheet_name="fit_params")
results_il["date"] = pd.to_datetime(results_il["date"])
results_il = results_il.sort_values("date").reset_index(drop=True)

b0 = results_il["b0_ext"].to_numpy(float)
b1 = results_il["b1_ext"].to_numpy(float)
b2 = results_il["b2_ext"].to_numpy(float)
k1 = results_il["k1_ext"].to_numpy(float)

zero_yields_il = pd.DataFrame({"date": results_il["date"]})
for m in EXPORT_MATURITIES_IL:
    zero_yields_il[f"y_{m}m"] = _ns_vec(m / 12.0, b0, b1, b2, k1)

with pd.ExcelWriter(ZERO_YIELDS_FILE, engine="openpyxl") as writer:
    zero_yields_il.to_excel(writer, sheet_name="zero_yields", index=False)
    results_il.to_excel(writer, sheet_name="fit_params", index=False)

print(f"Exported {len(zero_yields_il)} months to {ZERO_YIELDS_FILE}")


# ---- Step 2 — Term structure time series

def plot_il_term_structure(zero_yields: pd.DataFrame):
    dates = pd.to_datetime(zero_yields["date"])

    fig, ax = plt.subplots(figsize=(FIG_WIDTH * 1.4, FIG_HEIGHT))
    fig.patch.set_facecolor("white")
    ax.yaxis.grid(True, color="#CCCCCC", linewidth=0.5, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    ax.axhline(0, color="#666666", lw=0.8, ls="-", zorder=1)

    for (label, m), color in zip(TERM_STRUCTURE_MATURITIES_IL.items(),
                                  TERM_STRUCTURE_COLORS_IL):
        ax.plot(dates, zero_yields[f"y_{m}m"] * 100,
                color=color, lw=1.6, zorder=3, label=label)

    ax.set_ylabel("Yield (%)")
    ax.set_xlim(dates.min(), dates.max())
    ax.set_ylim(-4, 5)
    ax.set_yticks(range(-4, 6))
    ax.set_yticklabels([
        "" if v in (-4, 5) else str(v)
        for v in range(-4, 6)
    ])
    _style_ax(ax)
    ax.legend(loc="upper right", frameon=True, fancybox=False,
              edgecolor="#AAAAAA", facecolor="white", framealpha=1.0,
              borderpad=0.6, handlelength=2.2, ncol=3)
    plt.tight_layout(pad=0.6)
    _save_fig(fig, "term_structure_real")
    plt.show()
    plt.close(fig)


plot_il_term_structure(zero_yields_il)

# endregion

# region 12. LAG-ADJUSTED ZERO-COUPON YIELDS
# Strips the 3-month indexation-lag carry D_t from the estimated real
# zero-coupon yields to produce fully-indexed (lag-adjusted) real yields:
#
#   ỹ(n, t) = y(n, t) + (12 / n) · D_t
#
# where  D_t = π_t + π_{t-1} + π_{t-2}  is the sum of log monthly KPI
# changes over the current and previous two months (the lag window).
# A longer-maturity bond receives a smaller annualised adjustment because
# D_t is amortised over its full maturity.
#
# Step 1 — Build D_t series and compute lag-adjusted yield panel
# Step 2 — Export to zero_yields_SGBIL_lag_adj.xlsx
# Step 3 — Plot observed vs lag-adjusted at 2y, 5y, 10y

LAG_ADJ_OUTPUT    = "LINKER_CURVE_ESTIMATION/zero_yields_SGBIL_lag_adj.xlsx"
INDEX_LAG_MONTHS  = 3

LAG_ADJ_PLOT_MATS = {"1y": 12, "2y": 24, "5y": 60, "10y": 120}
LAG_ADJ_COLORS    = ["#A8C8E8", "#2878A8", "#3A3A3A", "#8B1A1A"]


# ---- Imports (section can be run independently)

zero_yields_il = pd.read_excel(ZERO_YIELDS_FILE, sheet_name="zero_yields")
zero_yields_il["date"] = pd.to_datetime(zero_yields_il["date"])

_kpi_raw      = pd.read_excel(KPI_FILE, sheet_name="basår 1980")[["date", "KPI"]].copy()
_kpi_raw["date"]  = pd.to_datetime(_kpi_raw["date"], errors="coerce")
_kpi_raw["KPI"]   = pd.to_numeric(_kpi_raw["KPI"], errors="coerce")
_kpi_raw["month"] = _kpi_raw["date"].dt.to_period("M")
kpi_by_month  = (
    _kpi_raw[["month", "KPI"]]
    .dropna()
    .drop_duplicates(subset=["month"])
    .sort_values("month")
    .set_index("month")["KPI"]
)

# ---- Step 1 — D_t and lag-adjusted yields

# Log monthly KPI changes: π_m = log(KPI_m / KPI_{m-1})
kpi_sorted   = kpi_by_month.sort_index()
log_pi       = pd.Series(
    np.concatenate([[np.nan], np.diff(np.log(kpi_sorted.values))]),
    index=kpi_sorted.index,
)

# D_t = rolling 3-month sum (positional; KPI is monthly, no gaps)
D_series = log_pi.rolling(window=INDEX_LAG_MONTHS).sum()   # Period-indexed

# Map D_t to each observation date in the zero-yield panel
periods_il = zero_yields_il["date"].dt.to_period("M")
D_t        = periods_il.map(D_series).values.astype(float)

n_nan = np.isnan(D_t).sum()
if n_nan:
    print(f"  Warning: {n_nan} dates with missing D_t (insufficient KPI history)")

print(f"\n  D_t  (ann. avg): {np.nanmean(D_t) * 12 * 100:+.2f}% p.a.")
print(f"  D_t  range: [{np.nanmin(D_t)*100:.2f}%, {np.nanmax(D_t)*100:.2f}%]  (monthly, log)")

# Lag-adjusted yields: add the annualised carry (12/n) · D_t
zero_yields_il_adj = zero_yields_il.copy()
for col in zero_yields_il_adj.columns:
    if col.startswith("y_"):
        n = int(col.split("_")[1].rstrip("m"))
        zero_yields_il_adj[col] = zero_yields_il[col].values + (12.0 / n) * D_t


# ---- Step 2 — Export

with pd.ExcelWriter(LAG_ADJ_OUTPUT, engine="openpyxl") as writer:
    zero_yields_il_adj.to_excel(writer, sheet_name="zero_yields_lag_adj", index=False)
    pd.DataFrame({"date": zero_yields_il["date"], "D_t": D_t}).to_excel(
        writer, sheet_name="carry_D_t", index=False
    )
print(f"  Exported lag-adjusted yields to {LAG_ADJ_OUTPUT}")


# ---- Step 3 — Plot: observed vs lag-adjusted (1y, 2y, 5y, 10y)

def plot_lag_adjusted_comparison(
    zero_yields: pd.DataFrame,
    zero_yields_adj: pd.DataFrame,
):
    """
    Three-panel figure: each panel shows the observed (lagged) real zero-coupon
    yield and its lag-adjusted counterpart for one key maturity.
    """
    dates = pd.to_datetime(zero_yields["date"])

    fig, axes = plt.subplots(
        len(LAG_ADJ_PLOT_MATS), 1,
        figsize=(FIG_WIDTH * 1.4, FIG_HEIGHT * len(LAG_ADJ_PLOT_MATS)),
        sharex=True,
    )
    fig.patch.set_facecolor("white")

    for ax, (label, m), color in zip(axes, LAG_ADJ_PLOT_MATS.items(), LAG_ADJ_COLORS):
        col = f"y_{m}m"
        ax.yaxis.grid(True, color="#CCCCCC", linewidth=0.5, linestyle="-", zorder=0)
        ax.set_axisbelow(True)
        ax.axhline(0, color="#666666", lw=0.8, ls="-", zorder=1)

        ax.plot(dates, zero_yields[col]     * 100,
                color=color, lw=1.6, ls="-",  zorder=3, label="SGBi")
        ax.plot(dates, zero_yields_adj[col] * 100,
                color=color, lw=1.4, ls="--", zorder=3, label="Carry adjusted")

        ax.set_ylabel("Yield (%)")
        ax.set_title(label, pad=4)
        _style_ax(ax)
        ax.legend(
            loc="best", frameon=True, fancybox=False,
            edgecolor="#AAAAAA", facecolor="white", framealpha=1.0,
            borderpad=0.6, handlelength=2.2,
        )

    axes[-1].set_xlim(dates.min(), dates.max())
    plt.tight_layout(pad=0.6)
    _save_fig(fig, "lag_adjusted_comparison")
    plt.show()
    plt.close(fig)


plot_lag_adjusted_comparison(zero_yields_il, zero_yields_il_adj)

# endregion
