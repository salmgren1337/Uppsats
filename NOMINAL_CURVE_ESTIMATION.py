import pandas as pd
import numpy as np
import QuantLib as ql
import matplotlib.pyplot as plt
import multiprocessing as mp
import os
 
# ------------------------------------------------------------------------------
#   Fits the Nelson-Sigel-Svensson yield curve to monthly cross-section of SGB
#   prices. Three instrument types enter the estimation:
#   coupon-bearing SGBs (the primary input), SSVX T-bills at the
#   1m/3m/6m tenors (to anchor the short end), and synthetic zero-coupon bonds
#   inserted at the midpoint between each pair of adjacent SGB maturities
#   (priced from a preliminary bootstrap, to fill maturity gaps).
#
#   For each date, two NSS curves are fitted:
#     - Actual:   fitted to observed SGB and SSVX prices only
#     - Extended: fitted to observed prices + synthetic midpoint instruments
# ------------------------------------------------------------------------------
#   Run the script once from 0 - 5 to load data and define all
#   functions. Then use the two RUN regions 6 and 7 independently:
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
#   All settings that govern the NSS estimation (parameter bounds, L2
#   regularisation weights, optimiser tolerances, and the multistart grid)
#   are collected in the "Estimation pipeline" sub-region inside CONFIGURATION.
#   Adjust them there only — they are referenced globally throughout the script.
# ------------------------------------------------------------------------------
 
# region 0.CONFIGURATION
# General settings that apply across the 0-7:
 
# ---- Data and output
SGB_FILE    = "statsobligationer_data.xlsx"
OUTPUT_FILE = "NOMINAL_CURVE_ESTIMATION/zero_yields_SGB.xlsx"       # panel export goes to working directory
 
# ---- Sample period
START = pd.Timestamp("2000-01-01")
END   = pd.Timestamp("2026-01-31")
 
# ---- TTM filter
MIN_TTM_DAYS  = 90
MAX_TTM_YEARS = 15
 
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
 
# ---- Figure settings:
# FIG_WIDTH / FIG_HEIGHT in inches. 5.5 x 3.4 fits a standard A4 LaTeX
FIG_WIDTH    = 5.5
FIG_HEIGHT   = 3.4
FIG_DPI      = 300          # resolution
FIG_FORMAT   = "pdf"        # "pdf" or "png"
FIG_SAVE_DIR = "NOMINAL_CURVE_ESTIMATION"   # set to a folder path to auto-save, or "None" if no save

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

os.makedirs("NOMINAL_CURVE_ESTIMATION", exist_ok=True)
 
    # region 0.1.Estimation pipeline
 
    # ---- Parameter bounds
    # Beta bounds are in continuously compounded rate space.
    # Tau bounds are in years; internally converted to kappa = 1/tau.
B0_BOUNDS   = (-0.03,  0.08)   # long-run level
B1_BOUNDS   = (-0.10,  0.10)   # slope
B2_BOUNDS   = (-0.05,  0.05)   # first hump
B3_BOUNDS   = (-0.05,  0.05)   # second hump
TAU1_BOUNDS = (0.25,   5.0)   # location/peak of first hump (years)
TAU2_BOUNDS = (4.0,  20.0)   # location/peak of second hump (years)
 
    # ---- L2 regularisation weights [b0, b1, b2, b3, k1, k2]
    # Humps (b2, b3) are penalised more strongly to discourage overfitting.
L2_WEIGHTS = [0.05, 0.01, 0.1, 0.1, 0.02, 0.02]
 
    # ---- Optimiser (Nelder-Mead simplex)
ACCURACY       = 1e-10
MAX_EVALS      = 15_000
SIMPLEX_LAMBDA = 0.005
 
    # ---- Multistart grid
    # b0 and b1 are anchored per date from the bootstrap.
    # Every (b2, b3, tau1, tau2) combination with tau2 > tau1 is tried.
B2_GRID   = [-0.02, -0.01, 0.0, 0.01, 0.02]
B3_GRID   = [-0.02, -0.01, 0.0, 0.01, 0.02]
TAU1_GRID = [0.5, 1.0, 2.0, 3.0, 3.5]
TAU2_GRID = [6.0, 8.0, 10.0, 12.0, 15.0, 18.0]
 
    # endregion
 
# endregion

# region 1.DATA PREPARATION
# Loads SGBs, SSVX T-bills, and the deposit rate; aligns all EOM sources to
# the last SGB trading day of each month; combines into a single df_panel.
# Run once at script startup — all downstream sections depend on df_panel and
# df_deposit.
 
    # region 1.1.SGBs
 
raw_sgb   = pd.read_excel(SGB_FILE, sheet_name="SGB long")
raw_short = pd.read_excel(SGB_FILE, sheet_name="korta räntor riksbanken")
 
df_SGBs = (
    raw_sgb.loc[:, ["Date", "PX_LAST", "YLD_YTM_MID", "BID_YIELD", "ASK_YIELD",
                    "Kupong", "Issue date", "Maturity date", "ISIN", "Serie"]]
    .rename(columns={
        "Date": "date",           "PX_LAST": "price",       "YLD_YTM_MID": "yield",
        "BID_YIELD": "bid_yield", "ASK_YIELD": "ask_yield", "Kupong": "coupon",
        "Issue date": "issue_date", "Maturity date": "maturity_date",
        "ISIN": "isin",           "Serie": "serie",
    })
    .copy()
)
 
for col in ["date", "issue_date", "maturity_date"]:
    df_SGBs[col] = pd.to_datetime(df_SGBs[col], errors="coerce")
for col in ["price", "yield", "bid_yield", "ask_yield", "coupon"]:
    df_SGBs[col] = pd.to_numeric(df_SGBs[col], errors="coerce")
 
df_SGBs[["yield", "bid_yield", "ask_yield", "coupon"]] /= 100.0
 
df_SGBs  = df_SGBs.loc[(df_SGBs["date"] >= START) & (df_SGBs["date"] <= END)]
ttm_days = (df_SGBs["maturity_date"] - df_SGBs["date"]).dt.days
df_SGBs  = df_SGBs[(ttm_days >= MIN_TTM_DAYS) & (ttm_days <= MAX_TTM_YEARS * 365)]
df_SGBs  = df_SGBs.sort_values(["date", "maturity_date"]).reset_index(drop=True)
 
# Build the last-trading-day index used to align all EOM data sources below
df_SGBs["_month"] = df_SGBs["date"].dt.to_period("M")
month_last_trade  = df_SGBs.groupby("_month")["date"].max()
df_SGBs           = df_SGBs.drop(columns="_month")
 
    # endregion
 
    # region 1.2.SSVX T-bills
    # SSVX rates are simple-interest yields on an act/360 basis.
    # Maturity dates use exact calendar-month offsets (pd.DateOffset).
    # Issue and maturity are anchored to the last SGB trading day of each month.
 
TENOR_MONTHS = {"SSVX_1m": 1, "SSVX_3m": 3, "SSVX_6m": 6}
 
df_SSVX = (
    raw_short.loc[:, ["date"] + list(TENOR_MONTHS.keys())]
    .melt(id_vars="date", var_name="serie", value_name="yield")
    .dropna(subset=["yield"])
    .reset_index(drop=True)
)
df_SSVX["date"]   = pd.to_datetime(df_SSVX["date"], errors="coerce")
df_SSVX["yield"]  = pd.to_numeric(df_SSVX["yield"], errors="coerce") / 100.0
df_SSVX["months"] = df_SSVX["serie"].map(TENOR_MONTHS).astype("Int64")
df_SSVX["coupon"] = 0.0
df_SSVX["isin"]   = ""
 
# Align EOM publication date to last SGB trading day of that month
df_SSVX["_month"] = df_SSVX["date"].dt.to_period("M")
df_SSVX["date"]   = df_SSVX["_month"].map(month_last_trade)
df_SSVX           = df_SSVX.drop(columns="_month").dropna(subset=["date"])
df_SSVX           = df_SSVX.loc[(df_SSVX["date"] >= START) & (df_SSVX["date"] <= END)]
 
# Issue date = aligned trading date.
# Maturity = issue date + calendar months, then adjusted to next good business day
# (ModifiedFollowing, Sweden calendar) — consistent with how QuantLib constructs
# the cash flow schedule internally.
# Settlement = issue date + SETTLEMENT_DAYS good business days using the Sweden
# calendar (not pd.offsets.BusinessDay, which ignores Swedish holidays).
df_SSVX["issue_date"] = df_SSVX["date"]

def _ql_adjust(ts: pd.Timestamp) -> pd.Timestamp:
    ql_d = ql.Date(ts.day, ts.month, ts.year)
    ql_a = CALENDAR.adjust(ql_d, BUS_CONV)
    return pd.Timestamp(ql_a.year(), ql_a.month(), ql_a.dayOfMonth())

def _ql_settle(ts: pd.Timestamp) -> pd.Timestamp:
    ql_d = ql.Date(ts.day, ts.month, ts.year)
    ql_s = CALENDAR.advance(ql_d, SETTLEMENT_DAYS, ql.Days)
    return pd.Timestamp(ql_s.year(), ql_s.month(), ql_s.dayOfMonth())

df_SSVX["maturity_date"]   = df_SSVX.apply(
    lambda r: _ql_adjust(r["issue_date"] + pd.DateOffset(months=int(r["months"]))), axis=1
)
df_SSVX["settlement_date"] = df_SSVX["issue_date"].apply(_ql_settle)

# Simple-interest price:  P = 100 / (1 + r * t),  t = act/360 from settlement
t_act360         = (df_SSVX["maturity_date"] - df_SSVX["settlement_date"]).dt.days / 360.0
df_SSVX["price"] = 100.0 / (1.0 + df_SSVX["yield"] * t_act360)
df_SSVX["bid_yield"] = np.nan
df_SSVX["ask_yield"] = np.nan
df_SSVX = df_SSVX.sort_values(["date", "maturity_date"]).reset_index(drop=True)
 
    # endregion
 
    # region 1.3.Deposit rate
    # Used to anchor the b0+b1 starting value (short-rate proxy).
 
df_deposit = raw_short.loc[:, ["date", "deposit"]].copy()
df_deposit["date"]    = pd.to_datetime(df_deposit["date"], errors="coerce")
df_deposit["deposit"] = pd.to_numeric(df_deposit["deposit"], errors="coerce") / 100.0
df_deposit["_month"]  = df_deposit["date"].dt.to_period("M")
df_deposit["date"]    = df_deposit["_month"].map(month_last_trade)
df_deposit = (
    df_deposit.drop(columns="_month")
    .dropna(subset=["date"])
    .sort_values("date")
    .reset_index(drop=True)
)
 
    # endregion
 
    # region 1.4.Combine into panel
 
PANEL_COLS = ["date", "price", "yield", "coupon", "issue_date",
              "maturity_date", "isin", "serie", "bid_yield", "ask_yield"]
 
df_panel = (
    pd.concat([df_SGBs.loc[:, PANEL_COLS], df_SSVX.loc[:, PANEL_COLS]],
              ignore_index=True)
    .sort_values(["date", "maturity_date"])
    .reset_index(drop=True)
)
 
    # endregion
 
# endregion

# region 2.ESTIMATION FUNCTIONS

def _to_ql_date(ts: pd.Timestamp) -> ql.Date:
    return ql.Date(ts.day, ts.month, ts.year)
 
 
def build_bond_helpers(cross_section: pd.DataFrame):
    """
    Build QuantLib bond objects and price helpers from a cross-section.

    Returns: helpers, bonds, quoted_prices, quoted_yields, maturities, is_zcb
    is_zcb is a boolean array — True for zero-coupon instruments (SSVX T-bills).
    """
    helpers, bonds, prices, yields, maturities, is_zcb = [], [], [], [], [], []

    for _, row in cross_section.iterrows():
        ql_issue    = _to_ql_date(row["issue_date"])
        ql_maturity = _to_ql_date(row["maturity_date"])
        coupon_rate = float(row["coupon"])
        clean_price = float(row["price"])

        schedule = ql.Schedule(
            ql_issue, ql_maturity, TENOR, CALENDAR,
            BUS_CONV, BUS_CONV, DATE_GEN_RULE, END_OF_MONTH
        )
        bond = ql.FixedRateBond(
            SETTLEMENT_DAYS, FACE, schedule,
            [coupon_rate], DAY_COUNT, BUS_CONV, REDEMPTION, ql_issue
        )
        helper = ql.FixedRateBondHelper(
            ql.QuoteHandle(ql.SimpleQuote(clean_price)),
            SETTLEMENT_DAYS, FACE, schedule,
            [coupon_rate], DAY_COUNT, BUS_CONV, REDEMPTION, ql_issue
        )
        bonds.append(bond)
        helpers.append(helper)
        prices.append(clean_price)
        yields.append(float(row["yield"]))
        maturities.append(row["maturity_date"])
        is_zcb.append(coupon_rate == 0.0)

    return helpers, bonds, np.array(prices), np.array(yields), maturities, np.array(is_zcb)
 
 
def build_bootstrap_curve(helpers: list) -> ql.PiecewiseLogCubicDiscount:
    """Piecewise log-cubic discount curve used as a non-parametric reference
    and for pricing synthetic midpoint instruments."""
    curve = ql.PiecewiseLogCubicDiscount(SETTLEMENT_DAYS, CALENDAR, helpers, DAY_COUNT)
    curve.enableExtrapolation()
    return curve
 
 
def build_synthetic_helpers(bonds_actual: list, pre_curve, ql_eval_date: ql.Date) -> list:
    """
    Insert a synthetic zero-coupon bond at the midpoint between each adjacent
    pair of actual bond maturities, priced from the bootstrapped pre-curve.
    Fills maturity gaps and regularises the NSS fit (used in the extended fit).
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
            BUS_CONV, BUS_CONV, ql.DateGeneration.Forward, False
        )
        synth_helpers.append(
            ql.FixedRateBondHelper(
                ql.QuoteHandle(ql.SimpleQuote(float(synth_price))),
                SETTLEMENT_DAYS, FACE, schedule,
                [0.0], DAY_COUNT, BUS_CONV, REDEMPTION, ref_date
            )
        )
 
    return synth_helpers
 
 
def compute_anchors(bonds_actual: list, pre_curve, deposit_rate: float):
    """
    Derive data-driven starting values for b0 and b1:
        b0      = long-end continuous zero rate from the bootstrap
        b0 + b1 = deposit rate (continuously compounded) ≈ short rate at t→0
    """
    max_maturity = max(b.maturityDate() for b in bonds_actual)
    b0 = pre_curve.zeroRate(max_maturity, DAY_COUNT, ql.Continuous).rate()
    b1 = np.log(1.0 + deposit_rate) - b0
    return b0, b1
 
 
def _build_fitting_method() -> ql.SvenssonFitting:
    """Assemble the QuantLib SvenssonFitting object from the pipeline settings."""
    kappa1_min, kappa1_max = 1.0 / TAU1_BOUNDS[1], 1.0 / TAU1_BOUNDS[0]
    kappa2_min, kappa2_max = 1.0 / TAU2_BOUNDS[1], 1.0 / TAU2_BOUNDS[0]
 
    lower, upper = ql.Array(6), ql.Array(6)
    for i, (lo, hi) in enumerate([
        B0_BOUNDS, B1_BOUNDS, B2_BOUNDS, B3_BOUNDS,
        (kappa1_min, kappa1_max), (kappa2_min, kappa2_max),
    ]):
        lower[i], upper[i] = lo, hi
 
    l2 = ql.Array(6)
    for i, w in enumerate(L2_WEIGHTS):
        l2[i] = w
 
    return ql.SvenssonFitting(
        ql.Array(), ql.Simplex(SIMPLEX_LAMBDA),
        l2, 0.0, 50.0,
        ql.NonhomogeneousBoundaryConstraint(lower, upper)
    )
 
 
def fit_multistart(helpers: list, b0: float, b1: float):
    """
    Run NSS optimisation from every point on the (b2, b3, tau1, tau2) grid;
    return the globally best result by minimum objective value.
 
    Returns: curve, params [b0..k2], objective value, winning start tuple
    """
    fitting = _build_fitting_method()
    best_obj, best_curve, best_params, best_start = float("inf"), None, None, None
 
    for b2s in B2_GRID:
        for b3s in B3_GRID:
            for tau1 in TAU1_GRID:
                for tau2 in TAU2_GRID:
                    if tau2 <= tau1:
                        continue
 
                    guess    = ql.Array(6)
                    guess[0], guess[1] = b0, b1
                    guess[2], guess[3] = b2s, b3s
                    guess[4], guess[5] = 1.0 / tau1, 1.0 / tau2
 
                    try:
                        curve = ql.FittedBondDiscountCurve(
                            SETTLEMENT_DAYS, CALENDAR, helpers,
                            DAY_COUNT, fitting, ACCURACY, MAX_EVALS, guess
                        )
                        obj = curve.fitResults().minimumCostValue()
 
                        if obj < best_obj:
                            best_obj    = obj
                            best_curve  = curve
                            best_params = list(curve.fitResults().solution())
                            best_start  = (b2s, b3s, tau1, tau2)
 
                    except RuntimeError:
                        continue
 
    return best_curve, best_params, best_obj, best_start
 
 
def evaluate_fit(bonds_actual: list, quoted_prices: np.ndarray,
                 quoted_yields: np.ndarray, curve, is_zcb: np.ndarray) -> dict:
    """
    Price bonds from a fitted curve; return price and yield RMSEs along with
    model-implied prices and yields.
    Yield RMSE is computed only over coupon bonds (not SSVX zeros).
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

    is_coupon = ~is_zcb
    if is_coupon.any():
        rmse_yield_bp = float(
            np.sqrt(np.mean((quoted_yields[is_coupon] - model_yields[is_coupon]) ** 2)) * 1e4
        )
    else:
        rmse_yield_bp = np.nan

    return {
        "rmse_price":    float(np.sqrt(np.mean((quoted_prices - model_prices) ** 2))),
        "rmse_yield_bp": rmse_yield_bp,
        "model_prices":  model_prices,
        "model_yields":  model_yields,
    }
 
# endregion

# region 3.SINGLE-DATE ESTIMATION
# estimate_nss() fits both the actual and extended NSS curves for one date.
# Called directly in RUN SINGLE DATE and inside run_panel() for RUN PANEL.
 
def estimate_nss(eval_date: pd.Timestamp, df_panel: pd.DataFrame,
                 df_deposit: pd.DataFrame) -> dict:
    """
    Fit the NSS curve for a single evaluation date.
 
    Two fits are produced:
        actual   — fitted to observed bond prices only
        extended — fitted to observed prices + bootstrapped synthetic midpoints
 
    Returns a dict with parameters, diagnostics, and QL objects.
    Keys prefixed with '_' hold QL objects for plotting; they are stripped
    automatically before panel export.
    """
    cross_section = (
        df_panel.loc[df_panel["date"] == eval_date]
        .dropna(subset=["price", "coupon", "issue_date", "maturity_date"])
        .sort_values(["maturity_date", "coupon"], ascending=[True, False])
        .reset_index(drop=True)
    )
    if cross_section.empty:
        raise ValueError(f"No instruments found for {eval_date.date()}")

    # If a SSVX T-bill and a coupon SGB share the same maturity date, drop the
    # SSVX — two instruments on the same pillar date cause the bootstrap to fail.
    # Sorting by coupon descending above ensures the coupon bond is kept first.
    n_before = len(cross_section)
    cross_section = cross_section.drop_duplicates(subset="maturity_date", keep="first")
    n_dropped = n_before - len(cross_section)
    if n_dropped > 0:
        print(f"  {eval_date.date()}: dropped {n_dropped} duplicate-maturity "
            f"instrument(s) before bootstrap")
 
    deposit_row = df_deposit.loc[df_deposit["date"] == eval_date]
    if deposit_row.empty:
        raise ValueError(f"No deposit rate for {eval_date.date()}")
    deposit_rate = float(deposit_row["deposit"].iloc[0])
 
    ql_eval_date = _to_ql_date(eval_date)
    ql.Settings.instance().evaluationDate = ql_eval_date
 
    helpers_actual, bonds_actual, quoted_prices, quoted_yields, maturities, is_zcb = \
        build_bond_helpers(cross_section)
 
    pre_curve        = build_bootstrap_curve(helpers_actual)
    synth_helpers    = build_synthetic_helpers(bonds_actual, pre_curve, ql_eval_date)
    helpers_extended = helpers_actual + synth_helpers
 
    b0, b1 = compute_anchors(bonds_actual, pre_curve, deposit_rate)
 
    curve_actual, params_actual, obj_actual, start_actual = fit_multistart(helpers_actual, b0, b1)
    if curve_actual is None:
        raise RuntimeError(f"Multistart failed (actual) on {eval_date.date()}")
 
    curve_ext, params_ext, obj_ext, start_ext = fit_multistart(helpers_extended, b0, b1)
    if curve_ext is None:
        raise RuntimeError(f"Multistart failed (extended) on {eval_date.date()}")
 
    diag_actual = evaluate_fit(bonds_actual, quoted_prices, quoted_yields, curve_actual, is_zcb)
    diag_ext    = evaluate_fit(bonds_actual, quoted_prices, quoted_yields, curve_ext,   is_zcb)
 
    return {
        # NSS parameters
        "b0_actual": params_actual[0], "b1_actual": params_actual[1],
        "b2_actual": params_actual[2], "b3_actual": params_actual[3],
        "k1_actual": params_actual[4], "k2_actual": params_actual[5],
        "b0_ext": params_ext[0], "b1_ext": params_ext[1],
        "b2_ext": params_ext[2], "b3_ext": params_ext[3],
        "k1_ext": params_ext[4], "k2_ext": params_ext[5],
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
        # Winning starting values
        "start_b2_actual":   start_actual[0], "start_b3_actual":   start_actual[1],
        "start_tau1_actual": start_actual[2], "start_tau2_actual": start_actual[3],
        "start_b2_ext":      start_ext[0],    "start_b3_ext":      start_ext[1],
        "start_tau1_ext":    start_ext[2],    "start_tau2_ext":    start_ext[3],
        # QL objects — for plots only, stripped before export
        "_eval_date":     eval_date,
        "_ql_eval_date":  ql_eval_date,
        "_curve_actual":  curve_actual,
        "_curve_ext":     curve_ext,
        "_pre_curve":     pre_curve,
        "_bonds_actual":  bonds_actual,
        "_quoted_prices": quoted_prices,
        "_quoted_yields": quoted_yields,
        "_is_zcb":        is_zcb,
        "_maturities":    maturities,
        "_diag_actual":   diag_actual,
        "_diag_ext":      diag_ext,
    }
 
# endregion
 
# region 4.DIAGNOSTICS AND PLOTS
# Diagnostic print functions and plot functions for both modes.
# All functions here take the result dict (single-date) or results DataFrame
# (panel) as input and are fully independent of the estimation step.
 
def print_single_diagnostics(result: dict):
    """Print NSS parameters and fit quality for a single-date result."""
    d = result["_eval_date"].date()
    print(f"\n{'─'*52}")
    print(f"  Single-date diagnostics — {d}")
    print(f"{'─'*52}")
    print(f"  Instruments:  {result['n_bonds']} bonds,  {result['n_synth']} synthetic")
    print()
    print(f"  {'Parameter':<12}  {'Actual':>12}  {'Extended':>12}")
    print(f"  {'─'*38}")
    for key in ["b0", "b1", "b2", "b3", "k1", "k2"]:
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
    """
    Save figure to FIG_SAVE_DIR if configured.
    stem is the base filename without extension, e.g. 'yield_curve_2025-12'.
    """
    if FIG_SAVE_DIR is None:
        return
    os.makedirs(FIG_SAVE_DIR, exist_ok=True)
    path = os.path.join(FIG_SAVE_DIR, f"{stem}.{FIG_FORMAT}")
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {path}")


def plot_curves(result: dict):
    """
    Figure dimensions, DPI, format, and output directory are controlled
    by the FIG_* constants in the CONFIGURATION region.
    Set FIG_SAVE_DIR to a folder path to auto-save the figure.
    """
    eval_date    = result["_eval_date"]
    ql_eval_date = result["_ql_eval_date"]
    ql.Settings.instance().evaluationDate = ql_eval_date

    curve_actual  = result["_curve_actual"]
    curve_ext     = result["_curve_ext"]
    pre_curve     = result["_pre_curve"]
    bonds_actual  = result["_bonds_actual"]
    quoted_yields = result["_quoted_yields"]
    quoted_prices = result["_quoted_prices"]
    is_zcb        = result["_is_zcb"]
    maturities    = result["_maturities"]

    settlement_date = _ql_settle(eval_date)
    max_date        = max(b.maturityDate() for b in bonds_actual)
    comp            = ql.Continuous

    # ---- Curve arrays
    times, z_actual, z_ext, z_boot = [], [], [], []
    for months in range(1, int(35 * 12) + 1):
        d = CALENDAR.advance(ql_eval_date, ql.Period(months, ql.Months))
        if d > max_date:
            break
        times.append(months / 12.0)
        z_actual.append(curve_actual.zeroRate(d, DAY_COUNT, comp).rate())
        z_ext.append(curve_ext.zeroRate(d, DAY_COUNT, comp).rate())
        z_boot.append(pre_curve.zeroRate(d, DAY_COUNT, comp).rate())

    # ---- Observed yields converted to CC zero rates (Thirty360)
    mat_years, obs_cc = [], []
    for m, p, ytm, zcb in zip(maturities, quoted_prices, quoted_yields, is_zcb):
        mat_years.append((pd.Timestamp(m) - eval_date).days / 365.25)
        if zcb:
            ql_s = _to_ql_date(settlement_date)
            ql_m = _to_ql_date(pd.Timestamp(m))
            t_yr = DAY_COUNT.yearFraction(ql_s, ql_m)
            obs_cc.append(-np.log(p / 100.0) / t_yr if t_yr > 0 else np.nan)
        else:
            obs_cc.append(np.log(1.0 + ytm))

    # ---- Figure
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    fig.patch.set_facecolor("white")

    # Horizontal grid — drawn first so it sits behind all data
    ax.yaxis.grid(True, color="#CCCCCC", linewidth=0.5, linestyle="-", zorder=0)
    ax.set_axisbelow(True)

    # Curves
    ax.plot(times, 100 * np.array(z_actual),
            color=COLOR_ACTUAL,   lw=2.0, ls="-",  zorder=3,
            label="Actual")
    ax.plot(times, 100 * np.array(z_ext),
            color=COLOR_EXTENDED, lw=2.0, ls="-",  zorder=3,
            label="Extended")
    ax.plot(times, 100 * np.array(z_boot),
            color=COLOR_BOOT,     lw=1.6, ls="-.", zorder=2,
            label="Bootstrap")

    # Observed yields
    ax.scatter(mat_years, 100 * np.array(obs_cc),
               s=50, color=COLOR_OBSERVED, marker="D", zorder=3,
               edgecolors="#4A4A4A", linewidths=0.5,
               label="Observed")

    # ---- Labels and formatting
    ax.set_xlabel("Years to maturity")
    ax.set_ylabel("Yield (%)")
    #ax.set_title(eval_date.strftime("%B %Y"), pad=8)
    ax.set_xlim(0, max(times))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))

    _style_ax(ax)

    ax.legend(loc="best", frameon=True, fancybox=False,
              edgecolor="#AAAAAA", facecolor="white", framealpha=1.0,
              borderpad=0.6, handlelength=2.2)

    plt.tight_layout(pad=0.6)
    _save_fig(fig, f"yield_curve_{eval_date.strftime('%Y-%m')}")
    plt.show()
    plt.close(fig)


def plot_price_residuals(result: dict):
    """
    Scatter of price residuals (observed minus model)
    against maturity for both the actual and extended fits.
    """
    eval_date    = result["_eval_date"]
    ql_eval_date = result["_ql_eval_date"]
    ql.Settings.instance().evaluationDate = ql_eval_date

    maturities    = pd.to_datetime(result["_maturities"])
    quoted_prices = result["_quoted_prices"]
    diag_actual   = result["_diag_actual"]
    diag_ext      = result["_diag_ext"]

    resid_actual = quoted_prices - diag_actual["model_prices"]
    resid_ext    = quoted_prices - diag_ext["model_prices"]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    fig.patch.set_facecolor("white")

    ax.yaxis.grid(True, color="#CCCCCC", linewidth=0.5, linestyle="-", zorder=0)
    ax.set_axisbelow(True)

    ax.scatter(maturities, resid_actual,
               s=50, color=COLOR_ACTUAL, marker="^", zorder=3,
               edgecolors="#4A4A4A", linewidths=0.5, label="Actual fit")
    ax.scatter(maturities, resid_ext,
               s=50, color=COLOR_EXTENDED, marker="o", zorder=3,
               edgecolors="#4A4A4A", linewidths=0.5, label="Extended fit")
    ax.axhline(0, color="#666666", ls="--", lw=1.0, zorder=1)

    ax.set_xlabel("Maturity")
    ax.set_ylabel("Observed \u2013 Model price")
    ax.set_title(eval_date.strftime("%B %Y"), pad=8)

    _style_ax(ax)

    ax.legend(loc="best", frameon=True, fancybox=False,
              edgecolor="#AAAAAA", facecolor="white", framealpha=1.0,
              borderpad=0.6, handlelength=2.2)

    plt.tight_layout(pad=0.6)
    _save_fig(fig, f"price_residuals_{eval_date.strftime('%Y-%m')}")
    plt.show()
    plt.close(fig)
 
# endregion
 
# region 5.PANEL
# _init_worker and _worker must be module-level functions so multiprocessing
# can pickle them when spawning worker processes.
#
# run_panel() distributes the monthly estimation across a pool of worker
# processes — one process per CPU core by default. Set N_WORKERS below to
# limit parallelism if needed.
#
# export_panel() writes the results to Excel. Kept separate so diagnostics
# can be reviewed before committing to disk.

N_WORKERS = None     # None = all available CPU cores; set to int to limit

EXPORT_COLS = [
    "date",
    "b0_actual", "b1_actual", "b2_actual", "b3_actual",
    "k1_actual",  "tau1_actual", "k2_actual", "tau2_actual",
    "b0_ext",    "b1_ext",    "b2_ext",    "b3_ext",
    "k1_ext",    "tau1_ext",  "k2_ext",    "tau2_ext",
    "rmse_price_actual", "rmse_price_ext",
    "rmse_yield_bp_actual", "rmse_yield_bp_ext",
    "objective_actual", "objective_ext",
    "n_bonds", "n_synth",
    "start_b2_actual", "start_b3_actual", "start_tau1_actual", "start_tau2_actual",
    "start_b2_ext",    "start_b3_ext",    "start_tau1_ext",    "start_tau2_ext",
]

# ---- Worker functions (module-level — required for multiprocessing pickling)

_shared_panel   = None
_shared_deposit = None

def _init_worker(panel, deposit):
    """Initialise shared DataFrames in each worker process."""
    global _shared_panel, _shared_deposit
    _shared_panel   = panel
    _shared_deposit = deposit


def _panel_worker(eval_date):
    """
    Estimate NSS for a single date. Runs inside a worker process.
    Returns ('ok', row_dict) on success or ('fail', error_dict) on failure.
    QL objects are stripped before returning — they cannot be pickled.
    """
    try:
        result = estimate_nss(eval_date, _shared_panel, _shared_deposit)
        row = {k: v for k, v in result.items() if not k.startswith("_")}
        row["date"]        = eval_date
        row["tau1_actual"] = 1.0 / row["k1_actual"]
        row["tau2_actual"] = 1.0 / row["k2_actual"]
        row["tau1_ext"]    = 1.0 / row["k1_ext"]
        row["tau2_ext"]    = 1.0 / row["k2_ext"]
        return "ok", row
    except Exception as e:
        return "fail", {"date": eval_date, "error": str(e)}


def run_panel(df_panel, df_deposit, n_workers=None):
    """
    Fit NSS for every month in the panel using a parallel worker pool.
    Each month is estimated independently in a separate process.

    Parameters
    ----------
    n_workers : int or None
        Number of worker processes. None uses all available CPU cores.

    Returns
    -------
    results   : pd.DataFrame of successful fits
    failed_df : pd.DataFrame of failed dates with error messages
    """
    if n_workers is None:
        n_workers = mp.cpu_count()

    month_last = (
        df_panel.assign(_month=df_panel["date"].dt.to_period("M"))
        .groupby("_month")["date"].max()
        .sort_index()
    )
    dates = [pd.Timestamp(d) for d in month_last]
    n_total = len(dates)
    print(f"Panel: {n_total} months, {n_workers} workers")

    rows, failed = [], []

    # fork is used on Linux/macOS — child processes inherit the parent's
    # namespace, so estimate_nss and all QL globals are available without
    # re-importing. Each worker then receives df_panel and df_deposit once
    # via the initializer rather than copying them with every task.
    ctx = mp.get_context("fork")
    with ctx.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(df_panel, df_deposit),
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
    Export panel results to Excel (OUTPUT_FILE, working directory).
    Sheet 'fit_params' holds results; sheet 'failed' is added if any months failed.
    """
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        results.to_excel(writer, sheet_name="fit_params", index=False)
        if not failed_df.empty:
            failed_df.to_excel(writer, sheet_name="failed", index=False)

    print(f"Exported {len(results)} months to {OUTPUT_FILE}")
 
# endregion

# region 6.RUN SINGLE DATE
# Set SINGLE_YM below, then run Step 1 and Step 2 individually in sequence.
# Step 2 requires Step 1 to have been run first (result must be in memory).
 
SINGLE_YM = "2020-08"   # yyyy-mm — resolved automatically to last trading day
 
def resolve_single_date(ym_str: str, df_panel: pd.DataFrame) -> pd.Timestamp:
    """Resolve a 'yyyy-mm' string to the last SGB trading day in that month."""
    target = pd.Period(ym_str, freq="M")
    month_last = (
        df_panel.assign(_month=df_panel["date"].dt.to_period("M"))
        .groupby("_month")["date"].max()
    )
    if target not in month_last.index:
        raise ValueError(f"No trading data found for '{ym_str}'.")
    return pd.Timestamp(month_last[target])
 
# Step 1 — Estimate
eval_date = resolve_single_date(SINGLE_YM, df_panel)
result    = estimate_nss(eval_date, df_panel, df_deposit)
 
# Step 2 — Diagnostics and plots
print_single_diagnostics(result)
plot_curves(result)
plot_price_residuals(result)
 
# endregion

# region 7.RUN PANEL
# Run Steps 1, 2, and 3 individually in sequence. Step 1 must complete before
# running 2 or 3. Steps 2 and 3 are independent of each other and can be
# re-run without re-running Step 1, as long as results is still in memory.
 
# Step 1 — Estimate all months  [slow]
results, failed_df = run_panel(df_panel, df_deposit)
 
# Step 2 — Print diagnostics
print_panel_diagnostics(results, failed_df)
 
# Step 3 — Export to Excel
export_panel(results, failed_df)
 
# endregion

# region 8.RMSE plots 
# Run after panel has been estimated and exported.
#
# Step 1 — Overall yield RMSE time series (actual and extended, all maturities)
# Step 2 — Binned yield RMSE: short 0-2y,
#           medium 2-5y, long 5y+, both fits shown in each subplot.
#
# Bin RMSE method: for each coupon bond, the model is priced by discounting
# every individual cash flow at the NSS zero rate for that cash flow's exact
# time to payment (continuously compounded), summing to a dirty price,
# subtracting accrued interest, then back-solving for the annual compounded
# YTM. This is consistent with the fitting objective and correctly accounts
# for all coupon timings. SSVX T-bills are excluded.

    # region 8.1.Functions

def _nss_zero_rate(t, b0, b1, b2, b3, k1, k2):
    """NSS continuously compounded zero rate at a single maturity t (scalar)."""
    x1 = k1 * t
    x2 = k2 * t
    l1 = 1.0 if abs(x1) < 1e-10 else (1.0 - np.exp(-x1)) / x1
    l2 = l1 - np.exp(-x1)
    l3 = 0.0 if abs(x2) < 1e-10 else (1.0 - np.exp(-x2)) / x2 - np.exp(-x2)
    return b0 + b1 * l1 + b2 * l2 + b3 * l3


def _bond_model_yield(row, eval_date, b0, b1, b2, b3, k1, k2):
    """
    Price a coupon bond from NSS parameters and return the annual compounded YTM.

    Each cash flow is discounted at the NSS zero rate for its own maturity,
    summed to a dirty price, accrued interest subtracted, then converted to YTM.
    Returns nan if pricing fails.
    """
    ql_eval = _to_ql_date(eval_date)
    ql.Settings.instance().evaluationDate = ql_eval

    ql_issue    = _to_ql_date(pd.Timestamp(row["issue_date"]))
    ql_maturity = _to_ql_date(pd.Timestamp(row["maturity_date"]))

    schedule = ql.Schedule(
        ql_issue, ql_maturity, TENOR, CALENDAR,
        BUS_CONV, BUS_CONV, DATE_GEN_RULE, END_OF_MONTH
    )
    bond = ql.FixedRateBond(
        SETTLEMENT_DAYS, FACE, schedule,
        [float(row["coupon"])], DAY_COUNT, BUS_CONV, REDEMPTION, ql_issue
    )

    sett = bond.settlementDate()
    dirty = 0.0
    for cf in bond.cashflows():
        if cf.hasOccurred(sett):
            continue
        t = DAY_COUNT.yearFraction(sett, cf.date())
        if t <= 0.0:
            continue
        z = _nss_zero_rate(t, b0, b1, b2, b3, k1, k2)
        dirty += cf.amount() * np.exp(-z * t)

    clean = dirty - bond.accruedAmount(sett)
    try:
        return float(bond.bondYield(
            ql.BondPrice(clean, ql.BondPrice.Clean),
            DAY_COUNT, ql.Compounded, ql.Annual
        ))
    except Exception:
        return np.nan


def _build_bin_rmse_panel(results: pd.DataFrame,
                           df_panel: pd.DataFrame) -> pd.DataFrame:
    """
    For every date in results, price each coupon bond using the fitted NSS
    parameters for both the actual and extended fits, compute the yield
    residual versus the observed Bloomberg YTM, and aggregate into monthly
    RMSE by maturity bin.

    Returns a long DataFrame with columns:
        date, bin, rmse_actual (bp), rmse_ext (bp)
    """
    bin_edges  = [0.0, 2.0, 5.0, np.inf]
    bin_labels = ["0\u20132y", "2\u20135y", "5y+"]

    params_actual = ["b0_actual", "b1_actual", "b2_actual", "b3_actual",
                     "k1_actual", "k2_actual"]
    params_ext    = ["b0_ext",    "b1_ext",    "b2_ext",    "b3_ext",
                     "k1_ext",    "k2_ext"]

    rows = []
    for _, p in results.iterrows():
        ed = pd.Timestamp(p["date"])

        cs = (df_panel
              .loc[(df_panel["date"] == ed) & (df_panel["coupon"] > 0)]
              .dropna(subset=["yield", "coupon", "issue_date", "maturity_date"])
              .copy())
        if cs.empty:
            continue

        cs["ttm_years"] = (cs["maturity_date"] - cs["date"]).dt.days / 365.25
        cs["bin"] = pd.cut(cs["ttm_years"], bins=bin_edges,
                           labels=bin_labels, right=False)

        for suffix, pcols in (("actual", params_actual), ("ext", params_ext)):
            b0, b1, b2, b3, k1, k2 = (float(p[c]) for c in pcols)
            cs[f"y_model_{suffix}"] = cs.apply(
                lambda r: _bond_model_yield(r, ed, b0, b1, b2, b3, k1, k2),
                axis=1
            )
            cs[f"resid_bp_{suffix}"] = (cs["yield"] - cs[f"y_model_{suffix}"]) * 1e4

        for bin_label, grp in cs.groupby("bin", observed=True):
            row = {"date": ed, "bin": bin_label}
            for suffix in ("actual", "ext"):
                resid = grp[f"resid_bp_{suffix}"].dropna()
                row[f"rmse_{suffix}"] = (
                    float(np.sqrt((resid ** 2).mean())) if len(resid) > 0 else np.nan
                )
            rows.append(row)

    return pd.DataFrame(rows).sort_values(["date", "bin"]).reset_index(drop=True)


# Step 1 — Overall yield RMSE time series
def plot_panel_rmse(results: pd.DataFrame):
    """Time series of overall yield RMSE for the actual and extended fits."""
    dates = pd.to_datetime(results["date"])

    fig, ax = plt.subplots(figsize=(FIG_WIDTH * 1.4, FIG_HEIGHT))
    fig.patch.set_facecolor("white")
    ax.yaxis.grid(True, color="#CCCCCC", linewidth=0.5, linestyle="-", zorder=0)
    ax.set_axisbelow(True)

    ax.plot(dates, results["rmse_yield_bp_actual"],
            color=COLOR_ACTUAL, lw=1.8, ls="--",  zorder=3,
            label="Actual fit")
    ax.plot(dates, results["rmse_yield_bp_ext"],
            color=COLOR_EXTENDED, lw=1.8, ls="-", zorder=2,
            label="Extended fit")

    #ax.set_xlabel("Date")
    ax.set_ylabel("Yield RMSE (bp)")
    ax.set_xlim(dates.min(), dates.max())
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))
    _style_ax(ax)
    ax.set_ylim(0, 16)
    ax.set_yticks([2, 4, 6, 8, 10, 12, 14])
    ax.legend(loc="upper right", frameon=True, fancybox=False,
              edgecolor="#AAAAAA", facecolor="white", framealpha=1.0,
              borderpad=0.6, handlelength=2.2)
    plt.tight_layout(pad=0.6)
    _save_fig(fig, "rmse_nominal")
    plt.show()
    plt.close(fig)


# Step 2 — Binned yield RMSE: three subplots side by side
def plot_panel_rmse_bins(bin_df: pd.DataFrame):
    """
    Three subplots (short / medium / long), each showing the monthly yield
    RMSE time series for both the actual (navy solid) and extended
    (charcoal dashed) fits. Subplots share the y-axis.
    """
    bin_labels = ["0\u20132y", "2\u20135y", "5y+"]

    fig, axes = plt.subplots(
        1, 3,
        figsize=(FIG_WIDTH * 2.5, FIG_HEIGHT),
        sharey=True
    )
    fig.patch.set_facecolor("white")

    for ax, label in zip(axes, bin_labels):
        sub = bin_df[bin_df["bin"] == label].sort_values("date")
        dates = pd.to_datetime(sub["date"])

        ax.yaxis.grid(True, color="#CCCCCC", linewidth=0.5, linestyle="-", zorder=0)
        ax.set_axisbelow(True)

        ax.plot(dates, sub["rmse_actual"],
                color=COLOR_ACTUAL, lw=1.6, ls="--",  zorder=3,
                label="Actual fit")
        ax.plot(dates, sub["rmse_ext"],
                color=COLOR_EXTENDED, lw=1.6, ls="-", zorder=2,
                label="Extended fit")

        ax.set_title(label, pad=8)
        #ax.set_xlabel("Date")
        ax.set_xlim(dates.min(), dates.max())
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))
        _style_ax(ax)
        ax.set_ylim(0, 22)
        ax.set_yticks([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])

    axes[0].set_ylabel("Yield RMSE (bp)")
    axes[2].legend(loc="upper right", frameon=True, fancybox=False,
                   edgecolor="#AAAAAA", facecolor="white", framealpha=1.0,
                   borderpad=0.6, handlelength=2.2)

    plt.tight_layout(pad=0.6)
    _save_fig(fig, "rmse_bins_nominal")
    plt.show()
    plt.close(fig)

    # endregion

    # region 8.2.Execution

# Reads results from OUTPUT_FILE (set in CONFIGURATION).
# df_panel is always available from the data preparation at script startup.

# Step 1 — Load panel results from Excel
results = pd.read_excel(OUTPUT_FILE, sheet_name="fit_params")
results["date"] = pd.to_datetime(results["date"])
results = results.loc[results["date"] >= pd.Timestamp("2004-01-01")].reset_index(drop=True)

# Step 2 — Plot overall RMSE
plot_panel_rmse(results)

# Step 3 — Build bin RMSE and plot  [slow: prices every bond via QuantLib]
bin_df = _build_bin_rmse_panel(results, df_panel)
plot_panel_rmse_bins(bin_df)

    # endregion

# endregion

# region 9.BID-ASK SPREADS
# Computes mean and median bid-ask yield spreads (in bp) across nominal SGBs
# for four maturity sets, exports to Excel, and plots the mean time series.
#
# Sets:
#   1-5y  — bonds with TTM in [1, 5) years
#   2-10y — bonds with TTM in [2, 10) years
#   1-10y — bonds with TTM in [1, 10) years
#   All   — all bonds with valid bid/ask quotes (no upper TTM limit)
#
# Output file: nominal_bid_ask.xlsx
# Sheet: "bid_ask_summary"

BID_ASK_OUTPUT = "NOMINAL_CURVE_ESTIMATION/nominal_bid_ask.xlsx"

SETS = {
    "1_5y":  (1.0,  5.0),
    "2_10y": (2.0, 10.0),
    "1_10y": (1.0, 10.0),
    "all":   (1.0, np.inf),
}


def compute_bid_ask_summary(df_sgbs: pd.DataFrame) -> pd.DataFrame:
    work = df_sgbs.dropna(subset=["bid_yield", "ask_yield"]).copy()
    work["ttm_years"] = (work["maturity_date"] - work["date"]).dt.days / 365.25
    work["spread_bp"] = (work["bid_yield"] - work["ask_yield"]).abs() * 1e4

    agg_frames = []
    for label, (lo, hi) in SETS.items():
        mask = (work["ttm_years"] >= lo) & (work["ttm_years"] < hi)
        grp  = (work.loc[mask]
                .groupby("date")["spread_bp"]
                .agg(mean="mean", median="median", count="count")
                .rename(columns={
                    "mean":   f"mean_{label}",
                    "median": f"median_{label}",
                    "count":  f"n_{label}",
                }))
        agg_frames.append(grp)

    summary = (pd.concat(agg_frames, axis=1)
               .reset_index()
               .sort_values("date")
               .reset_index(drop=True))
    return summary


def plot_bid_ask_means(summary: pd.DataFrame):
    """
    Time series of the cross-sectional mean bid-ask
    yield spread for the four maturity sets.
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

    for label in SETS:
        col = f"mean_{label}"
        if col not in summary.columns:
            continue
        ax.plot(dates, summary[col],
                color=colors[label], lw=1.8, ls=styles[label],
                zorder=3, label=labels[label])

    ax.set_ylabel("Mean bid\u2013ask spread (bp)")
    ax.set_xlim(dates.min(), dates.max())
    ax.set_ylim(0, 14)
    ax.set_yticks([2, 4, 6, 8, 10, 12])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))
    _style_ax(ax)
    ax.legend(loc="upper left", frameon=True, fancybox=False,
              edgecolor="#AAAAAA", facecolor="white", framealpha=1.0,
              borderpad=0.6, handlelength=2.2)

    plt.tight_layout(pad=0.6)
    _save_fig(fig, "bid_ask_nominal")
    plt.show()
    plt.close(fig)


# Execution

# Step 1 — Compute and export
bid_ask_summary = compute_bid_ask_summary(
    df_SGBs.loc[df_SGBs["date"] >= pd.Timestamp("2004-01-01")]
)

with pd.ExcelWriter(BID_ASK_OUTPUT, engine="openpyxl") as writer:
    bid_ask_summary.to_excel(writer, sheet_name="bid_ask_summary", index=False)

print(f"Bid-ask summary exported to {BID_ASK_OUTPUT} "
      f"({len(bid_ask_summary)} dates)")

# Step 2 — Plot mean time series
plot_bid_ask_means(bid_ask_summary)

# endregion

# region 10.BONDS OUTSTANDING
# Two-panel figure: upper panel shows each bond's TTM over time as a separate
# line; lower panel shows the monthly count of distinct bonds outstanding.
# Data comes from raw_sgb (full history, no TTM filter).
# Run independently — only requires raw_sgb from script startup.

def plot_bonds_outstanding():
    """
    Two-panel bonds outstanding figure.

    Upper panel: TTM trajectory of each SGB series over time.
    Lower panel: number of distinct bonds outstanding per date.
    """
    # ---- Data preparation
    cols = ["Date", "Issue date", "Maturity date", "ISIN", "Serie"]
    df = (raw_sgb.loc[:, cols]
          .rename(columns={"Date": "date", "Issue date": "issue_date",
                            "Maturity date": "maturity_date",
                            "ISIN": "isin", "Serie": "serie"})
          .copy())

    for col in ["date", "issue_date", "maturity_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df = df.dropna(subset=["date", "issue_date", "maturity_date", "serie"])
    df = df.loc[df["date"] >= START].copy()

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
        gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.06}
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
    #ax2.set_xlabel("Date")
    ax2.set_ylim(0, 15)   
    ax2.set_yticks([0, 5, 10])
    ax2.margins(x=0)

    # ---- Shared styling
    for ax in (ax1, ax2):
        _style_ax(ax)

    plt.tight_layout(pad=0.6)
    _save_fig(fig, "bonds_outstanding_nominal")
    plt.show()
    plt.close(fig)


plot_bonds_outstanding()

# endregion

# region 11.ZERO-COUPON YIELDS
# Computes monthly zero-coupon yields at maturities 1m–120m from the
# extended NSS fit, exports to zero_yields_SGB.xlsx,
# and produces two plots.
#
# Data source: reads results from OUTPUT_FILE (exported panel parameters).
#   df_SSVX and df_deposit are always available from script startup.
#
# Step 1 — Compute and export zero-coupon yield panel
# Step 2 — Time series of selected maturities (0.5y, 2y, 4y, 6y, 8y, 10y)
# Step 3 — Short end: curve-implied 1m/3m/6m vs SSVX T-bill yields and
#           deposit rate

ZERO_YIELDS_FILE = "NOMINAL_CURVE_ESTIMATION/zero_yields_SGB.xlsx"

# Maturities to export (months)
EXPORT_MATURITIES = list(range(1, 121))

# Maturities to plot in the term structure figure (months)
TERM_STRUCTURE_MATURITIES = {
    "0.5y":  6,
    "2y":   24,
    "4y":   48,
    "6y":   72,
    "8y":   96,
    "10y": 120,
}

# Colours for term structure plot — progression from light to deep blue
TERM_STRUCTURE_COLORS = [
    "#A8C8E8",   
    "#7AAFD4",   
    "#4D96C0",   
    "#2878A8",   
    "#0D5A90",   
    "#0D2B4E",   
]


def _nss_vec(t, b0, b1, b2, b3, k1, k2):
    """Vectorised NSS zero yield (continuously compounded)."""
    x1 = k1 * t
    x2 = k2 * t
    l1 = np.where(np.abs(x1) < 1e-12, 1.0, (1.0 - np.exp(-x1)) / x1)
    l2 = l1 - np.exp(-x1)
    l3 = np.where(np.abs(x2) < 1e-12, 0.0,
                  (1.0 - np.exp(-x2)) / x2 - np.exp(-x2))
    return b0 + b1 * l1 + b2 * l2 + b3 * l3


# ---- Step 1 — Compute and export zero-coupon yield panels

results_zy = pd.read_excel(OUTPUT_FILE, sheet_name="fit_params")
results_zy["date"] = pd.to_datetime(results_zy["date"])
results_zy = results_zy.sort_values("date").reset_index(drop=True)
results_zy = results_zy.loc[results_zy["date"] >= pd.Timestamp("2004-01-01")].reset_index(drop=True)

b0 = results_zy["b0_ext"].to_numpy(float)
b1 = results_zy["b1_ext"].to_numpy(float)
b2 = results_zy["b2_ext"].to_numpy(float)
b3 = results_zy["b3_ext"].to_numpy(float)
k1 = results_zy["k1_ext"].to_numpy(float)
k2 = results_zy["k2_ext"].to_numpy(float)

zero_yields = pd.DataFrame({"date": results_zy["date"]})
for m in EXPORT_MATURITIES:
    zero_yields[f"y_{m}m"] = _nss_vec(m / 12.0, b0, b1, b2, b3, k1, k2)

with pd.ExcelWriter(ZERO_YIELDS_FILE, engine="openpyxl") as writer:
    zero_yields.to_excel(writer, sheet_name="zero_yields", index=False)
    results_zy.to_excel(writer, sheet_name="fit_params", index=False)

print(f"Exported {len(zero_yields)} months to {ZERO_YIELDS_FILE}")


# ---- Step 2 — Term structure time series

def plot_term_structure(zero_yields: pd.DataFrame, df_deposit: pd.DataFrame):
    dates = pd.to_datetime(zero_yields["date"])

    fig, ax = plt.subplots(figsize=(FIG_WIDTH * 1.4, FIG_HEIGHT))
    fig.patch.set_facecolor("white")
    ax.yaxis.grid(True, color="#CCCCCC", linewidth=0.5, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    ax.axhline(0, color="#666666", lw=0.8, ls="-", zorder=1)

    dep_plot = df_deposit.loc[:, ["date", "deposit"]].dropna().copy()
    dep_plot["date"] = pd.to_datetime(dep_plot["date"])
    ax.plot(dep_plot["date"], dep_plot["deposit"] * 100,
            color="#222222", lw=1.4, ls=":", zorder=2,
            label="Deposit rate")

    for (label, m), color in zip(TERM_STRUCTURE_MATURITIES.items(),
                                  TERM_STRUCTURE_COLORS):
        ax.plot(dates, zero_yields[f"y_{m}m"] * 100,
                color=color, lw=1.6, zorder=3, label=label)

    #ax.set_xlabel("Date")
    ax.set_ylabel("Yield (%)")
    ax.set_xlim(dates.min(), dates.max())
    ax.set_ylim(-2, 7)
    ax.set_yticks(range(-2, 8))   
    ax.set_yticklabels([
        "" if v in (-2, 7) else str(v)
        for v in range(-2, 8)
    ])
    _style_ax(ax)
    ax.legend(loc="upper right", frameon=True, fancybox=False,
              edgecolor="#AAAAAA", facecolor="white", framealpha=1.0,
              borderpad=0.6, handlelength=2.2, ncol=4)
    plt.tight_layout(pad=0.6)
    _save_fig(fig, "term_structure_nominal")
    plt.show()
    plt.close(fig)


plot_term_structure(zero_yields, df_deposit)


# ---- Step 3 — Short end: curve-implied vs SSVX T-bills and deposit rate

def plot_short_end(zero_yields: pd.DataFrame, df_ssvx: pd.DataFrame,
                   df_deposit: pd.DataFrame):
    dates_zy = pd.to_datetime(zero_yields["date"])

    dep_plot = df_deposit.loc[:, ["date", "deposit"]].dropna().copy()
    dep_plot["date"] = pd.to_datetime(dep_plot["date"])

    curve = {
        "1m": zero_yields[["date", "y_1m"]].rename(columns={"y_1m": "y"}),
        "3m": zero_yields[["date", "y_3m"]].rename(columns={"y_3m": "y"}),
        "6m": zero_yields[["date", "y_6m"]].rename(columns={"y_6m": "y"}),
    }

    curve_colors = {"1m": "#0D2B4E", "3m": "#1A5EA8", "6m": "#5B9BD5"}
    ssvx_colors  = {"1m": "#1A6B3A", "3m": "#2E9E58", "6m": "#72C48A"}
    ssvx_tenors  = {"1m": "SSVX_1m", "3m": "SSVX_3m", "6m": "SSVX_6m"}

    fig, ax = plt.subplots(figsize=(FIG_WIDTH * 1.4, FIG_HEIGHT))
    fig.patch.set_facecolor("white")
    ax.yaxis.grid(True, color="#CCCCCC", linewidth=0.5, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    ax.axhline(0, color="#666666", lw=0.8, ls="-", zorder=1)

    ax.plot(dep_plot["date"], dep_plot["deposit"] * 100,
            color="#222222", lw=1.4, ls=":", zorder=2,
            label="Deposit rate")

    for tenor, serie in ssvx_tenors.items():
        sub = df_ssvx.loc[df_ssvx["serie"] == serie].dropna(
            subset=["price", "settlement_date", "maturity_date"]
        ).sort_values("date").copy()

        t_yr = sub.apply(
            lambda r: DAY_COUNT.yearFraction(
                _to_ql_date(pd.Timestamp(r["settlement_date"])),
                _to_ql_date(pd.Timestamp(r["maturity_date"]))
            ), axis=1
        )
        sub["y_cc"] = -np.log(sub["price"] / 100.0) / t_yr

        ax.plot(sub["date"], sub["y_cc"] * 100,
                color=ssvx_colors[tenor], lw=1.2, ls="--", zorder=3,
                label=f"SSVX {tenor}")

    for tenor, df_c in curve.items():
        df_c = df_c.sort_values("date")
        ax.plot(pd.to_datetime(df_c["date"]), df_c["y"] * 100,
                color=curve_colors[tenor], lw=1.8, ls="-", zorder=4,
                label=f"Curve {tenor}")

    #ax.set_xlabel("Date")
    ax.set_ylabel("Yield (%)")
    ax.set_xlim(dates_zy.min(), dates_zy.max())
    ax.set_ylim(-2, 7)
    ax.set_yticks(range(-2, 8))
    ax.set_yticklabels([
        "" if v in (-2, 7) else str(v)
        for v in range(-2, 8)
    ])
    _style_ax(ax)
    ax.legend(loc="upper right", frameon=True, fancybox=False,
              edgecolor="#AAAAAA", facecolor="white", framealpha=1.0,
              borderpad=0.6, handlelength=2.2, ncol=4)
    plt.tight_layout(pad=0.6)
    _save_fig(fig, "short_end_fit_nominal")
    plt.show()
    plt.close(fig)


plot_short_end(zero_yields, df_SSVX, df_deposit)

# endregion


