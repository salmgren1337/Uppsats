import pandas as pd
import QuantLib as ql
import numpy as np
import matplotlib.pyplot as plt

# IMPORTS 
raw_data_nominal = pd.read_excel("statsobligationer_data.xlsx", sheet_name = "SGB long")
short_rates = pd.read_excel("statsobligationer_data.xlsx", sheet_name = "korta räntor riksbanken")

START = pd.Timestamp("2000-01-01")
END = pd.Timestamp("2026-01-31")
ENTER_DATE = "2022-04-19"

MAX_MAT = 15
MIN_MAT = 90

# region PREPARE BOND DATA

# ---- 1) PREPARE SGB DATA

keep_cols = [
    "Date",
    "PX_LAST",
    "YLD_YTM_MID",
    "BID_YIELD",
    "ASK_YIELD",
    "Kupong",
    "Issue date",
    "Maturity date",
    "ISIN",
    "Serie",
]

df_SGBs = raw_data_nominal.loc[:, keep_cols].copy()

# Value formatting:
df_SGBs["Date"] = pd.to_datetime(df_SGBs["Date"], errors="coerce")
df_SGBs["Issue date"] = pd.to_datetime(df_SGBs["Issue date"], errors="coerce")
df_SGBs["Maturity date"] = pd.to_datetime(df_SGBs["Maturity date"], errors="coerce")

for col in ["PX_LAST", "YLD_YTM_MID", "BID_YIELD", "ASK_YIELD", "Kupong"]:
    df_SGBs[col] = pd.to_numeric(df_SGBs[col], errors="coerce")

df_SGBs["YLD_YTM_MID"] = df_SGBs["YLD_YTM_MID"] / 100.0
df_SGBs["BID_YIELD"] = df_SGBs["BID_YIELD"] / 100.0
df_SGBs["ASK_YIELD"] = df_SGBs["ASK_YIELD"] / 100.0
df_SGBs["Kupong"] = df_SGBs["Kupong"] / 100.0

# Raname columns:
df_SGBs = df_SGBs.rename(columns={
    "Date": "date",
    "PX_LAST": "price",
    "YLD_YTM_MID": "yield",
    "BID_YIELD": "bid_yield",
    "ASK_YIELD": "ask_yield",
    "Kupong": "coupon",
    "Issue date": "issue_date",
    "Maturity date": "maturity_date",
    "ISIN": "isin",
    "Serie": "serie"
})

# Set sample period and sort by date and maturity date:
df_SGBs = df_SGBs.loc[(df_SGBs["date"] >= START) & (df_SGBs["date"] <= END)]
df_SGBs = df_SGBs.sort_values(["date", "maturity_date"]).reset_index(drop=True)

# Filter bonds with less than 90 days and more than approx 15 years to maturity:
time_to_maturity = (df_SGBs["maturity_date"] - df_SGBs["date"]).dt.days
df_SGBs = df_SGBs[(time_to_maturity >= MIN_MAT) & (time_to_maturity <= MAX_MAT * 365)]
df_SGBs = df_SGBs.reset_index(drop=True)

# ---- 2) PREPARE SHORT RATE DATA

short_cols = [
    "SSVX_1m",
    "SSVX_3m",
    "SSVX_6m",
]
sr_wide = short_rates.loc[:, ["date"] + short_cols].copy()

# Convert wide to long format:
df_SSVX = sr_wide.melt(             
    id_vars = ["date"],
    value_vars=short_cols,
    var_name="serie",
    value_name="yield"
).dropna(subset=["yield"]).reset_index(drop=True)

# Build zero-coupon bonds from estimated zero-coupon yields:
tenor_months = {"SSVX_1m": 1, "SSVX_3m": 3, "SSVX_6m": 6}
df_SSVX["months"] = df_SSVX["serie"].map(tenor_months).astype("Int64")

df_SSVX["coupon"] = 0.0
df_SSVX["issue_date"] = df_SSVX["date"]
df_SSVX["maturity_date"] = df_SSVX["issue_date"] + pd.to_timedelta(df_SSVX["months"] * 30, unit="D")
df_SSVX["isin"] = ""

# Compute zero-coupon prices from yields: 
df_SSVX["yield"] = pd.to_numeric(df_SSVX["yield"], errors="coerce") / 100.0
days = (df_SSVX["maturity_date"] - df_SSVX["issue_date"]).dt.days
t = days / 360.0 
df_SSVX["price"] = 100.0 / (1.0 + df_SSVX["yield"]) ** t

# Prepare merge with SGBs:
df_SSVX = df_SSVX.sort_values(["date", "maturity_date"]).reset_index(drop=True) 
df_SSVX = df_SSVX.loc[:, ["date", "price", "yield", "coupon", "issue_date", "maturity_date", "isin", "serie"]].copy()

df_SSVX = df_SSVX.loc[(df_SSVX["date"] >= START) & (df_SSVX["date"] <= END)]

# ---- 3) COMBINE SSVX & SGBs

# SSVX is EOM and SGB is last trading day > align date column in SSVX
df_SGBs["month"] = df_SGBs["date"].dt.to_period("M")
df_SSVX["month"] = df_SSVX["date"].dt.to_period("M")

month_trading_date = df_SGBs.groupby("month")["date"].max()
df_SSVX["date"] = df_SSVX["month"].map(month_trading_date)
df_SSVX["issue_date"] = df_SSVX["date"]

df_SGBs = df_SGBs.drop(columns=["month"])
df_SSVX = df_SSVX.drop(columns=["month"])

# Align dataframes:
required_cols = [
    "date",
    "price",
    "yield",
    "coupon",
    "issue_date",
    "maturity_date",
    "isin",
    "serie",
    "bid_yield",
    "ask_yield",
]

df_SGBs = df_SGBs.loc[:, required_cols].copy()

df_SSVX["bid_yield"] = np.nan
df_SSVX["ask_yield"] = np.nan
df_SSVX = df_SSVX.loc[:, required_cols].copy()

for c in ["date", "issue_date", "maturity_date"]:
    df_SGBs[c] = pd.to_datetime(df_SGBs[c], errors="coerce")
    df_SSVX[c] = pd.to_datetime(df_SSVX[c], errors="coerce")

for c in ["price", "yield", "coupon", "bid_yield", "ask_yield"]:
    df_SGBs[c] = pd.to_numeric(df_SGBs[c], errors="coerce")
    df_SSVX[c] = pd.to_numeric(df_SSVX[c], errors="coerce")

df_SGBs["isin"] = df_SGBs["isin"].astype(str)
df_SSVX["isin"] = df_SSVX["isin"].astype(str)
df_SGBs["serie"] = df_SGBs["serie"].astype(str)
df_SSVX["serie"] = df_SSVX["serie"].astype(str)

# Concatenate rows & sort:
df_SGBs = pd.concat([df_SGBs, df_SSVX], ignore_index=True).copy()
df_SGBs = df_SGBs.sort_values(["date", "maturity_date"]).reset_index(drop=True)

# ---- 4) PREPARE DEPOSIT RATE DATA

df_deposit = short_rates.loc[:, ["date", "deposit"]].copy()
df_deposit["date"] = pd.to_datetime(df_deposit["date"], errors="coerce") 
df_deposit["deposit"] = pd.to_numeric(df_deposit["deposit"], errors="coerce") / 100.0

# Map deposit rate EOM to SGB last trading day:
df_deposit["month"] = df_deposit["date"].dt.to_period("M")
df_deposit["date"] = df_deposit["month"].map(month_trading_date) 
df_deposit = df_deposit.drop(columns=["month"]).dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

# endregion

# region CROSS SECTION

    # region SETUP

# ---- 1) SETTINGS

eval_date = pd.Timestamp(ENTER_DATE)
cross_section = df_SGBs.loc[df_SGBs["date"] == eval_date].copy()
cross_section = cross_section.sort_values("maturity_date").reset_index(drop=True)

if cross_section.empty:
    raise ValueError(f"No instruments found for evaluation date {eval_date.date()}")

ql_eval_date = ql.Date(eval_date.day, eval_date.month, eval_date.year)
ql.Settings.instance().evaluationDate = ql_eval_date

# Day-count and bond conventions
day_count = ql.Thirty360(ql.Thirty360.European)
tenor = ql.Period(ql.Annual)
date_gen_rule = ql.DateGeneration.Backward
calendar = ql.Sweden()
business_convention = ql.ModifiedFollowing
end_of_month = False

settlement_days = 2
face = 100.0
redemption = 100.0

# ---- 2) BUILD HELPERS FOR ACTUAL BONDS

helpers_actual = []
bonds_actual = []
quoted_price_actual = []
quoted_yield_actual = []
maturities_actual = []

for _, row in cross_section.iterrows():
    issue = row["issue_date"]
    maturity = row["maturity_date"]

    ql_issue = ql.Date(issue.day, issue.month, issue.year)
    ql_maturity = ql.Date(maturity.day, maturity.month, maturity.year)

    schedule = ql.Schedule(
        ql_issue,
        ql_maturity,
        tenor,
        calendar,
        business_convention,
        business_convention,
        date_gen_rule,
        end_of_month
    )

    coupon_rate = float(row["coupon"])
    clean_price = float(row["price"])
    quoted_yield = float(row["yield"])

    bond = ql.FixedRateBond(
        settlement_days,
        face,
        schedule,
        [coupon_rate],
        day_count,
        business_convention,
        redemption,
        ql_issue
    )

    helper = ql.FixedRateBondHelper(
        ql.QuoteHandle(ql.SimpleQuote(clean_price)),
        settlement_days,
        face,
        schedule,
        [coupon_rate],
        day_count,
        business_convention,
        redemption,
        ql_issue
    )

    bonds_actual.append(bond)
    helpers_actual.append(helper)
    quoted_price_actual.append(clean_price)
    quoted_yield_actual.append(quoted_yield)
    maturities_actual.append(row["maturity_date"])

quoted_price_actual = np.array(quoted_price_actual, dtype=float)
quoted_yield_actual = np.array(quoted_yield_actual, dtype=float)
maturities_actual = pd.to_datetime(maturities_actual)

print("Actual helpers built:", len(helpers_actual))

# ---- 3) CREATE SYNTHETIC ZERO-COUPON BONDS FROM BOOTSTRAPPED CURVE WITH CUBIC SPLINE INTERPOLATION

pre_curve = ql.PiecewiseLogCubicDiscount(
    settlement_days, calendar, helpers_actual, day_count
)
pre_curve.enableExtrapolation()

curve_reference_date = calendar.advance(ql_eval_date, settlement_days, ql.Days)
actual_maturity_dates = sorted({b.maturityDate() for b in bonds_actual})
curve_nodes = sorted(actual_maturity_dates)

synthetic_helpers = []

for d1, d2 in zip(curve_nodes[:-1], curve_nodes[1:]):
    t1 = day_count.yearFraction(curve_reference_date, d1)
    t2 = day_count.yearFraction(curve_reference_date, d2)

    if t1 <= 0.0 or t2 <= 0.0 or t2 <= t1:
        continue

    t_mid = 0.5 * (t1 + t2)
    months_mid = max(1, int(round(t_mid * 12)))
    mid_date = calendar.advance(curve_reference_date, ql.Period(months_mid, ql.Months))

    if not (d1 < mid_date < d2):
        continue

    synth_price = face * pre_curve.discount(mid_date)

    schedule_mid = ql.Schedule(
        curve_reference_date,
        mid_date,
        ql.Period(ql.Once),
        calendar,
        business_convention,
        business_convention,
        ql.DateGeneration.Forward,
        False
    )

    synth_helper = ql.FixedRateBondHelper(
        ql.QuoteHandle(ql.SimpleQuote(float(synth_price))),
        settlement_days,
        face,
        schedule_mid,
        [0.0],
        day_count,
        business_convention,
        redemption,
        curve_reference_date
    )

    synthetic_helpers.append(synth_helper)

helpers_extended = helpers_actual + synthetic_helpers

print("Synthetic helpers built:", len(synthetic_helpers))
print("Total helpers in extended set:", len(helpers_extended))

    # endregion

    # region PREPARE ESTIMATION

# ---- 1) SHORT & LONG END ANCHORS FOR STARTING VALUES

deposit_row = df_deposit.loc[df_deposit["date"] == eval_date]
if deposit_row.empty:
    raise ValueError(f"No deposit rate found for {eval_date}")

deposit_rate = float(deposit_row["deposit"].iloc[0])
deposit_rate_cc = np.log(1.0 + deposit_rate)

max_maturity = max(b.maturityDate() for b in bonds_actual)
long_rate_cc = pre_curve.zeroRate(max_maturity, day_count, ql.Continuous).rate()

b0 = long_rate_cc
b1 = deposit_rate_cc - b0

print(
    f"Start anchors: short={deposit_rate_cc:.6f}, "
    f"long={long_rate_cc:.6f}, b0={b0:.6f}, b1={b1:.6f}"
)

# ---- 2) BOUNDS

b0_min, b0_max = -0.03, 0.06
b1_min, b1_max = -0.10, 0.10
b2_min, b2_max = -0.05, 0.05
b3_min, b3_max = -0.05, 0.05

tau1_min, tau1_max = 1.0, 5.0
tau2_min, tau2_max = 4.0, 20.0

kappa1_min, kappa1_max = 1.0 / tau1_max, 1.0 / tau1_min
kappa2_min, kappa2_max = 1.0 / tau2_max, 1.0 / tau2_min

lower = ql.Array(6)
upper = ql.Array(6)

lower[0], upper[0] = b0_min, b0_max
lower[1], upper[1] = b1_min, b1_max
lower[2], upper[2] = b2_min, b2_max
lower[3], upper[3] = b3_min, b3_max
lower[4], upper[4] = kappa1_min, kappa1_max
lower[5], upper[5] = kappa2_min, kappa2_max

constraint = ql.NonhomogeneousBoundaryConstraint(lower, upper)

# ---- 3) PENALTIES AROUND STARTING VALUES

l2 = ql.Array(6)
l2[0] = 0.05   
l2[1] = 0.01   
l2[2] = 0.20   
l2[3] = 0.20   
l2[4] = 0.05   
l2[5] = 0.02   

# ---- 4) FIT SETUP

accuracy = 1e-10
max_evaluations = 15000
simplex_lambda = 0.005

fitting = ql.SvenssonFitting(
    ql.Array(),
    ql.Simplex(simplex_lambda),
    l2,
    0.0,
    50.0,
    constraint
)

# ---- 5) MULTISTART GRID

b2_grid = [-0.02, -0.01, 0.0, 0.01, 0.02]
b3_grid = [-0.02, -0.01, 0.0, 0.01, 0.02]
tau1_grid = [1.5, 2.0, 3.0, 4.5]
tau2_grid = [6.0, 8.0, 10.0, 12.0]

    # endregion

    # region MULTIPLE START NSS ESTIMATION

# ---- 1) PRICE ACTUAL BONDS FROM ESTIMATED CURVE

def price_actual_bonds(curve):
    curve_handle = ql.YieldTermStructureHandle(curve)
    engine = ql.DiscountingBondEngine(curve_handle)

    model_prices = []
    model_yields = []

    for bond in bonds_actual:
        bond.setPricingEngine(engine)
        model_prices.append(float(bond.cleanPrice()))
        model_yields.append(float(bond.bondYield(day_count, ql.Compounded, ql.Annual)))

    return np.array(model_prices), np.array(model_yields)

# ---- 2) MULTISTART ON ACTUAL BONDS

best_objective_actual = float("inf")
best_curve_actual = None
best_params_actual = None
best_start_actual = None

for b2_start in b2_grid:
    for b3_start in b3_grid:
        for tau1 in tau1_grid:
            for tau2 in tau2_grid:
                if tau2 <= tau1:
                    continue

                guess_ms = ql.Array(6)
                guess_ms[0] = b0
                guess_ms[1] = b1
                guess_ms[2] = b2_start
                guess_ms[3] = b3_start
                guess_ms[4] = 1.0 / tau1
                guess_ms[5] = 1.0 / tau2

                try:
                    curve_try = ql.FittedBondDiscountCurve(
                        settlement_days,
                        calendar,
                        helpers_actual,
                        day_count,
                        fitting,
                        accuracy,
                        max_evaluations,
                        guess_ms
                    )

                    fit_result = curve_try.fitResults()
                    objective_value = fit_result.minimumCostValue()

                    if objective_value < best_objective_actual:
                        best_objective_actual = objective_value
                        best_curve_actual = curve_try
                        best_params_actual = list(fit_result.solution())
                        best_start_actual = (b2_start, b3_start, tau1, tau2)

                except RuntimeError:
                    continue

if best_curve_actual is None:
    raise RuntimeError(f"Multistart failed for ACTUAL on {eval_date}")

print(
    f"MULTISTART ACTUAL best start: "
    f"b2={best_start_actual[0]}, b3={best_start_actual[1]}, "
    f"tau1={best_start_actual[2]}, tau2={best_start_actual[3]}, "
    f"objective={best_objective_actual:.8f}"
)
print("MULTISTART ACTUAL params:", best_params_actual)

# ---- 3) MULTISTART ON EXTENDED SET 

best_objective_ext = float("inf")
best_curve_ext = None
best_params_ext = None
best_start_ext = None

for b2_start in b2_grid:
    for b3_start in b3_grid:
        for tau1 in tau1_grid:
            for tau2 in tau2_grid:
                if tau2 <= tau1:
                    continue

                guess_ms = ql.Array(6)
                guess_ms[0] = b0
                guess_ms[1] = b1
                guess_ms[2] = b2_start
                guess_ms[3] = b3_start
                guess_ms[4] = 1.0 / tau1
                guess_ms[5] = 1.0 / tau2

                try:
                    curve_try = ql.FittedBondDiscountCurve(
                        settlement_days,
                        calendar,
                        helpers_extended,
                        day_count,
                        fitting,
                        accuracy,
                        max_evaluations,
                        guess_ms
                    )

                    fit_result = curve_try.fitResults()
                    objective_value = fit_result.minimumCostValue()

                    if objective_value < best_objective_ext:
                        best_objective_ext = objective_value
                        best_curve_ext = curve_try
                        best_params_ext = list(fit_result.solution())
                        best_start_ext = (b2_start, b3_start, tau1, tau2)

                except RuntimeError:
                    continue

if best_curve_ext is None:
    raise RuntimeError(f"Multistart failed for EXTENDED on {eval_date}")

print(
    f"MULTISTART EXTENDED best start: "
    f"b2={best_start_ext[0]}, b3={best_start_ext[1]}, "
    f"tau1={best_start_ext[2]}, tau2={best_start_ext[3]}, "
    f"objective={best_objective_ext:.8f}"
)
print("MULTISTART EXTENDED params:", best_params_ext)

# ---- 4) FIT DIAGNOSTICS ON ACTUAL BONDS

model_prices_actualfit, model_yields_actualfit = price_actual_bonds(best_curve_actual)
model_prices_extendedfit, model_yields_extendedfit = price_actual_bonds(best_curve_ext)

rmse_price_actualfit = float(np.sqrt(np.mean((quoted_price_actual - model_prices_actualfit) ** 2)))
rmse_price_extfit = float(np.sqrt(np.mean((quoted_price_actual - model_prices_extendedfit) ** 2)))

rmse_yield_actualfit_bp = float(np.sqrt(np.mean((quoted_yield_actual - model_yields_actualfit) ** 2)) * 1e4)
rmse_yield_extfit_bp = float(np.sqrt(np.mean((quoted_yield_actual - model_yields_extendedfit) ** 2)) * 1e4)

print("RMSE price (actual-only fit):", rmse_price_actualfit)
print("RMSE price (extended fit):   ", rmse_price_extfit)
print("RMSE yield bp (actual-only fit):", rmse_yield_actualfit_bp)
print("RMSE yield bp (extended fit):   ", rmse_yield_extfit_bp)

        # endregion

    # region CROSS SECTION PLOTS

# ---- PLOT SETTINGS

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

dc_plot = day_count
comp = ql.Continuous

max_date = max(b.maturityDate() for b in bonds_actual)
times = np.arange(1.0 / 12.0, 35.0 + 1.0 / 12.0, 1.0 / 12.0)
maturities_plot = pd.to_datetime(maturities_actual)

color_actual = "#000000"
color_extended = "#0B1EFF"
color_boot = "#E31A1C"
color_observed = "#E31A1C"

# ---- ZERO-COUPON CURVES

times_used = []
zero_actual = []
zero_extended = []
zero_boot = []

for t in times:
    months = int(round(t * 12))
    d = calendar.advance(ql_eval_date, ql.Period(months, ql.Months))

    if d > max_date:
        break

    times_used.append(months / 12.0)
    zero_actual.append(best_curve_actual.zeroRate(d, dc_plot, comp).rate())
    zero_extended.append(best_curve_ext.zeroRate(d, dc_plot, comp).rate())
    zero_boot.append(pre_curve.zeroRate(d, dc_plot, comp).rate())

fig, ax = plt.subplots(figsize=(8.2, 4.8))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

ax.plot(
    times_used,
    100.0 * np.array(zero_actual),
    color=color_actual,
    linewidth=2.6,
    linestyle="-",
    label="Actual"
)

ax.plot(
    times_used,
    100.0 * np.array(zero_extended),
    color=color_extended,
    linewidth=2.7,
    linestyle="-",
    label="Extended"
)

ax.plot(
    times_used,
    100.0 * np.array(zero_boot),
    color=color_boot,
    linewidth=2.0,
    linestyle="--",
    label="Bootstrapped"
)

# --- Anchor text box

b0_pct = 100.0 * b0
b0b1_pct = 100.0 * (b0 + b1)

anchor_text = (
    f"$b_0$ start: {b0_pct:.2f}%\n"
    f"$b_0 + b_1$ start: {b0b1_pct:.2f}%"
)

ax.text(
    0.98, 0.98,
    anchor_text,
    transform=ax.transAxes,
    fontsize=9,
    verticalalignment="top",
    horizontalalignment="right",
    bbox=dict(
        boxstyle="square",
        facecolor="white",
        edgecolor="black",
        linewidth=0.8
    )
)

ax.set_xlabel("Years")
ax.set_ylabel("Yield (%)")
ax.set_title(f"{eval_date.date()}", pad=12)

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
    width=1.0,
    pad=6
)

ax.set_xlim(min(times_used), max(times_used))
ax.margins(x=0)

ax.legend(
    loc="upper left",
    frameon=True,
    fancybox=False,
    edgecolor="black",
    facecolor="white",
    framealpha=1.0,
    borderpad=0.5,
    handlelength=3.0,
    handletextpad=0.5
)

plt.tight_layout()
plt.show()

# ---- PRICE RESIDUALS ON ACTUAL BONDS

price_resid_actualfit = quoted_price_actual - model_prices_actualfit
price_resid_extfit = quoted_price_actual - model_prices_extendedfit

fig, ax = plt.subplots(figsize=(8.2, 4.8))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

ax.scatter(
    maturities_plot,
    price_resid_actualfit,
    s=68,
    color=color_actual,
    marker="^",
    edgecolor="black",
    linewidth=0.4,
    label="Actual fit"
)

ax.scatter(
    maturities_plot,
    price_resid_extfit,
    s=62,
    color=color_extended,
    marker="o",
    edgecolor="black",
    linewidth=0.4,
    label="Extended fit"
)

ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)

ax.set_xlabel("Maturity")
ax.set_ylabel("Observed price - model price")
ax.set_title(f"Price residuals ({eval_date.date()})", pad=12)

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
    width=1.0,
    pad=6
)

ax.legend(
    loc="best",
    frameon=True,
    fancybox=False,
    edgecolor="black",
    facecolor="white",
    framealpha=1.0,
    borderpad=0.5
)

plt.tight_layout()
plt.show()

# ---- OBSERVED AND MODEL-IMPLIED YIELDS

fig, ax = plt.subplots(figsize=(8.2, 4.8))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

ax.scatter(
    maturities_plot,
    100.0 * quoted_yield_actual,
    s=70,
    color=color_observed,
    marker="D",
    edgecolor="black",
    linewidth=0.4,
    label="Observed"
)

ax.scatter(
    maturities_plot,
    100.0 * model_yields_actualfit,
    s=70,
    color=color_actual,
    marker="^",
    edgecolor="black",
    linewidth=0.4,
    label="Actual"
)

ax.scatter(
    maturities_plot,
    100.0 * model_yields_extendedfit,
    s=62,
    color=color_extended,
    marker="o",
    edgecolor="black",
    linewidth=0.4,
    label="Extended"
)

ax.set_xlabel("Maturity")
ax.set_ylabel("Yield (%)")
ax.set_title(f"Observed and model-implied yields ({eval_date.date()})", pad=12)

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
    width=1.0,
    pad=6
)

ax.legend(
    loc="best",
    frameon=True,
    fancybox=False,
    edgecolor="black",
    facecolor="white",
    framealpha=1.0,
    borderpad=0.5
)

plt.tight_layout()
plt.show()

    # endregion

# endregion

# region PANEL

    # region FUNCTION: ESTIMATE NSS WITH MULTIPLE STARTS

def run_estimation_for_date_multistart(
    df_panel: pd.DataFrame,
    df_deposit: pd.DataFrame,
    eval_date: pd.Timestamp
):

    cross_section = df_panel.loc[df_panel["date"] == eval_date].copy()
    cross_section = cross_section.sort_values("maturity_date").reset_index(drop=True)

    if cross_section.empty:
        raise ValueError(f"No instruments found for evaluation date {eval_date}")

    # 1) SETTINGS
    ql_eval_date = ql.Date(eval_date.day, eval_date.month, eval_date.year)
    ql.Settings.instance().evaluationDate = ql_eval_date

    day_count = ql.Thirty360(ql.Thirty360.European)
    tenor = ql.Period(ql.Annual)
    date_gen_rule = ql.DateGeneration.Backward
    calendar = ql.Sweden()
    business_convention = ql.ModifiedFollowing
    end_of_month = False

    settlement_days = 2
    face = 100.0
    redemption = 100.0

    # 2) BUILD HELPERS FOR ACTUAL BONDS
    helpers_actual = []
    bonds_actual = []
    quoted_price = []
    quoted_yield = []

    for _, row in cross_section.iterrows():
        issue = row["issue_date"]
        maturity = row["maturity_date"]

        ql_issue = ql.Date(issue.day, issue.month, issue.year)
        ql_maturity = ql.Date(maturity.day, maturity.month, maturity.year)

        schedule = ql.Schedule(
            ql_issue,
            ql_maturity,
            tenor,
            calendar,
            business_convention,
            business_convention,
            date_gen_rule,
            end_of_month
        )

        coupon_rate = float(row["coupon"])
        clean_price = float(row["price"])
        quoted_ytm = float(row["yield"])

        bond = ql.FixedRateBond(
            settlement_days,
            face,
            schedule,
            [coupon_rate],
            day_count,
            business_convention,
            redemption,
            ql_issue
        )

        helper = ql.FixedRateBondHelper(
            ql.QuoteHandle(ql.SimpleQuote(clean_price)),
            settlement_days,
            face,
            schedule,
            [coupon_rate],
            day_count,
            business_convention,
            redemption,
            ql_issue
        )

        bonds_actual.append(bond)
        helpers_actual.append(helper)
        quoted_price.append(clean_price)
        quoted_yield.append(quoted_ytm)

    quoted_price_actual = np.array(quoted_price, dtype=float)
    quoted_yield_actual = np.array(quoted_yield, dtype=float)

    # 3) BOOTSTRAP PRELIMINARY CURVE
    pre_curve = ql.PiecewiseLogCubicDiscount(
        settlement_days,
        calendar,
        helpers_actual,
        day_count
    )
    pre_curve.enableExtrapolation()

    curve_reference_date = calendar.advance(ql_eval_date, settlement_days, ql.Days)
    actual_maturity_dates = sorted({b.maturityDate() for b in bonds_actual})

    # 4) BUILD SYNTHETIC MIDPOINT ZERO-COUPON BONDS
    synthetic_helpers = []

    for d1, d2 in zip(actual_maturity_dates[:-1], actual_maturity_dates[1:]):
        t1 = day_count.yearFraction(curve_reference_date, d1)
        t2 = day_count.yearFraction(curve_reference_date, d2)

        if t1 <= 0.0 or t2 <= 0.0 or t2 <= t1:
            continue

        t_mid = 0.5 * (t1 + t2)
        months_mid = max(1, int(round(t_mid * 12)))
        mid_date = calendar.advance(curve_reference_date, ql.Period(months_mid, ql.Months))

        if not (d1 < mid_date < d2):
            continue

        synth_price = face * pre_curve.discount(mid_date)

        schedule_mid = ql.Schedule(
            curve_reference_date,
            mid_date,
            ql.Period(ql.Once),
            calendar,
            business_convention,
            business_convention,
            ql.DateGeneration.Forward,
            False
        )

        synth_helper = ql.FixedRateBondHelper(
            ql.QuoteHandle(ql.SimpleQuote(float(synth_price))),
            settlement_days,
            face,
            schedule_mid,
            [0.0],
            day_count,
            business_convention,
            redemption,
            curve_reference_date
        )

        synthetic_helpers.append(synth_helper)

    helpers_extended = helpers_actual + synthetic_helpers

    # 5) STARTING VALUES FROM SHORT AND LONG END ANCHORS
    deposit_row = df_deposit.loc[df_deposit["date"] == eval_date]
    if deposit_row.empty:
        raise ValueError(f"No deposit rate found for {eval_date}")

    deposit_rate = float(deposit_row["deposit"].iloc[0])
    deposit_rate_cc = np.log(1.0 + deposit_rate)

    max_maturity = max(b.maturityDate() for b in bonds_actual)
    long_rate_cc = pre_curve.zeroRate(max_maturity, day_count, ql.Continuous).rate()

    b0 = long_rate_cc
    b1 = deposit_rate_cc - b0

    # 6) BOUNDS
    b0_min, b0_max = -0.03, 0.06
    b1_min, b1_max = -0.10, 0.10
    b2_min, b2_max = -0.05, 0.05
    b3_min, b3_max = -0.05, 0.05

    tau1_min, tau1_max = 1.0, 5.0
    tau2_min, tau2_max = 4.0, 20.0

    kappa1_min, kappa1_max = 1.0 / tau1_max, 1.0 / tau1_min
    kappa2_min, kappa2_max = 1.0 / tau2_max, 1.0 / tau2_min

    lower = ql.Array(6)
    upper = ql.Array(6)

    lower[0], upper[0] = b0_min, b0_max
    lower[1], upper[1] = b1_min, b1_max
    lower[2], upper[2] = b2_min, b2_max
    lower[3], upper[3] = b3_min, b3_max
    lower[4], upper[4] = kappa1_min, kappa1_max
    lower[5], upper[5] = kappa2_min, kappa2_max

    constraint = ql.NonhomogeneousBoundaryConstraint(lower, upper)

    # 7) PENALTY AND FIT SETUP
    l2 = ql.Array(6)
    l2[0] = 0.05
    l2[1] = 0.01
    l2[2] = 0.2
    l2[3] = 0.2
    l2[4] = 0.05
    l2[5] = 0.02

    accuracy = 1e-10
    max_evaluations = 15000
    simplex_lambda = 0.005

    fitting = ql.SvenssonFitting(
        ql.Array(),
        ql.Simplex(simplex_lambda),
        l2,
        0.0,
        50.0,
        constraint
    )

    # 8) MULTISTART GRID
    b2_grid = [-0.02, -0.01, 0.0, 0.01, 0.02]
    b3_grid = [-0.02, -0.01, 0.0, 0.01, 0.02]
    tau1_grid = [1.5, 2.0, 3.0, 4.5]
    tau2_grid = [6.0, 8.0, 10.0, 12.0]

    # 9) PRICE ACTUAL BONDS FROM ESTIMATED CURVE
    def price_actual_bonds(curve):
        curve_handle = ql.YieldTermStructureHandle(curve)
        engine = ql.DiscountingBondEngine(curve_handle)

        model_prices = []
        model_yields = []

        for bond in bonds_actual:
            bond.setPricingEngine(engine)
            model_prices.append(float(bond.cleanPrice()))
            model_yields.append(float(bond.bondYield(day_count, ql.Compounded, ql.Annual)))

        return np.array(model_prices), np.array(model_yields)

    # 10) RUN MULTISTART FOR ONE HELPER SET
    def fit_multistart(helpers):
        best_objective = float("inf")
        best_curve = None
        best_params = None
        best_start = None

        for b2_start in b2_grid:
            for b3_start in b3_grid:
                for tau1 in tau1_grid:
                    for tau2 in tau2_grid:
                        if tau2 <= tau1:
                            continue

                        guess = ql.Array(6)
                        guess[0] = b0
                        guess[1] = b1
                        guess[2] = b2_start
                        guess[3] = b3_start
                        guess[4] = 1.0 / tau1
                        guess[5] = 1.0 / tau2

                        try:
                            curve_try = ql.FittedBondDiscountCurve(
                                settlement_days,
                                calendar,
                                helpers,
                                day_count,
                                fitting,
                                accuracy,
                                max_evaluations,
                                guess
                            )

                            fit_result = curve_try.fitResults()
                            objective_value = fit_result.minimumCostValue()

                            if objective_value < best_objective:
                                best_objective = objective_value
                                best_curve = curve_try
                                best_params = list(fit_result.solution())
                                best_start = (b2_start, b3_start, tau1, tau2)

                        except RuntimeError:
                            continue

        return best_curve, best_params, best_objective, best_start

    # 11) MULTISTART FITS
    best_curve_actual, best_params_actual, best_objective_actual, best_start_actual = fit_multistart(helpers_actual)
    if best_curve_actual is None:
        raise RuntimeError(f"Multistart failed for ACTUAL on {eval_date}")

    best_curve_ext, best_params_ext, best_objective_ext, best_start_ext = fit_multistart(helpers_extended)
    if best_curve_ext is None:
        raise RuntimeError(f"Multistart failed for EXTENDED on {eval_date}")

    # 12) FIT DIAGNOSTICS ON ACTUAL BONDS
    model_prices_actualfit, model_yields_actualfit = price_actual_bonds(best_curve_actual)
    model_prices_extendedfit, model_yields_extendedfit = price_actual_bonds(best_curve_ext)

    rmse_price_actualfit = float(np.sqrt(np.mean((quoted_price_actual - model_prices_actualfit) ** 2)))
    rmse_price_extfit = float(np.sqrt(np.mean((quoted_price_actual - model_prices_extendedfit) ** 2)))

    rmse_yield_actualfit_bp = float(np.sqrt(np.mean((quoted_yield_actual - model_yields_actualfit) ** 2)) * 1e4)
    rmse_yield_extfit_bp = float(np.sqrt(np.mean((quoted_yield_actual - model_yields_extendedfit) ** 2)) * 1e4)

    return {
        "date": eval_date,
        "b0_actual": best_params_actual[0],
        "b1_actual": best_params_actual[1],
        "b2_actual": best_params_actual[2],
        "b3_actual": best_params_actual[3],
        "k1_actual": best_params_actual[4],
        "k2_actual": best_params_actual[5],
        "b0_ext": best_params_ext[0],
        "b1_ext": best_params_ext[1],
        "b2_ext": best_params_ext[2],
        "b3_ext": best_params_ext[3],
        "k1_ext": best_params_ext[4],
        "k2_ext": best_params_ext[5],
        "rmse_price_actual": rmse_price_actualfit,
        "rmse_price_ext": rmse_price_extfit,
        "rmse_yield_bp_actual": rmse_yield_actualfit_bp,
        "rmse_yield_bp_ext": rmse_yield_extfit_bp,
        "objective_actual": best_objective_actual,
        "objective_ext": best_objective_ext,
        "n_bonds": len(helpers_actual),
        "n_synth": len(synthetic_helpers),
        "start_b2_actual": best_start_actual[0],
        "start_b3_actual": best_start_actual[1],
        "start_tau1_actual": best_start_actual[2],
        "start_tau2_actual": best_start_actual[3],
        "start_b2_ext": best_start_ext[0],
        "start_b3_ext": best_start_ext[1],
        "start_tau1_ext": best_start_ext[2],
        "start_tau2_ext": best_start_ext[3],
    }

    # endregion

    # region RUN 

# ---- BUILD MONTHLY INDEX

month_to_date = (
    df_SGBs.assign(month=df_SGBs["date"].dt.to_period("M"))
    .groupby("month")["date"]
    .max()
    .sort_index()
)

# ---- RUN ESTIMATION MONTH BY MONTH

rows = []
failed = []

for month, d in month_to_date.items():
    eval_date = pd.Timestamp(d)

    try:
        out = run_estimation_for_date_multistart(df_SGBs, df_deposit, eval_date)
        out["month"] = str(month)
        rows.append(out)

    except Exception as e:
        failed.append({
            "month": str(month),
            "date": eval_date,
            "error": str(e),
        })

results = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
failed_df = pd.DataFrame(failed)

print("Successful months:", len(results))
print("Failed months:", len(failed_df))

# ---- COMPUTE TAUS

results["tau1_actual"] = 1.0 / results["k1_actual"]
results["tau2_actual"] = 1.0 / results["k2_actual"]
results["tau1_ext"] = 1.0 / results["k1_ext"]
results["tau2_ext"] = 1.0 / results["k2_ext"]

# ---- ARRANGE AND EXPORT TO EXCEL

col_order = [
    "date", "month",

    # actual-only params
    "b0_actual", "b1_actual", "b2_actual", "b3_actual",
    "k1_actual", "tau1_actual",
    "k2_actual", "tau2_actual",

    # extended params
    "b0_ext", "b1_ext", "b2_ext", "b3_ext",
    "k1_ext", "tau1_ext",
    "k2_ext", "tau2_ext",

    # fit diagnostics
    "rmse_price_actual", "rmse_price_ext",
    "rmse_yield_bp_actual", "rmse_yield_bp_ext",
    "objective_actual", "objective_ext",

    # counts
    "n_bonds", "n_synth",

    # winning starting values
    "start_b2_actual", "start_b3_actual", "start_tau1_actual", "start_tau2_actual",
    "start_b2_ext", "start_b3_ext", "start_tau1_ext", "start_tau2_ext",
]

results = results.loc[:, col_order]

with pd.ExcelWriter("nominal_panel_results.xlsx", engine="openpyxl") as writer:
    results.to_excel(writer, sheet_name="results", index=False)

    if not failed_df.empty:
        failed_df.to_excel(writer, sheet_name="failed", index=False)

print("Exported nominal_panel_results.xlsx")

# ---- PANEL SUMMARY STATISTICS

panel_cols = [
    # parameters (actual-only)
    "b0_actual", "b1_actual", "b2_actual", "b3_actual",
    "k1_actual", "tau1_actual",
    "k2_actual", "tau2_actual",

    # parameters (extended)
    "b0_ext", "b1_ext", "b2_ext", "b3_ext",
    "k1_ext", "tau1_ext",
    "k2_ext", "tau2_ext",

    # RMSE diagnostics
    "rmse_price_actual", "rmse_price_ext",
    "rmse_yield_bp_actual", "rmse_yield_bp_ext",

    # objective values
    "objective_actual", "objective_ext",
]

print("Panel summary statistics")

for c in panel_cols:
    avg_val = results[c].mean()
    min_val = results[c].min()
    max_val = results[c].max()

    print(f"{c:20s}  avg={avg_val: .6f}  min={min_val: .6f}  max={max_val: .6f}")

    # endregion

    # region PLOT PANEL RMSEs

results_plot = pd.read_excel("nominal_panel_results.xlsx")
results_plot["date"] = pd.to_datetime(results_plot["date"])

dates = results_plot["date"]

# Colors
color_ext = "#2CA02C"         
color_actual = "#3A3A3A"

# Plot settings consistent with cross-sectional figures
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

fig, ax = plt.subplots(figsize=(8.2, 4.8))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

ax.plot(
    dates,
    results_plot["rmse_yield_bp_ext"],
    color=color_ext,
    linewidth=2.0,
    linestyle="-",
    label="Extended"
)

ax.plot(
    dates,
    results_plot["rmse_yield_bp_actual"],
    color=color_actual,
    linewidth=2.0,
    linestyle="-.",
    dashes=(6, 3),
    label="Actual"
)

ax.set_xlabel("Date")
ax.set_ylabel("Yield RMSE (bp)")
ax.set_title("", pad=12)

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
    width=1.0,
    pad=6
)

ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.45)

ax.legend(
    loc="upper right",
    frameon=True,
    fancybox=False,
    edgecolor="black",
    facecolor="white",
    framealpha=1.0,
    borderpad=0.5
)

plt.tight_layout()
plt.show()

    # endregion

# endregion

# region EXPORT ZERO-COUPON YIELDS

# Export zero-coupon yields from EXTENDED multistart NSS panel results

maturities_months = list(range(1, 121))   # 1m to 120m

# ---- NSS ZERO-YIELD FUNCTION

def nss_zero_yield(t, b0, b1, b2, b3, k1, k2):
    x1 = k1 * t
    x2 = k2 * t

    term1 = np.where(np.abs(x1) < 1e-12, 1.0, (1.0 - np.exp(-x1)) / x1)
    term2 = term1 - np.exp(-x1)
    term3 = np.where(np.abs(x2) < 1e-12, 0.0, (1.0 - np.exp(-x2)) / x2 - np.exp(-x2))

    return b0 + b1 * term1 + b2 * term2 + b3 * term3

# ---- EXTRACT EXTENDED PARAMETERS

b0 = results["b0_ext"].to_numpy(dtype=float)
b1 = results["b1_ext"].to_numpy(dtype=float)
b2 = results["b2_ext"].to_numpy(dtype=float)
b3 = results["b3_ext"].to_numpy(dtype=float)
k1 = results["k1_ext"].to_numpy(dtype=float)
k2 = results["k2_ext"].to_numpy(dtype=float)

# ---- BUILD ZERO-YIELD PANEL

zero_yields = pd.DataFrame({
    "date": pd.to_datetime(results["date"])
})

for m in maturities_months:
    t = m / 12.0
    zero_yields[f"y_{m}m"] = nss_zero_yield(t, b0, b1, b2, b3, k1, k2)

# ---- PARAMETER AND DIAGNOSTIC SHEET

fit_params = results.loc[:, [
    "date",
    "b0_ext", "b1_ext", "b2_ext", "b3_ext",
    "k1_ext", "tau1_ext",
    "k2_ext", "tau2_ext",
    "rmse_price_ext", "rmse_yield_bp_ext",
    "objective_ext",
    "n_bonds", "n_synth",
    "start_b2_ext", "start_b3_ext",
    "start_tau1_ext", "start_tau2_ext",
]].copy()

# ---- EXPORT TO EXCEL

with pd.ExcelWriter("zero_yields_SGB.xlsx", engine="openpyxl") as writer:
    zero_yields.to_excel(writer, sheet_name="zero_yields", index=False)
    fit_params.to_excel(writer, sheet_name="fit_params", index=False)

print("Exported zero_yields_SGB.xlsx")

# endregion

# region BONDS OUTSTANDING PLOT 

plot_cols = [
    "Date",
    "Issue date",
    "Maturity date",
    "ISIN",
    "Serie",
]

df_outstanding = raw_data_nominal.loc[:, plot_cols].copy()

df_outstanding["Date"] = pd.to_datetime(df_outstanding["Date"], errors="coerce")
df_outstanding["Issue date"] = pd.to_datetime(df_outstanding["Issue date"], errors="coerce")
df_outstanding["Maturity date"] = pd.to_datetime(df_outstanding["Maturity date"], errors="coerce")

df_outstanding = df_outstanding.dropna(subset=["Date", "Issue date", "Maturity date", "Serie"]).copy()

# keep 2000 onward
df_outstanding = df_outstanding[df_outstanding["Date"] >= pd.Timestamp("2000-01-01")].copy()

# remaining maturity in years
df_outstanding["ttm_years"] = (
    (df_outstanding["Maturity date"] - df_outstanding["Date"]).dt.days / 365.25
)

# keep only bonds that are still outstanding
df_outstanding = df_outstanding[df_outstanding["ttm_years"] >= 0.0].copy()

# sort for clean line plotting
df_outstanding = df_outstanding.sort_values(["Serie", "Date"]).reset_index(drop=True)

# monthly count of outstanding bonds
count_df = (
    df_outstanding.groupby("Date")["Serie"]
    .nunique()
    .rename("n_bonds")
    .reset_index()
)

# ---- PLOT SETTINGS

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

# ---- MONTHLY COUNT + OUTSTANDING BONDS OVER TIME

fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(8.2, 5.4),
    sharex=True,
    gridspec_kw={"height_ratios": [3.4, 1.0], "hspace": 0.05}
)

fig.patch.set_facecolor("white")
ax1.set_facecolor("white")
ax2.set_facecolor("white")

# upper panel: outstanding bonds
for serie, g in df_outstanding.groupby("Serie"):
    ax1.plot(g["Date"], g["ttm_years"], color="black", linewidth=1.1, alpha=0.95)

ax1.set_ylabel("Years to maturity")
ax1.set_ylim(0, max(35, np.ceil(df_outstanding["ttm_years"].max())))
ax1.margins(x=0)
ax1.tick_params(axis="x", which="both", labelbottom=False)

# lower panel: monthly number of bonds
ax2.plot(count_df["Date"], count_df["n_bonds"], color="black", linewidth=1.4)

ax2.set_ylabel("Count")
ax2.set_xlabel("Date")
ax2.set_ylim(2, max(14, count_df["n_bonds"].max() + 1))
ax2.set_yticks([3, 6, 9, 12])
ax2.margins(x=0)

# frame and ticks
for ax in (ax1, ax2):
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
        width=1.0,
        pad=6
    )

    ax.grid(False)

plt.tight_layout()
plt.show()

# endregion

# region FILL MISSING SSVX_6m IN statsobligationer_data.xlsx

sgb_date = pd.Timestamp("2018-09-28")       # date used in zero_yields_SGB.xlsx
target_date = pd.Timestamp("2018-09-30")    # date to fill in 'korta räntor riksbanken'

# ---- 1) GET ESTIMATED 6M NOMINAL YIELD FROM ZERO-YIELD PANEL

zero_panel = pd.read_excel("zero_yields_SGB.xlsx", sheet_name="zero_yields")
zero_panel["date"] = pd.to_datetime(zero_panel["date"], errors="coerce")

match = zero_panel.loc[zero_panel["date"] == sgb_date, "y_6m"]

if match.empty or pd.isna(match.iloc[0]):
    raise ValueError(f"No estimated y_6m found in zero_yields_SGB.xlsx for {sgb_date.date()}")

estimated_6m_pct = float(match.iloc[0]) * 100.0   # decimal -> percent

print(f"Estimated SSVX_6m from nominal curve on {sgb_date.date()}: {estimated_6m_pct:.6f}")

# ---- 2) LOAD ALL SHEETS FROM statsobligationer_data.xlsx

xlsx = pd.ExcelFile("statsobligationer_data.xlsx")
sheets = {
    sheet_name: pd.read_excel("statsobligationer_data.xlsx", sheet_name=sheet_name)
    for sheet_name in xlsx.sheet_names
}

# ---- 3) UPDATE ONLY THE 2018-09-30 ROW IN 'korta räntor riksbanken'

short_df = sheets["korta räntor riksbanken"].copy()
short_df["date"] = pd.to_datetime(short_df["date"], errors="coerce")

row_mask = short_df["date"] == target_date

if not row_mask.any():
    raise ValueError(f"No row found in 'korta räntor riksbanken' for {target_date.date()}")

short_df.loc[row_mask, "SSVX_6m"] = estimated_6m_pct
sheets["korta räntor riksbanken"] = short_df

print(
    "Filled row:",
    short_df.loc[row_mask, ["date", "SSVX_6m"]]
)

# ---- 4) OVERWRITE THE WORKBOOK WITH THE UPDATED VALUE, PRESERVING ALL SHEETS/DATA

with pd.ExcelWriter("statsobligationer_data.xlsx", engine="openpyxl", mode="w") as writer:
    for sheet_name, df_sheet in sheets.items():
        df_sheet.to_excel(writer, sheet_name=sheet_name, index=False)

print("Updated statsobligationer_data.xlsx")

# endregion

# region BID-ASK YIELD SPREADS

def compute_bid_ask_spread_summary_nominal(df_panel: pd.DataFrame) -> pd.DataFrame:
    spread_df = df_panel.copy()

    # Keep actual nominal bond observations only:
    # SSVX synthetic short instruments have missing bid/ask yields.
    spread_df = spread_df.dropna(subset=["bid_yield", "ask_yield"]).copy()

    spread_df["bid_ask_spread_bp"] = (
        (spread_df["bid_yield"] - spread_df["ask_yield"]).abs() * 1e4
    )

    summary = (
        spread_df.groupby("date", as_index=False)["bid_ask_spread_bp"]
        .agg(
            median_bid_ask_yield_spread_bp="median",
            mean_bid_ask_yield_spread_bp="mean",
            n_bonds="count",
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    return summary


df_nominal_bid_ask_summary = compute_bid_ask_spread_summary_nominal(df_SGBs)

print(df_nominal_bid_ask_summary.tail())
df_nominal_bid_ask_summary.to_excel("nominal_bid_ask.xlsx", index=False)

# endregion