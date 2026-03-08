import pandas as pd
import QuantLib as ql
import numpy as np
import matplotlib.pyplot as plt

# IMPORTS 
raw_data = pd.read_excel("statsobligationer_data.xlsx", sheet_name = "SGB long")
short_rates = pd.read_excel("statsobligationer_data.xlsx", sheet_name = "korta räntor riksbanken")

START = pd.Timestamp("2000-01-01")
END = pd.Timestamp("2026-01-31")

# region PREPARE BOND DATA

# ---- PREPARE SGB DATA

keep_cols = [
    "Date",
    "PX_LAST",
    "YLD_YTM_MID",
    "Kupong",
    "Issue date",
    "Maturity date",
    "ISIN",
    "Serie",
]

df_SGBs = raw_data.loc[:, keep_cols].copy()

df_SGBs["Date"] = pd.to_datetime(df_SGBs["Date"], errors="coerce")
df_SGBs["Issue date"] = pd.to_datetime(df_SGBs["Issue date"], errors="coerce")
df_SGBs["Maturity date"] = pd.to_datetime(df_SGBs["Maturity date"], errors="coerce")

df_SGBs = df_SGBs.rename(columns={
    "Date": "date",
    "PX_LAST": "price",
    "YLD_YTM_MID": "yield",
    "Kupong": "coupon",
    "Issue date": "issue_date",
    "Maturity date": "maturity_date",
    "ISIN": "isin",
    "Serie": "serie"
})

df_SGBs = df_SGBs.loc[(df_SGBs["date"] >= START) & (df_SGBs["date"] <= END)]

for col in ["price", "yield", "coupon"]:
    df_SGBs[col] = pd.to_numeric(df_SGBs[col], errors="coerce")

df_SGBs["yield"] = df_SGBs["yield"] / 100.0
df_SGBs["coupon"] = df_SGBs["coupon"] / 100.0

df_SGBs = df_SGBs.sort_values(["date", "maturity_date"]).reset_index(drop=True)

# filter bonds with less than 90 days and more than 20 years to maturity
time_to_maturity = (df_SGBs["maturity_date"] - df_SGBs["date"]).dt.days
df_SGBs = df_SGBs[(time_to_maturity >= 90) & (time_to_maturity <= 20 * 365)]
df_SGBs = df_SGBs.reset_index(drop=True)

# ----

# ---- PREPARE SHORT RATE DATA

short_cols = [
    "SSVX_1m",
    "SSVX_3m",
    "SSVX_6m",
]
sr_wide = short_rates.loc[:, ["date"] + short_cols].copy()

df_SSVX = sr_wide.melt(             #wide to long format
    id_vars = ["date"],
    value_vars=short_cols,
    var_name="serie",
    value_name="yield"
).dropna(subset=["yield"]).reset_index(drop=True)

tenor_months = {"SSVX_1m": 1, "SSVX_3m": 3, "SSVX_6m": 6}
df_SSVX["months"] = df_SSVX["serie"].map(tenor_months).astype("Int64")

df_SSVX["coupon"] = 0.0
df_SSVX["issue_date"] = df_SSVX["date"]
df_SSVX["maturity_date"] = df_SSVX["issue_date"] + pd.to_timedelta(df_SSVX["months"] * 30, unit="D")
df_SSVX["isin"] = ""

# compute prices based on 30/360 convention
df_SSVX["yield"] = pd.to_numeric(df_SSVX["yield"], errors="coerce") / 100.0
days = (df_SSVX["maturity_date"] - df_SSVX["issue_date"]).dt.days
t = days / 360.0 
df_SSVX["price"] = 100.0 / (1.0 + df_SSVX["yield"]) ** t

df_SSVX = df_SSVX.sort_values(["date", "maturity_date"]).reset_index(drop=True) 
df_SSVX = df_SSVX.loc[:, ["date", "price", "yield", "coupon", "issue_date", "maturity_date", "isin", "serie"]].copy()

df_SSVX = df_SSVX.loc[(df_SSVX["date"] >= START) & (df_SSVX["date"] <= END)]

# ----

# ---- COMBINE SSVX & SGBs

# SSVX is EOM and SGB is last trading day > align date column in SSVX
df_SGBs["month"] = df_SGBs["date"].dt.to_period("M")
df_SSVX["month"] = df_SSVX["date"].dt.to_period("M")

month_trading_date = df_SGBs.groupby("month")["date"].max()
df_SSVX["date"] = df_SSVX["month"].map(month_trading_date)
df_SSVX["issue_date"] = df_SSVX["date"]

df_SGBs = df_SGBs.drop(columns=["month"])
df_SSVX = df_SSVX.drop(columns=["month"])

# order and dtypes
required_cols = ["date", "price", "yield", "coupon", "issue_date", "maturity_date", "isin", "serie"]
df_SGBs = df_SGBs.loc[:, required_cols].copy()
df_SSVX = df_SSVX.loc[:, required_cols].copy()

for c in ["date", "issue_date", "maturity_date"]:
    df_SGBs[c] = pd.to_datetime(df_SGBs[c], errors="coerce")
    df_SSVX[c] = pd.to_datetime(df_SSVX[c], errors="coerce")

for c in ["price", "yield", "coupon"]:
    df_SGBs[c] = pd.to_numeric(df_SGBs[c], errors="coerce")
    df_SSVX[c] = pd.to_numeric(df_SSVX[c], errors="coerce")

df_SGBs["isin"] = df_SGBs["isin"].astype(str)
df_SSVX["isin"] = df_SSVX["isin"].astype(str)
df_SGBs["serie"] = df_SGBs["serie"].astype(str)
df_SSVX["serie"] = df_SSVX["serie"].astype(str)

# concatenate rows & sort
df_SGBs = pd.concat([df_SGBs, df_SSVX], ignore_index=True).copy()
df_SGBs = df_SGBs.sort_values(["date", "maturity_date"]).reset_index(drop=True)

# ----

# ---- PREPARE DEPOSIT DATA

df_deposit = short_rates.loc[:, ["date", "deposit"]].copy()
df_deposit["date"] = pd.to_datetime(df_deposit["date"], errors="coerce") 
df_deposit["deposit"] = pd.to_numeric(df_deposit["deposit"], errors="coerce") / 100.0
df_deposit["month"] = df_deposit["date"].dt.to_period("M")
df_deposit["date"] = df_deposit["month"].map(month_trading_date) # map to SDG last trading day 
df_deposit = df_deposit.drop(columns=["month"]).dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
# ----

# endregion

# region CROSS SECTION

    # region SETUP (SETTINGS & HELPERS)

# ---- SETTINGS: 

eval_date = pd.Timestamp("2026-01-30")
cross_section = df_SGBs.loc[df_SGBs["date"] == eval_date].copy()

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
# ----

# ---- BUILD FixedRateBondHelper PER ACTUAL BOND WHICH PAIRS PRICE AND CASHFLOWS

helpers = []
bonds = []
quoted_price = []
quoted_yield = []
maturities = []

for _, row in cross_section.iterrows():
    issue = row["issue_date"]
    maturity = row["maturity_date"]
    ql_issue = ql.Date(issue.day, issue.month, issue.year)
    ql_maturity = ql.Date(maturity.day, maturity.month, maturity.year)

    schedule = ql.Schedule(     #payment schedule
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

    helpers.append(
        ql.FixedRateBondHelper(
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
    )

    # store inputs for later plots 
    bonds.append(bond)
    quoted_price.append(clean_price)
    quoted_yield.append(row["yield"])
    maturities.append(row["maturity_date"])

print("Helpers build:", len(helpers))

# ----

# ---- CREATE SYNTHETIC ZERO COUPON BONDS (midpoints) FROM A PIECEWISE CUBIC SPLINE

# keep a copy of the observed sets (used later for diagnostics)
helpers_actual = list(helpers)
bonds_actual = list(bonds)
quoted_price_actual = np.array(quoted_price, dtype=float)
quoted_yield_actual = np.array(quoted_yield, dtype=float)

# 1) build a preliminary bootstrapped curve using cubic spline 
pre_curve = ql.PiecewiseLogCubicDiscount(settlement_days, calendar, helpers_actual, day_count)
pre_curve.enableExtrapolation()
curve_reference_date = calendar.advance(ql_eval_date, settlement_days, ql.Days)

# 2) create one synthetic maturity between each adjacent pair of actual bond maturities
ql_actual_maturity = []
for b in bonds_actual:
    ql_actual_maturity.append(b.maturityDate())

ql_actual_maturity = sorted(ql_actual_maturity)

synthetic_helpers = []
synthetic_bonds = []

for d1, d2 in zip(ql_actual_maturity[:-1], ql_actual_maturity[1:]):
    mid_serial = (d1.serialNumber() + d2.serialNumber()) // 2
    mid_date = ql.Date(mid_serial)

    # get discount factor for preliminary curve and compute synthetic clean price
    discount_factor_mid = pre_curve.discount(mid_date)
    synth_price = face * discount_factor_mid

    # build FixedRateBond for the synthetic zero coupon bond with maturity at mid_date
    schedule_mid = ql.Schedule([curve_reference_date, mid_date])
    synth_bond = ql.FixedRateBond(
        settlement_days,
        face,
        schedule_mid,
        [0.0],
        day_count,
        business_convention, 
        redemption, 
        curve_reference_date
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

    synthetic_bonds.append(synth_bond)
    synthetic_helpers.append(synth_helper)
    
print("Synthetic helpers build:", len(synthetic_helpers))

# 3) extend intstrument set (helpers + bonds):
helpers_extended = helpers_actual + synthetic_helpers
bonds_extended = bonds_actual + synthetic_bonds 

# ----

    # endregion

    # region FIT NSS CURVES

        # region PREPARE FITS 

# ---- INITIAL GUESS FROM SHORT (DEPOSIT) + LONG (BOOTSTRAPPED ANCHORS) + BOUNDS

# 1) Short rate: Riksbank ON deposit rate
deposit_row = df_deposit.loc[df_deposit["date"] == eval_date]
deposit_rate = float(deposit_row["deposit"].iloc[0])
deposit_rate_cc = np.log(1.0 + deposit_rate)

# 2) Long rate: zero rate from preliminary bootstrapped curve at the max observed maturity
max_maturity = max(b.maturityDate() for b in bonds_actual)
long_rate_cc = pre_curve.zeroRate(max_maturity, day_count, ql.Continuous).rate()

# 3) Build initial guesses: 
b0 = long_rate_cc
b1 = deposit_rate_cc - b0
b2 = 0.0
b3 = 0.0

tau1 = 2.0
tau2 = 9.0
kappa1 = 1.0 / tau1
kappa2 = 1.0 / tau2

guess = ql.Array(6)
guess[0] = b0
guess[1] = b1
guess[2] = b2
guess[3] = b3
guess[4] = kappa1
guess[5] = kappa2

print(f"Initial guess from anchors: short={deposit_rate_cc:.6f}, long={long_rate_cc:.6f}, "
      f"b0={b0:.6f}, b1={b1:.6f}, kappa1={kappa1:.6f}, kappa2={kappa2:.6f}")

# ----

# ---- BOUNDS

b0_min, b0_max = -0.15, 0.15
b1_min, b1_max = -0.15, 0.15
b2_min, b2_max = -0.5, 0.5
b3_min, b3_max = -0.5, 0.5

tau_min, tau_max = 0.05, 50.0
kappa_min, kappa_max = 1.0/tau_max, 1.0/tau_min

lower = ql.Array(6)
upper = ql.Array(6)

lower[0], upper[0] = b0_min, b0_max
lower[1], upper[1] = b1_min, b1_max
lower[2], upper[2] = b2_min, b2_max
lower[3], upper[3] = b3_min, b3_max
lower[4], upper[4] = kappa_min, kappa_max
lower[5], upper[5] = kappa_min, kappa_max

constraint = ql.NonhomogeneousBoundaryConstraint(lower, upper)

# ----

# ---- PENALTY 

l2 = ql.Array(6)
l2[0] = 1.0     # b0 penalty strength
l2[1] = 1.0     # b1 penalty strength
l2[2] = 0.02     # b2 unpenalized
l2[3] = 0.02     # b3 unpenalized
l2[4] = 0.02     # kappa1 penalty strength
l2[5] = 0.005     # kappa2 penalty strength

# ----

# ---- FIT SETUP 

accuracy = 1e-10
max_evaluations = 15000
simplex_lambda = 0.01
fitting = ql.SvenssonFitting(
    ql.Array(),                     # weights, default inverse duration
    ql.Simplex(simplex_lambda),     # optimizer, simplex (nelder-mead) default
    l2,                     # L2 penalty, ql.Array() is default no penalty (penalizes values different from starting vals)
    0.0,                            # Min cutoff time (maturity)
    50.0,                           # Max cutoff time (maturity)
    constraint
)

# ----

        # endregion

        # region SINGLE START FITS

# ---- FIT 1: NSS DISCOUNT COURVE TO PRICES ON ACTUAL BONDS ONLY

#staring values (c0 = b0, c1 = b1, c2 = b2, c3=b3, kappa = 1/tau1, kappa1 = 1/tau2)

nss_curve_actual = ql.FittedBondDiscountCurve(
    settlement_days,
    calendar,
    helpers,
    day_count,
    fitting,
    accuracy,
    max_evaluations,
    guess
)

params_actual = list(nss_curve_actual.fitResults().solution())

print("ACTUAL-ONLY params:", list(params_actual))
print("ACTUAL-ONLY iterations:", nss_curve_actual.fitResults().numberOfIterations())

# ----

# ---- FIT 2: NSS USING EXTENDED SET 

nss_curve_extended = ql.FittedBondDiscountCurve(
    settlement_days,
    calendar,
    helpers_extended,
    day_count,
    fitting,
    accuracy,
    max_evaluations,
    guess
)

params_extended = list(nss_curve_extended.fitResults().solution())

print("EXTENDED params:", list(params_extended))
print("EXTENDED iterations:", nss_curve_extended.fitResults().numberOfIterations())

# ----

# ---- PRICE ACTUAL BONDS OF ALL THREE CURVES & COMPUTE RMSEs

def price_actual_bonds(curve):
    curve_handle = ql.YieldTermStructureHandle(curve)
    engine = ql.DiscountingBondEngine(curve_handle)

    model_prices = []
    model_yields = []

    for b in bonds_actual:
        b.setPricingEngine(engine)
        mp = b.cleanPrice()
        model_prices.append(float(mp))
        my = b.bondYield(day_count, ql.Compounded, ql.Annual)
        model_yields.append(float(my))
    
    return np.array(model_prices), np.array(model_yields)

model_prices_actualfit, model_yields_actualfit = price_actual_bonds(nss_curve_actual)
model_prices_extendedfit, model_yields_extendedfit = price_actual_bonds(nss_curve_extended)

# RMSE on actual bonds only
rmse_price_actualfit = float(np.sqrt(np.mean((quoted_price_actual - model_prices_actualfit) ** 2)))
rmse_price_extfit = float(np.sqrt(np.mean((quoted_price_actual - model_prices_extendedfit) ** 2)))

rmse_yield_actualfit_bp = float(np.sqrt(np.mean((quoted_yield_actual - model_yields_actualfit) ** 2)) * 1e4)
rmse_yield_extfit_bp = float(np.sqrt(np.mean((quoted_yield_actual - model_yields_extendedfit) ** 2)) * 1e4)

print("RMSE price (actual-only fit):", rmse_price_actualfit)
print("RMSE price (extended fit):   ", rmse_price_extfit)

print("RMSE yield bp (actual-only fit):", rmse_yield_actualfit_bp)
print("RMSE yield bp (extended fit):   ", rmse_yield_extfit_bp)

# ----

        # endregion

        # region MULTISTART FITS 

def yield_rmse_bp_from_curve(curve):
    _, model_yields = price_actual_bonds(curve)
    return float(np.sqrt(np.mean((quoted_yield_actual - model_yields) ** 2)) * 1e4)

tau1_grid = [0.5, 2.0, 4.0, 6.0, 8.0]
tau2_grid = [5.0, 10.0, 15.0, 25.0, 30.0]

# --- MUTLISTART ON ACTUAL BONDS

best_rmse_bp_actual = float("inf")
best_curve_actual = None
best_params_actual = None
best_start_actual = None

for tau1 in tau1_grid:
    for tau2 in tau2_grid:
        if tau2 <= tau1:
            continue

        guess_ms = ql.Array(6)
        guess_ms[0] = b0
        guess_ms[1] = b1
        guess_ms[2] = 0.0
        guess_ms[3] = 0.0
        guess_ms[4] = 1.0 / tau1
        guess_ms[5] = 1.0 / tau2

        try:
            curve_try = ql.FittedBondDiscountCurve(
                settlement_days, calendar, helpers, day_count,
                fitting, accuracy, max_evaluations, guess_ms
            )
            rmse_bp = yield_rmse_bp_from_curve(curve_try)
            if rmse_bp < best_rmse_bp_actual:
                best_rmse_bp_actual = rmse_bp
                best_curve_actual = curve_try
                best_params_actual = list(curve_try.fitResults().solution())
                best_start_actual = (tau1, tau2)
        except RuntimeError:
            continue

print(f"MULTISTART ACTUAL best start: tau1={best_start_actual[0]}, tau2={best_start_actual[1]}, "
      f"yield RMSE={best_rmse_bp_actual:.3f} bp")
print("MULTISTART ACTUAL params:", best_params_actual)

# ----

# --- MUTLISTART ON EXTENDED INSTRUMENTS (helpers_extended)

best_rmse_bp_ext = float("inf")
best_curve_ext = None
best_params_ext = None
best_start_ext = None

for tau1 in tau1_grid:
    for tau2 in tau2_grid:
        if tau2 <= tau1:
            continue

        guess_ms = ql.Array(6)
        guess_ms[0] = b0
        guess_ms[1] = b1
        guess_ms[2] = 0.0
        guess_ms[3] = 0.0
        guess_ms[4] = 1.0 / tau1
        guess_ms[5] = 1.0 / tau2

        try:
            curve_try = ql.FittedBondDiscountCurve(
                settlement_days, calendar, helpers_extended, day_count,
                fitting, accuracy, max_evaluations, guess_ms
            )
            # Selection criterion is STILL RMSE on ACTUAL yields (your request)
            rmse_bp = yield_rmse_bp_from_curve(curve_try)
            if rmse_bp < best_rmse_bp_ext:
                best_rmse_bp_ext = rmse_bp
                best_curve_ext = curve_try
                best_params_ext = list(curve_try.fitResults().solution())
                best_start_ext = (tau1, tau2)
        except RuntimeError:
            continue

print(f"MULTISTART EXTENDED best start: tau1={best_start_ext[0]}, tau2={best_start_ext[1]}, "
      f"yield RMSE={best_rmse_bp_ext:.3f} bp")
print("MULTISTART EXTENDED params:", best_params_ext)

# ----

        #endregion

    # endregion

    # region PLOTS 

maturities = pd.to_datetime(maturities)

# ---- PLOT 1: ZERO COUPON CURVE FROM NSS (ACTUAL & EXTENDED)

dc_plot = ql.Actual365Fixed()
comp = ql.Continuous

max_date = nss_curve_actual.maxDate()
print("Curve maxDate:", max_date, "maxTime:", nss_curve_actual.maxTime())

times = np.linspace(0.25, 30.0, 120)

zero_actual, zero_extended, zero_boot = [], [], []
times_used = []

for t in times:
    months = int(round(t * 12))
    d = calendar.advance(ql_eval_date, ql.Period(months, ql.Months))

    if d > max_date:   # <- this is the key guard
        break

    times_used.append(months / 12.0)
    zero_actual.append(nss_curve_actual.zeroRate(d, dc_plot, comp).rate())
    zero_extended.append(nss_curve_extended.zeroRate(d, dc_plot, comp).rate())
    zero_boot.append(pre_curve.zeroRate(d, dc_plot, comp).rate())

plt.figure()
plt.plot(times_used, zero_actual, label="NSS fit: actual-only")
plt.plot(times_used, zero_extended, label="NSS fit: extended")
plt.plot(times_used, zero_boot, label="Bootstrapped (piecewise)")
plt.xlabel("Maturity (years)")
plt.ylabel("Zero rate (continuous, decimal)")
plt.title(f"Zero curves (as-of {eval_date.date()})")
plt.legend()
plt.show()

# ----

# ---- PLOT 2: PRICE ERRORS (QUOTED - MODEL) ON ACTUAL BONDS

price_resid_actualfit = quoted_price_actual - model_prices_actualfit
price_resid_extfit = quoted_price_actual - model_prices_extendedfit

plt.figure()
plt.scatter(maturities, price_resid_actualfit, label="Actual-only fit")
plt.scatter(maturities, price_resid_extfit, label="Extended fit")
plt.axhline(0.0)
plt.xlabel("Maturity date")
plt.ylabel("Quoted clean price - model clean price")
plt.title(f"Price residuals on actual bonds (as-of {eval_date.date()})")
plt.legend()
plt.show()

# ----

# ---- PLOT 3: CURVE YIELDS AND QUOTES YIELDS

plt.figure()
plt.scatter(maturities, quoted_yield_actual, label="Quoted YTM")
plt.scatter(maturities, model_yields_actualfit, label="Model YTM (actual-only fit)")
plt.scatter(maturities, model_yields_extendedfit, label="Model YTM (extended fit)")
plt.xlabel("Maturity date")
plt.ylabel("Yield (decimal)")
plt.title(f"Quoted vs model-implied yields (as-of {eval_date.date()})")
plt.legend()
plt.show()

# ----

# endregion

# endregion

# region PANEL: PUT CROSS SECTION CODE IN A FUNCTION & ITERATE OVER DATES

    # region ESTIMATE NSS FOR GIVEN DATE FUNCTION (SINGLE START)

def run_estimation_for_date(df_panel: pd.DataFrame, df_deposit: pd.DataFrame, eval_date: pd.Timestamp):

    cross_section = df_panel.loc[df_panel["date"] == eval_date].copy()
    if cross_section.empty:
        raise ValueError(f"No instruments found for evaluation date {eval_date}")

    # 1: SETTINGS
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

    # 2: BUILD BOND HELPERS FOR ACTUAL BONDS
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
        quoted_yield.append(row["yield"])

    quoted_price_actual = np.array(quoted_price, dtype=float)
    quoted_yield_actual = np.array(quoted_yield, dtype=float)

    # 3: COMPUTE BOOTSTRAPPED CURVE WITH CUBIC SPLINE 
    pre_curve = ql.PiecewiseLogCubicDiscount(settlement_days, calendar, helpers_actual, day_count)
    pre_curve.enableExtrapolation()
    curve_reference_date = calendar.advance(ql_eval_date, settlement_days, ql.Days)

    # 4: INITIAL GUESS FROM SHORT (DEPOSIT) + LONG (BOOSTRAPPED) ANCHORS
    deposit_row = df_deposit.loc[df_deposit["date"] == eval_date]
    deposit_rate = float(deposit_row["deposit"].iloc[0])
    deposit_rate_cc = np.log(1.0 + deposit_rate)

    max_maturity = max(b.maturityDate() for b in bonds_actual)
    long_rate_cc = pre_curve.zeroRate(max_maturity, day_count, ql.Continuous).rate()

    b0 = long_rate_cc
    b1 = deposit_rate_cc - b0
    b2 = 0.0
    b3 = 0.0

    tau1 = 2.0
    tau2 = 9.0
    kappa1 = 1.0 / tau1
    kappa2 = 1.0 / tau2

    guess = ql.Array(6)
    guess[0] = b0
    guess[1] = b1
    guess[2] = b2
    guess[3] = b3
    guess[4] = kappa1
    guess[5] = kappa2

    # 5: BOUNDS
    b0_min, b0_max = -0.15, 0.15
    b1_min, b1_max = -0.15, 0.15
    b2_min, b2_max = -10000.0, 10000.0
    b3_min, b3_max = -10000.0, 10000.0

    tau_min, tau_max = 0.05, 50.0
    kappa_min, kappa_max = 1.0 / tau_max, 1.0 / tau_min

    lower = ql.Array(6)
    upper = ql.Array(6)

    lower[0], upper[0] = b0_min, b0_max
    lower[1], upper[1] = b1_min, b1_max
    lower[2], upper[2] = b2_min, b2_max
    lower[3], upper[3] = b3_min, b3_max
    lower[4], upper[4] = kappa_min, kappa_max
    lower[5], upper[5] = kappa_min, kappa_max

    constraint = ql.NonhomogeneousBoundaryConstraint(lower, upper)

    # 6: PENALTY 

    l2 = ql.Array(6)
    l2[0] = 0.1     # b0 penalty strength
    l2[1] = 0.1     # b1 penalty strength
    l2[2] = 0.0     # b2 unpenalized
    l2[3] = 0.0     # b3 unpenalized
    l2[4] = 0.005     # kappa1 penalty strength
    l2[5] = 0.005     # kappa2 penalty strength

    # 7: FIT SETUP
    accuracy = 1e-10
    max_evaluations = 15000
    simplex_lambda = 0.005

    fitting = ql.SvenssonFitting(
        ql.Array(),                     # weights (inverse duration default)
        ql.Simplex(simplex_lambda),     # optimizer (simplex default)
        l2,                     # L2 penalty (none ql.Array() is default)
        0.0,                            # minCutoffTime (maturity)
        50.0,                           # maxCutoffTime (maturity)
        constraint
)

    # 8: CREATE MID SYNTHETIC ZERO COUPON BONDS
    ql_actual_maturity = sorted([b.maturityDate() for b in bonds_actual])
    synthetic_helpers = []

    for d1, d2 in zip(ql_actual_maturity[:-1], ql_actual_maturity[1:]):
        mid_serial = (d1.serialNumber() + d2.serialNumber()) // 2
        mid_date = ql.Date(mid_serial)

        # get discount factor for preliminary curve and compute synthetic clean price
        discount_factor_mid = pre_curve.discount(mid_date)
        synth_price = face * discount_factor_mid

        schedule_mid = ql.Schedule([curve_reference_date, mid_date])

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

    # 9: FIT 1: ACTUAL ONLY
    nss_curve_actual = ql.FittedBondDiscountCurve(
        settlement_days,
        calendar,
        helpers_actual,
        day_count,
        fitting,
        accuracy,
        max_evaluations,
        guess
    )

    params_actual = list(nss_curve_actual.fitResults().solution())

    # 10: FIT 2: EXTENDED 

    nss_curve_extended = ql.FittedBondDiscountCurve(
        settlement_days,
        calendar,
        helpers_extended,
        day_count,
        fitting,
        accuracy,
        max_evaluations,
        guess
    )

    params_extended = list(nss_curve_extended.fitResults().solution())

    # 11: PRICING AND RMSEs ON ACTUAL BONDS ONLY
    def price_actual_bonds(curve):
        curve_handle = ql.YieldTermStructureHandle(curve)
        engine = ql.DiscountingBondEngine(curve_handle)

        model_prices = []
        model_yields = []

        for b in bonds_actual:
            b.setPricingEngine(engine)
            mp = b.cleanPrice()
            model_prices.append(float(mp))
            my = b.bondYield(day_count, ql.Compounded, ql.Annual)
            model_yields.append(float(my))
        
        return np.array(model_prices), np.array(model_yields)

    model_prices_actualfit, model_yields_actualfit = price_actual_bonds(nss_curve_actual)
    model_prices_extendedfit, model_yields_extendedfit = price_actual_bonds(nss_curve_extended)

    rmse_price_actualfit = float(np.sqrt(np.mean((quoted_price_actual - model_prices_actualfit) ** 2)))
    rmse_price_extfit = float(np.sqrt(np.mean((quoted_price_actual - model_prices_extendedfit) ** 2)))

    rmse_yield_actualfit_bp = float(np.sqrt(np.mean((quoted_yield_actual - model_yields_actualfit) ** 2)) * 1e4)
    rmse_yield_extfit_bp = float(np.sqrt(np.mean((quoted_yield_actual - model_yields_extendedfit) ** 2)) * 1e4)

    return {
        "date" : eval_date,
        # parameters (actual-only)
        "b0_actual": params_actual[0], "b1_actual": params_actual[1], "b2_actual": params_actual[2],
        "b3_actual": params_actual[3], "k1_actual": params_actual[4], "k2_actual": params_actual[5],
        # parameters (extended)
        "b0_ext": params_extended[0], "b1_ext": params_extended[1], "b2_ext": params_extended[2],
        "b3_ext": params_extended[3], "k1_ext": params_extended[4], "k2_ext": params_extended[5],       
        # diagnostics
        "rmse_price_actual": rmse_price_actualfit,
        "rmse_price_ext": rmse_price_extfit,
        "rmse_yield_bp_actual": rmse_yield_actualfit_bp,
        "rmse_yield_bp_ext": rmse_yield_extfit_bp,
        "n_bonds": len(helpers_actual),
        "n_synth": len(synthetic_helpers),   
    }

# endregion

    # region ESTIMATE NSS FOR GIVEN DATE FUNCTION (MULTI START)

def run_estimation_for_date_multistart(df_panel: pd.DataFrame, df_deposit: pd.DataFrame, eval_date: pd.Timestamp):

    cross_section = df_panel.loc[df_panel["date"] == eval_date].copy()
    if cross_section.empty:
        raise ValueError(f"No instruments found for evaluation date {eval_date}")

    # 1: SETTINGS
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

    # 2: BUILD BOND HELPERS FOR ACTUAL BONDS
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
        quoted_yield.append(row["yield"])

    quoted_price_actual = np.array(quoted_price, dtype=float)
    quoted_yield_actual = np.array(quoted_yield, dtype=float)

    # 3: COMPUTE BOOTSTRAPPED CURVE WITH CUBIC SPLINE
    pre_curve = ql.PiecewiseLogCubicDiscount(settlement_days, calendar, helpers_actual, day_count)
    pre_curve.enableExtrapolation()
    curve_reference_date = calendar.advance(ql_eval_date, settlement_days, ql.Days)

    # 4: INITIAL GUESS FROM SHORT (DEPOSIT) + LONG (BOOSTRAPPED) ANCHORS
    deposit_row = df_deposit.loc[df_deposit["date"] == eval_date]
    deposit_rate = float(deposit_row["deposit"].iloc[0])
    deposit_rate_cc = np.log(1.0 + deposit_rate)

    max_maturity = max(b.maturityDate() for b in bonds_actual)
    long_rate_cc = pre_curve.zeroRate(max_maturity, day_count, ql.Continuous).rate()

    b0 = long_rate_cc
    b1 = deposit_rate_cc - b0
    b2 = 0.0
    b3 = 0.0

    # 5: BOUNDS
    b0_min, b0_max = -0.15, 0.15
    b1_min, b1_max = -0.15, 0.15
    b2_min, b2_max = -0.5, 0.5
    b3_min, b3_max = -0.5, 0.5

    tau_min, tau_max = 0.05, 50.0
    kappa_min, kappa_max = 1.0 / tau_max, 1.0 / tau_min

    lower = ql.Array(6)
    upper = ql.Array(6)

    lower[0], upper[0] = b0_min, b0_max
    lower[1], upper[1] = b1_min, b1_max
    lower[2], upper[2] = b2_min, b2_max
    lower[3], upper[3] = b3_min, b3_max
    lower[4], upper[4] = kappa_min, kappa_max
    lower[5], upper[5] = kappa_min, kappa_max

    constraint = ql.NonhomogeneousBoundaryConstraint(lower, upper)

    # 6: PENALTY
    l2 = ql.Array(6)
    l2[0] = 0.2
    l2[1] = 0.2
    l2[2] = 0.03
    l2[3] = 0.03
    l2[4] = 0.015
    l2[5] = 0.01

    # 7: FIT SETUP
    accuracy = 1e-10
    max_evaluations = 15000
    simplex_lambda = 0.005

    fitting = ql.SvenssonFitting(
        ql.Array(),
        ql.Simplex(simplex_lambda),
        l2,         # ql.Array() > No penalty, l2 > Penalty
        0.0,
        50.0,
        constraint
    )

    # 8: CREATE MID SYNTHETIC ZERO COUPON BONDS
    ql_actual_maturity = sorted([b.maturityDate() for b in bonds_actual])
    synthetic_helpers = []

    for d1, d2 in zip(ql_actual_maturity[:-1], ql_actual_maturity[1:]):
        mid_serial = (d1.serialNumber() + d2.serialNumber()) // 2
        mid_date = ql.Date(mid_serial)

        discount_factor_mid = pre_curve.discount(mid_date)
        synth_price = face * discount_factor_mid

        schedule_mid = ql.Schedule([curve_reference_date, mid_date])

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

    # 9: PRICING FUNCTION (used to evaluate candidates)
    def price_actual_bonds(curve):
        curve_handle = ql.YieldTermStructureHandle(curve)
        engine = ql.DiscountingBondEngine(curve_handle)

        model_prices = []
        model_yields = []

        for b in bonds_actual:
            b.setPricingEngine(engine)
            model_prices.append(float(b.cleanPrice()))
            model_yields.append(float(b.bondYield(day_count, ql.Compounded, ql.Annual)))

        return np.array(model_prices), np.array(model_yields)

    def yield_rmse_bp_from_curve(curve):
        _, model_yields = price_actual_bonds(curve)
        return float(np.sqrt(np.mean((quoted_yield_actual - model_yields) ** 2)) * 1e4)

    # 10: MULTISTART FIT 1: ACTUAL ONLY (helpers_actual)
    tau1_grid = [0.5, 2.0, 4.0, 6.0, 8.0]
    tau2_grid = [5.0, 10.0, 15.0, 25.0, 30.0]

    best_rmse_bp_actual = float("inf")
    best_curve_actual = None
    best_params_actual = None

    for tau1 in tau1_grid:
        for tau2 in tau2_grid:
            if tau2 <= tau1:
                continue

            guess_ms = ql.Array(6)
            guess_ms[0] = b0
            guess_ms[1] = b1
            guess_ms[2] = b2
            guess_ms[3] = b3
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
                rmse_bp = yield_rmse_bp_from_curve(curve_try)
                if rmse_bp < best_rmse_bp_actual:
                    best_rmse_bp_actual = rmse_bp
                    best_curve_actual = curve_try
                    best_params_actual = list(curve_try.fitResults().solution())
            except RuntimeError:
                continue

    if best_curve_actual is None:
        raise RuntimeError(f"Multistart failed for ACTUAL on {eval_date}")

    # 11: MULTISTART FIT 2: EXTENDED (helpers_extended)
    best_rmse_bp_ext = float("inf")
    best_curve_ext = None
    best_params_ext = None

    for tau1 in tau1_grid:
        for tau2 in tau2_grid:
            if tau2 <= tau1:
                continue

            guess_ms = ql.Array(6)
            guess_ms[0] = b0
            guess_ms[1] = b1
            guess_ms[2] = b2
            guess_ms[3] = b3
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
                # Selection is still based on ACTUAL bond yield RMSE (consistent with your cross-section multistart)
                rmse_bp = yield_rmse_bp_from_curve(curve_try)
                if rmse_bp < best_rmse_bp_ext:
                    best_rmse_bp_ext = rmse_bp
                    best_curve_ext = curve_try
                    best_params_ext = list(curve_try.fitResults().solution())
            except RuntimeError:
                continue

    if best_curve_ext is None:
        raise RuntimeError(f"Multistart failed for EXTENDED on {eval_date}")

    # 12: PRICING AND RMSEs ON ACTUAL BONDS ONLY (same as your original)
    model_prices_actualfit, model_yields_actualfit = price_actual_bonds(best_curve_actual)
    model_prices_extendedfit, model_yields_extendedfit = price_actual_bonds(best_curve_ext)

    rmse_price_actualfit = float(np.sqrt(np.mean((quoted_price_actual - model_prices_actualfit) ** 2)))
    rmse_price_extfit = float(np.sqrt(np.mean((quoted_price_actual - model_prices_extendedfit) ** 2)))

    rmse_yield_actualfit_bp = float(np.sqrt(np.mean((quoted_yield_actual - model_yields_actualfit) ** 2)) * 1e4)
    rmse_yield_extfit_bp = float(np.sqrt(np.mean((quoted_yield_actual - model_yields_extendedfit) ** 2)) * 1e4)

    return {
        "date": eval_date,
        # parameters (actual-only)
        "b0_actual": best_params_actual[0], "b1_actual": best_params_actual[1], "b2_actual": best_params_actual[2],
        "b3_actual": best_params_actual[3], "k1_actual": best_params_actual[4], "k2_actual": best_params_actual[5],
        # parameters (extended)
        "b0_ext": best_params_ext[0], "b1_ext": best_params_ext[1], "b2_ext": best_params_ext[2],
        "b3_ext": best_params_ext[3], "k1_ext": best_params_ext[4], "k2_ext": best_params_ext[5],
        # diagnostics
        "rmse_price_actual": rmse_price_actualfit,
        "rmse_price_ext": rmse_price_extfit,
        "rmse_yield_bp_actual": rmse_yield_actualfit_bp,
        "rmse_yield_bp_ext": rmse_yield_extfit_bp,
        "n_bonds": len(helpers_actual),
        "n_synth": len(synthetic_helpers),
    }

    # endregion

    # region RUN FUNCTION BY DATE 

# ---- BUILD MONTHLY INDEX AND MAP 

month_to_date = (
    df_SGBs.assign(month=df_SGBs["date"].dt.to_period("M"))
        .groupby("month")["date"].max()
        .sort_index()
)

# ----

# ---- RUN ESTIMATION MONTH BY MONTH (oldest -> newest)
rows = []
failed =[]

for month, d in month_to_date.items():
    eval_date = pd.Timestamp(d)
    try:
        out = run_estimation_for_date_multistart(df_SGBs, df_deposit, eval_date)
        #out = run_estimation_for_date(df_SGBs, df_deposit, eval_date)
        out["month"] = str(month)
        rows.append(out)                # collect dict from each month
    except Exception as e:
        failed.append({"month": str(month), "date": eval_date, "error": str(e)})

results = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

print("Successful months:", len(results))
print("Failed months:", len(failed))

# ----

# ---- ARRANGE AND EXPORT TO EXCEL
col_order = [
    "date", "month",
    # actual-only params
    "b0_actual", "b1_actual", "b2_actual", "b3_actual", "k1_actual", "k2_actual",
    # extended params
    "b0_ext", "b1_ext", "b2_ext", "b3_ext", "k1_ext", "k2_ext",
    # RMSEs
    "rmse_price_actual", "rmse_price_ext", "rmse_yield_bp_actual", "rmse_yield_bp_ext",
    # counts
    "n_bonds", "n_synth",
]
results = results.loc[:, col_order]

results.to_excel("nss_nominal_results.xlsx", index=False)
 
# ----

# ---- PANEL SUMMARY STATISTICS 

results["tau1_actual"] = 1.0 / results["k1_actual"]
results["tau2_actual"] = 1.0 / results["k2_actual"]

results["tau1_ext"] = 1.0 / results["k1_ext"]
results["tau2_ext"] = 1.0 / results["k2_ext"]

panel_cols = [
    # parameters (actual-only)
    "b0_actual","b1_actual","b2_actual","b3_actual","k1_actual","k2_actual",
    # parameters (extended)
    "b0_ext","b1_ext","b2_ext","b3_ext","k1_ext","k2_ext",
    # taus
    "tau1_actual","tau2_actual","tau1_ext","tau2_ext",
    # RMSE diagnostics
    "rmse_price_actual","rmse_price_ext",
    "rmse_yield_bp_actual","rmse_yield_bp_ext"
]

print("Panel summary statistics")

for c in panel_cols:
    avg_val = results[c].mean()
    min_val = results[c].min()
    max_val = results[c].max()

    print(f"{c:20s}  avg={avg_val: .6f}  min={min_val: .6f}  max={max_val: .6f}")

# ----
    # endregion

    # region PANEL PLOTS 

# ---- TERM STRUCUTRE TIME SERIES 
param_set= "ext"    # can change to "actual"

tenors = {
    "1m": 1.0 / 12.0,
    "6m": 6.0 / 12.0,
    "1y": 1.0,
    "2y": 2.0,
    "4y": 4.0,
    "6y": 6.0,
    "8y": 8.0,
    "10y": 10.0,
}

b0 = results[f"b0_{param_set}"].to_numpy()
b1 = results[f"b1_{param_set}"].to_numpy()
b2 = results[f"b2_{param_set}"].to_numpy()
b3 = results[f"b3_{param_set}"].to_numpy()
k1 = results[f"k1_{param_set}"].to_numpy()
k2 = results[f"k2_{param_set}"].to_numpy()

dates = pd.to_datetime(results["date"])

zero_ts = pd.DataFrame({"date": dates})

for label, t in tenors.items():
    t = float(t)

    x1 = k1 * t
    x2 = k2 * t

    term1 = (1.0 - np.exp(-x1)) / x1
    term2 = term1 - np.exp(-x1)
    term3 = (1.0 - np.exp(-x2)) / x2 - np.exp(-x2)

    zero_ts[label] = b0 + b1 * term1 + b2 * term2 + b3 * term3

# Convert to percent for plotting
plot_df = zero_ts.copy()
for c in tenors.keys():
    plot_df[c] = 100.0 * plot_df[c]

plt.figure(figsize=(11, 6))
for c in tenors.keys():
    plt.plot(plot_df["date"], plot_df[c], label=c)

plt.xlabel("Date")
plt.ylabel("Zero-coupon yield (%)")
plt.title(f"NSS zero yields by tenor")
plt.legend(ncol=4)
plt.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.7)
plt.tight_layout()
plt.show()

# ----

# ---- PLOT RMSEs 

dates = pd.to_datetime(results["date"])

fig, ax1 = plt.subplots(figsize=(11, 6))

# Left axis: yield RMSE (bp)
ax1.plot(dates, results["rmse_yield_bp_ext"], label="Yield RMSE (bp) - extended", linewidth=1.5)
ax1.plot(dates, results["rmse_yield_bp_actual"], label="Yield RMSE (bp) - actual-only", linewidth=1.5)

ax1.set_xlabel("Date")
ax1.set_ylabel("Yield RMSE (bp)")
ax1.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.7)

# Right axis: price RMSE
ax2 = ax1.twinx()
ax2.plot(dates, results["rmse_price_ext"], linestyle="--", label="Price RMSE - extended", linewidth=1.5)
ax2.plot(dates, results["rmse_price_actual"], linestyle="--", label="Price RMSE - actual-only", linewidth=1.5)
ax2.set_ylabel("Price RMSE (clean price points)")

# One combined legend
lines = ax1.get_lines() + ax2.get_lines()
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc="upper right")

plt.title("RMSE of yields (bp) and prices (dual axis)")
plt.tight_layout()
plt.show()

# ---- 

    # endregion

# region EXPORT ZERO-COUPON YIELDS: NOMINAL NSS MULTISTART EXTENDED

# Uses the panel results already stored in `results`
# and takes the EXTENDED multistart NSS parameters:
# b0_ext, b1_ext, b2_ext, b3_ext, k1_ext, k2_ext

maturities_months = list(range(1, 121))   # 1m, 2m, ..., 120m

# ---- ZERO YIELDS FROM EXTENDED NSS PARAMETERS
b0 = results["b0_ext"].to_numpy(dtype=float)
b1 = results["b1_ext"].to_numpy(dtype=float)
b2 = results["b2_ext"].to_numpy(dtype=float)
b3 = results["b3_ext"].to_numpy(dtype=float)
k1 = results["k1_ext"].to_numpy(dtype=float)
k2 = results["k2_ext"].to_numpy(dtype=float)

zero_yields_dict = {
    "date": pd.to_datetime(results["date"])
}

for m in maturities_months:
    t = m / 12.0
    x1 = k1 * t
    x2 = k2 * t

    term1 = np.where(np.abs(x1) < 1e-12, 1.0, (1.0 - np.exp(-x1)) / x1)
    term2 = term1 - np.exp(-x1)
    term3 = np.where(np.abs(x2) < 1e-12, 0.0, (1.0 - np.exp(-x2)) / x2 - np.exp(-x2))

    zero_yields_dict[f"y_{m}m"] = b0 + b1 * term1 + b2 * term2 + b3 * term3

zero_yields = pd.DataFrame(zero_yields_dict)

# ---- PARAMETER / DIAGNOSTIC SHEET
fit_params = results.loc[:, [
    "date",
    "b0_ext", "b1_ext", "b2_ext", "b3_ext",
    "k1_ext", "k2_ext",
    "rmse_price_ext", "rmse_yield_bp_ext",
    "n_bonds", "n_synth",
]].copy()

fit_params["tau1_ext"] = 1.0 / fit_params["k1_ext"]
fit_params["tau2_ext"] = 1.0 / fit_params["k2_ext"]

fit_params = fit_params.loc[:, [
    "date",
    "b0_ext", "b1_ext", "b2_ext", "b3_ext",
    "k1_ext", "tau1_ext",
    "k2_ext", "tau2_ext",
    "rmse_price_ext", "rmse_yield_bp_ext",
    "n_bonds", "n_synth",
]]

# ---- EXPORT TO EXCEL
with pd.ExcelWriter("zero_yields_SGB.xlsx", engine="openpyxl") as writer:
    zero_yields.to_excel(writer, sheet_name="zero yields", index=False)
    fit_params.to_excel(writer, sheet_name="fit params", index=False)

print("Exported zero_yields_SGB.xlsx")

# endregion