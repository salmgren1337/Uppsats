import pandas as pd
import QuantLib as ql
import numpy as np
import matplotlib.pyplot as plt

# IMPORTS 
raw_data = pd.read_excel("statsobligationer_data.xlsx", sheet_name = "SGBi long")
short_rates = pd.read_excel("statsobligationer_data.xlsx", sheet_name = "korta räntor riksbanken")
kpi = pd.read_excel("kpi_data.xlsx", sheet_name = "basår 1980")

START = pd.Timestamp("2000-01-01")
END = pd.Timestamp("2025-12-31")

# region PREPARE BOND DATA

# ---- PREPARE SGB DATA

BasKPI_tofill = raw_data.loc[raw_data["Serie"] == 3001, "BasKPI"].iloc[0]
Bas_tofill = raw_data.loc[raw_data["Serie"] == 3001, "Bas"].iloc[0]
Slutindexmånad_tofill = raw_data.loc[raw_data["Serie"] == 3001, "Slutindexmånad"].iloc[0]

raw_data.loc[raw_data["Serie"].isin([3002, 3003]), "BasKPI"] = BasKPI_tofill
raw_data.loc[raw_data["Serie"].isin([3002, 3003]), "Bas"] = Bas_tofill
raw_data.loc[raw_data["Serie"].isin([3002, 3003]), "Slutindexmånad"] = Slutindexmånad_tofill

keep_cols = [
    "Date",
    "PX_LAST",
    "YLD_YTM_MID",
    "Kupong",
    "Issue date",
    "Maturity date",
    "ISIN",
    "Serie",
    "BasKPI"
]

df_SGBILs = raw_data.loc[:, keep_cols].copy()

df_SGBILs["Date"] = pd.to_datetime(df_SGBILs["Date"], errors="coerce")
df_SGBILs["Issue date"] = pd.to_datetime(df_SGBILs["Issue date"], errors="coerce")
df_SGBILs["Maturity date"] = pd.to_datetime(df_SGBILs["Maturity date"], errors="coerce")

df_SGBILs = df_SGBILs.rename(columns={
    "Date": "date",
    "PX_LAST": "price",
    "YLD_YTM_MID": "yield",
    "Kupong": "coupon",
    "Issue date": "issue_date",
    "Maturity date": "maturity_date",
    "ISIN": "isin",
    "Serie": "serie",
    "BasKPI": "bas_kpi"
})

df_SGBILs = df_SGBILs.loc[(df_SGBILs["date"] >= START) & (df_SGBILs["date"] <= END)]
# ---- Remove illiquid bond with duplicate maturity (3113)
df_SGBILs = df_SGBILs[df_SGBILs["serie"] != 3103].copy()

for col in ["price", "yield", "coupon", "bas_kpi"]:
    df_SGBILs[col] = pd.to_numeric(df_SGBILs[col], errors="coerce")

df_SGBILs["yield"] = df_SGBILs["yield"] / 100.0
df_SGBILs["coupon"] = df_SGBILs["coupon"] / 100.0

df_SGBILs = df_SGBILs.sort_values(["date", "maturity_date"]).reset_index(drop=True)

#filter bonds with less than 2 year and more than 20 years to maturity
time_to_maturity = (df_SGBILs["maturity_date"] - df_SGBILs["date"]).dt.days
df_SGBILs = df_SGBILs[(time_to_maturity >= 2 * 365) & (time_to_maturity <= 20 * 365)]
df_SGBILs = df_SGBILs.reset_index(drop=True)

# ----

# ---- PREPARE DEPOSIT DATA

df_SGBILs["month"] = df_SGBILs["date"].dt.to_period("M")
month_trading_date = df_SGBILs.groupby("month")["date"].max()
df_SGBILs = df_SGBILs.drop(columns=["month"])

df_deposit = short_rates.loc[:, ["date", "deposit"]].copy()
df_deposit["date"] = pd.to_datetime(df_deposit["date"], errors="coerce") 
df_deposit["deposit"] = pd.to_numeric(df_deposit["deposit"], errors="coerce") / 100.0
df_deposit["month"] = df_deposit["date"].dt.to_period("M")
df_deposit["date"] = df_deposit["month"].map(month_trading_date) # map to SGBIL last trading day 
df_deposit = df_deposit.drop(columns=["month"]).dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

# ----

# ---- PREPARE INFLATION DATA 

df_inflation = kpi.loc[:, ["date", "inflation year"]].copy()
df_inflation["date"] = pd.to_datetime (df_inflation["date"], errors="coerce")
df_inflation["inflation year"] = pd.to_numeric(df_inflation["inflation year"], errors="coerce") 
df_inflation["month"] = df_inflation["date"].dt.to_period("M")
df_inflation["date"] = df_inflation["month"].map(month_trading_date) # map to SGBIL last trading day 
df_inflation = df_inflation.drop(columns=["month"]).dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

# ----

# ---- COMPUTE REAL SHORT RATE ANCHOR

df_real_deposit = pd.merge(df_deposit, df_inflation, on="date", how="inner")
df_real_deposit["real_deposit"] = df_real_deposit["deposit"] - df_real_deposit["inflation year"]

df_real_deposit = (
    df_real_deposit[["date", "real_deposit"]]
    .dropna(subset=["date", "real_deposit"])
    .drop_duplicates(subset=["date"])
    .sort_values("date")
    .reset_index(drop=True)
)

# endregion

# region CROSS SECTION 

    # region SETUP (SETTINGS & HELPERS)

# ---- SETTINGS: 
eval_date = pd.Timestamp("2011-07-29")
cross_section = df_SGBILs.loc[df_SGBILs["date"] == eval_date].copy()

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

# ---- KPI HELPER: Swedish linker reference CPI uses 3-month lag + linear interpolation
df_kpi_monthly = kpi.loc[:, ["date", "KPI"]].copy()
df_kpi_monthly["date"] = pd.to_datetime(df_kpi_monthly["date"], errors="coerce")
df_kpi_monthly["KPI"] = pd.to_numeric(df_kpi_monthly["KPI"], errors="coerce")
df_kpi_monthly["month"] = df_kpi_monthly["date"].dt.to_period("M")
df_kpi_monthly = (
    df_kpi_monthly[["month", "KPI"]]
    .dropna()
    .drop_duplicates(subset=["month"])
    .sort_values("month")
    .reset_index(drop=True)
)

kpi_by_month = df_kpi_monthly.set_index("month")["KPI"]

def swedish_reference_kpi(settlement_date: pd.Timestamp, kpi_by_month: pd.Series, lag_months: int = 3) -> float:
    """
    Swedish inflation-linked bond reference CPI:
    - reference index on day 1 of month M = CPI from M - 3 months
    - otherwise linear interpolation toward CPI from M - 2 months
    - all months assumed to have 30 days
    - if day = 31, use 30
    - Feb 28 remains 28
    """
    current_month = settlement_date.to_period("M")
    ref_month_0 = current_month - lag_months       # CPI for month M-3
    ref_month_1 = current_month - (lag_months - 1) # CPI for month M-2

    if ref_month_0 not in kpi_by_month.index or ref_month_1 not in kpi_by_month.index:
        raise ValueError(f"Missing KPI needed for interpolation: {ref_month_0}, {ref_month_1}")

    cpi_0 = float(kpi_by_month.loc[ref_month_0])
    cpi_1 = float(kpi_by_month.loc[ref_month_1])

    day = settlement_date.day
    if day == 31:
        day = 30

    # Debt Office convention: linear interpolation with 30-day months
    return cpi_0 + (day - 1) / 30.0 * (cpi_1 - cpi_0)

ql_settlement_date = calendar.advance(ql_eval_date, settlement_days, ql.Days)
settlement_date = pd.Timestamp(
    ql_settlement_date.year(),
    ql_settlement_date.month(),
    ql_settlement_date.dayOfMonth()
)

reference_kpi = swedish_reference_kpi(settlement_date, kpi_by_month, lag_months=3)

cross_section["index_factor"] = reference_kpi / cross_section["bas_kpi"]

# real price: 
cross_section["real_price"] = cross_section["price"] / cross_section["index_factor"]

# real yield to maturity:
real_ytm_list = []

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

    bond = ql.FixedRateBond(
        settlement_days,
        face,
        schedule,
        [float(row["coupon"])],
        day_count,
        business_convention,
        redemption,
        ql_issue
    )

    real_clean_price = float(row["real_price"])
    price_obj = ql.BondPrice(real_clean_price, ql.BondPrice.Clean)

    real_ytm = ql.BondFunctions.bondYield(
        bond,
        price_obj,
        day_count,
        ql.Compounded,
        ql.Annual,          # compounding frequency
        ql_settlement_date,
        1.0e-12,
        1000,               # max number of evaluations
        0.02                # intitial guess 
    )

    real_ytm_list.append(float(real_ytm))

cross_section["real_ytm"] = real_ytm_list


# ---- BUILD FixedRateBondHelper PER ACTUAL BOND WHICH PAIRS PRICE AND CASHFLOWS
helpers_actual = []
bonds_actual = []
quoted_price = []
quoted_yield = []
maturities = []

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
    # Use real clean price in the ordinary fixed-rate helper
    clean_price = float(row["real_price"])
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
    quoted_yield.append(row["real_ytm"])
    maturities.append(row["maturity_date"])

print("Actual helpers built:", len(helpers_actual))
print(f"Settlement date: {settlement_date.date()}, reference KPI: {reference_kpi:.6f}")

# ---- BUILD SYNTHETIC MIDPOINT ZERO-COUPON BONDS (REAL)
pre_curve = ql.PiecewiseLogCubicDiscount(settlement_days, calendar, helpers_actual, day_count)
pre_curve.enableExtrapolation()

curve_reference_date = calendar.advance(ql_eval_date, settlement_days, ql.Days)
actual_maturity_dates = sorted({b.maturityDate() for b in bonds_actual})

synthetic_helpers = []
synthetic_bonds = []

for d1, d2 in zip(actual_maturity_dates[:-1], actual_maturity_dates[1:]):
    t1 = day_count.yearFraction(curve_reference_date, d1)
    t2 = day_count.yearFraction(curve_reference_date, d2)

    if t1 <= 0.0 or t2 <= 0.0:
        continue

    t_mid = 0.5 * (t1 + t2)
    months_mid = int(round(t_mid * 12.0))
    mid_date = calendar.advance(curve_reference_date, ql.Period(months_mid, ql.Months))

    if mid_date <= curve_reference_date or mid_date >= d2:
        continue

    discount_factor_mid = pre_curve.discount(mid_date)
    synth_price = face * discount_factor_mid

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

helpers_extended = helpers_actual + synthetic_helpers
bonds_extended = bonds_actual + synthetic_bonds

quoted_price_actual = np.array(quoted_price, dtype=float)
quoted_yield_actual = np.array(quoted_yield, dtype=float)

print("Synthetic helpers built:", len(synthetic_helpers))
print("Total helpers in extended set:", len(helpers_extended))

    # endregion

    # region FIT NSS CURVES 

# ---- INITIAL GUESS FROM SHORT (DEPOSIT) + LONG (BOOTSTRAPPED ANCHORS)

# 1) Short rate: Riksbank ON deposit rate - annual inflation rate at eval_date
real_deposit_rate = float(
    df_real_deposit.loc[df_real_deposit["date"] == eval_date, "real_deposit"].iloc[0]
)
real_deposit_rate_cc = np.log(1.0 + real_deposit_rate)

# 2) Long rate: zero rate from preliminary bootstrapped REAL curve at max observed maturity
max_maturity = max(b.maturityDate() for b in bonds_actual)
long_rate_cc = pre_curve.zeroRate(max_maturity, day_count, ql.Continuous).rate()

# 3) Build initial guesses
b0 = long_rate_cc
b1 = real_deposit_rate_cc - b0
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

print(f"Initial guess from anchors: short={real_deposit_rate_cc:.6f}, long={long_rate_cc:.6f}, "
      f"b0={b0:.6f}, b1={b1:.6f}, kappa1={kappa1:.6f}, kappa2={kappa2:.6f}")

# ---- BOUNDS
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

# ---- PENALTY
l2 = ql.Array(6)
l2[0] = 0.2
l2[1] = 0.2
l2[2] = 0.03
l2[3] = 0.03
l2[4] = 0.015
l2[5] = 0.01

# ---- FIT SETUP
accuracy = 1e-10
max_evaluations = 15000
simplex_lambda = 0.01
min_cutoff = 0.0
max_cutoff = 50.0

fitting = ql.SvenssonFitting(
    ql.Array(),
    ql.Simplex(simplex_lambda),
    l2,
    min_cutoff,
    max_cutoff,
    constraint
)

# ---- FIT 1: NSS USING ACTUAL BONDS ONLY
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

print("ACTUAL-ONLY params:", list(params_actual))
print("ACTUAL-ONLY iterations:", nss_curve_actual.fitResults().numberOfIterations())

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

rmse_price_actualfit = float(np.sqrt(np.mean((quoted_price_actual - model_prices_actualfit) ** 2)))
rmse_price_extfit = float(np.sqrt(np.mean((quoted_price_actual - model_prices_extendedfit) ** 2)))

rmse_yield_actualfit_bp = float(np.sqrt(np.mean((quoted_yield_actual - model_yields_actualfit) ** 2)) * 1e4)
rmse_yield_extfit_bp = float(np.sqrt(np.mean((quoted_yield_actual - model_yields_extendedfit) ** 2)) * 1e4)

print("RMSE price (actual-only fit):", rmse_price_actualfit)
print("RMSE price (extended fit):   ", rmse_price_extfit)

print("RMSE yield bp (actual-only fit):", rmse_yield_actualfit_bp)
print("RMSE yield bp (extended fit):   ", rmse_yield_extfit_bp)

    # endregion

    # region FIT NS CURVES 

# ---- INITIAL GUESS FROM SHORT (DEPOSIT) + LONG (BOOTSTRAPPED ANCHORS)
# Short rate: Riksbank ON deposit rate - annual inflation rate at eval_date
real_deposit_rate = float(
    df_real_deposit.loc[df_real_deposit["date"] == eval_date, "real_deposit"].iloc[0]
)
real_deposit_rate_cc = np.log(1.0 + real_deposit_rate)

# Long rate: zero rate from preliminary bootstrapped REAL curve at max observed maturity
max_maturity = max(b.maturityDate() for b in bonds_actual)
long_rate_cc = pre_curve.zeroRate(max_maturity, day_count, ql.Continuous).rate()

# Build initial guesses:
b0_ns = long_rate_cc
b1_ns = real_deposit_rate_cc - b0_ns
b2_ns = 0.0

tau_ns = 2.0
kappa_ns = 1.0 / tau_ns

guess_ns = ql.Array(4)
guess_ns[0] = b0_ns
guess_ns[1] = b1_ns
guess_ns[2] = b2_ns
guess_ns[3] = kappa_ns

print(
    f"NS initial guess from anchors: short={real_deposit_rate_cc:.6f}, "
    f"long={long_rate_cc:.6f}, b0={b0_ns:.6f}, b1={b1_ns:.6f}, "
    f"kappa={kappa_ns:.6f}"
)

# ---- BOUNDS
b0_min, b0_max = -0.15, 0.15
b1_min, b1_max = -0.15, 0.15
b2_min, b2_max = -0.5, 0.5

tau_min, tau_max = 0.05, 50.0
kappa_min, kappa_max = 1.0 / tau_max, 1.0 / tau_min

lower_ns = ql.Array(4)
upper_ns = ql.Array(4)

lower_ns[0], upper_ns[0] = b0_min, b0_max
lower_ns[1], upper_ns[1] = b1_min, b1_max
lower_ns[2], upper_ns[2] = b2_min, b2_max
lower_ns[3], upper_ns[3] = kappa_min, kappa_max

constraint_ns = ql.NonhomogeneousBoundaryConstraint(lower_ns, upper_ns)

# ---- PENALTY
l2_ns = ql.Array(4)
l2_ns[0] = 0.2
l2_ns[1] = 0.2
l2_ns[2] = 0.03
l2_ns[3] = 0.015

# ---- FIT SETUP
accuracy_ns = 1e-10
max_evaluations_ns = 15000
simplex_lambda_ns = 0.01
min_cutoff_ns = 0.0
max_cutoff_ns = 50.0

fitting_ns = ql.NelsonSiegelFitting(
    ql.Array(),
    ql.Simplex(simplex_lambda_ns),
    l2_ns,
    min_cutoff_ns,
    max_cutoff_ns,
    constraint_ns
)

# ---- FIT 1: NS USING ACTUAL BONDS ONLY
ns_curve_actual = ql.FittedBondDiscountCurve(
    settlement_days,
    calendar,
    helpers_actual,
    day_count,
    fitting_ns,
    accuracy_ns,
    max_evaluations_ns,
    guess_ns
)

params_ns_actual = list(ns_curve_actual.fitResults().solution())

print("NS ACTUAL-ONLY params:", params_ns_actual)
print("NS ACTUAL-ONLY iterations:", ns_curve_actual.fitResults().numberOfIterations())

# ---- FIT 2: NS USING EXTENDED SET
ns_curve_extended = ql.FittedBondDiscountCurve(
    settlement_days,
    calendar,
    helpers_extended,
    day_count,
    fitting_ns,
    accuracy_ns,
    max_evaluations_ns,
    guess_ns
)

params_ns_extended = list(ns_curve_extended.fitResults().solution())

print("NS EXTENDED params:", params_ns_extended)
print("NS EXTENDED iterations:", ns_curve_extended.fitResults().numberOfIterations())

# ---- PRICE ACTUAL BONDS OF BOTH NS CURVES & COMPUTE RMSEs

model_prices_ns_actualfit, model_yields_ns_actualfit = price_actual_bonds(ns_curve_actual)
model_prices_ns_extendedfit, model_yields_ns_extendedfit = price_actual_bonds(ns_curve_extended)

rmse_price_ns_actualfit = float(
    np.sqrt(np.mean((quoted_price_actual - model_prices_ns_actualfit) ** 2))
)
rmse_price_ns_extendedfit = float(
    np.sqrt(np.mean((quoted_price_actual - model_prices_ns_extendedfit) ** 2))
)

rmse_yield_ns_actualfit_bp = float(
    np.sqrt(np.mean((quoted_yield_actual - model_yields_ns_actualfit) ** 2)) * 1e4
)
rmse_yield_ns_extendedfit_bp = float(
    np.sqrt(np.mean((quoted_yield_actual - model_yields_ns_extendedfit) ** 2)) * 1e4
)

print("NS RMSE price (actual-only fit):", rmse_price_ns_actualfit)
print("NS RMSE price (extended fit):   ", rmse_price_ns_extendedfit)

print("NS RMSE yield bp (actual-only fit):", rmse_yield_ns_actualfit_bp)
print("NS RMSE yield bp (extended fit):   ", rmse_yield_ns_extendedfit_bp)

    # endregion

    # region CROSS SECTION SINGLE-START PLOTS

# ---- PLOT SETTINGS

dc_plot = day_count
comp = ql.Continuous

max_date = max(b.maturityDate() for b in bonds_actual)
times = np.arange(1.0 / 12.0, 35.0 + 1.0 / 12.0, 1.0 / 12.0)

# ---- PLOT 1: SINGLE-START ZERO CURVES (NS, NSS, BOOTSTRAPPED IN SAME FIGURE)

times_used = []

zero_boot = []

zero_nss_actual = []
zero_nss_extended = []

zero_ns_actual = []
zero_ns_extended = []

for t in times:
    months = int(round(t * 12.0))
    d = calendar.advance(ql_eval_date, ql.Period(months, ql.Months))

    if d > max_date:
        break

    times_used.append(months / 12.0)

    zero_boot.append(pre_curve.zeroRate(d, dc_plot, comp).rate())

    zero_nss_actual.append(nss_curve_actual.zeroRate(d, dc_plot, comp).rate())
    zero_nss_extended.append(nss_curve_extended.zeroRate(d, dc_plot, comp).rate())

    zero_ns_actual.append(ns_curve_actual.zeroRate(d, dc_plot, comp).rate())
    zero_ns_extended.append(ns_curve_extended.zeroRate(d, dc_plot, comp).rate())

plt.figure(figsize=(11, 6))
plt.plot(times_used, zero_boot, label="Bootstrapped (piecewise)")
plt.plot(times_used, zero_nss_actual, label="NSS single-start: actual-only")
plt.plot(times_used, zero_nss_extended, label="NSS single-start: extended")
plt.plot(times_used, zero_ns_actual, label="NS single-start: actual-only")
plt.plot(times_used, zero_ns_extended, label="NS single-start: extended")
plt.xlabel("Maturity (years)")
plt.ylabel("Real zero rate (continuous, decimal)")
plt.title(f"Inflation-linked single-start zero curves (as-of {eval_date.date()})")
plt.legend()
plt.show()

# ---- PLOT 2: PRICE RESIDUALS OF SINGLE-START FITS ON ACTUAL BONDS

price_resid_nss_actual = quoted_price_actual - model_prices_actualfit
price_resid_nss_extended = quoted_price_actual - model_prices_extendedfit

price_resid_ns_actual = quoted_price_actual - model_prices_ns_actualfit
price_resid_ns_extended = quoted_price_actual - model_prices_ns_extendedfit

plt.figure(figsize=(11, 6))
plt.scatter(maturities, price_resid_nss_actual, label="NSS single-start: actual-only")
plt.scatter(maturities, price_resid_nss_extended, label="NSS single-start: extended")
plt.scatter(maturities, price_resid_ns_actual, label="NS single-start: actual-only")
plt.scatter(maturities, price_resid_ns_extended, label="NS single-start: extended")
plt.axhline(0.0)
plt.xlabel("Maturity date")
plt.ylabel("Real clean price - model clean price")
plt.title(f"Single-start price residuals on actual bonds (as-of {eval_date.date()})")
plt.legend()
plt.show()

# ---- PLOT 3: QUOTED VS MODEL YIELDS FOR ALL SINGLE-START FITS

plt.figure(figsize=(11, 6))
plt.scatter(maturities, quoted_yield_actual, label="Quoted YTM")
plt.scatter(maturities, model_yields_actualfit, label="NSS single-start: actual-only")
plt.scatter(maturities, model_yields_extendedfit, label="NSS single-start: extended")
plt.scatter(maturities, model_yields_ns_actualfit, label="NS single-start: actual-only")
plt.scatter(maturities, model_yields_ns_extendedfit, label="NS single-start: extended")
plt.xlabel("Maturity date")
plt.ylabel("Yield (decimal)")
plt.title(f"Quoted vs single-start model yields (as-of {eval_date.date()})")
plt.legend()
plt.show()

    # endregion

    # region MULTISTART NSS FITS

def yield_rmse_bp_from_curve(curve):
    _, model_yields = price_actual_bonds(curve)
    return float(np.sqrt(np.mean((quoted_yield_actual - model_yields) ** 2)) * 1e4)

tau1_grid = [0.5, 2.0, 4.0, 6.0, 8.0]
tau2_grid = [5.0, 10.0, 15.0, 25.0, 30.0]

# ---- MULTISTART ON ACTUAL BONDS

best_rmse_bp_nss_actual = float("inf")
best_curve_nss_actual = None
best_params_nss_actual = None
best_start_nss_actual = None

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

            if rmse_bp < best_rmse_bp_nss_actual:
                best_rmse_bp_nss_actual = rmse_bp
                best_curve_nss_actual = curve_try
                best_params_nss_actual = list(curve_try.fitResults().solution())
                best_start_nss_actual = (tau1, tau2)

        except RuntimeError:
            continue

if best_curve_nss_actual is None:
    raise RuntimeError(f"NSS multistart failed on ACTUAL helpers for {eval_date}")

print(
    f"NSS MULTISTART ACTUAL best start: "
    f"tau1={best_start_nss_actual[0]}, tau2={best_start_nss_actual[1]}, "
    f"yield RMSE={best_rmse_bp_nss_actual:.3f} bp"
)
print("NSS MULTISTART ACTUAL params:", best_params_nss_actual)

# ---- MULTISTART ON EXTENDED INSTRUMENTS

best_rmse_bp_nss_ext = float("inf")
best_curve_nss_ext = None
best_params_nss_ext = None
best_start_nss_ext = None

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
                settlement_days,
                calendar,
                helpers_extended,
                day_count,
                fitting,
                accuracy,
                max_evaluations,
                guess_ms
            )

            # selection criterion remains RMSE on ACTUAL bond yields
            rmse_bp = yield_rmse_bp_from_curve(curve_try)

            if rmse_bp < best_rmse_bp_nss_ext:
                best_rmse_bp_nss_ext = rmse_bp
                best_curve_nss_ext = curve_try
                best_params_nss_ext = list(curve_try.fitResults().solution())
                best_start_nss_ext = (tau1, tau2)

        except RuntimeError:
            continue

if best_curve_nss_ext is None:
    raise RuntimeError(f"NSS multistart failed on EXTENDED helpers for {eval_date}")

print(
    f"NSS MULTISTART EXTENDED best start: "
    f"tau1={best_start_nss_ext[0]}, tau2={best_start_nss_ext[1]}, "
    f"yield RMSE={best_rmse_bp_nss_ext:.3f} bp"
)
print("NSS MULTISTART EXTENDED params:", best_params_nss_ext)

# ---- PRICE ACTUAL BONDS OF BOTH WINNING NSS MULTISTART CURVES

model_prices_nss_ms_actual, model_yields_nss_ms_actual = price_actual_bonds(best_curve_nss_actual)
model_prices_nss_ms_ext, model_yields_nss_ms_ext = price_actual_bonds(best_curve_nss_ext)

rmse_price_nss_ms_actual = float(
    np.sqrt(np.mean((quoted_price_actual - model_prices_nss_ms_actual) ** 2))
)
rmse_price_nss_ms_ext = float(
    np.sqrt(np.mean((quoted_price_actual - model_prices_nss_ms_ext) ** 2))
)

rmse_yield_nss_ms_actual_bp = float(
    np.sqrt(np.mean((quoted_yield_actual - model_yields_nss_ms_actual) ** 2)) * 1e4
)
rmse_yield_nss_ms_ext_bp = float(
    np.sqrt(np.mean((quoted_yield_actual - model_yields_nss_ms_ext) ** 2)) * 1e4
)

print("NSS MULTISTART RMSE price (actual-only fit):", rmse_price_nss_ms_actual)
print("NSS MULTISTART RMSE price (extended fit):   ", rmse_price_nss_ms_ext)

print("NSS MULTISTART RMSE yield bp (actual-only fit):", rmse_yield_nss_ms_actual_bp)
print("NSS MULTISTART RMSE yield bp (extended fit):   ", rmse_yield_nss_ms_ext_bp)

    # endregion

    # region MULTISTART NS FITS

def yield_rmse_bp_from_curve_ns(curve):
    _, model_yields = price_actual_bonds(curve)
    return float(np.sqrt(np.mean((quoted_yield_actual - model_yields) ** 2)) * 1e4)

tau_grid_ns = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0]

# ---- MULTISTART ON ACTUAL BONDS

best_rmse_bp_ns_actual = float("inf")
best_curve_ns_actual = None
best_params_ns_actual = None
best_start_ns_actual = None

for tau in tau_grid_ns:
    guess_ns_ms = ql.Array(4)
    guess_ns_ms[0] = b0_ns
    guess_ns_ms[1] = b1_ns
    guess_ns_ms[2] = 0.0
    guess_ns_ms[3] = 1.0 / tau

    try:
        curve_try = ql.FittedBondDiscountCurve(
            settlement_days,
            calendar,
            helpers_actual,
            day_count,
            fitting_ns,
            accuracy_ns,
            max_evaluations_ns,
            guess_ns_ms
        )

        rmse_bp = yield_rmse_bp_from_curve_ns(curve_try)

        if rmse_bp < best_rmse_bp_ns_actual:
            best_rmse_bp_ns_actual = rmse_bp
            best_curve_ns_actual = curve_try
            best_params_ns_actual = list(curve_try.fitResults().solution())
            best_start_ns_actual = tau

    except RuntimeError:
        continue

if best_curve_ns_actual is None:
    raise RuntimeError(f"NS multistart failed on ACTUAL helpers for {eval_date}")

print(
    f"NS MULTISTART ACTUAL best start: "
    f"tau={best_start_ns_actual}, yield RMSE={best_rmse_bp_ns_actual:.3f} bp"
)
print("NS MULTISTART ACTUAL params:", best_params_ns_actual)

# ---- MULTISTART ON EXTENDED INSTRUMENTS

best_rmse_bp_ns_ext = float("inf")
best_curve_ns_ext = None
best_params_ns_ext = None
best_start_ns_ext = None

for tau in tau_grid_ns:
    guess_ns_ms = ql.Array(4)
    guess_ns_ms[0] = b0_ns
    guess_ns_ms[1] = b1_ns
    guess_ns_ms[2] = 0.0
    guess_ns_ms[3] = 1.0 / tau

    try:
        curve_try = ql.FittedBondDiscountCurve(
            settlement_days,
            calendar,
            helpers_extended,
            day_count,
            fitting_ns,
            accuracy_ns,
            max_evaluations_ns,
            guess_ns_ms
        )

        # selection criterion remains RMSE on ACTUAL bond yields
        rmse_bp = yield_rmse_bp_from_curve_ns(curve_try)

        if rmse_bp < best_rmse_bp_ns_ext:
            best_rmse_bp_ns_ext = rmse_bp
            best_curve_ns_ext = curve_try
            best_params_ns_ext = list(curve_try.fitResults().solution())
            best_start_ns_ext = tau

    except RuntimeError:
        continue

if best_curve_ns_ext is None:
    raise RuntimeError(f"NS multistart failed on EXTENDED helpers for {eval_date}")

print(
    f"NS MULTISTART EXTENDED best start: "
    f"tau={best_start_ns_ext}, yield RMSE={best_rmse_bp_ns_ext:.3f} bp"
)
print("NS MULTISTART EXTENDED params:", best_params_ns_ext)

# ---- PRICE ACTUAL BONDS OF BOTH WINNING NS MULTISTART CURVES

model_prices_ns_ms_actual, model_yields_ns_ms_actual = price_actual_bonds(best_curve_ns_actual)
model_prices_ns_ms_ext, model_yields_ns_ms_ext = price_actual_bonds(best_curve_ns_ext)

rmse_price_ns_ms_actual = float(
    np.sqrt(np.mean((quoted_price_actual - model_prices_ns_ms_actual) ** 2))
)
rmse_price_ns_ms_ext = float(
    np.sqrt(np.mean((quoted_price_actual - model_prices_ns_ms_ext) ** 2))
)

rmse_yield_ns_ms_actual_bp = float(
    np.sqrt(np.mean((quoted_yield_actual - model_yields_ns_ms_actual) ** 2)) * 1e4
)
rmse_yield_ns_ms_ext_bp = float(
    np.sqrt(np.mean((quoted_yield_actual - model_yields_ns_ms_ext) ** 2)) * 1e4
)

print("NS MULTISTART RMSE price (actual-only fit):", rmse_price_ns_ms_actual)
print("NS MULTISTART RMSE price (extended fit):   ", rmse_price_ns_ms_ext)

print("NS MULTISTART RMSE yield bp (actual-only fit):", rmse_yield_ns_ms_actual_bp)
print("NS MULTISTART RMSE yield bp (extended fit):   ", rmse_yield_ns_ms_ext_bp)

    # endregion

    # region CROSS SECTION MULTISTART PLOTS

# ---- PLOT SETTINGS

dc_plot = day_count
comp = ql.Continuous

max_date = max(b.maturityDate() for b in bonds_actual)
times = np.arange(1.0 / 12.0, 35.0 + 1.0 / 12.0, 1.0 / 12.0)

# ---- PLOT 1: ALL MULTISTART ZERO CURVES IN THE SAME FIGURE

times_used = []
zero_boot = []

zero_nss_ms_actual = []
zero_nss_ms_ext = []

zero_ns_ms_actual = []
zero_ns_ms_ext = []

for t in times:
    months = int(round(t * 12.0))
    d = calendar.advance(ql_eval_date, ql.Period(months, ql.Months))

    if d > max_date:
        break

    times_used.append(months / 12.0)

    zero_boot.append(pre_curve.zeroRate(d, dc_plot, comp).rate())

    zero_nss_ms_actual.append(best_curve_nss_actual.zeroRate(d, dc_plot, comp).rate())
    zero_nss_ms_ext.append(best_curve_nss_ext.zeroRate(d, dc_plot, comp).rate())

    zero_ns_ms_actual.append(best_curve_ns_actual.zeroRate(d, dc_plot, comp).rate())
    zero_ns_ms_ext.append(best_curve_ns_ext.zeroRate(d, dc_plot, comp).rate())

plt.figure(figsize=(11, 6))
plt.plot(times_used, zero_boot, label="Bootstrapped (piecewise)")
plt.plot(times_used, zero_nss_ms_actual, label="NSS multistart: actual-only")
plt.plot(times_used, zero_nss_ms_ext, label="NSS multistart: extended")
plt.plot(times_used, zero_ns_ms_actual, label="NS multistart: actual-only")
plt.plot(times_used, zero_ns_ms_ext, label="NS multistart: extended")
plt.xlabel("Maturity (years)")
plt.ylabel("Real zero rate (continuous, decimal)")
plt.title(f"Inflation-linked multistart zero curves (as-of {eval_date.date()})")
plt.legend()
plt.show()

# ---- PLOT 2: PRICE RESIDUALS OF MULTISTART FITS ON ACTUAL BONDS

price_resid_nss_ms_actual = quoted_price_actual - model_prices_nss_ms_actual
price_resid_nss_ms_ext = quoted_price_actual - model_prices_nss_ms_ext
price_resid_ns_ms_actual = quoted_price_actual - model_prices_ns_ms_actual
price_resid_ns_ms_ext = quoted_price_actual - model_prices_ns_ms_ext

plt.figure(figsize=(11, 6))
plt.scatter(maturities, price_resid_nss_ms_actual, label="NSS ms actual-only")
plt.scatter(maturities, price_resid_nss_ms_ext, label="NSS ms extended")
plt.scatter(maturities, price_resid_ns_ms_actual, label="NS ms actual-only")
plt.scatter(maturities, price_resid_ns_ms_ext, label="NS ms extended")
plt.axhline(0.0)
plt.xlabel("Maturity date")
plt.ylabel("Real clean price - model clean price")
plt.title(f"Multistart price residuals on actual bonds (as-of {eval_date.date()})")
plt.legend()
plt.show()

# ---- PLOT 3: QUOTED VS MODEL YIELDS FOR ALL MULTISTART FITS

plt.figure(figsize=(11, 6))
plt.scatter(maturities, quoted_yield_actual, label="Quoted YTM")
plt.scatter(maturities, model_yields_nss_ms_actual, label="NSS ms actual-only")
plt.scatter(maturities, model_yields_nss_ms_ext, label="NSS ms extended")
plt.scatter(maturities, model_yields_ns_ms_actual, label="NS ms actual-only")
plt.scatter(maturities, model_yields_ns_ms_ext, label="NS ms extended")
plt.xlabel("Maturity date")
plt.ylabel("Yield (decimal)")
plt.title(f"Quoted vs multistart model yields (as-of {eval_date.date()})")
plt.legend()
plt.show()

    # endregion

# endregion

# region PANEL 

    # region SETUP

# ---- KPI HELPER FOR PANEL
df_kpi_monthly = kpi.loc[:, ["date", "KPI"]].copy()
df_kpi_monthly["date"] = pd.to_datetime(df_kpi_monthly["date"], errors="coerce")
df_kpi_monthly["KPI"] = pd.to_numeric(df_kpi_monthly["KPI"], errors="coerce")
df_kpi_monthly["month"] = df_kpi_monthly["date"].dt.to_period("M")
df_kpi_monthly = (
    df_kpi_monthly[["month", "KPI"]]
    .dropna()
    .drop_duplicates(subset=["month"])
    .sort_values("month")
    .reset_index(drop=True)
)
kpi_by_month = df_kpi_monthly.set_index("month")["KPI"]

# ---- PREPARE COMMON REAL DATA
def build_real_context(df_panel: pd.DataFrame,
                       df_real_deposit: pd.DataFrame,
                       eval_date: pd.Timestamp,
                       kpi_by_month: pd.Series):

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

    # 2: CONVERT NOMINAL LINKER PRICES TO REAL CLEAN PRICES
    ql_settlement_date = calendar.advance(ql_eval_date, settlement_days, ql.Days)
    settlement_date = pd.Timestamp(
        ql_settlement_date.year(),
        ql_settlement_date.month(),
        ql_settlement_date.dayOfMonth()
    )

    reference_kpi = swedish_reference_kpi(settlement_date, kpi_by_month, lag_months=3)

    cross_section["index_factor"] = reference_kpi / cross_section["bas_kpi"]
    # real/deflated price:
    cross_section["real_price"] = cross_section["price"] / cross_section["index_factor"]
    # real/deflated yield to maturiy: 
    real_ytm_list = []

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

        bond = ql.FixedRateBond(
            settlement_days,
            face,
            schedule,
            [float(row["coupon"])],
            day_count,
            business_convention,
            redemption,
            ql_issue
        )

        real_clean_price = float(row["real_price"])
        price_obj = ql.BondPrice(real_clean_price, ql.BondPrice.Clean)

        real_ytm = ql.BondFunctions.bondYield(
            bond,
            price_obj,
            day_count,
            ql.Compounded,
            ql.Annual,
            ql_settlement_date,
            1.0e-12,
            1000,
            0.02
        )

        real_ytm_list.append(float(real_ytm))

    cross_section["real_ytm"] = real_ytm_list   

    # 3: BUILD BOND HELPERS FOR ACTUAL BONDS
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
        clean_price = float(row["real_price"])

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
        quoted_yield.append(float(row["real_ytm"]))

    quoted_price_actual = np.array(quoted_price, dtype=float)
    quoted_yield_actual = np.array(quoted_yield, dtype=float)

    # 4: COMPUTE BOOTSTRAPPED REAL CURVE
    pre_curve = ql.PiecewiseLogCubicDiscount(settlement_days, calendar, helpers_actual, day_count)
    pre_curve.enableExtrapolation()
    curve_reference_date = calendar.advance(ql_eval_date, settlement_days, ql.Days)

    # 5: BUILD SYNTHETIC MIDPOINT ZERO-COUPON BONDS
    actual_maturity_dates = sorted({b.maturityDate() for b in bonds_actual})

    synthetic_helpers = []
    synthetic_bonds = []

    for d1, d2 in zip(actual_maturity_dates[:-1], actual_maturity_dates[1:]):
        t1 = day_count.yearFraction(curve_reference_date, d1)
        t2 = day_count.yearFraction(curve_reference_date, d2)

        if t1 <= 0.0 or t2 <= 0.0:
            continue

        t_mid = 0.5 * (t1 + t2)
        months_mid = int(round(t_mid * 12.0))
        mid_date = calendar.advance(curve_reference_date, ql.Period(months_mid, ql.Months))

        if mid_date <= curve_reference_date or mid_date >= d2:
            continue

        discount_factor_mid = pre_curve.discount(mid_date)
        synth_price = face * discount_factor_mid

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

    helpers_extended = helpers_actual + synthetic_helpers
    bonds_extended = bonds_actual + synthetic_bonds

    # 6: REAL DEPOSIT ANCHOR
    real_deposit_row = df_real_deposit.loc[df_real_deposit["date"] == eval_date]
    if real_deposit_row.empty:
        raise ValueError(f"No real deposit anchor found for evaluation date {eval_date}")

    real_deposit_rate = float(real_deposit_row["real_deposit"].iloc[0])
    real_deposit_rate_cc = np.log(1.0 + real_deposit_rate)

    # 7: LONG END ANCHOR FROM BOOTSTRAPPED REAL CURVE
    max_maturity = max(b.maturityDate() for b in bonds_actual)
    long_rate_cc = pre_curve.zeroRate(max_maturity, day_count, ql.Continuous).rate()

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

    return {
        "eval_date": eval_date,
        "ql_eval_date": ql_eval_date,
        "day_count": day_count,
        "tenor": tenor,
        "date_gen_rule": date_gen_rule,
        "calendar": calendar,
        "business_convention": business_convention,
        "end_of_month": end_of_month,
        "settlement_days": settlement_days,
        "face": face,
        "redemption": redemption,
        "helpers_actual": helpers_actual,
        "helpers_extended": helpers_extended,
        "bonds_actual": bonds_actual,
        "bonds_extended": bonds_extended,
        "quoted_price_actual": quoted_price_actual,
        "quoted_yield_actual": quoted_yield_actual,
        "pre_curve": pre_curve,
        "curve_reference_date": curve_reference_date,
        "synthetic_helpers": synthetic_helpers,
        "real_deposit_rate": real_deposit_rate,
        "real_deposit_rate_cc": real_deposit_rate_cc,
        "long_rate_cc": long_rate_cc,
        "reference_kpi": reference_kpi,
        "price_actual_bonds": price_actual_bonds,
    }

    # endregion

    # region ESTIMATE NSS FOR GIVEN DATE FUNCTION (SINGLE START)

def run_estimation_for_date_nss(df_panel: pd.DataFrame,
                                df_real_deposit: pd.DataFrame,
                                eval_date: pd.Timestamp):

    ctx = build_real_context(df_panel, df_real_deposit, eval_date, kpi_by_month)

    day_count = ctx["day_count"]
    calendar = ctx["calendar"]
    settlement_days = ctx["settlement_days"]
    helpers_actual = ctx["helpers_actual"]
    helpers_extended = ctx["helpers_extended"]
    quoted_price_actual = ctx["quoted_price_actual"]
    quoted_yield_actual = ctx["quoted_yield_actual"]
    price_actual_bonds = ctx["price_actual_bonds"]

    # INITIAL GUESS FROM SHORT + LONG ANCHORS
    b0 = ctx["long_rate_cc"]
    b1 = ctx["real_deposit_rate_cc"] - b0
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

    # BOUNDS
    b0_min, b0_max = -0.15, 0.15
    b1_min, b1_max = -0.15, 0.15
    b2_min, b2_max = -0.25, 0.25
    b3_min, b3_max = -0.25, 0.25

    tau_min, tau_max = 0.05, 30.0
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

    l2 = ql.Array(6)
    l2[0] = 0.4
    l2[1] = 0.4
    l2[2] = 0.06
    l2[3] = 0.09
    l2[4] = 0.02
    l2[5] = 0.03

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

    # FIT 1: ACTUAL ONLY
    curve_actual = ql.FittedBondDiscountCurve(
        settlement_days,
        calendar,
        helpers_actual,
        day_count,
        fitting,
        accuracy,
        max_evaluations,
        guess
    )
    params_actual = list(curve_actual.fitResults().solution())

    # FIT 2: EXTENDED
    curve_extended = ql.FittedBondDiscountCurve(
        settlement_days,
        calendar,
        helpers_extended,
        day_count,
        fitting,
        accuracy,
        max_evaluations,
        guess
    )
    params_extended = list(curve_extended.fitResults().solution())

    # PRICE ACTUAL BONDS AND COMPUTE RMSEs
    model_prices_actualfit, model_yields_actualfit = price_actual_bonds(curve_actual)
    model_prices_extendedfit, model_yields_extendedfit = price_actual_bonds(curve_extended)

    rmse_price_actualfit = float(np.sqrt(np.mean((quoted_price_actual - model_prices_actualfit) ** 2)))
    rmse_price_extfit = float(np.sqrt(np.mean((quoted_price_actual - model_prices_extendedfit) ** 2)))

    rmse_yield_actualfit_bp = float(np.sqrt(np.mean((quoted_yield_actual - model_yields_actualfit) ** 2)) * 1e4)
    rmse_yield_extfit_bp = float(np.sqrt(np.mean((quoted_yield_actual - model_yields_extendedfit) ** 2)) * 1e4)

    return {
        "date": eval_date,
        "b0_actual": params_actual[0], "b1_actual": params_actual[1], "b2_actual": params_actual[2],
        "b3_actual": params_actual[3], "k1_actual": params_actual[4], "k2_actual": params_actual[5],
        "b0_ext": params_extended[0], "b1_ext": params_extended[1], "b2_ext": params_extended[2],
        "b3_ext": params_extended[3], "k1_ext": params_extended[4], "k2_ext": params_extended[5],
        "rmse_price_actual": rmse_price_actualfit,
        "rmse_price_ext": rmse_price_extfit,
        "rmse_yield_bp_actual": rmse_yield_actualfit_bp,
        "rmse_yield_bp_ext": rmse_yield_extfit_bp,
        "n_bonds": len(helpers_actual),
        "n_synth": len(ctx["synthetic_helpers"]),
        "real_deposit": ctx["real_deposit_rate"],
        "reference_kpi": ctx["reference_kpi"],
    }

    # endregion

    # region ESTIMATE NSS FOR GIVEN DATE FUNCTION (MULTI START)

def run_estimation_for_date_multistart_nss(df_panel: pd.DataFrame,
                                           df_real_deposit: pd.DataFrame,
                                           eval_date: pd.Timestamp):

    ctx = build_real_context(df_panel, df_real_deposit, eval_date, kpi_by_month)

    day_count = ctx["day_count"]
    calendar = ctx["calendar"]
    settlement_days = ctx["settlement_days"]
    helpers_actual = ctx["helpers_actual"]
    helpers_extended = ctx["helpers_extended"]
    quoted_price_actual = ctx["quoted_price_actual"]
    quoted_yield_actual = ctx["quoted_yield_actual"]
    price_actual_bonds = ctx["price_actual_bonds"]

    b0 = ctx["long_rate_cc"]
    b1 = ctx["real_deposit_rate_cc"] - b0

    b0_min, b0_max = -0.15, 0.15
    b1_min, b1_max = -0.15, 0.15
    b2_min, b2_max = -0.25, 0.25
    b3_min, b3_max = -0.25, 0.25

    tau_min, tau_max = 0.05, 30.0
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

    l2 = ql.Array(6)
    l2[0] = 0.4
    l2[1] = 0.4
    l2[2] = 0.06
    l2[3] = 0.09
    l2[4] = 0.02
    l2[5] = 0.03

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

    def yield_rmse_bp_from_curve(curve):
        _, model_yields = price_actual_bonds(curve)
        return float(np.sqrt(np.mean((quoted_yield_actual - model_yields) ** 2)) * 1e4)

    tau1_grid = [0.5, 2.0, 4.0, 6.0, 8.0]
    tau2_grid = [5.0, 10.0, 15.0, 25.0, 30.0]

    best_rmse_bp_actual = float("inf")
    best_curve_actual = None
    best_params_actual = None
    best_start_actual = None

    for tau1 in tau1_grid:
        for tau2 in tau2_grid:
            if tau2 <= tau1:
                continue

            guess = ql.Array(6)
            guess[0] = b0
            guess[1] = b1
            guess[2] = 0.0
            guess[3] = 0.0
            guess[4] = 1.0 / tau1
            guess[5] = 1.0 / tau2

            try:
                curve_try = ql.FittedBondDiscountCurve(
                    settlement_days,
                    calendar,
                    helpers_actual,
                    day_count,
                    fitting,
                    accuracy,
                    max_evaluations,
                    guess
                )

                rmse_bp = yield_rmse_bp_from_curve(curve_try)

                if rmse_bp < best_rmse_bp_actual:
                    best_rmse_bp_actual = rmse_bp
                    best_curve_actual = curve_try
                    best_params_actual = list(curve_try.fitResults().solution())
                    best_start_actual = (tau1, tau2)

            except RuntimeError:
                continue

    if best_curve_actual is None:
        raise RuntimeError(f"NSS multistart failed for ACTUAL on {eval_date}")

    best_rmse_bp_ext = float("inf")
    best_curve_ext = None
    best_params_ext = None
    best_start_ext = None

    for tau1 in tau1_grid:
        for tau2 in tau2_grid:
            if tau2 <= tau1:
                continue

            guess = ql.Array(6)
            guess[0] = b0
            guess[1] = b1
            guess[2] = 0.0
            guess[3] = 0.0
            guess[4] = 1.0 / tau1
            guess[5] = 1.0 / tau2

            try:
                curve_try = ql.FittedBondDiscountCurve(
                    settlement_days,
                    calendar,
                    helpers_extended,
                    day_count,
                    fitting,
                    accuracy,
                    max_evaluations,
                    guess
                )

                rmse_bp = yield_rmse_bp_from_curve(curve_try)

                if rmse_bp < best_rmse_bp_ext:
                    best_rmse_bp_ext = rmse_bp
                    best_curve_ext = curve_try
                    best_params_ext = list(curve_try.fitResults().solution())
                    best_start_ext = (tau1, tau2)

            except RuntimeError:
                continue

    if best_curve_ext is None:
        raise RuntimeError(f"NSS multistart failed for EXTENDED on {eval_date}")

    model_prices_actualfit, model_yields_actualfit = price_actual_bonds(best_curve_actual)
    model_prices_extendedfit, model_yields_extendedfit = price_actual_bonds(best_curve_ext)

    rmse_price_actualfit = float(np.sqrt(np.mean((quoted_price_actual - model_prices_actualfit) ** 2)))
    rmse_price_extfit = float(np.sqrt(np.mean((quoted_price_actual - model_prices_extendedfit) ** 2)))

    rmse_yield_actualfit_bp = float(np.sqrt(np.mean((quoted_yield_actual - model_yields_actualfit) ** 2)) * 1e4)
    rmse_yield_extfit_bp = float(np.sqrt(np.mean((quoted_yield_actual - model_yields_extendedfit) ** 2)) * 1e4)

    return {
        "date": eval_date,
        "b0_actual": best_params_actual[0], "b1_actual": best_params_actual[1], "b2_actual": best_params_actual[2],
        "b3_actual": best_params_actual[3], "k1_actual": best_params_actual[4], "k2_actual": best_params_actual[5],
        "b0_ext": best_params_ext[0], "b1_ext": best_params_ext[1], "b2_ext": best_params_ext[2],
        "b3_ext": best_params_ext[3], "k1_ext": best_params_ext[4], "k2_ext": best_params_ext[5],
        "rmse_price_actual": rmse_price_actualfit,
        "rmse_price_ext": rmse_price_extfit,
        "rmse_yield_bp_actual": rmse_yield_actualfit_bp,
        "rmse_yield_bp_ext": rmse_yield_extfit_bp,
        "n_bonds": len(helpers_actual),
        "n_synth": len(ctx["synthetic_helpers"]),
        "real_deposit": ctx["real_deposit_rate"],
        "reference_kpi": ctx["reference_kpi"],
        "start_tau1_actual": best_start_actual[0],
        "start_tau2_actual": best_start_actual[1],
        "start_tau1_ext": best_start_ext[0],
        "start_tau2_ext": best_start_ext[1],
    }

    # endregion

    # region ESTIMATE NS FOR GIVEN DATE FUNCTION (SINGLE START)

def run_estimation_for_date_ns(df_panel: pd.DataFrame,
                               df_real_deposit: pd.DataFrame,
                               eval_date: pd.Timestamp):

    ctx = build_real_context(df_panel, df_real_deposit, eval_date, kpi_by_month)

    day_count = ctx["day_count"]
    calendar = ctx["calendar"]
    settlement_days = ctx["settlement_days"]
    helpers_actual = ctx["helpers_actual"]
    helpers_extended = ctx["helpers_extended"]
    quoted_price_actual = ctx["quoted_price_actual"]
    quoted_yield_actual = ctx["quoted_yield_actual"]
    price_actual_bonds = ctx["price_actual_bonds"]

    # INITIAL GUESS FROM SHORT + LONG ANCHORS
    b0 = ctx["long_rate_cc"]
    b1 = ctx["real_deposit_rate_cc"] - b0
    b2 = 0.0

    tau = 2.0
    kappa = 1.0 / tau

    guess = ql.Array(4)
    guess[0] = b0
    guess[1] = b1
    guess[2] = b2
    guess[3] = kappa

    # BOUNDS
    b0_min, b0_max = -0.15, 0.15
    b1_min, b1_max = -0.15, 0.15
    b2_min, b2_max = -0.25, 0.25

    tau_min, tau_max = 0.1, 30.0
    kappa_min, kappa_max = 1.0 / tau_max, 1.0 / tau_min

    lower = ql.Array(4)
    upper = ql.Array(4)

    lower[0], upper[0] = b0_min, b0_max
    lower[1], upper[1] = b1_min, b1_max
    lower[2], upper[2] = b2_min, b2_max
    lower[3], upper[3] = kappa_min, kappa_max

    constraint = ql.NonhomogeneousBoundaryConstraint(lower, upper)

    l2 = ql.Array(4)
    l2[0] = 0.3
    l2[1] = 0.3
    l2[2] = 0.04
    l2[3] = 0.02

    accuracy = 1e-10
    max_evaluations = 15000
    simplex_lambda = 0.005

    fitting = ql.NelsonSiegelFitting(
        ql.Array(),
        ql.Simplex(simplex_lambda),
        l2,
        0.0,
        50.0,
        constraint
    )

    # FIT 1: ACTUAL ONLY
    curve_actual = ql.FittedBondDiscountCurve(
        settlement_days,
        calendar,
        helpers_actual,
        day_count,
        fitting,
        accuracy,
        max_evaluations,
        guess
    )
    params_actual = list(curve_actual.fitResults().solution())

    # FIT 2: EXTENDED
    curve_extended = ql.FittedBondDiscountCurve(
        settlement_days,
        calendar,
        helpers_extended,
        day_count,
        fitting,
        accuracy,
        max_evaluations,
        guess
    )
    params_extended = list(curve_extended.fitResults().solution())

    # PRICE ACTUAL BONDS AND COMPUTE RMSEs
    model_prices_actualfit, model_yields_actualfit = price_actual_bonds(curve_actual)
    model_prices_extendedfit, model_yields_extendedfit = price_actual_bonds(curve_extended)

    rmse_price_actualfit = float(np.sqrt(np.mean((quoted_price_actual - model_prices_actualfit) ** 2)))
    rmse_price_extfit = float(np.sqrt(np.mean((quoted_price_actual - model_prices_extendedfit) ** 2)))

    rmse_yield_actualfit_bp = float(np.sqrt(np.mean((quoted_yield_actual - model_yields_actualfit) ** 2)) * 1e4)
    rmse_yield_extfit_bp = float(np.sqrt(np.mean((quoted_yield_actual - model_yields_extendedfit) ** 2)) * 1e4)

    return {
        "date": eval_date,
        "b0_actual": params_actual[0], "b1_actual": params_actual[1], "b2_actual": params_actual[2],
        "k1_actual": params_actual[3],
        "b0_ext": params_extended[0], "b1_ext": params_extended[1], "b2_ext": params_extended[2],
        "k1_ext": params_extended[3],
        "rmse_price_actual": rmse_price_actualfit,
        "rmse_price_ext": rmse_price_extfit,
        "rmse_yield_bp_actual": rmse_yield_actualfit_bp,
        "rmse_yield_bp_ext": rmse_yield_extfit_bp,
        "n_bonds": len(helpers_actual),
        "n_synth": len(ctx["synthetic_helpers"]),
        "real_deposit": ctx["real_deposit_rate"],
        "reference_kpi": ctx["reference_kpi"],
    }

    # endregion

    # region ESTIMATE NS FOR GIVEN DATE FUNCTION (MULTI START)

def run_estimation_for_date_multistart_ns(df_panel: pd.DataFrame,
                                          df_real_deposit: pd.DataFrame,
                                          eval_date: pd.Timestamp):

    ctx = build_real_context(df_panel, df_real_deposit, eval_date, kpi_by_month)

    day_count = ctx["day_count"]
    calendar = ctx["calendar"]
    settlement_days = ctx["settlement_days"]
    helpers_actual = ctx["helpers_actual"]
    helpers_extended = ctx["helpers_extended"]
    quoted_price_actual = ctx["quoted_price_actual"]
    quoted_yield_actual = ctx["quoted_yield_actual"]
    price_actual_bonds = ctx["price_actual_bonds"]

    b0 = ctx["long_rate_cc"]
    b1 = ctx["real_deposit_rate_cc"] - b0

    b0_min, b0_max = -0.15, 0.15
    b1_min, b1_max = -0.15, 0.15
    b2_min, b2_max = -0.25, 0.25

    tau_min, tau_max = 0.1, 30.0
    kappa_min, kappa_max = 1.0 / tau_max, 1.0 / tau_min

    lower = ql.Array(4)
    upper = ql.Array(4)

    lower[0], upper[0] = b0_min, b0_max
    lower[1], upper[1] = b1_min, b1_max
    lower[2], upper[2] = b2_min, b2_max
    lower[3], upper[3] = kappa_min, kappa_max

    constraint = ql.NonhomogeneousBoundaryConstraint(lower, upper)

    l2 = ql.Array(4)
    l2[0] = 0.3
    l2[1] = 0.3
    l2[2] = 0.04
    l2[3] = 0.02

    accuracy = 1e-10
    max_evaluations = 15000
    simplex_lambda = 0.005

    fitting = ql.NelsonSiegelFitting(
        ql.Array(),
        ql.Simplex(simplex_lambda),
        l2,
        0.0,
        50.0,
        constraint
    )

    def yield_rmse_bp_from_curve(curve):
        _, model_yields = price_actual_bonds(curve)
        return float(np.sqrt(np.mean((quoted_yield_actual - model_yields) ** 2)) * 1e4)

    tau_grid = [0.5, 1.0, 1.5, 2.5, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]

    best_rmse_bp_actual = float("inf")
    best_curve_actual = None
    best_params_actual = None
    best_start_actual = None

    for tau in tau_grid:
        guess = ql.Array(4)
        guess[0] = b0
        guess[1] = b1
        guess[2] = 0.0
        guess[3] = 1.0 / tau

        try:
            curve_try = ql.FittedBondDiscountCurve(
                settlement_days,
                calendar,
                helpers_actual,
                day_count,
                fitting,
                accuracy,
                max_evaluations,
                guess
            )

            rmse_bp = yield_rmse_bp_from_curve(curve_try)

            if rmse_bp < best_rmse_bp_actual:
                best_rmse_bp_actual = rmse_bp
                best_curve_actual = curve_try
                best_params_actual = list(curve_try.fitResults().solution())
                best_start_actual = tau

        except RuntimeError:
            continue

    if best_curve_actual is None:
        raise RuntimeError(f"NS multistart failed for ACTUAL on {eval_date}")

    best_rmse_bp_ext = float("inf")
    best_curve_ext = None
    best_params_ext = None
    best_start_ext = None

    for tau in tau_grid:
        guess = ql.Array(4)
        guess[0] = b0
        guess[1] = b1
        guess[2] = 0.0
        guess[3] = 1.0 / tau

        try:
            curve_try = ql.FittedBondDiscountCurve(
                settlement_days,
                calendar,
                helpers_extended,
                day_count,
                fitting,
                accuracy,
                max_evaluations,
                guess
            )

            rmse_bp = yield_rmse_bp_from_curve(curve_try)

            if rmse_bp < best_rmse_bp_ext:
                best_rmse_bp_ext = rmse_bp
                best_curve_ext = curve_try
                best_params_ext = list(curve_try.fitResults().solution())
                best_start_ext = tau

        except RuntimeError:
            continue

    if best_curve_ext is None:
        raise RuntimeError(f"NS multistart failed for EXTENDED on {eval_date}")

    model_prices_actualfit, model_yields_actualfit = price_actual_bonds(best_curve_actual)
    model_prices_extendedfit, model_yields_extendedfit = price_actual_bonds(best_curve_ext)

    rmse_price_actualfit = float(np.sqrt(np.mean((quoted_price_actual - model_prices_actualfit) ** 2)))
    rmse_price_extfit = float(np.sqrt(np.mean((quoted_price_actual - model_prices_extendedfit) ** 2)))

    rmse_yield_actualfit_bp = float(np.sqrt(np.mean((quoted_yield_actual - model_yields_actualfit) ** 2)) * 1e4)
    rmse_yield_extfit_bp = float(np.sqrt(np.mean((quoted_yield_actual - model_yields_extendedfit) ** 2)) * 1e4)

    return {
        "date": eval_date,
        "b0_actual": best_params_actual[0], "b1_actual": best_params_actual[1], "b2_actual": best_params_actual[2],
        "k1_actual": best_params_actual[3],
        "b0_ext": best_params_ext[0], "b1_ext": best_params_ext[1], "b2_ext": best_params_ext[2],
        "k1_ext": best_params_ext[3],
        "rmse_price_actual": rmse_price_actualfit,
        "rmse_price_ext": rmse_price_extfit,
        "rmse_yield_bp_actual": rmse_yield_actualfit_bp,
        "rmse_yield_bp_ext": rmse_yield_extfit_bp,
        "n_bonds": len(helpers_actual),
        "n_synth": len(ctx["synthetic_helpers"]),
        "real_deposit": ctx["real_deposit_rate"],
        "reference_kpi": ctx["reference_kpi"],
        "start_tau_actual": best_start_actual,
        "start_tau_ext": best_start_ext,
    }

    # endregion

    # region FUNCTIONS FOR RUNNING PANEL, EXPORTING TO EXCEL & SUMMARY STATISTICS

# ---- PANEL RUNNER FUNCTION
def run_panel_by_method(df_panel: pd.DataFrame,
                        df_real_deposit: pd.DataFrame,
                        estimation_function):

    month_to_date = (
        df_panel.assign(month=df_panel["date"].dt.to_period("M"))
            .groupby("month")["date"].max()
            .sort_index()
    )

    rows = []
    failed = []

    for month, d in month_to_date.items():
        eval_date = pd.Timestamp(d)
        try:
            out = estimation_function(df_panel, df_real_deposit, eval_date)
            out["month"] = str(month)
            rows.append(out)
        except Exception as e:
            failed.append({"month": str(month), "date": eval_date, "error": str(e)})

    results = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    failed = pd.DataFrame(failed)

    print("Successful months:", len(results))
    print("Failed months:", len(failed))

    return results, failed

# ---- EXPORT TO EXCEL FUNCTION
def export_panel_results_to_excel(results_dict: dict,
                                  failed_dict: dict,
                                  output_path: str):

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, df in results_dict.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)

        for sheet_name, df in failed_dict.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)

# ---- SUMMARY STATISTICS FUNCTION
def print_panel_summary_statistics(results: pd.DataFrame):

    results = results.copy()

    if "k1_actual" in results.columns:
        results["tau1_actual"] = 1.0 / results["k1_actual"]
    if "k1_ext" in results.columns:
        results["tau1_ext"] = 1.0 / results["k1_ext"]

    if "k2_actual" in results.columns:
        results["tau2_actual"] = 1.0 / results["k2_actual"]
    if "k2_ext" in results.columns:
        results["tau2_ext"] = 1.0 / results["k2_ext"]

    panel_cols = []

    if "b0_actual" in results.columns: panel_cols.append("b0_actual")
    if "b1_actual" in results.columns: panel_cols.append("b1_actual")
    if "b2_actual" in results.columns: panel_cols.append("b2_actual")
    if "b3_actual" in results.columns: panel_cols.append("b3_actual")
    if "k1_actual" in results.columns: panel_cols.append("k1_actual")
    if "k2_actual" in results.columns: panel_cols.append("k2_actual")

    if "b0_ext" in results.columns: panel_cols.append("b0_ext")
    if "b1_ext" in results.columns: panel_cols.append("b1_ext")
    if "b2_ext" in results.columns: panel_cols.append("b2_ext")
    if "b3_ext" in results.columns: panel_cols.append("b3_ext")
    if "k1_ext" in results.columns: panel_cols.append("k1_ext")
    if "k2_ext" in results.columns: panel_cols.append("k2_ext")

    if "tau1_actual" in results.columns: panel_cols.append("tau1_actual")
    if "tau2_actual" in results.columns: panel_cols.append("tau2_actual")
    if "tau1_ext" in results.columns: panel_cols.append("tau1_ext")
    if "tau2_ext" in results.columns: panel_cols.append("tau2_ext")

    panel_cols += [
        "rmse_price_actual",
        "rmse_price_ext",
        "rmse_yield_bp_actual",
        "rmse_yield_bp_ext",
        "n_bonds",
        "n_synth",
        "real_deposit",
        "reference_kpi",
    ]

    print("Panel summary statistics")

    for c in panel_cols:
        avg_val = results[c].mean()
        min_val = results[c].min()
        max_val = results[c].max()

        print(f"{c:20s}  avg={avg_val: .6f}  min={min_val: .6f}  max={max_val: .6f}")

    # endregion

    # region RUN 

# ---- RUN SINGLE-START NSS
results_nss_ss, failed_nss_ss = run_panel_by_method(
    df_SGBILs,
    df_real_deposit,
    run_estimation_for_date_nss
)

# ---- RUN SINGLE-START NS
results_ns_ss, failed_ns_ss = run_panel_by_method(
    df_SGBILs,
    df_real_deposit,
    run_estimation_for_date_ns
)

# ---- RUN MULTISTART NSS
results_nss_ms, failed_nss_ms = run_panel_by_method(
    df_SGBILs,
    df_real_deposit,
    run_estimation_for_date_multistart_nss
)

# ---- RUN MULTISTART NS
results_ns_ms, failed_ns_ms = run_panel_by_method(
    df_SGBILs,
    df_real_deposit,
    run_estimation_for_date_multistart_ns
)

# ---- EXPORT ALL RESULTS TO ONE EXCEL FILE WITH SEPARATE SHEETS
export_panel_results_to_excel(
    results_dict={
        "nss_single_start": results_nss_ss,
        "ns_single_start": results_ns_ss,
        "nss_multistart": results_nss_ms,
        "ns_multistart": results_ns_ms,
    },
    failed_dict={
        "nss_ss_failed": failed_nss_ss,
        "ns_ss_failed": failed_ns_ss,
        "nss_ms_failed": failed_nss_ms,
        "ns_ms_failed": failed_ns_ms,
    },
    output_path="inflation_linked_panel_results.xlsx"
)

# ---- RUN SUMMARY STATISTICS SEPARATELY FOR EACH METHOD
print_panel_summary_statistics(results_nss_ss)
print_panel_summary_statistics(results_ns_ss)
print_panel_summary_statistics(results_nss_ms)
print_panel_summary_statistics(results_ns_ms)

    # endregion

    # region PANEL PLOTS

# ---- TERM STRUCTURE TIME SERIES: NSS MULTISTART
results = results_nss_ms
param_set = "ext"    # can change to "actual"

tenors = {
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

    term1 = np.where(np.abs(x1) < 1e-12, 1.0, (1.0 - np.exp(-x1)) / x1)
    term2 = term1 - np.exp(-x1)
    term3 = np.where(np.abs(x2) < 1e-12, 0.0, (1.0 - np.exp(-x2)) / x2 - np.exp(-x2))

    zero_ts[label] = b0 + b1 * term1 + b2 * term2 + b3 * term3

# Convert to percent for plotting
plot_df = zero_ts.copy()
for c in tenors.keys():
    plot_df[c] = 100.0 * plot_df[c]

plt.figure(figsize=(11, 6))
for c in tenors.keys():
    plt.plot(plot_df["date"], plot_df[c], label=c)

plt.xlabel("Date")
plt.ylabel("Real zero-coupon yield (%)")
plt.title("NSS multistart zero yields by tenor")
plt.legend(ncol=5)
plt.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.7)
plt.tight_layout()
plt.show()

# ---- PLOT RMSEs: NSS MULTISTART

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
ax2.set_ylabel("Price RMSE (real clean price points)")

# One combined legend
lines = ax1.get_lines() + ax2.get_lines()
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc="upper right")

plt.title("NSS multistart: RMSE of yields (bp) and prices (dual axis)")
plt.tight_layout()
plt.show()

# ---- TERM STRUCTURE TIME SERIES: NS MULTISTART
results = results_ns_ms
param_set = "ext"    # can change to "actual"

tenors = {
    "2y": 2.0,
    "4y": 4.0,
    "6y": 6.0,
    "8y": 8.0,
    "10y": 10.0,
}

b0 = results[f"b0_{param_set}"].to_numpy()
b1 = results[f"b1_{param_set}"].to_numpy()
b2 = results[f"b2_{param_set}"].to_numpy()
k1 = results[f"k1_{param_set}"].to_numpy()

dates = pd.to_datetime(results["date"])

zero_ts = pd.DataFrame({"date": dates})

for label, t in tenors.items():
    t = float(t)

    x1 = k1 * t

    term1 = np.where(np.abs(x1) < 1e-12, 1.0, (1.0 - np.exp(-x1)) / x1)
    term2 = term1 - np.exp(-x1)

    zero_ts[label] = b0 + b1 * term1 + b2 * term2

# Convert to percent for plotting
plot_df = zero_ts.copy()
for c in tenors.keys():
    plot_df[c] = 100.0 * plot_df[c]

plt.figure(figsize=(11, 6))
for c in tenors.keys():
    plt.plot(plot_df["date"], plot_df[c], label=c)

plt.xlabel("Date")
plt.ylabel("Real zero-coupon yield (%)")
plt.title("NS multistart zero yields by tenor")
plt.legend(ncol=5)
plt.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.7)
plt.tight_layout()
plt.show()

# ---- PLOT RMSEs: NS MULTISTART

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
ax2.set_ylabel("Price RMSE (real clean price points)")

# One combined legend
lines = ax1.get_lines() + ax2.get_lines()
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc="upper right")

plt.title("NS multistart: RMSE of yields (bp) and prices (dual axis)")
plt.tight_layout()
plt.show()

    # endregion

# endregion
