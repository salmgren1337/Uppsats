import pandas as pd
import QuantLib as ql
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# IMPORTS 
raw_data_linked = pd.read_excel("statsobligationer_data.xlsx", sheet_name = "SGBi long")
short_rates = pd.read_excel("statsobligationer_data.xlsx", sheet_name = "korta räntor riksbanken")
kpi = pd.read_excel("kpi_data.xlsx", sheet_name = "basår 1980")
kpi_exp = pd.read_excel("CPI_Inflation_Expectations.xlsx", sheet_name = "CPI Expectations")

START = pd.Timestamp("2004-01-30")
END = pd.Timestamp("2025-12-31")

MAX_MAT = 30
MIN_MAT = 1.5

#HELPER_PRICE_MODE = "standard_real"
HELPER_PRICE_MODE = "lag_adjusted"
ENTER_DATE = "2025-12-31"

# region FIXED SETUP OPTIONS

B0_MIN, B0_MAX = -0.03, 0.06
B1_MIN, B1_MAX = -0.15, 0.15
B2_MIN, B2_MAX = -0.05, 0.05

TAU_MIN, TAU_MAX = 1.5, 8.0

L2_0 = 10.0
L2_1 = 10.0
L2_2 = 0.2
L2_3 = 0.05

ACCURACY = 1.0e-10
MAX_EVALUTATIONS = 10000
SIMPLEX_LAMBDA = 0.005

B2_GRID = [-0.02, -0.01, 0.0, 0.01, 0.02]
TAU_GRID = [1.5, 2.0, 3.0, 4.5, 6.0]

# endregion

# region PREPARE BOND DATA

# ---- 1) PREPARE SGB IL DATA

# Fill missing base CPI for early linkers using series 3001
base_kpi_3001 = raw_data_linked.loc[raw_data_linked["Serie"] == 3001, "BasKPI"].iloc[0]
raw_data_linked.loc[raw_data_linked["Serie"].isin([3002, 3003]), "BasKPI"] = base_kpi_3001

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
    "BasKPI",
]

df_SGBILs = raw_data_linked.loc[:, keep_cols].copy()

# Value formatting:
df_SGBILs["Date"] = pd.to_datetime(df_SGBILs["Date"], errors="coerce")
df_SGBILs["Issue date"] = pd.to_datetime(df_SGBILs["Issue date"], errors="coerce")
df_SGBILs["Maturity date"] = pd.to_datetime(df_SGBILs["Maturity date"], errors="coerce")

for col in ["PX_LAST", "YLD_YTM_MID", "BID_YIELD", "ASK_YIELD", "Kupong", "BasKPI"]:
    df_SGBILs[col] = pd.to_numeric(df_SGBILs[col], errors="coerce")

df_SGBILs["YLD_YTM_MID"] = df_SGBILs["YLD_YTM_MID"] / 100.0
df_SGBILs["BID_YIELD"] = df_SGBILs["BID_YIELD"] / 100.0
df_SGBILs["ASK_YIELD"] = df_SGBILs["ASK_YIELD"] / 100.0
df_SGBILs["Kupong"] = df_SGBILs["Kupong"] / 100.0

# Rename columns:
df_SGBILs = df_SGBILs.rename(columns={
    "Date": "date",
    "PX_LAST": "price",
    "YLD_YTM_MID": "yield_quoted",
    "BID_YIELD": "bid_yield_quoted",
    "ASK_YIELD": "ask_yield_quoted",
    "Kupong": "coupon",
    "Issue date": "issue_date",
    "Maturity date": "maturity_date",
    "ISIN": "isin",
    "Serie": "serie",
    "BasKPI": "bas_kpi",
})

# Set sample period and sort by date and maturity date:
df_SGBILs = df_SGBILs.loc[(df_SGBILs["date"] >= START) & (df_SGBILs["date"] <= END)].copy()
df_SGBILs = df_SGBILs.sort_values(["date", "maturity_date"]).reset_index(drop=True)

# Remove illiquid bond with duplicate maturity
df_SGBILs = df_SGBILs.loc[df_SGBILs["serie"] != 3103].copy()

# Keep only bonds in the real curve estimation range
time_to_maturity = (df_SGBILs["maturity_date"] - df_SGBILs["date"]).dt.days
df_SGBILs = df_SGBILs.loc[
    (time_to_maturity >= MIN_MAT * 365.25) &
    (time_to_maturity <= MAX_MAT * 365.25)
].reset_index(drop=True)

# ---- 2) ALIGN RAW DATA TO SGB IL LAST TRADING DAY

df_SGBILs["month"] = df_SGBILs["date"].dt.to_period("M")
month_trading_date = df_SGBILs.groupby("month")["date"].max()

# ---- 3) PREPARE DEPOSIT RATE AND T-BILL DATA

# Deposit rate
df_deposit = short_rates.loc[:, ["date", "deposit"]].copy()
df_deposit["date"] = pd.to_datetime(df_deposit["date"], errors="coerce")
df_deposit["deposit"] = pd.to_numeric(df_deposit["deposit"], errors="coerce") / 100.0
df_deposit["month"] = df_deposit["date"].dt.to_period("M")

# Map deposit rate EOM to SGB IL last trading day
df_deposit["date"] = df_deposit["month"].map(month_trading_date)
df_deposit = (
    df_deposit.drop(columns=["month"])
    .dropna(subset=["date", "deposit"])
    .drop_duplicates(subset=["date"])
    .sort_values("date")
    .reset_index(drop=True)
)

# 1m nominal zero yield from nominal estimation
nominal_zero = pd.read_excel("zero_yields_SGB.xlsx", sheet_name="zero_yields")
nominal_zero["date"] = pd.to_datetime(nominal_zero["date"], errors="coerce")
nominal_zero["y_1m"] = pd.to_numeric(nominal_zero["y_1m"], errors="coerce")

df_tbill = nominal_zero.loc[:, ["date", "y_1m"]].copy()
df_tbill = df_tbill.rename(columns={"y_1m": "yield_quoted"})
df_tbill["serie"] = "1m_zero"
df_tbill["coupon"] = 0.0
df_tbill["issue_date"] = df_tbill["date"]
df_tbill["maturity_date"] = df_tbill["issue_date"] + pd.to_timedelta(30, unit="D")
df_tbill["isin"] = ""
df_tbill["bas_kpi"] = np.nan

# Zero-coupon price from quoted annualized yield
days = (df_tbill["maturity_date"] - df_tbill["issue_date"]).dt.days
t = days / 360.0
df_tbill["price"] = 100.0 / (1.0 + df_tbill["yield_quoted"]) ** t

# Align 1m zero yield date to SGB IL last trading day
df_tbill["month"] = df_tbill["date"].dt.to_period("M")
df_tbill["date"] = df_tbill["month"].map(month_trading_date)
df_tbill["issue_date"] = df_tbill["date"]

df_tbill = (
    df_tbill.drop(columns=["month"])
    .dropna(subset=["date", "price", "yield_quoted", "issue_date", "maturity_date"])
    .sort_values(["date", "maturity_date"])
    .reset_index(drop=True)
)

# ---- 4) PREPARE CPI AND CPI EXPECTATIONS

# CPI inflation
df_cpi = kpi.loc[:, ["date", "inflation year"]].copy()
df_cpi["date"] = pd.to_datetime(df_cpi["date"], errors="coerce")
df_cpi["inflation year"] = pd.to_numeric(df_cpi["inflation year"], errors="coerce")
df_cpi["month"] = df_cpi["date"].dt.to_period("M")

# Map CPI data to SGB IL last trading day
df_cpi["date"] = df_cpi["month"].map(month_trading_date)
df_cpi = (
    df_cpi.drop(columns=["month"])
    .dropna(subset=["date"])
    .drop_duplicates(subset=["date"])
    .sort_values("date")
    .reset_index(drop=True)
)

# Survey CPI expectations
df_cpi_exp = kpi_exp.loc[:, ["Month", "1 Year", "2 Year", "5 Year"]].copy()
df_cpi_exp["Month"] = pd.to_datetime(df_cpi_exp["Month"], format="%b-%Y", errors="coerce")

for col in ["1 Year", "2 Year", "5 Year"]:
    df_cpi_exp[col] = pd.to_numeric(df_cpi_exp[col], errors="coerce") / 100.0

df_cpi_exp["month"] = df_cpi_exp["Month"].dt.to_period("M")

df_cpi_exp = (
    df_cpi_exp.loc[:, ["month", "1 Year", "2 Year", "5 Year"]]
    .dropna(subset=["month"])
    .drop_duplicates(subset=["month"])
    .sort_values("month")
    .reset_index(drop=True)
)

# Build a complete monthly grid and assign each month the nearest survey observation
all_months = pd.period_range(
    start=df_cpi_exp["month"].min(),
    end=df_cpi_exp["month"].max(),
    freq="M"
)

df_cpi_exp_monthly = pd.DataFrame({"month": all_months})
df_cpi_exp_monthly["month_ts"] = df_cpi_exp_monthly["month"].dt.to_timestamp()
df_cpi_exp["month_ts"] = df_cpi_exp["month"].dt.to_timestamp()

df_cpi_exp_monthly = pd.merge_asof(
    df_cpi_exp_monthly.sort_values("month_ts"),
    df_cpi_exp.sort_values("month_ts"),
    on="month_ts",
    direction="nearest"
)

df_cpi_exp_monthly["month"] = df_cpi_exp_monthly["month_x"]
df_cpi_exp_monthly = (
    df_cpi_exp_monthly.drop(columns=["month_x", "month_y", "month_ts"])
    .rename(columns={
        "1 Year": "cpi_exp_1y",
        "2 Year": "cpi_exp_2y",
        "5 Year": "cpi_exp_5y",
    })
)

# Map monthly expectations to SGB IL last trading day
df_cpi_exp_monthly["date"] = df_cpi_exp_monthly["month"].map(month_trading_date)

df_cpi_exp = (
    df_cpi_exp_monthly
    .drop(columns=["month"])
    .dropna(subset=["date"])
    .drop_duplicates(subset=["date"])
    .sort_values("date")
    .reset_index(drop=True)
)

# Keep CPI and CPI expectations in one dataframe
df_cpi = (
    pd.merge(df_cpi, df_cpi_exp, on="date", how="left")
    .drop_duplicates(subset=["date"])
    .sort_values("date")
    .reset_index(drop=True)
)

# CPI HELPERS:

def backward_1y_inflation(inflation_year: pd.Series) -> pd.Series:
    return inflation_year


def forward_1y_inflation(inflation_year: pd.Series) -> pd.Series:
    return inflation_year.shift(-12)


def real_rate_from_1y_inflation(y_nominal: pd.Series, inflation_1y: pd.Series) -> pd.Series:
    return y_nominal - inflation_1y

    #region 5) PLOT SHORT-RATE ANCHOR ALTERNATIVES

# Keep one macro dataframe for inflation and expectations
df_anchor_plot = df_cpi.copy()

# Add deposit rate
df_anchor_plot = pd.merge(
    df_anchor_plot,
    df_deposit,
    on="date",
    how="left"
)

# Add 1m nominal zero
df_tbill_plot = (
    df_tbill.loc[:, ["date", "yield_quoted"]]
    .drop_duplicates(subset=["date"])
    .rename(columns={"yield_quoted": "y_1m"})
)

df_anchor_plot = pd.merge(
    df_anchor_plot,
    df_tbill_plot,
    on="date",
    how="left"
)

df_anchor_plot = df_anchor_plot.sort_values("date").reset_index(drop=True)

# Inflation measures
df_anchor_plot["infl_1y_back"] = backward_1y_inflation(df_anchor_plot["inflation year"])
df_anchor_plot["infl_1y_fwd"] = forward_1y_inflation(df_anchor_plot["inflation year"])
df_anchor_plot["infl_1y_exp"] = df_anchor_plot["cpi_exp_1y"]

# Deposit: synthetic real rates
df_anchor_plot["deposit_real_back"] = real_rate_from_1y_inflation(
    df_anchor_plot["deposit"],
    df_anchor_plot["infl_1y_back"]
)

df_anchor_plot["deposit_real_fwd"] = real_rate_from_1y_inflation(
    df_anchor_plot["deposit"],
    df_anchor_plot["infl_1y_fwd"]
)

df_anchor_plot["deposit_real_exp"] = real_rate_from_1y_inflation(
    df_anchor_plot["deposit"],
    df_anchor_plot["infl_1y_exp"]
)

# 1m zero: synthetic real rates
df_anchor_plot["y1m_real_back"] = real_rate_from_1y_inflation(
    df_anchor_plot["y_1m"],
    df_anchor_plot["infl_1y_back"]
)

df_anchor_plot["y1m_real_fwd"] = real_rate_from_1y_inflation(
    df_anchor_plot["y_1m"],
    df_anchor_plot["infl_1y_fwd"]
)

df_anchor_plot["y1m_real_exp"] = real_rate_from_1y_inflation(
    df_anchor_plot["y_1m"],
    df_anchor_plot["infl_1y_exp"]
)

# ---- PLOT SETTINGS

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

color_deposit = "#3A3A3A"
color_1m = "#2CA02C" 

fig, ax = plt.subplots(figsize=(8.2, 4.8))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# Deposit
ax.plot(
    df_anchor_plot["date"],
    100.0 * df_anchor_plot["deposit_real_back"],
    color=color_deposit,
    linewidth=1.7,
    linestyle="-",
    label="Deposit - 1y backward"
)

ax.plot(
    df_anchor_plot["date"],
    100.0 * df_anchor_plot["deposit_real_fwd"],
    color=color_deposit,
    linewidth=1.7,
    linestyle="-.",
    label="Deposit - 1y forward"
)

ax.plot(
    df_anchor_plot["date"],
    100.0 * df_anchor_plot["deposit_real_exp"],
    color=color_deposit,
    linewidth=2.2,
    linestyle=":",
    label="Deposit - 1y expected"
)

# 1m zero
ax.plot(
    df_anchor_plot["date"],
    100.0 * df_anchor_plot["y1m_real_back"],
    color=color_1m,
    linewidth=1.7,
    linestyle="-",
    label="1m zero - 1y backward"
)

ax.plot(
    df_anchor_plot["date"],
    100.0 * df_anchor_plot["y1m_real_fwd"],
    color=color_1m,
    linewidth=1.7,
    linestyle="-.",
    label="1m zero - 1y forward"
)

ax.plot(
    df_anchor_plot["date"],
    100.0 * df_anchor_plot["y1m_real_exp"],
    color=color_1m,
    linewidth=2.2,
    linestyle=":",
    label="1m zero - 1y expected"
)

ax.axhline(0.0, color="black", linestyle=":", linewidth=1.0)

ax.set_xlabel("Date")
ax.set_ylabel("Real rate (%)")
ax.set_title("Real short rate alternatives", pad=12)

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
ax.margins(x=0)

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

# ---- 6) BUILD 1M EXPECTATION-ADJUSTED START ANCHOR SERIES

df_anchor_1m_exp = pd.merge(
    df_tbill.loc[:, ["date", "yield_quoted"]],
    df_cpi.loc[:, ["date", "cpi_exp_1y"]],
    on="date",
    how="left"
)

df_anchor_1m_exp["real_1m_exp"] = (
    df_anchor_1m_exp["yield_quoted"] - df_anchor_1m_exp["cpi_exp_1y"]
)

df_anchor_1m_exp = (
    df_anchor_1m_exp.loc[:, ["date", "real_1m_exp"]]
    .dropna(subset=["date", "real_1m_exp"])
    .drop_duplicates(subset=["date"])
    .sort_values("date")
    .reset_index(drop=True)
)

df_SGBILs = df_SGBILs.drop(columns=["month"], errors="ignore")

df_SGBILs = df_SGBILs.drop(columns=["month"], errors="ignore")

# ---- 7) BUILD MONTHLY KPI LOOKUP FOR CROSS SECTION AND PANEL

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

def swedish_reference_kpi(
    settlement_date: pd.Timestamp,
    kpi_by_month: pd.Series,
    lag_months: int = 3
) -> float:

    current_month = settlement_date.to_period("M")
    ref_month_0 = current_month - lag_months
    ref_month_1 = current_month - (lag_months - 1)

    if ref_month_0 not in kpi_by_month.index or ref_month_1 not in kpi_by_month.index:
        raise ValueError(f"Missing KPI needed for interpolation: {ref_month_0}, {ref_month_1}")

    kpi_0 = float(kpi_by_month.loc[ref_month_0])
    kpi_1 = float(kpi_by_month.loc[ref_month_1])

    day = settlement_date.day
    if day == 31:
        day = 30

    return kpi_0 + (day - 1) / 30.0 * (kpi_1 - kpi_0)


# ---- FUNCTION: PREPARE HELPER INPUTS

def eval_month_cpi(
    eval_date: pd.Timestamp,
    kpi_by_month: pd.Series,
) -> float:
    """
    Monthly CPI for the eval-date month.
    Example: if eval_date is 2025-12-31, use CPI for 2025-12.
    """
    cpi_month = eval_date.to_period("M")

    if cpi_month not in kpi_by_month.index:
        raise ValueError(f"Missing CPI for eval-date month: {cpi_month}")

    return float(kpi_by_month.loc[cpi_month])


def prepare_helper_prices(
    cross_section: pd.DataFrame,
    eval_date: pd.Timestamp,
    settlement_date: pd.Timestamp,
    kpi_by_month: pd.Series,
    helper_price_mode: str = "standard_real",
) -> tuple[pd.DataFrame, float, float, str]:

    cs = cross_section.copy()

    # Standard Swedish market-convention reference KPI:
    # 3-month lag with linear interpolation at settlement date
    reference_kpi = swedish_reference_kpi(
        settlement_date,
        kpi_by_month,
        lag_months=3
    )

    # CPI of the eval-date month
    eval_month_kpi = eval_month_cpi(
        eval_date,
        kpi_by_month
    )

    # Standard real price: current script behavior
    cs["index_factor"] = reference_kpi / cs["bas_kpi"]
    cs["real_price"] = cs["price"] / cs["index_factor"]

    # Lag-adjusted price: divide by eval-month CPI / base CPI
    cs["index_factor_lag_adj"] = eval_month_kpi / cs["bas_kpi"]
    cs["real_price_lag_adj"] = cs["price"] / cs["index_factor_lag_adj"]

    if helper_price_mode == "standard_real":
        price_col = "real_price"
    elif helper_price_mode == "lag_adjusted":
        price_col = "real_price_lag_adj"
    else:
        raise ValueError(
            f"Unknown HELPER_PRICE_MODE: {helper_price_mode}. "
            f"Use 'standard_real' or 'lag_adjusted'."
        )

    cs["price_for_helper"] = cs[price_col]

    return cs, reference_kpi, eval_month_kpi, price_col

# endregion

# region CROSS SECTION 

    # region SETUP

# ---- 1) SETTINGS

eval_date = pd.Timestamp(ENTER_DATE)

cross_section = df_SGBILs.loc[df_SGBILs["date"] == eval_date].copy()
cross_section = cross_section.sort_values("maturity_date").reset_index(drop=True)

if cross_section.empty:
    raise ValueError(f"No instruments found for evaluation date {eval_date.date()}")

ql_eval_date = ql.Date(eval_date.day, eval_date.month, eval_date.year)
ql.Settings.instance().evaluationDate = ql_eval_date

# Bond conventions
day_count = ql.Thirty360(ql.Thirty360.European)
tenor = ql.Period(ql.Annual)
date_gen_rule = ql.DateGeneration.Backward
calendar = ql.Sweden()
business_convention = ql.ModifiedFollowing
end_of_month = False

settlement_days = 2
face = 100.0
redemption = 100.0

# ---- 4) REFERENCE KPI + HELPER PRICES 

ql_settlement_date = calendar.advance(ql_eval_date, settlement_days, ql.Days)
settlement_date = pd.Timestamp(
    ql_settlement_date.year(),
    ql_settlement_date.month(),
    ql_settlement_date.dayOfMonth()
)

cross_section, reference_kpi, eval_month_kpi, price_col = prepare_helper_prices(
    cross_section=cross_section,
    eval_date=eval_date,
    settlement_date=settlement_date,
    kpi_by_month=kpi_by_month,
    helper_price_mode=HELPER_PRICE_MODE,
)

print(f"HELPER_PRICE_MODE = {HELPER_PRICE_MODE}")
print(f"Using helper price column: {price_col}")
print(f"reference_kpi = {reference_kpi:.6f}")
print(f"eval_month_kpi = {eval_month_kpi:.6f}")

# ---- 6) REAL YIELDS

yield_real = []

for _, row in cross_section.iterrows():

    ql_issue = ql.Date(row["issue_date"].day, row["issue_date"].month, row["issue_date"].year)
    ql_maturity = ql.Date(row["maturity_date"].day, row["maturity_date"].month, row["maturity_date"].year)

    schedule = ql.Schedule(
        ql_issue, ql_maturity, tenor, calendar,
        business_convention, business_convention,
        date_gen_rule, end_of_month
    )

    bond = ql.FixedRateBond(
        settlement_days, face, schedule,
        [float(row["coupon"])],
        day_count, business_convention, redemption, ql_issue
    )

    price_obj = ql.BondPrice(float(row["price_for_helper"]), ql.BondPrice.Clean)

    ytm_real = ql.BondFunctions.bondYield(
        bond, price_obj, day_count,
        ql.Compounded, ql.Annual,
        ql_settlement_date,
        1.0e-12, 1000, 0.02
    )

    yield_real.append(float(ytm_real))

cross_section["yield_real"] = yield_real

# ---- 7) BUILD ACTUAL HELPERS

helpers_actual = []
bonds_actual = []
quoted_price_actual = []
yield_real_actual = []
maturities_actual = []

for _, row in cross_section.iterrows():

    ql_issue = ql.Date(row["issue_date"].day, row["issue_date"].month, row["issue_date"].year)
    ql_maturity = ql.Date(row["maturity_date"].day, row["maturity_date"].month, row["maturity_date"].year)

    schedule = ql.Schedule(
        ql_issue, ql_maturity, tenor, calendar,
        business_convention, business_convention,
        date_gen_rule, end_of_month
    )

    bond = ql.FixedRateBond(
        settlement_days, face, schedule,
        [float(row["coupon"])],
        day_count, business_convention, redemption, ql_issue
    )

    helper = ql.FixedRateBondHelper(
        ql.QuoteHandle(ql.SimpleQuote(float(row["price_for_helper"]))),
        settlement_days, face, schedule,
        [float(row["coupon"])],
        day_count, business_convention, redemption, ql_issue
    )

    bonds_actual.append(bond)
    helpers_actual.append(helper)
    quoted_price_actual.append(float(row["price_for_helper"]))
    yield_real_actual.append(float(row["yield_real"]))
    maturities_actual.append(row["maturity_date"])

quoted_price_actual = np.array(quoted_price_actual)
yield_real_actual = np.array(yield_real_actual)
maturities_actual = pd.to_datetime(maturities_actual)

print("Actual helpers built:", len(helpers_actual))

# ---- 10) BUILD SYNTHETIC HELPERS

pre_curve = ql.PiecewiseLogCubicDiscount(
    settlement_days,
    calendar,
    helpers_actual,
    day_count
)
pre_curve.enableExtrapolation()

actual_maturity_dates = sorted({b.maturityDate() for b in bonds_actual})
curve_nodes = actual_maturity_dates.copy()

synthetic_helpers = []

curve_reference_date = calendar.advance(ql_eval_date, settlement_days, ql.Days)

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

    price_mid = face * pre_curve.discount(mid_date)

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

    helper = ql.FixedRateBondHelper(
        ql.QuoteHandle(ql.SimpleQuote(price_mid)),
        settlement_days,
        face,
        schedule_mid,
        [0.0],
        day_count,
        business_convention,
        redemption,
        curve_reference_date
    )

    synthetic_helpers.append(helper)

helpers_extended = helpers_actual + synthetic_helpers

print("Synthetic helpers:", len(synthetic_helpers))
print("Total helpers:", len(helpers_extended))

    # endregion

    # region PREPARE NS ESTIMATION

# ---- DEPOSIT-BASED STARTING ANCHOR FOR b0 + b1

anchor_1m_row = df_anchor_1m_exp.loc[df_anchor_1m_exp["date"] == eval_date]

if anchor_1m_row.empty:
    raise ValueError(f"No 1m expected-inflation anchor found for evaluation date {eval_date.date()}")

real_1m_exp_rate = float(anchor_1m_row["real_1m_exp"].iloc[0])
real_1m_exp_rate_cc = np.log(1.0 + real_1m_exp_rate)

print(
    f"1m expected-inflation start anchor: simple={real_1m_exp_rate:.6f}, "
    f"cc={real_1m_exp_rate_cc:.6f}"
)

# ---- 1) ANCHORS

max_maturity = max(b.maturityDate() for b in bonds_actual)
long_rate_cc = pre_curve.zeroRate(max_maturity, day_count, ql.Continuous).rate()

b0 = long_rate_cc

b1 = real_1m_exp_rate_cc - b0

print(
    f"Start anchors: short={real_1m_exp_rate_cc:.6f}, "
    f"long={long_rate_cc:.6f}, b0={b0:.6f}, b1={b1:.6f}"
)

# ---- 2) BOUNDS

b0_min, b0_max = B0_MIN, B0_MAX
b1_min, b1_max = B1_MIN, B1_MAX
b2_min, b2_max = B2_MIN, B2_MAX

tau_min, tau_max = TAU_MIN, TAU_MAX
kappa_min, kappa_max = 1.0 / tau_max, 1.0 / tau_min

lower = ql.Array(4)
upper = ql.Array(4)

lower[0], upper[0] = b0_min, b0_max
lower[1], upper[1] = b1_min, b1_max
lower[2], upper[2] = b2_min, b2_max
lower[3], upper[3] = kappa_min, kappa_max

constraint = ql.NonhomogeneousBoundaryConstraint(lower, upper)

# ---- 5) PENALTIES AROUND STARTING VALUES

l2 = ql.Array(4)
l2[0] = L2_0
l2[1] = L2_1
l2[2] = L2_2
l2[3] = L2_3

# ---- 4) FIT SETUP

accuracy = ACCURACY
max_evaluations = MAX_EVALUTATIONS
simplex_lambda = SIMPLEX_LAMBDA

fitting = ql.NelsonSiegelFitting(
    ql.Array(),
    ql.Simplex(simplex_lambda),
    l2,
    0.0,
    50.0,
    constraint
)

    # endregion

    # region MULTIPLE START NS ESTIMATION

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

# ---- 2) MULTISTART GRID

b2_grid = B2_GRID
tau_grid = TAU_GRID

# ---- 3) FUNCTION: FIT A SET OF HELPERS

def fit_one_helper_set(helpers, name):

    best_objective = float("inf")
    best_curve = None
    best_params = None
    best_start = None

    for b2_start in b2_grid:
        for tau_start in tau_grid:
            guess = ql.Array(4)
            guess[0] = b0
            guess[1] = b1
            guess[2] = b2_start
            guess[3] = 1.0 / tau_start

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
                objective_value = float(fit_result.minimumCostValue())

                if objective_value < best_objective:
                    best_objective = objective_value
                    best_curve = curve_try
                    best_params = list(fit_result.solution())
                    best_start = (b2_start, tau_start)

            except RuntimeError:
                continue

    if best_curve is None:
        raise RuntimeError(f"NS multistart failed on {name} helpers for {eval_date.date()}")

    print(
        f"NS MULTISTART {name} best start: "
        f"b2={best_start[0]}, tau={best_start[1]}, "
        f"objective={best_objective:.8f}"
    )
    print(f"NS MULTISTART {name} params:", best_params)

    return best_curve, best_params, best_objective, best_start

# ---- 4) FIT ACTUAL AND EXTENDED SETS

best_curve_actual, best_params_actual, best_objective_actual, best_start_actual = fit_one_helper_set(
    helpers_actual, "ACTUAL"
)

best_curve_ext, best_params_ext, best_objective_ext, best_start_ext = fit_one_helper_set(
    helpers_extended, "EXTENDED"
)

# ---- 5) PRICE ACTUAL BONDS OF EACH CURVE

model_prices_actual, model_yields_actual = price_actual_bonds(best_curve_actual)
model_prices_ext, model_yields_ext = price_actual_bonds(best_curve_ext)

rmse_price_actual = float(
    np.sqrt(np.mean((quoted_price_actual - model_prices_actual) ** 2))
)
rmse_price_ext = float(
    np.sqrt(np.mean((quoted_price_actual - model_prices_ext) ** 2))
)

rmse_yield_bp_actual = float(
    np.sqrt(np.mean((yield_real_actual - model_yields_actual) ** 2)) * 1e4
)
rmse_yield_bp_ext = float(
    np.sqrt(np.mean((yield_real_actual - model_yields_ext) ** 2)) * 1e4
)

print("NS MULTISTART RMSE price (actual-only fit):   ", rmse_price_actual)
print("NS MULTISTART RMSE price (extended fit):      ", rmse_price_ext)

print("NS MULTISTART RMSE yield bp (actual-only fit):", rmse_yield_bp_actual)
print("NS MULTISTART RMSE yield bp (extended fit):   ", rmse_yield_bp_ext)

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
    linewidth=2.7,
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
    linestyle="-.",
    label="Bootstrapped"
)

# --- Anchor text box

b0_pct = 100.0 * b0
b0b1_pct = 100.0 * (b0 + b1)

anchor_text = (
    f"Short rate anchor: {b0b1_pct:.2f}%"
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
ymin, ymax = plt.gca().get_ylim()
plt.ylim(ymin - 0.0, ymax + 1.0)
plt.show()

# ---- PRICE RESIDUALS ON ACTUAL BONDS

price_resid_actualfit = quoted_price_actual - model_prices_actual
price_resid_extfit = quoted_price_actual - model_prices_ext

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
ax.set_ylabel("Observed real price - model price")
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

# ---- OBSERVED AND MODEL-IMPLIED REAL YIELDS

fig, ax = plt.subplots(figsize=(8.2, 4.8))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

ax.scatter(
    maturities_plot,
    100.0 * yield_real_actual,
    s=70,
    color=color_observed,
    marker="D",
    edgecolor="black",
    linewidth=0.4,
    label="Observed"
)

ax.scatter(
    maturities_plot,
    100.0 * model_yields_actual,
    s=70,
    color=color_actual,
    marker="^",
    edgecolor="black",
    linewidth=0.4,
    label="Actual"
)

ax.scatter(
    maturities_plot,
    100.0 * model_yields_ext,
    s=62,
    color=color_extended,
    marker="o",
    edgecolor="black",
    linewidth=0.4,
    label="Extended"
)

ax.set_xlabel("Maturity date")
ax.set_ylabel("Yield to maturity (%)")
ax.set_title(f"Observed vs. estimated ({eval_date.date()})", pad=12)

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
    borderpad=0.5,
    handlelength=1.8,
    handletextpad=0.5
)

fig.autofmt_xdate()
plt.tight_layout()
plt.show()

    # endregion

# endregion

# region PANEL 

    # region FUNCTION: ESTIMATE NS WITH MULTIPLE STARTS

def run_estimation_for_date_multistart_ns(
    df_panel: pd.DataFrame,
    df_anchor_1m_exp: pd.DataFrame,
    eval_date: pd.Timestamp,
    helper_price_mode: str = "standard_real",
):

    cross_section = df_panel.loc[df_panel["date"] == eval_date].copy()
    cross_section = cross_section.loc[cross_section["bas_kpi"].notna()].copy()
    cross_section = cross_section.sort_values("maturity_date").reset_index(drop=True)

    if cross_section.empty:
        raise ValueError(f"No instruments found for evaluation date {eval_date.date()}")

    ql_eval_date = ql.Date(eval_date.day, eval_date.month, eval_date.year)
    ql.Settings.instance().evaluationDate = ql_eval_date

    # Bond conventions
    day_count = ql.Thirty360(ql.Thirty360.European)
    tenor = ql.Period(ql.Annual)
    date_gen_rule = ql.DateGeneration.Backward
    calendar = ql.Sweden()
    business_convention = ql.ModifiedFollowing
    end_of_month = False

    settlement_days = 2
    face = 100.0
    redemption = 100.0

    # ---- 1) REFERENCE KPI + HELPER PRICE PREPARATION

    ql_settlement_date = calendar.advance(ql_eval_date, settlement_days, ql.Days)
    settlement_date = pd.Timestamp(
        ql_settlement_date.year(),
        ql_settlement_date.month(),
        ql_settlement_date.dayOfMonth()
    )

    cross_section, reference_kpi, eval_month_kpi, price_col = prepare_helper_prices(
        cross_section=cross_section,
        eval_date=eval_date,
        settlement_date=settlement_date,
        kpi_by_month=kpi_by_month,
        helper_price_mode=helper_price_mode,
    )

    # ---- 3) REAL YIELDS

    yield_real = []

    for _, row in cross_section.iterrows():
        ql_issue = ql.Date(row["issue_date"].day, row["issue_date"].month, row["issue_date"].year)
        ql_maturity = ql.Date(row["maturity_date"].day, row["maturity_date"].month, row["maturity_date"].year)

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

        price_obj = ql.BondPrice(float(row["price_for_helper"]), ql.BondPrice.Clean)

        ytm_real = ql.BondFunctions.bondYield(
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

        yield_real.append(float(ytm_real))

    cross_section["yield_real"] = yield_real

    # ---- 4) BUILD ACTUAL HELPERS

    helpers_actual = []
    bonds_actual = []
    quoted_price_actual = []
    yield_real_actual = []

    for _, row in cross_section.iterrows():
        ql_issue = ql.Date(row["issue_date"].day, row["issue_date"].month, row["issue_date"].year)
        ql_maturity = ql.Date(row["maturity_date"].day, row["maturity_date"].month, row["maturity_date"].year)

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

        helper = ql.FixedRateBondHelper(
            ql.QuoteHandle(ql.SimpleQuote(float(row["price_for_helper"]))),
            settlement_days,
            face,
            schedule,
            [float(row["coupon"])],
            day_count,
            business_convention,
            redemption,
            ql_issue
        )

        bonds_actual.append(bond)
        helpers_actual.append(helper)
        quoted_price_actual.append(float(row["price_for_helper"]))
        yield_real_actual.append(float(row["yield_real"]))

    quoted_price_actual = np.array(quoted_price_actual, dtype=float)
    yield_real_actual = np.array(yield_real_actual, dtype=float)

    # ---- 5) PRE-CURVE + SYNTHETICS FROM ACTUAL BONDS ONLY

    pre_curve = ql.PiecewiseLogCubicDiscount(
        settlement_days,
        calendar,
        helpers_actual,
        day_count
    )
    pre_curve.enableExtrapolation()

    actual_maturity_dates = sorted({b.maturityDate() for b in bonds_actual})
    curve_nodes = actual_maturity_dates.copy()

    synthetic_helpers = []

    curve_reference_date = calendar.advance(ql_eval_date, settlement_days, ql.Days)

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

        price_mid = face * pre_curve.discount(mid_date)

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

        helper = ql.FixedRateBondHelper(
            ql.QuoteHandle(ql.SimpleQuote(price_mid)),
            settlement_days,
            face,
            schedule_mid,
            [0.0],
            day_count,
            business_convention,
            redemption,
            curve_reference_date
        )

        synthetic_helpers.append(helper)

    helpers_extended = helpers_actual + synthetic_helpers

    # ---- 6) DEPOSIT-BASED STARTING ANCHOR FOR b0 + b1

    anchor_1m_row = df_anchor_1m_exp.loc[df_anchor_1m_exp["date"] == eval_date]

    if anchor_1m_row.empty:
        raise ValueError(f"No 1m expected-inflation anchor found for evaluation date {eval_date.date()}")

    real_1m_exp_rate = float(anchor_1m_row["real_1m_exp"].iloc[0])
    real_1m_exp_rate_cc = np.log(1.0 + real_1m_exp_rate)

    # ---- 7) ANCHORS

    max_maturity = max(b.maturityDate() for b in bonds_actual)
    long_rate_cc = pre_curve.zeroRate(max_maturity, day_count, ql.Continuous).rate()

    b0 = long_rate_cc
    b1 = real_1m_exp_rate_cc - b0

    # ---- 8) BOUNDS

    b0_min, b0_max = -0.03, 0.06
    b1_min, b1_max = -0.15, 0.15
    b2_min, b2_max = -0.05, 0.05

    tau_min, tau_max = 1.5, 8.0
    kappa_min, kappa_max = 1.0 / tau_max, 1.0 / tau_min

    lower = ql.Array(4)
    upper = ql.Array(4)

    lower[0], upper[0] = b0_min, b0_max
    lower[1], upper[1] = b1_min, b1_max
    lower[2], upper[2] = b2_min, b2_max
    lower[3], upper[3] = kappa_min, kappa_max

    constraint = ql.NonhomogeneousBoundaryConstraint(lower, upper)

    # ---- 9) PENALTIES AROUND STARTING VALUES

    l2 = ql.Array(4)
    l2[0] = 10.0
    l2[1] = 10.0
    l2[2] = 0.2
    l2[3] = 0.05

    # ---- 10) FIT SETUP

    accuracy = 1.0e-10
    max_evaluations = 10000
    simplex_lambda = 0.005

    fitting = ql.NelsonSiegelFitting(
        ql.Array(),
        ql.Simplex(simplex_lambda),
        l2,
        0.0,
        50.0,
        constraint
    )

    # ---- 11) PRICE ACTUAL BONDS FROM ESTIMATED CURVE

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

    # ---- 12) MULTISTART GRID

    b2_grid = [-0.02, -0.01, 0.0, 0.01, 0.02]
    tau_grid = [1.5, 2.0, 3.0, 4.5, 6.0]

    # ---- 13) FUNCTION: FIT A SET OF HELPERS

    def fit_one_helper_set(helpers, name):

        best_objective = float("inf")
        best_curve = None
        best_params = None
        best_start = None

        for b2_start in b2_grid:
            for tau_start in tau_grid:
                guess = ql.Array(4)
                guess[0] = b0
                guess[1] = b1
                guess[2] = b2_start
                guess[3] = 1.0 / tau_start

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
                    objective_value = float(fit_result.minimumCostValue())

                    if objective_value < best_objective:
                        best_objective = objective_value
                        best_curve = curve_try
                        best_params = list(fit_result.solution())
                        best_start = (b2_start, tau_start)

                except RuntimeError:
                    continue

        if best_curve is None:
            raise RuntimeError(f"NS multistart failed on {name} helpers for {eval_date.date()}")

        return best_curve, best_params, best_objective, best_start

    # ---- 14) FIT ACTUAL, ACTUAL+6M, AND EXTENDED SETS

    curve_actual, params_actual, objective_actual, start_actual = fit_one_helper_set(
        helpers_actual, "ACTUAL"
    )
    curve_extended, params_extended, objective_ext, start_ext = fit_one_helper_set(
        helpers_extended, "EXTENDED"
    )

    # ---- 15) PRICE ACTUAL BONDS AND COMPUTE RMSEs

    model_prices_actualfit, model_yields_actualfit = price_actual_bonds(curve_actual)
    model_prices_extendedfit, model_yields_extendedfit = price_actual_bonds(curve_extended)

    rmse_price_actualfit = float(
        np.sqrt(np.mean((quoted_price_actual - model_prices_actualfit) ** 2))
    )

    rmse_price_extfit = float(
        np.sqrt(np.mean((quoted_price_actual - model_prices_extendedfit) ** 2))
    )

    rmse_yield_actualfit_bp = float(
        np.sqrt(np.mean((yield_real_actual - model_yields_actualfit) ** 2)) * 1e4
    )

    rmse_yield_extfit_bp = float(
        np.sqrt(np.mean((yield_real_actual - model_yields_extendedfit) ** 2)) * 1e4
    )

    return {
    "date": eval_date,

    "b0_actual": params_actual[0],
    "b1_actual": params_actual[1],
    "b2_actual": params_actual[2],
    "k1_actual": params_actual[3],

    "b0_ext": params_extended[0],
    "b1_ext": params_extended[1],
    "b2_ext": params_extended[2],
    "k1_ext": params_extended[3],

    "rmse_price_actual": rmse_price_actualfit,
    "rmse_price_ext": rmse_price_extfit,

    "rmse_yield_bp_actual": rmse_yield_actualfit_bp,
    "rmse_yield_bp_ext": rmse_yield_extfit_bp,

    "objective_actual": objective_actual,
    "objective_ext": objective_ext,

    "n_bonds": len(helpers_actual),
    "n_synth": len(synthetic_helpers),

    "real_1m_exp": real_1m_exp_rate,
    "reference_kpi": reference_kpi,
    "helper_price_mode": helper_price_mode,
    "eval_month_kpi": eval_month_kpi,

    "start_b2_actual": start_actual[0],
    "start_tau_actual": start_actual[1],

    "start_b2_ext": start_ext[0],
    "start_tau_ext": start_ext[1],
}

    # endregion

    # region RUN

month_to_date = (
    df_SGBILs.assign(month=df_SGBILs["date"].dt.to_period("M"))
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
        out = run_estimation_for_date_multistart_ns(
            df_SGBILs,
            df_anchor_1m_exp,
            eval_date,
            helper_price_mode=HELPER_PRICE_MODE,
        )
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

# ---- COMPUTE TAU

results["tau_actual"] = 1.0 / results["k1_actual"]
results["tau_ext"] = 1.0 / results["k1_ext"]

# ---- ARRANGE AND EXPORT TO EXCEL

col_order = [
    "date", "month",

    "b0_actual", "b1_actual", "b2_actual",
    "k1_actual", "tau_actual",

    "b0_ext", "b1_ext", "b2_ext",
    "k1_ext", "tau_ext",

    "rmse_price_actual", "rmse_price_ext",
    "rmse_yield_bp_actual", "rmse_yield_bp_ext",
    "objective_actual", "objective_ext",

    "n_bonds", "n_synth",

    "real_1m_exp", "reference_kpi", "eval_month_kpi", "helper_price_mode",

    "start_b2_actual", "start_tau_actual",
    "start_b2_ext", "start_tau_ext",
]

results = results.loc[:, col_order]

with pd.ExcelWriter("inflation_linked_panel_results.xlsx", engine="openpyxl") as writer:
    results.to_excel(writer, sheet_name="results", index=False)

    if not failed_df.empty:
        failed_df.to_excel(writer, sheet_name="failed", index=False)

print("Exported inflation_linked_panel_results.xlsx")

# ---- PANEL SUMMARY STATISTICS

panel_cols = [
    "b0_actual", "b1_actual", "b2_actual",
    "k1_actual", "tau_actual",

    "b0_ext", "b1_ext", "b2_ext",
    "k1_ext", "tau_ext",

    "rmse_price_actual", "rmse_price_ext",
    "rmse_yield_bp_actual", "rmse_yield_bp_ext",

    "objective_actual", "objective_ext",

    "n_bonds", "n_synth",
    "real_1m_exp", "reference_kpi",
]

print("Panel summary statistics")

for c in panel_cols:
    avg_val = results[c].mean()
    min_val = results[c].min()
    max_val = results[c].max()

    print(f"{c:20s}  avg={avg_val: .6f}  min={min_val: .6f}  max={max_val: .6f}")

    # endregion

    # region PLOT PANEL RMSEs

results_plot = pd.read_excel("inflation_linked_panel_results.xlsx")
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

# Export zero-coupon real yields from EXTENDED NS panel results

maturities_months = list(range(23, 121))   # 23m to 120m

# ---- NS ZERO-YIELD FUNCTION

def ns_zero_yield(t, b0, b1, b2, k1):
    x1 = k1 * t

    term1 = np.where(np.abs(x1) < 1e-12, 1.0, (1.0 - np.exp(-x1)) / x1)
    term2 = term1 - np.exp(-x1)

    return b0 + b1 * term1 + b2 * term2

# ---- EXTRACT EXTENDED PARAMETERS

b0 = results["b0_ext"].to_numpy(dtype=float)
b1 = results["b1_ext"].to_numpy(dtype=float)
b2 = results["b2_ext"].to_numpy(dtype=float)
k1 = results["k1_ext"].to_numpy(dtype=float)

# ---- BUILD ZERO-YIELD PANEL

zero_yields = pd.DataFrame({
    "date": pd.to_datetime(results["date"])
})

for m in maturities_months:
    t = m / 12.0
    zero_yields[f"y_{m}m"] = ns_zero_yield(t, b0, b1, b2, k1)

# ---- PARAMETER AND DIAGNOSTIC SHEET

fit_params = results.loc[:, [
    "date",
    "b0_ext", "b1_ext", "b2_ext",
    "k1_ext", "tau_ext",
    "rmse_price_ext", "rmse_yield_bp_ext",
    "objective_ext",
    "n_bonds", "n_synth",
    "real_1m_exp", "reference_kpi",
    "start_b2_ext", "start_tau_ext",
]].copy()

# ---- EXPORT TO EXCEL

with pd.ExcelWriter("zero_yields_SGBIL.xlsx", engine="openpyxl") as writer:
    zero_yields.to_excel(writer, sheet_name="zero_yields", index=False)
    fit_params.to_excel(writer, sheet_name="fit_params", index=False)

print("Exported zero_yields_SGBIL.xlsx")

# endregion

# region BONDS OUTSTANDING PLOT

plot_cols = [
    "Date",
    "Issue date",
    "Maturity date",
    "ISIN",
    "Serie",
]

df_outstanding = raw_data_linked.loc[:, plot_cols].copy()

df_outstanding["Date"] = pd.to_datetime(df_outstanding["Date"], errors="coerce")
df_outstanding["Issue date"] = pd.to_datetime(df_outstanding["Issue date"], errors="coerce")
df_outstanding["Maturity date"] = pd.to_datetime(df_outstanding["Maturity date"], errors="coerce")

df_outstanding = df_outstanding.dropna(
    subset=["Date", "Issue date", "Maturity date", "Serie"]
).copy()

# keep 2000 onward
df_outstanding = df_outstanding[df_outstanding["Date"] >= pd.Timestamp("2000-01-01")].copy()

# remaining maturity in years
df_outstanding["ttm_years"] = (
    (df_outstanding["Maturity date"] - df_outstanding["Date"]).dt.days / 365.25
)

# keep only bonds that are still outstanding
df_outstanding = df_outstanding[df_outstanding["ttm_years"] >= 0.0].copy()

# drop bond 3103
df_outstanding = df_outstanding[df_outstanding["Serie"] != 3103].copy()

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

# region BID-ASK YIELD SPREADS

# 1) start from quoted bid/ask nominal yields,
# 2) convert to nominal clean prices,
# 3) deflate by the standard Swedish reference index factor
#    (3-month lag, linear interpolation at settlement),
# 4) convert the resulting real clean prices back to real YTMs,
# 5) summarize the bid-ask spread in bp by date.

def linker_bond_from_row(row, settlement_days=2):
    day_count = ql.Thirty360(ql.Thirty360.European)
    tenor = ql.Period(ql.Annual)
    calendar = ql.Sweden()
    business_convention = ql.ModifiedFollowing
    date_gen_rule = ql.DateGeneration.Backward
    end_of_month = False

    ql_issue = ql.Date(
        int(row["issue_date"].day),
        int(row["issue_date"].month),
        int(row["issue_date"].year)
    )
    ql_maturity = ql.Date(
        int(row["maturity_date"].day),
        int(row["maturity_date"].month),
        int(row["maturity_date"].year)
    )

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
        100.0,
        schedule,
        [float(row["coupon"])],
        day_count,
        business_convention,
        100.0,
        ql_issue
    )

    return bond


def quoted_yield_to_nominal_clean_price(
    row,
    quoted_yield,
    ql_settlement_date
):
    day_count = ql.Thirty360(ql.Thirty360.European)

    bond = linker_bond_from_row(row)

    clean_price = ql.BondFunctions.cleanPrice(
        bond,
        float(quoted_yield),
        day_count,
        ql.Compounded,
        ql.Annual,
        ql_settlement_date,
    )

    return float(clean_price)


def real_clean_price_from_nominal_clean_price(
    nominal_clean_price,
    bas_kpi,
    reference_kpi
):
    index_factor = reference_kpi / float(bas_kpi)
    return float(nominal_clean_price) / index_factor


def real_ytm_from_real_clean_price(
    row,
    real_clean_price,
    ql_settlement_date
):
    day_count = ql.Thirty360(ql.Thirty360.European)

    bond = linker_bond_from_row(row)

    price_obj = ql.BondPrice(float(real_clean_price), ql.BondPrice.Clean)

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

    return float(real_ytm)


def compute_bid_ask_spread_summary_linkers(
    df_panel: pd.DataFrame,
    kpi_by_month: pd.Series
) -> pd.DataFrame:
    calendar = ql.Sweden()
    settlement_days = 2

    work = df_panel.copy()

    work = work.dropna(subset=[
        "date",
        "issue_date",
        "maturity_date",
        "coupon",
        "bas_kpi",
        "bid_yield_quoted",
        "ask_yield_quoted",
    ]).copy()

    out_rows = []

    for eval_date, g in work.groupby("date", sort=True):
        ql_eval_date = ql.Date(eval_date.day, eval_date.month, eval_date.year)
        ql_settlement_date = calendar.advance(ql_eval_date, settlement_days, ql.Days)

        settlement_date = pd.Timestamp(
            ql_settlement_date.year(),
            ql_settlement_date.month(),
            ql_settlement_date.dayOfMonth()
        )

        # Use the standard market-convention reference KPI:
        # 3-month lag with linear interpolation.
        reference_kpi = swedish_reference_kpi(
            settlement_date=settlement_date,
            kpi_by_month=kpi_by_month,
            lag_months=3
        )

        spreads_bp = []

        for _, row in g.iterrows():
            try:
                bid_nominal_clean = quoted_yield_to_nominal_clean_price(
                    row=row,
                    quoted_yield=row["bid_yield_quoted"],
                    ql_settlement_date=ql_settlement_date
                )

                ask_nominal_clean = quoted_yield_to_nominal_clean_price(
                    row=row,
                    quoted_yield=row["ask_yield_quoted"],
                    ql_settlement_date=ql_settlement_date
                )

                bid_real_clean = real_clean_price_from_nominal_clean_price(
                    nominal_clean_price=bid_nominal_clean,
                    bas_kpi=row["bas_kpi"],
                    reference_kpi=reference_kpi
                )

                ask_real_clean = real_clean_price_from_nominal_clean_price(
                    nominal_clean_price=ask_nominal_clean,
                    bas_kpi=row["bas_kpi"],
                    reference_kpi=reference_kpi
                )

                bid_real_ytm = real_ytm_from_real_clean_price(
                    row=row,
                    real_clean_price=bid_real_clean,
                    ql_settlement_date=ql_settlement_date
                )

                ask_real_ytm = real_ytm_from_real_clean_price(
                    row=row,
                    real_clean_price=ask_real_clean,
                    ql_settlement_date=ql_settlement_date
                )

                spread_bp = abs(bid_real_ytm - ask_real_ytm) * 1e4
                spreads_bp.append(spread_bp)

            except Exception:
                continue

        if len(spreads_bp) == 0:
            out_rows.append({
                "date": eval_date,
                "median_bid_ask_yield_spread_bp": np.nan,
                "mean_bid_ask_yield_spread_bp": np.nan,
                "n_bonds": 0,
            })
        else:
            out_rows.append({
                "date": eval_date,
                "median_bid_ask_yield_spread_bp": float(np.median(spreads_bp)),
                "mean_bid_ask_yield_spread_bp": float(np.mean(spreads_bp)),
                "n_bonds": int(len(spreads_bp)),
            })

    return pd.DataFrame(out_rows).sort_values("date").reset_index(drop=True)


df_linker_bid_ask_summary = compute_bid_ask_spread_summary_linkers(
    df_panel=df_SGBILs,
    kpi_by_month=kpi_by_month
)

print(df_linker_bid_ask_summary.tail())
df_linker_bid_ask_summary.to_excel("inflation_linked_bid_ask.xlsx", index=False)

# endregion

# region EXTRA RMSE PLOTS 
# must run prep > df_SGBILs, kpi_by_month, prepare_helper_prices before

# ---- SETTINGS

RMSE_START = pd.Timestamp("2004-01-30")
RMSE_END = pd.Timestamp("2025-12-31")
rmse_color = "#0B3D91"   # deep blue

# Plot settings consistent with the other figures
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

# ---- LOAD EXTENDED PARAMETERS FROM PANEL EXCEL

panel_params = pd.read_excel("inflation_linked_panel_results.xlsx", sheet_name="results")
panel_params["date"] = pd.to_datetime(panel_params["date"], errors="coerce")
panel_params = panel_params.sort_values("date").reset_index(drop=True)

panel_params = panel_params.loc[
    (panel_params["date"] >= RMSE_START) &
    (panel_params["date"] <= RMSE_END)
].copy()

# ---- NS ZERO-YIELD FUNCTION (CONTINUOUSLY COMPOUNDED)

def ns_zero_yield_cc(t, b0, b1, b2, k1):
    x1 = k1 * t

    term1 = 1.0 if abs(x1) < 1e-12 else (1.0 - np.exp(-x1)) / x1
    term2 = term1 - np.exp(-x1)

    return b0 + b1 * term1 + b2 * term2


# ---- MODEL YIELD FROM NS PARAMETERS

def model_real_yield_from_ns(row, eval_date, b0, b1, b2, k1):
    ql_eval_date = ql.Date(eval_date.day, eval_date.month, eval_date.year)
    ql.Settings.instance().evaluationDate = ql_eval_date

    day_count = ql.Thirty360(ql.Thirty360.European)
    tenor = ql.Period(ql.Annual)
    calendar = ql.Sweden()
    business_convention = ql.ModifiedFollowing
    date_gen_rule = ql.DateGeneration.Backward
    end_of_month = False

    settlement_days = 2
    face = 100.0
    redemption = 100.0

    ql_issue = ql.Date(row["issue_date"].day, row["issue_date"].month, row["issue_date"].year)
    ql_maturity = ql.Date(row["maturity_date"].day, row["maturity_date"].month, row["maturity_date"].year)

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

    settlement_date = bond.settlementDate()

    dirty_price = 0.0
    for cf in bond.cashflows():
        if cf.hasOccurred(settlement_date):
            continue

        cf_date = cf.date()
        t = day_count.yearFraction(settlement_date, cf_date)
        if t <= 0.0:
            continue

        z = ns_zero_yield_cc(t, b0, b1, b2, k1)
        df = np.exp(-z * t)
        dirty_price += cf.amount() * df

    accrued = bond.accruedAmount(settlement_date)
    clean_price = dirty_price - accrued

    price_obj = ql.BondPrice(clean_price, ql.BondPrice.Clean)

    model_yield = ql.BondFunctions.bondYield(
        bond,
        price_obj,
        day_count,
        ql.Compounded,
        ql.Annual,
        settlement_date,
        1.0e-12,
        1000,
        0.02
    )

    return float(model_yield)


# ---- COMPUTE REAL-YIELD RMSEs BY DATE AND MATURITY BUCKET

bucket_rows = []

for _, p in panel_params.iterrows():
    eval_date = pd.Timestamp(p["date"])

    cross_section = df_SGBILs.loc[df_SGBILs["date"] == eval_date].copy()
    cross_section = cross_section.loc[cross_section["bas_kpi"].notna()].copy()
    cross_section = cross_section.sort_values("maturity_date").reset_index(drop=True)

    if cross_section.empty:
        continue

    ql_eval_date = ql.Date(eval_date.day, eval_date.month, eval_date.year)
    calendar = ql.Sweden()
    settlement_days = 2
    ql_settlement_date = calendar.advance(ql_eval_date, settlement_days, ql.Days)

    settlement_date = pd.Timestamp(
        ql_settlement_date.year(),
        ql_settlement_date.month(),
        ql_settlement_date.dayOfMonth()
    )

    # same helper-price treatment as in the fit
    cross_section, _, _, _ = prepare_helper_prices(
        cross_section=cross_section,
        eval_date=eval_date,
        settlement_date=settlement_date,
        kpi_by_month=kpi_by_month,
        helper_price_mode=HELPER_PRICE_MODE,
    )

    # observed real yields from helper prices
    observed_real_yields = []

    for _, row in cross_section.iterrows():
        day_count = ql.Thirty360(ql.Thirty360.European)
        tenor = ql.Period(ql.Annual)
        business_convention = ql.ModifiedFollowing
        date_gen_rule = ql.DateGeneration.Backward
        end_of_month = False
        face = 100.0
        redemption = 100.0

        ql_issue = ql.Date(row["issue_date"].day, row["issue_date"].month, row["issue_date"].year)
        ql_maturity = ql.Date(row["maturity_date"].day, row["maturity_date"].month, row["maturity_date"].year)

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

        price_obj = ql.BondPrice(float(row["price_for_helper"]), ql.BondPrice.Clean)

        ytm_real = ql.BondFunctions.bondYield(
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

        observed_real_yields.append(float(ytm_real))

    cross_section["yield_real"] = np.array(observed_real_yields, dtype=float)

    # model-implied real yields from extended-fit parameters
    model_yields = []
    for _, bond_row in cross_section.iterrows():
        y_model = model_real_yield_from_ns(
            bond_row,
            eval_date,
            p["b0_ext"], p["b1_ext"], p["b2_ext"], p["k1_ext"]
        )
        model_yields.append(y_model)

    cross_section["model_yield_ext"] = np.array(model_yields, dtype=float)

    cross_section["ttm_years"] = (
        (cross_section["maturity_date"] - cross_section["date"]).dt.days / 365.25
    )

    cross_section["sq_err_bp2"] = (
        (cross_section["yield_real"] - cross_section["model_yield_ext"]) * 1e4
    ) ** 2

    bucket_defs = {
        "Below 5 years": cross_section["ttm_years"] < 5.0,
        "5-10 years": (cross_section["ttm_years"] >= 5.0) & (cross_section["ttm_years"] <= 10.0),
        "Above 10 years": cross_section["ttm_years"] > 10.0,
    }

    out_row = {"date": eval_date}

    for bucket_name, mask in bucket_defs.items():
        g = cross_section.loc[mask]
        col_name = bucket_name.lower().replace(" ", "_").replace("-", "_")

        if len(g) == 0:
            out_row[f"rmse_{col_name}"] = np.nan
        else:
            out_row[f"rmse_{col_name}"] = float(np.sqrt(g["sq_err_bp2"].mean()))

    bucket_rows.append(out_row)

rmse_bucket_df = pd.DataFrame(bucket_rows).sort_values("date").reset_index(drop=True)

# ---- PLOT: THREE SQUARE SUBPLOTS SIDE BY SIDE

fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.4), sharey=True)
fig.patch.set_facecolor("white")

plot_map = [
    ("rmse_below_5_years", "Below 5 years"),
    ("rmse_5_10_years", "5-10 years"),
    ("rmse_above_10_years", "Above 10 years"),
]

for ax, (col, title) in zip(axes, plot_map):
    ax.set_facecolor("white")

    ax.plot(
        rmse_bucket_df["date"],
        rmse_bucket_df[col],
        color=rmse_color,
        linewidth=1.8,
        linestyle="-"
    )

    ax.set_title(title, pad=10)
    ax.set_xlabel("Date")
    ax.set_box_aspect(1)

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
    ax.set_xlim(RMSE_START, RMSE_END)
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

axes[0].set_ylabel("Real-yield RMSE (bp)")

plt.tight_layout()
plt.show()

# endregion

