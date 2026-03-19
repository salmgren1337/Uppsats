import pandas as pd
import QuantLib as ql
import numpy as np
import matplotlib.pyplot as plt

# IMPORTS 
raw_data_linked = pd.read_excel("statsobligationer_data.xlsx", sheet_name = "SGBi long")
short_rates = pd.read_excel("statsobligationer_data.xlsx", sheet_name = "korta räntor riksbanken")
kpi = pd.read_excel("kpi_data.xlsx", sheet_name = "basår 1980")
kpi_exp = pd.read_excel("CPI_Inflation_Expectations.xlsx", sheet_name = "CPI Expectations")

START = pd.Timestamp("2005-10-01")
END = pd.Timestamp("2025-12-31")

# region PREPARE BOND DATA

# ---- 1) PREPARE SGB IL DATA

# Fill missing base CPI for early linkers using series 3001
base_kpi_3001 = raw_data_linked.loc[raw_data_linked["Serie"] == 3001, "BasKPI"].iloc[0]
raw_data_linked.loc[raw_data_linked["Serie"].isin([3002, 3003]), "BasKPI"] = base_kpi_3001

keep_cols = [
    "Date",
    "PX_LAST",
    "YLD_YTM_MID",
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

for col in ["PX_LAST", "YLD_YTM_MID", "Kupong", "BasKPI"]:
    df_SGBILs[col] = pd.to_numeric(df_SGBILs[col], errors="coerce")

df_SGBILs["YLD_YTM_MID"] = df_SGBILs["YLD_YTM_MID"] / 100.0
df_SGBILs["Kupong"] = df_SGBILs["Kupong"] / 100.0

# Rename columns:
df_SGBILs = df_SGBILs.rename(columns={
    "Date": "date",
    "PX_LAST": "price",
    "YLD_YTM_MID": "yield_quoted",
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
    (time_to_maturity >= 2 * 365) &
    (time_to_maturity <= 20 * 365)
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

# T-bill zero-coupon yields
tbill_cols = ["SSVX_1m", "SSVX_3m", "SSVX_6m"]

df_tbill = short_rates.loc[:, ["date"] + tbill_cols].copy()
df_tbill["date"] = pd.to_datetime(df_tbill["date"], errors="coerce")

df_tbill = df_tbill.melt(
    id_vars=["date"],
    value_vars=tbill_cols,
    var_name="serie",
    value_name="yield_quoted"
).dropna(subset=["yield_quoted"]).reset_index(drop=True)

df_tbill["yield_quoted"] = pd.to_numeric(df_tbill["yield_quoted"], errors="coerce") / 100.0

tenor_months = {
    "SSVX_1m": 1,
    "SSVX_3m": 3,
    "SSVX_6m": 6,
}
df_tbill["months"] = df_tbill["serie"].map(tenor_months).astype("Int64")

df_tbill["coupon"] = 0.0
df_tbill["issue_date"] = df_tbill["date"]
df_tbill["maturity_date"] = df_tbill["issue_date"] + pd.to_timedelta(df_tbill["months"] * 30, unit="D")
df_tbill["isin"] = ""
df_tbill["bas_kpi"] = np.nan

# Zero-coupon prices from quoted yields
days = (df_tbill["maturity_date"] - df_tbill["issue_date"]).dt.days
t = days / 360.0
df_tbill["price"] = 100.0 / (1.0 + df_tbill["yield_quoted"]) ** t

# Align T-bills from EOM to SGB IL last trading day
df_tbill["month"] = df_tbill["date"].dt.to_period("M")
df_tbill["date"] = df_tbill["month"].map(month_trading_date)
df_tbill["issue_date"] = df_tbill["date"]

df_tbill = (
    df_tbill.drop(columns=["month", "months"])
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
    .dropna(subset=["date", "inflation year"])
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

# ---- 5) PLOT SHORT-RATE ANCHOR ALTERNATIVES

# Keep one macro dataframe for inflation and expectations
df_anchor_plot = df_cpi.copy()

# Add deposit rate
df_anchor_plot = pd.merge(
    df_anchor_plot,
    df_deposit,
    on="date",
    how="left"
)

# Reshape T-bill data to wide format
df_tbill_plot = (
    df_tbill.loc[:, ["date", "serie", "yield_quoted"]]
    .drop_duplicates(subset=["date", "serie"])
    .pivot(index="date", columns="serie", values="yield_quoted")
    .reset_index()
)

# Merge T-bill rates into plotting dataframe
df_anchor_plot = pd.merge(
    df_anchor_plot,
    df_tbill_plot,
    on="date",
    how="left"
)

# Construct alternative real short-rate proxies
df_anchor_plot["deposit_minus_infl"] = df_anchor_plot["deposit"] - df_anchor_plot["inflation year"]

df_anchor_plot["tbill_1m_minus_infl"] = df_anchor_plot["SSVX_1m"] - df_anchor_plot["inflation year"]
df_anchor_plot["tbill_3m_minus_infl"] = df_anchor_plot["SSVX_3m"] - df_anchor_plot["inflation year"]
df_anchor_plot["tbill_6m_minus_infl"] = df_anchor_plot["SSVX_6m"] - df_anchor_plot["inflation year"]

df_anchor_plot["tbill_1m_minus_exp"] = df_anchor_plot["SSVX_1m"] - df_anchor_plot["cpi_exp_1y"]
df_anchor_plot["tbill_3m_minus_exp"] = df_anchor_plot["SSVX_3m"] - df_anchor_plot["cpi_exp_1y"]
df_anchor_plot["tbill_6m_minus_exp"] = df_anchor_plot["SSVX_6m"] - df_anchor_plot["cpi_exp_1y"]

df_anchor_plot = df_anchor_plot.sort_values("date").reset_index(drop=True)

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
color_1m = "#D55E00"
color_3m = "#43AD73"  
color_6m = "#3677C8"       

fig, ax = plt.subplots(figsize=(8.2, 4.8))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# Deposit minus realized inflation
ax.plot(
    df_anchor_plot["date"],
    100.0 * df_anchor_plot["deposit_minus_infl"],
    color=color_deposit,
    linewidth=1.8,
    linestyle="-",
    label="Deposit - 1y realized"
)

# 1m T-bill
ax.plot(
    df_anchor_plot["date"],
    100.0 * df_anchor_plot["tbill_1m_minus_infl"],
    color=color_1m,
    linewidth=1.8,
    alpha=0.85,
    linestyle="-",
    label="1m T-bill - 1y realized"
)

ax.plot(
    df_anchor_plot["date"],
    100.0 * df_anchor_plot["tbill_1m_minus_exp"],
    color=color_1m,
    linewidth=2.2,
    linestyle=":",
    label="1m T-bill - 1y expected"
)

# 3m T-bill
ax.plot(
    df_anchor_plot["date"],
    100.0 * df_anchor_plot["tbill_3m_minus_infl"],
    color=color_3m,
    linewidth=1.8,
    alpha=0.85,
    linestyle="-",
    label="3m T-bill - 1y realized"
)

ax.plot(
    df_anchor_plot["date"],
    100.0 * df_anchor_plot["tbill_3m_minus_exp"],
    color=color_3m,
    linewidth=2.2,
    linestyle=":",
    label="3m T-bill - 1y expected"
)

# 6m T-bill
ax.plot(
    df_anchor_plot["date"],
    100.0 * df_anchor_plot["tbill_6m_minus_infl"],
    color=color_6m,
    linewidth=1.8,
    alpha=0.85,
    linestyle="-",
    label="6m T-bill - 1y realized"
)

ax.plot(
    df_anchor_plot["date"],
    100.0 * df_anchor_plot["tbill_6m_minus_exp"],
    color=color_6m,
    linewidth=2.2,
    linestyle=":",
    label="6m T-bill - 1y expected"
)

ax.axhline(0.0, color="black", linestyle=":", linewidth=1.0)

ax.set_xlabel("Date")
ax.set_ylabel("money market rate - inflation (pp)")
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

# ---- 6) BUILD REAL SHORT-RATE INSTRUMENTS

# Real deposit series: deposit - realized inflation
df_real_deposit = pd.merge(
    df_deposit,
    df_cpi.loc[:, ["date", "inflation year"]],
    on="date",
    how="inner"
)

df_real_deposit["real_deposit"] = (
    df_real_deposit["deposit"] - df_real_deposit["inflation year"]
)

df_real_deposit = (
    df_real_deposit.loc[:, ["date", "real_deposit"]]
    .dropna(subset=["date", "real_deposit"])
    .drop_duplicates(subset=["date"])
    .sort_values("date")
    .reset_index(drop=True)
)

# Real T-bill instruments: nominal T-bill yield - 1y expected inflation
df_tbill_real = pd.merge(
    df_tbill,
    df_cpi.loc[:, ["date", "cpi_exp_1y"]],
    on="date",
    how="left"
)

df_tbill_real["yield_quoted"] = (
    df_tbill_real["yield_quoted"] - df_tbill_real["cpi_exp_1y"]
)

df_tbill_real["serie"] = df_tbill_real["serie"].map({
    "SSVX_1m": "1m_real",
    "SSVX_3m": "3m_real",
    "SSVX_6m": "6m_real",
})

days = (df_tbill_real["maturity_date"] - df_tbill_real["issue_date"]).dt.days
t = days / 360.0
df_tbill_real["price"] = 100.0 / (1.0 + df_tbill_real["yield_quoted"]) ** t

df_tbill_real = (
    df_tbill_real.loc[:, [
        "date", "price", "yield_quoted", "coupon",
        "issue_date", "maturity_date", "isin", "serie", "bas_kpi"
    ]]
    .dropna(subset=["date", "price", "yield_quoted", "issue_date", "maturity_date", "serie"])
    .sort_values(["date", "maturity_date"])
    .reset_index(drop=True)
)

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

# endregion

# region CROSS SECTION 

    # region SETUP

# ---- 1) SETTINGS

ENTER_DATE = "2022-12-30"
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

# ---- 4) REFERENCE KPI

ql_settlement_date = calendar.advance(ql_eval_date, settlement_days, ql.Days)
settlement_date = pd.Timestamp(
    ql_settlement_date.year(),
    ql_settlement_date.month(),
    ql_settlement_date.dayOfMonth()
)

reference_kpi = swedish_reference_kpi(settlement_date, kpi_by_month, lag_months=3)

# ---- 5) REAL PRICES

cross_section["index_factor"] = reference_kpi / cross_section["bas_kpi"]
cross_section["real_price"] = cross_section["price"] / cross_section["index_factor"]

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

    price_obj = ql.BondPrice(float(row["real_price"]), ql.BondPrice.Clean)

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
        ql.QuoteHandle(ql.SimpleQuote(float(row["real_price"]))),
        settlement_days, face, schedule,
        [float(row["coupon"])],
        day_count, business_convention, redemption, ql_issue
    )

    bonds_actual.append(bond)
    helpers_actual.append(helper)
    quoted_price_actual.append(float(row["real_price"]))
    yield_real_actual.append(float(row["yield_real"]))
    maturities_actual.append(row["maturity_date"])

quoted_price_actual = np.array(quoted_price_actual)
yield_real_actual = np.array(yield_real_actual)
maturities_actual = pd.to_datetime(maturities_actual)

print("Actual helpers built:", len(helpers_actual))

# ---- 8) 6M REAL T-BILL ANCHOR

tbill_6m_row = df_tbill_real.loc[
    (df_tbill_real["date"] == eval_date) &
    (df_tbill_real["serie"] == "6m_real")
].copy()

if tbill_6m_row.empty:
    raise ValueError(f"No 6m real T-bill anchor found for {eval_date.date()}")

tbill_6m_row = tbill_6m_row.iloc[0]

real_6m_rate = float(tbill_6m_row["yield_quoted"])
real_6m_price = float(tbill_6m_row["price"])

real_6m_issue_date = pd.Timestamp(tbill_6m_row["issue_date"])
real_6m_maturity_date = pd.Timestamp(tbill_6m_row["maturity_date"])

real_6m_rate_cc = np.log(1.0 + real_6m_rate)

# ---- 9) ADD 6M HELPER

ql_issue_6m = ql.Date(real_6m_issue_date.day, real_6m_issue_date.month, real_6m_issue_date.year)
ql_maturity_6m = ql.Date(real_6m_maturity_date.day, real_6m_maturity_date.month, real_6m_maturity_date.year)

schedule_6m = ql.Schedule(
    ql_issue_6m, ql_maturity_6m,
    ql.Period(ql.Once), calendar,
    business_convention, business_convention,
    ql.DateGeneration.Forward, False
)

tbill_6m_helper = ql.FixedRateBondHelper(
    ql.QuoteHandle(ql.SimpleQuote(real_6m_price)),
    settlement_days, face, schedule_6m,
    [0.0], day_count,
    business_convention, redemption, ql_issue_6m
)

helpers_actual_6m = helpers_actual + [tbill_6m_helper]

# ---- 10) PRE-CURVE + SYNTHETICS

pre_curve = ql.PiecewiseLogCubicDiscount(
    settlement_days, calendar, helpers_actual_6m, day_count
)
pre_curve.enableExtrapolation()

actual_maturity_dates = sorted({b.maturityDate() for b in bonds_actual})
curve_nodes = sorted({ql_maturity_6m, *actual_maturity_dates})

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
        curve_reference_date, mid_date,
        ql.Period(ql.Once), calendar,
        business_convention, business_convention,
        ql.DateGeneration.Forward, False
    )

    helper = ql.FixedRateBondHelper(
        ql.QuoteHandle(ql.SimpleQuote(price_mid)),
        settlement_days, face, schedule_mid,
        [0.0], day_count,
        business_convention, redemption,
        curve_reference_date
    )

    synthetic_helpers.append(helper)

helpers_extended = helpers_actual_6m + synthetic_helpers

print("Synthetic helpers:", len(synthetic_helpers))
print("Total helpers:", len(helpers_extended))

    # endregion

    # region PREPARE NS ESTIMATION

# ---- DEPOSIT-BASED STARTING ANCHOR FOR b0 + b1

real_deposit_row = df_real_deposit.loc[df_real_deposit["date"] == eval_date]

if real_deposit_row.empty:
    raise ValueError(f"No real deposit anchor found for evaluation date {eval_date.date()}")

real_deposit_rate = float(real_deposit_row["real_deposit"].iloc[0])
real_deposit_rate_cc = np.log(1.0 + real_deposit_rate)

print(
    f"Deposit-based start anchor: simple={real_deposit_rate:.6f}, "
    f"cc={real_deposit_rate_cc:.6f}"
)

# ---- 1) ANCHORS

max_maturity = max(b.maturityDate() for b in bonds_actual)
long_rate_cc = pre_curve.zeroRate(max_maturity, day_count, ql.Continuous).rate()

b0 = long_rate_cc
b1 = real_deposit_rate_cc - b0

print(
    f"Start anchors: short={real_deposit_rate_cc:.6f}, "
    f"long={long_rate_cc:.6f}, b0={b0:.6f}, b1={b1:.6f}"
)

# ---- 2) BOUNDS

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

# ---- 5) PENALTIES AROUND STARTING VALUES

l2 = ql.Array(4)
l2[0] = 0.05
l2[1] = 0.001
l2[2] = 0.2
l2[3] = 0.05

# ---- 4) FIT SETUP

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

b2_grid = [-0.02, -0.01, 0.0, 0.01, 0.02]
tau_grid = [1.5, 2.0, 3.0, 4.5, 6.0]

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

# ---- 4) FIT ACTUAL, ACTUAL+6M, AND EXTENDED SETS

best_curve_actual, best_params_actual, best_objective_actual, best_start_actual = fit_one_helper_set(
    helpers_actual, "ACTUAL"
)

best_curve_actual_6m, best_params_actual_6m, best_objective_actual_6m, best_start_actual_6m = fit_one_helper_set(
    helpers_actual_6m, "ACTUAL+6M"
)

best_curve_ext, best_params_ext, best_objective_ext, best_start_ext = fit_one_helper_set(
    helpers_extended, "EXTENDED"
)

# ---- 5) PRICE ACTUAL BONDS OF EACH CURVE

model_prices_actual, model_yields_actual = price_actual_bonds(best_curve_actual)
model_prices_actual_6m, model_yields_actual_6m = price_actual_bonds(best_curve_actual_6m)
model_prices_ext, model_yields_ext = price_actual_bonds(best_curve_ext)

rmse_price_actual = float(
    np.sqrt(np.mean((quoted_price_actual - model_prices_actual) ** 2))
)
rmse_price_actual_6m = float(
    np.sqrt(np.mean((quoted_price_actual - model_prices_actual_6m) ** 2))
)
rmse_price_ext = float(
    np.sqrt(np.mean((quoted_price_actual - model_prices_ext) ** 2))
)

rmse_yield_bp_actual = float(
    np.sqrt(np.mean((yield_real_actual - model_yields_actual) ** 2)) * 1e4
)
rmse_yield_bp_actual_6m = float(
    np.sqrt(np.mean((yield_real_actual - model_yields_actual_6m) ** 2)) * 1e4
)
rmse_yield_bp_ext = float(
    np.sqrt(np.mean((yield_real_actual - model_yields_ext) ** 2)) * 1e4
)

print("NS MULTISTART RMSE price (actual-only fit):   ", rmse_price_actual)
print("NS MULTISTART RMSE price (actual+6m fit):     ", rmse_price_actual_6m)
print("NS MULTISTART RMSE price (extended fit):      ", rmse_price_ext)

print("NS MULTISTART RMSE yield bp (actual-only fit):", rmse_yield_bp_actual)
print("NS MULTISTART RMSE yield bp (actual+6m fit):  ", rmse_yield_bp_actual_6m)
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
color_actual_6m = "#7A7A7A"
color_extended = "#0B1EFF"
color_boot = "#E31A1C"
color_observed = "#E31A1C"
color_6m_point = "#FF5A0B"

# ---- ZERO-COUPON CURVES

times_used = []
zero_actual = []
zero_actual_6m = []
zero_extended = []
zero_boot = []

for t in times:
    months = int(round(t * 12))
    d = calendar.advance(ql_eval_date, ql.Period(months, ql.Months))

    if d > max_date:
        break

    times_used.append(months / 12.0)
    zero_actual.append(best_curve_actual.zeroRate(d, dc_plot, comp).rate())
    zero_actual_6m.append(best_curve_actual_6m.zeroRate(d, dc_plot, comp).rate())
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
    100.0 * np.array(zero_actual_6m),
    color=color_actual_6m,
    linewidth=2.7,
    linestyle="-",
    label="Actual + 6m real T-bill"
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

price_resid_actualfit = quoted_price_actual - model_prices_actual
price_resid_actual6mfit = quoted_price_actual - model_prices_actual_6m
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
    price_resid_actual6mfit,
    s=60,
    color=color_actual_6m,
    marker="o",
    edgecolor="black",
    linewidth=0.4,
    label="Actual + 6m real T-bill fit"
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

ax.scatter(
    pd.Timestamp(real_6m_maturity_date),
    0.0,
    s=78,
    color=color_6m_point,
    marker="^",
    edgecolor="black",
    linewidth=0.5,
    label="6m real T-bill"
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
    100.0 * model_yields_actual_6m,
    s=60,
    color=color_actual_6m,
    marker="o",
    edgecolor="black",
    linewidth=0.4,
    label="Actual + 6m real T-bill"
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

ax.scatter(
    pd.Timestamp(real_6m_maturity_date),
    100.0 * real_6m_rate,
    s=82,
    color=color_6m_point,
    marker="D",
    edgecolor="black",
    linewidth=0.5,
    label="6m real T-bill"
)

ax.set_xlabel("Maturity date")
ax.set_ylabel("Yield to maturity (%)")
ax.set_title(f"Observed vs. implied ({eval_date.date()})", pad=12)

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
    df_real_deposit: pd.DataFrame,
    df_tbill_real: pd.DataFrame,
    eval_date: pd.Timestamp
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

    # ---- 1) REFERENCE KPI

    ql_settlement_date = calendar.advance(ql_eval_date, settlement_days, ql.Days)
    settlement_date = pd.Timestamp(
        ql_settlement_date.year(),
        ql_settlement_date.month(),
        ql_settlement_date.dayOfMonth()
    )

    reference_kpi = swedish_reference_kpi(settlement_date, kpi_by_month, lag_months=3)

    # ---- 2) REAL PRICES

    cross_section["index_factor"] = reference_kpi / cross_section["bas_kpi"]
    cross_section["real_price"] = cross_section["price"] / cross_section["index_factor"]

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

        price_obj = ql.BondPrice(float(row["real_price"]), ql.BondPrice.Clean)

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
            ql.QuoteHandle(ql.SimpleQuote(float(row["real_price"]))),
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
        quoted_price_actual.append(float(row["real_price"]))
        yield_real_actual.append(float(row["yield_real"]))

    quoted_price_actual = np.array(quoted_price_actual, dtype=float)
    yield_real_actual = np.array(yield_real_actual, dtype=float)

    # ---- 5) 6M REAL T-BILL ANCHOR

    tbill_6m_row = df_tbill_real.loc[
        (df_tbill_real["date"] == eval_date) &
        (df_tbill_real["serie"] == "6m_real")
    ].copy()

    if tbill_6m_row.empty:
        raise ValueError(f"No 6m real T-bill anchor found for {eval_date.date()}")

    tbill_6m_row = tbill_6m_row.iloc[0]

    real_6m_rate = float(tbill_6m_row["yield_quoted"])
    real_6m_price = float(tbill_6m_row["price"])
    real_6m_issue_date = pd.Timestamp(tbill_6m_row["issue_date"])
    real_6m_maturity_date = pd.Timestamp(tbill_6m_row["maturity_date"])

    real_6m_rate_cc = np.log(1.0 + real_6m_rate)

    # ---- 6) ADD 6M HELPER

    ql_issue_6m = ql.Date(real_6m_issue_date.day, real_6m_issue_date.month, real_6m_issue_date.year)
    ql_maturity_6m = ql.Date(real_6m_maturity_date.day, real_6m_maturity_date.month, real_6m_maturity_date.year)

    schedule_6m = ql.Schedule(
        ql_issue_6m,
        ql_maturity_6m,
        ql.Period(ql.Once),
        calendar,
        business_convention,
        business_convention,
        ql.DateGeneration.Forward,
        False
    )

    tbill_6m_helper = ql.FixedRateBondHelper(
        ql.QuoteHandle(ql.SimpleQuote(real_6m_price)),
        settlement_days,
        face,
        schedule_6m,
        [0.0],
        day_count,
        business_convention,
        redemption,
        ql_issue_6m
    )

    helpers_actual_6m = helpers_actual + [tbill_6m_helper]

    # ---- 7) PRE-CURVE + SYNTHETICS

    pre_curve = ql.PiecewiseLogCubicDiscount(
        settlement_days,
        calendar,
        helpers_actual_6m,
        day_count
    )
    pre_curve.enableExtrapolation()

    actual_maturity_dates = sorted({b.maturityDate() for b in bonds_actual})
    curve_nodes = sorted({ql_maturity_6m, *actual_maturity_dates})

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

    helpers_extended = helpers_actual_6m + synthetic_helpers

    # ---- 8) DEPOSIT-BASED STARTING ANCHOR FOR b0 + b1

    real_deposit_row = df_real_deposit.loc[df_real_deposit["date"] == eval_date]

    if real_deposit_row.empty:
        raise ValueError(f"No real deposit anchor found for evaluation date {eval_date.date()}")

    real_deposit_rate = float(real_deposit_row["real_deposit"].iloc[0])
    real_deposit_rate_cc = np.log(1.0 + real_deposit_rate)

    # ---- 9) ANCHORS

    max_maturity = max(b.maturityDate() for b in bonds_actual)
    long_rate_cc = pre_curve.zeroRate(max_maturity, day_count, ql.Continuous).rate()

    b0 = long_rate_cc
    b1 = real_deposit_rate_cc - b0

    # ---- 10) BOUNDS

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

    # ---- 11) PENALTIES AROUND STARTING VALUES

    l2 = ql.Array(4)
    l2[0] = 0.05
    l2[1] = 0.001
    l2[2] = 0.2
    l2[3] = 0.05

    # ---- 12) FIT SETUP

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

    # ---- 13) PRICE ACTUAL BONDS FROM ESTIMATED CURVE

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

    # ---- 14) MULTISTART GRID

    b2_grid = [-0.02, -0.01, 0.0, 0.01, 0.02]
    tau_grid = [1.5, 2.0, 3.0, 4.5, 6.0]

    # ---- 15) FUNCTION: FIT A SET OF HELPERS

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

    # ---- 16) FIT ACTUAL, ACTUAL+6M, AND EXTENDED SETS

    curve_actual, params_actual, objective_actual, start_actual = fit_one_helper_set(
        helpers_actual, "ACTUAL"
    )
    curve_actual_6m, params_actual_6m, objective_actual_6m, start_actual_6m = fit_one_helper_set(
        helpers_actual_6m, "ACTUAL+6M"
    )
    curve_extended, params_extended, objective_ext, start_ext = fit_one_helper_set(
        helpers_extended, "EXTENDED"
    )

    # ---- 17) PRICE ACTUAL BONDS AND COMPUTE RMSEs

    model_prices_actualfit, model_yields_actualfit = price_actual_bonds(curve_actual)
    model_prices_actual6mfit, model_yields_actual6mfit = price_actual_bonds(curve_actual_6m)
    model_prices_extendedfit, model_yields_extendedfit = price_actual_bonds(curve_extended)

    rmse_price_actualfit = float(
        np.sqrt(np.mean((quoted_price_actual - model_prices_actualfit) ** 2))
    )
    rmse_price_actual6mfit = float(
        np.sqrt(np.mean((quoted_price_actual - model_prices_actual6mfit) ** 2))
    )
    rmse_price_extfit = float(
        np.sqrt(np.mean((quoted_price_actual - model_prices_extendedfit) ** 2))
    )

    rmse_yield_actualfit_bp = float(
        np.sqrt(np.mean((yield_real_actual - model_yields_actualfit) ** 2)) * 1e4
    )
    rmse_yield_actual6mfit_bp = float(
        np.sqrt(np.mean((yield_real_actual - model_yields_actual6mfit) ** 2)) * 1e4
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

        "b0_actual_6m": params_actual_6m[0],
        "b1_actual_6m": params_actual_6m[1],
        "b2_actual_6m": params_actual_6m[2],
        "k1_actual_6m": params_actual_6m[3],

        "b0_ext": params_extended[0],
        "b1_ext": params_extended[1],
        "b2_ext": params_extended[2],
        "k1_ext": params_extended[3],

        "rmse_price_actual": rmse_price_actualfit,
        "rmse_price_actual_6m": rmse_price_actual6mfit,
        "rmse_price_ext": rmse_price_extfit,

        "rmse_yield_bp_actual": rmse_yield_actualfit_bp,
        "rmse_yield_bp_actual_6m": rmse_yield_actual6mfit_bp,
        "rmse_yield_bp_ext": rmse_yield_extfit_bp,

        "objective_actual": objective_actual,
        "objective_actual_6m": objective_actual_6m,
        "objective_ext": objective_ext,

        "n_bonds": len(helpers_actual),
        "n_synth": len(synthetic_helpers),

        "real_deposit": real_deposit_rate,
        "real_6m": real_6m_rate,
        "reference_kpi": reference_kpi,

        "start_b2_actual": start_actual[0],
        "start_tau_actual": start_actual[1],

        "start_b2_actual_6m": start_actual_6m[0],
        "start_tau_actual_6m": start_actual_6m[1],

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
            df_real_deposit,
            df_tbill_real,
            eval_date
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
results["tau_actual_6m"] = 1.0 / results["k1_actual_6m"]
results["tau_ext"] = 1.0 / results["k1_ext"]

# ---- ARRANGE AND EXPORT TO EXCEL

col_order = [
    "date", "month",

    # actual-only params
    "b0_actual", "b1_actual", "b2_actual",
    "k1_actual", "tau_actual",

    # actual + 6m params
    "b0_actual_6m", "b1_actual_6m", "b2_actual_6m",
    "k1_actual_6m", "tau_actual_6m",

    # extended params
    "b0_ext", "b1_ext", "b2_ext",
    "k1_ext", "tau_ext",

    # fit diagnostics
    "rmse_price_actual", "rmse_price_actual_6m", "rmse_price_ext",
    "rmse_yield_bp_actual", "rmse_yield_bp_actual_6m", "rmse_yield_bp_ext",
    "objective_actual", "objective_actual_6m", "objective_ext",

    # counts
    "n_bonds", "n_synth",

    # anchors / indexation inputs
    "real_deposit", "real_6m", "reference_kpi",

    # winning starting values
    "start_b2_actual", "start_tau_actual",
    "start_b2_actual_6m", "start_tau_actual_6m",
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

    "b0_actual_6m", "b1_actual_6m", "b2_actual_6m",
    "k1_actual_6m", "tau_actual_6m",

    "b0_ext", "b1_ext", "b2_ext",
    "k1_ext", "tau_ext",

    "rmse_price_actual", "rmse_price_actual_6m", "rmse_price_ext",
    "rmse_yield_bp_actual", "rmse_yield_bp_actual_6m", "rmse_yield_bp_ext",

    "objective_actual", "objective_actual_6m", "objective_ext",

    "n_bonds", "n_synth",
    "real_deposit", "real_6m", "reference_kpi",
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
color_ext = "#0B1EFF"        
color_actual_6m = "#2CA02C" 
color_actual = "#000000"

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
    results_plot["rmse_yield_bp_actual_6m"],
    color=color_actual_6m,
    linewidth=2.0,
    linestyle="-.",
    label="Actual + 6m real T-bill"
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
    "real_deposit", "real_6m", "reference_kpi",
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