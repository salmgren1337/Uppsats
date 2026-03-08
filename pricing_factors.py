import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---- IMPORTS
fitted_zero_SGB = pd.read_excel("zero_yields_SGB.xlsx", sheet_name = "zero yields")
params_SGB = pd.read_excel("zero_yields_SGB.xlsx", sheet_name = "fit params")

fitted_zero_SGBIL = pd.read_excel("zero_yields_SGBIL.xlsx", sheet_name = "zero yields")
params_SGBIL = pd.read_excel("zero_yields_SGBIL.xlsx", sheet_name = "fit params")

riksbank_SGB = pd.read_excel("riksbank_zero_SGB.xlsx")
riksbank_SGBIL = pd.read_excel("riksbank_zero_SGBIL.xlsx")

# ---- FORMAT DATES
fitted_zero_SGB["date"] = pd.to_datetime(fitted_zero_SGB["date"])
params_SGB["date"] = pd.to_datetime(params_SGB["date"])

fitted_zero_SGBIL["date"] = pd.to_datetime(fitted_zero_SGBIL["date"])
params_SGBIL["date"] = pd.to_datetime(params_SGBIL["date"])

riksbank_SGB["date"] = pd.to_datetime(riksbank_SGB["date"])
riksbank_SGBIL["date"] = pd.to_datetime(riksbank_SGBIL["date"])


# region PLOTS

# ---- 1) FITTED NOMINAL ZERO-COUPON YIELDS: SELECTED MATURITIES
# manual date range
start_date = "2000-01-01"
end_date   = "2025-12-31"

plot_df = fitted_zero_SGB.loc[
    (fitted_zero_SGB["date"] >= start_date) &
    (fitted_zero_SGB["date"] <= end_date)
].copy()

maturity_cols = [
    "y_1m",   # 1 month
    "y_12m",  # 1 year
    "y_24m",  # 2 years
    "y_48m",  # 4 years
    "y_72m",  # 6 years
    "y_96m",  # 8 years
    "y_120m", # 10 years
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

fig, ax = plt.subplots(figsize=(13, 7))

for col in maturity_cols:
    ax.plot(plot_df["date"], plot_df[col], linewidth=2.0, label=labels[col])

ax.set_title("Nominal Zero-Coupon Yields (Fitted)", fontsize=16, pad=15)
ax.set_xlabel("")
ax.set_ylabel("Yield (decimal)", fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend(frameon=False, ncol=3)
ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
plt.show()

# ---- 2) FITTED INFLATION-LINKED ZERO-COUPON YIELDS: SELECTED MATURITIES
# manual date range
start_date = "2000-01-01"
end_date   = "2025-12-31"

plot_df = fitted_zero_SGBIL.loc[
    (fitted_zero_SGBIL["date"] >= start_date) &
    (fitted_zero_SGBIL["date"] <= end_date)
].copy()

maturity_cols = [
    "y_24m",  # 2 years
    "y_48m",  # 4 years
    "y_72m",  # 6 years
    "y_96m",  # 8 years
    "y_120m", # 10 years
]

labels = {
    "y_24m": "2y",
    "y_48m": "4y",
    "y_72m": "6y",
    "y_96m": "8y",
    "y_120m": "10y",
}

fig, ax = plt.subplots(figsize=(13, 7))

for col in maturity_cols:
    ax.plot(plot_df["date"], plot_df[col], linewidth=2.0, label=labels[col])

ax.set_title("Inflation-Linked Zero-Coupon Yields (Fitted)", fontsize=16, pad=15)
ax.set_xlabel("")
ax.set_ylabel("Yield (decimal)", fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend(frameon=False, ncol=3)
ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
plt.show()

# ---- 3A) RIKSBANK NOMINAL ZERO-COUPON YIELDS: SELECTED MATURITIES
# manual date range
start_date = "2000-01-01"
end_date   = "2025-12-31"

plot_df = riksbank_SGB.loc[
    (riksbank_SGB["date"] >= start_date) &
    (riksbank_SGB["date"] <= end_date)
].copy()

maturity_cols = [
    "y_12m",
    "y_24m",
    "y_48m",
    "y_72m",
    "y_96m",
    "y_120m",
]

labels = {
    "y_12m": "1y",
    "y_24m": "2y",
    "y_48m": "4y",
    "y_72m": "6y",
    "y_96m": "8y",
    "y_120m": "10y",
}

fig, ax = plt.subplots(figsize=(13, 7))

for col in maturity_cols:
    ax.plot(plot_df["date"], plot_df[col], linewidth=2.0, label=labels[col])

ax.set_title("Nominal Zero-Coupon Yields (Riksbank)", fontsize=16, pad=15)
ax.set_xlabel("")
ax.set_ylabel("Yield (percent)", fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend(frameon=False, ncol=3)
ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
plt.show()

# ---- 3B) RIKSBANK INFLATION-LINKED ZERO-COUPON YIELDS: SELECTED MATURITIES
# manual date range
start_date = "2000-01-01"
end_date   = "2025-12-31"

plot_df = riksbank_SGBIL.loc[
    (riksbank_SGBIL["date"] >= start_date) &
    (riksbank_SGBIL["date"] <= end_date)
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

fig, ax = plt.subplots(figsize=(13, 7))

for col in maturity_cols:
    ax.plot(plot_df["date"], plot_df[col], linewidth=2.0, label=labels[col])

ax.set_title("Inflation-Linked Zero-Coupon Yields (Riksbank)", fontsize=16, pad=15)
ax.set_xlabel("")
ax.set_ylabel("Yield (percent)", fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend(frameon=False, ncol=3)
ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
plt.show()

# ---- 4A) NOMINAL FIT RMSE: PRICE AND YIELD
# manual date range
start_date = "2000-01-01"
end_date   = "2025-12-31"

plot_df = params_SGB.loc[
    (params_SGB["date"] >= start_date) &
    (params_SGB["date"] <= end_date)
].copy()

fig, ax1 = plt.subplots(figsize=(13, 7))
ax2 = ax1.twinx()

line1 = ax1.plot(
    plot_df["date"],
    plot_df["rmse_price_ext"],
    linewidth=2.0,
    label="RMSE price"
)

line2 = ax2.plot(
    plot_df["date"],
    plot_df["rmse_yield_bp_ext"],
    linewidth=2.0,
    linestyle="--",
    label="RMSE yield"
)

ax1.set_title("Nominal Fit RMSE", fontsize=16, pad=15)
ax1.set_xlabel("")
ax1.set_ylabel("RMSE on price", fontsize=12)
ax2.set_ylabel("RMSE on yield (bp)", fontsize=12)

ax1.grid(True, alpha=0.3)
ax1.tick_params(axis="x", rotation=45)

lines = line1 + line2
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, frameon=False, loc="upper right")

fig.tight_layout()
plt.show()

# ---- 4B) INFLATION-LINKED FIT RMSE: PRICE AND YIELD
# manual date range
start_date = "2000-01-01"
end_date   = "2025-12-31"

plot_df = params_SGBIL.loc[
    (params_SGBIL["date"] >= start_date) &
    (params_SGBIL["date"] <= end_date)
].copy()

fig, ax1 = plt.subplots(figsize=(13, 7))
ax2 = ax1.twinx()

line1 = ax1.plot(
    plot_df["date"],
    plot_df["rmse_price_ext"],
    linewidth=2.0,
    label="RMSE price"
)

line2 = ax2.plot(
    plot_df["date"],
    plot_df["rmse_yield_bp_ext"],
    linewidth=2.0,
    linestyle="--",
    label="RMSE yield"
)

ax1.set_title("Inflation-Linked Fit RMSE", fontsize=16, pad=15)
ax1.set_xlabel("")
ax1.set_ylabel("RMSE on price", fontsize=12)
ax2.set_ylabel("RMSE on yield (bp)", fontsize=12)

ax1.grid(True, alpha=0.3)
ax1.tick_params(axis="x", rotation=45)

lines = line1 + line2
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, frameon=False, loc="upper right")

fig.tight_layout()
plt.show()

# endregion