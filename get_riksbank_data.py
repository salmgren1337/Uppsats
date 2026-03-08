import time
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

BASE_URL = "https://api.riksbank.se/swea/v1"

def fetch_data(series_id, date_from, date_to):
    url = f"{BASE_URL}/observations/{series_id}/{date_from}/{date_to}"
    r = requests.get(url)
    r.raise_for_status()

    data = r.json()
    obs = data["observations"] if isinstance(data, dict) else data

    df = pd.DataFrame(obs)
    if df.empty:
        return pd.DataFrame(columns=[series_id])

    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    return (
        df[["date", "value"]]
        .set_index("date")
        .sort_index()
        .rename(columns={"value": series_id})
    )

# region SSVX
series_ids = [
    "SETB1MBENCHC",   
    "SETB3MBENCH",   
    "SETB6MBENCH",   
    "SETB12MBENCH",  
]

dfs = []
for s in series_ids:
    print(f"Fetching {s}...")
    df = fetch_data(s, "1998-01-01", "2026-01-31")
    dfs.append(df)
    time.sleep(1)

df_ssvx = pd.concat(dfs, axis=1)
df_ssvx.head()
df_ssvx = df_ssvx.loc["1998-01-01":]
df_ssvx_eom = df_ssvx.resample("ME").last()
df_ssvx_eom.head()
df_ssvx_eom.reset_index().to_excel("korta_räntor_riksbanken.xlsx", sheet_name="ssvx", index=False)

# plot
df_plot = df_ssvx_eom.rename(columns={
    "SETB1MBENCHC": "1M",
    "SETB3MBENCH": "3M",
    "SETB6MBENCH": "6M",
    "SETB12MBENCH": "12M",
})

ax = df_plot.plot(figsize=(10, 6))

ax.set_title("Swedish Treasury Bills (Statsskuldväxlar)")
ax.set_xlabel("Date")
ax.set_ylabel("Yield (%)")
ax.legend(title="Maturity")

plt.show()

# endregion

# region Inlåningsränta (SECBDEPOEFF)
series_id_depo = "SECBDEPOEFF"

print(f"Fetching {series_id_depo} (inlåningsränta)...")
df_depo = fetch_data(series_id_depo, "1998-01-01", "2026-01-31")
time.sleep(1)

# End-of-month series (same approach as your SSVX block)
df_depo_eom = df_depo.resample("ME").last()

# Append as a new sheet in the existing workbook (or create if missing)
out_xlsx = "korta_räntor_riksbanken.xlsx"
sheet_name = "inlåningsränta"

try:
    with pd.ExcelWriter(out_xlsx, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        df_depo_eom.reset_index().to_excel(writer, sheet_name=sheet_name, index=False)
except FileNotFoundError:
    with pd.ExcelWriter(out_xlsx, engine="openpyxl", mode="w") as writer:
        df_depo_eom.reset_index().to_excel(writer, sheet_name=sheet_name, index=False)

# Optional plot
ax = df_depo_eom.rename(columns={series_id_depo: "Inlåningsränta"}).plot(figsize=(10, 4))
ax.set_title("Riksbankens inlåningsränta (SECBDEPOEFF)")
ax.set_xlabel("Date")
ax.set_ylabel("Rate (%)")
plt.show()
# endregion

# region KPI from SCB 
# KPI BAS 2020 https://www.statistikdatabasen.scb.se/pxweb/sv/ssd/START__PR__PR0101__PR0101A/KPI2020M/
df_KPI_2020 = pd.read_csv("TAB6596_sv.csv", encoding="latin1")

df_KPI_2020 = df_KPI_2020[df_KPI_2020["tabellinnehåll"] == "KPI, skuggindex"].copy()
df_KPI_2020["date"] = (
    df_KPI_2020["månad"]
    .str.replace("M", "-", regex=False)
    .pipe(pd.PeriodIndex, freq="M")
)

df_KPI_2020 = df_KPI_2020.drop(columns=["månad", "tabellinnehåll"])

df_KPI_2020 = df_KPI_2020.rename(
    columns={"Konsumentprisindex (KPI), totalt, 2020=100": "KPI"}
)

df_KPI_2020["KPI"] = pd.to_numeric(df_KPI_2020["KPI"], errors="coerce")
df_KPI_2020 = df_KPI_2020[["date", "KPI"]]

df_KPI_2020["log year"] = np.log(df_KPI_2020["KPI"]) - np.log(df_KPI_2020["KPI"]).shift(12)
df_KPI_2020["log month"] = np.log(df_KPI_2020["KPI"]) - np.log(df_KPI_2020["KPI"]).shift(1)

df_KPI_2020["inflation year"] = np.exp(df_KPI_2020["log year"]) - 1
df_KPI_2020["inflation month"] = np.exp(df_KPI_2020["log month"]) - 1

# KPI BAS 1980 https://www.statistikdatabasen.scb.se/pxweb/sv/ssd/START__PR__PR0101__PR0101Z/KPIRBBas1980/
df_KPI_1980 = pd.read_csv("TAB2079_sv.csv", encoding="latin1")

df_KPI_1980 = df_KPI_1980[
    (df_KPI_1980["varu-/tjänstegrupp"] == "999 Konsumentprisindex totalt") &
    (df_KPI_1980["tabellinnehåll"] == "Index, 1980=100")
].copy()

df_KPI_1980["date"] = (
    df_KPI_1980["månad"]
    .str.replace("M", "-", regex=False)
    .pipe(pd.PeriodIndex, freq="M")
)

df_KPI_1980 = df_KPI_1980.drop(
    columns=["varu-/tjänstegrupp", "tabellinnehåll", "månad"]
)

df_KPI_1980 = df_KPI_1980.rename(
    columns={"Konsumentprisindex (Riksbanken)": "KPI"}
)

df_KPI_1980["KPI"] = pd.to_numeric(df_KPI_1980["KPI"], errors="coerce")

df_KPI_1980 = df_KPI_1980[["date", "KPI"]]

df_KPI_1980["log year"] = np.log(df_KPI_1980["KPI"]) - np.log(df_KPI_1980["KPI"]).shift(12)
df_KPI_1980["log month"] = np.log(df_KPI_1980["KPI"]) - np.log(df_KPI_1980["KPI"]).shift(1)

df_KPI_1980["inflation year"] = np.exp(df_KPI_1980["log year"]) - 1
df_KPI_1980["inflation month"] = np.exp(df_KPI_1980["log month"]) - 1

# Plot KPI
plt.figure()

plt.plot(df_KPI_2020["date"].dt.to_timestamp(),
         df_KPI_2020["log year"],
         label="KPI base 2020")

plt.plot(df_KPI_1980["date"].dt.to_timestamp(),
         df_KPI_1980["log year"],
         label="KPI base 1980")

plt.xlabel("Date")
plt.ylabel("Yearly log change")
plt.title("Yearly log change in CPI")
plt.legend()

plt.show()

# Export to excel
with pd.ExcelWriter("kpi_data.xlsx") as writer:
    df_KPI_2020.to_excel(writer, sheet_name="basår 2020", index=False)
    df_KPI_1980.to_excel(writer, sheet_name="basår 1980", index=False)