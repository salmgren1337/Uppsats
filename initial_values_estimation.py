import numpy as np
import pandas as pd
import statsmodels.api as sm 
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from statsmodels.stats.diagnostic import acorr_ljungbox

# ---- Imports:

# K = 6 pricing factors. Monthly frequency (2003-12-31 - 2025-12-31)
df_factors = pd.read_excel("model_factors_nominal_real_liquidity.xlsx").set_index("date").sort_index().asfreq("ME")

# 1-month annually compounded annualized yield in decimal. Monthly frequency (2000-01-31 - 2026-01-31)
df_short_rate = pd.read_excel("short_rate.xlsx").set_index("date").sort_index().asfreq("ME") / 100
# The model is based on monthly time intervalls --> short rate is the monthly continously compounded return on a 1-month zero coupon bond
# 1) annually compunded rate --> continuously compounded rate, 2) annual rate --> monthly rate
df_short_rate["y_1m_cc"] = np.log1p(df_short_rate["y_1m"] / 12)

# Zero coupon annually compunded annualized yields in decimal on SGBs with maturities 5months to 120months. Monthly frequency (2000-01-31 - 2026-01-31)
df_nominal = pd.read_excel("nominal_eom_zero_coupon_yields_monthly_grid.xlsx").set_index("date").sort_index().asfreq("ME") / 100
# Zero coupon annually compunded annualized yields in decimal on SGB ILs with maturities 23months to 120months. Monthly frequency (2000-01-31 - 2026-01-31)
df_real = pd.read_excel("real_eom_zero_coupon_yields_monthly_grid.xlsx").set_index("date").sort_index().asfreq("ME") / 100
# annaul compunding --> continuous compounding
df_nominal_cc = np.log1p(df_nominal.copy()) 
df_real_cc = np.log1p(df_real.copy())

# CPI, annual log inflation and monthly log inflation in percent. Monthly frequency (2002-01-31 - 2025-12-31)
df_inflation = pd.read_excel("CPI_and_rate.xlsx", sheet_name="inflation").set_index("date").sort_index().asfreq("ME")
# Monthly inflation in decimal. pi_t is is the monthly inflationrate in decimal from t-1 to t (2002-02-28 - 2025-12-31)
pi = (df_inflation["monthly log"] / 100).dropna(how="all")

# ----

# region Helpers 

def enforce_stationarity_spectral_radius(Phi: np.ndarray, rho_max: float = 0.995):
    """
    Enforce VAR(1) stationarity by scaling Phi so that spectral radius <= rho_max.
    Returns: (Phi_stable, rho_before, rho_after, scale_factor)
    """
    eigvals = np.linalg.eigvals(Phi)
    rho = float(np.max(np.abs(eigvals)))

    # If already stable (or pathological), return unchanged
    if (not np.isfinite(rho)) or rho <= rho_max:
        return Phi, rho, rho, 1.0

    scale = rho_max / rho
    Phi_stable = Phi * scale

    eigvals_after = np.linalg.eigvals(Phi_stable)
    rho_after = float(np.max(np.abs(eigvals_after)))

    return Phi_stable, rho, rho_after, scale

# endregion

# region Phi, mu, Sigma 
# ---- Estimate demeaned VAR(1) (equation 1) with OLS

factor_names = df_factors.columns.tolist()

mu = df_factors.mean()
df_factors_dm = df_factors - mu

model = VAR(df_factors_dm)
var_res = model.fit(maxlags=1, trend="n")

Phi = pd.DataFrame(var_res.coefs[0], index=factor_names, columns=factor_names)                  #(K,K)
Sigma = pd.DataFrame(var_res.sigma_u, index=factor_names, columns=factor_names)                 #(K,K)
var_residuals = pd.DataFrame(var_res.resid, index=var_res.resid.index, columns=factor_names)    #(T,K)
mu = pd.Series(mu, index=factor_names)                                                          #(K,)

# ---- Stationarity: Eigenvalues inside unit circle 

eigvals = np.linalg.eigvals(Phi.values)
max_abs = np.max(np.abs(eigvals))
print("Eigenvalues:")
print(eigvals)
if max_abs < 1:
    print("VAR is stable")
else:
    print("VAR is not stable")

# ---- Autocorrelation in residuals: Ljung-Box test

print("\n--- Ljung-Box test (12 lags) ---")
for col in var_residuals.columns:
    pval = acorr_ljungbox(var_residuals[col], lags=[12], return_df=True)["lb_pvalue"].iloc[0]
    print(f"{col}: p-value = {pval:.4f}")
print("p < 0.05  -> reject white noise process")
print("p > 0.05 -> fail to reject white noise process")

# ---- Plot residuals 

fig, axes = plt.subplots(var_residuals.shape[1], 1, figsize=(10, 2.5 * 6), sharex=True)
for i, col in enumerate(var_residuals.columns):
    axes[i].plot(var_residuals.index, var_residuals[col])
    axes[i].axhline(0, linestyle="--")
    axes[i].set_title(f"Residuals: {col}")

plt.tight_layout()
plt.show()

# endregion

# region delta_0, delta_1 
# OLS regression of the short rate on a constant and the pricing factors 

df_short_rate_reg = df_short_rate.join(df_factors, how="inner")

y = df_short_rate_reg["y_1m_cc"]
X = sm.add_constant(df_short_rate_reg.drop(columns=["y_1m_cc", "y_1m"]))
ols_res = sm.OLS(y, X).fit()

delta_0 = ols_res.params["const"]          
delta_1 = ols_res.params.drop("const")  

print(ols_res.summary())

# endregion

# region Observed monthly excess returns & observed monthly real excess returns + inflation compensation 

# ---- (1) Compute observed 1-month nominal and real excess returns (equation 13)

# Compute log prices from yields: log P^{n} = - (n/12) * y^{n} (n is maturity in months)
maturity_months_nom = (
    {f"y_{i}m": i for i in range(5, 121)}
)

maturity_months_real = (
    {f"y_{i}m": i for i in range(23, 121)}
    )

df_logP_nominal = pd.DataFrame(
    {f"logP_{n_months}m": - (n_months / 12) * df_nominal_cc[col]
     for col, n_months in maturity_months_nom.items()},
    index=df_nominal.index
)

df_logP_real = pd.DataFrame(
    {f"logP_{n_months}m": - (n_months / 12) * df_real_cc[col]
     for col, n_months in maturity_months_real.items()},
    index=df_real.index
)

# Compute observed excess 1-month holding period returns (equation 13): rx^{n-1}_{t+1} = log P^{n-1}_{t+1} - log P^{n}_{t} - r_{t}
# For maturities 6m, 12m, 24m, 36m, ... , 120m for SGBs, 24m, 36m, ... , 120m for SGB ILs
df_hpr_nominal = df_logP_nominal.copy().join(df_short_rate["y_1m_cc"], how="inner")
df_hpr_real = df_logP_real.copy().join(df_short_rate["y_1m_cc"], how="inner")

list_nominal_maturities = [6] + list(range(12, 121, 12)) 
list_real_maturities = list(range(24, 121, 12)) 

df_rx = pd.DataFrame(index=df_hpr_nominal.index)    # index: EOM (2001-01-31 - 2026-01-31)
df_rx_real = pd.DataFrame(index=df_hpr_real.index)  # index: EOM (2001-01-31 - 2026-01-31)

month_interval = 1 

for n in list_nominal_maturities:
    col_n = f"logP_{n}m"
    col_n_minus_1 = f"logP_{n-1}m"

    _1m_hpr = df_hpr_nominal[col_n_minus_1].shift(-month_interval) - df_hpr_nominal[col_n]
    rx_t_to_tp1 = _1m_hpr - df_hpr_nominal["y_1m_cc"]

    df_rx[f"rx_1m_{n}m"] = rx_t_to_tp1.shift(month_interval)

for n in list_real_maturities:
    col_n = f"logP_{n}m"
    col_n_minus_1 = f"logP_{n-1}m"

    _1m_hpr_real = df_hpr_real[col_n_minus_1].shift(-month_interval) - df_hpr_real[col_n]
    rx_real_t_to_tp1 = _1m_hpr_real - df_hpr_real["y_1m_cc"]

    df_rx_real[f"rx_real_1m_{n}m"] = rx_real_t_to_tp1.shift(month_interval)

df_rx = df_rx.dropna(how="all")
df_rx_real = df_rx_real.dropna(how="all")

# IMPORTANT: Each row t in df_rx and df_rx_real is the 1m excess hpr realized at t (i.e. from t-1 to t)
# df_rx and df_rx_real therefore "loose" the first date in the sample (2000-02-29 - 2026-01-31)

# ----
# ---- (2) Compute real excess returns + inflation compensation (equation 23)

idx_rx_real_pi = df_rx_real.index.intersection(pi.index)
pi_tmp = pi.loc[idx_rx_real_pi].sort_index()
df_rx_real_tmp = df_rx_real.loc[idx_rx_real_pi].sort_index()

df_rx_real_pi = df_rx_real_tmp.add(pi_tmp, axis=0)
# rx_R + pi at t is 1-month real excess return + inflation compensation from t-1 to t 
# (2002-02-28 - 2025-12-31)

# endregion

# region Initial values: First stage regression (equation 28)

idx_tmp = df_factors.index.intersection(idx_rx_real_pi)
df_X_tm1 = df_factors.loc[idx_tmp].sort_index().shift(1).dropna(how="all")
df_X_t = df_factors.loc[df_X_tm1.index].sort_index()
df_rx_reg = df_rx.loc[df_X_tm1.index].sort_index()
df_rx_real_pi_reg = df_rx_real_pi.loc[df_X_tm1.index].sort_index()
# Index: Monthly (2004-01-31 - 2025-12-31)

# R_{t} = a i' - B Phi_tilde X_{t-1} + B X_{t} + E_{t}
# --> R_{t} = A + B X_{t-1} + C X_{t} + E_{t}

R_pi = pd.concat([df_rx_reg, df_rx_real_pi_reg], axis=1)

X = pd.concat([df_X_tm1.add_prefix("Xtm1_"), df_X_t.add_prefix("Xt_")], axis=1)
X = sm.add_constant(X, has_constant="add")

E_hat = pd.DataFrame(index=R_pi.index, columns=R_pi.columns, dtype=float)
B_hat = pd.DataFrame(index=R_pi.columns, columns=df_X_tm1.columns, dtype=float)
C_hat = pd.DataFrame(index=R_pi.columns, columns=df_X_t.columns, dtype=float)

# --- SUR(1) pt.1: run OLS separately per maturity (columns in pd.R_pi are maturities, rows in model R_pi are maturities)
# B = - (B Phi_tilde) , C = B (equation 23)

for col in R_pi.columns:
    model_sur1 = sm.OLS(R_pi[col].astype(float), X.astype(float), missing="drop")
    res_sur1 = model_sur1.fit()
    E_hat[col] = res_sur1.resid
    coefficients = res_sur1.params 
    B_hat.loc[col] = coefficients.filter(like="Xtm1_").values
    C_hat.loc[col] = coefficients.filter(like="Xt_").values

# Compute sample residual covariance matrix: 
T = E_hat.shape[0]
Sigma_e = (E_hat.T @ E_hat) / T

# Plot residual series per maturity:
E_nominal = E_hat.iloc[:, :11]
E_real = E_hat.iloc[:, 11:]

# Nominal maturities 
plt.figure(figsize=(12, 5))
for col in E_nominal.columns:
    plt.plot(E_nominal.index, E_nominal[col])
plt.axhline(0)
plt.title("OLS Residuals – Nominal Bonds")
plt.xlabel("Date")
plt.ylabel("Residual")
plt.tight_layout()
plt.show()

# Real maturities 
plt.figure(figsize=(12, 5))
for col in E_real.columns:
    plt.plot(E_real.index, E_real[col])
plt.axhline(0)
plt.title("OLS Residuals – Real Bonds")
plt.xlabel("Date")
plt.ylabel("Residual")
plt.tight_layout()
plt.show()

# All maturities 
plt.figure(figsize=(12, 5))
for col in E_hat.columns:
    plt.plot(E_hat.index, E_hat[col])
plt.axhline(0)
plt.title("OLS Residuals – All Bonds")
plt.xlabel("Date")
plt.ylabel("Residual")
plt.tight_layout()
plt.show()

# endregion 

# region Initial values: Get initial value for Phi_tilde by GLS (equation 29)
# Phi_tilde_GLS = - (C_OLS' Sigma_e_OLS^{-1} C_OLS) C_OLS' Sigma_e_OLS^{-1} B_OLS

B_OLS = C_hat.to_numpy(dtype=float)                             # (N,K)
B_PHI_OLS = (-B_hat).to_numpy(dtype=float)                       # (N,K) (sign)
Sigma_e_inverse = np.linalg.inv(Sigma_e.to_numpy(dtype=float))  # (N,N)

# --- SUR(1) pt.2: 
# Phi_tilde_GLS = - (C_OLS' Sigma_e_OLS^{-1} C_OLS) C_OLS' Sigma_e_OLS^{-1} B_OLS
term1 = B_OLS.T @ Sigma_e_inverse @ B_OLS                       # (K,K)
term2 = B_OLS.T @ Sigma_e_inverse @ B_PHI_OLS                   # (K,K)
Phi_GLS = np.linalg.solve(term1, term2)                         # (K,K)

# Check stability of Phi_tilde:
eigvals = np.linalg.eigvals(Phi_GLS)
max_abs = np.max(np.abs(eigvals))
print("Eigenvalues:")
print(eigvals)
if max_abs < 1:
    print("VAR is stable")
else:
    print("VAR is not stable")

# ---- ENFORCE STABILITY OF PHI_TILDE ----

#Phi_GLS, rho_before, rho_after, scale = enforce_stationarity_spectral_radius(Phi_GLS, rho_max=0.995)

#print(f"Spectral radius before = {rho_before:.6f}")
#print(f"Applied scale factor   = {scale:.6f}")
#print(f"Spectral radius after  = {rho_after:.6f}")

#eigvals = np.linalg.eigvals(Phi_GLS)
#print("Eigenvalues after enforcement:")
#print(eigvals)

# ---- END ----

initial_Phi_tilde = pd.DataFrame(Phi_GLS, index=df_X_t.columns, columns=df_X_t.columns)     # (K,K)

# endregion

# region Initial values: Get initial value for alpha and B by GLS (Additional SUR equation 28 with Phi_tilde_GLS)
# R_pi_{t} = (alpha i') - (B Phi_tilde_GLS) X_{t-1} + B X_{t} + E_{t}
# --> R_pi_{t} = (alpha i') + B (X_{t} - Phi_tilde_GLS X_{t-1}) + E_{t}
# --> SUR(2): R_pi_{t} = A + B Z + E_{t}

# Build regressor Z = (X_{t} - Phi_tilde_GLS X_{t-1})
Phi_GLS = initial_Phi_tilde.to_numpy(dtype=float)                      # (K,K)
X_t = df_X_t.transpose().to_numpy(dtype=float)                         # (K,T)
X_tm1 = df_X_tm1.transpose().to_numpy(dtype=float)                     # (K,T)

Z = (X_t - Phi_GLS @ X_tm1)                                            # (K,T)
df_Z = pd.DataFrame(Z.T, index=df_X_t.index, columns=df_X_t.columns)   # (T,K)
X = sm.add_constant(df_Z, has_constant="add")

alpha_GLS = pd.Series(index=R_pi.columns, dtype=float)                 
B_GLS = pd.DataFrame(index=R_pi.columns, columns=df_X_t.columns, dtype=float)

# SUR(2): Run OLS for each maturity (column by column in pd.R_pi)
for col in R_pi.columns:
    model_sur2 = sm.OLS(R_pi[col].astype(float), X.astype(float), missing="drop")
    res_sur2 = model_sur2.fit()

    alpha_GLS.loc[col] = res_sur2.params["const"]
    B_GLS.loc[col] = res_sur2.params.drop("const").values

initial_B = pd.DataFrame(B_GLS, index=R_pi.columns, columns=df_X_t.columns)     # (N,K)
initial_alpha = pd.Series(alpha_GLS, index=R_pi.columns)                        # (N,)

# endregion

# region Initial values: Get intial value for mu_tilde using Phi_tilde_GLS, B_GLS and Sigma_e_OLS (equation 30)
# mu_tilde_GLS = (B_GLS' Sigma_e_OLS^{-1} B_GLS)^{-1} (B_GLS' Sigma_e_OLS^{-1}) (alpha_GLS + 0.5 gamma_GLS)

# --- (1) gamma: (N,1) vector, is a function each B_n, B_n,R and Sigma 

Sigma_X = Sigma.to_numpy(dtype=float)                           # (K,K)
gamma_GLS = pd.Series(index=R_pi.columns, dtype=float)          # (N,)

for col in R_pi.columns:
    b = initial_B.loc[col].to_numpy(dtype=float).reshape(-1,1)  # Extract rows in B (B_n), pd.Series --> numpy vector (K,)
    gamma_GLS.loc[col] = (b.T @ Sigma_X @ b).item()             # Element n in gamma_GLS is scalar ( 1xK KxK Kx1 = 1x1)

# --- (2) equation 30:

Sigma_e_inv = np.linalg.inv(Sigma_e.to_numpy(dtype=float))          # (N,N)
B_GLS = initial_B.to_numpy(dtype=float)                             # (N,K)
alpha_term =  (initial_alpha + 0.5 * gamma_GLS).to_numpy(dtype=float).reshape(-1,1)     # (N,1)

term1_mu = B_GLS.T @ Sigma_e_inv @ B_GLS
term1_mu_inv = np.linalg.inv(term1_mu)
term2_mu = B_GLS.T @ Sigma_e_inv @ alpha_term

mu_tilde_GLS = - (term1_mu_inv @ term2_mu)
initial_mu_tilde = pd.Series(mu_tilde_GLS.flatten(), index=df_X_t.columns)

# endregion

# region Initial values: Get initial values for pi_0 and pi_1 by minimizing SSE observed excess return and model implied excess returns (equation 31)

# ---- Inputs: 
Phi_tilde = initial_Phi_tilde.to_numpy(dtype=float)    # (K,K)
mu_tilde = initial_mu_tilde.to_numpy(dtype=float)      # (K,)
Sigma_X = Sigma.to_numpy(dtype=float)                  # (K,K)
_delta_1 = delta_1.to_numpy(dtype=float)               # (K,)    

df_X_tm1 = df_factors.shift(1).dropna(how="all")
df_X_t = df_factors.loc[df_X_tm1.index].sort_index()
# Index: Monthly (2004-01-31 - 2025-12-31)         

# ---- Starting values for pi_0, pi_1: 
# OLS: pi_{t} = pi_0 + pi_1' X_{t} + u_{t}

df_y_pi = pi.loc[df_X_tm1.index].sort_index()
df_X_pi = df_factors.loc[df_X_tm1.index].sort_index()
df_X_pi = sm.add_constant(df_X_pi, has_constant="add")

pi_res = sm.OLS(df_y_pi, df_X_pi).fit()

pi_0_start = float(pi_res.params["const"])
pi_1_start = pi_res.params[df_X_t.columns].to_numpy(dtype=float)

print(pi_res.summary())

x0 = np.r_[pi_0_start, pi_1_start]

# ---- Functions: 

# ---- 1) Compute B_{n,R} for all months given pi_0 and pi_1 (equation 11): 
# B'_{n,R} = (B_{n-1,R} + pi_1)' Phi_tilde - delta_1' , B_0 = 0   (1,K)
# --> B_{n,R} = Phi_tilde' (B_{n-1,R} + pi_1) - delta_1           (K,1)

def compute_Bn(
        max_maturity: int, 
        Phi: np.ndarray,
        delta_1: np.ndarray,
        pi_1: np.ndarray
) -> np.ndarray:
    K = Phi.shape[0]
    B = np.zeros((max_maturity + 1, K), dtype=float) # +1 because B_{0,R} is included
    for n in range(1, max_maturity + 1):
        B[n] = Phi.T @ (B[n - 1] + pi_1) - delta_1
    return B            # (max_maturity + 1, K), row n is B_{n,R}


# ---- 2) Compute alpha_{n,R} for all months given pi_0, pi_1 and B_{n,R} (equation 19):
# alpha_{n-1,R} = -[ pi_0 + (B_{n-1,R} + pi_1)' mu_tilde + 1/2 (B_{n-1,R} + pi_1)' Sigma_X (B_{n-1,R} + pi_1) ]
# Objective: Given element B_{n,R} in B, compute alpha_{n,R}

def compute_alpha(
        B_n: np.ndarray,    # shape (K,)
        mu: np.ndarray,
        Sigma: np.ndarray, 
        pi_0: float,
        pi_1: np.ndarray
) -> float:
    B_pi = B_n + pi_1
    quadratic_term = B_pi @ Sigma @ B_pi    # 1D array --> no need to transpose
    return - (pi_0 + B_pi @ mu + 0.5 * quadratic_term)


# ---- 3) Compute model implied excess returns given (1), (2), initial values, pi_0 and pi_1 (equation 18):
# rx^{n-1}_{t+1,R} = alpha_{n-1,R} - (B_{n-1,R}+pi_1)' Phi_tilde X_t + B_{n-1,R}' X_{t+1} 
# Indexing: Excess return evaluated at {t+1} is t -> {t+1}. -> df_X_t = X_{t+1} & df_X_tm1 = X_{t} above

def model_implied_rx(
        annual_maturities_in_months: list[int],     # [24,36,...,120]
        X_prev: np.ndarray,                         # X_t in equation 18
        X_current: np.ndarray,                      # X_{t+1} in equation 18
        Phi: np.ndarray,
        mu: np.ndarray, 
        Sigma: np.ndarray,
        delta_1: np.ndarray,
        pi_0: float,
        pi_1: np.ndarray
) -> np.ndarray:
    
    max_maturity = max(annual_maturities_in_months)
    B = compute_Bn(max_maturity, Phi, delta_1, pi_1)

    T, K = X_prev.shape
    rx_R_hat = np.empty((T, len(annual_maturities_in_months)), dtype=float)

    for j, n in enumerate(annual_maturities_in_months):     # -> e.g. first iteration is: j=0, n=24
        b = B[n-1]                                          # B_{n-1,R}
        b_pi = b + pi_1
        alpha = compute_alpha(b, mu, Sigma, pi_0, pi_1)

        # - bpi' Phi X_t = - X_t' Phi' b_pi   (convenient since X_prev is (T,K), not (K,T))
        prev_term = X_prev @ Phi.T @ b_pi

        # b' X_{t+1} = X_{t+1} @ b
        current_term = X_current @ b

        rx_R_hat[:, j] = alpha - prev_term + current_term    # column j, shape(T,)
    
    return rx_R_hat                                         # (T, N_R)


# ---- 4) Compute fitting errors between observed and model implied returns, then stack in (T,) vector: 
# Objective function in optimizer 

def get_error_vector( 
        free_params: np.ndarray,
        annual_maturities_in_months: list[int],
        X_prev: np.ndarray,
        X_current: np.ndarray,
        rx_R_observed: np.ndarray,                          # (T, N_R)
        Phi: np.ndarray,
        mu: np.ndarray,
        Sigma: np.ndarray,
        delta_1: np.ndarray
) -> np.ndarray:
    
    pi_0 = float(free_params[0])
    pi_1 = np.asarray(free_params[1:], dtype=float)

    rx_R_hat = model_implied_rx(
        annual_maturities_in_months = annual_maturities_in_months,
        X_prev = X_prev,
        X_current = X_current,
        Phi = Phi,
        mu = mu,
        Sigma = Sigma, 
        delta_1 = delta_1,
        pi_0 = pi_0,
        pi_1 = pi_1
    )

    errors_matrix = rx_R_observed - rx_R_hat                # (T, N_R)
    # Turn into (1, T*N_R) vector. Each element is the error of a T,N_R combination
    errors_vector = errors_matrix.ravel(order="C")
    return errors_vector


# ---- Set up and run optimizer using least squares algorithm:

# Format arrays: 
real_maturities = list(range(24, 121, 12))      # [24,36,...,120]

rx_real_cols = [f"rx_real_1m_{n}m" for n in real_maturities]
df_rx_real_reg = df_rx_real.loc[df_X_tm1.index, rx_real_cols].sort_index()

X_prev = df_X_tm1.to_numpy(dtype=float)       # (T,K) : X_{t-1}
X_current = df_X_t.to_numpy(dtype=float)         # (T,K) : X_t
rx_real_obs = df_rx_real_reg.to_numpy(float)  # (T, NR) where NR=len(real_maturities)

N_R = len(real_maturities)                      # number of return series 

# Least squares for {pi_0, pi_1} (equation 31) (solves NL least squares): 

lsq_res = least_squares(
    fun=get_error_vector,
    x0=x0,
    jac="3-point",                  # optional: 2-point, simpler 
    method="trf",
    x_scale="jac",                  # optional: scales steps using the Jacobian
    args=(
        real_maturities,
        X_prev,
        X_current,
        rx_real_obs,
        Phi_tilde,
        mu_tilde,
        Sigma_X, 
        _delta_1
    ),
    max_nfev = 2000,
    ftol=1e-12,                      # relative change in cost for termination 
    xtol=1e-12,                      # relative change in parameters for termination 
    gtol=1e-12                       # termination by gradient norm 
)

pi_0_hat = float(lsq_res.x[0])
pi_1_hat = lsq_res.x[1:]

print("Estimation status:", lsq_res.status, lsq_res.message)
print("pi0_hat =", pi_0_hat)
print("pi1_hat =", pi_1_hat)
print("SSE =", np.sum(lsq_res.fun ** 2))

# ---- PLOTS: (1) actual vs estimated monthly inflation 

actual_pi = pi.loc[df_X_tm1.index].sort_index()                 
df_pifactors  = df_factors.loc[df_y_pi.index].sort_index()         

# pi_hat_t = pi_0_hat + pi_1_hat' X_t
pi_hat = pd.Series(
    pi_0_hat + df_pifactors.to_numpy(dtype=float) @ pi_1_hat,
    index=df_y_pi.index,
    name="pi_hat"
)

plt.figure(figsize=(10, 5))
plt.plot(df_y_pi.index, df_y_pi.to_numpy(dtype=float), label="Actual monthly inflation")
plt.plot(pi_hat.index, pi_hat.to_numpy(dtype=float), linestyle="--", label="Estimated monthly inflation")
plt.title("Actual vs Estimated Monthly Inflation")
plt.xlabel("Time")
plt.ylabel("Monthly inflation (decimal)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ---- PLOTS: (2) (actual 1-month real excess return) - (fitted 1-month real excess return) by maturity

rx_hat = model_implied_rx(
    annual_maturities_in_months=real_maturities,
    X_prev=X_prev,
    X_current=X_current,
    Phi=Phi_tilde,
    mu=mu_tilde,
    Sigma=Sigma_X,
    delta_1=_delta_1,
    pi_0=pi_0_hat,
    pi_1=pi_1_hat
)  

residuals = rx_real_obs - rx_hat 
dates = df_X_tm1.index 

plt.figure(figsize=(12, 6))
for j, n in enumerate(real_maturities):
    plt.plot(dates, residuals[:, j], label=f"{n}m")

plt.title("Residuals: Actual − Fitted Real Excess Returns (by Maturity)")
plt.xlabel("Time")
plt.ylabel("Residual (decimal)")
plt.grid(True)
plt.legend(title="Maturity", ncol=2, fontsize=8)
plt.tight_layout()
plt.show()

# ---- PLOTS: (3) mean fitting error by maturity

mean_errors = residuals.mean(axis=0)   # shape (N_R,)

plt.figure(figsize=(8, 5))
plt.plot(real_maturities, mean_errors, marker="o")

plt.axhline(0.0)  # reference line at zero error
plt.title("Mean Fitting Error by Maturity")
plt.xlabel("Maturity (months)")
plt.ylabel("Mean residual (actual - fitted)")
plt.grid(True)

plt.tight_layout()
plt.show()


# endregion



