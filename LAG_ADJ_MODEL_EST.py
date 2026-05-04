import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from scipy.optimize import minimize

# Three lag-adjusted specifications with targeted il_PC2 yield-space identification:
# Spec 1 — 5-factor adj YID : nom_PC1-3,           il_PC1-2 (carry-adjusted)
# Spec 2 — 6-factor adj YID : nom_PC1-3, liq,      il_PC1-2 (carry-adjusted)
# Spec 3 — 7-factor adj YID : nom_PC1-3, liq, inf, il_PC1-2 (carry-adjusted)
#
# All three specifications include both il_PC1 and il_PC2, unlike the original
# LAG_ADJ_MODEL_EST.py which dropped il_PC2 due to return-space near-collinearity.
#
# IDENTIFICATION CHANGE (Steps 1-2):
# In the standard Abrahams et al. approach, the B matrix (contemporaneous factor
# loadings on returns) is estimated freely from the joint return regression. With
# carry-adjusted yields, the carry adjustment adds a term D_t - D_{t+1} to all IL
# excess returns equally. This maturity-independent shift contaminates il_PC2_adj's
# return-space loading, making it nearly parallel to il_PC1_adj. The result is a
# near-singular BtSiB and an explosive Phi_tilde.
#
# The fix is surgical: only the il_PC2 column of B is identified from the adj SGBi
# yield cross-section. All other columns (il_PC1 and all nominal) remain estimated
# from returns as in the standard approach. The affine model implies:
#
#   B_{n-1, il_PC2}^{return} = -(n-1)/12 * beta_{n-1, il_PC2}^{yield}
#
# where beta_{n-1, il_PC2}^{yield} is the OLS loading of y_{n-1,t}^{adj} on
# il_PC2. In yield space il_PC2 is a genuine slope factor, free of the
# carry-induced flattening that distorts its return-space loading.
#
# il_PC2 is always the last factor in the factor vector (index K-1).
# BPhit is re-estimated from the return residuals after fixing B[:,K-1] from yields.
# Steps 3-5 are identical to LAG_ADJ_MODEL_EST.py.
# Run FACTORS.py first to populate the *_adj sheets in Factors.xlsx.

# region DATA

DIR_NOM  = "../Curve estimation/NOMINAL_CURVE_ESTIMATION"
DIR_IL   = "../Curve estimation/LINKER_CURVE_ESTIMATION"
DIR_CURV = "../Curve estimation"
DIR_FACT = "../Factors"

# ---- Raw files

raw_sgb    = pd.read_excel(f"{DIR_NOM}/zero_yields_SGB.xlsx",
                            sheet_name="zero_yields")
raw_il_adj = pd.read_excel(f"{DIR_IL}/zero_yields_SGBIL_lag_adj.xlsx",
                            sheet_name="zero_yields_lag_adj")
raw_sr     = pd.read_excel(f"{DIR_CURV}/statsobligationer_data.xlsx",
                            sheet_name="korta räntor riksbanken")[["date", "SSVX_1m"]]
raw_kpi    = pd.read_excel(f"{DIR_CURV}/kpi_data.xlsx",
                            sheet_name="basår 1980")[["date", "log month"]]

raw_sgb["date"]    = pd.to_datetime(raw_sgb["date"])
raw_il_adj["date"] = pd.to_datetime(raw_il_adj["date"])
raw_sr["date"]     = pd.to_datetime(raw_sr["date"])
raw_kpi["date"]    = pd.to_datetime(raw_kpi["date"], errors="coerce")

# ---- Align on year-month period index and convert to annual CC rates

def _ym(df):
    """Set a year-month PeriodIndex from the date column."""
    return df.assign(ym=df["date"].dt.to_period("M")).set_index("ym").sort_index()

sgb_cc    = _ym(raw_sgb).drop(columns="date")
sgb_cc    = sgb_cc[[c for c in sgb_cc.columns if c.startswith("y_")]].apply(np.log1p)

il_adj_cc = _ym(raw_il_adj).drop(columns="date")
il_adj_cc = il_adj_cc[[c for c in il_adj_cc.columns if c.startswith("y_")]].apply(np.log1p)

sr_cc     = (_ym(raw_sr)
             .drop(columns="date")
             .assign(r_cc=lambda d: np.log1p(d["SSVX_1m"] / 100.0))[["r_cc"]])

kpi_m     = (_ym(raw_kpi)
             .drop(columns="date")
             .rename(columns={"log month": "pi_m"}))

print("Data ranges loaded:")
print(f"  SGB  zero yields     : {sgb_cc.index.min()} – {sgb_cc.index.max()}")
print(f"  SGBi adj zero yields : {il_adj_cc.index.min()} – {il_adj_cc.index.max()}")
print(f"  Short rate           : {sr_cc.index.min()} – {sr_cc.index.max()}")
print(f"  Monthly CPI log      : {kpi_m.index.min()} – {kpi_m.index.max()}")

# endregion

# region CONFIGURATION

SAMPLE_START = pd.Timestamp("2004-01-01")
SAMPLE_END   = pd.Timestamp("2025-12-31")
INDEX_LAG    = 3    # indexation lag in months

est_ym     = pd.period_range(start=SAMPLE_START.to_period("M"),
                              end=SAMPLE_END.to_period("M"), freq="M")
T          = len(est_ym)
dates_plot = est_ym.to_timestamp()

NOM_RET_MATS = [12, 24, 36, 48, 60, 72, 84, 96, 108, 120]
IL_RET_MATS  = [24, 36, 48, 60, 72, 84, 96, 108, 120]

BEI_MATS = [24, 60, 120]
BEI_LABS  = {24: "2y", 60: "5y", 120: "10y"}

N_STARTS      = 40
SEED          = 42
PI_0_BAND_ANN = 0.02
PI_1_SCALE    = 3.0
PI_1_FLOOR    = 0.003
RANK_THRESH   = 1e-8

FIG_W   = 5.5
FIG_H   = 3.4
FIG_DPI = 300
FIG_FMT = "pdf"

TICK_YEARS = [2004, 2008, 2012, 2016, 2020, 2024]
TICK_DATES = [pd.Timestamp(f"{y}-01-01") for y in TICK_YEARS]

C_NOM_3 = ["#4D96C0", "#1A5EA8", "#0D2B4E"]
C_EI_3  = ["#7DC99A", "#2D9F5D", "#1A5C38"]
C_IRP_3 = ["#E8948F", "#C0453A", "#7B0000"]
C_LIQ_3 = ["#F0A868", "#C4620A", "#7D3C00"]
C_ZERO  = "#666666"

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":         11,
    "axes.labelsize":    11,
    "axes.titlesize":    11,
    "legend.fontsize":   10,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size":  4.0,
    "ytick.major.size":  4.0,
})

# endregion

# region SHARED DATA OBJECTS

sgb_est    = sgb_cc.loc[est_ym]
il_adj_est = il_adj_cc.loc[est_ym]
r_arr      = sr_cc.loc[est_ym, "r_cc"].values
pi_arr     = kpi_m.loc[est_ym, "pi_m"].values

assert not np.isnan(r_arr).any(),  "NaN in short rate"
assert not np.isnan(pi_arr).any(), "NaN in CPI"

# ---- Build carry term D_t = pi_t + pi_{t-1} + pi_{t-2}

pre_start = est_ym[0] - INDEX_LAG
per_ext   = pd.period_range(start=pre_start, end=est_ym[-1], freq="M")
pi_ext    = kpi_m.loc[per_ext, "pi_m"].values

assert not np.isnan(pi_ext).any(), (
    f"KPI 'log month' missing in pre-sample. Need data back to {pre_start}."
)

D_full = pd.Series(pi_ext).rolling(window=INDEX_LAG).sum().values
D_t    = D_full[INDEX_LAG : INDEX_LAG + T]

assert not np.isnan(D_t).any() and len(D_t) == T

print(f"\n  Carry D_t (ann. mean): {D_t.mean() * 12 / INDEX_LAG * 100:+.2f}% p.a.")

t_idx  = est_ym[:-1]
t1_idx = est_ym[1:]
T_ret  = len(t_idx)
r_t    = r_arr[:-1]
pi_t1  = pi_arr[1:]

# Nominal excess holding-period returns
R_nom = np.column_stack([
    n / 12 * sgb_est.loc[t_idx,  f"y_{n}m"].values
    - (n - 1) / 12 * sgb_est.loc[t1_idx, f"y_{n-1}m"].values
    - 1 / 12 * r_t
    for n in NOM_RET_MATS
])

# Pure real excess returns on carry-adjusted SGBi
R_il_adj_raw = np.column_stack([
    n / 12 * il_adj_est.loc[t_idx,  f"y_{n}m"].values
    - (n - 1) / 12 * il_adj_est.loc[t1_idx, f"y_{n-1}m"].values
    - 1 / 12 * r_t
    for n in IL_RET_MATS
])

# Inflation-augmented adjusted returns for the joint return regression
R_il_adj_pi = R_il_adj_raw + pi_t1[:, None]

# Joint return matrix
R_all_adj = np.hstack([R_nom, R_il_adj_pi])
N_nom     = R_nom.shape[1]
N_il      = R_il_adj_raw.shape[1]
N         = N_nom + N_il

# Carry-adjusted observed IL yields for fitting error computation
il_adj_obs_full = il_adj_est[[f"y_{n}m" for n in IL_RET_MATS]].values

print(f"\nEstimation period : {est_ym[0]} – {est_ym[-1]}   T = {T}")
print(f"Returns           : T_ret = {T_ret},  N = {N}  ({N_nom} nominal + {N_il} IL adj)")

# endregion

# region ESTIMATION FUNCTIONS

# ---- Affine pricing recursions (unchanged from LAG_ADJ_MODEL_EST)

def _nom_ab(Phi_t, mu_t, Sig, d0m, d1m, mats, K):
    """
    Nominal bond affine pricing recursion at a monthly time step.
    Returns a dict mapping each maturity in mats to the pair (A_n, B_n).
    """
    tgt = set(mats)
    A   = 0.0
    B   = np.zeros(K)
    AB  = {}
    for step in range(1, max(mats) + 1):
        A = A + B @ mu_t + 0.5 * B @ Sig @ B - d0m
        B = Phi_t.T @ B - d1m
        if step in tgt:
            AB[step] = (A, B.copy())
    return AB


def _il_ab(pi0, pi1, Phi_t, mu_t, Sig, d0m, d1m, mats, K):
    """
    Fully-indexed IL bond affine pricing recursion at a monthly time step.
    Used to anchor the BEI decomposition on the true real yield.
    """
    tgt = set(mats)
    A   = 0.0
    B   = np.zeros(K)
    AB  = {}
    for step in range(1, max(mats) + 1):
        b = B + pi1
        A = A + b @ mu_t + 0.5 * b @ Sig @ b - d0m + pi0
        B = Phi_t.T @ b - d1m
        if step in tgt:
            AB[step] = (A, B.copy())
    return AB


def _il_ab_lag(pi0, pi1, Phi_t, mu_t, Sig, d0m, d1m, mats, K, lag):
    """
    Artificial-bond affine pricing recursion for an IL bond with a
    lag-month indexation lag.
    """
    assert min(mats) >= lag, "All maturities must be at least as long as the lag"
    tgt = set(mats)
    A   = 0.0
    B   = np.zeros(K)
    AB  = {}
    for step in range(1, lag + 1):
        A = A + B @ mu_t + 0.5 * B @ Sig @ B - d0m
        B = Phi_t.T @ B - d1m
        if step in tgt:
            AB[step] = (A, B.copy())
    for step in range(lag + 1, max(mats) + 1):
        b = B + pi1
        A = A + b @ mu_t + 0.5 * b @ Sig @ b - d0m + pi0
        B = Phi_t.T @ b - d1m
        if step in tgt:
            AB[step] = (A, B.copy())
    return AB


def _model_yields(AB, X, mats):
    """Convert affine coefficients to model-implied annual CC yields."""
    return np.column_stack([
        -(12.0 / n) * (AB[n][0] + X @ AB[n][1]) for n in mats
    ])


def _model_rx_il_lag(pi0, pi1, Phi_t, mu_t, Sig, d0m, d1m,
                     X_lag, X_cur, mats, K, lag):
    """
    Model-implied pure real excess returns for the artificial IL bond.
    """
    assert min(mats) > lag, "All IL maturities must exceed the indexation lag"
    T_ret   = X_lag.shape[0]
    RX_pred = np.zeros((T_ret, len(mats)))
    mats_s  = set(mats)
    B       = np.zeros(K)
    for step in range(1, lag + 1):
        B = Phi_t.T @ B - d1m
    for step in range(lag + 1, max(mats) + 1):
        b = B + pi1
        if step in mats_s:
            j     = mats.index(step)
            alpha = -(pi0 + b @ mu_t + 0.5 * b @ Sig @ b)
            RX_pred[:, j] = alpha - X_lag @ (Phi_t.T @ b) + X_cur @ B
        B = Phi_t.T @ b - d1m
    return RX_pred


# ---- IL PC2 yield-space identification

def _build_B_il_pc2_from_yields(X, K):
    """
    Estimate only the il_PC2 column of B from adj SGBi yield regressions.

    il_PC2 is always the last factor (index K-1). All other B columns
    (il_PC1 and all nominal factors) remain estimated from returns.

    The affine model implies that for each return maturity n, the loading
    of the excess return on il_PC2 at time t+1 equals:

        B_{n-1, il_PC2} = -(n-1)/12 * beta_{n-1, il_PC2}^yield

    where beta_{n-1, il_PC2}^yield is the OLS coefficient from regressing
    the (n-1)-period adj SGBi yield on il_PC2.

    In yield space, il_PC2 is a genuine slope factor with a clearly
    distinct loading profile (crossing zero around 48-60m). This is free
    of the carry-induced flattening that makes its return-space loading
    nearly parallel to il_PC1.

    Returns B_il_pc2 of shape (N,) — the full column for il_PC2 (both
    nominal rows, which are set to zero, and IL rows from yields).
    Nominal rows carry no information about il_PC2 pricing so they remain
    as estimated by the return regression; this function returns only the
    IL portion (N_il values) and the caller assembles the full column.

    Returns B_il_pc2_col of shape (N_il,) for the IL rows only.
    """
    Z_yield        = np.column_stack([np.ones(T), X])   # (T, 1+K)
    B_il_pc2_col   = np.zeros(len(IL_RET_MATS))
    il_pc2_pos     = K - 1    # il_PC2 is always last factor; Z_yield[:,K] = X[:,K-1]
    for i, n in enumerate(IL_RET_MATS):
        y    = il_adj_est[f"y_{n-1}m"].values
        beta = np.linalg.lstsq(Z_yield, y, rcond=None)[0]
        # beta[1:] are loadings on X[:,0]...X[:,K-1]; il_PC2 is beta[K]
        B_il_pc2_col[i] = -(n - 1) / 12.0 * beta[il_pc2_pos + 1]
    return B_il_pc2_col


# ---- Core estimation

def run_estimation_yieldid(X, R_all_adj, R_il_adj_raw, r_cc, pi_arr, K):
    """
    Run the five estimation steps for one lag-adjusted specification,
    using yield-space identification for the il_PC2 column of B only.

    Step 0: physical VAR by OLS on demeaned factors.
    Step 1 (modified):
      - Full joint return regression run first to estimate all B columns
        from returns (standard approach).
      - Only the il_PC2 column (index K-1) is then replaced with the
        yield-space estimate from _build_B_il_pc2_from_yields.
      - BPhit is re-estimated from the residuals R - B*X_cur projected
        onto X_lag, conditional on the updated B.
    Step 2: GLS recovery of Phi_tilde using the hybrid B.
    Step 3: risk-neutral drift mu_tilde and market prices of risk.
    Step 4: short-rate equation by OLS.
    Step 5: inflation parameters by multi-start L-BFGS-B.
    """
    T_ret = X.shape[0] - 1
    X_lag = X[:-1]
    X_cur = X[1:]

    # Step 0 — physical VAR
    mu_X  = X.mean(axis=0)
    X_dm  = X - mu_X
    Phi   = np.linalg.lstsq(X_dm[:-1], X_dm[1:], rcond=None)[0].T
    V_hat = X_dm[1:] - X_dm[:-1] @ Phi.T
    Sigma = V_hat.T @ V_hat / T_ret

    # Step 1 — hybrid B: all columns from returns, il_PC2 column from yields
    #
    # Full joint return regression (standard)
    Z         = np.column_stack([np.ones(T_ret), X_lag, X_cur])
    theta_ols = np.linalg.lstsq(Z, R_all_adj, rcond=None)[0]
    B_ret     = theta_ols[K + 1:, :].T    # (N, K) — all from returns

    # Yield-space il_PC2 column (IL rows only)
    B_il_pc2_col = _build_B_il_pc2_from_yields(X, K)   # shape (N_il,)

    # Assemble hybrid B: copy return B, replace il_PC2 column in IL rows
    B_ols                      = B_ret.copy()
    B_ols[N_nom:, K - 1]       = B_il_pc2_col    # only IL rows, only il_PC2

    # Re-estimate BPhit conditional on the hybrid B:
    # R - B_ols * X_cur  ~  [1, X_lag]  =>  BPhit from residual regression
    R_resid   = R_all_adj - X_cur @ B_ols.T
    Z_lag     = np.column_stack([np.ones(T_ret), X_lag])
    theta_lag = np.linalg.lstsq(Z_lag, R_resid, rcond=None)[0]
    BPhit_ols = -theta_lag[1:, :].T      # (N, K)

    # Residuals and error covariance
    E_hat   = R_resid - Z_lag @ theta_lag
    Sigma_e = E_hat.T @ E_hat / T_ret

    # Step 2 — GLS recovery of Phi_tilde
    Se_inv    = np.linalg.pinv(Sigma_e, rcond=RANK_THRESH)
    BtSiB     = B_ols.T @ Se_inv @ B_ols
    Phi_tilde = np.linalg.solve(BtSiB, B_ols.T @ Se_inv @ BPhit_ols)

    # Stationarity check
    ev_Q = np.abs(np.linalg.eigvals(Phi_tilde))
    if ev_Q.max() >= 1.0:
        print(f"  WARNING: Phi_tilde max|EV| = {ev_Q.max():.6f}  *** NON-STATIONARY ***")

    Z2        = np.column_stack([np.ones(T_ret), X_cur - X_lag @ Phi_tilde.T])
    theta2    = np.linalg.lstsq(Z2, R_all_adj, rcond=None)[0]
    alpha_gls = theta2[0, :]
    B_gls     = theta2[1:, :].T

    # Step 3 — risk-neutral drift and market prices of risk
    gamma    = np.array([B_gls[i] @ Sigma @ B_gls[i] for i in range(N)])
    BtSiB2   = B_gls.T @ Se_inv @ B_gls
    mu_tilde = -np.linalg.solve(BtSiB2,
                                 B_gls.T @ Se_inv @ (alpha_gls + 0.5 * gamma))
    lambda_0 = (np.eye(K) - Phi) @ mu_X - mu_tilde
    lambda_1 = Phi - Phi_tilde

    # Step 4 — short-rate equation
    Z_sr    = np.column_stack([np.ones(X.shape[0]), X])
    sr_ols  = np.linalg.lstsq(Z_sr, r_cc, rcond=None)[0]
    delta_0 = sr_ols[0]
    delta_1 = sr_ols[1:]
    d0m     = delta_0 / 12
    d1m     = delta_1 / 12
    r2_sr   = 1.0 - np.var(r_cc - Z_sr @ sr_ols) / np.var(r_cc)

    # Step 5 — inflation parameters via artificial-bond return criterion
    Z_pi     = np.column_stack([np.ones(X.shape[0]), X])
    pi_ols   = np.linalg.lstsq(Z_pi, pi_arr, rcond=None)[0]
    pi_0_ols = pi_ols[0]
    pi_1_ols = pi_ols[1:]

    pi_0_band = PI_0_BAND_ANN / 12
    dk        = np.maximum(PI_1_SCALE * np.abs(pi_1_ols), PI_1_FLOOR)
    bounds    = ([(pi_0_ols - pi_0_band, pi_0_ols + pi_0_band)]
                 + list(zip(pi_1_ols - dk, pi_1_ols + dk)))
    p0_ols    = np.concatenate([[pi_0_ols], pi_1_ols])

    def _sse(p):
        rx_pred = _model_rx_il_lag(
            p[0], p[1:], Phi_tilde, mu_tilde, Sigma,
            d0m, d1m, X_lag, X_cur, IL_RET_MATS, K, INDEX_LAG
        )
        return float(np.sum((R_il_adj_raw - rx_pred) ** 2))

    sse_init = _sse(p0_ols)
    rng      = np.random.default_rng(SEED)
    best     = None
    for s in range(N_STARTS):
        p0 = (p0_ols.copy() if s == 0
              else np.array([rng.uniform(lo, hi) for lo, hi in bounds]))
        res = minimize(_sse, p0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 8_000, "ftol": 1e-14, "gtol": 1e-9})
        if best is None or res.fun < best.fun:
            best = res

    pi_0          = best.x[0]
    pi_1          = best.x[1:]
    sse_reduction = 100.0 * (1.0 - best.fun / sse_init) if sse_init > 0 else 0.0

    atol       = 1e-7
    bound_hits = []
    if best.x[0] <= bounds[0][0] + atol or best.x[0] >= bounds[0][1] - atol:
        bound_hits.append("pi_0")
    for i in range(K):
        lo, hi = bounds[i + 1]
        if best.x[i + 1] <= lo + atol or best.x[i + 1] >= hi - atol:
            bound_hits.append(f"pi_1[{i}]")

    sv_BtSiB = np.linalg.svd(BtSiB, compute_uv=False)

    return dict(
        mu_X=mu_X, Phi=Phi, Sigma=Sigma, V_hat=V_hat,
        B_ols=B_ols, B_ret=B_ret, B_il_pc2_col=B_il_pc2_col,
        E_hat=E_hat, Sigma_e=Sigma_e, BtSiB=BtSiB, sv_BtSiB=sv_BtSiB,
        Phi_tilde=Phi_tilde, mu_tilde=mu_tilde,
        B_gls=B_gls, alpha_gls=alpha_gls,
        lambda_0=lambda_0, lambda_1=lambda_1,
        delta_0=delta_0, delta_1=delta_1, d0m=d0m, d1m=d1m, r2_sr=r2_sr,
        pi_0=pi_0, pi_1=pi_1, pi_0_ols=pi_0_ols, pi_1_ols=pi_1_ols,
        sse_reduction=sse_reduction, pi_converged=best.success,
        bound_hits=bound_hits,
    )


# ---- BEI decomposition (unchanged from LAG_ADJ_MODEL_EST)

def decompose_bei_lag(est, X, liq_idx=None):
    """
    Compute model BEI and decompose it into expected inflation, the
    inflation risk premium, and optionally a liquidity premium.
    """
    K   = est["Phi"].shape[0]
    d0m = est["d0m"]
    d1m = est["d1m"]
    pi0 = est["pi_0"]
    pi1 = est["pi_1"]

    # Q-measure: artificial-bond recursion for yield fitting
    AB_il_lag_Q = _il_ab_lag(pi0, pi1, est["Phi_tilde"], est["mu_tilde"],
                              est["Sigma"], d0m, d1m, IL_RET_MATS, K, INDEX_LAG)
    il_lag_Q    = _model_yields(AB_il_lag_Q, X, IL_RET_MATS)

    # Q-measure: nominal and fully-indexed IL for BEI
    AB_nom_Q = _nom_ab(est["Phi_tilde"], est["mu_tilde"], est["Sigma"],
                       d0m, d1m, NOM_RET_MATS, K)
    AB_il_Q  = _il_ab(pi0, pi1, est["Phi_tilde"], est["mu_tilde"],
                      est["Sigma"], d0m, d1m, IL_RET_MATS, K)
    nom_Q    = _model_yields(AB_nom_Q, X, NOM_RET_MATS)
    il_Q     = _model_yields(AB_il_Q,  X, IL_RET_MATS)

    # P-measure: nominal and fully-indexed IL for expected inflation
    mu_P     = (np.eye(K) - est["Phi"]) @ est["mu_X"]
    AB_nom_P = _nom_ab(est["Phi"], mu_P, est["Sigma"], d0m, d1m, BEI_MATS, K)
    AB_il_P  = _il_ab(pi0, pi1, est["Phi"], mu_P, est["Sigma"],
                      d0m, d1m, BEI_MATS, K)
    nom_P    = _model_yields(AB_nom_P, X, BEI_MATS)
    il_P     = _model_yields(AB_il_P,  X, BEI_MATS)

    nom_obs = sgb_est[[f"y_{n}m" for n in NOM_RET_MATS]].values

    out = dict(
        nom_Q=nom_Q, il_Q=il_lag_Q,
        nom_err_bp=(nom_obs         - nom_Q)    * 1e4,
        il_err_bp =(il_adj_obs_full - il_lag_Q) * 1e4,
        BEI={}, EI={}, IRP={}, LIQ={}, IRP_adj={}, raw_BEI={},
    )

    for k, n in enumerate(BEI_MATS):
        jn  = NOM_RET_MATS.index(n)
        ji  = IL_RET_MATS.index(n)
        bei = nom_Q[:, jn] - il_Q[:, ji]
        ei  = nom_P[:, k]  - il_P[:, k]
        out["BEI"][n]     = bei
        out["EI"][n]      = ei
        out["IRP"][n]     = bei - ei
        out["raw_BEI"][n] = (sgb_est[f"y_{n}m"].values
                             - il_adj_est[f"y_{n}m"].values)

    if liq_idx is not None:
        L_t = X[:, liq_idx]
        for n in BEI_MATS:
            liq = -(12.0 / n) * (AB_nom_Q[n][1][liq_idx] - AB_il_Q[n][1][liq_idx]) * L_t
            out["LIQ"][n]     = liq
            out["IRP_adj"][n] = out["IRP"][n] - liq
    else:
        for n in BEI_MATS:
            out["LIQ"][n]     = np.zeros(T)
            out["IRP_adj"][n] = out["IRP"][n]

    return out


# ---- Diagnostic helpers

def _lb_pval(x, lag):
    n  = len(x)
    ac = np.array([np.corrcoef(x[:n - k], x[k:])[0, 1] for k in range(1, lag + 1)])
    Q  = n * (n + 2) * np.sum(ac ** 2 / (n - np.arange(1, lag + 1)))
    return float(stats.chi2.sf(Q, df=lag))


def _lb_rejrate(E, lags=(1, 6, 12)):
    ns = E.shape[1]
    return {lag: sum(_lb_pval(E[:, i], lag) < 0.05 for i in range(ns)) / ns
            for lag in lags}


def print_diagnostics(est, bei, factor_names, spec_label):
    """
    Print a compact diagnostic block for one specification.
    Includes a yield-ID specific section reporting how much the IL B
    rows changed relative to the return-estimated values.
    """
    K = len(factor_names)

    print(f"\n{'='*60}")
    print(f"  Diagnostics — {spec_label}")
    print(f"{'='*60}")

    # Stationarity
    print("\n  Stationarity")
    for label, M in [("Phi  (P)", est["Phi"]), ("Phi_tilde  (Q)", est["Phi_tilde"])]:
        ev  = np.sort(np.abs(np.linalg.eigvals(M)))[::-1]
        flg = ("  *** NON-STATIONARY" if ev[0] >= 1.0
               else "  near unit root"    if ev[0] >= 0.99
               else "  high persistence"  if ev[0] >= 0.97
               else "")
        print(f"    {label:18s}  rho = {ev[0]:.6f}{flg}")
        print(f"      eigenvalues: {' '.join(f'{v:.4f}' for v in ev)}")

    # BtSiB conditioning
    sv = est["sv_BtSiB"]
    print(f"\n  BtSiB conditioning  (yield-ID hybrid B)")
    print(f"    min SV = {sv[-1]:.4f}   max SV = {sv[0]:.4f}   "
          f"cond = {sv[0]/sv[-1]:.2e}")

    # Identification
    print("\n  Identification  (rank of B_gls must equal K)")
    sv2 = np.linalg.svd(est["B_gls"], compute_uv=False)
    rk  = int(np.sum(sv2 > sv2.max() * 1e-10))
    flg = "" if rk == K else "  *** RANK DEFICIENT"
    print(f"    rank(B_gls) = {rk}/{K}   min sv = {sv2.min():.3e}{flg}")

    # Yield-ID: il_PC2 column shift
    print("\n  il_PC2 column — return-estimated vs yield-estimated (x1000)")
    print(f"  {'Mat':>5}  {'il_PC1 ret':>12}  {'il_PC2 ret':>12}  {'il_PC2 yld':>12}  {'zero-cross':>10}")
    for i, n in enumerate(IL_RET_MATS):
        lbl = f"{n//12}y"
        b1r = est["B_ret"][N_nom + i, K - 2] * 1e3   # il_PC1 from returns (unchanged)
        b2r = est["B_ret"][N_nom + i, K - 1] * 1e3   # il_PC2 from returns
        b2y = est["B_il_pc2_col"][i] * 1e3            # il_PC2 from yields
        cross = "← zero cross" if i > 0 and (b2y * est["B_il_pc2_col"][i-1] < 0) else ""
        print(f"  {lbl:>5}  {b1r:>12.3f}  {b2r:>12.3f}  {b2y:>12.3f}  {cross}")

    # Return error autocorrelation
    print("\n  Return pricing error autocorrelation")
    rr = _lb_rejrate(est["E_hat"])
    for lag, rate in rr.items():
        flg = "  ***" if rate > 0.25 else ""
        print(f"    Lag {lag:>2d}:  {rate:.0%} rejection rate{flg}")

    # VAR residual autocorrelation
    print("\n  VAR residual autocorrelation")
    rr2 = _lb_rejrate(est["V_hat"])
    for lag, rate in rr2.items():
        flg = "  ***" if rate > 0.25 else ""
        print(f"    Lag {lag:>2d}:  {rate:.0%} rejection rate{flg}")

    # Short rate
    print("\n  Short rate fit")
    flg = "" if est["r2_sr"] > 0.90 else "  moderate" if est["r2_sr"] > 0.70 else "  *** POOR"
    print(f"    R² = {est['r2_sr']:.4f}{flg}")
    print(f"    delta_0 = {est['delta_0'] * 100:.3f}% p.a.")
    for i, nm in enumerate(factor_names):
        print(f"    delta_1[{nm}] = {est['delta_1'][i]:+.5f}")

    # Inflation optimisation
    print("\n  Inflation parameter optimisation  (artificial-bond return SSE)")
    print(f"    pi_0 = {est['pi_0'] * 12 * 100:.3f}% p.a.  "
          f"(OLS start: {est['pi_0_ols'] * 12 * 100:.3f}% p.a.)")
    print(f"    SSE reduction from OLS start: {est['sse_reduction']:.1f}%")
    print(f"    Converged: {est['pi_converged']}   "
          f"Bound hits: {est['bound_hits'] if est['bound_hits'] else 'none'}")

    # Yield fitting errors
    print("\n  Yield fitting errors (bp)  — IL vs carry-adjusted observed")
    print(f"  {'Mat':>5}  {'NOM mean':>9}  {'NOM RMSE':>9}  "
          f"{'IL mean':>8}  {'IL RMSE':>8}")
    for j, n in enumerate(NOM_RET_MATS):
        lbl = f"{n // 12}y" if n % 12 == 0 else f"{n}m"
        en  = bei["nom_err_bp"][:, j]
        if n in IL_RET_MATS:
            ei   = bei["il_err_bp"][:, IL_RET_MATS.index(n)]
            s_il = (f"{ei.mean():>8.2f}  "
                    f"{np.sqrt(np.mean(ei**2)):>8.2f}"
                    + ("***" if np.sqrt(np.mean(ei**2)) > 20 else ""))
        else:
            s_il = "        —         —"
        flg = "***" if np.sqrt(np.mean(en**2)) > 20 else ""
        print(f"  {lbl:>5}  {en.mean():>9.2f}  "
              f"{np.sqrt(np.mean(en**2)):>8.2f}{flg}  {s_il}")
    print(f"  {'RMSE':>5}  {'':>9}  "
          f"{np.sqrt(np.mean(bei['nom_err_bp']**2)):>9.2f}  "
          f"{'':>8}  {np.sqrt(np.mean(bei['il_err_bp']**2)):>8.2f}")
    print()


# ---- Plot helpers (unchanged from LAG_ADJ_MODEL_EST)

def _style_ax(ax):
    ax.set_facecolor("white")
    ax.yaxis.grid(True, color="#CCCCCC", linewidth=0.5, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)
        sp.set_color("black")
    ax.tick_params(axis="both", which="major", direction="in",
                   top=True, right=True, pad=5)


def _fmt_xaxis(ax):
    ax.set_xticks(TICK_DATES)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=0)
    ax.set_xlim(dates_plot[0], dates_plot[-1])
    ax.margins(x=0)


def _save(fig, out_dir, stem):
    path = os.path.join(out_dir, f"{stem}.{FIG_FMT}")
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {path}")


def _finalise_yaxis(ax, tick_spacing, label_spacing=None):
    if label_spacing is None:
        label_spacing = tick_spacing
    lo, hi = ax.get_ylim()
    lo_snap = np.floor(lo / tick_spacing) * tick_spacing - tick_spacing
    hi_snap = np.ceil(hi  / tick_spacing) * tick_spacing
    ax.set_ylim(lo_snap, hi_snap)
    ax.yaxis.set_major_locator(plt.MultipleLocator(tick_spacing))
    ticks = ax.get_yticks()
    ax.set_yticklabels([
        "" if (abs(t - lo_snap) < tick_spacing * 0.01
               or abs(t - hi_snap) < tick_spacing * 0.01
               or abs(round(t / label_spacing) * label_spacing - t) > tick_spacing * 0.01)
        else f"{t:g}"
        for t in ticks
    ])


# ---- Publication plots (unchanged from LAG_ADJ_MODEL_EST)

def plot_yield_fit(bei, spec_label, out_dir):
    """Observed vs model yields at 2y, 5y, 10y for both curves."""
    HL_MATS = [24, 60, 120]
    HL_LABS = {24: "2y", 60: "5y", 120: "10y"}

    nom_obs    = sgb_est[[f"y_{n}m" for n in NOM_RET_MATS]].values
    il_adj_obs = il_adj_est[[f"y_{n}m" for n in IL_RET_MATS]].values

    panels = [
        ("SGB",                   nom_obs,    bei["nom_Q"], NOM_RET_MATS),
        ("SGBi (carry-adjusted)", il_adj_obs, bei["il_Q"],  IL_RET_MATS),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(FIG_W * 1.4, FIG_H * 2.0), sharex=True)
    fig.patch.set_facecolor("white")

    for ax, (title, obs, mod, mats) in zip(axes, panels):
        for k, n in enumerate(HL_MATS):
            j = mats.index(n)
            ax.plot(dates_plot, obs[:, j] * 100,
                    color=C_NOM_3[k], lw=1.8, label=f"{HL_LABS[n]} obs")
            ax.plot(dates_plot, mod[:, j] * 100,
                    color=C_NOM_3[k], lw=1.0, ls="--", label=f"{HL_LABS[n]} model")
        ax.axhline(0, color=C_ZERO, lw=0.8)
        ax.set_ylabel("Yield (%)")
        ax.set_title(title)
        _finalise_yaxis(ax, 1)
        ax.legend(ncol=3, loc="lower left", frameon=True, fancybox=False,
                  edgecolor="#AAAAAA", facecolor="white")
        _style_ax(ax)

    _fmt_xaxis(axes[-1])
    plt.tight_layout(pad=0.8)
    _save(fig, out_dir, "yield_fit")
    plt.show()
    plt.close(fig)


def plot_bei_decomposition(bei, spec_label, out_dir, has_liq=False):
    """BEI decomposition panels."""
    if not has_liq:
        panels = [
            ("Model vs observed BEI", bei["BEI"],     C_NOM_3, bei["raw_BEI"], 0.5,  1.0),
            ("Expected inflation",     bei["EI"],      C_EI_3,  None,           0.25, 0.5),
            ("Total risk premium",     bei["IRP"],     C_IRP_3, None,           0.5,  1.0),
        ]
    else:
        panels = [
            ("Model vs observed BEI", bei["BEI"],     C_NOM_3, bei["raw_BEI"], 0.5,  1.0),
            ("Expected inflation",     bei["EI"],      C_EI_3,  None,           0.25, 0.5),
            ("Liquidity premium",      bei["LIQ"],     C_LIQ_3, None,           0.25, 0.5),
            ("Inflation risk premium", bei["IRP_adj"], C_IRP_3, None,           0.5,  1.0),
        ]

    fig, axes = plt.subplots(len(panels), 1,
                             figsize=(FIG_W * 1.4, FIG_H * len(panels) * 0.95),
                             sharex=True)
    fig.patch.set_facecolor("white")

    for ax, (title, series, colors, overlay, tick_sp, label_sp) in zip(axes, panels):
        for k, n in enumerate(BEI_MATS):
            ax.plot(dates_plot, series[n] * 100,
                    color=colors[k], lw=1.8, label=BEI_LABS[n])
        if overlay:
            for k, n in enumerate(BEI_MATS):
                ax.plot(dates_plot, overlay[n] * 100,
                        color=colors[k], lw=1.0, ls="--", label=f"{BEI_LABS[n]} raw")
        ax.axhline(0, color=C_ZERO, lw=0.8)
        ax.set_ylabel("% p.a.")
        ax.set_title(title)
        _finalise_yaxis(ax, tick_sp, label_sp)
        ax.legend(ncol=3 if not overlay else 6, loc="lower left", frameon=True,
                  fancybox=False, edgecolor="#AAAAAA", facecolor="white")
        _style_ax(ax)

    _fmt_xaxis(axes[-1])
    plt.tight_layout(pad=0.8)
    _save(fig, out_dir, "bei_decomposition")
    plt.show()
    plt.close(fig)


def plot_inflation_fit(est, X, spec_label, out_dir):
    """Model-implied vs realised inflation as 12-month rolling sums."""
    pi_mod      = est["pi_0"] + X @ est["pi_1"]
    pi_obs_roll = pd.Series(pi_arr, index=dates_plot).rolling(12).sum() * 100
    pi_mod_roll = pd.Series(pi_mod,  index=dates_plot).rolling(12).sum() * 100

    fig, ax = plt.subplots(figsize=(FIG_W * 1.4, FIG_H))
    fig.patch.set_facecolor("white")
    ax.plot(dates_plot, pi_obs_roll, color=C_EI_3[0], lw=1.8,
            label="Realised CPI (12m rolling)")
    ax.plot(dates_plot, pi_mod_roll, color=C_EI_3[0], lw=1.0, ls="--",
            label=r"Model: $\pi_0 + \pi_1^\prime X_t$")
    ax.axhline(0, color=C_ZERO, lw=0.8)
    ax.set_ylabel("% p.a.")
    _finalise_yaxis(ax, 1)
    ax.legend(loc="lower left", frameon=True, fancybox=False,
              edgecolor="#AAAAAA", facecolor="white")
    _style_ax(ax)
    _fmt_xaxis(ax)
    plt.tight_layout(pad=0.8)
    _save(fig, out_dir, "inflation_fit")
    plt.show()
    plt.close(fig)


def plot_term_premia(est, X, spec_label, out_dir):
    """Nominal and real term premia and the implied inflation risk premium."""
    K    = est["Phi"].shape[0]
    d0m  = est["d0m"]
    d1m  = est["d1m"]
    mu_P = (np.eye(K) - est["Phi"]) @ est["mu_X"]

    AB_nom_Q = _nom_ab(est["Phi_tilde"], est["mu_tilde"], est["Sigma"],
                       d0m, d1m, BEI_MATS, K)
    AB_il_Q  = _il_ab(est["pi_0"], est["pi_1"], est["Phi_tilde"], est["mu_tilde"],
                      est["Sigma"], d0m, d1m, BEI_MATS, K)
    AB_nom_P = _nom_ab(est["Phi"], mu_P, est["Sigma"], d0m, d1m, BEI_MATS, K)
    AB_il_P  = _il_ab(est["pi_0"], est["pi_1"], est["Phi"], mu_P,
                      est["Sigma"], d0m, d1m, BEI_MATS, K)

    nom_Q = _model_yields(AB_nom_Q, X, BEI_MATS)
    il_Q  = _model_yields(AB_il_Q,  X, BEI_MATS)
    nom_P = _model_yields(AB_nom_P, X, BEI_MATS)
    il_P  = _model_yields(AB_il_P,  X, BEI_MATS)

    TP_nom = nom_Q - nom_P
    TP_il  = il_Q  - il_P
    IRP_tp = TP_nom - TP_il

    fig, axes = plt.subplots(3, 1, figsize=(FIG_W * 1.4, FIG_H * 3.0), sharex=True)
    fig.patch.set_facecolor("white")

    for k, n in enumerate(BEI_MATS):
        axes[0].plot(dates_plot, TP_nom[:, k] * 100, color=C_NOM_3[k], lw=1.8,
                     label=BEI_LABS[n])
        axes[1].plot(dates_plot, TP_il[:, k]  * 100, color=C_EI_3[k],  lw=1.8,
                     label=BEI_LABS[n])
        axes[2].plot(dates_plot, IRP_tp[:, k] * 100, color=C_IRP_3[k], lw=1.8,
                     label=BEI_LABS[n])

    for ax, title in zip(axes, ["Nominal term premium", "Real term premium",
                                 "Inflation risk premium"]):
        ax.axhline(0, color=C_ZERO, lw=0.8)
        ax.set_ylabel("% p.a.")
        ax.set_title(title)
        _finalise_yaxis(ax, 0.5, 1.0)
        ax.legend(ncol=3, loc="lower left", frameon=True, fancybox=False,
                  edgecolor="#AAAAAA", facecolor="white")
        _style_ax(ax)

    _fmt_xaxis(axes[-1])
    plt.tight_layout(pad=0.8)
    _save(fig, out_dir, "term_premia")
    plt.show()
    plt.close(fig)


# endregion

# region SPEC 1 — 5-FACTOR ADJ YID
# Factors: nom_PC1, nom_PC2, nom_PC3, il_PC1, il_PC2 (carry-adjusted)
# Nominal PCs from raw SGB yields.
# il_PC1 and il_PC2 from PCA on residuals of carry-adjusted SGBi yields
# after projecting out nominal PCs.
# IL B rows identified from adj SGBi yield cross-section.

SPEC1_LABEL = "5-factor adj YID"
SPEC1_DIR   = "Model output/5_factor_adj_yid"
SPEC1_NAMES = ["nom_PC1", "nom_PC2", "nom_PC3", "il_PC1", "il_PC2"]
os.makedirs(SPEC1_DIR, exist_ok=True)

raw_f1 = pd.read_excel(f"{DIR_FACT}/Factors.xlsx", sheet_name="5_factor_adj")
raw_f1["ym"] = raw_f1["date"].dt.to_period("M")
factors1 = (raw_f1[["ym"] + SPEC1_NAMES]
            .set_index("ym").sort_index()
            .loc[est_ym].dropna())

assert len(factors1) == T, (
    f"Spec 1: {len(factors1)} factor rows, expected {T}. Re-run FACTORS.py.")

X1 = factors1.values
K1 = X1.shape[1]

print(f"\n{'='*60}")
print(f"  {SPEC1_LABEL}   K = {K1}")
print(f"{'='*60}")

est1 = run_estimation_yieldid(X1, R_all_adj, R_il_adj_raw, r_arr, pi_arr, K1)
bei1 = decompose_bei_lag(est1, X1, liq_idx=None)
print_diagnostics(est1, bei1, SPEC1_NAMES, SPEC1_LABEL)
plot_yield_fit(bei1, SPEC1_LABEL, SPEC1_DIR)
plot_bei_decomposition(bei1, SPEC1_LABEL, SPEC1_DIR, has_liq=False)
plot_inflation_fit(est1, X1, SPEC1_LABEL, SPEC1_DIR)
plot_term_premia(est1, X1, SPEC1_LABEL, SPEC1_DIR)

# endregion

# region SPEC 2 — 6-FACTOR ADJ YID
# Factors: nom_PC1, nom_PC2, nom_PC3, liq, il_PC1, il_PC2 (carry-adjusted)
# IL PCs from residuals after projecting out nominal PCs and liq.
# IL B rows identified from adj SGBi yield cross-section.

SPEC2_LABEL = "6-factor adj YID"
SPEC2_DIR   = "Model output/6_factor_adj_yid"
SPEC2_NAMES = ["nom_PC1", "nom_PC2", "nom_PC3", "liq", "il_PC1", "il_PC2"]
os.makedirs(SPEC2_DIR, exist_ok=True)

raw_f2 = pd.read_excel(f"{DIR_FACT}/Factors.xlsx", sheet_name="6_factor_adj")
raw_f2["ym"] = raw_f2["date"].dt.to_period("M")
factors2 = (raw_f2[["ym"] + SPEC2_NAMES]
            .set_index("ym").sort_index()
            .loc[est_ym].dropna())

assert len(factors2) == T, (
    f"Spec 2: {len(factors2)} factor rows, expected {T}. Re-run FACTORS.py.")

X2 = factors2.values
K2 = X2.shape[1]

print(f"\n{'='*60}")
print(f"  {SPEC2_LABEL}   K = {K2}")
print(f"{'='*60}")

est2 = run_estimation_yieldid(X2, R_all_adj, R_il_adj_raw, r_arr, pi_arr, K2)
bei2 = decompose_bei_lag(est2, X2, liq_idx=SPEC2_NAMES.index("liq"))
print_diagnostics(est2, bei2, SPEC2_NAMES, SPEC2_LABEL)
plot_yield_fit(bei2, SPEC2_LABEL, SPEC2_DIR)
plot_bei_decomposition(bei2, SPEC2_LABEL, SPEC2_DIR, has_liq=True)
plot_inflation_fit(est2, X2, SPEC2_LABEL, SPEC2_DIR)
plot_term_premia(est2, X2, SPEC2_LABEL, SPEC2_DIR)

# endregion

# region SPEC 3 — 7-FACTOR ADJ YID
# Factors: nom_PC1, nom_PC2, nom_PC3, liq, inf, il_PC1, il_PC2 (carry-adjusted)
# Adds an observable inflation factor residualised on nominal PCs and standardised.
# il_PC2 column of B identified from adj SGBi yield cross-section.
#
# Unlike the full-IL-yield-ID approach, the il_PC2-only fix produces a stationary
# Phi_tilde for this specification. The inf factor's cross-loadings are retained
# in the return regression and do not create instability when il_PC1's B column
# also remains return-estimated.

SPEC3_LABEL = "7-factor adj YID"
SPEC3_DIR   = "Model output/7_factor_adj_yid"
SPEC3_NAMES = ["nom_PC1", "nom_PC2", "nom_PC3", "liq", "inf", "il_PC1", "il_PC2"]
os.makedirs(SPEC3_DIR, exist_ok=True)

raw_f3 = pd.read_excel(f"{DIR_FACT}/Factors.xlsx", sheet_name="7_factor_adj")
raw_f3["ym"] = raw_f3["date"].dt.to_period("M")
factors3 = (raw_f3[["ym"] + SPEC3_NAMES]
            .set_index("ym").sort_index()
            .loc[est_ym].dropna())

assert len(factors3) == T, (
    f"Spec 3: {len(factors3)} factor rows, expected {T}. Re-run FACTORS.py.")

X3 = factors3.values
K3 = X3.shape[1]

print(f"\n{'='*60}")
print(f"  {SPEC3_LABEL}   K = {K3}")
print(f"{'='*60}")

est3 = run_estimation_yieldid(X3, R_all_adj, R_il_adj_raw, r_arr, pi_arr, K3)
bei3 = decompose_bei_lag(est3, X3, liq_idx=SPEC3_NAMES.index("liq"))
print_diagnostics(est3, bei3, SPEC3_NAMES, SPEC3_LABEL)
plot_yield_fit(bei3, SPEC3_LABEL, SPEC3_DIR)
plot_bei_decomposition(bei3, SPEC3_LABEL, SPEC3_DIR, has_liq=True)
plot_inflation_fit(est3, X3, SPEC3_LABEL, SPEC3_DIR)
plot_term_premia(est3, X3, SPEC3_LABEL, SPEC3_DIR)

# endregion