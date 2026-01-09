# -*- coding: utf-8 -*-

def _normcdf(x: float) -> float:
    """CDF da Normal(0,1) usando erf; evita dependências externas."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

"""
DGU backtest — v2.8.18 (patched)
---------------------------------
Ajustes principais desta versão:
1) Sharpe do **MV (in sample)** calculado sobre a faixa OOS com **filtro de NaN**
   e MLE consistente (cov_ddof1 * (T-1)/T) para evitar NaN/degenerescência.
2) Retorno mensal de **MV (in sample)** passa a ser calculado **sempre** na base
   que gerou os momentos (indústrias em excesso); define `rex_t_ind` antes de usar.
3) **EW–MIN (KZ 1/N)**: combinação fechada entre 1/N e GMV usando Sigma MLE.
4) **MV "paper"**: w = Σ^{-1}μ / |1'Σ^{-1}μ| com ridge e simetrização.
5) **BS (Jorion)** específico para Industry, normalizado risky-only (∑|w|=1).
6) **MP (MacKinlay–Pastor)** e **DM (Pástor–Stambaugh)** robustos, normalização
   risky-only; DM usa apenas MKT como fator do modelo.
7) Turnover "paper-like" (drift + 0.5*L1) e razões relativas a EW reportadas.

Observação: custos não são aplicados (GROSS), alinhado às Tabelas principais.
"""

from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
from math import erf, sqrt, isfinite  # added for normcdf / p-values
import numpy as np
import pandas as pd
from scipy.special import betainc  # Kan-Zhou exact (MATLAB betainc)

# ========================= Helpers básicos ========================= #
def _risky_only(v: np.ndarray) -> np.ndarray:
    s = float(np.sum(v))
    return v if s == 0 or not np.isfinite(s) else (v / abs(s))



def _alpha_to_w_ro(alpha: np.ndarray, denom_floor: float = 1e-16) -> np.ndarray:
    """Converte alfa (fração da riqueza em ativos arriscados) para risky-only: alpha / |1'alpha|."""
    a = np.asarray(alpha, float).reshape(-1)
    s = float(np.sum(a))
    if (not np.isfinite(s)) or abs(s) < denom_floor:
        # se soma ~0, devolve vetor original (evita explosão); caller decide
        return a.copy()
    return (a / abs(s))

def _buyhold_alpha_matlab(alpha_prev: np.ndarray, rexcess_t: np.ndarray, rf_t: float, denom_floor: float = 1e-16) -> np.ndarray:
    """Replica dgu_static_buyhold.m (MATLAB). alpha_prev e rexcess_t são vetores (N,).
    Usa: trp = (1-sum(alpha))*(1+rf) + alpha'*(1+(rexcess+rf))
    """
    a = np.asarray(alpha_prev, float).reshape(-1)
    rx = np.asarray(rexcess_t, float).reshape(-1)
    rf = float(rf_t)
    gross = 1.0 + (rx + rf)  # 1 + retorno total de cada ativo arriscado
    trp = (1.0 - float(np.sum(a))) * (1.0 + rf) + float(np.dot(a, gross))
    if (not np.isfinite(trp)) or abs(trp) < denom_floor:
        # fallback: não driftar
        return a.copy()
    return (a * gross) / trp

def _turnover_series_matlab_ro(alpha_df: pd.DataFrame, rexcess_df: pd.DataFrame, rf_s: pd.Series) -> pd.Series:
    """Turnover risky-only exatamente como em dgu_static_riskyonly.m:
      - target risky-only: alpha_t / |sum(alpha_t)|
      - buyhold alpha: dgu_static_buyhold(alpha_{t-1}, rexcess_t, rf_t)
      - buyhold risky-only: alphaBH_t / sum(alphaBH_t)   (SEM abs)
      - turnover_t = sum | targetRO_t - buyholdRO_t |
    """
    if alpha_df is None or rexcess_df is None or alpha_df.empty or rexcess_df.empty:
        return pd.Series(dtype=float)
    idx = alpha_df.index
    rexcess_df = rexcess_df.reindex(index=idx, columns=alpha_df.columns).fillna(0.0)
    rf_s = rf_s.reindex(idx).fillna(0.0)

    tau = np.full(len(idx), np.nan, float)
    alpha_vals = alpha_df.to_numpy(float)
    rex_vals = rexcess_df.to_numpy(float)
    rf_vals = rf_s.to_numpy(float)

    for i in range(len(idx)):
        alpha_t = alpha_vals[i, :]
        # target risky-only (abs no denominador)
        w_t = _alpha_to_w_ro(alpha_t)

        # buy-and-hold alpha (MATLAB)
        if i == 0:
            alpha_bh = alpha_t.copy()
        else:
            alpha_prev = alpha_vals[i-1, :]
            alpha_bh = _buyhold_alpha_matlab(alpha_prev, rex_vals[i, :], rf_vals[i])

        s_bh = float(np.sum(alpha_bh))
        if (not np.isfinite(s_bh)) or abs(s_bh) < 1e-16:
            w_bh = alpha_bh.copy()
        else:
            w_bh = alpha_bh / s_bh  # SEM abs (MATLAB)

        tau[i] = float(np.sum(np.abs(w_t - w_bh)))

    return pd.Series(tau, index=idx)
def _turnover_step_paper_risky_only(w_target: np.ndarray,
                                    w_prev: np.ndarray,
                                    gross_ret_prev: np.ndarray) -> float:
    """
    Um passo t-1 -> t do turnover 'paper' (sem 0.5), só para MV-min:
      - normaliza w_target e w_prev em 'risky-only' (divide por |∑w|)
      - aplica drift buy-and-hold em w_prev com (1 + R_{gross,t-1})
      - renormaliza 'risky-only'
      - retorna soma L1 das diferenças
    """
    w_t = _risky_only(np.asarray(w_target, float).reshape(-1))
    w_bh = np.asarray(w_prev, float).reshape(-1) * np.asarray(gross_ret_prev, float).reshape(-1)
    w_bh = _risky_only(w_bh)
    return float(np.sum(np.abs(w_t - w_bh)))

def _pinv(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, float)
    try:
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A, rcond=1e-12)


def _pinv_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _pinv(A) @ b


def _sanitize_index(obj):
    if obj is None:
        return None
    try:
        o = obj.copy()
        if not o.index.is_monotonic_increasing:
            o = o.sort_index()
        if not o.index.is_unique:
            o = o[~o.index.duplicated(keep="last")]
        return o
    except Exception:
        return obj


# ---- FF3 detection (factor-only dataset guard) ----
FF3_ALIASES = {
    "MktRF": ["MktRF","Mkt-RF","MKT_RF","MKT-RF","MktMinusRF"],
    "SMB":   ["SMB"],
    "HML":   ["HML"],
}

FACTOR_COLS = {"RF","SMB","HML","UMD","WML","IML","WORLD"}


def _detect_ff3_columns(df: pd.DataFrame) -> Tuple[bool, Dict[str, str]]:
    found = {}
    for key, opts in FF3_ALIASES.items():
        for c in opts:
            if c in df.columns:
                found[key] = c
                break
    only_ff3 = (len(found) == 3) and (set(df.columns) == set(found.values()))
    return only_ff3, found


def _risky_only_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if str(c) not in FACTOR_COLS]
    only_ff3, ff3_map = _detect_ff3_columns(df)
    if only_ff3:
        return [ff3_map['MktRF'], ff3_map['SMB'], ff3_map['HML']]
    if len(cols) == 0:
        prefer_order = ["MktRF","MKT_RF","MKT","Mkt-RF","MKT-RF","MktMinusRF","SMB","HML"]
        fb = [c for c in prefer_order if c in df.columns]
        seen, fb_unique = set(), []
        for c in fb:
            if c not in seen:
                fb_unique.append(c); seen.add(c)
        cols = fb_unique
    return cols


# ========================= Portfólios (pesos) ========================= #

# MV "paper": w = Σ^{-1} μ / |1' Σ^{-1} μ|

def _w_mv_paper(mu: np.ndarray,
                Sigma: np.ndarray,
                ridge: float = 0.0,
                denom_floor: float = 1e-20) -> np.ndarray:
    """
    MV no estilo do DGU/Matlab para gerar pesos "risky-only":

        w = inv(Sigma) @ mu / |1' inv(Sigma) mu|

    Diferenças vs. versão "robusta":
      - **sem fallback para 1/N** quando o denominador fica pequeno
      - usa apenas um *floor* mínimo para evitar divisão por zero

    Observação: essa escolha afeta fortemente o turnover (Tabela 5), mas pouco o Sharpe.
    """
    mu = np.asarray(mu, float).reshape(-1, 1)
    N = mu.shape[0]
    S = np.asarray(Sigma, float).reshape(N, N)
    S = 0.5 * (S + S.T)
    if ridge and ridge > 0:
        S = S + ridge * np.eye(N)

    invS = _pinv(S)
    one = np.ones((N, 1))
    w_unnorm = invS @ mu
    denom = float(np.abs(one.T @ w_unnorm))
    if (not np.isfinite(denom)) or denom <= denom_floor:
        denom = denom_floor
    return (w_unnorm / denom).ravel()

def _w_mv(mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    mu = np.asarray(mu, float).reshape(-1)
    N = mu.size
    w_raw = _pinv_solve(Sigma, mu)
    s = float(np.sum(w_raw))
    return np.ones(N)/N if s == 0 else (w_raw / s)

def _w_bs_c(mu: np.ndarray, Sigma: np.ndarray, T: int, iters: int = 600, ridge: float = 1e-12) -> np.ndarray:
    """
    Bayes–Stein (Jorion) com restrição w>=0, sum(w)=1 (simplex).
    Implementação alinhada ao uso no DGU:
      - alvo = mu_min = (1' Σ^{-1} μ)/(1' Σ^{-1} 1)
      - φ̂ = (N+2) / ((N+2) + M * (μ - mu_min 1)' Σ_hat^{-1} (μ - mu_min 1))
      - Σ_hat escalada ao estilo Jorion: ((M-1)/(M-N-2)) * S (quando M > N+2)
      - Otimização via PGD no simplex (grad = μ_bs − Σ_bs w)
    """
    import numpy as _np

    # ---- fallbacks locais (se helpers globais não existirem) ----
    def _pinv_local(A: _np.ndarray) -> _np.ndarray:
        try:
            # usa helper do arquivo, se existir
            return _pinv(A)  # type: ignore[name-defined]
        except Exception:
            try:
                return _np.linalg.inv(A)
            except _np.linalg.LinAlgError:
                return _np.linalg.pinv(A, rcond=1e-12)

    def _proj_simplex(v: _np.ndarray) -> _np.ndarray:
        try:
            # usa helper do arquivo, se existir
            return _safe_simplex_project(v)  # type: ignore[name-defined]
        except Exception:
            v = _np.asarray(v, dtype=float).ravel()
            n = v.size
            if n == 0:
                return v
            u = _np.sort(v)[::-1]
            cssv = _np.cumsum(u)
            rho_idx = _np.nonzero(u * _np.arange(1, n+1) > (cssv - 1))[0]
            if rho_idx.size == 0:
                return _np.ones(n)/n
            rho = int(rho_idx[-1])
            theta = (cssv[rho] - 1.0) / float(rho + 1)
            w = _np.maximum(v - theta, 0.0)
            s = float(w.sum())
            return w / s if s != 0 else _np.ones(n)/n

    mu = _np.asarray(mu, dtype=float).reshape(-1)
    N  = mu.size
    if N == 0:
        return _np.array([])

    S = _np.asarray(Sigma, dtype=float)
    S = 0.5 * (S + S.T)
    if ridge and ridge > 0.0:
        S = S + float(ridge) * _np.eye(N)

    M = int(T)
    # Σ_hat (Jorion): ((M-1)/(M-N-2)) * S_unbiased  — aqui usamos S já consistente com o restante do pipeline
    if M > N + 2:
        SigmaHat = ((M - 1.0) / (M - N - 2.0)) * S
    else:
        SigmaHat = S.copy()
    SigmaHat = 0.5 * (SigmaHat + SigmaHat.T)
    if ridge and ridge > 0.0:
        SigmaHat = SigmaHat + float(ridge) * _np.eye(N)

    invS = _pinv_local(SigmaHat)
    one  = _np.ones((N, 1))
    Ahat = float(one.T @ invS @ one)  # 1' Σ^{-1} 1
    if (not _np.isfinite(Ahat)) or Ahat <= 0:
        return _np.ones(N)/N

    mu_col = mu.reshape(-1,1)
    mu_min = float((one.T @ invS @ mu_col) / Ahat)
    diff   = (mu - mu_min)
    diff_c = diff.reshape(-1,1)

    # φ̂ (shrink da média)
    qform = float(diff_c.T @ (M * invS) @ diff_c)
    phi   = (N + 2.0) / ((N + 2.0) + qform) if ((N + 2.0) + qform) > 0 else 0.0

    # λ (shrink da covariância)
    q0  = float(diff_c.T @ invS @ diff_c)
    lam = (N + 2.0) / q0 if q0 > 0 else 0.0

    # μ_bs e Σ_bs
    mu_bs = (1.0 - phi) * mu + phi * mu_min * _np.ones(N)

    Sigma_bs = SigmaHat * (1.0 + 1.0 / (M + lam))
    Sigma_bs += (lam / (M * (M + 1.0 + lam))) * (one @ one.T) / Ahat
    Sigma_bs = 0.5 * (Sigma_bs + Sigma_bs.T)
    if ridge and ridge > 0.0:
        Sigma_bs = Sigma_bs + float(ridge) * _np.eye(N)

    # PGD no simplex (maximize w' mu_bs - 0.5 w' Σ_bs w)
    w   = _np.ones(N)/N
    L   = (float(_np.linalg.norm(Sigma_bs, ord=2)) + 1e-12)
    eta = 0.1 / L
    for _ in range(int(iters)):
        grad = mu_bs - (Sigma_bs @ w)
        w_new = _proj_simplex(w + eta * grad)
        if float(_np.linalg.norm(w_new - w, 1)) < 1e-10:
            w = w_new; break
        w = w_new
    # normalização defensiva
    s = float(w.sum())
    return (w / s) if s != 0 else (_np.ones(N)/N)


def _w_min(mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    N = len(mu); one = np.ones(N)
    w_raw = _pinv_solve(Sigma, one)
    s = float(np.sum(w_raw))
    return np.ones(N)/N if s == 0 else (w_raw / s)


def _safe_simplex_project(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float).ravel(); n = v.size
    if n == 0:
        return v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho_idx = np.nonzero(u * np.arange(1, n+1) > (cssv - 1))[0]
    if rho_idx.size == 0:
        return np.ones(n)/n
    rho = int(rho_idx[-1])
    theta = (cssv[rho] - 1.0) / float(rho + 1)
    w = np.maximum(v - theta, 0.0)
    s = w.sum()
    return w / s if s != 0 else np.ones(n)/n


def _safe_simplex_with_floor(v: np.ndarray, lbound: float) -> np.ndarray:
    v = np.asarray(v, float).ravel(); n = v.size
    mass = 1.0 - n*float(lbound)
    if mass <= 0:
        return np.ones(n)/n
    z = v - lbound
    u = np.sort(z)[::-1]
    cssv = np.cumsum(u)
    rho_idx = np.nonzero(u * np.arange(1, n+1) > (cssv - mass))[0]
    if rho_idx.size == 0:
        w = np.zeros(n)
    else:
        rho = int(rho_idx[-1])
        theta = (cssv[rho] - mass) / float(rho + 1)
        w = np.maximum(z - theta, 0.0)
    s = w.sum(); w = w * (mass/s) if s != 0 else np.zeros(n)
    w = w + lbound; w = np.maximum(w, lbound); w = w / w.sum()
    return w


def _w_min_c(Sigma: np.ndarray, iters: int = 600) -> np.ndarray:
    N = Sigma.shape[0]
    w = np.ones(N)/N
    L = (np.linalg.norm(Sigma, ord=2) + 1e-12)
    eta = 0.1 / L
    for _ in range(iters):
        grad = 2.0 * (Sigma @ w)
        w_new = _safe_simplex_project(w - eta * grad)
        if np.linalg.norm(w_new - w, 1) < 1e-10:
            w = w_new; break
        w = w_new
    return w


def _w_mv_c(mu: np.ndarray, Sigma: np.ndarray, iters: int = 600) -> np.ndarray:
    N = len(mu)
    w = np.ones(N)/N
    L = (np.linalg.norm(Sigma, ord=2) + 1e-12)
    eta = 0.1 / L
    for _ in range(iters):
        grad = mu - (Sigma @ w)
        w_new = _safe_simplex_project(w + eta * grad)
        if np.linalg.norm(w_new - w, 1) < 1e-10:
            w = w_new; break
        w = w_new
    return w


def _w_gmin_c(Sigma: np.ndarray, lbound: float = 0.0, iters: int = 600) -> np.ndarray:
    N = Sigma.shape[0]
    w = np.ones(N)/N
    L = (np.linalg.norm(Sigma, ord=2) + 1e-12)
    eta = 0.1 / L
    for _ in range(iters):
        grad = 2.0 * (Sigma @ w)
        w_new = _safe_simplex_with_floor(w - eta * grad, lbound=lbound)
        if np.linalg.norm(w_new - w, 1) < 1e-10:
            w = w_new; break
        w = w_new
    return w


# MV–MIN (Kan–Zhou)

def _w_mv_min_kz(mu: np.ndarray, S: np.ndarray, M: int, normalize: str = "risky_only", ridge: float = 1e-12) -> np.ndarray:
    """Kan–Zhou MV-MIN (equação (67) no MATLAB dgu_static_KanZhou.m).
    Entradas:
      - mu: muhat (médias em excesso) (N,)
      - S:  SigmaMLE (MLE) (N,N)
      - M:  T (tamanho da janela, nPoints no MATLAB)
    Saída:
      - se normalize == "alpha": alpha (N,)
      - caso contrário: risky-only w = alpha/|sum(alpha)|
    """
    mu = np.asarray(mu, float).reshape(-1)
    S  = 0.5*(np.asarray(S, float) + np.asarray(S, float).T)
    N  = mu.size
    if N == 0:
        return np.zeros(0, float)

    S_reg = S + (float(ridge) * np.eye(N) if ridge and ridge > 0 else 0.0)
    invSigma = _pinv(S_reg)

    one = np.ones((N,1))
    muhat = mu.reshape(-1,1)

    # mug = (muhat' invSigma 1) / (1' invSigma 1)
    denom_A = float((one.T @ invSigma @ one))
    mug = float((muhat.T @ invSigma @ one) / denom_A) if (np.isfinite(denom_A) and abs(denom_A) > 1e-16) else 0.0

    diff = (muhat - mug*one)
    phiHat2 = float((diff.T @ invSigma @ diff))
    if not np.isfinite(phiHat2) or phiHat2 < 0:
        phiHat2 = 0.0

    x = float(phiHat2/(1.0 + phiHat2)) if phiHat2 >= 0 else 0.0
    a = (N - 1.0)/2.0
    b = (M - N + 1.0)/2.0

    # MATLAB betainc é regularizada; scipy.special.betainc também é regularizada.
    B = float(betainc(a, b, x)) if (a > 0 and b > 0) else 1.0
    if (not np.isfinite(B)) or B <= 1e-300:
        B = 1e-300

    term = (2.0*(phiHat2**((N-1.0)/2.0))*((1.0+phiHat2)**(-(M-2.0)/2.0))) / (M*B) if M > 0 else 0.0
    phiHat2a = (((M-N-1.0)*phiHat2-(N-1.0))/M) + term if M != 0 else 0.0

    c3 = ((M-N-1.0)*(M-N-4.0)) / (M*(M-2.0)) if (M != 0 and abs(M-2.0) > 1e-16) else 0.0
    denom = float(phiHat2a + N/float(M)) if M != 0 else float('nan')
    if (not np.isfinite(denom)) or abs(denom) <= 1e-16:
        denom = 1e-16

    # alpha = c3/gam * ((phiHat2a/denom)*invSigma*muhat + ((N/M)/denom)*mug*invSigma*ones)
    gam = 1.0
    part1 = (phiHat2a/denom) * (invSigma @ muhat)
    part2 = ((N/float(M))/denom) * mug * (invSigma @ one) if M != 0 else 0.0
    alpha = (c3/gam) * (part1 + part2)
    alpha = np.asarray(alpha, float).reshape(-1)

    if normalize == "alpha":
        return alpha

    return _alpha_to_w_ro(alpha)


# EW–MIN (KZ 1/N)

def _w_ew_min_kz1n_from_S_mle(S_mle: np.ndarray, T: int, ridge: float = 1e-12) -> np.ndarray:
    S_mle = np.asarray(S_mle, float)
    N = S_mle.shape[0]
    Sigma = 0.5*(S_mle + S_mle.T)
    Sigma = Sigma * (T / (T - N - 2)) if (T > N + 2) else Sigma
    if ridge and ridge > 0:
        Sigma = Sigma + ridge * np.eye(N)
    Sigma = 0.5*(Sigma + Sigma.T)
    one  = np.ones((N,1))
    try:
        invS = np.linalg.pinv(Sigma)
    except Exception:
        return (one/N).ravel()
    esige = float(one.T @ Sigma @ one)
    einv  = float(one.T @ invS  @ one)
    k = (T**2 * (T - 2)) / ((T - N - 1) * (T - N - 2) * (T - N - 4)) if (T > N + 4) else 1.0
    num = (T - N - 2) * esige * einv - (N**2) * T
    den = (N**2) * (T - N - 2) * k * einv - 2 * T * (N**2) * einv + (T - N - 2) * (einv**2) * esige
    if not np.isfinite(den) or abs(den) < 1e-16:
        return (one/N).ravel()
    d = num / den
    c = 1.0 - d * einv
    w = c * (one / N) + d * (invS @ one)
    s = float(w.sum())
    if not np.isfinite(s) or abs(s) < 1e-16:
        return (one/N).ravel()
    return (w / s).ravel()


# BS (Jorion) — Industry

def _alpha_bs_industry_from_window(X_window: np.ndarray, gamma: float = 1.0, ridge: float = 1e-12):
    """Momento Bayes–Stein (Jorion) e alfa BS (MATLAB dgu_static.m).
    Retorna (alpha, phi, qform, Ahat, condSigmaHat).
    """
    X = np.asarray(X_window, float)
    if X.ndim != 2:
        raise ValueError("X_window deve ser 2D (T x N).")
    Mwin, N = X.shape
    if Mwin <= 1 or N < 1:
        alpha = np.ones(N)/N
        return alpha, 0.0, 0.0, 0.0, np.inf

    Y = X.mean(axis=0)
    S_unb = np.cov(X, rowvar=False, ddof=1)

    # SigmaHat = (M-1)/(M-N-2) * cov (como no MATLAB quando M > N+2)
    SigmaHat = ((Mwin - 1.0) / (Mwin - N - 2.0)) * S_unb if (Mwin > N + 2) else S_unb
    SigmaHat = 0.5*(SigmaHat + SigmaHat.T)
    if ridge and ridge > 0:
        SigmaHat = SigmaHat + ridge*np.eye(N)

    invSigmaHat = _pinv(SigmaHat)
    one = np.ones((N,1))
    Ahat = float(one.T @ invSigmaHat @ one)
    Y0 = float((one.T @ invSigmaHat @ Y.reshape(-1,1)) / Ahat)

    diff = (Y - Y0)
    qform = float((diff.reshape(-1,1)).T @ (Mwin*invSigmaHat) @ (diff.reshape(-1,1)))

    wshr  = (N + 2.0) / ((N + 2.0) + qform) if ((N + 2.0) + qform) > 0 else 0.0
    q0 = float((diff.reshape(-1,1)).T @ invSigmaHat @ (diff.reshape(-1,1)))
    lam = (N + 2.0) / q0 if q0 > 0 else 0.0

    mu_bs = (1.0 - wshr) * Y + wshr * Y0 * np.ones(N)

    Sigma_bs = SigmaHat * (1.0 + 1.0/(Mwin + lam))
    Sigma_bs += (lam / (Mwin * (Mwin + 1.0 + lam))) * (np.ones((N,1)) @ np.ones((1,N))) / Ahat
    Sigma_bs = 0.5*(Sigma_bs + Sigma_bs.T)
    if ridge and ridge > 0:
        Sigma_bs = Sigma_bs + ridge*np.eye(N)

    invSigmaBS = _pinv(Sigma_bs)
    alpha = (invSigmaBS @ mu_bs.reshape(-1,1)).ravel() / float(gamma)

    # cond do SigmaHat (diagnóstico igual ao MATLAB)
    try:
        ev = np.linalg.eigvalsh(SigmaHat)
        condS = float(np.max(ev) / np.min(ev)) if np.min(ev) > 0 else np.inf
    except Exception:
        condS = np.inf

    return alpha.astype(float), float(wshr), float(qform), float(Ahat), float(condS)


def _w_bs_industry_from_window(X_window: np.ndarray, gamma: float = 1.0, ridge: float = 1e-12):
    """Convenience: devolve risky-only w = alpha/|sum(alpha)| + diagnósticos."""
    alpha, phi, qform, Ahat, condS = _alpha_bs_industry_from_window(X_window, gamma=gamma, ridge=ridge)
    w = _alpha_to_w_ro(alpha)
    return w, phi, qform, Ahat, condS


# MP (MacKinlay–Pastor)

def _w_mp_mackinlay_pastor_new(mu, S, gamma: float = 1.0, normalize: Optional[str] = "risky_only", **kwargs):
    """Pastor–MacKinlay (aprox. Kan-Zhou), igual a dgu_static_PastorMacKinlay.m.
    Se normalize == "alpha": retorna alpha (sem normalização).
    Se normalize == "risky_only" (default): retorna alpha/|sum(alpha)|.
    """
    mu = np.asarray(mu, float).reshape(-1)
    S  = 0.5*(np.asarray(S, float) + np.asarray(S, float).T)
    N  = mu.size

    U  = S + np.outer(mu, mu)
    try:
        evals, evecs = np.linalg.eigh(U)
    except np.linalg.LinAlgError:
        evals, evecs = np.linalg.eigh(U + 1e-8*np.eye(N))

    l1  = float(evals[-1])
    q1  = np.asarray(evecs[:, -1], float)

    mu_tilde = float(q1 @ mu) * q1
    denom = l1 - float(mu_tilde @ mu_tilde)
    if (not np.isfinite(denom)) or abs(denom) <= 1e-16:
        alpha = np.ones(N)/N
    else:
        alpha = (mu_tilde / denom) / float(gamma)

    if normalize == "alpha":
        return alpha.astype(float).ravel()

    # risky-only (paper)
    return _alpha_to_w_ro(alpha).astype(float).ravel()


# DM (Pástor–Stambaugh)

def _w_dm_pastor(X_excess: np.ndarray, F_excess: Optional[np.ndarray], sigma_alpha_annual: float = 0.01) -> np.ndarray:
    X = np.asarray(X_excess, float)
    if X.ndim != 2 or X.shape[0] < 3 or X.shape[1] < 1:
        return np.zeros(0, float)
    T, N = X.shape
    if F_excess is None or (hasattr(F_excess, "size") and F_excess.size == 0):
        mu = X.mean(axis=0); S = np.cov(X, rowvar=False, ddof=1)
        try:
            invS = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            ev, U = np.linalg.eigh(S); ev = np.clip(ev, 1e-10, None)
            invS = U @ np.diag(1.0/ev) @ U.T
        w_raw = invS @ mu; s = float(np.sum(w_raw))
        return (w_raw/abs(s)).ravel() if s != 0 and np.isfinite(s) else np.ones(N)/N
    F = np.asarray(F_excess, float); F = F.reshape(-1, 1) if F.ndim == 1 else F
    if F.shape[0] != T:
        raise ValueError("F_excess deve ter shape (T, K) igual ao de X_excess.")
    F = F[:, :1]  # usa apenas MKT
    K = 1
    f = F[:, 0]
    corr_vals = []
    for j in range(N):
        xj = X[:, j]
        sx = np.std(xj, ddof=1); sf = np.std(f, ddof=1)
        c = -np.inf if (sx <= 0 or sf <= 0) else float(np.corrcoef(xj, f)[0,1])
        corr_vals.append(c)
    j_fac = int(np.nanargmax(corr_vals))
    mask_other = [i for i in range(N) if i != j_fac]
    ret = X[:, mask_other]; m = ret.shape[1]
    if m < 1:
        e = np.zeros(N); e[j_fac] = 1.0; return e
    mu1hat = ret.mean(axis=0)
    mu2hat = F.mean(axis=0)
    omega22hat = np.cov(F, rowvar=False, ddof=1)
    if np.ndim(omega22hat) == 0:
        omega22hat = np.array([[float(omega22hat)]], float)
    Shat2 = float(mu2hat @ np.linalg.inv(omega22hat) @ mu2hat)
    b = (T + 1.0) / (T - K - 2.0)
    h = T / (T - m - K - 1.0)
    delta_bar = (T*(T-2)+K)/T/(T-K-2) - (K+3)*(Shat2)/T/(T-K-2)/(1+Shat2)
    delta_hat = (T-2)*(T+1)/T/(T-K-2)
    X_int = np.column_stack([np.ones(T), F])
    beta_h, *_ = np.linalg.lstsq(X_int, ret, rcond=None)
    R  = ret - X_int @ beta_h
    beta_h = beta_h.T
    beta_hat = beta_h[:, 1:2]
    Sigma_hat = np.cov(R, rowvar=False, ddof=1)
    beta_bar, *_ = np.linalg.lstsq(F, ret, rcond=None)
    R1 = ret - F @ beta_bar
    beta_bar = beta_bar.T
    Sigma_bar = np.cov(R1, rowvar=False, ddof=1)
    s2 = float(np.max(np.diag(Sigma_hat))) if m>0 else 1.0
    s_alpha = float(sigma_alpha_annual) / 12.0
    omega = 1.0 / (1.0 + (s_alpha**2) * T / ((1.0 + Shat2) * s2))
    B_omega = omega*beta_bar + (1.0-omega)*beta_hat
    v11 = b * (B_omega @ omega22hat @ B_omega.T) + h * (omega*delta_bar + (1.0-omega)*delta_hat) * (omega*Sigma_bar + (1.0-omega)*Sigma_hat)
    v12 = b * (B_omega @ omega22hat)
    v22 = b * omega22hat
    ee_top = omega * (beta_bar @ mu2hat) + (1.0-omega) * mu1hat
    ee_bot = mu2hat
    ee = np.concatenate([ee_top, ee_bot])
    V = np.block([[v11, v12], [v12.T, v22]])
    try:
        Vinv = np.linalg.inv(V)
    except np.linalg.LinAlgError:
        ev, U = np.linalg.eigh(V); ev = np.clip(ev, 1e-10, None)
        Vinv = U @ np.diag(1.0/ev) @ U.T
    w_dm = (Vinv @ ee).reshape(-1)
    w_full = np.zeros(N, float)
    w_full[mask_other] = w_dm[:m]
    w_full[j_fac]      = w_dm[m]
    s = float(np.sum(w_full))
    return np.ones(N)/N if (s == 0 or not np.isfinite(s)) else (w_full / abs(s))


# ========================= Turnover "paper" ========================= #

def _norm_weights__paper(W: pd.DataFrame, mode: str = "sum") -> pd.DataFrame:
    s = W.sum(axis=1); s = s.abs() if mode == "abssum" else s
    s = s.replace(0, float('nan'))
    return W.div(s, axis=0)


def _turnover_series_paper(W: pd.DataFrame, R: pd.DataFrame, mode: str = "sum") -> pd.Series:
    if W is None or R is None or W.empty or R.empty:
        return pd.Series(dtype=float)
    R = R.reindex(index=W.index, columns=W.columns).fillna(0.0)
    Wn   = _norm_weights__paper(W.astype(float), mode=mode)
    Wlag = Wn.shift(1)
    Wpre = _norm_weights__paper(Wlag * (1.0 + R), mode=mode)
    I = Wn.index.intersection(Wpre.index)
    return (Wn.loc[I].sub(Wpre.loc[I])).abs().sum(axis=1)


# ========================= Result container ========================= #

@dataclass
class Result:
    rets_net_excess: pd.Series
    turnover: pd.Series
    diag: Optional[pd.DataFrame] = None
    # artefatos para turnover "paper"
    weights_oos: Optional[pd.DataFrame] = None
    asset_rets_oos: Optional[pd.DataFrame] = None
    turnover_paper: Optional[pd.Series] = None
    turnover_rel_to_EW: Optional[float] = None


    # === NOVO: insumos p/ p-valor (JK vs EW e Sharpe=0) ===
    pval_inputs: Optional[dict] = None
# ========================= Backtest principal ========================= #

def rolling_backtest(
    rets_m: pd.DataFrame,
    rf_m: pd.Series,
    M: int = 120,
    cost_bps: float = 50.0,   # aceito, mas **não aplicado** (GROSS)
    vw_weights_m: Optional[pd.DataFrame] = None,
    vw_mkt_m: Optional[pd.DataFrame] = None,
    mvmin_alpha_override: Optional[float] = None,
    drop_mkt_from_universe: bool = False,
    mktrf_m: Optional[pd.Series] = None,
    returns_are_excess: bool = False,
    ignore_col: Optional[str] = None,
) -> Dict[str, Result]:

    rets_m = _sanitize_index(rets_m)
    rf_m   = _sanitize_index(rf_m)
    if vw_weights_m is not None:
        vw_weights_m = _sanitize_index(vw_weights_m)
    if vw_mkt_m is not None:
        vw_mkt_m = _sanitize_index(vw_mkt_m)
    if mktrf_m is not None and hasattr(mktrf_m, 'index'):
        mktrf_m = _sanitize_index(mktrf_m)

    assert isinstance(rets_m, pd.DataFrame)
    assert isinstance(rf_m, pd.Series)

    rets_all = rets_m.copy()
    rf_m = rf_m.reindex(rets_all.index)

    # Universo investível
    risky_cols = _risky_only_cols(rets_all)
    only_ff3, ff3_map = _detect_ff3_columns(rets_all)
    if only_ff3:
        investible_cols = [ff3_map["MktRF"], ff3_map["SMB"], ff3_map["HML"]]
    else:
        investible_cols = list(risky_cols)

    # Subconjunto principal
    rets_gen = rets_all[investible_cols].copy()
    if ignore_col is not None and ignore_col in rets_gen.columns:
        rets_gen = rets_gen.drop(columns=[ignore_col])

    # Retornos em excesso (Industry: subtrai RF das indústrias; fatores de excesso ficam como estão)
    if only_ff3:
        rets_inds_excess = rets_gen.copy()
    else:
        excess_alias = ["MktRF","MKT_RF","Mkt-RF","MKT-RF","MktMinusRF"]
        cols_excess = [c for c in rets_gen.columns if c in excess_alias]
        cols_raw    = [c for c in rets_gen.columns if c not in cols_excess]
        rets_inds_excess = rets_gen.copy()
        if not returns_are_excess:
            rets_inds_excess[cols_raw] = rets_inds_excess[cols_raw].subtract(rf_m, axis=0)

    # Alinhamento de datas
    dates = rets_gen.index.intersection(rets_inds_excess.index)
    rets_gen = rets_gen.loc[dates]
    rets_inds_excess = rets_inds_excess.loc[dates]
    rf_m = rf_m.loc[dates]
    if vw_weights_m is not None:
        vw_weights_m = vw_weights_m.reindex(dates)
    if vw_mkt_m is not None:
        if isinstance(vw_mkt_m, pd.DataFrame):
            if vw_mkt_m.shape[1] == 1:
                vw_mkt_m = vw_mkt_m.iloc[:,0]
            elif "MKT" in vw_mkt_m.columns:
                vw_mkt_m = vw_mkt_m["MKT"]
            else:
                vw_mkt_m = vw_mkt_m.iloc[:,0]
        vw_mkt_m = vw_mkt_m.reindex(dates)
    if mktrf_m is not None and hasattr(mktrf_m, 'reindex'):
        mktrf_m = mktrf_m.reindex(dates)

    N_gen = rets_gen.shape[1]
    N_ind = rets_inds_excess.shape[1]

    # GROSS para drift do turnover Eq.16
    # GROSS para drift do turnover (paper/MATLAB)
    # - Se a entrada já é em excesso (returns_are_excess=True), então gross = excesso + RF
    #   para ativos financiados (indústrias, MKT etc.). Para fatores self-financing (SMB/HML/UMD/RMW/CMA...),
    #   NÃO soma RF.
    # - Se a entrada NÃO é em excesso, então as indústrias já são total; nesse caso só convertemos colunas
    #   conhecidas como excesso (ex.: MktRF) para total somando RF.
    SELF_FINANCING = {"SMB","HML","UMD","WML","MOM","RMW","CMA","IML"}
    EXCESS_ALIASES  = {"MktRF","MKT_RF","Mkt-RF","MKT-RF","MktMinusRF","MKT"}

    # Universo "gen" (estratégias gerais)
    rets_gen_gross = rets_gen.copy()
    if returns_are_excess:
        cols_add_rf = [c for c in rets_gen_gross.columns if str(c) not in SELF_FINANCING]
        if cols_add_rf:
            rets_gen_gross[cols_add_rf] = rets_gen_gross[cols_add_rf].add(rf_m, axis=0)
    else:
        cols_add_rf = [c for c in rets_gen_gross.columns if c in EXCESS_ALIASES and str(c) not in SELF_FINANCING]
        if cols_add_rf:
            rets_gen_gross[cols_add_rf] = rets_gen_gross[cols_add_rf].add(rf_m, axis=0)

    # Universo "ind" (MP/BS etc.)
    rets_inds_gross = rets_inds_excess.copy()
    cols_add_rf = [c for c in rets_inds_gross.columns if str(c) not in SELF_FINANCING]
    if cols_add_rf:
        rets_inds_gross[cols_add_rf] = rets_inds_gross[cols_add_rf].add(rf_m, axis=0)

    # Validação por janela (sem NaN)
    valid = np.zeros(len(dates), bool)
    X_all = rets_inds_excess.values
    rf_vals = rf_m.values
    for t in range(M, len(dates)):
        block_X  = X_all[t-M:t, :]
        block_rf = rf_vals[t-M:t]
        if np.isfinite(block_X).all() and np.isfinite(block_rf).all():
            valid[t] = True

    # ===== MV-IS OOS: pesos e Sharpe consistentes com filtro de NaN =====
    try:
        oos_slice = slice(M, len(dates))
        X_oos_df = _sanitize_index(rets_inds_excess).iloc[oos_slice].dropna(how="any")
        T_oos = X_oos_df.shape[0]
        if T_oos < 3:
            w_MV_IS_full = np.ones(N_ind)/max(N_ind,1)
            sr_mv_is = float("nan")
        else:
            X_oos = X_oos_df.to_numpy(float)
            mu_full = X_oos.mean(axis=0)
            Sigma_unb = np.cov(X_oos, rowvar=False, ddof=1)
            Sigma_full = Sigma_unb * ((T_oos - 1.0) / T_oos)
            w_MV_IS_full = _w_mv_paper(mu_full, Sigma_full, ridge=1e-12)
            denom = float(w_MV_IS_full @ Sigma_full @ w_MV_IS_full)
            denom = denom if np.isfinite(denom) and denom > 0 else np.finfo(float).eps
            num = float(w_MV_IS_full @ mu_full) if np.isfinite(w_MV_IS_full @ mu_full) else 0.0
            sr_mv_is = num / np.sqrt(denom)
    except Exception:
        w_MV_IS_full = np.ones(N_ind)/max(N_ind,1)
        sr_mv_is = float("nan")

    # ===== Preparação de coletores =====
    keys = ["EW","MV","BS","DM","MIN","VW","MP","MV-C","BS-C","MIN-C","G-MIN-C","MV-MIN","EW-MIN","MV (in sample)"]
    rex = {k: [] for k in keys}
    tov = {k: [] for k in keys}
    idx_oos: List[pd.Timestamp] = []

    weights_store: Dict[str, list] = {k: [] for k in keys}
    returns_store: Dict[str, list] = {k: [] for k in keys}

    # Alphas para turnover risky-only no estilo MATLAB (necessário p/ BS, MP, MV-MIN)
    alpha_store: Dict[str, list] = {k: [] for k in ["BS","MP","MV-MIN"]}
    rex_store_ind: list = []   # vetores de retornos em excesso (mesma dimensão de cols_ind) por data OOS
    rf_store_oos: list = []    # rf por data OOS
    cols_store: Dict[str, list] = {}
    cols_gen = list(rets_gen.columns)
    cols_ind = list(rets_inds_excess.columns) if isinstance(rets_inds_excess, pd.DataFrame) else []
    for k in ["EW","MV","DM","MIN","MV-C","BS-C","MIN-C","G-MIN-C","MV-MIN","EW-MIN","MV (in sample)","VW"]:
        cols_store[k] = cols_gen
    for k in ["MP","BS"]:
        cols_store[k] = cols_ind

    weighted_keys_gen = ["EW","MV","DM","MIN","MV-C","BS-C","MIN-C","G-MIN-C","MV-MIN","EW-MIN","MV (in sample)"]
    W_prev_gen = {k: np.ones(N_gen)/N_gen for k in weighted_keys_gen}
    W_prev_MP  = np.ones(N_ind)/N_ind if N_ind>0 else np.array([])
    W_prev_BS  = np.ones(N_ind)/N_ind if N_ind>0 else np.array([])

    if vw_weights_m is not None:
        sums = vw_weights_m.sum(axis=1).replace(0.0, np.nan)
        vw_weights_m = vw_weights_m.div(sums, axis=0).fillna(0.0)

    # ===== Loop OOS =====
    for t in range(M, len(dates)):
        if not valid[t]:
            continue

        is_slice_gen = rets_gen.iloc[t-M:t]
        rf_is = rf_m.iloc[t-M:t]
        T_is = is_slice_gen.shape[0]
        is_slice_ind = rets_inds_excess.loc[is_slice_gen.index]

        # Série de treino (excesso) para modelos gerais
        excess_alias = ["MktRF","MKT_RF","Mkt-RF","MKT-RF","MktMinusRF"]
        cols_excess = [c for c in is_slice_gen.columns if c in excess_alias]
        cols_raw    = [c for c in is_slice_gen.columns if c not in cols_excess]
        X_gen_df = is_slice_gen.copy()
        if not returns_are_excess:
            X_gen_df[cols_raw] = X_gen_df[cols_raw].sub(rf_is.values, axis=0)
        X_gen = X_gen_df.values
        mu_gen = X_gen.mean(axis=0)
        Sigma_gen = np.cov(X_gen, rowvar=False, ddof=1)

        X_ind = is_slice_ind.values
        mu_ind = X_ind.mean(axis=0)
        S_unbiased = np.cov(X_ind, rowvar=False, ddof=1)
        T_is = X_ind.shape[0]
        S_ind = S_unbiased * ((T_is - 1.0) / T_is)  # MLE

        mu_mp = np.asarray(mu_ind, float).reshape(-1)

        oneN_gen = np.ones(N_gen)/N_gen
        w_EW = oneN_gen

        # MV (OOS)
        if only_ff3:
            w_MV = _w_mv(mu_gen, Sigma_gen)
        else:
            w_MV = _w_mv_paper(mu_gen, Sigma_gen)

        # Fatores para DM
        investiveis = list(rets_gen.columns)
        investiveis_set = set(investiveis)
        aliases_mkt = ["MktRF", "MKT_RF", "MKT", "Mkt-RF", "MKT-RF", "MktMinusRF"]
        universo_ff3 = (
            any(c in investiveis_set for c in aliases_mkt) and
            "SMB" in investiveis_set and
            "HML" in investiveis_set and
            len(investiveis_set) == 3
        )
        if universo_ff3:
            factor_cols = []
            for c in ["MktRF","MKT_RF","MKT","Mkt-RF","MKT-RF","MktMinusRF"]:
                if c in rets_all.columns:
                    factor_cols.append(c); break
            for c in ["SMB","HML"]:
                if c in rets_all.columns:
                    factor_cols.append(c)
        else:
            factor_cols = []
            for c in ["MktRF","MKT_RF","MKT","Mkt-RF","MKT-RF","MktMinusRF"]:
                if c in rets_all.columns:
                    factor_cols.append(c); break

        F_block = None
        if len(factor_cols) > 0:
            F_full = rets_all[factor_cols].copy()
            if not F_full.index.is_monotonic_increasing:
                F_full = F_full.sort_index()
            if not F_full.index.is_unique:
                F_full = F_full[~F_full.index.duplicated(keep="last")]
            F_is = F_full.reindex(is_slice_gen.index).copy()
            if "MktRF" in F_is.columns:
                F_is = F_is.rename(columns={"MktRF":"MKT"})
            elif "MKT_RF" in F_is.columns:
                F_is = F_is.rename(columns={"MKT_RF":"MKT"})
            elif "MKT" in F_is.columns and "MKT" not in investiveis_set:
                F_is["MKT"] = F_is["MKT"].sub(rf_is)
            keep = [c for c in (["MKT","SMB","HML"] if universo_ff3 else ["MKT"]) if c in F_is.columns]
            F_is_df = F_is[keep]
            F_block = F_is_df.values if len(keep) > 0 else None

        R_is = _sanitize_index(rets_inds_excess).reindex(is_slice_gen.index)
        F_is_use = None
        if 'F_is_df' in locals() and isinstance(F_is_df, pd.DataFrame) and F_is_df.shape[1] > 0:
            F_is_use = _sanitize_index(F_is_df).reindex(is_slice_gen.index)
        mask = ~R_is.isna().any(axis=1)
        if F_is_use is not None:
            mask &= ~F_is_use.isna().any(axis=1)
        R_is = R_is.loc[mask]
        if F_is_use is not None:
            F_is_use = F_is_use.loc[mask]
        X_block = R_is.values
        F_block = F_is_use.values if F_is_use is not None else None

        w_DM   = _w_dm_pastor(X_block, F_block, sigma_alpha_annual=0.01)
        w_MIN  = _w_min(mu_gen, Sigma_gen)
        w_MINC = _w_min_c(Sigma_gen)
        w_MVC  = _w_mv_c(mu_gen, Sigma_gen)
        w_BSC  = _w_bs_c(mu_gen, Sigma_gen, T=T_is)
        w_GMINC = _w_gmin_c(Sigma_gen, lbound=1.0/(2.0*N_gen))

        # MV-MIN (Kan-Zhou) — guardar alpha para turnover MATLAB
        alpha_MV_MIN = _w_mv_min_kz(mu_ind, S_ind, M=T_is, normalize="alpha")
        w_MV_MIN = _alpha_to_w_ro(alpha_MV_MIN)

        w_EW_MIN = _w_ew_min_kz1n_from_S_mle(S_ind, T_is, ridge=1e-12)
        w_MV_IS  = w_MV_IS_full

        # Pastor–MacKinlay (MP) — guardar alpha
        alpha_MP = _w_mp_mackinlay_pastor_new(mu_mp, S_ind, gamma=1.0, normalize="alpha")
        w_MP = _alpha_to_w_ro(alpha_MP)

        # Bayes–Stein (BS) — guardar alpha
        alpha_BS, phi_val, q_val, denom_val, condS_val = _alpha_bs_industry_from_window(X_ind, gamma=1.0, ridge=1e-12)
        w_BS = _alpha_to_w_ro(alpha_BS)

        def drift(w_prev, r_prev):
            g = w_prev*(1.0 + r_prev); s = np.sum(g); return g/(s if s!=0 else 1.0)

        if len(idx_oos) == 0:
            w_before_gen = {k: W_prev_gen.get(k, oneN_gen) for k in weighted_keys_gen}
            w_before_MP = W_prev_MP.copy()
            w_before_BS = W_prev_BS.copy()
        else:
            last_date = idx_oos[-1]
            pos = dates.get_loc(last_date)
            r_prev_gen = rets_gen_gross.iloc[pos].values
            w_before_gen = {k: drift(W_prev_gen[k], r_prev_gen) for k in weighted_keys_gen}
            r_prev_ind = rets_inds_gross.iloc[pos].values
            w_before_MP = drift(W_prev_MP, r_prev_ind)
            w_before_BS = drift(W_prev_BS, r_prev_ind)

        # MV (in sample) mantém pesos fixos (sem drift)
        w_before_gen["MV (in sample)"] = w_MV_IS_full

        W_now_gen = {
            "EW": w_EW, "MV": w_MV, "DM": w_DM, "MIN": w_MIN,
            "MV-C": w_MVC, "BS-C": w_BSC, "MIN-C": w_MINC, "G-MIN-C": w_GMINC,
            "MV-MIN": w_MV_MIN, "EW-MIN": w_EW_MIN, "MV (in sample)": w_MV_IS,
        }

        TO_gen = {k: float(np.sum(np.abs(W_now_gen[k] - w_before_gen[k]))) for k in W_now_gen.keys()}
        # --- CORREÇÃO local só para MV-min: turnover 'paper' de 1 passo ---
        # gross returns do t-1 nas MESMAS colunas de rets_gen (investible_cols)
        R_prev_gross = (rets_gen_gross.iloc[t-1].values
                        if returns_are_excess else
                        rets_gen_gross.iloc[t-1].values)  # já está GROSS acima

        TO_gen["MV-MIN"] = _turnover_step_paper_risky_only(
            w_target=W_now_gen["MV-MIN"],
            w_prev=w_before_gen["MV-MIN"],
            gross_ret_prev=1.0 + R_prev_gross  # já contém RF + excesso nas colunas FF3
)
        TO_MP = float(np.sum(np.abs(w_MP - w_before_MP))) if N_ind>0 else 0.0
        TO_BS = float(np.sum(np.abs(w_BS - w_before_BS))) if N_ind>0 else 0.0

        rex_t_gen = rets_gen.iloc[t].values if returns_are_excess else (rets_gen.iloc[t].values - float(rf_m.iloc[t]))
        rex_t_ind = rets_inds_excess.iloc[t].values  # define ANTES de usar

        # Retornos por estratégia
        G = {k: float(np.dot(W_now_gen[k], rex_t_gen)) for k in W_now_gen.keys()}
        G["MV (in sample)"] = float(np.dot(w_MV_IS, rex_t_ind))  # medir na MESMA base

        # VW
        if mktrf_m is not None:
            G["VW"] = float(mktrf_m.iloc[t])
        elif vw_mkt_m is not None:
            G["VW"] = float(vw_mkt_m.iloc[t])
        else:
            mkt_alias = None
            for c in ["MktRF","Mkt-RF","MKT-RF","MKT_RF","MktMinusRF"]:
                if c in rets_all.columns:
                    mkt_alias = c; break
            if mkt_alias is not None:
                G["VW"] = float(rets_all.loc[dates[t], mkt_alias])
            elif "MKT" in rets_all.columns:
                r = rets_all.loc[dates[t], "MKT"]
                G["VW"] = float(r if returns_are_excess else (r - float(rf_m.iloc[t])))
            else:
                r_row = rets_all.loc[dates[t]]
                rex_ind_t = r_row.values if returns_are_excess else (r_row.values - float(rf_m.iloc[t]))
                G["VW"] = float(np.mean(rex_ind_t))

        G["MP"] = float(np.dot(w_MP, rex_t_ind))
        G["BS"] = float(np.dot(w_BS, rex_t_ind))

        for k in W_now_gen.keys():
            rex[k].append(G[k]); tov[k].append(TO_gen[k]); W_prev_gen[k] = W_now_gen[k]
        rex["VW"].append(G["VW"]); tov["VW"].append(0.0)
        rex["MP"].append(G["MP"]); tov["MP"].append(TO_MP); W_prev_MP = w_MP
        rex["BS"].append(G["BS"]); tov["BS"].append(TO_BS); W_prev_BS = w_BS

        # Artefatos p/ turnover "paper"
        Rg_t = rets_gen_gross.iloc[t].reindex(cols_gen).to_numpy()
        for _k in ["EW","MV","DM","MIN","MV-C","BS-C","MIN-C","G-MIN-C","MV-MIN","EW-MIN","MV (in sample)","VW"]:
            if _k in W_now_gen:
                weights_store[_k].append(W_now_gen[_k].copy())
                returns_store[_k].append(Rg_t.copy())
        if len(cols_ind) > 0:
            Ri_t = rets_inds_gross.iloc[t].reindex(cols_ind).to_numpy()
            weights_store["MP"].append(w_MP.copy()); returns_store["MP"].append(Ri_t.copy())
            weights_store["BS"].append(w_BS.copy()); returns_store["BS"].append(Ri_t.copy())


        # --- Alphas p/ turnover MATLAB (BS, MP, MV-MIN) ---
        if len(cols_ind) > 0:
            alpha_store["BS"].append(alpha_BS.copy())
            alpha_store["MP"].append(alpha_MP.copy())
            alpha_store["MV-MIN"].append(alpha_MV_MIN.copy())
            rex_store_ind.append(rets_inds_excess.iloc[t].reindex(cols_ind).to_numpy())
            rf_store_oos.append(float(rf_m.iloc[t]))
        idx_oos.append(dates[t])

    # ===== Montagem do resultado =====
    idx = pd.Index(idx_oos, name=rets_gen.index.name)
    out: Dict[str, Result] = {}
    for k in keys:
        out[k] = Result(
            rets_net_excess=pd.Series(rex[k], index=idx, name=f"{k}_excess"),
            turnover=pd.Series(tov[k], index=idx, name=f"{k}_turnover"),
        )

    # ===== Turnover "paper" e razões relativas (Tabela 5) =====
    # Observação: definimos turnover do RESULTADO como a métrica do paper:
    #   TO_t = sum_j | w_{j,t} - w^{BH}_{j,t} |
    # onde w^{BH} é o vetor de pesos do mês anterior "driftado" por (1+R_gross) e renormalizado por |1'w|.

    tau_EW = pd.Series(dtype=float)
    avg_tau_EW = float("nan")

    if len(weights_store.get("EW", [])) and len(returns_store.get("EW", [])):
        ew_cols = cols_store.get("EW", cols_gen)
        W_EW = pd.DataFrame(weights_store["EW"], index=idx, columns=ew_cols)
        R_EW = pd.DataFrame(returns_store["EW"], index=idx, columns=ew_cols)
        tau_EW = _turnover_series_paper(W_EW, R_EW, mode="abssum")
        avg_tau_EW = float(tau_EW.mean()) if len(tau_EW) else float("nan")

        # Pré-computa séries para turnover MATLAB (alfa + retornos em excesso + rf)
        tau_matlab = {}
        if len(alpha_store.get("BS", [])) and len(rex_store_ind) and len(rf_store_oos):
            idx_tmp = idx
            cols_tmp = list(cols_ind)
            rex_df = pd.DataFrame(rex_store_ind, index=idx_tmp, columns=cols_tmp)
            rf_s  = pd.Series(rf_store_oos, index=idx_tmp)

            for _nm in ["BS","MP","MV-MIN"]:
                a_df = pd.DataFrame(alpha_store[_nm], index=idx_tmp, columns=cols_tmp)
                tau_matlab[_nm] = _turnover_series_matlab_ro(a_df, rex_df, rf_s)

        out["EW"].weights_oos = W_EW
        out["EW"].asset_rets_oos = R_EW
        out["EW"].turnover_paper = tau_EW
        out["EW"].turnover = tau_EW.reindex(idx)
        out["EW"].turnover_rel_to_EW = 1.0

    for _k in keys:
        if _k == "EW":
            continue
        if len(weights_store.get(_k, [])) and len(returns_store.get(_k, [])):
            k_cols = cols_store.get(_k, cols_gen if _k not in ("MP", "BS") else cols_ind)
            Wk = pd.DataFrame(weights_store[_k], index=idx, columns=k_cols)
            Rk = pd.DataFrame(returns_store[_k], index=idx, columns=k_cols)
            tau_k = tau_matlab.get(_k, _turnover_series_paper(Wk, Rk, mode="abssum"))

            out[_k].weights_oos = Wk
            out[_k].asset_rets_oos = Rk
            out[_k].turnover_paper = tau_k

            # sobrescreve turnover para usar a definição do paper
            out[_k].turnover = tau_k.reindex(idx)

            out[_k].turnover_rel_to_EW = (
                float(tau_k.mean()) / avg_tau_EW
                if (len(tau_k) and np.isfinite(avg_tau_EW) and avg_tau_EW > 0)
                else float("nan")
            )

    # Diagnósticos agregados (mantém compatibilidade com notebooks)
    out["EW"].diag = pd.DataFrame({"Sharpe_MV_IS_OOS": [sr_mv_is]})

    return out
