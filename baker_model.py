# Author: Zhi Zheng
import numpy as np

from data_and_corey import (
    BETA_G,
    BETA_O,
    BETA_W,
    KRO_MAX,
    MU_G,
    MU_O,
    MU_W,
    SGC,
    SORG,
    SORW,
    SO_REF,
    SWC,
    SWC_WG,
)


def _phi_so(So):
    if SO_REF <= 0:
        return 1.0
    t = (So - SORG) / SO_REF
    if np.isscalar(t):
        return 0.0 if t <= 0.0 else 1.0 - np.exp(-t)
    return np.where(t <= 0.0, 0.0, 1.0 - np.exp(-t))


def baker_kro(Sw, Sg, cp, f2):
    So = 1.0 - Sw - Sg
    if Sw < SWC or So <= SORG:
        return 0.0
    krow = float(f2["krow_wo"](Sw))
    krog = float(f2["krog_og"](So))
    denom = (Sw - SWC) + (Sg - SGC)
    wg = 0.0 if denom < 1e-30 else np.clip((Sg - SGC) / denom, 0.0, 1.0)
    return max(0.0, min(((1.0 - wg) * krow + wg * krog) * _phi_so(So), KRO_MAX))


def baker_krw(Sw, Sg, cp, f2):
    So = 1.0 - Sw - Sg
    if Sw < SWC:
        return 0.0
    krw_wo = float(f2["krw_wo"](Sw))
    krw_wg = float(f2["krw_wg"](Sw))
    denom = max(0.0, So - SORG) + max(0.0, Sg - SGC)
    wg = 0.0 if denom < 1e-30 else np.clip(max(0.0, Sg - SGC) / denom, 0.0, 1.0)
    return max(0.0, (1.0 - wg) * krw_wo + wg * krw_wg)


def baker_krg(Sw, Sg, cp, f2):
    So = 1.0 - Sw - Sg
    if Sg <= SGC:
        return 0.0
    krg_og = float(f2["krg_og"](Sg))
    krg_wg = float(f2["krg_wg"](Sg))
    denom = max(0.0, So - SORG) + max(0.0, Sw - SWC)
    ww = 0.0 if denom < 1e-30 else np.clip(max(0.0, Sw - SWC) / denom, 0.0, 1.0)
    return max(0.0, (1.0 - ww) * krg_og + ww * krg_wg)


def fractional_flow(Sw, Sg, cp, f2):
    krw = baker_krw(Sw, Sg, cp, f2)
    kro = baker_kro(Sw, Sg, cp, f2)
    krg = baker_krg(Sw, Sg, cp, f2)
    lw = krw / MU_W
    lo = kro / MU_O
    lg = krg / MU_G
    lt = lw + lo + lg
    if lt < 1e-30:
        return 0.0, 0.0, 0.0
    return lw / lt, lo / lt, lg / lt


def fractional_flow_vec(Sw, Sg, cp):
    Sw = np.asarray(Sw, dtype=float)
    Sg = np.asarray(Sg, dtype=float)
    So = 1.0 - Sw - Sg

    dw_wo = 1.0 - SWC - SORW
    swn = np.clip((Sw - SWC) / dw_wo, 0.0, 1.0)
    krw_wo = cp.krw_end * (BETA_W * swn + (1.0 - BETA_W) * swn**cp.nw)
    krw_wo = np.where(Sw < SWC, 0.0, krw_wo)

    dw_wg = 1.0 - SWC_WG
    swn2 = np.clip((Sw - SWC_WG) / dw_wg, 0.0, 1.0)
    krw_wg = cp.krw_wg_end * (BETA_W * swn2 + (1.0 - BETA_W) * swn2**cp.nw_wg)
    krw_wg = np.where(Sw < SWC_WG, 0.0, krw_wg)

    dg_og = 1.0 - SWC - SORG - SGC
    sgn = np.clip((Sg - SGC) / dg_og, 0.0, 1.0)
    krg_og = cp.krg_end * (BETA_G * sgn + (1.0 - BETA_G) * sgn**cp.ng)
    krg_og = np.where(Sg <= SGC, 0.0, krg_og)

    sgn2 = np.clip(Sg / dw_wg, 0.0, 1.0)
    krg_wg = cp.krg_wg_end * (BETA_G * sgn2 + (1.0 - BETA_G) * sgn2**cp.ng_wg)
    krg_wg = np.where(Sg <= 0.0, 0.0, krg_wg)

    sow = np.clip((1.0 - Sw - SORW) / dw_wo, 0.0, 1.0)
    krow = cp.krow_end * (BETA_O * sow + (1.0 - BETA_O) * sow**cp.now)

    dog = 1.0 - SWC - SORG
    sog = np.clip((So - SORG) / dog, 0.0, 1.0)
    krog = cp.krog_end * (BETA_O * sog + (1.0 - BETA_O) * sog**cp.nog)

    kro = np.zeros_like(Sw)
    valid_o = (So > SORG) & (Sw >= SWC)
    if np.any(valid_o):
        denom = (Sw[valid_o] - SWC) + (Sg[valid_o] - SGC)
        wg = np.where(denom > 1e-30, np.clip((Sg[valid_o] - SGC) / denom, 0.0, 1.0), 0.0)
        kro_raw = (1.0 - wg) * krow[valid_o] + wg * krog[valid_o]
        kro[valid_o] = np.clip(kro_raw * _phi_so(So[valid_o]), 0.0, KRO_MAX)

    krw = np.zeros_like(Sw)
    valid_w = Sw >= SWC
    if np.any(valid_w):
        denom = np.maximum(0.0, So[valid_w] - SORG) + np.maximum(0.0, Sg[valid_w] - SGC)
        wg = np.where(denom > 1e-30, np.clip(np.maximum(0.0, Sg[valid_w] - SGC) / denom, 0.0, 1.0), 0.0)
        krw[valid_w] = np.maximum(0.0, (1.0 - wg) * krw_wo[valid_w] + wg * krw_wg[valid_w])

    krg = np.zeros_like(Sw)
    valid_g = Sg > SGC
    if np.any(valid_g):
        denom = np.maximum(0.0, So[valid_g] - SORG) + np.maximum(0.0, Sw[valid_g] - SWC)
        ww = np.where(denom > 1e-30, np.clip(np.maximum(0.0, Sw[valid_g] - SWC) / denom, 0.0, 1.0), 0.0)
        krg[valid_g] = np.maximum(0.0, (1.0 - ww) * krg_og[valid_g] + ww * krg_wg[valid_g])

    lw = krw / MU_W
    lo = kro / MU_O
    lg = krg / MU_G
    lt = np.maximum(lw + lo + lg, 1e-30)
    return lw / lt, lg / lt

