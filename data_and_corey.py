# Author: Zhi Zheng
import os
import numpy as np

from cases import get_case


ACTIVE_CASE = get_case(os.environ.get("THREE_PHASE_CASE", "water_wet"))

MU_W = ACTIVE_CASE.mu_w
MU_O = ACTIVE_CASE.mu_o
MU_G = ACTIVE_CASE.mu_g

SWC = ACTIVE_CASE.swc
SORW = ACTIVE_CASE.sorw
SORG = ACTIVE_CASE.sorg
SGC = ACTIVE_CASE.sgc
SWC_WG = ACTIVE_CASE.swc_wg

NW = ACTIVE_CASE.nw
NOW = ACTIVE_CASE.now
KRW_END = ACTIVE_CASE.krw_end
KROW_END = ACTIVE_CASE.krow_end
NG = ACTIVE_CASE.ng
NOG = ACTIVE_CASE.nog
KRG_END = ACTIVE_CASE.krg_end
KROG_END = ACTIVE_CASE.krog_end
NW_WG = ACTIVE_CASE.nw_wg
NG_WG = ACTIVE_CASE.ng_wg
KRW_WG_END = ACTIVE_CASE.krw_wg_end
KRG_WG_END = ACTIVE_CASE.krg_wg_end

KRW_MAX = max(KRW_END, KRW_WG_END)
KRO_MAX = ACTIVE_CASE.kro_max
KRG_MAX = max(KRG_END, KRG_WG_END)

BETA_W = ACTIVE_CASE.beta_w
BETA_O = ACTIVE_CASE.beta_o
BETA_G = ACTIVE_CASE.beta_g
SO_REF = ACTIVE_CASE.so_ref


def corey_kr(S, S_min, S_max, kr_end, n, beta=0.0):
    S = np.asarray(S, dtype=float)
    denom = max(S_max - S_min, 1e-30)
    Sn = np.clip((S - S_min) / denom, 0.0, 1.0)
    return np.maximum(0.0, kr_end * (beta * Sn + (1.0 - beta) * Sn**n))


class CoreyParams:
    def __init__(self):
        self.case_name = ACTIVE_CASE.name
        self.nw = NW
        self.now = NOW
        self.krw_end = KRW_END
        self.krow_end = KROW_END
        self.ng = NG
        self.nog = NOG
        self.krg_end = KRG_END
        self.krog_end = KROG_END
        self.nw_wg = NW_WG
        self.ng_wg = NG_WG
        self.krw_wg_end = KRW_WG_END
        self.krg_wg_end = KRG_WG_END

    def summary(self):
        return (
            f"{ACTIVE_CASE.title}: "
            f"mu=({MU_W:g}, {MU_O:g}, {MU_G:g}), "
            f"Swc={SWC:g}, Sorw={SORW:g}, Sorg={SORG:g}, "
            f"WO(nw={self.nw:g}, now={self.now:g}), "
            f"OG(ng={self.ng:g}, nog={self.nog:g}), "
            f"WG(nw={self.nw_wg:g}, ng={self.ng_wg:g})"
        )


def make_two_phase_funcs(cp: CoreyParams):
    dw_wo = 1.0 - SWC - SORW
    dog = 1.0 - SWC - SORG
    dg_og = 1.0 - SWC - SORG - SGC
    return {
        "krw_wo": lambda Sw: corey_kr(Sw, SWC, 1.0 - SORW, cp.krw_end, cp.nw, BETA_W),
        "krow_wo": lambda Sw: corey_kr(1.0 - Sw - SORW, 0.0, dw_wo, cp.krow_end, cp.now, BETA_O),
        "krg_og": lambda Sg: corey_kr(Sg, SGC, dg_og, cp.krg_end, cp.ng, BETA_G),
        "krog_og": lambda So: corey_kr(So - SORG, 0.0, dog, cp.krog_end, cp.nog, BETA_O),
        "krw_wg": lambda Sw: corey_kr(Sw, SWC_WG, 1.0, cp.krw_wg_end, cp.nw_wg, BETA_W),
        "krg_wg": lambda Sg: corey_kr(Sg, 0.0, 1.0 - SWC_WG, cp.krg_wg_end, cp.ng_wg, BETA_G),
    }


_cp0 = CoreyParams()
_f2 = make_two_phase_funcs(_cp0)

WO_Sw = np.linspace(SWC, 1.0 - SORW, 12)
WO_krw = _f2["krw_wo"](WO_Sw)
WO_kro = _f2["krow_wo"](WO_Sw)
OG_So = np.linspace(SORG, 1.0 - SWC, 12)
OG_kro = _f2["krog_og"](OG_So)
OG_Sg = np.linspace(SGC, 1.0 - SWC - SORG, 12)
OG_krg = _f2["krg_og"](OG_Sg)
WG_Sw = np.linspace(SWC_WG, 1.0, 12)
WG_krw = _f2["krw_wg"](WG_Sw)
WG_Sg = np.linspace(0.0, 1.0 - SWC_WG, 12)
WG_krg = _f2["krg_wg"](WG_Sg)

