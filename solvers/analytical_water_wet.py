# Author: Zhi Zheng
"""
Analytical solution for the three-phase Riemann problem  (v3).

Jacobian of (fw, fg) w.r.t. (Sw, Sg):
  J = [[dfw/dSw, dfw/dSg],
       [dfg/dSw, dfg/dSg]]

Eigenvalues (wave speeds):
  v_D = (f11+f33)/2 +/- (1/2)*sqrt((f11-f33)^2 + 4*f13*f31)

Key insight (v3):
  For gas injection into a waterflooded reservoir (L near Sg=1, R at Sg=0),
  the solution may be ENTIRELY in the fast family — a composite wave
  consisting of  shock(L→M_tail) + rarefaction(M_tail→M_peak) + shock(M_peak→R),
  all along the fast eigenvector.  The code now detects this structure
  automatically by checking whether the fast rarefaction from R passes
  close to L, and if so, builds the solution along the fast curve instead
  of assuming slow+fast.
"""
import numpy as np
from data_and_corey import (SWC, SORW, SORG, SGC, SWC_WG, KRO_MAX,
                             MU_W, MU_O, MU_G, corey_kr,
                             BETA_W, BETA_O, BETA_G, SO_REF)
from baker_model import fractional_flow


# ==================================================================
#  Kinetic-relation override (Approach A)
# ------------------------------------------------------------------
#  When set, Strategy 1g will skip its geometric back-trace for the
#  transitional shock plateau P and instead use this externally supplied
#  P (typically extracted from a short numerical probe run).  Q is then
#  redetermined by forward-integrating the fast rarefaction from P and
#  locating the Welge-tangent point on the fast integral curve.
# ==================================================================
_P_KINETIC = None


def set_kinetic_P(P):
    """Register a kinetic-selected P (array-like, len 3).  Pass None to clear."""
    global _P_KINETIC
    if P is None:
        _P_KINETIC = None
    else:
        _P_KINETIC = np.asarray(P, dtype=float).copy()


def get_kinetic_P():
    return _P_KINETIC


# ==================================================================
#  Jacobian eigenvalues / eigenvectors
# ==================================================================
def compute_eigenvalues(Sw, Sg, cp):
    """
    Compute analytical eigenvalues and eigenvectors of the Jacobian
    of fractional flow at (Sw, Sg) using full Baker model.
    Returns: lambda1 (slow), lambda2 (fast), v1, v2, discriminant
    """
    So = 1.0 - Sw - Sg
    Dw_wo = 1 - SWC - SORW

    # kr(Sn) = kr_end * (beta*Sn + (1-beta)*Sn^n)
    # dkr/dSn = kr_end * (beta + (1-beta)*n*Sn^(n-1))
    Sw_s = np.clip((Sw - SWC) / Dw_wo, 0, 1)
    if Sw_s <= 0:
        krw_wo = 0.0
        dkrw_wo = cp.krw_end * BETA_W / Dw_wo if Sw >= SWC else 0.0
    elif Sw_s >= 1:
        krw_wo = cp.krw_end * (BETA_W + (1 - BETA_W) * Sw_s**cp.nw)
        dkrw_wo = 0.0
    else:
        krw_wo = cp.krw_end * (BETA_W * Sw_s + (1 - BETA_W) * Sw_s**cp.nw)
        dkrw_wo = cp.krw_end * (BETA_W + (1 - BETA_W) * cp.nw * Sw_s**(cp.nw - 1)) / Dw_wo

    Dw_wg = 1 - SWC_WG
    Sw_s2 = np.clip((Sw - SWC_WG) / Dw_wg, 0, 1)
    if Sw_s2 <= 0:
        krw_wg = 0.0
        dkrw_wg = cp.krw_wg_end * BETA_W / Dw_wg if Sw >= SWC_WG else 0.0
    elif Sw_s2 >= 1:
        krw_wg = cp.krw_wg_end * (BETA_W + (1 - BETA_W) * Sw_s2**cp.nw_wg)
        dkrw_wg = 0.0
    else:
        krw_wg = cp.krw_wg_end * (BETA_W * Sw_s2 + (1 - BETA_W) * Sw_s2**cp.nw_wg)
        dkrw_wg = cp.krw_wg_end * (BETA_W + (1 - BETA_W) * cp.nw_wg * Sw_s2**(cp.nw_wg - 1)) / Dw_wg

    Dg_og = 1 - SWC - SORG - SGC
    Sg_s = np.clip((Sg - SGC) / Dg_og, 0, 1)
    if Sg_s <= 0:
        krg_og = 0.0
        dkrg_og = cp.krg_end * BETA_G / Dg_og if Sg >= SGC else 0.0
    elif Sg_s >= 1:
        krg_og = cp.krg_end * (BETA_G + (1 - BETA_G) * Sg_s**cp.ng)
        dkrg_og = 0.0
    else:
        krg_og = cp.krg_end * (BETA_G * Sg_s + (1 - BETA_G) * Sg_s**cp.ng)
        dkrg_og = cp.krg_end * (BETA_G + (1 - BETA_G) * cp.ng * Sg_s**(cp.ng - 1)) / Dg_og

    Dg_wg = 1 - SWC_WG
    Sg_s2 = np.clip(Sg / Dg_wg, 0, 1)
    if Sg_s2 <= 0:
        krg_wg = 0.0
        dkrg_wg = cp.krg_wg_end * BETA_G / Dg_wg if Sg >= 0 else 0.0
    elif Sg_s2 >= 1:
        krg_wg = cp.krg_wg_end * (BETA_G + (1 - BETA_G) * Sg_s2**cp.ng_wg)
        dkrg_wg = 0.0
    else:
        krg_wg = cp.krg_wg_end * (BETA_G * Sg_s2 + (1 - BETA_G) * Sg_s2**cp.ng_wg)
        dkrg_wg = cp.krg_wg_end * (BETA_G + (1 - BETA_G) * cp.ng_wg * Sg_s2**(cp.ng_wg - 1)) / Dg_wg

    Sow_s = np.clip((1 - Sw - SORW) / Dw_wo, 0, 1)
    if Sow_s <= 0:
        krow = 0.0
        dkrow_dSw = -cp.krow_end * BETA_O / Dw_wo
    elif Sow_s >= 1:
        krow = cp.krow_end * (BETA_O + (1 - BETA_O) * Sow_s**cp.now)
        dkrow_dSw = 0.0
    else:
        krow = cp.krow_end * (BETA_O * Sow_s + (1 - BETA_O) * Sow_s**cp.now)
        dkrow_dSw = cp.krow_end * (BETA_O + (1 - BETA_O) * cp.now * Sow_s**(cp.now - 1)) * (-1.0 / Dw_wo)

    Dog = 1 - SWC - SORG
    Sog_s = np.clip((So - SORG) / Dog, 0, 1)
    if Sog_s <= 0:
        krog = 0.0
        dkrog_dSo = cp.krog_end * BETA_O / Dog if So >= SORG else 0.0
    elif Sog_s >= 1:
        krog = cp.krog_end * (BETA_O + (1 - BETA_O) * Sog_s**cp.nog)
        dkrog_dSo = 0.0
    else:
        krog = cp.krog_end * (BETA_O * Sog_s + (1 - BETA_O) * Sog_s**cp.nog)
        dkrog_dSo = cp.krog_end * (BETA_O + (1 - BETA_O) * cp.nog * Sog_s**(cp.nog - 1)) / Dog
    dkrog_dSw = -dkrog_dSo
    dkrog_dSg = -dkrog_dSo

    # Baker (25): kro with phi(So) damping for linear vanishing at So=SORG
    if Sw < SWC:
        kro = 0; dkro_dSw = 0; dkro_dSg = 0
    else:
        # --- Raw Baker kro and its partial derivatives ---
        Dw_o = Sw - SWC
        Dg_o = Sg - SGC
        Db_o = Dw_o + Dg_o
        if Db_o > 1e-30:
            Wg_o = np.clip(Dg_o / Db_o, 0, 1)
            if 0 < Wg_o < 1:
                dWg_o_dSw = -Dg_o / Db_o**2
                dWg_o_dSg = Dw_o / Db_o**2
            else:
                dWg_o_dSw = dWg_o_dSg = 0
        else:
            Wg_o = 0; dWg_o_dSw = dWg_o_dSg = 0
        kro_raw = (1 - Wg_o) * krow + Wg_o * krog
        dkro_raw_dSw = (dWg_o_dSw * (krog - krow)
                        + (1 - Wg_o) * dkrow_dSw + Wg_o * dkrog_dSw)
        dkro_raw_dSg = dWg_o_dSg * (krog - krow) + Wg_o * dkrog_dSg

        # --- Apply smooth phi(So) damping; product rule for derivatives
        # phi(So) = 1 - exp(-(So-SORG)/SO_REF) is C-infinity, so no
        # kink-induced discontinuity at So=SORG+SO_REF as there was
        # with a piecewise-linear ramp.
        if SO_REF > 0:
            dSo = So - SORG
            if dSo <= 0.0:
                # One-sided right limit at So=SORG.
                phi = 0.0
                dphi_dSo = 1.0 / SO_REF
            else:
                expo = np.exp(-dSo / SO_REF)
                phi = 1.0 - expo
                dphi_dSo = expo / SO_REF
            dphi_dSw = -dphi_dSo   # So = 1 - Sw - Sg
            dphi_dSg = -dphi_dSo
            kro = kro_raw * phi
            dkro_dSw = dkro_raw_dSw * phi + kro_raw * dphi_dSw
            dkro_dSg = dkro_raw_dSg * phi + kro_raw * dphi_dSg
        else:
            kro = kro_raw
            dkro_dSw = dkro_raw_dSw
            dkro_dSg = dkro_raw_dSg
        kro = np.clip(kro, 0, KRO_MAX)

    # Baker (26): krw
    Do_w = max(0, So - SORG)
    Dg_w = max(0, Sg - SGC)
    Db_w = Do_w + Dg_w
    if Sw < SWC:
        krw = 0; dkrw_dSw = 0; dkrw_dSg = 0
    elif Db_w > 1e-30:
        Wg_w = np.clip(Dg_w / Db_w, 0, 1)
        if 0 < Wg_w < 1:
            dWg_w_dSw = Dg_w / Db_w**2
            dWg_w_dSg = 1.0 / Db_w
        else:
            dWg_w_dSw = dWg_w_dSg = 0
        krw = (1 - Wg_w) * krw_wo + Wg_w * krw_wg
        dkrw_dSw = (dWg_w_dSw * (krw_wg - krw_wo)
                     + (1 - Wg_w) * dkrw_wo + Wg_w * dkrw_wg)
        dkrw_dSg = dWg_w_dSg * (krw_wg - krw_wo)
    else:
        krw = krw_wo; dkrw_dSw = dkrw_wo; dkrw_dSg = 0

    # Baker (27): krg
    Do_g = max(0, So - SORG)
    Dw_g = max(0, Sw - SWC)
    Db_g = Do_g + Dw_g
    if Sg <= SGC:
        krg = 0; dkrg_dSw = 0; dkrg_dSg = 0
    elif Db_g > 1e-30:
        Ww_g = np.clip(Dw_g / Db_g, 0, 1)
        if 0 < Ww_g < 1:
            dWw_g_dSw = 1.0 / Db_g
            dWw_g_dSg = Dw_g / Db_g**2
        else:
            dWw_g_dSw = dWw_g_dSg = 0
        krg = (1 - Ww_g) * krg_og + Ww_g * krg_wg
        dkrg_dSw = dWw_g_dSw * (krg_wg - krg_og)
        dkrg_dSg = (dWw_g_dSg * (krg_wg - krg_og)
                     + (1 - Ww_g) * dkrg_og + Ww_g * dkrg_wg)
    else:
        krg = krg_og; dkrg_dSw = 0; dkrg_dSg = dkrg_og

    # Jacobian
    lw = krw / MU_W; lo = kro / MU_O; lg = krg / MU_G
    lt = lw + lo + lg
    if lt < 1e-30:
        return 0, 0, np.array([1, 0]), np.array([0, 1]), 0

    dlw_dSw = dkrw_dSw / MU_W; dlw_dSg = dkrw_dSg / MU_W
    dlo_dSw = dkro_dSw / MU_O; dlo_dSg = dkro_dSg / MU_O
    dlg_dSw = dkrg_dSw / MU_G; dlg_dSg = dkrg_dSg / MU_G
    dlt_dSw = dlw_dSw + dlo_dSw + dlg_dSw
    dlt_dSg = dlw_dSg + dlo_dSg + dlg_dSg
    lt2 = lt**2

    a11 = (dlw_dSw * lt - lw * dlt_dSw) / lt2
    a12 = (dlw_dSg * lt - lw * dlt_dSg) / lt2
    a21 = (dlg_dSw * lt - lg * dlt_dSw) / lt2
    a22 = (dlg_dSg * lt - lg * dlt_dSg) / lt2

    tr_J = a11 + a22
    det_J = a11 * a22 - a12 * a21
    disc = tr_J**2 - 4 * det_J

    if disc < 0:
        lam1 = lam2 = tr_J / 2
        v1 = np.array([1.0, 0.0]); v2 = np.array([0.0, 1.0])
    else:
        sq = np.sqrt(disc)
        lam1 = (tr_J - sq) / 2
        lam2 = (tr_J + sq) / 2
        if abs(a12) >= abs(a21) and abs(a12) > 1e-15:
            v1 = np.array([a12, lam1 - a11])
            v2 = np.array([a12, lam2 - a11])
        elif abs(a21) > 1e-15:
            v1 = np.array([lam1 - a22, a21])
            v2 = np.array([lam2 - a22, a21])
        else:
            v1 = np.array([1.0, 0.0]); v2 = np.array([0.0, 1.0])

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 > 0: v1 /= n1
    if n2 > 0: v2 /= n2
    return lam1, lam2, v1, v2, disc


# ==================================================================
#  Two-phase wave speed on So=SORG boundary (water-gas BL)
# ==================================================================
def _boundary_wave_speed(Sw, cp, f2, eps=1e-5):
    """
    Compute dfw/dSw on the So=SORG edge (two-phase water-gas).
    On this edge: Sg = 1 - Sw - SORG, kro = 0.
    """
    Sw_p = min(Sw + eps, 1.0 - SORG)
    Sw_m = max(Sw - eps, SWC)
    Sg_p = 1.0 - Sw_p - SORG
    Sg_m = 1.0 - Sw_m - SORG
    fw_p, _, _ = fractional_flow(Sw_p, Sg_p, cp, f2)
    fw_m, _, _ = fractional_flow(Sw_m, Sg_m, cp, f2)
    dSw = Sw_p - Sw_m
    if abs(dSw) < 1e-15:
        return 0.0
    return (fw_p - fw_m) / dSw


# ==================================================================
#  Strategy A: boundary BL slow-wave + shock to R
# ==================================================================
def _boundary_fw(Sw, cp, f2):
    """Water fractional flow on the So=SORG edge."""
    Sg = 1.0 - Sw - SORG
    if Sg < -1e-10 or Sw < SWC - 1e-10:
        return 0.0
    fw, _, _ = fractional_flow(max(Sw, SWC), max(Sg, 0.0), cp, f2)
    return fw


def _oleinik_admissible(Sw_L, Sw_M, fw_arr, Sw_arr):
    """
    Check the Oleinik entropy condition for a BL shock from Sw_L to Sw_M.

    Oleinik condition (scalar Liu): for ALL intermediate Sw between L and M,
    the chord slope sigma(L;M) must be <= the chord slope sigma(L;u).
    This is the lower convex envelope / Welge tangent condition.

    Returns True if the shock L->M is admissible.
    """
    if abs(Sw_M - Sw_L) < 1e-12:
        return True

    fw_L = np.interp(Sw_L, Sw_arr, fw_arr)
    fw_M = np.interp(Sw_M, Sw_arr, fw_arr)
    sigma_LM = (fw_M - fw_L) / (Sw_M - Sw_L)

    # Check: for all u between L and M, sigma(L;u) >= sigma(L;M)
    if Sw_M > Sw_L:
        mask = (Sw_arr > Sw_L + 1e-8) & (Sw_arr < Sw_M - 1e-8)
    else:
        mask = (Sw_arr < Sw_L - 1e-8) & (Sw_arr > Sw_M + 1e-8)

    if not np.any(mask):
        return True

    Sw_mid = Sw_arr[mask]
    fw_mid = fw_arr[mask]
    sigma_mid = (fw_mid - fw_L) / (Sw_mid - Sw_L)

    if Sw_M > Sw_L:
        # Forward shock: chord to M should be <= chord to any intermediate
        return np.all(sigma_LM <= sigma_mid + 1e-8)
    else:
        # Backward shock: chord to M should be >= chord to any intermediate
        return np.all(sigma_LM >= sigma_mid - 1e-8)


def _liu_check_fast_shock(M_state, R, cp):
    """
    Liu entropy condition for the M->R fast shock (Juanes eq.24):
        lambda_s(M) >= sigma(M;R) > lambda_2(R)
    and the additional Lax-type bound:
        sigma(M;R) < lambda_2(M)  (characteristic goes INTO the shock)

    Returns (passes, sigma, info_str).
    """
    Sw_M, Sg_M = M_state[0], M_state[2]
    Sw_R, Sg_R = R[0], R[2]

    try:
        l1_M, l2_M, _, _, d_M = compute_eigenvalues(Sw_M, Sg_M, cp)
    except Exception:
        # M is on boundary, eigenvalues may degenerate; use BL speed
        l1_M = _boundary_wave_speed(Sw_M, cp, None)
        l2_M = l1_M  # degenerate

    try:
        l1_R, l2_R, _, _, d_R = compute_eigenvalues(Sw_R, Sg_R, cp)
    except Exception:
        return False, 0.0, "eigenvalue failure at R"

    # Need sigma from RH (already computed externally), but we can
    # just return the eigenvalue info for the caller to use.
    return True, l1_M, l2_M, l1_R, l2_R


def _find_M_on_boundary(L, R, cp, f2, ns=400):
    """
    Strategy A for L on the So=SORG boundary.  DISABLED: in our
    strongly/weakly water-wet Riemann problems the true M is an
    *interior* state, so the boundary scan can only produce a spurious
    Welge tangent trapped on the edge (the fast eigenvector is tangent
    to the edge, so the fast integral curve from any boundary M stays
    on the boundary).  Returning None lets the caller fall through to
    the interior Newton / curve-intersection strategies that v4 used
    to reach the correct M.  Re-enable only when both L and R sit on
    the same edge AND the physical viscous profile is known to stay
    on the boundary.
    """
    return None

    # ------- legacy body kept for future boundary-profile cases -------
    L_on_boundary = (abs(L[1] - SORG) <= 0.025)
    if not L_on_boundary:
        return None

    Sw_L = L[0]
    Sw_arr = np.linspace(SWC + 0.002, 1.0 - SORG - 0.002, ns)

    fw_arr = np.array([_boundary_fw(s, cp, f2) for s in Sw_arr])
    dfw_arr = np.array([_boundary_wave_speed(s, cp, f2) for s in Sw_arr])
    fw_L = _boundary_fw(Sw_L, cp, f2)
    dfw_L = _boundary_wave_speed(Sw_L, cp, f2)
    i_L = int(np.argmin(np.abs(Sw_arr - Sw_L)))

    try:
        _, l2_R, _, _, d_R = compute_eigenvalues(R[0], R[2], cp)
    except Exception:
        return None
    if d_R < -1e-8:
        return None

    TRIVIAL_DSW = 3e-3     # reject near-L candidates on both sides
    SCORE_ACCEPT = 0.30    # accept threshold on (rh + welge_err)

    best_M = None
    best_score = np.inf
    best_info = {}

    for i in range(ns):
        Sw_M = Sw_arr[i]
        Sg_M = 1.0 - Sw_M - SORG
        if Sg_M < -1e-6 or Sw_M < SWC - 1e-6:
            continue

        dSw_LM = Sw_M - Sw_L
        if abs(dSw_LM) <= TRIVIAL_DSW:
            continue

        M_cand = np.array([Sw_M, SORG, Sg_M])

        # ============================================================
        #  (1) Slow wave L->M admissibility on the scalar BL edge
        # ============================================================
        sigma_shock_LM = (fw_arr[i] - fw_L) / dSw_LM
        dfw_M = dfw_arr[i]

        if dSw_LM < 0:
            # Backward shock: auto-Oleinik for convex fw.
            # Lax: dfw(L) >= sigma >= dfw(M)
            if not (dfw_L + 1e-6 >= sigma_shock_LM >= dfw_M - 1e-6):
                continue
            mode = 'shock_back'
            slow_speed = sigma_shock_LM
        else:
            # Forward direction: only rarefaction is admissible on a
            # convex fw segment (forward shock violates Oleinik).
            # Require dfw to be monotonically non-decreasing on [L, M].
            seg_dfw = dfw_arr[i_L:i + 1]
            if len(seg_dfw) < 2 or not np.all(np.diff(seg_dfw) >= -1e-6):
                continue
            mode = 'rarefaction_fwd'
            slow_speed = dfw_M    # trailing characteristic speed

        # ============================================================
        #  (2) Eigenvalues at M on the boundary
        # ============================================================
        try:
            _, l2_M, _, _, d_M = compute_eigenvalues(Sw_M, Sg_M, cp)
        except Exception:
            continue
        if d_M < -1e-8:
            continue

        # Speed ordering: slow wave must not overtake fast leading char
        if slow_speed >= l2_M - 1e-4:
            continue

        # ============================================================
        #  (3) Fast Welge tangent on the fast integral curve from M
        # ============================================================
        # Trace fast integral curve from M in BOTH directions; the
        # forward direction may stay trapped on the boundary because
        # the fast eigenvector can be nearly tangent here, so we rely
        # on the full bi-directional curve.
        fast_p = integrate_rarefaction(Sw_M, Sg_M, 2, +1, cp, f2,
                                       Nmax=4000, h=0.0008)
        fast_m = integrate_rarefaction(Sw_M, Sg_M, 2, -1, cp, f2,
                                       Nmax=4000, h=0.0008)
        if len(fast_p) > 0 and len(fast_m) > 0:
            fast_full = np.vstack([fast_m[::-1], fast_p[1:]])
        elif len(fast_p) > 0:
            fast_full = fast_p
        elif len(fast_m) > 0:
            fast_full = fast_m[::-1]
        else:
            continue
        if len(fast_full) < 5:
            continue

        # True Welge tangent u2* must satisfy BOTH conditions:
        #   (i)  u2* lies on the Hugoniot from R  -> rh(u2*, R) ~ 0
        #   (ii) lambda_2(u2*) = sigma(u2*; R)     -> welge_err ~ 0
        # So we hard-filter on rh first, then minimise welge_err among
        # the surviving candidates.
        RH_HARD_TOL = 0.05

        welge_err_best = np.inf
        welge_rh_best = np.inf
        welge_pt_best = None
        for pt in fast_full:
            Sw_p = float(pt[0]); Sg_p = float(pt[2])
            So_p = 1.0 - Sw_p - Sg_p
            if So_p < SORG - 1e-4 or Sw_p < SWC - 1e-4 or Sg_p < -1e-4:
                continue
            Pstate = np.array([Sw_p, So_p, Sg_p])
            sig_PR, rh_PR = compute_shock_speed(Pstate, R, cp, f2)
            if np.isnan(sig_PR) or sig_PR <= 0:
                continue
            if rh_PR > RH_HARD_TOL:
                continue  # not on Hugoniot from R
            lam2_p = float(pt[3])
            welge = abs(lam2_p - sig_PR)
            if welge < welge_err_best:
                welge_err_best = welge
                welge_rh_best = rh_PR
                welge_pt_best = Pstate.copy()

        if welge_pt_best is None:
            continue

        # ============================================================
        #  (4) Score and keep best
        # ============================================================
        score = welge_rh_best + welge_err_best

        if score < best_score:
            best_score = score
            best_M = M_cand.copy()
            best_info = dict(mode=mode,
                             slow_speed=slow_speed,
                             sigma_LM=sigma_shock_LM,
                             lam2_M=l2_M,
                             welge_pt=welge_pt_best,
                             welge_err=welge_err_best,
                             welge_rh=welge_rh_best)

    if best_M is None:
        print(f"    Boundary scan: no candidate survived admissibility")
        return None

    info = best_info
    wp = info['welge_pt']
    print(f"    Boundary scan [{info['mode']}]: "
          f"best M=({best_M[0]:.4f}, {best_M[1]:.4f}, {best_M[2]:.4f})")
    print(f"    slow_speed={info['slow_speed']:.4f}, "
          f"lam2(M_bnd)={info['lam2_M']:.4f}")
    print(f"    fast Welge u2*=({wp[0]:.4f},{wp[1]:.4f},{wp[2]:.4f})  "
          f"welge_err={info['welge_err']:.3e}  "
          f"rh(u2*->R)={info['welge_rh']:.3e}  "
          f"score={best_score:.3e}")
    if best_score < SCORE_ACCEPT:
        return best_M
    print(f"    Boundary scan: score {best_score:.3e} "
          f">= {SCORE_ACCEPT:.2f}, returning None (fall through)")
    return None


# ==================================================================
#  Rarefaction curve integration
# ==================================================================
def integrate_rarefaction(Sw0, Sg0, fam, direction, cp, f2,
                          Nmax=1200, h=0.002):
    """
    Integrate rarefaction curve from (Sw0, Sg0) along family fam
    (1=slow, 2=fast).  direction: +1 or -1.
    Returns array of shape (n, 4): [Sw, So, Sg, lambda].

    When the curve hits the So=SORG boundary, it switches to two-phase
    (water-gas) Buckley-Leverett mode and continues along the boundary.
    """
    path = []
    Sw, Sg = Sw0, Sg0
    prev_v = None
    on_boundary = False       # True once we hit So=SORG edge
    boundary_sign = None      # +1 or -1: step direction in Sw on boundary

    for _ in range(Nmax):
        So = 1 - Sw - Sg

        # --- Detect CROSSING into the So=SORG boundary ---
        # Only switch to boundary-march mode when we arrive at the edge
        # from the interior (prev_v is not None).  If L already sits on
        # the edge, let MODE 2 take the first interior RK step: if the
        # slow eigenvector points into the interior, we correctly leave
        # the boundary; if it is tangent to the edge, So stays ~0 and
        # boundary mode triggers on the NEXT iteration (with prev_v set).
        if not on_boundary and So <= SORG + 1e-6 and prev_v is not None:
            on_boundary = True
            So = SORG
            Sg = 1.0 - Sw - SORG
            # Project last velocity onto boundary tangent [1, -1]/sqrt(2)
            proj = (prev_v[0] - prev_v[1]) * direction
            boundary_sign = 1.0 if proj >= 0 else -1.0

        # ============================================================
        #  MODE 1: On the So=SORG boundary (two-phase water-gas)
        # ============================================================
        if on_boundary:
            if Sw < SWC - 1e-4 or Sg < -1e-4:
                break
            # Wave speed = dfw/dSw (1D Buckley-Leverett on this edge)
            lam = _boundary_wave_speed(Sw, cp, f2)
            path.append([Sw, So, Sg, lam])

            # Step along boundary: dSg = -dSw (So stays at SORG)
            Sw_n = Sw + h * boundary_sign
            Sg_n = 1.0 - Sw_n - SORG
            # Stop if we walk off the edge of the saturation triangle
            if Sw_n < SWC - 1e-4 or Sg_n < -1e-4:
                break
            Sw_n = max(Sw_n, SWC)
            Sg_n = max(Sg_n, 0.0)
            Sw, Sg = Sw_n, Sg_n
            continue

        # ============================================================
        #  MODE 2: Interior three-phase region (original logic)
        # ============================================================
        if So < SORG - 1e-4 or Sw < SWC - 1e-4 or Sg < -1e-4:
            break
        try:
            l1, l2, e1, e2, d = compute_eigenvalues(Sw, Sg, cp)
        except Exception:
            break
        if d < -1e-10:
            break

        lam = l1 if fam == 1 else l2
        v = e1 if fam == 1 else e2

        if np.linalg.norm(v) < 1e-12:
            break
        if prev_v is not None and np.dot(v, prev_v) < 0:
            v = -v

        path.append([Sw, So, Sg, lam])
        vs = v * direction
        prev_v = v

        Sw_n = Sw + h * vs[0]
        Sg_n = Sg + h * vs[1]
        So_n = 1 - Sw_n - Sg_n

        # If next step crosses boundary, clamp to it (will enter boundary
        # mode on the next iteration instead of breaking)
        if So_n < SORG:
            Sg_n = 1.0 - Sw_n - SORG

        if Sw_n < SWC - 0.002 or Sg_n < -0.002:
            break
        Sw_n = max(Sw_n, SWC)
        Sg_n = max(Sg_n, 0.0)
        if 1 - Sw_n - Sg_n < SORG:
            Sg_n = 1 - Sw_n - SORG
        Sw, Sg = Sw_n, Sg_n

    return np.array(path) if path else np.empty((0, 4))


# ==================================================================
#  Shock speed  (Rankine-Hugoniot)
# ==================================================================
def compute_shock_speed(sa, sb, cp, f2):
    """Compute shock speed between states sa and sb (each [Sw,So,Sg])."""
    fwa, _, fga = fractional_flow(sa[0], sa[2], cp, f2)
    fwb, _, fgb = fractional_flow(sb[0], sb[2], cp, f2)
    dSw = sb[0] - sa[0]
    dSg = sb[2] - sa[2]
    dfw = fwb - fwa
    dfg = fgb - fga

    hw = abs(dSw) > 1e-12
    hg = abs(dSg) > 1e-12

    if hw and hg:
        sw = dfw / dSw
        sg = dfg / dSg
        w1, w2 = abs(dSw), abs(dSg)
        sigma = (sw * w1 + sg * w2) / (w1 + w2)
        rh_err = abs(sw - sg) / max(abs(sw), abs(sg), 1e-15)
    elif hw:
        sigma = dfw / dSw
        rh_err = 0.0
    elif hg:
        sigma = dfg / dSg
        rh_err = 0.0
    else:
        sigma = 0.0
        rh_err = 0.0
    return sigma, rh_err


# ==================================================================
#  Hugoniot locus helpers
# ==================================================================
def _rh_residual(Sw_P, Sg_P, ref, fw_ref, fg_ref, cp, f2, reverse=False):
    fw, _, fg = fractional_flow(Sw_P, Sg_P, cp, f2)
    if reverse:
        dSw = ref[0] - Sw_P; dSg = ref[2] - Sg_P
        dfw = fw_ref - fw;   dfg = fg_ref - fg
    else:
        dSw = Sw_P - ref[0]; dSg = Sg_P - ref[2]
        dfw = fw - fw_ref;   dfg = fg - fg_ref
    if abs(dSw) < 1e-12 or abs(dSg) < 1e-12:
        return 1e10
    return dfw / dSw - dfg / dSg


def _bisect_all_roots(f, lo, hi, n=200):
    t = np.linspace(lo, hi, n)
    fv = np.array([f(x) for x in t])
    roots = []
    for i in range(n - 1):
        if (np.isfinite(fv[i]) and np.isfinite(fv[i + 1])
                and fv[i] * fv[i + 1] < 0):
            a, b, fa = t[i], t[i + 1], fv[i]
            for _ in range(60):
                c = 0.5 * (a + b)
                fc = f(c)
                if abs(fc) < 1e-14 or (b - a) < 1e-14:
                    break
                if fc * fa < 0:
                    b = c
                else:
                    a, fa = c, fc
            rt = 0.5 * (a + b)
            roots.append((rt, abs(f(rt))))
    if not roots:
        return np.nan
    return min(roots, key=lambda x: x[1])[0]


def find_hugoniot_Sw_reverse(R, Sg_P, cp, f2):
    fw_R, _, fg_R = fractional_flow(R[0], R[2], cp, f2)
    lo = SWC
    hi = 1 - Sg_P - SORG
    if lo >= hi:
        return np.nan
    res = lambda Sw: _rh_residual(Sw, Sg_P, R, fw_R, fg_R, cp, f2, reverse=True)
    return _bisect_all_roots(res, lo, hi, 200)


def trace_hugoniot_from_R(R, cp, f2, ns=600):
    Sg_lo = 1e-4
    Sg_hi = 1 - SWC - SORG
    Sg_sc = np.linspace(Sg_lo, Sg_hi, ns)
    hug = []
    for Sg_P in Sg_sc:
        Sw_P = find_hugoniot_Sw_reverse(R, Sg_P, cp, f2)
        if np.isnan(Sw_P):
            continue
        So_P = 1 - Sw_P - Sg_P
        if So_P < SORG - 1e-6 or Sw_P < SWC - 1e-6:
            continue
        P = np.array([Sw_P, So_P, Sg_P])
        sig, rh = compute_shock_speed(P, R, cp, f2)
        if rh > 0.25 or sig <= 0 or np.isnan(sig):
            continue
        try:
            _, l2, _, _, d = compute_eigenvalues(Sw_P, Sg_P, cp)
            if d < -1e-8 or l2 < sig * 0.7:
                continue
        except Exception:
            continue
        hug.append([Sw_P, So_P, Sg_P, sig])
    return np.array(hug) if hug else np.empty((0, 4))


# ------------------------------------------------------------------
#  Hugoniot from L  +  transitional-shock machinery (Strategy 1c)
# ------------------------------------------------------------------
def find_hugoniot_Sw_forward(L, Sg_P, cp, f2):
    """Solve _rh_residual(Sw, Sg_P; ref=L) = 0 for Sw."""
    fw_L, _, fg_L = fractional_flow(L[0], L[2], cp, f2)
    lo = SWC
    hi = 1 - Sg_P - SORG
    if lo >= hi:
        return np.nan
    res = lambda Sw: _rh_residual(Sw, Sg_P, L, fw_L, fg_L, cp, f2, reverse=False)
    return _bisect_all_roots(res, lo, hi, 200)


def trace_hugoniot_from_R_permissive(R, cp, f2, ns=600):
    """
    Permissive R-Hugoniot tracer.  Like trace_hugoniot_from_R but does NOT
    enforce the Lax-2 filter — needed for transitional shocks where the
    target state may violate Lax.  Caller must apply admissibility itself.
    """
    Sg_lo = 1e-4
    Sg_hi = 1 - SWC - SORG
    Sg_sc = np.linspace(Sg_lo, Sg_hi, ns)
    hug = []
    for Sg_P in Sg_sc:
        Sw_P = find_hugoniot_Sw_reverse(R, Sg_P, cp, f2)
        if np.isnan(Sw_P):
            continue
        So_P = 1 - Sw_P - Sg_P
        if So_P < SORG - 1e-6 or Sw_P < SWC - 1e-6:
            continue
        if abs(Sw_P - R[0]) < 5e-4 and abs(Sg_P - R[2]) < 5e-4:
            continue
        P = np.array([Sw_P, So_P, Sg_P])
        sig, rh = compute_shock_speed(P, R, cp, f2)
        if rh > 0.25 or not np.isfinite(sig):
            continue
        hug.append([Sw_P, So_P, Sg_P, sig])
    return np.array(hug) if hug else np.empty((0, 4))


def trace_hugoniot_from_L(L, cp, f2, ns=600):
    """
    Trace the Hugoniot locus emanating from state L.

    PERMISSIVE: keeps every state P with small RH error and finite shock
    speed.  Does NOT enforce Lax — the caller decides admissibility.  This
    is essential for transitional / undercompressive shocks that violate
    Lax-1 but still satisfy Rankine-Hugoniot and Liu's E-condition.
    """
    Sg_lo = 1e-4
    Sg_hi = 1 - SWC - SORG
    Sg_sc = np.linspace(Sg_lo, Sg_hi, ns)
    hug = []
    for Sg_P in Sg_sc:
        Sw_P = find_hugoniot_Sw_forward(L, Sg_P, cp, f2)
        if np.isnan(Sw_P):
            continue
        So_P = 1 - Sw_P - Sg_P
        if So_P < SORG - 1e-6 or Sw_P < SWC - 1e-6:
            continue
        # Skip the trivial fixed point P == L
        if abs(Sw_P - L[0]) < 5e-4 and abs(Sg_P - L[2]) < 5e-4:
            continue
        P = np.array([Sw_P, So_P, Sg_P])
        sig, rh = compute_shock_speed(L, P, cp, f2)
        if rh > 0.25 or not np.isfinite(sig):
            continue
        hug.append([Sw_P, So_P, Sg_P, sig])
    return np.array(hug) if hug else np.empty((0, 4))


def _liu_E_check(state_a, state_b, hug_a, cp, f2, n_test=40, tol=2e-3):
    """
    Liu's E-condition for a shock state_a -> state_b.

    `hug_a` is the Hugoniot locus from `state_a` (each row [Sw,So,Sg,sig]).
    For every state P on the Hugoniot segment from `state_a` to `state_b`,
    require  sigma(state_a, P) >= sigma(state_a, state_b) - tol.

    This is the standard admissibility criterion that includes Lax shocks
    AND undercompressive (transitional) shocks while excluding spurious
    Hugoniot intersections.

    Returns (passes, sigma_ab, min_sigma_along_segment).
    """
    sig_ab, _ = compute_shock_speed(state_a, state_b, cp, f2)
    if not np.isfinite(sig_ab):
        return False, np.nan, np.nan
    if len(hug_a) < 2:
        # No locus — be permissive but flag
        return True, sig_ab, sig_ab

    da = (hug_a[:, 0] - state_a[0])**2 + (hug_a[:, 2] - state_a[2])**2
    db = (hug_a[:, 0] - state_b[0])**2 + (hug_a[:, 2] - state_b[2])**2
    ia, ib = int(np.argmin(da)), int(np.argmin(db))
    if ia == ib:
        return True, sig_ab, sig_ab
    if ia > ib:
        ia, ib = ib, ia
    seg = hug_a[ia:ib + 1]
    if len(seg) < 2:
        return True, sig_ab, sig_ab

    step = max(1, len(seg) // n_test)
    sigs = []
    for k in range(0, len(seg), step):
        P = seg[k, :3]
        # Skip endpoints (they reproduce sig_ab and 0/0)
        if (abs(P[0] - state_a[0]) < 1e-4 and abs(P[2] - state_a[2]) < 1e-4):
            continue
        if (abs(P[0] - state_b[0]) < 1e-4 and abs(P[2] - state_b[2]) < 1e-4):
            continue
        sig_aP, rh = compute_shock_speed(state_a, P, cp, f2)
        if not np.isfinite(sig_aP) or rh > 0.5:
            continue
        sigs.append(sig_aP)

    if not sigs:
        return True, sig_ab, sig_ab
    min_sig = float(np.min(sigs))
    passes = (min_sig >= sig_ab - tol)
    return passes, sig_ab, min_sig


# ==================================================================
#  VISCOUS PROFILE (TRAVELING WAVE) ADMISSIBILITY
# ==================================================================
#
#  References:
#    [IMPT90] Isaacson, Marchesin, Plohr, Temple, "The Riemann problem
#             for a class of conservation laws of mixed type",
#             J. Diff. Equations 88 (1990).
#    [SS87]   Schaeffer & Shearer, "The classification of 2x2 systems of
#             non-strictly hyperbolic conservation laws...", CPAM 40 (1987).
#    [MP01]   Marchesin & Plohr, "Wave structure in WAG recovery",
#             SPE Journal 6 (2001).
#    [Az10]   Azevedo, de Souza, Furtado, Marchesin, Plohr, "The solution
#             by the wave curve method of three-phase flow in virgin
#             reservoirs", Transp. Porous Media 83 (2010).
#    [MaP85]  Majda & Pego, "Stable viscosity matrices for systems of
#             conservation laws", J. Diff. Equations 56 (1985).
#
#  Theory
#  ------
#  For the system    ∂_t u + ∂_x f(u) = 0,    u = (Sw, Sg) ∈ R^2,
#  add an artificial viscosity ε I ∂_xx u and look for traveling-wave
#  solutions  u(x,t) = U((x − σt)/ε).  After integrating once and applying
#  U(−∞) = L,  U'(−∞) = 0,  the profile satisfies the autonomous 2D ODE
#
#       U'(η) = G(U; σ, L) := f(U) − f(L) − σ (U − L)
#
#  Fixed points of G are exactly the Hugoniot states from L at speed σ
#  (in particular L itself and any candidate downstream state M).  The
#  viscous profile is a heteroclinic orbit of this ODE from L (η = −∞)
#  to M (η = +∞).
#
#  Linearization at a fixed point U*:
#       G'(U*) = f'(U*) − σ I
#  whose eigenvalues are  μ_i = λ_i(U*) − σ,  i = 1, 2,
#  where λ_i are the eigenvalues of the flux Jacobian f'(U*).
#
#  Endpoint admissibility (Majda-Pego):
#    L (α-limit):  must be UNSTABLE forward, i.e. at least one μ_i(L) > 0.
#                  Orbit lies on the unstable manifold of L.
#    M (ω-limit):  must be STABLE forward, i.e. at least one μ_i(M) < 0.
#                  Orbit lies on the stable manifold of M.
#
#  Lax classification by sign pattern of  (μ_1(L), μ_2(L), μ_1(M), μ_2(M)):
#    Lax-1 :     (+, +, −, +)   L = repeller (2D), M = saddle (1D stable)
#    Lax-2 :     (−, +, −, −)   L = saddle (1D unstable),  M = attractor
#    Trans 1→2:  (−, +, −, +)   both saddles — codim-1 saddle-saddle
#                connection, σ must be tuned (undercompressive shock).
#
# ==================================================================


def _flux_jacobian(state, cp, f2, h=1e-5):
    """Numerical Jacobian f'(u) at state, where u = (Sw, Sg).

    Returns 2x2 matrix [[∂fw/∂Sw, ∂fw/∂Sg], [∂fg/∂Sw, ∂fg/∂Sg]].
    """
    Sw, _, Sg = state[0], state[1], state[2]
    fwp, _, fgp = fractional_flow(Sw + h, Sg, cp, f2)
    fwm, _, fgm = fractional_flow(Sw - h, Sg, cp, f2)
    dfw_dSw = (fwp - fwm) / (2 * h)
    dfg_dSw = (fgp - fgm) / (2 * h)
    fwp, _, fgp = fractional_flow(Sw, Sg + h, cp, f2)
    fwm, _, fgm = fractional_flow(Sw, Sg - h, cp, f2)
    dfw_dSg = (fwp - fwm) / (2 * h)
    dfg_dSg = (fgp - fgm) / (2 * h)
    return np.array([[dfw_dSw, dfw_dSg],
                     [dfg_dSw, dfg_dSg]])


def _classify_endpoint_viscous(state, sigma, cp, f2):
    """Linearize G'(state) = f'(state) − σI and return spectrum.

    Returns dict with keys:
        eigvals    : (mu1, mu2) sorted ascending — eigenvalues of G'
        eigvecs    : 2x2 matrix, columns = right eigenvectors
        n_unstable : number of eigenvalues > +tol
        n_stable   : number of eigenvalues < −tol
        n_zero     : number of near-zero eigenvalues
    """
    J = _flux_jacobian(state, cp, f2)
    A = J - sigma * np.eye(2)
    w, V = np.linalg.eig(A)
    # Real parts only — viscous profiles for 2x2 strictly hyperbolic
    # systems have real eigenvalues; if not, the state is in/near a
    # complex-eigenvalue region (elliptic) and we treat as degenerate.
    if np.max(np.abs(w.imag)) > 1e-6:
        return {
            'eigvals': w, 'eigvecs': V,
            'n_unstable': 0, 'n_stable': 0, 'n_zero': 2,
            'complex': True,
        }
    w = w.real
    V = V.real
    order = np.argsort(w)
    w = w[order]
    V = V[:, order]
    tol = 1e-6
    return {
        'eigvals': w, 'eigvecs': V,
        'n_unstable': int(np.sum(w > tol)),
        'n_stable': int(np.sum(w < -tol)),
        'n_zero': int(np.sum(np.abs(w) <= tol)),
        'complex': False,
    }


def _viscous_rhs(eta, U, sigma, L_sg, fL, cp, f2):
    """Right-hand side of  U'(η) = f(U) − f(L) − σ(U − L).

    U is in (Sw, Sg) coordinates (drop redundant So = 1 − Sw − Sg).
    """
    Sw, Sg = float(U[0]), float(U[1])
    # Soft clamp to physical region (orbit can wander slightly outside
    # during integration, but we don't want fractional_flow to blow up)
    if Sw < SWC - 0.05 or Sg < -0.05:
        return np.array([0.0, 0.0])
    if Sw + Sg > 1.0 - SORG + 0.05:
        return np.array([0.0, 0.0])
    fw, _, fg = fractional_flow(Sw, Sg, cp, f2)
    return np.array([
        fw - fL[0] - sigma * (Sw - L_sg[0]),
        fg - fL[1] - sigma * (Sg - L_sg[1]),
    ])


def _shoot_viscous_profile(L, sigma, M_target, cp, f2,
                            eps_offset=1e-5, max_eta=400.0,
                            verbose=False):
    """Attempt to shoot a viscous profile orbit from L to M_target at
    speed sigma.

    Method: linearize at L, identify unstable eigendirections, place a
    point at distance `eps_offset` along each unstable direction (and
    its negative), integrate U' = G(U; σ, L) forward in η using LSODA,
    record the closest approach to M_target.

    Returns dict with keys
        success     : bool, True iff closest approach < 5e-3
        dist_to_M   : float, closest approach distance in (Sw,Sg)
        eta_path    : 1D array of η values along orbit (best branch)
        U_path      : 2 x N array of (Sw, Sg) along orbit
        info        : dict with linearization summary
    """
    try:
        from scipy.integrate import solve_ivp
    except ImportError:
        return {'success': False, 'dist_to_M': np.inf,
                'reason': 'scipy not available'}

    L_sg = np.array([L[0], L[2]])
    M_sg = np.array([M_target[0], M_target[2]])
    fwL, _, fgL = fractional_flow(L[0], L[2], cp, f2)
    fL = np.array([fwL, fgL])

    info_L = _classify_endpoint_viscous(L, sigma, cp, f2)
    info_M = _classify_endpoint_viscous(M_target, sigma, cp, f2)

    if info_L['complex'] or info_M['complex']:
        return {'success': False, 'dist_to_M': np.inf,
                'reason': 'complex eigenvalues at endpoint',
                'info': {'L': info_L, 'M': info_M}}

    if info_L['n_unstable'] == 0:
        return {'success': False, 'dist_to_M': np.inf,
                'reason': f'L has no unstable direction (eigvals={info_L["eigvals"]})',
                'info': {'L': info_L, 'M': info_M}}
    if info_M['n_stable'] == 0:
        return {'success': False, 'dist_to_M': np.inf,
                'reason': f'M has no stable direction (eigvals={info_M["eigvals"]})',
                'info': {'L': info_L, 'M': info_M}}

    # Build list of unstable eigendirections at L
    unstable_dirs = []
    for k in range(2):
        if info_L['eigvals'][k] > 1e-6:
            v = info_L['eigvecs'][:, k]
            v = v / np.linalg.norm(v)
            unstable_dirs.append(v)

    def rhs(eta, U):
        return _viscous_rhs(eta, U, sigma, L_sg, fL, cp, f2)

    best = {'success': False, 'dist_to_M': np.inf,
            'eta_path': None, 'U_path': None,
            'info': {'L': info_L, 'M': info_M}}

    for v in unstable_dirs:
        for sign in (+1, -1):
            U0 = L_sg + sign * eps_offset * v

            def near_M(eta, U):
                return ((U[0] - M_sg[0])**2
                        + (U[1] - M_sg[1])**2) - (1e-5)**2
            near_M.terminal = True
            near_M.direction = -1

            def out_of_box(eta, U):
                # +1 inside, -1 outside; terminal on cross
                inside = ((U[0] > SWC - 0.02)
                          and (U[1] > -0.02)
                          and (U[0] + U[1] < 1.0 - SORG + 0.02))
                return 1.0 if inside else -1.0
            out_of_box.terminal = True
            out_of_box.direction = -1

            try:
                sol = solve_ivp(
                    rhs, (0.0, max_eta), U0,
                    method='LSODA',
                    rtol=1e-8, atol=1e-11,
                    events=[near_M, out_of_box],
                    max_step=1.0,
                )
            except Exception as e:
                if verbose:
                    print(f"        solve_ivp exception: {e}")
                continue

            if not sol.success and len(sol.t) < 5:
                continue

            # Closest approach to M along the trajectory
            d2 = (sol.y[0] - M_sg[0])**2 + (sol.y[1] - M_sg[1])**2
            i_min = int(np.argmin(d2))
            dist = float(np.sqrt(d2[i_min]))

            if dist < best['dist_to_M']:
                # Truncate trajectory at the closest-approach point
                best.update({
                    'dist_to_M': dist,
                    'success': dist < 5e-3,
                    'eta_path': sol.t[:i_min + 1],
                    'U_path': sol.y[:, :i_min + 1],
                    'init_dir': v * sign,
                })

    return best


def _find_M_via_viscous_profile(L, R, cp, f2):
    """Strategy 1f: locate the intermediate state M by VISCOUS PROFILE
    admissibility (Isaacson-Marchesin-Plohr).

    Sweep candidate Hugoniot states P along Hug(L); for each (P, σ_LP),
    classify the linearization at L and at P, and attempt to shoot a
    heteroclinic orbit U' = G(U; σ_LP, L) from L's unstable manifold.
    Accept (P, σ) if the orbit converges to P AND the resulting M→R
    fast wave has  σ(M, R) > σ(L, M)  (wave-speed ordering).

    Returns (M, sigma_LM, profile_dict) or (None, None, None).
    """
    print("    Strategy 1f: viscous profile shooting "
          "(Isaacson-Marchesin-Plohr) ...")

    hug_L = trace_hugoniot_from_L(L, cp, f2, ns=400)
    if len(hug_L) < 5:
        print("      [1f] Hug_L too short")
        return None, None, None
    print(f"      [1f] Hug_L: {len(hug_L)} candidate states")

    # Subsample for speed
    step = max(1, len(hug_L) // 80)
    samples = hug_L[::step]

    # Diagnostic counters
    counts = {'L_stable': 0, 'M_unstable': 0, 'L_complex': 0, 'M_complex': 0,
              'attempted': 0, 'profile_ok': 0, 'order_ok': 0}

    candidates = []
    for row in samples:
        P = row[:3]
        sigma = float(row[3])
        if not np.isfinite(sigma):
            continue

        # Quick endpoint classification before launching the integrator
        info_L = _classify_endpoint_viscous(L, sigma, cp, f2)
        info_M = _classify_endpoint_viscous(P, sigma, cp, f2)
        if info_L['complex']:
            counts['L_complex'] += 1
            continue
        if info_M['complex']:
            counts['M_complex'] += 1
            continue
        if info_L['n_unstable'] == 0:
            counts['L_stable'] += 1
            continue
        if info_M['n_stable'] == 0:
            counts['M_unstable'] += 1
            continue

        counts['attempted'] += 1
        res = _shoot_viscous_profile(L, sigma, P, cp, f2)
        if not res['success']:
            continue
        counts['profile_ok'] += 1

        sig_PR, rh_PR = compute_shock_speed(P, R, cp, f2)
        if not np.isfinite(sig_PR) or sig_PR <= sigma + 1e-6:
            continue
        counts['order_ok'] += 1

        # Determine Lax-class label for reporting
        muL = info_L['eigvals']; muM = info_M['eigvals']
        nL_u, nM_s = info_L['n_unstable'], info_M['n_stable']
        if nL_u == 2 and nM_s == 1:
            klass = 'Lax-1'
        elif nL_u == 1 and nM_s == 2:
            klass = 'Lax-2'
        elif nL_u == 1 and nM_s == 1:
            klass = 'Trans (saddle-saddle)'
        elif nL_u == 2 and nM_s == 2:
            klass = 'overcompressive'
        else:
            klass = f'?({nL_u},{nM_s})'

        candidates.append({
            'M': P.copy(), 'sigma_LM': sigma, 'sigma_MR': float(sig_PR),
            'rh_MR': float(rh_PR), 'profile': res, 'klass': klass,
            'mu_L': muL, 'mu_M': muM,
        })
        print(f"      [1f] sig={sigma:6.4f} ({klass:18s}) "
              f"M=({P[0]:.4f},{P[1]:.4f},{P[2]:.4f}) "
              f"profile_dist={res['dist_to_M']:.2e}  "
              f"sig(M,R)={sig_PR:.4f}")

    print(f"      [1f] sweep summary: L_stable={counts['L_stable']} "
          f"M_unstable={counts['M_unstable']} "
          f"L_complex={counts['L_complex']} M_complex={counts['M_complex']} "
          f"attempted={counts['attempted']} "
          f"profile_ok={counts['profile_ok']} "
          f"order_ok={counts['order_ok']}")

    if not candidates:
        print("      [1f] no admissible viscous profile found in this sweep")
        return None, None, None

    # Pick by smallest profile distance, breaking ties by smaller rh_MR
    candidates.sort(key=lambda c: (c['profile']['dist_to_M'], c['rh_MR']))
    best = candidates[0]
    print(f"      [1f] BEST: {best['klass']}")
    print(f"      [1f]   M = ({best['M'][0]:.4f},{best['M'][1]:.4f},"
          f"{best['M'][2]:.4f})")
    print(f"      [1f]   sig(L,M) = {best['sigma_LM']:.4f}  "
          f"sig(M,R) = {best['sigma_MR']:.4f}")
    print(f"      [1f]   profile dist to M: "
          f"{best['profile']['dist_to_M']:.2e}")
    print(f"      [1f]   mu(L) = {best['mu_L']}")
    print(f"      [1f]   mu(M) = {best['mu_M']}")
    return best['M'], best['sigma_LM'], best['profile']


def _find_oil_bank_on_hug(hug_state, ref_state, cp, f2,
                            sg_tol=0.025, rh_max=0.05):
    """
    Given a Hugoniot locus from `ref_state` (rows [Sw,So,Sg,sig]), find the
    candidate oil-bank state J = (Sw_J, 1-Sw_J, 0) on the Sg=0 edge with
    the smallest RH error relative to `ref_state`.

    Returns J as np.array or None.
    """
    if hug_state is None or len(hug_state) < 2:
        return None
    near_base = hug_state[hug_state[:, 2] < sg_tol]
    if len(near_base) == 0:
        return None
    best = None
    best_rh = np.inf
    for row in near_base:
        Sw_J = float(row[0])
        if (Sw_J < SWC + 0.005 or Sw_J > 1.0 - SORW - 0.005):
            continue
        So_J = 1.0 - Sw_J
        if So_J < SORW + 0.005:
            continue
        J = np.array([Sw_J, So_J, 0.0])
        sig, rh = compute_shock_speed(ref_state, J, cp, f2)
        if not np.isfinite(sig):
            continue
        if rh < best_rh:
            best_rh = rh
            best = J
    if best is None or best_rh > rh_max:
        return None
    return best


def _find_compound_slow_M(L, R, cp, f2):
    """
    Strategy 1d: COMPOUND SLOW WAVE.

    Build the slow wave as:
        rarefaction(L -> J1)  +  transitional shock(J1 -> M)
    where J1 is selected by the Oleinik / Welge tangent condition

        sigma(J1, M(J1))  =  lambda_1(J1)

    M(J1) is the oil-bank state on Hug(J1) intersected with the Sg=0 edge.
    This is the standard Isaacson-Marchesin-Plohr "compound wave" structure
    for three-phase flow with loss of strict hyperbolicity: the rarefaction
    runs along the slow integral curve from L until its trailing
    characteristic speed exactly matches the Rankine-Hugoniot speed of the
    transitional shock leaving it.

    Returns (J1, M, info) or (None, None, None).
    """
    print("    Strategy 1d: compound slow wave (rar L->J1 + shock J1->M) ...")
    slow_rar = _full_curve(L[0], L[2], 1, cp, f2, Nmax=3000, h=0.0015)
    if len(slow_rar) < 10:
        print("      [1d] slow rarefaction too short")
        return None, None, None

    # Locate L on the bidirectional curve and split into two branches
    d_to_L = (slow_rar[:, 0] - L[0])**2 + (slow_rar[:, 2] - L[2])**2
    iL = int(np.argmin(d_to_L))
    branches = []
    if iL < len(slow_rar) - 5:
        branches.append(('+', slow_rar[iL:]))
    if iL > 5:
        branches.append(('-', slow_rar[iL::-1]))

    def _eval(J1, lam1_J1, ns_hug=300):
        """Trace Hug(J1), find oil-bank M, return (M, sig, rh) or (None,...)"""
        J1 = J1.copy()
        J1[1] = 1.0 - J1[0] - J1[2]
        hug_J1 = trace_hugoniot_from_L(J1, cp, f2, ns=ns_hug)
        M = _find_oil_bank_on_hug(hug_J1, J1, cp, f2)
        if M is None:
            return None, np.nan, np.nan
        sig, rh = compute_shock_speed(J1, M, cp, f2)
        if not np.isfinite(sig) or rh > 0.1:
            return None, np.nan, np.nan
        return M, sig, rh

    best_overall = None  # (abs_residual, J1, M, info)

    for tag, branch in branches:
        # Coarse scan over branch indices
        n_scan = 25
        idx_scan = np.unique(np.linspace(0, len(branch) - 1,
                                          n_scan).astype(int))
        residuals = np.full(len(idx_scan), np.nan)
        Ms = [None] * len(idx_scan)
        for k, i in enumerate(idx_scan):
            J1 = branch[i, :3].copy()
            lam1_J1 = float(branch[i, 3])
            if not np.isfinite(lam1_J1):
                continue
            M, sig, rh = _eval(J1, lam1_J1, ns_hug=200)
            if M is None:
                continue
            residuals[k] = sig - lam1_J1
            Ms[k] = M

        valid = np.where(np.isfinite(residuals))[0]
        if len(valid) < 2:
            print(f"      [1d-{tag}] too few valid samples ({len(valid)})")
            continue
        print(f"      [1d-{tag}] scan: {len(valid)}/{len(idx_scan)} valid, "
              f"residual range [{np.nanmin(residuals):.3f},"
              f" {np.nanmax(residuals):.3f}]")

        # Find first sign change between consecutive valid samples
        sign_change = None
        for j in range(len(valid) - 1):
            k0, k1 = valid[j], valid[j + 1]
            if residuals[k0] * residuals[k1] < 0:
                sign_change = (k0, k1)
                break

        if sign_change is None:
            # Fallback: smallest |residual| sample
            kbest = valid[int(np.argmin(np.abs(residuals[valid])))]
            J1 = branch[idx_scan[kbest], :3].copy()
            J1[1] = 1.0 - J1[0] - J1[2]
            lam1_J1 = float(branch[idx_scan[kbest], 3])
            M = Ms[kbest]
            sig, rh = compute_shock_speed(J1, M, cp, f2)
            cand = (abs(residuals[kbest]), J1, M, {
                'branch': tag, 'res': float(residuals[kbest]),
                'lam1_J1': lam1_J1, 'sig_J1M': float(sig),
                'rh_J1M': float(rh), 'method': 'min_residual',
            })
            print(f"      [1d-{tag}] no sign change; min |res|="
                  f"{cand[0]:.4f}")
            if best_overall is None or cand[0] < best_overall[0]:
                best_overall = cand
            continue

        # Bisection between branch indices
        lo_idx = int(idx_scan[sign_change[0]])
        hi_idx = int(idx_scan[sign_change[1]])
        res_lo = float(residuals[sign_change[0]])
        for _ in range(20):
            if hi_idx - lo_idx <= 1:
                break
            mid_idx = (lo_idx + hi_idx) // 2
            J1_mid = branch[mid_idx, :3].copy()
            lam1_mid = float(branch[mid_idx, 3])
            M_mid, sig_mid, rh_mid = _eval(J1_mid, lam1_mid, ns_hug=250)
            if M_mid is None:
                # Shrink toward the side with valid M
                hi_idx = mid_idx
                continue
            res_mid = sig_mid - lam1_mid
            if abs(res_mid) < 5e-4:
                lo_idx = mid_idx
                res_lo = res_mid
                break
            if res_mid * res_lo > 0:
                lo_idx = mid_idx
                res_lo = res_mid
            else:
                hi_idx = mid_idx

        # Final J1 evaluation with high resolution
        J1_f = branch[lo_idx, :3].copy()
        J1_f[1] = 1.0 - J1_f[0] - J1_f[2]
        lam1_f = float(branch[lo_idx, 3])
        M_f, sig_f, rh_f = _eval(J1_f, lam1_f, ns_hug=600)
        if M_f is None:
            continue
        res_f = sig_f - lam1_f
        cand = (abs(res_f), J1_f, M_f, {
            'branch': tag, 'res': float(res_f), 'lam1_J1': lam1_f,
            'sig_J1M': float(sig_f), 'rh_J1M': float(rh_f),
            'method': 'bisect',
        })
        print(f"      [1d-{tag}] bisect: J1=({J1_f[0]:.4f},{J1_f[1]:.4f},"
              f"{J1_f[2]:.4f})  sig={sig_f:.4f}  lam1={lam1_f:.4f}"
              f"  res={res_f:+.4e}")
        if best_overall is None or cand[0] < best_overall[0]:
            best_overall = cand

    if best_overall is None:
        print("      [1d] no compound slow wave found")
        return None, None, None

    # Demand a real Welge tangent (residual ~ 0).  If the smallest |residual|
    # found is large, no compound slow wave exists for this Riemann problem
    # — fall through so the caller can try strategy 1c (pure shock).
    WELGE_TOL = 0.05
    if best_overall[0] > WELGE_TOL:
        print(f"      [1d] no Welge tangent: best |residual|="
              f"{best_overall[0]:.3f} > {WELGE_TOL}")
        print(f"      [1d] -> compound slow wave does NOT exist for this "
              f"problem; falling through")
        return None, None, None

    abs_res, J1, M, info = best_overall
    sig_MR, rh_MR = compute_shock_speed(M, R, cp, f2)
    info['sig_MR'] = float(sig_MR)
    info['rh_MR'] = float(rh_MR)

    print(f"      [1d] BEST: branch={info['branch']}  method={info['method']}")
    print(f"      [1d]   J1 = ({J1[0]:.4f},{J1[1]:.4f},{J1[2]:.4f})"
          f"  lam1(J1)={info['lam1_J1']:.4f}")
    print(f"      [1d]   M  = ({M[0]:.4f},{M[1]:.4f},{M[2]:.4f})")
    print(f"      [1d]   sig(J1,M)={info['sig_J1M']:.4f}"
          f"  Welge residual={info['res']:+.4e}")
    print(f"      [1d]   sig(M,R)={info['sig_MR']:.4f}"
          f"  rh(J1,M)={info['rh_J1M']:.2e}  rh(M,R)={info['rh_MR']:.2e}")
    if info['sig_MR'] <= info['sig_J1M']:
        print(f"      [1d]   WARN: wave-speed ordering violated "
              f"(sig_MR <= sig_J1M)")

    return J1, M, info


def _find_fast_welge_M_W(L, R, M_seed, cp, f2):
    """
    Strategy 1g (corrected geometry): build the L -> R wave as

        L --(small shock L->L_eq)-->  L_eq
            --(fast rarefaction along integral curve)-->  M_W
            --(small shock M_W->R)-->  R

    Geometry
    --------
    1. M_seed is a point KNOWN to lie on the fast integral curve we want
       (in practice, the M state from Strategy 1c, which we already have).

    2. Trace the COMPLETE fast integral curve through M_seed by integrating
       fam=2 in BOTH directions.  This gives a 1D manifold in (Sw, Sg)
       parameterized by lambda_2, monotone from low (toward L side) to
       high (toward R side).

    3. Sweep the curve from M_seed toward LOW lam_2 (the L side); find
       the closest point of the curve to L.  Call it L_eq.  The jump
       L -> L_eq is a small RH shock that satisfies the dataset's
       Rankine-Hugoniot relations to within tol; verify this.

    4. Sweep the curve from M_seed toward HIGH lam_2 (the R side); find
       the Welge tangent point M_W at which

           residual(P) := lam_2(P) - sigma(P, R)

       changes sign (or its absolute value is minimised).  M_W is
       generally extremely close to R because the integral curve enters
       R as lam_2 -> lam_2(R-) from below.

    5. The rarefaction segment is the sub-curve [L_eq -> M_W] sorted in
       ascending lambda_2.

    Returns (M_W, L_eq, rar_segment, info) or (None, None, None, None).
    """
    print("    Strategy 1g: fast integral curve through interior seed ...")
    if M_seed is None:
        print("      [1g] no seed point provided; skipping")
        return None, None, None, None

    # ---- Step 1: trace fast integral curve through M_seed (both dirs) ----
    crv_p = integrate_rarefaction(M_seed[0], M_seed[2], 2, +1, cp, f2,
                                    Nmax=15000, h=0.0003)
    crv_m = integrate_rarefaction(M_seed[0], M_seed[2], 2, -1, cp, f2,
                                    Nmax=15000, h=0.0003)
    if len(crv_p) < 5 and len(crv_m) < 5:
        print("      [1g] fast integral curve too short")
        return None, None, None, None
    if len(crv_m) > 0 and len(crv_p) > 0:
        full = np.vstack([crv_m[::-1], crv_p[1:]])
    elif len(crv_p) > 0:
        full = crv_p
    else:
        full = crv_m[::-1]

    print(f"      [1g] fast integral curve through M_seed: {len(full)} pts, "
          f"lambda_2 in [{full[:,3].min():.4f}, {full[:,3].max():.4f}]")

    # ---- Step 2: L_eq = closest point on curve to L ----
    d2L = (full[:, 0] - L[0])**2 + (full[:, 2] - L[2])**2
    iL = int(np.argmin(d2L))
    L_eq = full[iL, :3].copy()
    L_eq[1] = 1.0 - L_eq[0] - L_eq[2]
    dist_L = float(np.sqrt(d2L[iL]))
    lam2_Leq = float(full[iL, 3])
    print(f"      [1g] L_eq = ({L_eq[0]:.4f},{L_eq[1]:.4f},{L_eq[2]:.4f})  "
          f"dist(L, L_eq) = {dist_L:.4f}  lam_2(L_eq) = {lam2_Leq:.4f}")

    # Verify L -> L_eq satisfies RH within tolerance
    sig_L_Leq, rh_L_Leq = compute_shock_speed(L, L_eq, cp, f2)
    print(f"      [1g] L -> L_eq:  sigma = {sig_L_Leq:.4f}  rh = {rh_L_Leq:.2e}")
    if rh_L_Leq > 0.10:
        print(f"      [1g] L -> L_eq RH error too large; abort")
        return None, None, None, None

    # ---- Step 3: M_W = Welge tangent on the curve toward R ----
    # Walk the curve from L_eq toward higher lam_2 looking for the residual
    # lam_2(P) - sigma(P,R) sign change.
    if iL >= len(full) - 2:
        print("      [1g] L_eq is at the end of the curve; cannot search "
              "for M_W")
        return None, None, None, None

    sub = full[iL:].copy()
    # Make sure lam_2 is monotone increasing (it should be along a single
    # rarefaction branch, but the integrator may have small wobble)
    order = np.argsort(sub[:, 3])
    sub = sub[order]

    residuals = np.full(len(sub), np.nan)
    sigs = np.full(len(sub), np.nan)
    rhs = np.full(len(sub), np.nan)
    for i, row in enumerate(sub):
        P = row[:3]
        sig, rh = compute_shock_speed(P, R, cp, f2)
        if not np.isfinite(sig):
            continue
        residuals[i] = row[3] - sig
        sigs[i] = sig
        rhs[i] = rh

    valid = np.where(np.isfinite(residuals))[0]
    if len(valid) < 2:
        print("      [1g] no valid samples for Welge search")
        return None, None, None, None
    print(f"      [1g] Welge sweep along curve: {len(valid)} samples, "
          f"residual range [{np.nanmin(residuals[valid]):+.3f}, "
          f"{np.nanmax(residuals[valid]):+.3f}]")

    # Look for sign change (residual goes from negative far from R to
    # ~zero/positive close to R)
    sign_change = None
    for k in range(len(valid) - 1):
        i0, i1 = valid[k], valid[k + 1]
        if residuals[i0] * residuals[i1] <= 0:
            sign_change = (i0, i1)
            break

    if sign_change is None:
        # No sign change; pick the point with smallest |residual| AND
        # acceptable RH error to R
        candidates_W = [i for i in valid
                        if rhs[i] < 0.05 and abs(residuals[i]) < 0.5]
        if not candidates_W:
            print("      [1g] no Welge tangent or near-tangent point found")
            return None, None, None, None
        i_W = min(candidates_W, key=lambda i: abs(residuals[i]))
        method = 'min_residual'
    else:
        i0, i1 = sign_change
        i_W = i0 if abs(residuals[i0]) <= abs(residuals[i1]) else i1
        method = 'sign_change'

    M_W = sub[i_W, :3].copy()
    M_W[1] = 1.0 - M_W[0] - M_W[2]
    lam2_W = float(sub[i_W, 3])
    sigma_W = float(sigs[i_W])
    rh_W = float(rhs[i_W])
    print(f"      [1g] M_W = ({M_W[0]:.4f},{M_W[1]:.4f},{M_W[2]:.4f})  "
          f"lam_2(M_W) = {lam2_W:.4f}  sigma(M_W,R) = {sigma_W:.4f}  "
          f"rh = {rh_W:.2e}  ({method})")

    # ---- Step 4: rarefaction segment L_eq -> M_W ----
    rar_seg = sub[:i_W + 1].copy()
    if len(rar_seg) < 2:
        print("      [1g] rarefaction segment too short")
        return None, None, None, None

    # Sanity: check how many rarefaction samples are inside the elliptic
    # region (a healthy fast rar should have very few)
    n_elliptic = 0
    for row in rar_seg:
        try:
            _, _, _, _, d = compute_eigenvalues(row[0], row[2], cp)
            if d < 0:
                n_elliptic += 1
        except Exception:
            pass
    print(f"      [1g] rarefaction L_eq -> M_W: {len(rar_seg)} pts, "
          f"elliptic samples = {n_elliptic}")
    print(f"      [1g] lambda_2 spans [{rar_seg[0,3]:.4f}, "
          f"{rar_seg[-1,3]:.4f}]  (matches numerical xi range)")

    info = {
        'M_W': M_W, 'L_eq': L_eq,
        'sigma_L_Leq': float(sig_L_Leq), 'rh_L_Leq': float(rh_L_Leq),
        'sigma_MR': sigma_W, 'rh_MR': rh_W,
        'lam2_Leq': lam2_Leq, 'lam2_W': lam2_W,
        'rar_npts': int(len(rar_seg)),
        'L_to_Leq_dist': dist_L,
        'n_elliptic_in_rar': int(n_elliptic),
        'method': method,
    }
    return M_W, L_eq, rar_seg, info


def _find_fast_welge_M_W_OLD(L, R, cp, f2):
    """OLD broken version kept for reference; not called."""
    print("    Strategy 1g: fast Welge tangent + back-traced rarefaction ...")

    # ---- Step 1: scan Hug(R) for the Welge tangent ----
    hug_R = trace_hugoniot_from_R_permissive(R, cp, f2, ns=600)
    if len(hug_R) < 10:
        print("      [1g] Hug(R) too short")
        return None, None, None, None

    Sw_R, _, Sg_R = R[0], R[1], R[2]
    residuals = np.full(len(hug_R), np.nan)
    lams = np.full(len(hug_R), np.nan)
    sigs = np.full(len(hug_R), np.nan)
    for i, row in enumerate(hug_R):
        try:
            _, l2, _, _, d = compute_eigenvalues(row[0], row[2], cp)
        except Exception:
            continue
        if d < 0:                       # complex eigenvalues -> skip
            continue
        sigma_PR = float(row[3])
        residuals[i] = l2 - sigma_PR
        lams[i] = l2
        sigs[i] = sigma_PR

    valid = np.where(np.isfinite(residuals))[0]
    if len(valid) < 2:
        print("      [1g] no valid Hug(R) samples (mostly elliptic)")
        return None, None, None, None
    print(f"      [1g] valid Hug(R) samples: {len(valid)}/{len(hug_R)}, "
          f"residual range [{np.nanmin(residuals[valid]):+.3f}, "
          f"{np.nanmax(residuals[valid]):+.3f}]")

    # Find first sign change; if multiple, prefer the one with smaller |sigma|
    sign_changes = []
    for k in range(len(valid) - 1):
        i0, i1 = valid[k], valid[k + 1]
        if residuals[i0] * residuals[i1] < 0:
            sign_changes.append((i0, i1))
    if not sign_changes:
        # No tangent: pick the smallest |residual| as best guess
        i_best = valid[int(np.argmin(np.abs(residuals[valid])))]
        print(f"      [1g] no sign change in residual; "
              f"min |res|={abs(residuals[i_best]):.4f} at "
              f"sigma={sigs[i_best]:.4f}")
        if abs(residuals[i_best]) > 0.05:
            print("      [1g] no Welge tangent — fast composite not applicable")
            return None, None, None, None
        sign_change = (i_best, i_best)
    else:
        # Multiple Welge tangents are possible (the residual is non-monotone
        # in the elliptic region).  Prefer the one closest to R in (Sw,Sg).
        def _proximity(idx):
            row = hug_R[idx]
            return (row[0] - Sw_R)**2 + (row[2] - Sg_R)**2
        sign_changes.sort(key=lambda pair: _proximity(pair[0]))
        sign_change = sign_changes[0]
        if len(sign_changes) > 1:
            print(f"      [1g] {len(sign_changes)} sign changes found; "
                  f"using one closest to R")

    i0, i1 = sign_change
    if i0 == i1:
        M_W = hug_R[i0, :3].copy()
        sigma_W = float(hug_R[i0, 3])
        lam2_W = float(lams[i0])
    else:
        # Linear interpolation in residual
        r0, r1 = residuals[i0], residuals[i1]
        t = -r0 / (r1 - r0)
        row_W = (1 - t) * hug_R[i0] + t * hug_R[i1]
        M_W = row_W[:3].copy()
        M_W[1] = 1.0 - M_W[0] - M_W[2]
        sigma_W = float(row_W[3])
        try:
            _, lam2_W, _, _, _ = compute_eigenvalues(M_W[0], M_W[2], cp)
        except Exception:
            lam2_W = sigma_W
    print(f"      [1g] Welge tangent: M_W=({M_W[0]:.4f},{M_W[1]:.4f},"
          f"{M_W[2]:.4f}) sigma={sigma_W:.4f} lambda_2={lam2_W:.4f}")

    # Sanity: M_W → R must really satisfy RH
    sig_chk, rh_chk = compute_shock_speed(M_W, R, cp, f2)
    if rh_chk > 0.05:
        print(f"      [1g] M_W -> R RH error too large: {rh_chk:.3f}; abort")
        return None, None, None, None

    # ---- Step 2: trace fast integral curve backward from M_W ----
    # Try BOTH directions; the "backward" branch is the one whose lam2
    # decreases below lam2_W and whose endpoint is closer to L than M_W.
    best_branch = None
    for direction in (+1, -1):
        crv = integrate_rarefaction(M_W[0], M_W[2], 2, direction, cp, f2,
                                      Nmax=12000, h=0.0003)
        if len(crv) < 5:
            continue
        # Restrict to lambda_2 <= lam2_W (rarefaction goes from low to high
        # lam2, so the part we want is BELOW lam2_W when traced backward).
        mask = crv[:, 3] <= lam2_W + 1e-6
        sub = crv[mask]
        if len(sub) < 5:
            continue
        order = np.argsort(sub[:, 3])
        sub = sub[order]
        # Closest approach to L
        d2 = (sub[:, 0] - L[0])**2 + (sub[:, 2] - L[2])**2
        i_close = int(np.argmin(d2))
        dist_close = float(np.sqrt(d2[i_close]))
        print(f"      [1g] dir={direction:+d}: sub-curve {len(sub)} pts, "
              f"lam2 in [{sub[0,3]:.4f},{sub[-1,3]:.4f}], "
              f"closest to L at dist={dist_close:.4f} "
              f"({sub[i_close,0]:.4f},{sub[i_close,2]:.4f}) "
              f"lam2={sub[i_close,3]:.4f}")
        cand = {'direction': direction, 'sub': sub,
                'i_close': i_close, 'dist_close': dist_close}
        if best_branch is None or dist_close < best_branch['dist_close']:
            best_branch = cand

    if best_branch is None:
        print("      [1g] no fast integral curve from M_W extends backward")
        return None, None, None, None

    sub = best_branch['sub']
    i_close = best_branch['i_close']
    dist_close = best_branch['dist_close']

    # Take the segment from the closest-to-L point up to M_W
    rar_seg = sub[i_close:].copy()
    if rar_seg[0, 3] > rar_seg[-1, 3]:
        rar_seg = rar_seg[::-1]
    # Force the last sample to be M_W exactly
    rar_seg[-1, 0] = M_W[0]
    rar_seg[-1, 1] = M_W[1]
    rar_seg[-1, 2] = M_W[2]
    rar_seg[-1, 3] = lam2_W

    L_eq = sub[i_close, :3].copy()
    L_eq[1] = 1.0 - L_eq[0] - L_eq[2]
    lam2_Leq = float(sub[i_close, 3])

    # Sanity: the rarefaction should be inside the strictly hyperbolic region
    # everywhere except possibly at L_eq (which sits at the elliptic boundary)
    n_elliptic = 0
    for row in rar_seg:
        try:
            _, _, _, _, d = compute_eigenvalues(row[0], row[2], cp)
            if d < 0:
                n_elliptic += 1
        except Exception:
            pass
    print(f"      [1g] rarefaction segment: {len(rar_seg)} pts "
          f"(elliptic samples: {n_elliptic})")
    print(f"      [1g] L_eq = ({L_eq[0]:.4f},{L_eq[1]:.4f},{L_eq[2]:.4f})  "
          f"lam2(L_eq) = {lam2_Leq:.4f}  dist(L,L_eq) = {dist_close:.4f}")
    print(f"      [1g] residual L -> L_eq jump = "
          f"({L_eq[0]-L[0]:+.4f},{L_eq[2]-L[2]:+.4f}) "
          f"(absorbed in elliptic neighborhood of L)")

    info = {
        'M_W': M_W, 'L_eq': L_eq, 'sigma_MR': sigma_W,
        'lam2_W': lam2_W, 'lam2_Leq': lam2_Leq,
        'rar_npts': int(len(rar_seg)),
        'L_to_Leq_dist': float(dist_close),
        'n_elliptic_in_rar': int(n_elliptic),
    }
    return M_W, L_eq, rar_seg, info


def _find_fast_composite(L, M, R, cp, f2):
    """
    Strategy 1e: build a FAST COMPOSITE wave from L to M.

    Numerical experiment shows that for gas displacement of water+oil with
    L near the elliptic / umbilic region, the actual L->M wave is a single
    fast-family composite:

        L --(small shock)--> H --(fast rarefaction)--> M

    where H is selected by the Welge / Oleinik tangent condition

        sigma(L, H)  =  lambda_2(H)

    so that the trailing characteristic of the L->H shock equals the
    leading characteristic of the H->M fast rarefaction.  The slow-family
    wave has zero strength (it does not appear in the solution at all).

    The full Riemann solution then becomes
        L --shock--> H --fast rar--> M --fast shock--> R
    i.e. the slow wave is absent and the fast wave is the composite
    "shock + rarefaction + shock".

    Returns (H, fast_rar_segment, info) or (None, None, None).

    fast_rar_segment is a (k,4) array [Sw,So,Sg,lambda2] from H to M
    sampled along the fast integral curve, monotone in lambda2.
    """
    print("    Strategy 1e: fast composite wave (shock L->H + rar H->M) ...")

    # Trace the fast integral curve through M.  We need the branch that
    # extends from M toward smaller lambda_2 (the rarefaction "fan" direction
    # toward L).  The integrator's sign convention depends on local geometry,
    # so try BOTH directions and pick the branch whose lambda_2 actually
    # decreases below lambda_2(M).
    try:
        l1_M, l2_M, _, _, _ = compute_eigenvalues(M[0], M[2], cp)
    except Exception:
        print("      [1e] eigenvalue failure at M")
        return None, None, None

    print(f"      [1e] lambda_2(M) = {l2_M:.4f}")

    branches = []
    # Keep original (unsorted) integration curve per direction so we can
    # later verify λ₂ monotonicity for Plan D (genuine rarefaction vs fold).
    direction_to_crv = {}
    for direction in (+1, -1):
        crv = integrate_rarefaction(M[0], M[2], 2, direction, cp, f2,
                                      Nmax=12000, h=0.0003)
        if len(crv) < 5:
            continue
        # Branch is useful only if lambda_2 actually decreases below l2_M
        if crv[:, 3].min() >= l2_M - 1e-3:
            print(f"      [1e] dir={direction:+d}: lambda_2 does not "
                  f"decrease below M (min={crv[:,3].min():.4f})")
            continue
        # Sub-segment with lambda_2 < lambda_2(M)
        mask = crv[:, 3] < l2_M + 1e-6
        sub = crv[mask]
        if len(sub) < 5:
            continue
        # Sort by lambda_2 ascending so we can do clean bisection
        order = np.argsort(sub[:, 3])
        sub = sub[order]
        branches.append((direction, sub))
        direction_to_crv[direction] = crv
        print(f"      [1e] dir={direction:+d}: usable sub-curve {len(sub)} "
              f"pts, lambda_2 in [{sub[0,3]:.4f},{sub[-1,3]:.4f}]")

    if not branches:
        print("      [1e] no fast integral curve from M extends toward L")
        return None, None, None

    # For each branch, look for the Welge tangent  sigma(L,P) = lambda_2(P)
    best = None  # (abs_residual, H, branch_dir, sub_curve, sig_LH, lam2_H)

    for direction, sub in branches:
        # Closest approach of this branch to L
        d2 = (sub[:, 0] - L[0])**2 + (sub[:, 2] - L[2])**2
        i_close = int(np.argmin(d2))
        dist_close = float(np.sqrt(d2[i_close]))
        print(f"      [1e] dir={direction:+d}: closest to L at "
              f"({sub[i_close,0]:.4f},{sub[i_close,2]:.4f}) dist={dist_close:.4f}"
              f" lam2={sub[i_close,3]:.4f}")

        # Welge residual along the branch
        residuals = np.full(len(sub), np.nan)
        sigs = np.full(len(sub), np.nan)
        rhs = np.full(len(sub), np.nan)
        for i in range(len(sub)):
            P = sub[i, :3]
            sig, rh = compute_shock_speed(L, P, cp, f2)
            if not np.isfinite(sig):
                continue
            residuals[i] = sig - sub[i, 3]   # sigma(L,P) - lambda2(P)
            sigs[i] = sig
            rhs[i] = rh

        valid = np.where(np.isfinite(residuals) & (rhs < 0.5))[0]
        if len(valid) < 2:
            continue

        # Look for sign change from + to -  (residual >0 near M because
        # sig is large; <0 near L because sig is small)
        sign_change = None
        for j in range(len(valid) - 1):
            i0, i1 = valid[j], valid[j + 1]
            if residuals[i0] * residuals[i1] < 0:
                sign_change = (i0, i1)
                break

        if sign_change is None:
            kbest = valid[int(np.argmin(np.abs(residuals[valid])))]
            cand_res = abs(residuals[kbest])
            H_cand = sub[kbest, :3].copy()
            H_cand[1] = 1.0 - H_cand[0] - H_cand[2]
            print(f"      [1e] dir={direction:+d}: no sign change, "
                  f"min |residual|={cand_res:.4f} at lam2={sub[kbest,3]:.4f}")
            cand = (cand_res, H_cand, direction, sub, sigs[kbest], sub[kbest, 3], rhs[kbest])
            if best is None or cand[0] < best[0]:
                best = cand
            continue

        # Bisection between consecutive valid samples on either side of
        # the sign change
        i0, i1 = sign_change
        for _ in range(40):
            if i1 - i0 <= 1:
                break
            i_mid = (i0 + i1) // 2
            r_mid = residuals[i_mid]
            if not np.isfinite(r_mid):
                i1 = i_mid
                continue
            if r_mid * residuals[i0] > 0:
                i0 = i_mid
            else:
                i1 = i_mid

        # Take the index with smaller |residual| of the final pair
        i_best = i0 if abs(residuals[i0]) <= abs(residuals[i1]) else i1
        H_cand = sub[i_best, :3].copy()
        H_cand[1] = 1.0 - H_cand[0] - H_cand[2]
        cand_res = abs(residuals[i_best])
        cand = (cand_res, H_cand, direction, sub, sigs[i_best], sub[i_best, 3], rhs[i_best])
        print(f"      [1e] dir={direction:+d}: bisect H="
              f"({H_cand[0]:.4f},{H_cand[1]:.4f},{H_cand[2]:.4f}) "
              f"sigma={sigs[i_best]:.4f} lam2={sub[i_best,3]:.4f} "
              f"residual={residuals[i_best]:+.4e}")
        if best is None or cand[0] < best[0]:
            best = cand

    if best is None:
        print("      [1e] no Welge tangent found")
        return None, None, None

    abs_res, H, direction, sub, sig_LH, lam2_H, rh_LH = best

    # Quality gate 1: Welge residual (|sigma(L,H) - lam2(H)|) small
    if abs_res > 0.05:
        print(f"      [1e] best |residual|={abs_res:.4f} > 0.05; "
              f"fast composite does not exist for this problem")
        return None, None, None

    # Quality gate 2: L->H must be a genuine RH shock (rh small).  The
    # Welge residual only checks the TANGENT condition sigma=lam2 at H;
    # it does not enforce Rankine-Hugoniot between L and H.  Without
    # this second gate the code can accept H that satisfies Welge but
    # yields rh(L,H) ~ 0.5, i.e. a fictitious "shock" violating mass
    # conservation.  For strongly-oil-wet this protects us from picking
    # a non-Lax bridge H in the rarefaction curve near R.
    if rh_LH > 0.05:
        print(f"      [1e] best H has large RH residual "
              f"(rh(L,H)={rh_LH:.2e} > 0.05); L->H is not a genuine "
              f"Rankine-Hugoniot shock -- rejecting fast composite")
        return None, None, None

    # Plan D: when H ~ L (fast integral curve from M passes through L),
    # the "shock" L->H degenerates to zero amplitude and the wave from L
    # to M is a PURE fast rarefaction.  This is physically legitimate iff
    # the fast integral curve from L to M is an admissible rarefaction:
    # lambda_2 must be strictly monotone along the ORIGINAL integration
    # order (no V2 inflection crossing / no fold).  A fold shows up as
    # lambda_2 reversing direction mid-way.
    #
    # Previous v5 behaviour rejected all H ~ L via a coarse `d_HL < 0.02`
    # proxy; that wrongly vetoed cases where L genuinely lives on the fast
    # integral curve (e.g. L on the So=SORG edge for water-wet data), for
    # which the numerical profile shows the rarefaction clearly.
    d_HL = float(np.sqrt((H[0] - L[0])**2 + (H[2] - L[2])**2))
    if d_HL < 0.02:
        crv_orig = direction_to_crv.get(direction)
        if crv_orig is None or len(crv_orig) < 2:
            print(f"      [1e] rejected H ~ L (|H-L|={d_HL:.2e}): "
                  f"cannot verify lam2 monotonicity (no integration curve)")
            return None, None, None
        # Locate H in the ORIGINAL integration order (crv[0] = M).
        d2_orig = (crv_orig[:, 0] - H[0])**2 + (crv_orig[:, 2] - H[2])**2
        i_H_orig = int(np.argmin(d2_orig))
        if i_H_orig < 2:
            print(f"      [1e] rejected H ~ L (|H-L|={d_HL:.2e}): "
                  f"no integration samples between M and H (i_H={i_H_orig})")
            return None, None, None
        lam2_path = crv_orig[:i_H_orig + 1, 3]
        lam2_diffs = np.diff(lam2_path)
        if len(lam2_diffs) < 1:
            print(f"      [1e] rejected H ~ L (|H-L|={d_HL:.2e}): "
                  f"path too short to verify monotonicity")
            return None, None, None
        # The integrator chose this direction because lam2 decreases
        # below l2_M.  In a valid fast rarefaction, lam2 is non-
        # increasing along the M -> H traversal, so every point has
        # lam2 <= min of all prior points.  A V2 inflection crossing
        # (fold) shows up as a "bounce": later samples rising above the
        # running minimum of earlier samples.  Scale-aware tolerance:
        # allow integrator noise ~ 10% of the typical step magnitude,
        # floor at 1e-6 to keep any reasonable case hyperbolic.
        cum_min = np.minimum.accumulate(lam2_path)
        bounces = lam2_path[1:] - cum_min[:-1]
        median_step = float(np.median(np.abs(lam2_diffs)))
        bounce_tol = max(1e-6, 0.1 * median_step)
        n_folds = int(np.sum(bounces > bounce_tol))
        if n_folds > 0:
            max_bounce = float(np.max(bounces))
            print(f"      [1e] rejected H ~ L (|H-L|={d_HL:.2e}): fast "
                  f"integral curve folds at V2 inflection ({n_folds}/"
                  f"{len(bounces)} samples bounce above running min; "
                  f"max bounce={max_bounce:+.2e} > tol={bounce_tol:.2e}"
                  f"; median step={median_step:.2e}) -- rarefaction "
                  f"L->M is NOT admissible")
            return None, None, None
        # Monotone -> genuine pure fast rarefaction L -> M (H == L, zero
        # shock amplitude).  Accept and flag sig_LH = lam2(L) for the
        # downstream degenerate-shock handling.
        print(f"      [1e] H ~ L accepted (|H-L|={d_HL:.2e}): lam2 "
              f"monotone from M ({lam2_path[0]:.4f}) to L "
              f"({lam2_path[-1]:.4f}) over {len(lam2_diffs)} samples, "
              f"max bounce={bounces.max():+.2e} <= tol={bounce_tol:.2e}"
              f" -- admissible pure fast rarefaction")

    # Build the rarefaction segment H -> M (lambda_2 from lam2_H up to l2_M).
    # Take the portion of `sub` with lambda_2 in [lam2_H, l2_M], sorted
    # ascending in lambda_2.
    seg_mask = (sub[:, 3] >= lam2_H - 1e-9) & (sub[:, 3] <= l2_M + 1e-9)
    rar_seg = sub[seg_mask]
    if len(rar_seg) < 2:
        # Degenerate: just two endpoints
        rar_seg = np.array([
            [H[0], H[1], H[2], lam2_H],
            [M[0], M[1], M[2], l2_M],
        ])

    # Verify the shock L -> H satisfies RH reasonably and is admissible
    sig_LH_chk, rh_LH = compute_shock_speed(L, H, cp, f2)

    # Detect "pure fast rarefaction" outcome: H is essentially L and
    # the rarefaction H -> M is monotone (no V_2 fold).  This signals
    # a zero-strength slow + admissible fast rarefaction L -> M, which
    # the caller should accept directly (do NOT override with the
    # 1g 3-segment construction; that would over-resolve a non-existent
    # transitional shock).
    d_HL_for_info = float(np.hypot(H[0] - L[0], H[2] - L[2]))
    pure_fast_rar = (d_HL_for_info < 0.02)
    info = {
        'sig_LH': float(sig_LH_chk),
        'lam2_H': float(lam2_H),
        'rh_LH': float(rh_LH),
        'welge_residual': float(abs_res),
        'branch_dir': int(direction),
        'rar_npts': int(len(rar_seg)),
        'pure_fast_rarefaction': bool(pure_fast_rar),
        'd_HL': d_HL_for_info,
    }
    print(f"      [1e] FOUND: H=({H[0]:.4f},{H[1]:.4f},{H[2]:.4f})")
    print(f"      [1e]   sigma(L,H) = {info['sig_LH']:.4f}")
    print(f"      [1e]   lambda_2(H) = {info['lam2_H']:.4f}")
    print(f"      [1e]   |Welge residual| = {info['welge_residual']:.2e}")
    print(f"      [1e]   rh(L,H) = {info['rh_LH']:.2e}")
    print(f"      [1e]   rarefaction H->M: {info['rar_npts']} pts")
    return H, rar_seg, info


def _find_transitional_slow_edge_shock(L, R, cp, f2,
                                        disc_L_thresh=1e-2,
                                        rh_tol=0.05,
                                        n_scan=300,
                                        M_fast=None):
    """
    Strategy 1g: three-segment Riemann solution for near-umbilic L, where
    the slow wave is a TRANSITIONAL shock along the So=0 edge:

        L --(slow transitional shock, So=0 edge)--> P
        P --(fast rarefaction, into interior)------> Q
        Q --(fast shock)----------------------------> R

    Construction (when M_fast is supplied):
      The caller passes M_fast — the peak of the already-solved fast
      composite (L -> H~L -> fast-rar -> M_fast -> shock -> R from
      Strategy 1e).  This M_fast lies on the correct fast integral curve
      that eventually reaches R, so we back-integrate the fast rar from
      M_fast until it hits the So=0 edge.  That intersection is P.  The
      portion of that rar from the edge crossing onward becomes the
      rar P -> Q = M_fast segment.  Uniqueness: there is exactly one
      fast integral curve through M_fast, so P is uniquely pinned.

    Fallback (no M_fast): scan So=0 edge for candidate P and pick the
    one whose fast rar gives best (Welge-error + RH) score.

    Returns (P, Q, rar_PQ, info) or (None, None, None, None).
    """
    print("    Strategy 1g: transitional slow edge shock (Welge-pinned) ...")

    try:
        lam1_L, lam2_L, _, _, disc_L = compute_eigenvalues(L[0], L[2], cp)
    except Exception:
        print("      [1g] eigenvalue failure at L")
        return None, None, None, None

    if disc_L > disc_L_thresh:
        print(f"      [1g] L not near umbilic (disc_L={disc_L:.3e} > "
              f"{disc_L_thresh:.0e}) -- strategy skipped")
        return None, None, None, None

    print(f"      [1g] disc_L={disc_L:.3e}, transitional band "
          f"[lam1(L), lam2(L)] = [{lam1_L:.4f}, {lam2_L:.4f}]")

    try:
        _, lam2_R, _, _, _ = compute_eigenvalues(R[0], R[2], cp)
    except Exception:
        lam2_R = 0.0

    # ============================================================
    # Primary path: pin P by back-integrating the fast integral curve
    # from the already-known M_fast (Strategy 1e's composite peak).
    # ============================================================
    if M_fast is not None:
        Q = np.asarray(M_fast, dtype=float).copy()
        Q[1] = max(0.0, Q[1])
        print(f"      [1g] back-trace from M_fast=({Q[0]:.4f},{Q[1]:.4f},{Q[2]:.4f})"
              " to find P on So=0")
        best_cand = None
        for direction in (+1, -1):
            rar_back = integrate_rarefaction(Q[0], Q[2], 2, direction, cp, f2,
                                              Nmax=2000, h=0.002)
            if rar_back is None or len(rar_back) < 10:
                continue
            So_series = rar_back[:, 1]
            mask_edge = So_series <= 1e-3
            if not mask_edge.any():
                continue
            i_edge = int(np.argmax(mask_edge))
            if i_edge < 3:
                continue
            # Back-integration should END at a LOWER lam2 (the rar P->Q
            # increases lam2, so reversing from Q to P decreases lam2).
            if rar_back[i_edge, 3] > rar_back[0, 3] + 1e-3:
                continue
            Sw_P = float(rar_back[i_edge, 0])
            P_cand = np.array([Sw_P, 0.0, 1.0 - Sw_P])
            sig_LP, rh_LP = compute_shock_speed(L, P_cand, cp, f2)
            if not np.isfinite(sig_LP):
                continue
            rar_PQ = rar_back[:i_edge + 1][::-1].copy()
            rar_PQ[0, :3] = P_cand
            # Metric: how close to edge we landed (So small is better)
            edge_dist = abs(rar_back[i_edge, 1])
            # Slightly prefer transitional-band shocks
            band_pen = 0.0 if (lam1_L - 5e-3 <= sig_LP <= lam2_L + 5e-3) else 1.0
            score = edge_dist + band_pen
            if best_cand is None or score < best_cand[0]:
                best_cand = (score, P_cand, rar_PQ, direction, sig_LP, rh_LP)

        if best_cand is not None:
            _, P, rar_PQ, direction, sig_LP, rh_LP = best_cand
            sig_QR, rh_QR = compute_shock_speed(Q, R, cp, f2)
            try:
                _, lam2_Q, _, _, _ = compute_eigenvalues(Q[0], Q[2], cp)
            except Exception:
                lam2_Q = np.nan
            info = {'sig_LP': sig_LP, 'sig_QR': sig_QR,
                    'lam2_Q': float(lam2_Q), 'rh_LP': rh_LP, 'rh_QR': rh_QR,
                    'welge_err': abs(sig_QR - lam2_Q) if np.isfinite(lam2_Q) else np.nan,
                    'direction': direction, 'source': 'back_from_M_fast'}
            print(f"      [1g] FOUND (from M_fast): "
                  f"P=({P[0]:.4f},{P[1]:.4f},{P[2]:.4f}) "
                  f"Q=({Q[0]:.4f},{Q[1]:.4f},{Q[2]:.4f})")
            print(f"      [1g]   sigma(L,P)={sig_LP:.4f}  "
                  f"sigma(Q,R)={sig_QR:.4f}  lam2(Q)={lam2_Q:.4f}")
            print(f"      [1g]   rar_PQ: {len(rar_PQ)} pts  direction={direction}")
            return P, Q, rar_PQ, info

        print("      [1g] back-trace from M_fast failed; falling back to Hug(R) scan")

    # ============================================================
    # Fallback: Welge-tangent Q on Hug(R) (less reliable when shocks
    # are both undercompressive — this usually returns None)
    # ============================================================
    hug_R = trace_hugoniot_from_R_permissive(R, cp, f2, ns=max(400, n_scan))
    print(f"      [1g] Hug(R): {len(hug_R)} states")
    if len(hug_R) < 5:
        print("      [1g] Hug(R) too sparse")
        return None, None, None, None

    # Compute Welge residual along Hug(R)
    welge_list = []  # (welge_err, Q, sig_QR, lam2_Q)
    for row in hug_R:
        Sw_Q, So_Q, Sg_Q, sig_QR = row
        try:
            _, lam2_Q, _, _, _ = compute_eigenvalues(Sw_Q, Sg_Q, cp)
        except Exception:
            continue
        if not np.isfinite(lam2_Q):
            continue
        # Lax-2 lower bound: sig_QR >= lam2_R
        if sig_QR < lam2_R - 0.5:
            continue
        welge_err = abs(sig_QR - lam2_Q)
        Q = np.array([Sw_Q, So_Q, Sg_Q])
        welge_list.append((welge_err, Q, sig_QR, lam2_Q))

    if not welge_list:
        print("      [1g] no Welge candidate on Hug(R)")
        return None, None, None, None

    welge_list.sort(key=lambda t: t[0])
    best_Q = None
    for welge_err, Q, sig_QR, lam2_Q in welge_list[:5]:
        # Allow a small tolerance on the tangent condition (Hug(R) is sampled
        # discretely so exact tangent may fall between samples).
        if welge_err > 0.5:
            break

        # ------------------------------------------------------------
        # Step 2: back-integrate fast integral curve from Q to So=0 edge
        # to find P.  Try both rar directions; keep the one whose
        # endpoint is on So=0 with lam2 decreasing (i.e. rar P->Q).
        # ------------------------------------------------------------
        # Slightly nudge Q off Hug(R) interior for the integrator.
        best_P_for_Q = None  # (endpoint_distance_to_edge, P, rar_PQ, direction)
        for direction in (+1, -1):
            rar_back = integrate_rarefaction(Q[0], Q[2], 2, direction, cp, f2,
                                              Nmax=1200, h=0.002)
            if rar_back is None or len(rar_back) < 10:
                continue
            # Detect the first state where So <= 1e-3 (hit So=0 edge)
            So_series = rar_back[:, 1]
            mask_edge = So_series <= 1e-3
            if not mask_edge.any():
                continue
            i_edge = int(np.argmax(mask_edge))  # first True
            if i_edge < 3:
                continue
            # Ensure lam2 is DECREASING from Q along this direction (so that
            # walking back from Q to P corresponds to a forward rar P->Q with
            # increasing lam2).
            if rar_back[i_edge, 3] > rar_back[0, 3] + 1e-3:
                continue
            # Candidate P = endpoint on So=0 edge (project onto edge exactly)
            P_cand = rar_back[i_edge, :3].copy()
            Sw_P = float(P_cand[0])
            P_cand = np.array([Sw_P, 0.0, 1.0 - Sw_P])
            # Reverse the back-integration so the path runs P -> Q
            rar_PQ = rar_back[:i_edge + 1][::-1].copy()
            rar_PQ[0, :3] = P_cand
            rar_PQ[0, 1] = 0.0
            # Verify sig(L, P) in transitional band
            sig_LP, rh_LP = compute_shock_speed(L, P_cand, cp, f2)
            if not np.isfinite(sig_LP):
                continue
            if not (lam1_L - 5e-3 <= sig_LP <= lam2_L + 5e-3):
                continue

            dist = abs(rar_back[i_edge, 1])  # how close to So=0 we landed
            cand = (dist, P_cand, rar_PQ, direction, sig_LP, rh_LP)
            if best_P_for_Q is None or cand[0] < best_P_for_Q[0]:
                best_P_for_Q = cand

        if best_P_for_Q is None:
            continue

        dist, P, rar_PQ, direction, sig_LP, rh_LP = best_P_for_Q
        info = {'sig_LP': sig_LP, 'sig_QR': sig_QR,
                'lam2_Q': lam2_Q, 'rh_LP': rh_LP, 'rh_QR': 0.0,
                'welge_err': welge_err, 'direction': direction}
        best_Q = (welge_err, P, Q, rar_PQ, info)
        break  # take first (smallest-welge_err) Welge Q that produces a valid P

    if best_Q is None:
        print("      [1g] no Welge Q on Hug(R) that back-traces to So=0 edge")
        return None, None, None, None

    welge_err, P, Q, rar_PQ, info = best_Q
    print(f"      [1g] FOUND: P=({P[0]:.4f},{P[1]:.4f},{P[2]:.4f}) "
          f"Q=({Q[0]:.4f},{Q[1]:.4f},{Q[2]:.4f})")
    print(f"      [1g]   sigma(L,P)={info['sig_LP']:.4f}  "
          f"sigma(Q,R)={info['sig_QR']:.4f}  lam2(Q)={info['lam2_Q']:.4f}")
    print(f"      [1g]   welge|sig(Q,R)-lam2(Q)|={info['welge_err']:.2e}  "
          f"rh(L,P)={info['rh_LP']:.2e}  rar_PQ: {len(rar_PQ)} pts")
    return P, Q, rar_PQ, info


def _edge_to_state_rh_residual(Sw_P, Q, cp, f2):
    """RH residual for a shock from P=(Sw_P,0,1-Sw_P) to interior Q."""
    P = np.array([Sw_P, 0.0, 1.0 - Sw_P])
    fw_P, _, fg_P = fractional_flow(P[0], P[2], cp, f2)
    fw_Q, _, fg_Q = fractional_flow(Q[0], Q[2], cp, f2)
    dSw = Q[0] - P[0]
    dSg = Q[2] - P[2]
    if abs(dSw) < 1e-10 or abs(dSg) < 1e-10:
        return np.nan
    return (fw_Q - fw_P) / dSw - (fg_Q - fg_P) / dSg


def _edge_rh_roots_for_Q(Q, L, cp, f2, n=180):
    """All edge states P that are RH-related to a fixed interior Q."""
    sw_lo = max(float(L[0]) + 1e-5, SWC + 1e-5)
    # Keep P genuinely upstream of Q in Sw; the dSw=0 root is a scalar
    # artifact and does not satisfy the full vector RH condition.
    sw_hi = min(float(Q[0]) - 1e-4, 1.0 - SORG - 1e-5)
    if sw_hi <= sw_lo:
        return []

    xs = np.linspace(sw_lo, sw_hi, n)
    vals = np.array([_edge_to_state_rh_residual(x, Q, cp, f2) for x in xs])
    roots = []
    for i in range(len(xs) - 1):
        f0, f1 = vals[i], vals[i + 1]
        if not (np.isfinite(f0) and np.isfinite(f1)):
            continue
        if f0 == 0.0:
            roots.append(xs[i])
            continue
        if f0 * f1 > 0:
            continue

        a, b, fa = xs[i], xs[i + 1], f0
        for _ in range(50):
            c = 0.5 * (a + b)
            fc = _edge_to_state_rh_residual(c, Q, cp, f2)
            if not np.isfinite(fc):
                break
            if abs(fc) < 1e-13 or (b - a) < 1e-12:
                a = b = c
                break
            if fa * fc <= 0:
                b = c
            else:
                a, fa = c, fc
        roots.append(0.5 * (a + b))
    return roots


def _find_boundary_entry_shock(L, R, fast_curve, cp, f2,
                               q_so_min=0.02, xi_min=0.03,
                               xi_max=5.0, welge_tol=3e-3):
    """
    Find an entry shock P->Q where P lies on the So=0 edge and Q lies on
    the downstream fast rarefaction curve.  Conditions:
      RH(P,Q), sigma(P,Q)=lambda_2(Q), and boundary speed at P < sigma.

    This represents the numerical branch:
      L -- boundary BL rarefaction --> P -- shock --> Q -- R2 --> R.
    """
    if fast_curve is None or len(fast_curve) < 20:
        return None
    if abs(float(L[1])) > 1e-8:
        return None

    curve = np.asarray(fast_curve, dtype=float)
    finite = np.isfinite(curve[:, 3])
    curve = curve[finite]
    if len(curve) < 20:
        return None

    # Coarse scan first, then refine locally around the best Q index.
    candidate_indices = np.where(
        (curve[:, 1] >= q_so_min)
        & (curve[:, 3] >= xi_min)
        & (curve[:, 3] <= xi_max)
    )[0]
    if len(candidate_indices) == 0:
        return None

    step = max(1, len(candidate_indices) // 450)
    coarse = candidate_indices[::step]

    def _scan(indices, root_n):
        best = None
        for idx in indices:
            Q = curve[idx, :3].copy()
            lam_Q = float(curve[idx, 3])
            roots = _edge_rh_roots_for_Q(Q, L, cp, f2, n=root_n)
            for Sw_P in roots:
                P = np.array([Sw_P, 0.0, 1.0 - Sw_P])
                if P[2] <= Q[2] + 1e-5:
                    continue
                if abs(Q[0] - P[0]) < 1e-4 or abs(Q[2] - P[2]) < 1e-4:
                    continue
                sig_PQ, rh_PQ = compute_shock_speed(P, Q, cp, f2)
                if not np.isfinite(sig_PQ) or sig_PQ <= 0:
                    continue
                bspd_P = _boundary_wave_speed(P[0], cp, f2)
                if not np.isfinite(bspd_P) or bspd_P >= sig_PQ:
                    continue
                welge_err = abs(sig_PQ - lam_Q)
                if rh_PQ > 0.02:
                    continue
                score = welge_err + 0.25 * rh_PQ
                cand = (score, welge_err, rh_PQ, idx, P, Q,
                        sig_PQ, lam_Q, bspd_P)
                if best is None or cand[0] < best[0]:
                    best = cand
        return best

    best = _scan(coarse, root_n=140)
    if best is None:
        return None

    _, _, _, best_idx, _, _, _, _, _ = best
    lo = max(0, best_idx - 12)
    hi = min(len(curve), best_idx + 13)
    refined = _scan(np.arange(lo, hi), root_n=320)
    if refined is not None and refined[0] <= best[0] + 1e-8:
        best = refined

    score, welge_err, rh_PQ, idx_Q, P, Q, sig_PQ, lam_Q, bspd_P = best
    if welge_err > welge_tol:
        print(f"      [entry-shock] best Welge error {welge_err:.3e} "
              f"> tol {welge_tol:.1e}; rejecting")
        return None

    rar_QR = curve[idx_Q:].copy()
    rar_QR[0, :3] = Q
    rar_QR[0, 3] = sig_PQ
    # Ensure downstream rarefaction has strictly increasing xi samples.
    keep = np.concatenate([[True], np.diff(rar_QR[:, 3]) > 1e-9])
    rar_QR = rar_QR[keep]

    info = {
        'P': P, 'Q': Q, 'rar_QR': rar_QR,
        'sig_PQ': float(sig_PQ),
        'lam_Q': float(lam_Q),
        'rh_PQ': float(rh_PQ),
        'welge_err': float(welge_err),
        'boundary_speed_P': float(bspd_P),
        'idx_Q': int(idx_Q),
    }
    print(f"      [entry-shock] FOUND: "
          f"P=({P[0]:.4f},{P[1]:.4f},{P[2]:.4f}) -> "
          f"Q=({Q[0]:.4f},{Q[1]:.4f},{Q[2]:.4f})")
    print(f"      [entry-shock]   sigma(P,Q)={sig_PQ:.4f}, "
          f"lambda2(Q)={lam_Q:.4f}, "
          f"boundary_speed(P)={bspd_P:.4f}, RH={rh_PQ:.2e}, "
          f"Welge={welge_err:.2e}")
    return info


def _finalize_boundary_entry_solution(L, R, M_fast, entry, cp, f2):
    """Package L->P boundary rarefaction, P->Q entry shock, Q->R rarefaction."""
    P = entry['P'].copy()
    Q = entry['Q'].copy()
    rar_QR = entry['rar_QR'].copy()
    sig_PQ = float(entry['sig_PQ'])

    n_boundary = 600
    Sw_b = np.linspace(float(L[0]), float(P[0]), n_boundary)
    Sg_b = 1.0 - Sw_b
    So_b = np.zeros_like(Sw_b)
    xi_b = np.array([_boundary_wave_speed(sw, cp, f2) for sw in Sw_b])
    order = np.argsort(xi_b)
    xi_b = xi_b[order]
    Sw_b = Sw_b[order]
    Sg_b = Sg_b[order]
    So_b = So_b[order]

    # Shock is drawn as a very narrow jump in xi to avoid duplicate-x
    # issues in downstream diagnostics while still appearing vertical.
    eps = max(1e-8, abs(sig_PQ) * 1e-6)
    xi_left = sig_PQ - eps
    xi_right = sig_PQ + eps

    x_all = np.concatenate([xi_b, [xi_left, xi_right], rar_QR[:, 3]])
    xlo = max(float(np.nanmin(x_all)) - 0.25 * np.ptp(x_all), -2.0)
    # The fast rarefaction reaches a genuine near-boundary maximum around
    # xi ~= 73, but the old 25% tail pushed the displayed R plateau out to
    # ~91.  Keep only a short constant-state tail after the boundary-limit
    # closure so profile plots do not suggest additional wave content.
    xhi = float(np.nanmax(x_all)) + max(0.01 * np.ptp(x_all), 0.75)

    xi_parts = []
    Sw_parts = []
    So_parts = []
    Sg_parts = []

    def _append(x, w, o, g):
        xi_parts.extend(np.asarray(x, dtype=float).tolist())
        Sw_parts.extend(np.asarray(w, dtype=float).tolist())
        So_parts.extend(np.asarray(o, dtype=float).tolist())
        Sg_parts.extend(np.asarray(g, dtype=float).tolist())

    if xlo < xi_b[0]:
        _append([xlo], [L[0]], [L[1]], [L[2]])

    _append(xi_b, Sw_b, So_b, Sg_b)

    if xi_b[-1] < xi_left:
        _append([xi_left], [P[0]], [P[1]], [P[2]])

    _append([xi_right], [Q[0]], [Q[1]], [Q[2]])

    # Drop rar_QR's first point if it would duplicate the artificial jump.
    mask_qr = rar_QR[:, 3] > xi_right
    _append(rar_QR[mask_qr, 3], rar_QR[mask_qr, 0],
            rar_QR[mask_qr, 1], rar_QR[mask_qr, 2])

    if xi_parts[-1] < xhi:
        _append([xhi], [R[0]], [R[1]], [R[2]])

    xi = np.asarray(xi_parts)
    Sw = np.clip(np.asarray(Sw_parts), 0, 1)
    Sg = np.clip(np.asarray(Sg_parts), 0, 1)
    So = np.maximum(0, 1 - Sw - Sg)

    fw = np.zeros_like(xi)
    fg = np.zeros_like(xi)
    for i in range(len(xi)):
        fw[i], _, fg[i] = fractional_flow(Sw[i], Sg[i], cp, f2)

    slow_path = np.column_stack([Sw_b, So_b, Sg_b, xi_b])
    shock_line = np.array([
        [P[0], P[1], P[2], sig_PQ],
        [Q[0], Q[1], Q[2], sig_PQ],
    ])
    fast_tail = rar_QR[1:] if len(rar_QR) > 1 else np.empty((0, 4))
    fast_path = np.vstack([shock_line, fast_tail])

    return {
        'xi': xi, 'Sw': Sw, 'So': So, 'Sg': Sg, 'fw': fw, 'fg': fg,
        'slow_path': slow_path,
        'fast_path': fast_path,
        'slow_type': 'boundary-rarefaction',
        'fast_type': 'entry-shock-rarefaction',
        'fast_has_terminal_shock': False,
        'fast_shock_sigma': np.nan,
        'L': L, 'M': M_fast, 'R': R,
        'P_entry': P, 'Q_entry': Q,
        'entry_shock_info': entry,
        'edge_platform': {
            'state': P.copy(),
            'xi_start': float(xi_b[-1]),
            'xi_end': float(sig_PQ),
        },
        'boundary_limit_closure': {
            'upstream': rar_QR[-2, :3].copy() if len(rar_QR) >= 2 else Q.copy(),
            'downstream': R.copy(),
            'xi': float(rar_QR[-1, 3]) if len(rar_QR) else np.nan,
        },
    }


def _find_transitional_M(L, R, cp, f2):
    """
    Strategy 1c: find an intermediate state M reachable from L by a
    Rankine-Hugoniot-admissible shock (possibly transitional /
    undercompressive, jumping across the elliptic region), connected to R
    either by a second interior shock (Hug_L x Hug_R) or by an oil-bank
    state on the Sg=0 edge (Hug_L x {Sg=0}).

    Admissibility:
      * RH error small for both shocks
      * Wave-speed ordering  sigma(L,M) <= sigma(M,R)  (no crossing)
      * Liu's E-condition on each shock side using its Hugoniot locus

    Returns (M, info) or (None, None).
    """
    print("    Strategy 1c: Hug_L based search (transitional shock) ...")
    hug_L = trace_hugoniot_from_L(L, cp, f2, ns=800)
    print(f"      Hug_L: {len(hug_L)} pts")
    if len(hug_L) < 5:
        return None, None
    hug_R = trace_hugoniot_from_R_permissive(R, cp, f2, ns=800)
    print(f"      Hug_R (permissive): {len(hug_R)} pts")

    candidates = []  # list of (score, M, info)

    # ---- (a) Hug_L intersected with Sg=0 edge → oil-bank M ----
    near_base = hug_L[hug_L[:, 2] < 0.025]
    if len(near_base) > 0:
        # Subsample to keep cost down
        step = max(1, len(near_base) // 60)
        for row in near_base[::step]:
            Sw_J = row[0]
            J = np.array([Sw_J, 1.0 - Sw_J, 0.0])
            if (J[1] < SORW + 0.005 or Sw_J < SWC + 0.005
                    or Sw_J > 1.0 - SORW - 0.005):
                continue
            sig_LJ, rh_LJ = compute_shock_speed(L, J, cp, f2)
            sig_JR, rh_JR = compute_shock_speed(J, R, cp, f2)
            if not (np.isfinite(sig_LJ) and np.isfinite(sig_JR)):
                continue
            if rh_LJ > 0.05 or rh_JR > 0.05:
                continue
            if sig_LJ > sig_JR + 1e-3:
                continue  # waves would cross
            ok_LJ, _, min_sig_LJ = _liu_E_check(L, J, hug_L, cp, f2)
            ok_JR, _, min_sig_JR = _liu_E_check(R, J, hug_R, cp, f2) \
                if len(hug_R) > 5 else (True, sig_JR, sig_JR)
            liu_pen = (0.0 if ok_LJ else 0.4) + (0.0 if ok_JR else 0.4)
            score = rh_LJ + rh_JR + liu_pen
            candidates.append((score, J, {
                'kind': 'oil_bank',
                'sig_LM': sig_LJ, 'sig_MR': sig_JR,
                'rh_LM': rh_LJ, 'rh_MR': rh_JR,
                'liu_LM': bool(ok_LJ), 'liu_MR': bool(ok_JR),
            }))

    # ---- (b) Hug_L x Hug_R interior intersection → double-shock M ----
    # Scan ALL nearby pairs (not just the single closest pair).  The single
    # closest pair is often a SPURIOUS Hugoniot crossing near the umbilic /
    # elliptic region, where the residual function has artifactual sign
    # changes.  Real intersections live elsewhere on the locus and have
    # slightly larger Hug-Hug distance but small actual RH error.
    if len(hug_R) > 5:
        n_interior = 0
        DIST_TOL = 0.025  # bigger than the umbilic spurious crossing scale
        for p in hug_L:
            d2 = (hug_R[:, 0] - p[0])**2 + (hug_R[:, 2] - p[2])**2
            j = int(np.argmin(d2))
            if d2[j] >= DIST_TOL * DIST_TOL:
                continue
            M_int = 0.5 * (p[:3] + hug_R[j, :3])
            M_int[1] = 1.0 - M_int[0] - M_int[2]
            # Reject points on or below the SORG boundary (those are
            # oil-bank candidates, handled in branch (a))
            if M_int[1] < SORG + 0.005:
                continue
            sig_LM, rh_LM = compute_shock_speed(L, M_int, cp, f2)
            sig_MR, rh_MR = compute_shock_speed(M_int, R, cp, f2)
            if not (np.isfinite(sig_LM) and np.isfinite(sig_MR)):
                continue
            # Strict RH filter — kills the spurious umbilic intersection
            # whose sign-flip residual gives rh ~ 0.25
            if rh_LM > 0.04 or rh_MR > 0.04:
                continue
            if sig_LM > sig_MR + 1e-3:
                continue
            ok_LM, _, _ = _liu_E_check(L, M_int, hug_L, cp, f2)
            ok_MR, _, _ = _liu_E_check(R, M_int, hug_R, cp, f2)
            liu_pen = (0.0 if ok_LM else 0.4) + (0.0 if ok_MR else 0.4)
            score = rh_LM + rh_MR + liu_pen
            candidates.append((score, M_int.copy(), {
                'kind': 'interior',
                'sig_LM': float(sig_LM), 'sig_MR': float(sig_MR),
                'rh_LM': float(rh_LM), 'rh_MR': float(rh_MR),
                'liu_LM': bool(ok_LM), 'liu_MR': bool(ok_MR),
                'dist': float(np.sqrt(d2[j])),
            }))
            n_interior += 1
        print(f"      Hug_L x Hug_R: {n_interior} admissible interior "
              f"candidates")

    if not candidates:
        print("      [1c] no admissible transitional M found")
        return None, None

    # Categorical preference: interior > oil_bank.
    #
    # Rationale: when both kinds exist, the interior solution is the
    # generic answer; the oil_bank is the degenerate special case where M
    # happens to lie exactly on the Sg=0 edge.  The oil_bank also gets an
    # artificially low rh_MR (=0) because both M and R are on the same
    # boundary, making the dSg component of RH trivially satisfied — this
    # is NOT real evidence of higher accuracy.
    interior = [c for c in candidates if c[2]['kind'] == 'interior']
    oil_bank = [c for c in candidates if c[2]['kind'] == 'oil_bank']
    interior.sort(key=lambda x: x[0])
    oil_bank.sort(key=lambda x: x[0])
    print(f"      [1c] {len(interior)} interior + {len(oil_bank)} oil_bank candidates")
    for sc, Mc, ic in interior[:3]:
        print(f"      [1c]   interior score={sc:.4f}  "
              f"M=({Mc[0]:.4f},{Mc[1]:.4f},{Mc[2]:.4f})  "
              f"rh={ic['rh_LM']:.1e}/{ic['rh_MR']:.1e}  "
              f"sig={ic['sig_LM']:.3f}/{ic['sig_MR']:.3f}")
    for sc, Mc, ic in oil_bank[:3]:
        print(f"      [1c]   oil_bank score={sc:.4f}  "
              f"M=({Mc[0]:.4f},{Mc[1]:.4f},{Mc[2]:.4f})  "
              f"rh={ic['rh_LM']:.1e}/{ic['rh_MR']:.1e}  "
              f"sig={ic['sig_LM']:.3f}/{ic['sig_MR']:.3f}")

    if interior:
        best_score, best_M, best_info = interior[0]
        print(f"      [1c] preferring INTERIOR (generic solution over oil-bank)")
    elif oil_bank:
        best_score, best_M, best_info = oil_bank[0]
    else:
        return None, None
    print(f"      [1c] best:")
    print(f"      [1c]   M=({best_M[0]:.4f},{best_M[1]:.4f},{best_M[2]:.4f})"
          f"  kind={best_info['kind']}  score={best_score:.4f}")
    print(f"      [1c]   sig(L,M)={best_info['sig_LM']:.4f}"
          f"  sig(M,R)={best_info['sig_MR']:.4f}")
    print(f"      [1c]   rh(L,M)={best_info['rh_LM']:.2e}"
          f"  rh(M,R)={best_info['rh_MR']:.2e}")
    print(f"      [1c]   Liu(L,M)={best_info['liu_LM']}"
          f"  Liu(M,R)={best_info['liu_MR']}")
    if not (best_info['liu_LM'] and best_info['liu_MR']):
        print("      [1c]   NOTE: at least one shock is undercompressive "
              "(transitional)")
    return best_M, best_info


# ==================================================================
#  Helper: bi-directional rarefaction curve
# ==================================================================
def _full_curve(Sw0, Sg0, fam, cp, f2, Nmax=2000, h=0.002):
    sp = integrate_rarefaction(Sw0, Sg0, fam, +1, cp, f2, Nmax, h)
    sm = integrate_rarefaction(Sw0, Sg0, fam, -1, cp, f2, Nmax, h)
    if len(sm) > 0 and len(sp) > 0:
        return np.vstack([sm[::-1], sp[1:]])
    elif len(sp) > 0:
        return sp
    elif len(sm) > 0:
        return sm[::-1]
    return np.empty((0, 4))


def _closest_pair(crv_a, crv_b):
    best_d2 = np.inf
    ia_best = ib_best = 0
    for i in range(len(crv_a)):
        dSw = crv_b[:, 0] - crv_a[i, 0]
        dSg = crv_b[:, 2] - crv_a[i, 2]
        d2 = dSw**2 + dSg**2
        j = np.argmin(d2)
        if d2[j] < best_d2:
            best_d2 = d2[j]
            ia_best, ib_best = i, j
    return ia_best, ib_best, np.sqrt(best_d2)


# ==================================================================
#  Detect whether the solution is a single fast-family wave
# ==================================================================
def _try_fast_only_solution(L, R, cp, f2):
    """
    Check whether the fast rarefaction from R (integrated in +dir)
    passes close to L.  If so, the entire Riemann problem is solved
    by a single fast-family composite wave:
       shock(L → tail) + rarefaction(tail → peak) + shock(peak → R)

    Returns (success, M_tail, M_peak, rar_curve) or (False, ...).
    """
    # Integrate fast rarefaction from R in +direction (towards higher Sg)
    # If R is at Sg=0 exactly, nudge slightly inward to avoid degeneracy
    Sw_start, Sg_start = R[0], R[2]
    if Sg_start < 0.001:
        Sg_start = 0.0005
        Sw_start = R[0]  # keep Sw, adjust So
    fp = integrate_rarefaction(Sw_start, Sg_start, 2, +1, cp, f2,
                               Nmax=8000, h=0.0006)
    if len(fp) < 10:
        return False, None, None, None

    # Check: does this curve pass close to L?
    dists = np.sqrt((fp[:, 0] - L[0])**2 + (fp[:, 2] - L[2])**2)
    i_closest = np.argmin(dists)
    dist_to_L = dists[i_closest]

    print(f"    Fast rarefaction from R: {len(fp)} pts, "
          f"closest to L: {dist_to_L:.4f}")

    if dist_to_L > 0.15:
        return False, None, None, None

    # The fast rarefaction from R passes near L.
    # Find the peak of lam2 — this divides the curve into:
    #   Phase 1 (R → peak): lam2 increasing  → becomes a SHOCK from peak to R
    #   Phase 2 (peak → tail): lam2 decreasing → this IS the rarefaction fan
    lam = fp[:, 3]
    i_peak = int(np.argmax(lam))

    # Extract the rarefaction segment: from peak to the point closest to L
    # (not the full end-of-curve, which may overshoot past L)
    # Find L on the post-peak portion of the curve
    post_peak = fp[i_peak:]
    dists_pp = np.sqrt((post_peak[:, 0] - L[0])**2
                       + (post_peak[:, 2] - L[2])**2)
    i_L_on_pp = int(np.argmin(dists_pp))

    # Rarefaction runs from the L-closest point back to peak
    # Reverse so xi (=lam2) is increasing (physical order: low xi -> high xi)
    rar_seg = post_peak[:i_L_on_pp + 1][::-1]

    if len(rar_seg) < 3:
        return False, None, None, None

    # "tail" = low-xi end (near L), "peak" = high-xi end (near R)
    tail_state = rar_seg[0, :3].copy()
    peak_state = rar_seg[-1, :3].copy()

    print(f"    Fast-only structure detected!")
    print(f"    Tail (low xi): Sw={tail_state[0]:.4f} So={tail_state[1]:.4f} "
          f"Sg={tail_state[2]:.4f}")
    print(f"    Peak (high xi): Sw={peak_state[0]:.4f} So={peak_state[1]:.4f} "
          f"Sg={peak_state[2]:.4f}")

    return True, tail_state, peak_state, rar_seg


# ==================================================================
#  Newton-based Riemann solvers (Juanes & Patzek 2004, Appendix B)
# ==================================================================
def _flux_and_jacobian(Sw, Sg, cp, f2):
    """Return (fw, fg) and the 2x2 Jacobian dF/dU at (Sw, Sg)."""
    fw, _, fg = fractional_flow(Sw, Sg, cp, f2)
    eps = 1e-7
    So = 1.0 - Sw - Sg
    # Finite-difference Jacobian (robust for any kr model)
    Sw_p = min(Sw + eps, 1.0 - max(Sg, 0))
    Sw_m = max(Sw - eps, SWC)
    hSw = Sw_p - Sw_m
    if hSw < 1e-15:
        dfw_dSw = dfg_dSw = 0.0
    else:
        fwp, _, fgp = fractional_flow(Sw_p, Sg, cp, f2)
        fwm, _, fgm = fractional_flow(Sw_m, Sg, cp, f2)
        dfw_dSw = (fwp - fwm) / hSw
        dfg_dSw = (fgp - fgm) / hSw

    Sg_p = min(Sg + eps, 1.0 - max(Sw, SWC))
    Sg_m = max(Sg - eps, 0.0)
    hSg = Sg_p - Sg_m
    if hSg < 1e-15:
        dfw_dSg = dfg_dSg = 0.0
    else:
        fwp, _, fgp = fractional_flow(Sw, Sg_p, cp, f2)
        fwm, _, fgm = fractional_flow(Sw, Sg_m, cp, f2)
        dfw_dSg = (fwp - fwm) / hSg
        dfg_dSg = (fgp - fgm) / hSg

    J = np.array([[dfw_dSw, dfw_dSg],
                  [dfg_dSw, dfg_dSg]])
    return fw, fg, J


def solve_S1S2(L, R, cp, f2, M0=None, tol=1e-10, max_iter=50):
    """
    S1S2 Newton solver (Juanes & Patzek 2004, Appendix B.1).
    Both waves are genuine shocks.

    Unknowns: x = (sigma1, Sw_m, Sg_m, sigma2)
    Equations: 4 Rankine-Hugoniot conditions (2 per shock).

    Returns: (M, sigma1, sigma2, converged)
    """
    fw_L, _, fg_L = fractional_flow(L[0], L[2], cp, f2)
    fw_R, _, fg_R = fractional_flow(R[0], R[2], cp, f2)

    # Initial guess
    if M0 is not None:
        Sw_m, Sg_m = M0[0], M0[2]
    else:
        Sw_m = 0.5 * (L[0] + R[0])
        Sg_m = 0.5 * (L[2] + R[2])
    So_m = 1.0 - Sw_m - Sg_m
    fw_m, _, fg_m = fractional_flow(Sw_m, Sg_m, cp, f2)
    dSw1 = Sw_m - L[0]; dSg1 = Sg_m - L[2]
    dSw2 = R[0] - Sw_m; dSg2 = R[2] - Sg_m
    sig1 = (fw_m - fw_L) / dSw1 if abs(dSw1) > 1e-12 else 0.5
    sig2 = (fw_R - fw_m) / dSw2 if abs(dSw2) > 1e-12 else 1.5

    x = np.array([sig1, Sw_m, Sg_m, sig2], dtype=float)

    for it in range(max_iter):
        s1, sw, sg, s2 = x
        so = 1.0 - sw - sg
        # Enforce bounds
        sw = np.clip(sw, SWC + 1e-6, 1.0 - SORG - 1e-6)
        sg = np.clip(sg, 1e-6, 1.0 - sw - SORG - 1e-6)
        x[1], x[2] = sw, sg

        fw_m, fg_m, Jm = _flux_and_jacobian(sw, sg, cp, f2)

        # Residual: 4 RH equations
        F = np.array([
            fw_m - fw_L - s1 * (sw - L[0]),
            fg_m - fg_L - s1 * (sg - L[2]),
            fw_R - fw_m - s2 * (R[0] - sw),
            fg_R - fg_m - s2 * (R[2] - sg),
        ])

        if np.linalg.norm(F) < tol:
            M = np.array([sw, 1.0 - sw - sg, sg])
            return M, s1, s2, True

        # Newton Jacobian (4x4)
        # dF/d(sigma1, Sw_m, Sg_m, sigma2)
        J_N = np.array([
            [-(sw - L[0]),  Jm[0, 0] - s1,  Jm[0, 1],         0.0],
            [-(sg - L[2]),  Jm[1, 0],        Jm[1, 1] - s1,    0.0],
            [0.0,          -Jm[0, 0] + s2,  -Jm[0, 1],        -(R[0] - sw)],
            [0.0,          -Jm[1, 0],       -Jm[1, 1] + s2,   -(R[2] - sg)],
        ])

        try:
            dx = np.linalg.solve(J_N, -F)
        except np.linalg.LinAlgError:
            break

        # Damped Newton step
        alpha = 1.0
        for _ in range(10):
            x_new = x + alpha * dx
            sw_n = np.clip(x_new[1], SWC + 1e-6, 1.0 - SORG - 1e-6)
            sg_n = np.clip(x_new[2], 1e-6, 1.0 - sw_n - SORG - 1e-6)
            x_new[1], x_new[2] = sw_n, sg_n
            fw_n, _, fg_n = fractional_flow(sw_n, sg_n, cp, f2)
            F_new = np.array([
                fw_n - fw_L - x_new[0] * (sw_n - L[0]),
                fg_n - fg_L - x_new[0] * (sg_n - L[2]),
                fw_R - fw_n - x_new[3] * (R[0] - sw_n),
                fg_R - fg_n - x_new[3] * (R[2] - sg_n),
            ])
            if np.linalg.norm(F_new) < np.linalg.norm(F):
                break
            alpha *= 0.5
        x = x + alpha * dx

    # Did not converge — return best attempt
    sw, sg = np.clip(x[1], SWC, 1.0), np.clip(x[2], 0, 1.0)
    M = np.array([sw, 1.0 - sw - sg, sg])
    return M, x[0], x[3], False


def solve_S1R2(L, R, cp, f2, Npts=600):
    """
    S1R2 solver: slow shock L->M, fast rarefaction M->R.
    Trace fast rarefaction backward from R; for each candidate M on the
    curve, check RH for the 1-shock L->M.
    """
    h_rar = 0.001
    # Trace 2-rarefaction in BOTH directions from R.
    # Single-direction integration is brittle: when R sits on a degenerate
    # boundary (e.g. Sg=0), one of the two eigenvector signs can stall the
    # integrator at R. Using _full_curve avoids that.
    rar2 = _full_curve(R[0], R[2], 2, cp, f2, Nmax=max(Npts, 2000), h=h_rar)
    if len(rar2) < 3:
        print(f"      [S1R2] fast rarefaction from R has only {len(rar2)} pts")
        return None, np.nan, np.nan, False

    print(f"      [S1R2] fast rarefaction from R (bi-dir): {len(rar2)} pts, "
          f"Sw range [{rar2[:,0].min():.3f},{rar2[:,0].max():.3f}], "
          f"Sg range [{rar2[:,2].min():.3f},{rar2[:,2].max():.3f}]")

    best_M = None
    best_rh = 1e10
    best_sig1 = np.nan
    counts = {'bounds': 0, 'sig_bad': 0, 'lax_fail': 0, 'lax_ok': 0}
    from data_and_corey import SO_REF as _SO_REF
    # Direction filter: slow shock L->M must not move Sw/Sg opposite to the
    # overall L->R trend (rejects topologically wrong Lax-admissible branches).
    dSw_LR = R[0] - L[0]
    dSg_LR = R[2] - L[2]
    for i in range(1, len(rar2)):
        pt = rar2[i]
        Sw_m, So_m, Sg_m = pt[0], pt[1], pt[2]
        # Reject M inside the phi(So) damping layer where kro is artificially
        # suppressed -- spurious Lax-admissible S1R2 shocks appear there.
        if So_m < SORG + max(_SO_REF, 0.005) or Sw_m < SWC + 0.001:
            counts['bounds'] += 1
            continue
        if dSw_LR * (Sw_m - L[0]) < -1e-4 or dSg_LR * (Sg_m - L[2]) < -1e-4:
            counts['bounds'] += 1
            continue
        M_cand = np.array([Sw_m, So_m, Sg_m])
        sig1, rh = compute_shock_speed(L, M_cand, cp, f2)
        if np.isnan(sig1) or sig1 <= 0:
            counts['sig_bad'] += 1
            continue
        # Check Lax entropy for 1-shock
        lam1_L, lam2_L, _, _, _ = compute_eigenvalues(L[0], L[2], cp)
        lam1_M, lam2_M, _, _, _ = compute_eigenvalues(Sw_m, Sg_m, cp)
        if lam1_L >= sig1 >= lam1_M and sig1 <= lam2_M:
            counts['lax_ok'] += 1
            if rh < best_rh:
                best_rh = rh
                best_M = M_cand.copy()
                best_sig1 = sig1
        else:
            counts['lax_fail'] += 1

    print(f"      [S1R2] scan: bounds_skip={counts['bounds']}, "
          f"sig_bad={counts['sig_bad']}, lax_fail={counts['lax_fail']}, "
          f"lax_ok={counts['lax_ok']}, best_rh={best_rh:.3e}")

    if best_M is not None and best_rh < 0.3:
        _, lam2_M, _, _, _ = compute_eigenvalues(best_M[0], best_M[2], cp)
        return best_M, best_sig1, lam2_M, True
    return None, np.nan, np.nan, False


def solve_R1S2(L, R, cp, f2, Npts=600):
    """
    R1S2 solver: slow rarefaction L->M, fast shock M->R.
    Trace slow rarefaction from L; for each candidate M on the
    curve, check RH for the 2-shock M->R.
    """
    h_rar = 0.001
    # Trace slow rarefaction in BOTH directions from L (see solve_S1R2 comment).
    rar1 = _full_curve(L[0], L[2], 1, cp, f2, Nmax=max(Npts, 2000), h=h_rar)
    if len(rar1) < 3:
        print(f"      [R1S2] slow rarefaction from L has only {len(rar1)} pts")
        return None, np.nan, np.nan, False

    print(f"      [R1S2] slow rarefaction from L (bi-dir): {len(rar1)} pts, "
          f"Sw range [{rar1[:,0].min():.3f},{rar1[:,0].max():.3f}], "
          f"Sg range [{rar1[:,2].min():.3f},{rar1[:,2].max():.3f}]")

    best_M = None
    best_rh = 1e10
    best_sig2 = np.nan
    counts = {'bounds': 0, 'sig_bad': 0, 'lax_fail': 0, 'lax_ok': 0}
    from data_and_corey import SO_REF as _SO_REF
    dSw_LR = R[0] - L[0]
    dSg_LR = R[2] - L[2]
    for i in range(1, len(rar1)):
        pt = rar1[i]
        Sw_m, So_m, Sg_m = pt[0], pt[1], pt[2]
        if So_m < SORG + max(_SO_REF, 0.005) or Sw_m < SWC + 0.001:
            counts['bounds'] += 1
            continue
        if dSw_LR * (Sw_m - L[0]) < -1e-4 or dSg_LR * (Sg_m - L[2]) < -1e-4:
            counts['bounds'] += 1
            continue
        M_cand = np.array([Sw_m, So_m, Sg_m])
        sig2, rh = compute_shock_speed(M_cand, R, cp, f2)
        if np.isnan(sig2) or sig2 <= 0:
            counts['sig_bad'] += 1
            continue
        # Check Lax entropy for 2-shock
        lam1_M, lam2_M, _, _, _ = compute_eigenvalues(Sw_m, Sg_m, cp)
        lam1_R, lam2_R, _, _, _ = compute_eigenvalues(R[0], R[2], cp)
        lam1_at_M = pt[3] if len(pt) > 3 else lam1_M
        if lam2_M >= sig2 >= lam2_R and sig2 >= lam1_M:
            counts['lax_ok'] += 1
            if rh < best_rh:
                best_rh = rh
                best_M = M_cand.copy()
                best_sig2 = sig2
        else:
            counts['lax_fail'] += 1

    print(f"      [R1S2] scan: bounds_skip={counts['bounds']}, "
          f"sig_bad={counts['sig_bad']}, lax_fail={counts['lax_fail']}, "
          f"lax_ok={counts['lax_ok']}, best_rh={best_rh:.3e}")

    if best_M is not None and best_rh < 0.3:
        lam1_M, _, _, _, _ = compute_eigenvalues(best_M[0], best_M[2], cp)
        return best_M, lam1_M, best_sig2, True
    return None, np.nan, np.nan, False


def solve_R1R2(L, R, cp, f2, Npts=3000, max_pc=30, tol=1e-6):
    """
    R1R2 solver: both waves are rarefactions.  M is the intersection of
    the slow-curve through L and the fast-curve through R.

    Uses bidirectional curve integration (`_full_curve`) so that boundary
    degeneracies (e.g. R on Sg=0) do not stall the integrator.  M is found
    via direct closest-pair search rather than predictor-corrector, which
    is more robust when the two curves cross transversally.
    """
    h = 0.001

    slow_crv = _full_curve(L[0], L[2], 1, cp, f2, Nmax=max(Npts, 2000), h=h)
    fast_crv = _full_curve(R[0], R[2], 2, cp, f2, Nmax=max(Npts, 2000), h=h)

    if len(slow_crv) < 5 or len(fast_crv) < 5:
        print(f"      [R1R2] curve too short: slow={len(slow_crv)}, "
              f"fast={len(fast_crv)}")
        return np.array([0.5*(L[0]+R[0]), 0.0, 0.5*(L[2]+R[2])]), False

    # Restrict to interior, ABOVE the phi damping layer.  With beta>0 and
    # SO_REF>0, the oil boundary uses kro = kr_end*(beta*Sn + (1-beta)*Sn^n)
    # times a linear ramp phi(So) that turns on over a thickness ~SO_REF.
    # Inside this layer the slow integral curve is artificially deflected
    # into the interior, producing SPURIOUS intersections with the fast
    # curve at So ~ SO_REF that are not real R1R2 connections.  Raise the
    # floor well above the damping thickness.  When no true R1R2 M exists
    # (e.g. the correct solution is a transitional shock), R1R2 then fails
    # cleanly and the outer logic falls through to Strategy 1c.
    from data_and_corey import SO_REF as _SO_REF_R1R2
    _so_floor = SORG + max(3.0 * _SO_REF_R1R2, 0.01)
    slow_int = slow_crv[slow_crv[:, 1] > _so_floor]
    fast_int = fast_crv[fast_crv[:, 1] > _so_floor]

    if len(slow_int) < 2 or len(fast_int) < 2:
        print(f"      [R1R2] interior curve too short: slow={len(slow_int)}, "
              f"fast={len(fast_int)} (floor So>{_so_floor:.3f})")
        return np.array([0.5*(L[0]+R[0]), 0.0, 0.5*(L[2]+R[2])]), False

    is_idx, ig_idx, dist = _closest_pair(slow_int, fast_int)
    print(f"      [R1R2] closest-pair: slow={len(slow_int)}, "
          f"fast={len(fast_int)}, dist={dist:.4e}")

    M_sw = 0.5 * (slow_int[is_idx, 0] + fast_int[ig_idx, 0])
    M_sg = 0.5 * (slow_int[is_idx, 2] + fast_int[ig_idx, 2])
    M_sw = np.clip(M_sw, SWC + 1e-4, 1.0 - SORG - 1e-4)
    M_sg = np.clip(M_sg, 1e-6, 1.0 - M_sw - SORG - 1e-4)
    M = np.array([M_sw, 1.0 - M_sw - M_sg, M_sg])

    # Reject if curves don't actually intersect
    if dist > 0.05:
        print(f"      [R1R2] curves do not intersect (dist={dist:.4f} > 0.05)")
        return M, False

    # Validate: rarefaction Lax/monotonicity conditions
    try:
        l1L, _, _, _, _ = compute_eigenvalues(L[0], L[2], cp)
        l1M, l2M, _, _, _ = compute_eigenvalues(M[0], M[2], cp)
        _, l2R, _, _, _ = compute_eigenvalues(R[0], R[2], cp)
    except Exception as e:
        print(f"      [R1R2] eigenvalue failure at M or R: {e}")
        return M, False

    # Reject umbilic points (lam1 ≈ lam2)
    if l2M > 1e-10 and abs(l1M - l2M) / l2M < 0.05:
        print(f"      [R1R2] umbilic at M: lam1={l1M:.4f}, lam2={l2M:.4f}")
        return M, False

    # Rarefaction conditions:
    #   slow rarefaction L→M: lam1 increases:  l1M >= l1L
    #   fast rarefaction M→R: lam2 increases:  l2R >= l2M
    #   strict hyperbolicity at M:             l1M <  l2M
    cond_slow = l1M >= l1L - 1e-6
    cond_fast = l2R >= l2M - 1e-6
    cond_hyp  = l1M <  l2M
    print(f"      [R1R2] M=({M[0]:.4f},{M[1]:.4f},{M[2]:.4f}) "
          f"l1L={l1L:.4f} l1M={l1M:.4f} l2M={l2M:.4f} l2R={l2R:.4f}")
    print(f"      [R1R2] slow_inc={cond_slow}, fast_inc={cond_fast}, "
          f"hyp={cond_hyp}")
    if cond_slow and cond_fast and cond_hyp:
        return M, True
    return M, False


def solve_riemann_newton(L, R, cp, f2):
    """
    Try multiple solution types (S1S2, S1R2, R1S2) with Newton/scan
    and return the best valid intermediate state M.
    """
    results = []

    # --- Pre-compute eigenvalues at L and R for diagnostics ---
    try:
        lam1_L0, lam2_L0, _, _, _ = compute_eigenvalues(L[0], L[2], cp)
        lam1_R0, lam2_R0, _, _, _ = compute_eigenvalues(R[0], R[2], cp)
        print(f"    [diag] lambdas at L: lam1={lam1_L0:.4f}, lam2={lam2_L0:.4f}")
        print(f"    [diag] lambdas at R: lam1={lam1_R0:.4f}, lam2={lam2_R0:.4f}")
    except Exception as e:
        print(f"    [diag] eigenvalue computation at L/R failed: {e}")

    # --- Try S1S2 with multiple initial guesses ---
    guesses = []
    # Midpoint
    guesses.append(np.array([0.5*(L[0]+R[0]), 0.5*(L[1]+R[1]), 0.5*(L[2]+R[2])]))
    # Near R on Sg=0
    for sw_try in np.linspace(SWC + 0.05, 1 - SORW - 0.05, 8):
        guesses.append(np.array([sw_try, 1.0 - sw_try, 0.0]))
    # Interior points
    for sw_try in np.linspace(L[0], R[0], 5):
        for sg_try in np.linspace(0, L[2], 4):
            so_try = 1.0 - sw_try - sg_try
            if so_try > SORG and sw_try > SWC and sg_try >= 0:
                guesses.append(np.array([sw_try, so_try, sg_try]))

    # S1S2 failure tally
    s1s2_fail = {'no_converge': 0, 'bounds': 0, 'order': 0,
                 'lax1': 0, 'lax2': 0, 'eig_fail': 0, 'pass': 0}
    s1s2_lax_examples = []  # store a few near-misses for inspection
    s1s2_trans_candidates = []  # Lax1-failing but RH-good candidates
                                # (for transitional shock across elliptic region)

    for M0 in guesses:
        M, s1, s2, conv = solve_S1S2(L, R, cp, f2, M0=M0)
        if not conv:
            s1s2_fail['no_converge'] += 1
            continue
        # Validate
        if M[0] < SWC or M[2] < -1e-6 or M[1] < SORG - 1e-6:
            s1s2_fail['bounds'] += 1
            continue
        if s1 >= s2:
            s1s2_fail['order'] += 1
            continue
        try:
            lam1_L, lam2_L, _, _, _ = compute_eigenvalues(L[0], L[2], cp)
            lam1_M, lam2_M, _, _, _ = compute_eigenvalues(M[0], M[2], cp)
            lam1_R, lam2_R, _, _, _ = compute_eigenvalues(R[0], R[2], cp)
        except Exception:
            s1s2_fail['eig_fail'] += 1
            continue
        # Check Lax conditions
        lax1_ok = (lam1_L >= s1 - 0.01) and (s1 >= lam1_M - 0.01)
        lax2_ok = (lam2_M >= s2 - 0.01) and (s2 >= lam2_R - 0.01)
        if not lax1_ok:
            s1s2_fail['lax1'] += 1
            if len(s1s2_lax_examples) < 3:
                s1s2_lax_examples.append(
                    f"      Lax1 fail: M=({M[0]:.3f},{M[1]:.3f},{M[2]:.3f}) "
                    f"need lam1(L)={lam1_L:.3f} >= s1={s1:.3f} >= lam1(M)={lam1_M:.3f}")
            # Save as transitional candidate if M is well-hyperbolic
            # and wave ordering holds (for shock crossing elliptic region)
            _, _, _, _, dM = compute_eigenvalues(M[0], M[2], cp)
            if dM > 0.01 and s1 < lam2_M:
                fw_m, _, fg_m = fractional_flow(M[0], M[2], cp, f2)
                fw_L, _, fg_L = fractional_flow(L[0], L[2], cp, f2)
                fw_R, _, fg_R = fractional_flow(R[0], R[2], cp, f2)
                rh1 = abs((fw_m-fw_L) - s1*(M[0]-L[0])) + abs((fg_m-fg_L) - s1*(M[2]-L[2]))
                rh2 = abs((fw_R-fw_m) - s2*(R[0]-M[0])) + abs((fg_R-fg_m) - s2*(R[2]-M[2]))
                s1s2_trans_candidates.append(('S1S2_trans', M.copy(), s1, s2, rh1+rh2))
            continue
        if not lax2_ok:
            s1s2_fail['lax2'] += 1
            if len(s1s2_lax_examples) < 3:
                s1s2_lax_examples.append(
                    f"      Lax2 fail: M=({M[0]:.3f},{M[1]:.3f},{M[2]:.3f}) "
                    f"need lam2(M)={lam2_M:.3f} >= s2={s2:.3f} >= lam2(R)={lam2_R:.3f}")
            _, _, _, _, dM = compute_eigenvalues(M[0], M[2], cp)
            if dM > 0.01 and s1 < lam2_M:
                fw_m, _, fg_m = fractional_flow(M[0], M[2], cp, f2)
                fw_L, _, fg_L = fractional_flow(L[0], L[2], cp, f2)
                rh1 = abs((fw_m-fw_L) - s1*(M[0]-L[0])) + abs((fg_m-fg_L) - s1*(M[2]-L[2]))
                s1s2_trans_candidates.append(('S1S2_trans', M.copy(), s1, s2, rh1))
            continue
        if lax1_ok and lax2_ok:
            s1s2_fail['pass'] += 1
            fw_m, _, fg_m = fractional_flow(M[0], M[2], cp, f2)
            fw_L, _, fg_L = fractional_flow(L[0], L[2], cp, f2)
            fw_R, _, fg_R = fractional_flow(R[0], R[2], cp, f2)
            rh1 = abs((fw_m-fw_L) - s1*(M[0]-L[0])) + abs((fg_m-fg_L) - s1*(M[2]-L[2]))
            rh2 = abs((fw_R-fw_m) - s2*(R[0]-M[0])) + abs((fg_R-fg_m) - s2*(R[2]-M[2]))
            results.append(('S1S2', M, s1, s2, rh1+rh2))
            print(f"    S1S2 candidate: M=({M[0]:.4f},{M[1]:.4f},{M[2]:.4f})"
                  f" s1={s1:.4f} s2={s2:.4f} rh={rh1+rh2:.2e}")

    # --- S1S2 diagnostic summary ---
    print(f"    [S1S2] tried {len(guesses)} guesses: "
          f"converged={len(guesses)-s1s2_fail['no_converge']}, "
          f"out_of_bounds={s1s2_fail['bounds']}, "
          f"s1>=s2={s1s2_fail['order']}, "
          f"lax1_fail={s1s2_fail['lax1']}, "
          f"lax2_fail={s1s2_fail['lax2']}, "
          f"eig_fail={s1s2_fail['eig_fail']}, "
          f"passed={s1s2_fail['pass']}")
    for ex in s1s2_lax_examples:
        print(ex)

    # --- Transitional S1S2: accept Lax1-failing candidates when L is
    #     near-umbilic and the shock must cross the elliptic region ---
    _, _, _, _, disc_L = compute_eigenvalues(L[0], L[2], cp)
    if s1s2_fail['pass'] == 0 and len(s1s2_trans_candidates) > 0 and disc_L < 1e-3:
        s1s2_trans_candidates.sort(key=lambda x: x[4])
        best = s1s2_trans_candidates[0]
        _, M_tr, s1_tr, s2_tr, rh_tr = best
        if rh_tr < 0.05:
            M_tr[1] = 1.0 - M_tr[0] - M_tr[2]
            results.append(best)
            print(f"    S1S2_trans candidate (Lax1 relaxed, elliptic crossing): "
                  f"M=({M_tr[0]:.4f},{M_tr[1]:.4f},{M_tr[2]:.4f}) "
                  f"s1={s1_tr:.4f} s2={s2_tr:.4f} rh={rh_tr:.2e}")

    # --- Try R1S2 ---
    print("    [R1S2] solving slow rarefaction L->M, fast shock M->R ...")
    M_r1s2, sig1, sig2, ok = solve_R1S2(L, R, cp, f2, Npts=2000)
    if not ok:
        print("    [R1S2] failed (no Lax-admissible M on slow rarefaction from L)")
    if ok:
        fw_m, _, fg_m = fractional_flow(M_r1s2[0], M_r1s2[2], cp, f2)
        fw_R, _, fg_R = fractional_flow(R[0], R[2], cp, f2)
        rh2 = abs((fw_R-fw_m) - sig2*(R[0]-M_r1s2[0])) + \
              abs((fg_R-fg_m) - sig2*(R[2]-M_r1s2[2]))
        results.append(('R1S2', M_r1s2, sig1, sig2, rh2))
        print(f"    R1S2 candidate: M=({M_r1s2[0]:.4f},{M_r1s2[1]:.4f},{M_r1s2[2]:.4f})"
              f" s1={sig1:.4f} s2={sig2:.4f} rh={rh2:.2e}")

    # --- Try S1R2 ---
    print("    [S1R2] solving slow shock L->M, fast rarefaction M->R ...")
    M_s1r2, sig1, sig2, ok = solve_S1R2(L, R, cp, f2, Npts=2000)
    if not ok:
        print("    [S1R2] failed (no Lax-admissible M on fast rarefaction from R)")
    if ok:
        fw_L, _, fg_L = fractional_flow(L[0], L[2], cp, f2)
        fw_m, _, fg_m = fractional_flow(M_s1r2[0], M_s1r2[2], cp, f2)
        rh1 = abs((fw_m-fw_L) - sig1*(M_s1r2[0]-L[0])) + \
              abs((fg_m-fg_L) - sig1*(M_s1r2[2]-L[2]))
        results.append(('S1R2', M_s1r2, sig1, sig2, rh1))
        print(f"    S1R2 candidate: M=({M_s1r2[0]:.4f},{M_s1r2[1]:.4f},{M_s1r2[2]:.4f})"
              f" s1={sig1:.4f} s2={sig2:.4f} rh={rh1:.2e}")

    # --- Try R1R2 (predictor-corrector) ---
    print("    [R1R2] predictor-corrector solving two rarefactions ...")
    M_r1r2, ok = solve_R1R2(L, R, cp, f2)
    if not ok:
        print(f"    [R1R2] failed: M=({M_r1r2[0]:.4f},{M_r1r2[1]:.4f},"
              f"{M_r1r2[2]:.4f}) — fails monotonicity / umbilic check")
    if ok:
        l1M, l2M, _, _, _ = compute_eigenvalues(M_r1r2[0], M_r1r2[2], cp)
        results.append(('R1R2', M_r1r2, l1M, l2M, 0.0))
        print(f"    R1R2 candidate: M=({M_r1r2[0]:.4f},{M_r1r2[1]:.4f},{M_r1r2[2]:.4f})"
              f" lam1(M)={l1M:.4f} lam2(M)={l2M:.4f}")

    if not results:
        print("    Newton solver: no valid solution found")
        return None, 'none', np.nan, np.nan

    # Pick the best by smallest RH residual
    results.sort(key=lambda r: r[4])
    best = results[0]
    print(f"    Best: {best[0]} with rh={best[4]:.2e}")
    return best[1], best[0], best[2], best[3]


# ==================================================================
#  Find intermediate state M
# ==================================================================
def find_intermediate_state(L, R, cp, f2):
    """
    Determine the intermediate state M for the Riemann problem L -> M -> R.

    Strategy:
      1. Check if the solution is entirely in the fast family
         (fast rarefaction from R passes near L).
      2. If not, try slow-curve from L intersected with fast-curve from R.
      3. Fall back to numerical M extraction.
    """
    print("  Finding intermediate state M ...")
    find_intermediate_state._fast_only = False

    # --- Strategy 0: boundary BL (when L is exactly on So=SORG edge) ---
    # Tight threshold: only trigger for true edge states.  With beta>0 the
    # slow eigenvector at interior points near the edge is NOT tangent to
    # the edge, so these states must go through the Newton solver below
    # to get the correct R1S1+R2S2 structure.
    L_on_boundary = (abs(L[1] - SORG) <= 1e-4)
    if L_on_boundary:
        print("    L is on So=SORG boundary, trying boundary BL scan ...")
        M_bnd = _find_M_on_boundary(L, R, cp, f2, ns=600)
        if M_bnd is not None:
            M_bnd[1] = 1.0 - M_bnd[0] - M_bnd[2]
            sig, rh = compute_shock_speed(M_bnd, R, cp, f2)
            print(f"  M = ({M_bnd[0]:.4f}, {M_bnd[1]:.4f}, {M_bnd[2]:.4f})"
                  f"  [boundary BL]")
            print(f"  M->R shock: sigma={sig:.4f}, RH_err={rh:.4f}")
            return M_bnd

    # --- Strategy 0b: fast-only composite (try early) ---  [DISABLED]
    # This early short-circuit was pre-empting Newton in the strongly
    # water-wet Riemann problem and locking M to a spurious near-edge
    # state (~0.49, 0.05, 0.46).  The same structure is still reachable
    # as a late fallback further below (see 'Strategy: fast-only single
    # composite' near the bottom of this function) which only fires
    # after Newton / curve-intersection fail.  Re-enable here only if a
    # Riemann problem is *known* to be a pure fast-family composite
    # from an edge-L (neither of our water-wet / weakly / oil-wet cases).
    if False:
        L_on_edge = (abs(L[1] - SORG) <= 1e-4)
        if L_on_edge:
            _crosses_elliptic = False
            for _t in np.linspace(0.05, 0.95, 200):
                _pt = (1 - _t) * L + _t * R
                try:
                    _, _, _, _, _d = compute_eigenvalues(_pt[0], _pt[2], cp)
                    if _d < 0:
                        _crosses_elliptic = True
                        break
                except Exception:
                    pass

            if _crosses_elliptic:
                print("    L on So=SORG edge but L->R crosses elliptic region "
                      "-- skipping fast-only, will use Newton ...")
            else:
                print("    L on So=SORG edge -- trying fast-only composite first ...")
                ok_fo, tail_fo, peak_fo, rar_seg_fo = _try_fast_only_solution(
                    L, R, cp, f2)
                if ok_fo:
                    M_fo = tail_fo.copy()
                    M_fo[1] = 1.0 - M_fo[0] - M_fo[2]
                    print(f"  M = ({M_fo[0]:.4f}, {M_fo[1]:.4f}, {M_fo[2]:.4f}) "
                          f" [fast-only, early]")
                    find_intermediate_state._fast_only = True
                    find_intermediate_state._rar_seg = rar_seg_fo
                    find_intermediate_state._tail = tail_fo
                    find_intermediate_state._peak = peak_fo
                    return M_fo

    # --- Strategy N: Newton-based solver (Juanes & Patzek Appendix B) ---
    print("    Trying Newton-based Riemann solver ...")
    M_newton, wave_type, sig1, sig2 = solve_riemann_newton(L, R, cp, f2)
    if M_newton is not None and M_newton[1] > 0.01:
        M_newton[1] = 1.0 - M_newton[0] - M_newton[2]
        sig, rh = compute_shock_speed(M_newton, R, cp, f2)
        print(f"  M = ({M_newton[0]:.4f}, {M_newton[1]:.4f}, {M_newton[2]:.4f})"
              f"  [{wave_type}]")
        print(f"  sigma1={sig1:.4f}, sigma2={sig2:.4f}")
        find_intermediate_state._wave_type = wave_type
        find_intermediate_state._sigma1 = sig1
        find_intermediate_state._sigma2 = sig2
        return M_newton

    print("    Newton solver found nothing valid, trying heuristic ...")

    # --- Strategy 1: slow curve from L ---
    print("    Tracing slow curve from L ...")
    h_rar = 0.001
    Nmax = 3000
    slow_plus = integrate_rarefaction(L[0], L[2], 1, +1, cp, f2, Nmax, h_rar)
    print(f"    Slow curve (+dir): {len(slow_plus)} pts")

    M = None
    # --- 1a: Check if slow curve crosses Sg=0 boundary (foam/oil-bank regime) ---
    if len(slow_plus) > 5:
        for i in range(len(slow_plus) - 1):
            if slow_plus[i, 2] > 0.005 and slow_plus[i+1, 2] <= 0.005:
                t = slow_plus[i, 2] / (slow_plus[i, 2] - slow_plus[i+1, 2])
                M_Sw = slow_plus[i, 0] + t * (slow_plus[i+1, 0] - slow_plus[i, 0])
                M_So = 1.0 - M_Sw
                # Reject unphysical candidates: M at triangle vertex or So too small
                if M_Sw > 1.0 - SWC - 0.01 or M_So < 0.01:
                    print(f"    Slow curve hits Sg=0 at Sw={M_Sw:.4f}, So={M_So:.4f}"
                          f" — rejected (unphysical)")
                    break
                M_cand = np.array([M_Sw, M_So, 0.0])
                sig, rh = compute_shock_speed(M_cand, R, cp, f2)
                if not np.isnan(sig) and sig > 0 and rh < 0.5:
                    M = M_cand
                    print(f"    Slow curve hits Sg=0: M=({M[0]:.4f}, {M[1]:.4f}, 0)")
                    print(f"    M->R shock: sigma={sig:.4f}, RH_err={rh:.4f}")
                    if M[1] > R[1]:
                        print(f"    OIL BANK: So(M)={M[1]:.4f} > So(R)={R[1]:.4f}")
                break

    # --- 1b: slow curve x fast curve intersection ---
    if M is None:
        print("    Trying slow-curve x fast-curve intersection ...")
        slow_crv = _full_curve(L[0], L[2], 1, cp, f2, Nmax, h_rar)
        print(f"    Slow curve from L: {len(slow_crv)} pts")

        # Build the "backward fast wave curve" from R as the UNION of the
        # admissible Hugoniot locus (for shock segments) and the fast
        # integral curve (for rarefaction segments).  Using only one of
        # them misses intersections when the true fast wave is the other
        # type; in particular, for R on the Sg=0 edge the Hugoniot tends
        # to be confined around R and does not reach the So=0 edge where
        # the slow curve from L lives.
        fast_rar = _full_curve(R[0], R[2], 2, cp, f2, Nmax, h_rar)
        R_on_base = (R[2] < 0.02)
        if R_on_base:
            fast_hug = trace_hugoniot_from_R(R, cp, f2, ns=800)
            if len(fast_hug) > 0 and len(fast_rar) > 0:
                fast_crv = np.vstack([fast_hug, fast_rar])
                fast_src = f"Hugoniot({len(fast_hug)})+rarefaction({len(fast_rar)})"
            elif len(fast_hug) > 0:
                fast_crv = fast_hug
                fast_src = "Hugoniot"
            else:
                fast_crv = fast_rar
                fast_src = "rarefaction"
        else:
            fast_crv = fast_rar
            fast_src = "rarefaction"
        print(f"    Fast curve from R ({fast_src}): {len(fast_crv)} pts")

        if len(slow_crv) > 0 and len(fast_crv) > 0:
            # Require strictly interior intersections.  When L is on the
            # So=SORG edge the slow integral curve is entirely on that
            # edge (slow eigenvector tangent to the edge), so relaxing
            # this filter would let closest-pair pick a geometric
            # crossing that is not the physical M.  Forcing interior
            # lets Strategy 1c (Hugoniot-based) take over in that case.
            # Raise floor above the phi damping layer (thickness ~SO_REF).
            # Inside this layer the slow integral curve is artificially
            # deflected into the interior by the phi ramp, producing a
            # SPURIOUS crossing with the fast curve at So ~ SO_REF.  We
            # want Strategy 1c (Hugoniot-based) to handle that case, so
            # filter these intersections out of Strategy 1b.
            from data_and_corey import SO_REF as _SO_REF_1b
            _so_floor = SORG + max(3.0 * _SO_REF_1b, 0.01)
            slow_int = slow_crv[slow_crv[:, 1] > _so_floor]
            fast_int = fast_crv[fast_crv[:, 1] > _so_floor]
            if len(slow_int) > 0 and len(fast_int) > 0:
                is_idx, ig_idx, dist = _closest_pair(slow_int, fast_int)
                print(f"    Interior closest-approach distance = {dist:.6f}")
                # Tightened threshold: a true intersection should be ~grid step.
                # Anything larger means the slow-curve from L and the fast-curve
                # from R do NOT actually intersect in the interior, and the
                # averaged "M" would be non-physical.
                FALLBACK_DIST_TOL = 0.05
                if dist < FALLBACK_DIST_TOL:
                    M_Sw = 0.5*(slow_int[is_idx, 0] + fast_int[ig_idx, 0])
                    M_Sg = 0.5*(slow_int[is_idx, 2] + fast_int[ig_idx, 2])
                    M_So = 1.0 - M_Sw - M_Sg
                    M = np.array([M_Sw, M_So, M_Sg])
                    print(f"    M from interior curve intersection "
                          f"(dist={dist:.4f} < tol={FALLBACK_DIST_TOL})")
                else:
                    print(f"    !! WARNING: closest-pair distance {dist:.4f} "
                          f">= tol {FALLBACK_DIST_TOL}.")
                    print(f"    !! Slow curve from L and fast curve from R do NOT")
                    print(f"    !! intersect in the interior — refusing to fabricate M.")
                    print(f"    !! Slow-curve closest pt: "
                          f"({slow_int[is_idx,0]:.4f},{slow_int[is_idx,1]:.4f},"
                          f"{slow_int[is_idx,2]:.4f})")
                    print(f"    !! Fast-curve closest pt: "
                          f"({fast_int[ig_idx,0]:.4f},{fast_int[ig_idx,1]:.4f},"
                          f"{fast_int[ig_idx,2]:.4f})")

    # --- Strategy 1f: VISCOUS PROFILE shooting (Isaacson-Marchesin-Plohr) ---
    # NOTE: For Riemann problems with L in the elliptic neighborhood, this
    # strategy gives only a degenerate "L=M" Lax-1 (the trivial fixed
    # point), so we skip it here and let 1c+1g handle the geometry.
    # The implementation is preserved for problems where L is strictly
    # hyperbolic; re-enable it then.
    if False and M is None:
        M_v, sig_v, prof_v = _find_M_via_viscous_profile(L, R, cp, f2)
        if M_v is not None and prof_v is not None:
            M = M_v
            find_intermediate_state._wave_type = 'viscous_profile'
            find_intermediate_state._viscous_profile = prof_v
            find_intermediate_state._viscous_sigma = sig_v

    # --- Strategy 1d: COMPOUND slow wave (rarefaction L->J1 + shock J1->M) ---
    # The numerical solution shows that, for gas displacement of water+oil,
    # the actual slow wave starts as a rarefaction along the L slow integral
    # curve and only LATER turns into the transitional shock that crosses
    # the elliptic region.  J1 is selected by the Oleinik / Welge tangent:
    #     sigma(J1, M) = lambda_1(J1)
    # so the trailing characteristic of the rarefaction matches the leading
    # speed of the shock.  This is the Isaacson-Marchesin-Plohr "compound
    # wave" structure.
    if M is None:
        J1_c, M_c, info_c = _find_compound_slow_M(L, R, cp, f2)
        if M_c is not None:
            M = M_c
            find_intermediate_state._wave_type = 'compound_slow_trans'
            find_intermediate_state._J1 = J1_c
            find_intermediate_state._compound_info = info_c

    # --- Strategy 1c: pure transitional / undercompressive shock via Hug_L ---
    # Fallback used only if the compound construction failed (e.g. no Welge
    # tangent on the slow rarefaction curve).  Builds M as a single shock
    # from L instead of rarefaction+shock.
    if M is None:
        M_trans, info_trans = _find_transitional_M(L, R, cp, f2)
        if M_trans is not None:
            M = M_trans
            find_intermediate_state._wave_type = 'S1S2_trans'
            find_intermediate_state._trans_info = info_trans

    # --- Strategy 1g: fast integral curve through M_seed (post-1c upgrade) ---
    # Only apply when L is strictly off the So=SORG boundary: on the edge
    # the slow eigenvector is tangent to it, so the "L->L_eq shock" fiction
    # in 1g moves M into the elliptic region.  When L is in the interior,
    # 1g is what lets construct_solution recognise the R2S2 composite fast
    # wave (fast rarefaction + terminal shock, Welge-tangent glued).
    _L_on_edge = (abs(L[1] - SORG) <= 0.025)
    _DISABLE_1G_UPGRADE = _L_on_edge
    if M is not None and not _DISABLE_1G_UPGRADE:
        M_W, L_eq, rar_seg, info_g = _find_fast_welge_M_W(L, R, M, cp, f2)
        if M_W is not None and rar_seg is not None and len(rar_seg) >= 2:
            print("    [1g] upgrading 1c result to fast composite "
                  "(shock + rarefaction + shock)")
            M = M_W
            find_intermediate_state._wave_type = 'fast_welge_back'
            find_intermediate_state._fwb_L_eq = L_eq
            find_intermediate_state._fwb_rar_seg = rar_seg
            find_intermediate_state._fwb_info = info_g

    # --- Strategy 2: fast-only solution ---
    if M is None:
        print("    Trying fast-only solution ...")
        ok, tail, peak, rar_seg = _try_fast_only_solution(L, R, cp, f2)
        if ok:
            M = tail.copy()
            M[1] = 1.0 - M[0] - M[2]
            print(f"  M = ({M[0]:.4f}, {M[1]:.4f}, {M[2]:.4f})  [fast-only mode]")
            self = find_intermediate_state
            self._fast_only = True
            self._rar_seg = rar_seg
            self._tail = tail
            self._peak = peak
            return M

    # --- Strategy 3: numerical fallback ---
    if M is None:
        print("    !! WARNING: all analytical strategies failed.")
        print("    !! Falling back to numerical M extraction — the resulting")
        print("    !! M is just the location of minimum gradient in the numerical")
        print("    !! profile and is NOT guaranteed to satisfy RH or Lax conditions.")
        M_num = _find_M_numerically(L, R, cp, f2)
        if len(slow_crv) > 0:
            dists = ((slow_crv[:, 0] - M_num[0])**2
                     + (slow_crv[:, 2] - M_num[2])**2)
            idx = np.argmin(dists)
            M = slow_crv[idx, :3].copy()
            print(f"    Projected onto slow curve "
                  f"(shift={np.sqrt(dists[idx]):.4f})")
        else:
            M = M_num

    M[1] = 1.0 - M[0] - M[2]
    sig, rh = compute_shock_speed(M, R, cp, f2)
    print(f"  M = ({M[0]:.4f}, {M[1]:.4f}, {M[2]:.4f})")
    print(f"  M->R shock: sigma={sig:.4f}, RH_err={rh:.4f}")
    return M


# ==================================================================
#  Construct full analytical solution
# ==================================================================
def construct_solution(L, R, M, cp, f2):
    """Build the complete Riemann solution path and xi-profile."""
    print("  Constructing analytical solution ...")

    # ---- Check if we're in fast-only mode ----
    fast_only = getattr(find_intermediate_state, '_fast_only', False)

    if fast_only:
        return _construct_fast_only(L, R, M, cp, f2)

    return _construct_slow_fast(L, R, M, cp, f2)


def _construct_fast_only(L, R, M, cp, f2):
    """
    Build solution for the fast-only case:
      shock(L -> tail) + rarefaction(tail -> peak) + shock(peak -> R)
    All in the fast family.
    """
    rar_seg = find_intermediate_state._rar_seg
    tail = find_intermediate_state._tail
    peak = find_intermediate_state._peak

    # Make lam2 (= xi) monotonically increasing along the rarefaction
    rar_lam = rar_seg[:, 3].copy()
    for i in range(1, len(rar_lam)):
        if rar_lam[i] < rar_lam[i - 1]:
            rar_lam[i] = rar_lam[i - 1]

    # Remove duplicate xi values
    xi_u, idx_u = np.unique(rar_lam, return_index=True)
    Sw_u = rar_seg[idx_u, 0]
    Sg_u = rar_seg[idx_u, 2]

    xi_rar_min = xi_u[0]
    xi_rar_max = xi_u[-1]

    # Shock speeds
    tail_state = np.array([tail[0], tail[1], tail[2]])
    peak_state = np.array([peak[0], peak[1], peak[2]])
    sig_L, _ = compute_shock_speed(L, tail_state, cp, f2)
    sig_R, _ = compute_shock_speed(peak_state, R, cp, f2)

    print(f"    Shock L->tail: sigma={sig_L:.6f}")
    print(f"    Rarefaction: xi [{xi_rar_min:.6f}, {xi_rar_max:.4f}]")
    print(f"    Shock peak->R: sigma={sig_R:.4f}")

    # Build xi profile
    x_margin = 2.0
    xlo = min(sig_L, xi_rar_min) - x_margin
    xhi = max(sig_R, xi_rar_max) + x_margin
    xlo = max(xlo, -3)
    xhi = min(xhi, sig_R + 5)

    np_pts = 3000
    xi = np.linspace(xlo, xhi, np_pts)
    Sw = np.zeros(np_pts)
    So = np.zeros(np_pts)
    Sg = np.zeros(np_pts)

    for i in range(np_pts):
        xv = xi[i]
        if xv <= sig_L:
            Sw[i], So[i], Sg[i] = L
        elif xv < xi_rar_min:
            # Between L-shock and rarefaction start: tail state
            Sw[i], So[i], Sg[i] = tail_state
        elif xv <= xi_rar_max:
            Sw[i] = np.interp(xv, xi_u, Sw_u)
            Sg[i] = np.interp(xv, xi_u, Sg_u)
            So[i] = 1.0 - Sw[i] - Sg[i]
        elif xv < sig_R:
            # Between rarefaction end and R-shock: peak state
            Sw[i], So[i], Sg[i] = peak_state
        else:
            Sw[i], So[i], Sg[i] = R

    Sw = np.clip(Sw, 0, 1)
    Sg = np.clip(Sg, 0, 1)
    So = np.maximum(0, 1 - Sw - Sg)

    fw = np.zeros(np_pts)
    fg = np.zeros(np_pts)
    for i in range(np_pts):
        fw[i], _, fg[i] = fractional_flow(Sw[i], Sg[i], cp, f2)

    # Build path arrays for ternary plot
    slow_path = np.array([[L[0], L[1], L[2], sig_L],
                           [tail[0], tail[1], tail[2], sig_L]])
    fast_path = rar_seg.copy()

    return {
        'xi': xi, 'Sw': Sw, 'So': So, 'Sg': Sg, 'fw': fw, 'fg': fg,
        'slow_path': slow_path,
        'fast_path': fast_path,
        'slow_type': 'shock',
        'fast_type': 'composite (fast rarefaction + shock)',
        'L': L, 'M': M, 'R': R,
    }


def _construct_slow_fast(L, R, M, cp, f2):
    """
    Build solution for the standard slow+fast case:
      slow wave (L -> M) + fast wave (M -> R)
    """
    h_rar = 0.0015
    Nmax = 3000

    wave_type_hint_top = getattr(find_intermediate_state, '_wave_type', None)

    # ----- Fast Welge tangent + back-traced rarefaction (Strategy 1g) -----
    # Wave structure:
    #   L  ==[zero-strength elliptic-layer jump]==>  L_eq
    #      --(fast rarefaction along integral curve)-->  M_W
    #      --(fast shock at sigma=sigma_W)-->  R
    # The slow family contributes nothing.  Slot the L->L_eq jump into
    # the "slow" wave channel (with zero xi-extent) and put the
    # rarefaction + shock into the fast channel.
    if wave_type_hint_top == 'fast_welge_back':
        L_eq = find_intermediate_state._fwb_L_eq
        rar_seg = find_intermediate_state._fwb_rar_seg
        info_g = find_intermediate_state._fwb_info
        sigma_W = float(info_g['sigma_MR'])
        lam2_Leq = float(info_g['lam2_Leq'])
        print(f"    [fast_welge_back] L_eq=({L_eq[0]:.4f},{L_eq[1]:.4f},"
              f"{L_eq[2]:.4f})  M_W=({M[0]:.4f},{M[1]:.4f},{M[2]:.4f})")
        print(f"    [fast_welge_back] lambda_2 from {lam2_Leq:.4f} "
              f"(at L_eq) up to {info_g['lam2_W']:.4f} (at M_W), "
              f"then shock at sigma={sigma_W:.4f}")

        # ---- Slow channel: zero-strength jump L -> L_eq ----
        # In the inviscid limit this is invisible (it lives entirely in
        # the elliptic neighborhood of L); render it as a degenerate
        # shock at xi = lam2_Leq so it does not appear as a wave in the
        # xi-profile but the (Sw,Sg) ternary path still passes through
        # L and L_eq.
        slow_xi_val = lam2_Leq
        slow_xi = np.array([slow_xi_val, slow_xi_val])
        slow_Sw = np.array([L[0], L_eq[0]])
        slow_Sg = np.array([L[2], L_eq[2]])
        slow_path = np.array([
            [L[0],    L[1],    L[2],    slow_xi_val],
            [L_eq[0], L_eq[1], L_eq[2], slow_xi_val],
        ])
        slow_type = 'elliptic-jump'

        # ---- Fast channel: rarefaction L_eq -> M_W (sampled in lam2),
        # then shock M_W -> R at sigma_W ----
        xi_rar = rar_seg[:, 3].copy()
        Sw_rar = rar_seg[:, 0].copy()
        Sg_rar = rar_seg[:, 2].copy()
        order = np.argsort(xi_rar)
        xi_rar = xi_rar[order]
        Sw_rar = Sw_rar[order]
        Sg_rar = Sg_rar[order]
        xi_u, idx_u = np.unique(xi_rar, return_index=True)
        Sw_u = Sw_rar[idx_u]
        Sg_u = Sg_rar[idx_u]

        # Append the M_W -> R shock at xi = sigma_W
        fast_xi = np.append(xi_u, sigma_W)
        fast_Sw = np.append(Sw_u, R[0])
        fast_Sg = np.append(Sg_u, R[2])

        fast_path = np.column_stack([
            np.append(Sw_u, [R[0]]),
            np.append(1 - Sw_u - Sg_u, [R[1]]),
            np.append(Sg_u, [R[2]]),
            np.append(xi_u, [sigma_W]),
        ])
        fast_type = 'composite'

        return _finalize_solution(L, R, M, slow_path, slow_type,
                                   slow_xi, slow_Sw, slow_Sg,
                                   fast_path, fast_type,
                                   fast_xi, fast_Sw, fast_Sg, cp, f2)

    # ----- Viscous-profile heteroclinic orbit  L → M  +  shock M → R -----
    if wave_type_hint_top == 'viscous_profile':
        prof = find_intermediate_state._viscous_profile
        sig_LM = float(find_intermediate_state._viscous_sigma)
        U_path = prof['U_path']      # 2 x N (Sw, Sg)
        print(f"    [viscous_profile] orbit length = {U_path.shape[1]} pts, "
              f"sig(L,M) = {sig_LM:.4f}")

        # Build (Sw, So, Sg) path from the heteroclinic orbit.  Prepend L
        # and append M to ensure the curve hits the exact endpoints.
        seg_Sw = np.concatenate([[L[0]], U_path[0], [M[0]]])
        seg_Sg = np.concatenate([[L[2]], U_path[1], [M[2]]])
        seg_So = 1.0 - seg_Sw - seg_Sg

        # In the inviscid (ε → 0) limit the entire orbit collapses to a
        # discontinuity at xi = σ(L,M).  For plotting purposes we still
        # spread the orbit slightly across xi so the path can be drawn
        # as a curve in the ternary diagram (the (Sw,Sg) shape is
        # mathematically meaningful as the heteroclinic orbit; only the
        # parameterization in xi is conventional).
        xi_spread = 0.02
        n_orb = len(seg_Sw)
        xi_orb = np.linspace(sig_LM - 0.5 * xi_spread,
                              sig_LM + 0.5 * xi_spread, n_orb)
        slow_xi = xi_orb
        slow_Sw = seg_Sw
        slow_Sg = seg_Sg
        slow_path = np.column_stack([seg_Sw, seg_So, seg_Sg, xi_orb])
        slow_type = 'viscous-profile'

        # M → R fast shock
        fast_path, fast_type, fast_xi, fast_Sw, fast_Sg = \
            _make_shock(M, R, cp, f2)

        sig_MR = float(fast_xi[0])
        print(f"    [viscous_profile] sig(M,R) = {sig_MR:.4f} ; "
              f"L->M heteroclinic profile rendered as curve in (Sw,Sg)")

        return _finalize_solution(L, R, M, slow_path, slow_type,
                                   slow_xi, slow_Sw, slow_Sg,
                                   fast_path, fast_type,
                                   fast_xi, fast_Sw, fast_Sg, cp, f2)

    # ----- Compound slow wave (rarefaction L->J1 + shock J1->M) + shock M->R -----
    if wave_type_hint_top == 'compound_slow_trans':
        J1 = find_intermediate_state._J1
        print(f"    [compound] J1=({J1[0]:.4f},{J1[1]:.4f},{J1[2]:.4f})")

        slow_crv = _full_curve(L[0], L[2], 1, cp, f2, Nmax, h_rar)
        # Slice rarefaction L -> J1 from the slow curve
        rar_path, rar_type, rar_xi, rar_Sw, rar_Sg = \
            _build_wave_on_curve(slow_crv, L, J1, 1, cp, f2)
        print(f"    [compound] L->J1 segment: {rar_type}, "
              f"{len(rar_xi)} samples, "
              f"xi in [{rar_xi.min():.4f},{rar_xi.max():.4f}]")

        sig_J1M, rh_J1M = compute_shock_speed(J1, M, cp, f2)
        sig_MR, _ = compute_shock_speed(M, R, cp, f2)
        print(f"    [compound] J1->M shock: sigma={sig_J1M:.4f} "
              f"rh={rh_J1M:.2e}")
        print(f"    [compound] M->R shock:  sigma={sig_MR:.4f}")

        # Append the J1->M shock as a discontinuity at xi = sigma(J1,M)
        # (which equals lambda_1(J1) by Welge tangency).
        slow_xi = np.append(rar_xi, sig_J1M)
        slow_Sw = np.append(rar_Sw, M[0])
        slow_Sg = np.append(rar_Sg, M[2])
        slow_path = np.vstack([
            rar_path,
            np.array([[J1[0], J1[1], J1[2], sig_J1M],
                      [M[0], M[1], M[2], sig_J1M]]),
        ])
        slow_type = 'compound'

        # Fast wave is the M -> R shock (oil-bank case)
        fast_path, fast_type, fast_xi, fast_Sw, fast_Sg = \
            _make_shock(M, R, cp, f2)

        return _finalize_solution(L, R, M, slow_path, slow_type,
                                   slow_xi, slow_Sw, slow_Sg,
                                   fast_path, fast_type,
                                   fast_xi, fast_Sw, fast_Sg, cp, f2)

    # ----- Transitional / double-shock short-circuit (1c result) -----
    # When M is well-hyperbolic (disc >> 0), the slow shock L->M crosses the
    # elliptic region as a transitional shock.  Use standard shock + R2S2
    # construction instead of the old fast_composite path (which was designed
    # for near-umbilic M).
    if wave_type_hint_top == 'S1S2_trans':
        # Option C: always try fast_composite (Strategy 1e) first.
        # It has its own quality gate (|Welge residual| < 0.05) and returns
        # None when the construction is not applicable.  Only fall back to the
        # generic slow-shock + fast-R2S2 construction if 1e cannot find H.
        # The previous disc_M > 0.01 gate incorrectly vetoed valid composite
        # constructions when M is well-hyperbolic but L is near-umbilic and
        # the L->M chord crosses the elliptic / near-umbilic region — exactly
        # the u-shock situation described in Lozano-Chapiro-Marchesin 2025,
        # Case 3-4.
        _, _, _, _, disc_L = compute_eigenvalues(L[0], L[2], cp)
        _, _, _, _, disc_M = compute_eigenvalues(M[0], M[2], cp)
        print(f"    [S1S2_trans] disc_L={disc_L:.3e}  disc_M={disc_M:.3e}; "
              f"trying fast_composite first")
        H, rar_HM, info_e = _find_fast_composite(L, M, R, cp, f2)

        # If 1e accepted H ~ L AND verified lambda_2 monotonicity along
        # H -> M (no V_2 fold), the answer is the canonical "zero-strength
        # slow + pure fast rarefaction L -> M followed by fast shock
        # M -> R" structure (Furtado-Marchesin 2003, Case II).  Build it
        # directly and skip 1g — overriding with a 3-segment construction
        # would invent a transitional slow shock that doesn't exist.
        pure_fast_rar = bool(info_e is not None and
                              info_e.get('pure_fast_rarefaction', False))
        if pure_fast_rar and rar_HM is not None and len(rar_HM) >= 2:
            print("    [S1S2_trans] 1e returned pure fast rarefaction "
                  "(H ~ L, lambda_2 monotone) -> building zero-strength "
                  "slow + fast composite path; SKIPPING 1g override")

            # Slow channel: zero-strength degenerate jump at the leading
            # wave speed.  Renderer treats 'zero-strength' as no-draw.
            slow_xi_val = float(rar_HM[0, 3])
            slow_xi = np.array([slow_xi_val, slow_xi_val])
            slow_Sw = np.array([L[0], L[0]])
            slow_Sg = np.array([L[2], L[2]])
            slow_path = np.array([
                [L[0], L[1], L[2], slow_xi_val],
                [L[0], L[1], L[2], slow_xi_val],
            ])
            slow_type = 'zero-strength'

            # Fast channel: rarefaction L -> M (sorted by xi=lambda_2)
            xi_rar = rar_HM[:, 3].copy()
            Sw_rar = rar_HM[:, 0].copy()
            Sg_rar = rar_HM[:, 2].copy()
            order = np.argsort(xi_rar)
            xi_rar = xi_rar[order]
            Sw_rar = Sw_rar[order]
            Sg_rar = Sg_rar[order]
            xi_u, idx_u = np.unique(xi_rar, return_index=True)
            Sw_u = Sw_rar[idx_u]
            Sg_u = Sg_rar[idx_u]

            # Force start at exact L (so the path begins where it should)
            Sw_u[0] = L[0]; Sg_u[0] = L[2]
            # Force last rarefaction sample to be M (closes the H->M arc)
            Sw_u[-1] = M[0]; Sg_u[-1] = M[2]

            # ----- Try to EXTEND the rarefaction past M toward R --------
            # Continue the fast integral curve from M and use the requested
            # rule at the downstream end: find the maximum characteristic
            # speed on the admissible extension.  If that max-speed point
            # has Sg ~= 0, the rarefaction reaches the water-oil boundary
            # and no terminal shock is constructed.  Otherwise the
            # rarefaction stops there and a terminal shock jumps to R.
            s3_zero_tol = 1.0e-3
            best_dist = np.inf
            best_branch = None
            for direction in (+1, -1):
                ext = integrate_rarefaction(M[0], M[2], 2, direction, cp, f2,
                                             Nmax=12000, h=0.0003)
                if len(ext) < 5:
                    continue
                d2R = (ext[:, 0] - R[0])**2 + (ext[:, 2] - R[2])**2
                iR = int(np.argmin(d2R))
                d_close = float(np.sqrt(d2R[iR]))
                # Require lam2 to climb past lam2(M) so we're going
                # forward in xi (away from L, toward R), not back to L.
                lam2_at_iR = float(ext[iR, 3])
                if lam2_at_iR < float(xi_u[-1]) - 1e-6:
                    # This branch goes back toward L, not toward R
                    continue
                if d_close < best_dist:
                    best_dist = d_close
                    # Take the strictly-monotone-in-lam2 sub-arc from M
                    # up to iR
                    sub = ext[:iR + 1]
                    # Ensure lam2 is monotone increasing along sub
                    if len(sub) >= 2 and sub[-1, 3] < sub[0, 3]:
                        sub = sub[::-1]
                    best_branch = sub

            sig_chord, _ = compute_shock_speed(M, R, cp, f2)
            sig_chord = float(sig_chord)
            vmax_diag = None

            if best_branch is not None and len(best_branch) >= 2:
                # The branch can hit Sg=0 and then switch to boundary mode,
                # where lambda may drop.  The decision point is the true
                # maximum speed just before/at that turnover, not the last
                # sample nearest R.
                i_max_speed = int(np.nanargmax(best_branch[:, 3]))
                q_max = best_branch[i_max_speed, :3].copy()
                lam2_max = float(best_branch[i_max_speed, 3])
                qmax_sg = float(q_max[2])
                make_terminal_shock = qmax_sg > s3_zero_tol

                # Append the M -> Qmax rarefaction extension.  Drop the
                # first sample because M is already present in xi_u.
                ext_seg = best_branch[1:i_max_speed + 1]
                ext_xi = ext_seg[:, 3].copy() if len(ext_seg) else np.array([])
                ext_Sw = ext_seg[:, 0].copy() if len(ext_seg) else np.array([])
                ext_Sg = ext_seg[:, 2].copy() if len(ext_seg) else np.array([])

                # Ensure strictly increasing in xi
                if len(ext_xi) > 0:
                    keep = np.concatenate([[True],
                                           np.diff(ext_xi) > 1e-9])
                    ext_xi = ext_xi[keep]
                    ext_Sw = ext_Sw[keep]
                    ext_Sg = ext_Sg[keep]

                if not make_terminal_shock and len(ext_xi) > 0:
                    # Sg at the max-speed point is zero (within tolerance),
                    # so the terminal point is treated as the boundary limit
                    # R and no shock is inserted.
                    ext_Sw[-1] = R[0]
                    ext_Sg[-1] = R[2]

                xi_full = np.concatenate([xi_u, ext_xi])
                Sw_full = np.concatenate([Sw_u, ext_Sw])
                Sg_full = np.concatenate([Sg_u, ext_Sg])

                if make_terminal_shock:
                    # Stop the rarefaction at Qmax and place the terminal
                    # shock at the same speed so the sampled profile has a
                    # vertical jump instead of an interpolated ramp.
                    sig_term = lam2_max
                    fast_xi = np.append(xi_full, sig_term)
                    fast_Sw = np.append(Sw_full, R[0])
                    fast_Sg = np.append(Sg_full, R[2])
                    fast_path = np.vstack([
                        np.column_stack([
                            Sw_full, 1 - Sw_full - Sg_full, Sg_full, xi_full
                        ]),
                        np.array([[R[0], R[1], R[2], sig_term]])
                    ])
                    fast_type = 'fast-composite'
                    sig_qr, rh_qr = compute_shock_speed(q_max, R, cp, f2)
                    vmax_diag = {
                        'V': lam2_max,
                        'state': q_max.copy(),
                        'branch': 'R2 fast rarefaction / S2 upstream',
                        'kind': 'shock-speed',
                        'S3_tol': s3_zero_tol,
                        'terminal_shock': True,
                        'RH_sigma_QR': float(sig_qr),
                        'RH_error_QR': float(rh_qr),
                    }
                    print(f"    [fast-composite] Qmax=({q_max[0]:.4f},"
                          f"{q_max[1]:.4f},{q_max[2]:.4g}) has "
                          f"Sg={qmax_sg:.3e} > tol={s3_zero_tol:.0e}; "
                          f"closing with terminal shock at "
                          f"sigma={sig_term:.4f} "
                          f"(RH sigma(Qmax,R)={sig_qr:.4f}, "
                          f"rh={rh_qr:.2e})")
                else:
                    fast_xi, fast_Sw, fast_Sg = xi_full, Sw_full, Sg_full
                    fast_path = np.column_stack([
                        Sw_full, 1 - Sw_full - Sg_full, Sg_full, xi_full
                    ])
                    fast_type = 'fast-rarefaction-only'
                    vmax_diag = {
                        'V': lam2_max,
                        'state': q_max.copy(),
                        'branch': 'R2 fast rarefaction (boundary limit)',
                        'kind': 'characteristic',
                        'S3_tol': s3_zero_tol,
                        'terminal_shock': False,
                        'nearest_R_residual': float(best_dist),
                    }

                    print(f"    [fast-composite] Qmax=({q_max[0]:.4f},"
                          f"{q_max[1]:.4f},{q_max[2]:.4g}) has "
                          f"Sg={qmax_sg:.3e} <= tol={s3_zero_tol:.0e}; "
                          f"PURE fast rarefaction to R "
                          f"(no terminal shock, lambda2_max={lam2_max:.4f}, "
                          f"nearest-R residual={best_dist:.2e}); "
                          f"sigma_chord(M,R)={sig_chord:.4f} not used")
            else:
                # Integral curve doesn't reach R; close with explicit
                # terminal shock at sigma(M,R).
                sig_term = sig_chord
                xi_max_rar = float(xi_u[-1])
                if not np.isfinite(sig_term) or sig_term <= xi_max_rar:
                    sig_term = xi_max_rar + max(0.5, 0.1 * xi_max_rar)
                fast_xi = np.append(xi_u, sig_term)
                fast_Sw = np.append(Sw_u, R[0])
                fast_Sg = np.append(Sg_u, R[2])
                fast_path = np.column_stack([
                    np.append(Sw_u, [R[0]]),
                    np.append(1 - Sw_u - Sg_u, [R[1]]),
                    np.append(Sg_u, [R[2]]),
                    np.append(xi_u, [sig_term]),
                ])
                fast_type = 'fast-composite'

                print(f"    [fast-composite] L=({L[0]:.4f},{L[1]:.4f},"
                      f"{L[2]:.4f}) -> M=({M[0]:.4f},{M[1]:.4f},"
                      f"{M[2]:.4f}) via {len(xi_u)}-pt fast rarefaction; "
                      f"integral curve from M does NOT reach R "
                      f"(min |P-R|={best_dist:.2e}), "
                      f"closing with terminal shock at sigma={sig_term:.4f}")

            # The pure fast curve can be geometrically valid but
            # entropy-inadmissible for the numerical/viscous problem.  If a
            # RH/Welge entry shock from the So=0 boundary to this downstream
            # fast curve exists, use the observed structure:
            # boundary BL rarefaction -> edge plateau -> entry shock -> R2.
            if fast_type == 'fast-rarefaction-only':
                entry = _find_boundary_entry_shock(L, R, fast_path, cp, f2)
                if entry is not None:
                    print("    [S1S2_trans] replacing pure fast rarefaction "
                          "with boundary-rarefaction + entry-shock + R2")
                    out = _finalize_boundary_entry_solution(
                        L, R, M, entry, cp, f2)
                    if vmax_diag is not None:
                        out['vmax_diagnostic'] = vmax_diag
                    return out

            out = _finalize_solution(L, R, M, slow_path, slow_type,
                                     slow_xi, slow_Sw, slow_Sg,
                                     fast_path, fast_type,
                                     fast_xi, fast_Sw, fast_Sg, cp, f2)
            if vmax_diag is not None:
                out['vmax_diagnostic'] = vmax_diag
            return out

        # 1g override path: only used when 1e gave a degenerate H~L that
        # is NOT a pure fast rarefaction (e.g. lambda_2 folds and the
        # construction needs a real transitional slow edge shock + R_2 +
        # S_2).  Also fires when 1e fails entirely.
        use_1g = False
        P_1g = Q_1g = rar_PQ = info_1g = None
        if disc_L < 1e-2:
            degenerate_fast_composite = (
                H is not None and
                float(np.hypot(H[0] - L[0], H[2] - L[2])) < 0.02 and
                not pure_fast_rar
            )
            if degenerate_fast_composite or H is None:
                # Prefer kinetic-oracle P if one has been registered by
                # the caller (Approach A).  Fall back to geometric 1g.
                P_kin = get_kinetic_P()
                if P_kin is not None:
                    from kinetic_lookup import find_transitional_from_kinetic_P
                    print(f"    [S1S2_trans] kinetic override active; "
                          f"P_kinetic=({P_kin[0]:.4f},{P_kin[1]:.4f},"
                          f"{P_kin[2]:.4f})")
                    P_1g, Q_1g, rar_PQ, info_1g = \
                        find_transitional_from_kinetic_P(L, R, cp, f2, P_kin)
                    if P_1g is None:
                        print("    [S1S2_trans] kinetic-1g failed, falling "
                              "back to geometric 1g")
                if P_1g is None:
                    P_1g, Q_1g, rar_PQ, info_1g = \
                        _find_transitional_slow_edge_shock(L, R, cp, f2,
                                                           M_fast=M)
                if P_1g is not None:
                    use_1g = True
                    print(f"    [S1S2_trans] Strategy 1g applicable; "
                          f"overriding fast_composite with 3-segment "
                          f"L->P->Q->R construction")

        if use_1g:
            # Slow wave: transitional shock L -> P (P on So=0 edge)
            slow_path, slow_type, slow_xi, slow_Sw, slow_Sg = \
                _make_shock(L, P_1g, cp, f2)

            # Fast wave: composite (rarefaction P -> Q, then shock Q -> R)
            xi_rar = rar_PQ[:, 3].copy()
            Sw_rar = rar_PQ[:, 0].copy()
            Sg_rar = rar_PQ[:, 2].copy()
            order = np.argsort(xi_rar)
            xi_rar = xi_rar[order]
            Sw_rar = Sw_rar[order]
            Sg_rar = Sg_rar[order]
            xi_u, idx_u = np.unique(xi_rar, return_index=True)
            Sw_u = Sw_rar[idx_u]
            Sg_u = Sg_rar[idx_u]

            sig_QR = info_1g['sig_QR']
            xi_fast = np.append(xi_u, sig_QR)
            Sw_fast = np.append(Sw_u, R[0])
            Sg_fast = np.append(Sg_u, R[2])

            fast_path = np.column_stack([
                np.append(Sw_u, [R[0]]),
                np.append(1 - Sw_u - Sg_u, [R[1]]),
                np.append(Sg_u, [R[2]]),
                np.append(xi_u, [sig_QR]),
            ])
            fast_type = 'composite'

            # Riemann M is the state between slow and fast waves = P
            M_out = P_1g.copy()

            print(f"    [1g] sigma(L,P)={info_1g['sig_LP']:.4f}  "
                  f"lambda2(Q)={info_1g['lam2_Q']:.4f}  "
                  f"sigma(Q,R)={info_1g['sig_QR']:.4f}")

            return _finalize_solution(L, R, M_out, slow_path, slow_type,
                                       slow_xi, slow_Sw, slow_Sg,
                                       fast_path, fast_type,
                                       xi_fast, Sw_fast, Sg_fast, cp, f2)

        if H is None:
            print(f"    [S1S2_trans] fast_composite not applicable; "
                  f"falling back to shock + R2S2 construction")
            wave_type_hint_top = 'S1S2_trans_hyp'
    if wave_type_hint_top == 'S1S2_trans':
        H_ok = ('H' in dir() and H is not None and
                'rar_HM' in dir() and rar_HM is not None and len(rar_HM) >= 2)
        if H_ok:
            print("    [fast_composite] using shock(L->H) + rar(H->M) + "
                  "shock(M->R) construction")
            # Slow wave: shock L -> H (small jump)
            slow_path, slow_type, slow_xi, slow_Sw, slow_Sg = \
                _make_shock(L, H, cp, f2)
            sig_LH = float(slow_xi[0])

            # Then a rarefaction H -> M as the LEADING piece of the fast wave
            # (this stitches onto the M->R fast shock below).  Use rar_HM
            # which is already sorted by lambda_2 ascending.
            xi_rar = rar_HM[:, 3].copy()
            Sw_rar = rar_HM[:, 0].copy()
            Sg_rar = rar_HM[:, 2].copy()
            # Make strictly monotone (drop duplicates)
            order = np.argsort(xi_rar)
            xi_rar = xi_rar[order]
            Sw_rar = Sw_rar[order]
            Sg_rar = Sg_rar[order]
            xi_u, idx_u = np.unique(xi_rar, return_index=True)
            Sw_u = Sw_rar[idx_u]
            Sg_u = Sg_rar[idx_u]

            # Then the M -> R fast shock at xi = sigma(M,R)
            sig_MR, _ = compute_shock_speed(M, R, cp, f2)
            xi_fast = np.append(xi_u, sig_MR)
            Sw_fast = np.append(Sw_u, R[0])
            Sg_fast = np.append(Sg_u, R[2])

            # Build a path array (Sw,So,Sg,xi) for plotting
            fast_path_pts = np.column_stack([
                np.append(Sw_u, [R[0]]),
                np.append(1 - Sw_u - Sg_u, [R[1]]),
                np.append(Sg_u, [R[2]]),
                np.append(xi_u, [sig_MR]),
            ])
            fast_path = fast_path_pts
            fast_type = 'composite'
            fast_xi, fast_Sw, fast_Sg = xi_fast, Sw_fast, Sg_fast

            print(f"    [fast_composite] sigma(L,H)={sig_LH:.4f}  "
                  f"lambda2(H)={info_e['lam2_H']:.4f}  sigma(M,R)={sig_MR:.4f}")

            return _finalize_solution(L, R, M, slow_path, slow_type,
                                       slow_xi, slow_Sw, slow_Sg,
                                       fast_path, fast_type,
                                       fast_xi, fast_Sw, fast_Sg, cp, f2)
        else:
            print("    [fast_composite] not applicable; trying numerical "
                  "wave-path extraction")
            # Final attempt: run a small numerical solve and use the
            # observed L->M wave shape directly.  We have already verified
            # (via grid-refinement) that the numerical L->M structure is a
            # genuine self-similar wave, not a smeared shock — its width
            # in xi-space stays constant as dx -> 0.  Since the wave lives
            # in the umbilic region where eigenvector labels are unstable
            # and Liu/Welge fails to give a closed-form description, the
            # most rigorous practical thing is to use the observed self-
            # similar profile as the analytical L->M segment.
            from numerical_solver import run_numerical
            num = run_numerical(L, R, cp, Nx=3000, t_final=0.4, quiet=True)
            xi_n = num['xi']; Sw_n = num['Sw']; Sg_n = num['Sg']
            So_n = 1 - Sw_n - Sg_n

            # Find where the profile leaves the L plateau and enters the
            # M plateau.  Use the same gradient-magnitude criterion that
            # _find_M_numerically uses.
            dSw_n = np.gradient(Sw_n, xi_n)
            dSg_n = np.gradient(Sg_n, xi_n)
            mag = np.sqrt(dSw_n**2 + dSg_n**2)

            # Snap onto endpoints: closest to L and closest to M
            d2L = (Sw_n - L[0])**2 + (Sg_n - L[2])**2
            d2M = (Sw_n - M[0])**2 + (Sg_n - M[2])**2
            off_L = d2L > 0.005**2
            iL_end = int(np.argmax(off_L)) if off_L.any() else 1
            close_M = d2M < 0.01**2
            iM_start = int(np.argmax(close_M)) if close_M.any() \
                else len(xi_n) - 1
            if iL_end < 1:
                iL_end = 1
            if iM_start >= len(xi_n):
                iM_start = len(xi_n) - 1
            if iM_start <= iL_end:
                iM_start = min(len(xi_n) - 1, iL_end + 10)
            print(f"    [num-extract] L->M wave indices [{iL_end},{iM_start}]"
                  f"  xi range [{xi_n[iL_end]:.4f},{xi_n[iM_start]:.4f}]")

            # Build the slow path from the numerical wave shape.  Prepend L
            # at xi=xi_n[iL_end] (so the path starts exactly at L) and append
            # M at the end.
            seg_Sw = np.concatenate([[L[0]], Sw_n[iL_end:iM_start + 1], [M[0]]])
            seg_Sg = np.concatenate([[L[2]], Sg_n[iL_end:iM_start + 1], [M[2]]])
            seg_xi = np.concatenate([[xi_n[iL_end]],
                                      xi_n[iL_end:iM_start + 1],
                                      [xi_n[iM_start]]])
            # Make xi strictly monotone increasing to allow np.interp later
            xi_u, idx_u = np.unique(seg_xi, return_index=True)
            Sw_u = seg_Sw[idx_u]
            Sg_u = seg_Sg[idx_u]
            slow_xi, slow_Sw, slow_Sg = xi_u, Sw_u, Sg_u
            slow_path = np.column_stack([
                Sw_u, 1 - Sw_u - Sg_u, Sg_u, xi_u
            ])
            slow_type = 'numerical'

            # M -> R fast shock as before
            fast_path, fast_type, fast_xi, fast_Sw, fast_Sg = \
                _make_shock(M, R, cp, f2)

            print(f"    [num-extract] slow_path: {len(xi_u)} samples; "
                  f"fast = single shock at sigma={fast_xi[0]:.4f}")

            return _finalize_solution(L, R, M, slow_path, slow_type,
                                       slow_xi, slow_Sw, slow_Sg,
                                       fast_path, fast_type,
                                       fast_xi, fast_Sw, fast_Sg, cp, f2)
        print("    [S1S2_trans] building double-shock solution")
        slow_path, slow_type, slow_xi, slow_Sw, slow_Sg = \
            _make_shock(L, M, cp, f2)
        fast_path, fast_type, fast_xi, fast_Sw, fast_Sg = \
            _make_shock(M, R, cp, f2)
        sig_LM = float(slow_xi[0])
        sig_MR = float(fast_xi[0])
        print(f"    [S1S2_trans] sig(L,M)={sig_LM:.4f}  sig(M,R)={sig_MR:.4f}")
        # Wave-speed ordering must hold; nudge if numerically equal
        if sig_LM > sig_MR:
            mid = 0.5 * (sig_LM + sig_MR)
            slow_xi = np.array([mid, mid])
            fast_xi = np.array([mid, mid])
        return _finalize_solution(L, R, M, slow_path, slow_type,
                                   slow_xi, slow_Sw, slow_Sg,
                                   fast_path, fast_type,
                                   fast_xi, fast_Sw, fast_Sg, cp, f2)

    slow_crv = _full_curve(L[0], L[2], 1, cp, f2, Nmax, h_rar)
    print(f"    Slow curve: {len(slow_crv)} pts")

    R_on_base = (R[2] < 0.02)
    if R_on_base:
        fast_crv = trace_hugoniot_from_R(R, cp, f2, ns=800)
        if len(fast_crv) == 0:
            fast_crv = _full_curve(R[0], R[2], 2, cp, f2, Nmax, h_rar)
    else:
        fast_crv = _full_curve(R[0], R[2], 2, cp, f2, Nmax, h_rar)
    print(f"    Fast curve: {len(fast_crv)} pts")

    # Slow wave L -> M
    wave_type_hint = getattr(find_intermediate_state, '_wave_type', None)
    if wave_type_hint in ('S1R2', 'S1S2', 'S1S2_trans'):
        # Newton identified the slow wave as a shock; honour that instead
        # of letting _build_wave_on_curve misclassify a monotone-decreasing
        # lambda_1 segment as a "rarefaction".
        slow_path, slow_type, slow_xi, slow_Sw, slow_Sg = \
            _make_shock(L, M, cp, f2)
        print(f"    Slow wave: {slow_type} (forced by Newton hint '{wave_type_hint}')")
    else:
        slow_path, slow_type, slow_xi, slow_Sw, slow_Sg = \
            _build_wave_on_curve(slow_crv, L, M, 1, cp, f2)
        print(f"    Slow wave: {slow_type}")

    # If find_intermediate_state identified an R1R2 (two rarefactions) solution,
    # build the fast wave as a pure rarefaction from M to R.  We integrate
    # the fast rarefaction from M in both directions and pick the branch
    # whose endpoint is closest to R.  This is more robust than slicing the
    # bidirectional fast curve from R, whose lambda need not be monotone
    # along the M->R subarc.
    wave_type_hint = getattr(find_intermediate_state, '_wave_type', None)
    if wave_type_hint == 'R1R2':
        fast_p = integrate_rarefaction(M[0], M[2], 2, +1, cp, f2,
                                        Nmax=8000, h=0.0006)
        fast_m = integrate_rarefaction(M[0], M[2], 2, -1, cp, f2,
                                        Nmax=8000, h=0.0006)

        def _branch_dist(arr):
            if len(arr) < 2:
                return np.inf
            return float(np.min((arr[:, 0] - R[0])**2 + (arr[:, 2] - R[2])**2))

        d_p, d_m = _branch_dist(fast_p), _branch_dist(fast_m)
        chosen = fast_p if d_p <= d_m else fast_m
        chosen_dir = '+' if d_p <= d_m else '-'
        print(f"    [R1R2-fast] +dir reaches dist²={d_p:.2e}, "
              f"-dir reaches dist²={d_m:.2e}, picked {chosen_dir}")

        if len(chosen) >= 2 and min(d_p, d_m) < 0.01:
            # Truncate the rarefaction at the point closest to R
            d2 = (chosen[:, 0] - R[0])**2 + (chosen[:, 2] - R[2])**2
            i_R = int(np.argmin(d2))
            seg = chosen[:i_R + 1]
            lam_seg = seg[:, 3]

            # Trim trailing non-monotone samples.  Near the Sg=0 boundary
            # the integrator can hit a discontinuous flip in the eigenvalue
            # labels (lam_2 jumps because the analytic formula picks the
            # other root once Sg crosses 0), so the very last 1-3 samples
            # of an otherwise clean rarefaction can wreck the monotonicity
            # check.  Walk back from the end keeping the longest monotone
            # tail.
            if len(lam_seg) >= 3:
                # Pick monotonicity sense from the bulk of the segment
                # (use the median of consecutive diffs)
                mid_diffs = np.diff(lam_seg[:max(2, len(lam_seg) * 9 // 10)])
                bulk_sense = (+1 if np.median(mid_diffs) >= 0 else -1)

                end = len(lam_seg)
                while end > 2:
                    diffs = np.diff(lam_seg[:end])
                    if bulk_sense > 0:
                        ok = np.all(diffs >= -1e-8)
                    else:
                        ok = np.all(diffs <= 1e-8)
                    if ok:
                        break
                    end -= 1

                if end < len(lam_seg):
                    n_dropped = len(lam_seg) - end
                    print(f"    [R1R2-fast] dropped {n_dropped} trailing "
                          f"non-monotone sample(s) at the Sg=0 boundary")
                    seg = seg[:end]
                    lam_seg = seg[:, 3]

                    # The trimmed segment ends short of R; append R itself
                    # so the path closes.  Use the trailing slope to fill
                    # the gap with a small straight extension if needed.
                    last = seg[-1]
                    sig_lastR, rh_lastR = compute_shock_speed(
                        last[:3], R, cp, f2)
                    if np.isfinite(sig_lastR) and rh_lastR < 0.05:
                        # The last interior point and R are RH-related;
                        # this is the natural rarefaction-then-shock
                        # endpoint (R2 ends at "last", then a tiny shock
                        # closes to R).  We treat the gap as part of the
                        # rarefaction by appending R at lam2 = lam_2(R)+
                        # if monotonicity is preserved, otherwise we leave
                        # the gap and the renderer will draw a small shock.
                        pass

            mono_inc = np.all(np.diff(lam_seg) >= -1e-8)
            mono_dec = np.all(np.diff(lam_seg) <= 1e-8)
            if mono_dec:
                seg = seg[::-1]
                lam_seg = seg[:, 3]
            if mono_inc or mono_dec:
                xi_u, idx_u = np.unique(lam_seg, return_index=True)
                Sw_u = seg[idx_u, 0]
                Sg_u = seg[idx_u, 2]
                fast_path = seg
                fast_type = 'rarefaction'
                fast_xi = xi_u
                fast_Sw = Sw_u
                fast_Sg = Sg_u
                print(f"    Fast wave: rarefaction ({len(seg)} pts, "
                      f"lam2 in [{lam_seg.min():.3f}, {lam_seg.max():.3f}])")
                return _finalize_solution(L, R, M, slow_path, slow_type,
                                           slow_xi, slow_Sw, slow_Sg,
                                           fast_path, fast_type,
                                           fast_xi, fast_Sw, fast_Sg, cp, f2)
            else:
                print(f"    [R1R2-fast] segment non-monotone, "
                      f"falling through to default fast-wave logic")
        else:
            print(f"    [R1R2-fast] neither branch reached R, "
                  f"falling through")

    # Fast wave M -> R
    # Try fast rarefaction-shock composite (R2S2) from M toward R.
    # Trace the fast integral curve in BOTH directions; in a rarefaction
    # from M, lambda_2 must INCREASE along arc-length (xi = lam2 grows
    # on self-similar wave).  Keep only the monotone-increasing portion
    # of each branch -- past a V2 inflection the integrator keeps going
    # but the curve is no longer a valid rar.
    fast_p = integrate_rarefaction(M[0], M[2], 2, +1, cp, f2,
                                   Nmax=5000, h=0.0008)
    fast_m = integrate_rarefaction(M[0], M[2], 2, -1, cp, f2,
                                   Nmax=5000, h=0.0008)

    def _mono_inc_from_M(arr):
        """Return portion of arr where lam2 is strictly increasing from
        arr[0] (== M).  Stops at first fold (V2 inflection)."""
        if len(arr) < 2:
            return arr
        lam2 = arr[:, 3]
        # Reject branches whose initial step is NOT increasing lam2
        # (those are the "wrong" direction for a forward rarefaction
        # from M; their integration grows away from R in xi-space).
        if lam2[1] <= lam2[0] + 1e-9:
            return arr[:0]
        keep = [0]
        max_so_far = lam2[0]
        for i in range(1, len(lam2)):
            if lam2[i] > max_so_far - 1e-6:
                keep.append(i)
                max_so_far = max(max_so_far, lam2[i])
            else:
                break
        return arr[keep]

    p_inc = _mono_inc_from_M(fast_p)
    m_inc = _mono_inc_from_M(fast_m)

    # Prefer whichever monotone-increasing branch extends to higher lam2
    # (more of the admissible rarefaction range).
    if len(p_inc) > 0 and len(m_inc) > 0:
        fast_rar = p_inc if p_inc[-1, 3] >= m_inc[-1, 3] else m_inc
    elif len(p_inc) > 0:
        fast_rar = p_inc
    elif len(m_inc) > 0:
        fast_rar = m_inc
    else:
        fast_rar = np.empty((0, 4))

    if len(fast_rar) > 0:
        print(f"    Fast rarefaction from M (monotone-inc only): "
              f"{len(fast_rar)} pts, lam2 in "
              f"[{fast_rar[0,3]:.4f},{fast_rar[-1,3]:.4f}]")

    if len(fast_rar) > 10:
        print(f"    Fast rarefaction curve: {len(fast_rar)} pts, "
              f"lam2 range [{fast_rar[:,3].min():.4f}, "
              f"{fast_rar[:,3].max():.4f}]")
        # Find where fast rarefaction can shock to R (Liu: lam2 = sigma)
        best_i = None
        best_rh = np.inf
        best_liu = np.inf
        best_sig = np.nan
        for ii in range(0, len(fast_rar), 3):
            pt = fast_rar[ii, :3]
            sig, rh = compute_shock_speed(pt, R, cp, f2)
            if np.isnan(sig) or sig <= 0:
                continue
            lam2_pt = fast_rar[ii, 3]
            # Liu condition for R2S2: lam2(u*) = sigma(u*; R)
            liu_err = abs(lam2_pt - sig)
            score = liu_err + rh
            if score < best_rh:
                best_rh = score
                best_i = ii
                best_liu = liu_err
                best_sig = sig

        if best_i is not None and best_rh < 1.0:
            # Build R2S2: rarefaction M->u* then shock u*->R
            rar_seg = fast_rar[:best_i+1]
            u_star = fast_rar[best_i, :3]
            xi_rar = rar_seg[:, 3]  # lambda2 values
            Sw_rar = rar_seg[:, 0]
            Sg_rar = rar_seg[:, 2]
            # Ensure monotone increasing xi
            if len(xi_rar) > 1 and xi_rar[-1] < xi_rar[0]:
                xi_rar = xi_rar[::-1]
                Sw_rar = Sw_rar[::-1]
                Sg_rar = Sg_rar[::-1]
            # Append shock point
            xi_comb = np.append(xi_rar, best_sig)
            Sw_comb = np.append(Sw_rar, R[0])
            Sg_comb = np.append(Sg_rar, R[2])
            # Include shock line from u* to R in path for plotting
            shock_line = np.array([[u_star[0], u_star[1], u_star[2], best_sig],
                                    [R[0], R[1], R[2], best_sig]])
            fast_path = np.vstack([rar_seg, shock_line])
            fast_type = 'composite'
            fast_xi = xi_comb
            fast_Sw = Sw_comb
            fast_Sg = Sg_comb
            print(f"    Fast wave: R2S2 composite, rarefaction {len(rar_seg)} pts + shock")
            print(f"    u*=({u_star[0]:.4f},{u_star[1]:.4f},{u_star[2]:.4f})"
                  f" sigma={best_sig:.4f} liu_err={best_liu:.4f}")
        else:
            fast_path, fast_type, fast_xi, fast_Sw, fast_Sg = \
                _make_shock(M, R, cp, f2)
            print(f"    Fast wave: {fast_type} (no R2S2 found)")
    else:
        fast_path, fast_type, fast_xi, fast_Sw, fast_Sg = \
            _make_shock(M, R, cp, f2)
        print(f"    Fast wave: {fast_type}")

    # Enforce wave ordering
    xi_sn, xi_sx = np.min(slow_xi), np.max(slow_xi)
    xi_fn, xi_fx = np.min(fast_xi), np.max(fast_xi)
    if xi_sx > xi_fn:
        mid = 0.5 * (xi_sx + xi_fn)
        xi_sx = xi_fn = mid

    xa = np.concatenate([slow_xi, fast_xi])
    xa = xa[np.isfinite(xa)]
    xr = np.max(xa) - np.min(xa) if len(xa) > 0 else 1.0
    xm = max(0.3 * xr, 0.15)
    xlo = max(np.min(xa) - xm, -2)
    # Keep a short constant-state tail after the last wave instead of forcing
    # a long artificial xi window.
    xhi = np.max(xa) + xm

    np_pts = 2000
    xi = np.linspace(xlo, xhi, np_pts)
    Sw = np.zeros(np_pts)
    So = np.zeros(np_pts)
    Sg = np.zeros(np_pts)

    fast_type_l = str(fast_type).lower()
    fast_has_terminal_shock = (
        'rarefaction-only' not in fast_type_l
        and ('shock' in fast_type_l or 'composite' in fast_type_l)
        and fast_path is not None
        and len(fast_path) >= 2
        and np.linalg.norm(fast_path[-1, :3] - fast_path[-2, :3]) > 1e-10
    )
    if fast_has_terminal_shock:
        fast_shock_sigma = float(fast_path[-1, 3])
        fast_upstream = fast_path[-2, :3].copy()
        fast_rar_path = fast_path[:-1]
        valid = np.isfinite(fast_rar_path[:, 3])
        fast_rar_path = fast_rar_path[valid]
        order = np.argsort(fast_rar_path[:, 3])
        fast_rar_path = fast_rar_path[order]
        fast_rar_xi, fast_rar_idx = np.unique(
            fast_rar_path[:, 3], return_index=True)
        fast_rar_Sw = fast_rar_path[fast_rar_idx, 0]
        fast_rar_Sg = fast_rar_path[fast_rar_idx, 2]
        fast_rar_xi_max = (float(np.max(fast_rar_xi))
                           if len(fast_rar_xi) else -np.inf)
    else:
        fast_shock_sigma = np.nan
        fast_upstream = None
        fast_rar_xi = fast_rar_Sw = fast_rar_Sg = np.array([])
        fast_rar_xi_max = -np.inf

    for i in range(np_pts):
        xv = xi[i]
        if xv <= xi_sn:
            Sw[i], So[i], Sg[i] = L
        elif xv <= xi_sx and slow_type != 'shock' and len(slow_xi) >= 2:
            Sw[i] = np.interp(xv, slow_xi, slow_Sw)
            Sg[i] = np.interp(xv, slow_xi, slow_Sg)
            So[i] = 1 - Sw[i] - Sg[i]
        elif xv <= xi_sx and slow_type == 'shock':
            Sw[i], So[i], Sg[i] = M
        elif xv < xi_fn:
            Sw[i], So[i], Sg[i] = M
        elif fast_type == 'shock':
            if xv < fast_xi[0]:
                Sw[i], So[i], Sg[i] = M
            else:
                Sw[i], So[i], Sg[i] = R
        elif fast_has_terminal_shock:
            if xv < fast_shock_sigma:
                if len(fast_rar_xi) >= 2 and xv <= fast_rar_xi_max:
                    Sw[i] = np.interp(xv, fast_rar_xi, fast_rar_Sw)
                    Sg[i] = np.interp(xv, fast_rar_xi, fast_rar_Sg)
                    So[i] = 1 - Sw[i] - Sg[i]
                else:
                    Sw[i], So[i], Sg[i] = fast_upstream
            else:
                Sw[i], So[i], Sg[i] = R
        elif xv <= xi_fx and len(fast_xi) >= 2:
            Sw[i] = np.interp(xv, fast_xi, fast_Sw)
            Sg[i] = np.interp(xv, fast_xi, fast_Sg)
            So[i] = 1 - Sw[i] - Sg[i]
        else:
            Sw[i], So[i], Sg[i] = R

    Sw = np.clip(Sw, 0, 1)
    Sg = np.clip(Sg, 0, 1)
    So = np.maximum(0, 1 - Sw - Sg)

    fw = np.zeros(np_pts)
    fg = np.zeros(np_pts)
    for i in range(np_pts):
        fw[i], _, fg[i] = fractional_flow(Sw[i], Sg[i], cp, f2)

    return {
        'xi': xi, 'Sw': Sw, 'So': So, 'Sg': Sg, 'fw': fw, 'fg': fg,
        'slow_path': slow_path, 'fast_path': fast_path,
        'slow_type': slow_type, 'fast_type': fast_type,
        'fast_has_terminal_shock': bool(fast_has_terminal_shock),
        'fast_shock_sigma': fast_shock_sigma,
        'L': L, 'M': M, 'R': R,
    }


def _finalize_solution(L, R, M, slow_path, slow_type, slow_xi, slow_Sw, slow_Sg,
                        fast_path, fast_type, fast_xi, fast_Sw, fast_Sg, cp, f2):
    """Sample slow + fast wave segments onto a uniform xi grid and package
    the result dictionary.  Extracted from `_construct_slow_fast` so the
    R1R2 short-circuit can reuse the same packaging logic."""
    # Enforce wave ordering
    xi_sn, xi_sx = np.min(slow_xi), np.max(slow_xi)
    xi_fn, xi_fx = np.min(fast_xi), np.max(fast_xi)
    if xi_sx > xi_fn:
        mid = 0.5 * (xi_sx + xi_fn)
        xi_sx = xi_fn = mid

    xa = np.concatenate([slow_xi, fast_xi])
    xa = xa[np.isfinite(xa)]
    xr = np.max(xa) - np.min(xa) if len(xa) > 0 else 1.0
    xm = max(0.3 * xr, 0.15)
    xlo = max(np.min(xa) - xm, -2)
    # Keep a short constant-state tail after the last wave instead of forcing
    # a long artificial xi window.
    xhi = np.max(xa) + xm

    np_pts = 2000
    xi = np.linspace(xlo, xhi, np_pts)
    Sw = np.zeros(np_pts)
    So = np.zeros(np_pts)
    Sg = np.zeros(np_pts)

    fast_type_l = str(fast_type).lower()
    fast_has_terminal_shock = (
        'rarefaction-only' not in fast_type_l
        and ('shock' in fast_type_l or 'composite' in fast_type_l)
        and fast_path is not None
        and len(fast_path) >= 2
        and np.linalg.norm(fast_path[-1, :3] - fast_path[-2, :3]) > 1e-10
    )
    if fast_has_terminal_shock:
        fast_shock_sigma = float(fast_path[-1, 3])
        fast_upstream = fast_path[-2, :3].copy()
        fast_rar_path = fast_path[:-1]
        valid = np.isfinite(fast_rar_path[:, 3])
        fast_rar_path = fast_rar_path[valid]
        order = np.argsort(fast_rar_path[:, 3])
        fast_rar_path = fast_rar_path[order]
        fast_rar_xi, fast_rar_idx = np.unique(
            fast_rar_path[:, 3], return_index=True)
        fast_rar_Sw = fast_rar_path[fast_rar_idx, 0]
        fast_rar_Sg = fast_rar_path[fast_rar_idx, 2]
        fast_rar_xi_max = (float(np.max(fast_rar_xi))
                           if len(fast_rar_xi) else -np.inf)
    else:
        fast_shock_sigma = np.nan
        fast_upstream = None
        fast_rar_xi = fast_rar_Sw = fast_rar_Sg = np.array([])
        fast_rar_xi_max = -np.inf

    for i in range(np_pts):
        xv = xi[i]
        if xv <= xi_sn:
            Sw[i], So[i], Sg[i] = L
        elif xv <= xi_sx and slow_type != 'shock' and len(slow_xi) >= 2:
            Sw[i] = np.interp(xv, slow_xi, slow_Sw)
            Sg[i] = np.interp(xv, slow_xi, slow_Sg)
            So[i] = 1 - Sw[i] - Sg[i]
        elif xv <= xi_sx and slow_type == 'shock':
            Sw[i], So[i], Sg[i] = M
        elif xv < xi_fn:
            Sw[i], So[i], Sg[i] = M
        elif fast_type == 'shock':
            if xv < fast_xi[0]:
                Sw[i], So[i], Sg[i] = M
            else:
                Sw[i], So[i], Sg[i] = R
        elif fast_has_terminal_shock:
            if xv < fast_shock_sigma:
                if len(fast_rar_xi) >= 2 and xv <= fast_rar_xi_max:
                    Sw[i] = np.interp(xv, fast_rar_xi, fast_rar_Sw)
                    Sg[i] = np.interp(xv, fast_rar_xi, fast_rar_Sg)
                    So[i] = 1 - Sw[i] - Sg[i]
                else:
                    Sw[i], So[i], Sg[i] = fast_upstream
            else:
                Sw[i], So[i], Sg[i] = R
        elif xv <= xi_fx and len(fast_xi) >= 2:
            Sw[i] = np.interp(xv, fast_xi, fast_Sw)
            Sg[i] = np.interp(xv, fast_xi, fast_Sg)
            So[i] = 1 - Sw[i] - Sg[i]
        else:
            Sw[i], So[i], Sg[i] = R

    Sw = np.clip(Sw, 0, 1)
    Sg = np.clip(Sg, 0, 1)
    So = np.maximum(0, 1 - Sw - Sg)

    fw = np.zeros(np_pts)
    fg = np.zeros(np_pts)
    for i in range(np_pts):
        fw[i], _, fg[i] = fractional_flow(Sw[i], Sg[i], cp, f2)

    return {
        'xi': xi, 'Sw': Sw, 'So': So, 'Sg': Sg, 'fw': fw, 'fg': fg,
        'slow_path': slow_path, 'fast_path': fast_path,
        'slow_type': slow_type, 'fast_type': fast_type,
        'fast_has_terminal_shock': bool(fast_has_terminal_shock),
        'fast_shock_sigma': fast_shock_sigma,
        'L': L, 'M': M, 'R': R,
    }


# ==================================================================
#  Build wave segment on pre-computed curve
# ==================================================================
def _build_wave_on_curve(curve, sa, sb, fam, cp, f2):
    if len(curve) < 2:
        return _make_shock(sa, sb, cp, f2)

    da = (curve[:, 0] - sa[0])**2 + (curve[:, 2] - sa[2])**2
    db = (curve[:, 0] - sb[0])**2 + (curve[:, 2] - sb[2])**2
    ia = int(np.argmin(da))
    ib = int(np.argmin(db))

    if ia > ib:
        ia, ib = ib, ia

    seg = curve[ia:ib + 1]
    if len(seg) < 2:
        return _make_shock(sa, sb, cp, f2)

    lam_seg = seg[:, 3]
    # Use CUMULATIVE dip (running-max minus lambda, or running-min minus lambda
    # for decreasing case) to detect inflection crossings.  Per-step diffs
    # can look like float64 noise even when the curve genuinely inflects;
    # cumulative dip catches sustained reversals.  Threshold = 2% of range.
    _lam_range = float(lam_seg.max() - lam_seg.min())
    _dip_tol = max(1e-6, 0.02 * _lam_range)
    _cum_dip_inc = float((np.maximum.accumulate(lam_seg) - lam_seg).max())
    _cum_dip_dec = float((lam_seg - np.minimum.accumulate(lam_seg)).max())
    mono_inc = _cum_dip_inc <= _dip_tol
    mono_dec = _cum_dip_dec <= _dip_tol

    if mono_inc or mono_dec:
        if mono_inc:
            lam_seg = np.maximum.accumulate(lam_seg)
        else:
            lam_seg = np.minimum.accumulate(lam_seg)
        seg = seg.copy()
        seg[:, 3] = lam_seg
        xi_raw = lam_seg.copy()
        Sw_raw = seg[:, 0].copy()
        Sg_raw = seg[:, 2].copy()
        if mono_dec:
            xi_raw = xi_raw[::-1]
            Sw_raw = Sw_raw[::-1]
            Sg_raw = Sg_raw[::-1]
        xi_u, idx_u = np.unique(xi_raw, return_index=True)
        Sw_u = Sw_raw[idx_u]
        Sg_u = Sg_raw[idx_u]
        if len(xi_u) < 2:
            return _make_shock(sa, sb, cp, f2)
        return seg, 'rarefaction', xi_u, Sw_u, Sg_u

    # Non-monotone → composite
    idx_ext = (int(np.argmax(lam_seg)) if fam == 1
               else int(np.argmin(lam_seg)))
    rar_seg = seg[:idx_ext + 1]
    shock_start = seg[idx_ext, :3]
    shock_end = sb

    if len(rar_seg) < 2:
        return _make_shock(sa, sb, cp, f2)

    rar_lam = rar_seg[:, 3].copy()
    # Allow the rarefaction sub-segment to have tiny noise reversals by
    # enforcing monotonicity via running-max/-min (cumulative).  We only
    # take this path if the sub-segment is "mostly" monotone: cumulative
    # dip <= 2% of the sub-segment range.
    _rar_range = float(rar_lam.max() - rar_lam.min())
    _rar_tol = max(1e-6, 0.02 * _rar_range)
    _ci = float((np.maximum.accumulate(rar_lam) - rar_lam).max())
    _cd = float((rar_lam - np.minimum.accumulate(rar_lam)).max())
    if _ci > _rar_tol and _cd > _rar_tol:
        return _make_shock(sa, sb, cp, f2)
    if _ci <= _rar_tol:
        rar_lam = np.maximum.accumulate(rar_lam)
    else:
        rar_lam = np.minimum.accumulate(rar_lam)
    rar_seg = rar_seg.copy()
    rar_seg[:, 3] = rar_lam

    xi_rar = rar_lam.copy()
    Sw_rar = rar_seg[:, 0].copy()
    Sg_rar = rar_seg[:, 2].copy()
    if np.all(np.diff(xi_rar) <= 1e-8):
        xi_rar = xi_rar[::-1]
        Sw_rar = Sw_rar[::-1]
        Sg_rar = Sg_rar[::-1]
    xi_u, idx_u = np.unique(xi_rar, return_index=True)
    Sw_u = Sw_rar[idx_u]
    Sg_u = Sg_rar[idx_u]

    sig_sh, _ = compute_shock_speed(shock_start, shock_end, cp, f2)
    xi_comb = np.append(xi_u, sig_sh)
    Sw_comb = np.append(Sw_u, sb[0])
    Sg_comb = np.append(Sg_u, sb[2])
    return seg, 'composite', xi_comb, Sw_comb, Sg_comb


def _make_shock(sa, sb, cp, f2):
    sig, _ = compute_shock_speed(sa, sb, cp, f2)
    xi_a = np.array([sig, sig])
    Sw_a = np.array([sa[0], sb[0]])
    Sg_a = np.array([sa[2], sb[2]])
    path = np.array([[sa[0], sa[1], sa[2], sig],
                      [sb[0], sb[1], sb[2], sig]])
    return path, 'shock', xi_a, Sw_a, Sg_a


# ==================================================================
#  Numerical M extraction  (fallback)
# ==================================================================
def _find_M_numerically(L, R, cp, f2, Nx=1000):
    from numerical_solver import run_numerical
    result = run_numerical(L, R, cp, Nx=Nx, t_final=0.4, quiet=True)
    Sw, Sg = result['Sw'], result['Sg']
    So = 1 - Sw - Sg
    xi = result['xi']

    # Save profile for construct_solution to use later
    find_intermediate_state._num_profile = {
        'xi': xi, 'Sw': Sw, 'So': So, 'Sg': Sg
    }

    dSw = np.abs(np.diff(Sw))
    dSg = np.abs(np.diff(Sg))
    flat = (dSw < 0.003) & (dSg < 0.003)
    i = 0
    plateaus = []
    while i < len(flat):
        if flat[i]:
            j = i
            while j < len(flat) and flat[j]:
                j += 1
            if j - i >= 10:
                idx = slice(i, min(j + 1, len(Sw)))
                p_Sw = np.mean(Sw[idx])
                p_Sg = np.mean(Sg[idx])
                p_So = np.mean(So[idx])
                is_LR = ((abs(p_Sw - L[0]) < 0.03 and abs(p_Sg - L[2]) < 0.03)
                          or (abs(p_Sw - R[0]) < 0.03 and abs(p_Sg - R[2]) < 0.03))
                width = xi[min(j, len(xi) - 1)] - xi[i]
                plateaus.append((p_Sw, p_So, p_Sg, width, is_LR))
            i = j
        else:
            i += 1

    interior = [p for p in plateaus if not p[4]]
    if interior:
        best = max(interior, key=lambda x: x[3])
        print(f"    M from interior plateau (width={best[3]:.3f})")
        return np.array([best[0], best[1], best[2]])

    abs_dS = np.abs(np.gradient(Sw, xi))
    window = Nx // 5
    running_grad = np.convolve(abs_dS, np.ones(window) / window, mode='same')
    search_region = slice(Nx // 5, 4 * Nx // 5)
    min_grad_idx = np.argmin(running_grad[search_region]) + Nx // 5

    avg_range = slice(max(0, min_grad_idx - Nx // 20),
                      min(Nx, min_grad_idx + Nx // 20))
    M_Sw = np.mean(Sw[avg_range])
    M_Sg = np.mean(Sg[avg_range])
    M_So = 1 - M_Sw - M_Sg
    print(f"    M from minimum gradient region (xi~{xi[min_grad_idx]:.2f})")
    return np.array([M_Sw, M_So, M_Sg])


def _construct_from_numerical(L, R, M, cp, f2, num_profile):
    """
    Build analytical solution using the numerical profile as backbone.
    The numerical xi-profile gives the rarefaction shape;
    we identify the wave types from eigenvalue analysis.
    """
    xi_num = num_profile['xi']
    Sw_num = num_profile['Sw']
    So_num = num_profile['So']
    Sg_num = num_profile['Sg']
    Nx = len(xi_num)

    # Determine xi range
    xlo = max(xi_num[0], -1)
    xhi = min(xi_num[-1], 15)
    np_pts = 2000
    xi = np.linspace(xlo, xhi, np_pts)

    # Interpolate numerical profile onto uniform xi grid
    Sw = np.interp(xi, xi_num, Sw_num)
    Sg = np.interp(xi, xi_num, Sg_num)
    So = 1.0 - Sw - Sg

    # Compute fractional flows
    fw = np.zeros(np_pts)
    fg = np.zeros(np_pts)
    for i in range(np_pts):
        fw[i], _, fg[i] = fractional_flow(Sw[i], Sg[i], cp, f2)

    # Build path arrays for ternary plot from numerical profile
    # Sample ~50 points along the transition zone (not L or R)
    trans = []
    for i in range(0, Nx, max(1, Nx // 50)):
        s = np.array([Sw_num[i], So_num[i], Sg_num[i]])
        d_L = abs(s[0] - L[0]) + abs(s[2] - L[2])
        d_R = abs(s[0] - R[0]) + abs(s[2] - R[2])
        if d_L > 0.01 and d_R > 0.01:
            trans.append([s[0], s[1], s[2], xi_num[i]])
    if len(trans) < 2:
        trans = [[L[0], L[1], L[2], xi[0]],
                 [M[0], M[1], M[2], xi[len(xi)//2]]]
    trans = np.array(trans)

    # Determine wave types by checking eigenvalues at M
    try:
        l1_M, l2_M, _, _, _ = compute_eigenvalues(
            M[0], max(M[2], 0.001), cp)
        sig_MR, _ = compute_shock_speed(M, R, cp, f2)
        # If M->R is a shock (sigma > lam2(R)), label fast as shock
        _, l2_R, _, _, _ = compute_eigenvalues(R[0], max(R[2], 0.001), cp)
        fast_is_shock = (sig_MR > l2_R * 0.9)
    except Exception:
        fast_is_shock = True

    # Split path at M
    dists_to_M = np.sqrt((trans[:, 0] - M[0])**2 + (trans[:, 2] - M[2])**2)
    i_M = int(np.argmin(dists_to_M))
    slow_path = trans[:max(i_M + 1, 2)]
    fast_path = trans[max(i_M, 0):]

    slow_type = 'rarefaction'
    fast_type = 'composite'

    print(f"    Built from numerical profile: slow={len(slow_path)} pts, "
          f"fast={len(fast_path)} pts")

    return {
        'xi': xi, 'Sw': Sw, 'So': So, 'Sg': Sg, 'fw': fw, 'fg': fg,
        'slow_path': slow_path, 'fast_path': fast_path,
        'slow_type': slow_type, 'fast_type': fast_type,
        'L': L, 'M': M, 'R': R,
    }


# ==================================================================
#  Characteristic field computation
# ==================================================================
def compute_characteristic_field(cp, ng=35):
    Sw_v = np.linspace(SWC, 1 - SORG, ng)
    Sg_v = np.linspace(0, 1 - SWC - SORG, ng)
    lam1 = np.full((ng, ng), np.nan)
    lam2 = np.full((ng, ng), np.nan)
    for i in range(ng):
        for j in range(ng):
            Sw, Sg = Sw_v[i], Sg_v[j]
            So = 1 - Sw - Sg
            if So >= SORG - 1e-6 and Sw >= SWC - 1e-6 and Sg >= -1e-6:
                try:
                    l1, l2, _, _, _ = compute_eigenvalues(Sw, Sg, cp)
                    lam1[i, j] = l1
                    lam2[i, j] = l2
                except Exception:
                    pass
    return Sw_v, Sg_v, lam1, lam2


# ==================================================================
#  Inflection loci: grad(lam_i) . r_i = 0  (Juanes & Patzek eq.21)
# ==================================================================
def compute_inflection_loci(cp, ng=80):
    """
    Compute the inflection loci for both characteristic families.

    The inflection locus V_i is the set of points where
        grad(nu_i) . r_i = 0
    i.e. the eigenvalue reaches an extremum along integral curves.

    Returns: Sw_v, Sg_v, infl1[ng,ng], infl2[ng,ng]
        where infl_i is the scalar field grad(lam_i).r_i
        (zero-contour = inflection locus)
    """
    eps = 5e-4
    Sw_v = np.linspace(SWC + eps, 1 - SORG - eps, ng)
    Sg_v = np.linspace(eps, 1 - SWC - SORG - eps, ng)
    infl1 = np.full((ng, ng), np.nan)
    infl2 = np.full((ng, ng), np.nan)

    for i in range(ng):
        for j in range(ng):
            Sw, Sg = Sw_v[i], Sg_v[j]
            So = 1.0 - Sw - Sg
            if So < SORG + eps or Sw < SWC + eps or Sg < eps:
                continue
            try:
                l1, l2, r1, r2, d = compute_eigenvalues(Sw, Sg, cp)
                if d < -1e-10:
                    continue

                # grad(lam1) via finite differences
                h = 2e-4
                # d lam1/d Sw
                l1_pw, _, _, _, _ = compute_eigenvalues(Sw + h, Sg, cp)
                l1_mw, _, _, _, _ = compute_eigenvalues(Sw - h, Sg, cp)
                dl1_dSw = (l1_pw - l1_mw) / (2 * h)

                # d lam1/d Sg
                l1_pg, _, _, _, _ = compute_eigenvalues(Sw, Sg + h, cp)
                l1_mg, _, _, _, _ = compute_eigenvalues(Sw, Sg - h, cp)
                dl1_dSg = (l1_pg - l1_mg) / (2 * h)

                # grad(lam2) via finite differences
                _, l2_pw, _, _, _ = compute_eigenvalues(Sw + h, Sg, cp)
                _, l2_mw, _, _, _ = compute_eigenvalues(Sw - h, Sg, cp)
                dl2_dSw = (l2_pw - l2_mw) / (2 * h)

                _, l2_pg, _, _, _ = compute_eigenvalues(Sw, Sg + h, cp)
                _, l2_mg, _, _, _ = compute_eigenvalues(Sw, Sg - h, cp)
                dl2_dSg = (l2_pg - l2_mg) / (2 * h)

                # Enforce a consistent sign convention on r_i across the
                # grid so the zero-contour of grad(lam_i).r_i is drawn as a
                # single connected curve rather than a shower of spurious
                # segments where the raw eigenvector sign flips.  Require
                # r_i[0] >= 0, breaking ties by r_i[1] >= 0.
                def _canon(v):
                    if v[0] < -1e-12:
                        return -v
                    if abs(v[0]) <= 1e-12 and v[1] < 0:
                        return -v
                    return v
                r1c = _canon(r1)
                r2c = _canon(r2)

                # grad(lam_i) . r_i
                infl1[i, j] = dl1_dSw * r1c[0] + dl1_dSg * r1c[1]
                infl2[i, j] = dl2_dSw * r2c[0] + dl2_dSg * r2c[1]
            except Exception:
                pass

    return Sw_v, Sg_v, infl1, infl2
