# Author: Zhi Zheng
import numpy as np

from baker_model import fractional_flow_vec
from data_and_corey import SORG, SWC


def enforce_admissible_triangle(Sw, Sg):
    Sw = np.clip(Sw, 0.0, 1.0)
    Sg = np.clip(Sg, 0.0, 1.0)
    over = Sw + Sg > 1.0 - SORG
    if np.any(over):
        total = Sw[over] + Sg[over]
        Sw[over] = (1.0 - SORG) * Sw[over] / total
        Sg[over] = (1.0 - SORG) * Sg[over] / total
    return Sw, Sg


def estimate_max_speed(cp, n=70):
    from analytical_solver import compute_eigenvalues

    max_lam = 1.0
    for Sw in np.linspace(SWC, 1.0 - SORG, n):
        for Sg in np.linspace(0.0, 1.0 - SWC - SORG, n):
            if 1.0 - Sw - Sg < SORG - 1e-8:
                continue
            try:
                lam1, lam2, *_ = compute_eigenvalues(Sw, Sg, cp)
                max_lam = max(max_lam, abs(float(lam1)), abs(float(lam2)))
            except Exception:
                pass
    return 1.35 * max_lam


def run_numerical(
    L,
    R,
    cp,
    Nx=10000,
    CFL=0.45,
    t_final=0.40,
    x_max=3.0,
    phi=1.0,
    quiet=False,
    snap_times=None,
    filter_size=41,
):
    dx = x_max / Nx
    x = (np.arange(Nx) + 0.5) * dx
    max_lam = estimate_max_speed(cp)
    dt = CFL * dx / max_lam

    if snap_times is None:
        snap_times = [0.05, 0.10, 0.20, t_final]
    snap_times = sorted(s for s in set(snap_times) if s <= t_final + 1e-12)

    Sw = np.full(Nx, R[0], dtype=float)
    Sg = np.full(Nx, R[2], dtype=float)
    fw_in, fg_in = fractional_flow_vec(np.array([L[0]]), np.array([L[2]]), cp)
    fw_in, fg_in = float(fw_in[0]), float(fg_in[0])

    if not quiet:
        steps = int(np.ceil(t_final / dt))
        print(f"  Numerical grid: Nx={Nx}, x_max={x_max:g}, dt={dt:.3e}, steps~{steps}")

    snaps = []
    snap_i = 0
    t = 0.0
    step = 0
    while t < t_final - 1e-14:
        dt_use = min(dt, t_final - t)
        fw, fg = fractional_flow_vec(Sw, Sg, cp)

        Sw_new = Sw.copy()
        Sg_new = Sg.copy()
        c = dt_use / (dx * phi)
        Sw_new[0] += c * (fw_in - fw[0])
        Sg_new[0] += c * (fg_in - fg[0])
        Sw_new[1:] += c * (fw[:-1] - fw[1:])
        Sg_new[1:] += c * (fg[:-1] - fg[1:])

        Sw, Sg = enforce_admissible_triangle(Sw_new, Sg_new)
        t += dt_use
        step += 1

        if snap_i < len(snap_times) and t >= snap_times[snap_i] - 1e-12:
            snaps.append({"t": t, "Sw": Sw.copy(), "Sg": Sg.copy(), "So": 1.0 - Sw - Sg})
            snap_i += 1
        if not quiet and step % 5000 == 0:
            print(f"    step {step}, t={t:.4f}")

    try:
        from scipy.ndimage import median_filter

        size = min(int(filter_size), Nx if Nx % 2 == 1 else Nx - 1)
        if size >= 3:
            if size % 2 == 0:
                size -= 1
            Sw = median_filter(Sw, size=size)
            Sg = median_filter(Sg, size=size)
    except Exception:
        pass

    So = np.maximum(0.0, 1.0 - Sw - Sg)
    return {
        "x": x,
        "Sw": Sw,
        "So": So,
        "Sg": Sg,
        "xi": x / t_final,
        "t": t_final,
        "snaps": snaps,
        "Nx": Nx,
        "dx": dx,
        "dt": dt,
    }

