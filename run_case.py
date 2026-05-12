# Author: Zhi Zheng
import argparse
import contextlib
import importlib
import importlib.util
import os
from pathlib import Path
import sys
import time

import numpy as np

from cases import CASES, get_case


ROOT = Path(__file__).resolve().parent


def _clear_case_modules():
    for name in [
        "data_and_corey",
        "baker_model",
        "numerical_solver",
        "plotting",
        "analytical_solver",
    ]:
        sys.modules.pop(name, None)


def _load_backend(case):
    path = ROOT / "solvers" / case.solver_file
    spec = importlib.util.spec_from_file_location("analytical_solver", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["analytical_solver"] = module
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def _maybe_log(path, verbose):
    if verbose:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as f, contextlib.redirect_stdout(f):
        yield


def run_single(case_name, args):
    case = get_case(case_name)
    os.environ["THREE_PHASE_CASE"] = case.name
    _clear_case_modules()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    data = importlib.import_module("data_and_corey")
    analytical = _load_backend(case)
    numerical = importlib.import_module("numerical_solver")
    plotting = importlib.import_module("plotting")

    cp = data.CoreyParams()
    f2 = data.make_two_phase_funcs(cp)
    L = np.asarray(case.left, dtype=float)
    R = np.asarray(case.right, dtype=float)
    outdir = ROOT / "results" / case.name
    outdir.mkdir(parents=True, exist_ok=True)

    nx = args.nx if args.nx is not None else case.nx
    x_max = args.x_max if args.x_max is not None else case.x_max
    t_final = args.t_final if args.t_final is not None else case.t_final

    print(f"\n[{case.title}]")
    print(f"  L = ({L[0]:.3f}, {L[1]:.3f}, {L[2]:.3f})")
    print(f"  R = ({R[0]:.3f}, {R[1]:.3f}, {R[2]:.3f})")
    print(f"  {cp.summary()}")

    log_path = outdir / "solver_log.txt"
    t0 = time.time()
    print("  analytical solution ...")
    with _maybe_log(log_path, args.verbose):
        M = analytical.find_intermediate_state(L, R, cp, f2)
        sol = analytical.construct_solution(L, R, M, cp, f2)

    print(f"  numerical solution: Nx={nx}, x_max={x_max:g}, t={t_final:g} ...")
    with _maybe_log(log_path, args.verbose):
        num = numerical.run_numerical(
            L,
            R,
            cp,
            Nx=nx,
            CFL=case.cfl,
            t_final=t_final,
            x_max=x_max,
            quiet=not args.verbose,
        )

    print("  plotting ...")
    plotting.plot_three_phase_relative_permeability(
        cp, f2, outdir / "01_three_phase_relative_permeability.png"
    )
    plotting.plot_analytical_ternary_path(
        sol, outdir / "02_analytical_ternary_path.png", cp=cp, f2=f2
    )
    plotting.plot_numerical_ternary(
        num, sol, outdir / "03_numerical_ternary.png", cp=cp, f2=f2
    )
    plotting.plot_num_vs_analytical_logxi(
        num, sol, outdir / "04_num_vs_analytical_logxi.png"
    )

    elapsed = time.time() - t0
    M_out = np.asarray(sol.get("M", M), dtype=float)
    print(f"  M = ({M_out[0]:.4f}, {M_out[1]:.4f}, {M_out[2]:.4f})")
    print(f"  waves = {sol.get('slow_type')} | {sol.get('fast_type')}")
    print(f"  saved to {outdir}")
    print(f"  done in {elapsed:.1f} s")


def parse_args():
    parser = argparse.ArgumentParser(description="Run three-phase flow wetting cases.")
    parser.add_argument(
        "--case",
        default="all",
        choices=["all", *CASES.keys()],
        help="case to run",
    )
    parser.add_argument("--nx", type=int, default=None, help="override grid cells")
    parser.add_argument("--x-max", type=float, default=None, help="override domain length")
    parser.add_argument("--t-final", type=float, default=None, help="override final time")
    parser.add_argument("--verbose", action="store_true", help="print backend solver details")
    return parser.parse_args()


def main():
    args = parse_args()
    names = CASES.keys() if args.case == "all" else [args.case]
    for name in names:
        run_single(name, args)


if __name__ == "__main__":
    main()
