# Author: Zhi Zheng
from dataclasses import dataclass


@dataclass(frozen=True)
class CaseConfig:
    name: str
    title: str
    solver_file: str
    left: tuple[float, float, float]
    right: tuple[float, float, float]
    nx: int
    x_max: float
    t_final: float = 0.40
    cfl: float = 0.45
    mu_w: float = 1.0
    mu_o: float = 1.5
    mu_g: float = 0.1
    swc: float = 0.2
    sorw: float = 0.2
    sorg: float = 0.0
    sgc: float = 0.0
    swc_wg: float = 0.2
    nw: float = 2.0
    now: float = 2.0
    krw_end: float = 1.0
    krow_end: float = 1.0
    ng: float = 1.0
    nog: float = 2.0
    krg_end: float = 1.0
    krog_end: float = 1.0
    nw_wg: float = 2.0
    ng_wg: float = 2.0
    krw_wg_end: float = 1.0
    krg_wg_end: float = 1.0
    beta_w: float = 0.05
    beta_o: float = 0.05
    beta_g: float = 0.05
    so_ref: float = 0.05

    @property
    def kro_max(self) -> float:
        return self.krow_end


CASES = {
    "water_wet": CaseConfig(
        name="water_wet",
        title="Water-wet",
        solver_file="analytical_water_wet.py",
        left=(0.30, 0.00, 0.70),
        right=(0.70, 0.30, 0.00),
        nx=6000,
        x_max=30.0,
        sorw=0.30,
        nw=3.0,
        now=1.0,
        krw_end=0.15,
        krow_end=0.90,
        ng=1.0,
        nog=3.0,
        krg_end=0.90,
        krog_end=0.90,
        nw_wg=3.0,
        ng_wg=1.0,
        krw_wg_end=0.15,
        krg_wg_end=0.90,
        so_ref=0.00,
    ),
    "strongly_oil_wet": CaseConfig(
        name="strongly_oil_wet",
        title="Strongly oil-wet",
        solver_file="analytical_strongly_oil_wet.py",
        left=(0.30, 0.00, 0.70),
        right=(0.90, 0.10, 0.00),
        nx=8000,
        x_max=24.0,
        sorw=0.10,
        nw=2.0,
        now=5.0,
        krw_end=0.70,
        krow_end=0.90,
        ng=1.0,
        nog=3.0,
        krg_end=0.90,
        krog_end=0.90,
        nw_wg=3.0,
        ng_wg=3.0,
        krw_wg_end=0.90,
        krg_wg_end=0.10,
        so_ref=0.05,
    ),
    "weakly_water_wet": CaseConfig(
        name="weakly_water_wet",
        title="Weakly water-wet",
        solver_file="analytical_weakly_water_wet.py",
        left=(0.30, 0.00, 0.70),
        right=(0.80, 0.20, 0.00),
        nx=6000,
        x_max=4.1,
        sorw=0.20,
        nw=6.0,
        now=3.0,
        krw_end=0.50,
        krow_end=0.90,
        ng=1.0,
        nog=1.2,
        krg_end=0.90,
        krog_end=0.90,
        nw_wg=3.0,
        ng_wg=1.5,
        krw_wg_end=0.90,
        krg_wg_end=0.80,
        so_ref=0.05,
    ),
}


def get_case(name: str) -> CaseConfig:
    key = name.lower().replace("-", "_")
    if key not in CASES:
        options = ", ".join(CASES)
        raise KeyError(f"Unknown case '{name}'. Choose one of: {options}")
    return CASES[key]
