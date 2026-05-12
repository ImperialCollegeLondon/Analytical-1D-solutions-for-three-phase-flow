# Author: Zhi Zheng

# Three-Phase Saturation Path Prediction

This repository contains research code for predicting saturation paths in
gas-oil-water three-phase flow. The main goal is to support the design and
interpretation of three-phase flow experiments by comparing analytical
Riemann-solution paths with direct numerical simulations.

The code currently focuses on three wettability cases:

- `water_wet`
- `strongly_oil_wet`
- `weakly_water_wet`

The implementation is under active development. Several related functions are
still being developed, and many parts of the workflow still need further
testing, benchmarking, and physical validation. Feedback, questions, and
discussion are very welcome.

Contact: z.zheng25@imperial.ac.uk

## What The Code Does

For each wettability case, the script:

1. Defines Corey-type two-phase relative permeability parameters.
2. Builds Baker-style three-phase relative permeability surfaces.
3. Constructs an analytical saturation path from the left state `L` to the
   right state `R`.
4. Runs a one-dimensional finite-volume numerical simulation.
5. Compares analytical and numerical saturation profiles in self-similar
   coordinates.

The ternary plots include sparse rarefaction-network curves as background
guides for the slow and fast wave families.

## Folder Structure

```text
Share Code/
  run_case.py              Main entry point
  cases.py                 Wettability cases and model parameters
  data_and_corey.py        Corey relative permeability setup
  baker_model.py           Baker three-phase interpolation and fractional flow
  numerical_solver.py      1D numerical solver
  plotting.py              Plotting utilities
  solvers/                 Case-specific analytical solvers
  results/                 Generated figures
  requirements.txt         Python dependencies
```

## Requirements

The code was written for Python 3 and uses:

```text
numpy
scipy
matplotlib
```

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Quick Start

From this folder, run all cases:

```bash
python run_case.py --case all
```

Run a single case:

```bash
python run_case.py --case water_wet
python run_case.py --case strongly_oil_wet
python run_case.py --case weakly_water_wet
```

Print detailed analytical-solver messages:

```bash
python run_case.py --case weakly_water_wet --verbose
```

## Output Files

Outputs are written to:

```text
results/<case>/
```

For each case, four figures are generated:

```text
01_three_phase_relative_permeability.png
02_analytical_ternary_path.png
03_numerical_ternary.png
04_num_vs_analytical_logxi.png
```

Figure meanings:

- `01_three_phase_relative_permeability.png`: Baker three-phase relative
  permeability surfaces for water, oil, and gas.
- `02_analytical_ternary_path.png`: analytical saturation path from `L` to
  `R`, with sparse rarefaction-network background curves.
- `03_numerical_ternary.png`: numerical saturation path in the ternary diagram,
  with analytical path overlay.
- `04_num_vs_analytical_logxi.png`: numerical and analytical saturation
  profiles plotted against `xi = x / t` on a logarithmic axis.

## Numerical Resolution

The default grid sizes in `cases.py` are moderate so the examples can be rerun
without a very long wait. For final figures, increase the numerical resolution:

```bash
python run_case.py --case water_wet --nx 20000
python run_case.py --case strongly_oil_wet --nx 40000
python run_case.py --case weakly_water_wet --nx 20000
```

You can also override the domain length and final time:

```bash
python run_case.py --case weakly_water_wet --nx 20000 --x-max 4.1 --t-final 0.4
```

## Changing Case Parameters

Most user-facing parameters are collected in `cases.py`, including:

- left and right Riemann states `L` and `R`
- residual saturations
- Corey exponents
- endpoint relative permeabilities
- numerical grid size and domain length

After changing a case, rerun:

```bash
python run_case.py --case <case_name>
```

## Development Notes

This is research code rather than a finished simulation package. In particular:

- the analytical construction is still being tested for broader parameter
  ranges;
- some non-classical or transitional wave structures may need additional
  validation;
- numerical resolution can affect sharp shocks and small transition regions;
- the code is currently organized for clarity and reproducibility of the three
  shared cases, not for a fully general user interface.

Please treat generated figures as modelling predictions that should be checked
against physical assumptions, numerical convergence, and experimental context.

