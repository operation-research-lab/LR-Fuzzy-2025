# LR-Fuzzy-2025
# An O(1) Direct Algorithm for Comparing LR Fuzzy Numbers with Fixed-Form Parametric Shape Functions

This repository contains the complete reproducibility package for the paper:

> **"An O(1) direct algorithm for comparing LR fuzzy numbers with fixed-form parametric shape functions"**
>
> Sahar Rahdar, Reza Ghanbari, Khatere Ghorbani-Moghadam

The proposed method approximates the integral terms in Ghanbari et al.'s modified Kerre comparison measure using Simpson's rule, yielding a set of direct case-based formulas with `O(1)` time complexity under fixed-form parametric assumptions.

---

## 📌 Overview

The main contributions of this repository are:

- **Proposed method** (`Proposed_method.py`): Implementation of the Simpson-based closed-form formulas (Theorem 3) for comparing Gaussian fuzzy numbers represented in LR form.
- **Reference method** (`Ghanbari_method.py`): Implementation of the integral-based reference method (Theorem 2).
- **Gaussian reference** (`Calculate_rG.py`): Reference computation of `r_G` on the original (non-approximated) Gaussian membership functions.
- **Shared utilities** (`fuzzy_utils.py`): Lagrange interpolation, cubic spline fitting, and one-step Newton–Raphson intersection computation.
- **Intersection reference** (`x_bar_reference.py`): High-accuracy Brent-based reference for validating the one-step Newton–Raphson intersection computation.
- **Shape families** (`shape_family_*.py`): Implementations of the four LR shape families — linear, quadratic, cubic, and quintic.
- **Alternative ranking methods** (`Yager.py`, `Liou_Wang.py`): Implementations of Yager's level-set valuation functional and Liou-Wang's integral-value ranking method.

---

## 🖥️ Computational Environment

To ensure reproducibility, all experiments reported in the paper were performed under the following environment (Section 4.3):

| Component | Value |
|-----------|-------|
| **OS** | Windows 11 Pro |
| **CPU** | AMD Ryzen 5 5600H with Radeon Graphics (3.3 GHz) |
| **RAM** | 16 GB |
| **Python** | 3.11.0 |

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/operation-research-lab/LR-Fuzzy-2025.git
cd LR-Fuzzy-2025
```

### 2. (Optional) Create and activate a virtual environment

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Linux / macOS:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Reproducing the Results

All experiments use the random seed `12`, as reported in the paper. Each script must be executed from the project root directory. Output files (CSV tables and PNG figures) will be automatically saved under the `results/` directory.

### Table 1 — Newton–Raphson vs. Brent intersection

Compares the one-step Newton–Raphson refinement against a high-accuracy Brent-based reference on 10,000 overlapping Gaussian pairs.

```bash
python x_bar_comparison.py --num_pairs 10000 --seed 12
```

**Output:** `results/tables/x_bar_comparison.csv`

---

### Tables 2 & 3 — Execution time and accuracy statistics (Gaussian)

Runs the proposed method and Ghanbari et al.'s reference method on 10,000 Gaussian test pairs, and produces the accuracy statistics and distribution plots.

```bash
python run_experiments.py --num_pairs 10000 --seed 12
python analyze_results.py
python check_ordering_signs.py
```

**Outputs:**
- `results/tables/experiment_results.csv`
- `results/tables/abs_error_stats.csv`
- `results/tables/rel_error_stats.csv`
- `results/tables/sign_mismatches.csv`
- `results/figures/absolute_error_histogram.png`
- `results/figures/relative_error_histogram.png`
- `results/figures/relative_error_ecdf.png`

---

### Table 4 — Empirical decomposition of the approximation error

Decomposes the total error into the representation component `E_rep` and the Simpson component `e_S`.

```bash
python error_decomposition.py --num_pairs 10000 --seed 12
```

**Output:** `results/tables/error_decomposition.csv`

---

### Table 5 — Validation on additional LR shape-function families

Compares the proposed method with Ghanbari et al.'s method for linear, quadratic, cubic, and quintic shape functions.

```bash
python run_shape_family_experiments.py --num_pairs 10000 --seed 12
```

**Output:** `results/tables/shape_family_results.csv`

---

### Table 6 — Ordering agreement with alternative ranking methods

Compares the ordering decisions of the proposed method with Yager's and Liou-Wang's ranking approaches.

```bash
python compare_ranking_methods.py --num_pairs 10000 --seed 12
```

**Output:** `results/tables/ranking_comparison.csv`

---

## 📂 Repository Structure

```text
LR-Fuzzy-2025/
│
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── .gitignore                             # Files ignored by Git
├── LICENSE                                # MIT License
│
├── # ── Core implementations ──────────────────────────────────
├── Proposed_method.py                     # Proposed Simpson-based formulas (Theorem 3)
├── Ghanbari_method.py                     # Reference integral-based method (Theorem 2)
├── Calculate_rG.py                        # Reference computation of r_G on original Gaussians
├── fuzzy_utils.py                         # Shared utilities: Lagrange interpolation,
│                                          #   spline fitting, one-step Newton–Raphson
├── x_bar_reference.py                     # Brent-based reference for intersection point
│
├── # ── Shape families (Section 4.5) ──────────────────────────
├── shape_family_linear.py                 # Linear shape function S1
├── shape_family_quadratic.py              # Quadratic shape function S2
├── shape_family_cubic.py                  # Cubic shape function S3
├── shape_family_Quintic.py                # Quintic shape function S5
│
├── # ── Alternative ranking methods (Section 4.6) ──────────────
├── Yager.py                               # Yager's level-set valuation functional
├── Liou_Wang.py                           # Liou-Wang's integral-value ranking method
│
├── # ── Execution scripts ──────────────────────────────────────
├── x_bar_comparison.py                    # Table 1
├── run_experiments.py                     # Tables 2 & 3 (data generation)
├── analyze_results.py                     # Tables 2 & 3 (statistics and figures)
├── check_ordering_signs.py                # Verifies ordering preservation
├── error_decomposition.py                 # Table 4
├── run_shape_family_experiments.py        # Table 5
└── compare_ranking_methods.py             # Table 6
```

---

## 📊 Summary of Results

All scripts reproduce the numerical results reported in the paper:

| Table | Description | Script |
|:-----:|-------------|--------|
| 1 | Newton–Raphson vs. Brent intersection | `x_bar_comparison.py` |
| 2 | Execution time comparison | `run_experiments.py` |
| 3 | Relative error statistics | `analyze_results.py` |
| 4 | Error decomposition (`E_rep`, `e_S`, `E_tot`) | `error_decomposition.py` |
| 5 | Shape-family validation | `run_shape_family_experiments.py` |
| 6 | Ranking-method agreement | `compare_ranking_methods.py` |

> **Note on execution times:** The absolute execution times reported in Table 2 and Table 5 of the paper were measured on the system specified in Section 4.3 of the paper. The exact timings may vary when the code is executed on different hardware or with different software versions, but the *relative* speedup of the proposed method compared to the reference method (typically two orders of magnitude) remains consistent.

---

## 📝 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{rahdar2025lr,
  title   = {An O(1) direct algorithm for comparing LR fuzzy numbers
             with fixed-form parametric shape functions},
  author  = {Rahdar, Sahar and Ghanbari, Reza and Ghorbani-Moghadam, Khatere},
  journal = {[Journal Name]},
  year    = {2025}
}
```

---

## 📄 License

This project is released under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

For questions, issues, or suggestions, please open an issue on GitHub or contact:

- **Sahar Rahdar** — s.rahdar92@gmail.com
- **Reza Ghanbari** — rghanbari@um.ac.ir
- **Khatere Ghorbani-Moghadam** — k.ghorbani@khu.ac.ir
