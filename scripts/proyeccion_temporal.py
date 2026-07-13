"""Proyección temporal 2026: ajuste logístico sobre tasa de discrepancia
y extrapolación de AUC intra-año.

Outputs:
    outputs/08_temporal_proyeccion.png
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"

# ── datos observados ────────────────────────────────────────────────
YEARS = np.array([2023, 2024, 2025], dtype=float)
TASA_BASE = np.array([0.153, 0.185, 0.271])
AUC_INTRA = np.array([0.577, 0.599, 0.639])

# ── tres escenarios de techo ────────────────────────────────────────
# L < 0.271 no es fittable (ya superamos el techo).
# L = 0.45 → 2025 ya pasó el punto de inflexión (L/2 = 0.225 < 0.271).
# L = 0.55 → 2025 en el punto de inflexión (L/2 = 0.275 ≈ 0.271).
# L = 0.70 → 2025 todavía en fase de aceleración (L/2 = 0.35 > 0.271).
SCENARIOS = [
    ("Techo 45 % — moderado",   0.45, "#16a34a", "-"),
    ("Techo 55 % — intermedio", 0.55, "#f59e0b", "--"),
    ("Techo 70 % — acelerado",  0.70, "#dc2626", ":"),
]

years_plot = np.linspace(2022.5, 2027.5, 400)


def logistic(t: np.ndarray, k: float, t0: float, L: float) -> np.ndarray:
    return L / (1.0 + np.exp(-k * (t - t0)))


# ── figura ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    "Proyección temporal condicionada — con tres supuestos de techo",
    fontsize=11, y=1.01
)

# ── Panel A: tasa de discrepancia ───────────────────────────────────
ax = axes[0]
proj_vals: dict[str, float] = {}

for label, L, color, ls in SCENARIOS:
    try:
        popt, _ = curve_fit(
            lambda t, k, t0: logistic(t, k, t0, L),
            YEARS, TASA_BASE,
            p0=[1.5, 2025.5],
            maxfev=80000,
        )
        y_curve = logistic(years_plot, *popt, L)
        y_2026 = float(logistic(2026.0, *popt, L))
        proj_vals[label] = y_2026

        ax.plot(years_plot, y_curve * 100, color=color, lw=2, ls=ls,
                label=f"{label}: {y_2026*100:.1f} % en 2026")
        ax.scatter([2026], [y_2026 * 100], color=color, s=65,
                   marker="D", zorder=6)
    except RuntimeError:
        pass  # si el ajuste no converge para algún L, se omite

# banda de rango entre escenarios
vals = list(proj_vals.values())
if len(vals) >= 2:
    lo, hi = min(vals), max(vals)
    ax.axvspan(2025.6, 2026.4, alpha=0.09, color="gray",
               label=f"Rango 2026: {lo*100:.1f} %–{hi*100:.1f} %")

# puntos observados
ax.scatter(YEARS, TASA_BASE * 100, color="black", s=110, zorder=10,
           label="Observado")
for yr, v in zip(YEARS, TASA_BASE):
    ax.annotate(f"{v*100:.1f} %", (yr, v * 100),
                textcoords="offset points", xytext=(7, 6), fontsize=9)

ax.axvline(2025.5, color="gray", ls=":", lw=1)
ax.text(2025.65, 3, "→ proyección", color="gray", fontsize=8)
ax.set_xlabel("Año del survey")
ax.set_ylabel("Tasa de discrepancia entre usuarios (%)")
ax.set_title("Tasa de discrepancia: ajuste logístico")
ax.legend(fontsize=8, loc="upper left")
ax.set_xlim(2022.5, 2027.5)
ax.set_ylim(0, 75)
ax.grid(True, alpha=0.3)

# ── Panel B: AUC intra-año ──────────────────────────────────────────
ax = axes[1]
t_idx = YEARS - 2023.0
t_plot = years_plot - 2023.0

# extrapolación cuadrática (captura la aceleración)
c_quad = np.polyfit(t_idx, AUC_INTRA, 2)
auc_quad = np.polyval(c_quad, t_plot)
auc_quad_2026 = float(np.polyval(c_quad, 3.0))

# extrapolación lineal (cota conservadora)
c_lin = np.polyfit(t_idx, AUC_INTRA, 1)
auc_lin = np.polyval(c_lin, t_plot)
auc_lin_2026 = float(np.polyval(c_lin, 3.0))

ax.plot(years_plot, np.clip(auc_quad, 0.5, 1.0), color="#3b82f6", lw=2,
        ls="--", label=f"Cuadrática: {auc_quad_2026:.3f} en 2026")
ax.plot(years_plot, np.clip(auc_lin, 0.5, 1.0), color="#8b5cf6", lw=2,
        ls=":", label=f"Lineal: {auc_lin_2026:.3f} en 2026")

ax.scatter([2026], [auc_quad_2026], color="#3b82f6", s=65, marker="D", zorder=6)
ax.scatter([2026], [auc_lin_2026], color="#8b5cf6", s=65, marker="D", zorder=6)
ax.fill_between(
    [2025.6, 2026.4],
    [min(auc_lin_2026, auc_quad_2026)] * 2,
    [max(auc_lin_2026, auc_quad_2026)] * 2,
    alpha=0.12, color="gray",
    label=f"Rango 2026: {min(auc_lin_2026, auc_quad_2026):.3f}–{max(auc_lin_2026, auc_quad_2026):.3f}",
)

ax.scatter(YEARS, AUC_INTRA, color="black", s=110, zorder=10, label="Observado")
for yr, v in zip(YEARS, AUC_INTRA):
    ax.annotate(f"{v:.3f}", (yr, v),
                textcoords="offset points", xytext=(7, 6), fontsize=9)

ax.axvline(2025.5, color="gray", ls=":", lw=1)
ax.text(2025.65, 0.503, "→ proyección", color="gray", fontsize=8)
ax.set_xlabel("Año del survey")
ax.set_ylabel("ROC-AUC (modelo intra-año, LogReg)")
ax.set_title("Poder discriminativo: extrapolación")
ax.legend(fontsize=8, loc="upper left")
ax.set_xlim(2022.5, 2027.5)
ax.set_ylim(0.50, 0.80)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = OUTPUTS / "08_temporal_proyeccion.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"✓ {out_path}")
print()
print("Proyecciones 2026:")
print(f"  Tasa de discrepancia: {min(vals)*100:.1f} %–{max(vals)*100:.1f} %")
print(f"  ROC-AUC:              {min(auc_lin_2026, auc_quad_2026):.3f}–{max(auc_lin_2026, auc_quad_2026):.3f}")
