"""Escenarios para 2026 (outputs/temporal_proyeccion.png).

Ajusta una curva logística a la tasa de discrepancia bajo tres supuestos de techo y
extrapola el AUC de los modelos dentro de cada año. Es ilustrativo: con tres años
observados, el resultado depende del techo que se supone.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

import numpy as np
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

# Las tasas y los AUC se leen de los resultados de modelado_temporal.py
_res = json.loads((OUTPUTS / "temporal_resultados_cv.json").read_text())["auc_results"]
YEARS = np.array([2023, 2024, 2025], dtype=float)
TASA_BASE = np.array([_res[f"within_{y}"]["target_0_pct"] for y in (2023, 2024, 2025)])
AUC_INTRA = np.array([_res[f"within_{y}"]["roc_auc"] for y in (2023, 2024, 2025)])

# La tasa observada (15,3 %, 23,4 % y 27,6 %) crece cada vez menos (+8,1 y +4,1 puntos).
# Con tres puntos no se puede estimar el techo L, así que se fija en tres valores: 0,30
# (el fenómeno ya está cerca de saturar), 0,45 (la desaceleración sigue, con margen) y
# 0,70 (la desaceleración sería transitoria).
SCENARIOS = [
    ("Techo 30 % — saturando",  0.30, "#16a34a", "-"),
    ("Techo 45 % — moderado",   0.45, "#f59e0b", "--"),
    ("Techo 70 % — alto",       0.70, "#dc2626", ":"),
]

years_plot = np.linspace(2022.5, 2027.5, 400)


def logistic(t: np.ndarray, k: float, t0: float, L: float) -> np.ndarray:
    return L / (1.0 + np.exp(-k * (t - t0)))


def _c(x: float, dec: int) -> str:
    """Número con coma decimal."""
    return f"{x:.{dec}f}".replace(".", ",")


fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    "Proyección condicionada a tres supuestos de techo",
    fontsize=11, y=1.01
)

ax = axes[0]
proj_vals: dict[str, float] = {}

for label, L, color, ls in SCENARIOS:
    try:
        popt, _ = curve_fit(
            lambda t, k, t0: logistic(t, k, t0, L),
            YEARS, TASA_BASE,
            p0=[1.0, 2024.0],
            maxfev=80000,
        )
        y_curve = logistic(years_plot, *popt, L)
        y_2026 = float(logistic(2026.0, *popt, L))
        proj_vals[label] = y_2026

        ax.plot(years_plot, y_curve * 100, color=color, lw=2, ls=ls,
                label=f"{label}: {_c(y_2026*100, 1)} % en 2026")
        ax.scatter([2026], [y_2026 * 100], color=color, s=65,
                   marker="D", zorder=6)
    except RuntimeError:
        # si el ajuste no converge para algún techo, ese escenario se omite
        pass

vals = list(proj_vals.values())
if len(vals) >= 2:
    lo, hi = min(vals), max(vals)
    ax.axvspan(2025.6, 2026.4, alpha=0.09, color="gray",
               label=f"Rango 2026: {_c(lo*100, 1)} %–{_c(hi*100, 1)} %")

ax.scatter(YEARS, TASA_BASE * 100, color="black", s=110, zorder=10,
           label="Observado")
for yr, v in zip(YEARS, TASA_BASE):
    ax.annotate(f"{_c(v*100, 1)} %", (yr, v * 100),
                textcoords="offset points", xytext=(7, 6), fontsize=9)

ax.axvline(2025.5, color="gray", ls=":", lw=1)
ax.text(2025.65, 3, "proyección", color="gray", fontsize=8)
ax.set_xlabel("Año de la encuesta")
ax.set_ylabel("Tasa de discrepancia entre usuarios (%)")
ax.set_title("Tasa de discrepancia: ajuste logístico")
ax.legend(fontsize=8, loc="upper left")
ax.set_xlim(2022.5, 2027.5)
ax.set_ylim(0, 75)
ax.grid(True, alpha=0.3)

ax = axes[1]
t_idx = YEARS - 2023.0
t_plot = years_plot - 2023.0

# Dos extrapolaciones: la cuadrática sigue la aceleración observada y la lineal sirve de
# cota conservadora
c_quad = np.polyfit(t_idx, AUC_INTRA, 2)
auc_quad = np.polyval(c_quad, t_plot)
auc_quad_2026 = float(np.polyval(c_quad, 3.0))

c_lin = np.polyfit(t_idx, AUC_INTRA, 1)
auc_lin = np.polyval(c_lin, t_plot)
auc_lin_2026 = float(np.polyval(c_lin, 3.0))

ax.plot(years_plot, np.clip(auc_quad, 0.5, 1.0), color="#3b82f6", lw=2,
        ls="--", label=f"Cuadrática: {_c(auc_quad_2026, 3)} en 2026")
ax.plot(years_plot, np.clip(auc_lin, 0.5, 1.0), color="#8b5cf6", lw=2,
        ls=":", label=f"Lineal: {_c(auc_lin_2026, 3)} en 2026")

ax.scatter([2026], [auc_quad_2026], color="#3b82f6", s=65, marker="D", zorder=6)
ax.scatter([2026], [auc_lin_2026], color="#8b5cf6", s=65, marker="D", zorder=6)
ax.fill_between(
    [2025.6, 2026.4],
    [min(auc_lin_2026, auc_quad_2026)] * 2,
    [max(auc_lin_2026, auc_quad_2026)] * 2,
    alpha=0.12, color="gray",
    label=f"Rango 2026: {_c(min(auc_lin_2026, auc_quad_2026), 3)}–{_c(max(auc_lin_2026, auc_quad_2026), 3)}",
)

ax.scatter(YEARS, AUC_INTRA, color="black", s=110, zorder=10, label="Observado")
for yr, v in zip(YEARS, AUC_INTRA):
    ax.annotate(f"{_c(v, 3)}", (yr, v),
                textcoords="offset points", xytext=(7, 6), fontsize=9)

ax.axvline(2025.5, color="gray", ls=":", lw=1)
ax.text(2025.65, 0.503, "proyección", color="gray", fontsize=8)
ax.set_xlabel("Año de la encuesta")
ax.set_ylabel("ROC-AUC (regresión logística dentro del año)")
ax.set_title("Capacidad de discriminar: extrapolación")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: _c(v, 2)))
ax.legend(fontsize=8, loc="upper left")
ax.set_xlim(2022.5, 2027.5)
ax.set_ylim(0.50, 0.80)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = OUTPUTS / "temporal_proyeccion.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"Guardado {out_path.relative_to(ROOT)}")
print("\nProyecciones 2026:")
print(f"  Tasa de discrepancia: {min(vals)*100:.1f} %–{max(vals)*100:.1f} %")
print(f"  ROC-AUC:              {min(auc_lin_2026, auc_quad_2026):.3f}–{max(auc_lin_2026, auc_quad_2026):.3f}")
