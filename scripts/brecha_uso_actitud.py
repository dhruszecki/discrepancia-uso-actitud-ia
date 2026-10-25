"""Brecha entre uso y actitud en toda la población (outputs/brecha_uso_actitud.png).

Cruza la pregunta de uso de IA (AISelect) con la de actitud (AISent) en cuatro grupos y
muestra la distribución global, la evolución por año y el porcentaje de usuarios sin
actitud favorable según años programando.
"""
import importlib.util
import io
import contextlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location("pipeline", ROOT / "scripts" / "pipeline.py")
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)

GRUPOS = [
    ("Usa y actitud favorable", "#0072B2"),
    ("Usa sin actitud favorable", "#D55E00"),
    ("No usa, actitud favorable", "#009E73"),
    ("No usa, sin actitud favorable", "#999999"),
]
ORDEN = [g for g, _ in GRUPOS]
COLORES = dict(GRUPOS)


def anios_programando(v):
    if pd.isna(v):
        return np.nan
    if v == "Less than 1 year":
        return 0.5
    if v == "More than 50 years":
        return 51.0
    try:
        return float(v)
    except ValueError:
        return np.nan


def cargar() -> pd.DataFrame:
    with contextlib.redirect_stdout(io.StringIO()):
        d23, d24, d25 = pipeline.cargar_y_normalizar()
    df = pd.concat([d23, d24, d25], ignore_index=True)
    df = df[
        (df["MainBranch"] == pipeline.DEV_LABEL) & df["AISelect"].notna() & df["AISent"].notna()
    ].copy()
    usa = df["AISelect"] == "Yes"
    favorable = df["AISent"].isin(pipeline.ACTITUD_POSITIVA)
    df["grupo"] = np.select(
        [usa & favorable, usa & ~favorable, ~usa & favorable],
        ORDEN[:3],
        default=ORDEN[3],
    )
    df["anios"] = df["YearsCode"].map(anios_programando)
    return df


def figura(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    # Distribución global
    ax = axes[0]
    dist = df["grupo"].value_counts(normalize=True).reindex(ORDEN) * 100
    barras = ax.bar(range(4), dist.values, color=[COLORES[g] for g in ORDEN])
    for b, v in zip(barras, dist.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f} %", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(4))
    ax.set_xticklabels([g.replace(", ", ",\n").replace(" y ", " y\n").replace(" sin ", "\nsin ") for g in ORDEN], fontsize=8)
    ax.set_ylabel("% de desarrolladores profesionales")
    ax.set_ylim(0, 68)
    ax.set_title("Distribución de los cuatro grupos\n(2023–2025 combinados)")

    # Evolución por año
    ax = axes[1]
    tabla = (
        df.groupby("survey_year")["grupo"].value_counts(normalize=True).unstack().reindex(columns=ORDEN) * 100
    )
    base = np.zeros(len(tabla))
    for g in ORDEN:
        ax.bar(tabla.index.astype(str), tabla[g], bottom=base, color=COLORES[g], label=g, width=0.75)
        for i, (v, b) in enumerate(zip(tabla[g], base)):
            if v > 5:
                ax.text(i, b + v / 2, f"{v:.0f} %", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        base += tabla[g].values
    ax.set_ylabel("% de desarrolladores profesionales")
    ax.set_ylim(0, 100)
    ax.set_title("Evolución por año de la encuesta")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=8, frameon=False)

    # Usuarios sin actitud favorable según años programando
    ax = axes[2]
    usuarios = df[df["AISelect"] == "Yes"].dropna(subset=["anios"]).copy()
    usuarios["tramo"] = pd.cut(usuarios["anios"], [-0.1, 5, 10, 20, 100], labels=["0–5", "6–10", "11–20", "más de 20"])
    sin_fav = (usuarios["grupo"] == ORDEN[1]).groupby(usuarios["tramo"], observed=True).mean() * 100
    barras = ax.bar(range(len(sin_fav)), sin_fav.values, color=COLORES[ORDEN[1]])
    for b, v in zip(barras, sin_fav.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.1f} %", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(sin_fav)))
    ax.set_xticklabels(sin_fav.index.astype(str))
    ax.set_xlabel("Años programando")
    ax.set_ylabel("% de usuarios sin actitud favorable")
    ax.set_ylim(0, 30)
    ax.set_title("Usuarios sin actitud favorable\nsegún años programando")

    fig.tight_layout()
    return fig


def main():
    df = cargar()
    print(f"N = {len(df):,}")
    print((df["grupo"].value_counts(normalize=True) * 100).round(1).to_string())
    fig = figura(df)
    fig.savefig(OUTPUTS / "brecha_uso_actitud.png", dpi=150, bbox_inches="tight")
    print("Guardado outputs/brecha_uso_actitud.png")


if __name__ == "__main__":
    main()
