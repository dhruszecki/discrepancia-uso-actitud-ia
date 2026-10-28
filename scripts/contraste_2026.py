"""Contraste con la edición 2026 de la encuesta.

La edición 2026 cambió las dos preguntas que definen la variable a predecir: la de uso
pasa a preguntar por herramientas o agentes de IA «en el trabajo», con opción múltiple por
tipo de herramienta, y la de actitud usa otra escala (Somewhat favorable, Neutral, Somewhat
unfavorable). También cambia el filtro de profesionales (CodeMain en lugar de MainBranch).
Por eso 2026 no se suma a la serie 2023-2025: se calcula la tasa con una definición
equivalente y se la compara con los escenarios de proyeccion_temporal.py, junto con H1
(experiencia) y H2 (rol).

Genera la figura temporal_contraste_2026.png y el informe docs/analisis/contraste_2026.md.
"""
from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import pointbiserialr

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_2026 = ROOT / "data" / "2026" / "survey_results_public.csv"
DATA_USERS = ROOT / "data" / "df_clean_users.parquet"
OUTPUTS = ROOT / "outputs"
DOCS = ROOT / "docs" / "analisis"
OUTPUTS.mkdir(parents=True, exist_ok=True)
DOCS.mkdir(parents=True, exist_ok=True)

_spec = importlib.util.spec_from_file_location("pipeline", ROOT / "scripts" / "pipeline.py")
pipeline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pipeline)

# Definición equivalente en 2026
PROFESIONAL_2026 = "I create code as a primary part of my job"
NO_USA_2026 = "I don’t use AI tools"
FAVORABLE_2026 = ["Very favorable", "Somewhat favorable"]

COLORS_YEAR = {2023: "#3b82f6", 2024: "#8b5cf6", 2025: "#f59e0b", 2026: "#dc2626"}
SCENARIOS = [
    ("Techo 30 %", 0.30, "#16a34a", "-"),
    ("Techo 45 %", 0.45, "#f59e0b", "--"),
    ("Techo 70 %", 0.70, "#6b7280", ":"),
]
TRAMOS = ["0–2", "3–5", "6–10", "11–20", ">20"]


def _c(x: float, dec: int = 1) -> str:
    """Número con coma decimal."""
    return f"{x:.{dec}f}".replace(".", ",")


def _miles(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def tramo_experiencia(years: pd.Series) -> pd.Categorical:
    return pd.cut(years, bins=[-0.1, 2, 5, 10, 20, 100], labels=TRAMOS)


def cargar_2026() -> pd.DataFrame:
    """Usuarios profesionales de 2026 con la discrepancia definida como en 2023-2025."""
    if not DATA_2026.exists():
        raise FileNotFoundError(
            f"No se encontró {DATA_2026}. Hay que bajar la edición 2026 de "
            "https://survey.stackoverflow.co/"
        )
    df = pd.read_csv(DATA_2026, low_memory=False)
    sel = df["AISelect"].astype("string")
    # Quien marca «no uso» junto con alguna herramienta se cuenta como no usuario
    usa = sel.notna() & ~sel.str.contains(NO_USA_2026, regex=False).fillna(False)
    df = df[(df["CodeMain"] == PROFESIONAL_2026) & usa].copy()
    # Igual que en el pipeline, quien no responde la actitud queda con discrepancia
    df["discrepancia"] = (~df["AISentiment"].isin(FAVORABLE_2026)).astype(int)
    df["YearsCode"] = pd.to_numeric(df["YearsCode"], errors="coerce")
    df["Role"] = df["Role"].map(lambda v: pipeline.DEVTYPE_REMAP.get(v, v))
    return df


def cargar_serie() -> pd.DataFrame:
    if not DATA_USERS.exists():
        raise FileNotFoundError(f"No se encontró {DATA_USERS}. Hay que correr antes scripts/pipeline.py")
    df = pd.read_parquet(DATA_USERS)
    df["discrepancia"] = 1 - df["target"]
    return df


def sensibilidad(df: pd.DataFrame) -> pd.DataFrame:
    """Tasa de 2026 con definiciones alternativas de usuario, profesional y actitud."""
    s = df["AISentiment"]
    filas = [
        ("Definición equivalente (principal)", len(df), df["discrepancia"].mean()),
        ("Solo quienes respondieron la actitud", int(s.notna().sum()), (~s.dropna().isin(FAVORABLE_2026)).mean()),
        ("Solo empleados", int((df["Employment"] == "Employed").sum()),
         df.loc[df["Employment"] == "Employed", "discrepancia"].mean()),
        ("Solo usuarios de asistentes de código", int(df["AISelect"].str.contains("coding assistants", regex=False).sum()),
         df.loc[df["AISelect"].str.contains("coding assistants", regex=False), "discrepancia"].mean()),
    ]
    tabla = pd.DataFrame(filas, columns=["definicion", "n", "tasa"])
    tabla["tasa"] = (tabla["tasa"] * 100).round(1)
    return tabla


def proyeccion() -> tuple[np.ndarray, dict]:
    """Repite el ajuste de proyeccion_temporal.py para cada techo."""
    res = json.loads((OUTPUTS / "temporal_resultados_cv.json").read_text())["auc_results"]
    tasas = np.array([res[f"within_{y}"]["target_0_pct"] for y in (2023, 2024, 2025)])
    years = np.array([2023, 2024, 2025], dtype=float)
    curvas = {}
    for label, L, _, _ in SCENARIOS:
        f = lambda t, k, t0, L=L: L / (1.0 + np.exp(-k * (t - t0)))
        popt, _ = curve_fit(f, years, tasas, p0=[1.0, 2024.0], maxfev=80000)
        curvas[label] = (popt, L)
    return tasas, curvas


def h1(serie: pd.DataFrame, d26: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    tablas = {}
    for year in (2023, 2024, 2025):
        sub = serie[serie["survey_year"] == year]
        tablas[year] = sub.groupby(tramo_experiencia(sub["YearsCode"]), observed=True)["discrepancia"].mean()
    tablas[2026] = d26.groupby(tramo_experiencia(d26["YearsCode"]), observed=True)["discrepancia"].mean()
    tabla = (pd.DataFrame(tablas) * 100).round(1)

    corr = {}
    for year in (2023, 2024, 2025):
        sub = serie[serie["survey_year"] == year]
        corr[year] = pointbiserialr(sub["discrepancia"], sub["YearsCode"])
    m = d26["YearsCode"].notna()
    corr[2026] = pointbiserialr(d26.loc[m, "discrepancia"], d26.loc[m, "YearsCode"])
    return tabla, corr


def h2(serie: pd.DataFrame, d26: pd.DataFrame, n_min: int = 100) -> pd.DataFrame:
    """Discrepancia por rol en 2025 y 2026, para los roles con al menos n_min casos en ambos."""
    s25 = serie[serie["survey_year"] == 2025]
    filas = []
    for col in [c for c in serie.columns if c.startswith("DevType_")]:
        rol = col.replace("DevType_", "")
        m25 = s25[col] == 1
        m26 = d26["Role"] == rol
        if m25.sum() < n_min or m26.sum() < n_min:
            continue
        filas.append({
            "rol": rol,
            "tasa_2025": round(s25.loc[m25, "discrepancia"].mean() * 100, 1),
            "n_2025": int(m25.sum()),
            "tasa_2026": round(d26.loc[m26, "discrepancia"].mean() * 100, 1),
            "n_2026": int(m26.sum()),
        })
    return pd.DataFrame(filas).sort_values("tasa_2026", ascending=False).reset_index(drop=True)


def figura(tasas: np.ndarray, curvas: dict, sens: pd.DataFrame, tabla_h1: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Contraste con la edición 2026 (preguntas de uso y actitud distintas)", fontsize=11, y=1.01)

    ax = axes[0]
    t = np.linspace(2022.5, 2026.5, 300)
    for label, _, color, ls in SCENARIOS:
        popt, L = curvas[label]
        y = L / (1.0 + np.exp(-popt[0] * (t - popt[1])))
        y26 = L / (1.0 + np.exp(-popt[0] * (2026 - popt[1])))
        ax.plot(t, y * 100, color=color, lw=1.8, ls=ls, label=f"Proyección, {label.lower()}: {_c(y26 * 100)} %")
    ax.scatter([2023, 2024, 2025], tasas * 100, color="black", s=90, zorder=10, label="Observado 2023–2025")
    for yr, v in zip((2023, 2024, 2025), tasas):
        ax.annotate(f"{_c(v * 100)} %", (yr, v * 100), textcoords="offset points", xytext=(7, 6), fontsize=9)
    principal = sens["tasa"].iloc[0]
    ax.errorbar([2026], [principal], yerr=[[principal - sens["tasa"].min()], [sens["tasa"].max() - principal]],
                fmt="D", color=COLORS_YEAR[2026], ms=8, capsize=5, zorder=11,
                label="2026 (barra: otras definiciones)")
    ax.annotate(f"{_c(principal)} %", (2026, principal), textcoords="offset points", xytext=(-42, 6), fontsize=9,
                color=COLORS_YEAR[2026])
    ax.axvline(2025.5, color="gray", ls=":", lw=1)
    ax.set_xlabel("Año de la encuesta")
    ax.set_ylabel("Tasa de discrepancia entre usuarios (%)")
    ax.set_title("Tasa observada en 2026 y escenarios proyectados")
    ax.set_xticks([2023, 2024, 2025, 2026])
    ax.set_xlim(2022.6, 2026.4)
    ax.set_ylim(0, 50)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    x = np.arange(len(TRAMOS))
    for year in tabla_h1.columns:
        ax.plot(x, tabla_h1[year].reindex(TRAMOS), marker="o", color=COLORS_YEAR[year],
                lw=2.4 if year == 2026 else 1.6, ls="-" if year == 2026 else "--", label=str(year))
    ax.set_xticks(x)
    ax.set_xticklabels(TRAMOS)
    ax.set_xlabel("Años programando")
    ax.set_ylabel("Tasa de discrepancia (%)")
    ax.set_title("Discrepancia por experiencia y año")
    ax.set_ylim(0, 50)
    ax.legend(title="Año", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def generar_doc(d26, sens, tasas, curvas, tabla_h1, corr, tabla_h2) -> None:
    proy = {label: L / (1.0 + np.exp(-popt[0] * (2026 - popt[1]))) * 100 for label, (popt, L) in curvas.items()}
    lineas = [
        "# Contraste con la edición 2026 de la encuesta",
        "",
        "Generado por `scripts/contraste_2026.py`.",
        "",
        "## Cambios en el instrumento",
        "",
        "- Uso: «Do you currently use AI tools or AI agents in your role at work?», opción múltiple por tipo de herramienta. En 2023-2025: «Do you currently use AI tools in your development process?».",
        "- Actitud: `AISentiment` (Very favorable, Somewhat favorable, Neutral, Somewhat unfavorable, Very unfavorable, Not sure). En 2023-2025: `AISent` (Very favorable, Favorable, Indifferent, Unfavorable, Very unfavorable, Unsure).",
        f"- Profesionales: `CodeMain` = «{PROFESIONAL_2026}». En 2023-2025: `MainBranch` = «I am a developer by profession».",
        "",
        "## Tasa de discrepancia",
        "",
        f"Usuarios profesionales de 2026: {_miles(len(d26))}.",
        "",
        "| Definición | N | Tasa (%) |",
        "|---|---:|---:|",
    ]
    lineas += [f"| {r.definicion} | {_miles(r.n)} | {_c(r.tasa)} |" for r in sens.itertuples()]
    lineas += [
        "",
        "Serie 2023-2025: " + ", ".join(f"{_c(v * 100)} %" for v in tasas) + ".",
        "",
        "Proyección para 2026: " + ", ".join(f"{k.lower()} {_c(v)} %" for k, v in proy.items()) + ".",
        "",
        "## H1: discrepancia por años programando",
        "",
        "| Tramo | 2023 | 2024 | 2025 | 2026 |",
        "|---|---:|---:|---:|---:|",
    ]
    lineas += [f"| {t} | " + " | ".join(_c(tabla_h1.loc[t, y]) for y in tabla_h1.columns) + " |" for t in TRAMOS]
    lineas += ["", "Correlación punto-biserial con los años programando:", ""]
    lineas += [f"- {y}: r = {_c(r.statistic, 3)}, p = {r.pvalue:.1e}" for y, r in corr.items()]
    lineas += [
        "",
        "## H2: discrepancia por rol (roles con al menos 100 casos en 2025 y 2026)",
        "",
        "| Rol | 2025 (%) | N 2025 | 2026 (%) | N 2026 |",
        "|---|---:|---:|---:|---:|",
    ]
    lineas += [f"| {r.rol} | {_c(r.tasa_2025)} | {_miles(r.n_2025)} | {_c(r.tasa_2026)} | {_miles(r.n_2026)} |"
               for r in tabla_h2.itertuples()]
    (DOCS / "contraste_2026.md").write_text("\n".join(lineas) + "\n")


def main():
    d26 = cargar_2026()
    serie = cargar_serie()
    sens = sensibilidad(d26)
    tasas, curvas = proyeccion()
    tabla_h1, corr = h1(serie, d26)
    tabla_h2 = h2(serie, d26)

    print(sens.to_string(index=False))
    print(tabla_h1.to_string())
    for y, r in corr.items():
        print(y, round(r.statistic, 3), f"{r.pvalue:.1e}")
    print(tabla_h2.to_string(index=False))

    fig = figura(tasas, curvas, sens, tabla_h1)
    fig.savefig(OUTPUTS / "temporal_contraste_2026.png", dpi=150, bbox_inches="tight")
    generar_doc(d26, sens, tasas, curvas, tabla_h1, corr, tabla_h2)


if __name__ == "__main__":
    main()
