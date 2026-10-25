"""Tasa de discrepancia por año entre usuarios de IA.

Repite H1 (experiencia) y H2 (rol) por año de la encuesta sobre los usuarios de IA
(df_clean_users.parquet, N = 80.369), como complemento del análisis conjunto de eda.py.
Genera las figuras temporal_tasa_discrepancia.png, temporal_experiencia.png y
temporal_rol.png, y el informe docs/analisis/resultados_temporal_usuarios.md.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

TARGET_LABEL_0 = "Con discrepancia uso-actitud"
COLOR_GLOBAL = "#1f2937"
COLORS_YEAR = {2023: "#3b82f6", 2024: "#8b5cf6", 2025: "#f59e0b"}
YEAR_EXCLUSIVE_NOTE = "* Rol presente solo en algunas ediciones de la encuesta"

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "df_clean_users.parquet"
OUTPUTS = ROOT / "outputs"
DOCS = ROOT / "docs" / "analisis"
OUTPUTS.mkdir(parents=True, exist_ok=True)
DOCS.mkdir(parents=True, exist_ok=True)


def cargar() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró {DATA_PATH}. "
            "Hay que correr antes scripts/pipeline.py"
        )
    df = pd.read_parquet(DATA_PATH)
    df["discrepancia"] = 1 - df["target"]
    return df


def _num(x: float, dec: int = 0) -> str:
    """Formato argentino: punto de miles y coma decimal."""
    return f"{x:,.{dec}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def tramo_experiencia(years: pd.Series) -> pd.Categorical:
    bins = [-0.1, 2, 5, 10, 20, 100]
    labels = ["0–2", "3–5", "6–10", "11–20", ">20"]
    return pd.cut(years, bins=bins, labels=labels)


def tasa_base_por_anio(df: pd.DataFrame) -> tuple[pd.DataFrame, plt.Figure]:
    """Tasa de discrepancia por año de la encuesta."""
    grp = df.groupby("survey_year")["discrepancia"]
    tabla = pd.DataFrame({
        "tasa_pct": (grp.mean() * 100).round(1),
        "n_total": grp.count(),
        "n_disc": grp.sum().astype(int),
    })

    tasa_global = df["discrepancia"].mean() * 100

    fig, ax = plt.subplots(figsize=(7, 5))
    years = tabla.index.tolist()
    x = np.arange(len(years))

    ax.bar(x, tabla["tasa_pct"], color=[COLORS_YEAR[y] for y in years], width=0.5)

    for i, (year, row) in enumerate(tabla.iterrows()):
        ax.text(i, row["tasa_pct"] + 0.4,
                f"{_num(row['tasa_pct'], 1)} %\n(N = {_num(row['n_total'])})",
                ha="center", va="bottom", fontsize=10)

    ax.axhline(tasa_global, color=COLOR_GLOBAL, linestyle="--", lw=1.5,
               label=f"Tres años juntos: {_num(tasa_global, 1)} %")

    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years], fontsize=11)
    ax.set_xlabel("Año de la encuesta")
    ax.set_ylabel(f"% {TARGET_LABEL_0}")
    ax.set_ylim(0, tabla["tasa_pct"].max() * 1.4)
    ax.set_title(
        f"Tasa de discrepancia por año de la encuesta\n(usuarios de IA, N = {_num(len(df))})"
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()

    return tabla, fig


def h1_temporal_usuarios(df: pd.DataFrame) -> tuple[pd.DataFrame, plt.Figure]:
    """H1 por año: discrepancia por tramo de experiencia."""
    df = df.copy()
    df["exp_tramo"] = tramo_experiencia(df["YearsCode"])

    global_rate = (
        df.groupby("exp_tramo", observed=True)["discrepancia"]
        .mean()
        .rename("global")
    )
    por_anio = (
        df.groupby(["survey_year", "exp_tramo"], observed=True)["discrepancia"]
        .mean()
        .unstack(level=0)
    )
    por_anio.columns = [str(int(c)) for c in por_anio.columns]
    tabla = pd.concat([global_rate, por_anio], axis=1) * 100
    tabla = tabla.round(1)

    for col in tabla.columns:
        vals = tabla[col].dropna().values
        mono = all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))
        estado = "creciente" if mono else "no creciente"
        diferencia = vals[-1] - vals[0]
        print(f"  H1 {col}: {estado} (de {vals[0]:.1f}% a {vals[-1]:.1f}%, diferencia {diferencia:+.1f} puntos)")

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(tabla.index))

    ax.plot(x, tabla["global"], "--o", color=COLOR_GLOBAL, lw=2, ms=6,
            label="Tres años juntos", zorder=5)
    for year in [2023, 2024, 2025]:
        col = str(year)
        if col in tabla.columns:
            ax.plot(x, tabla[col], "-o", color=COLORS_YEAR[year], lw=2, ms=6, label=col)

    ax.set_xticks(x)
    ax.set_xticklabels(tabla.index)
    ax.set_xlabel("Años programando")
    ax.set_ylabel(f"% {TARGET_LABEL_0}")
    ax.set_ylim(0, tabla.max().max() * 1.25)
    ax.set_title("Discrepancia según años programando, por año de la encuesta\n(usuarios de IA)")
    ax.legend(title="", loc="upper left", fontsize=9)
    ax.grid(linestyle="--", alpha=0.3)
    fig.tight_layout()

    return tabla, fig


def h2_temporal_usuarios(df: pd.DataFrame) -> tuple[pd.DataFrame, plt.Figure]:
    """H2 por año: discrepancia por rol."""
    devtype_cols = [c for c in df.columns if c.startswith("DevType_")]

    rows = []
    for dt_col in devtype_cols:
        label = dt_col.replace("DevType_", "")
        for year in [2023, 2024, 2025]:
            mask = (df["survey_year"] == year) & (df[dt_col] == 1)
            n = int(mask.sum())
            if n < 100:
                continue
            tasa = df.loc[mask, "discrepancia"].mean() * 100
            rows.append({"devtype": label, "year": year, "tasa": round(tasa, 1), "n": n})

    tabla_long = pd.DataFrame(rows)

    # Para comparar entre años, solo quedan los roles con al menos 100 casos en
    # dos ediciones o más.
    years_ok = tabla_long.groupby("devtype")["year"].count()
    devtypes_incluidos = years_ok[years_ok >= 2].index.tolist()
    tabla_long = tabla_long[tabla_long["devtype"].isin(devtypes_incluidos)]

    tabla = tabla_long.pivot(index="devtype", columns="year", values="tasa")
    tabla.columns.name = None
    tabla["_avg"] = tabla.mean(axis=1)
    tabla = tabla.sort_values("_avg", ascending=True).drop(columns="_avg")

    # Algunos roles no existen en las tres ediciones; se marcan con asterisco.
    year_exclusive = [dt for dt in tabla.index if years_ok.get(dt, 0) < 3]
    ytick_labels = [
        f"{dt[:38]}*" if dt in year_exclusive else dt[:42]
        for dt in tabla.index
    ]

    fig, ax = plt.subplots(figsize=(11, max(6, 0.45 * len(tabla))))
    y = np.arange(len(tabla.index))
    height = 0.27

    for i, year in enumerate([2023, 2024, 2025]):
        if year not in tabla.columns:
            continue
        ax.barh(y + (i - 1) * height, tabla[year], height,
                label=str(year), color=COLORS_YEAR[year])

    for year in [2023, 2024, 2025]:
        global_year = df.loc[df["survey_year"] == year, "discrepancia"].mean() * 100
        ax.axvline(global_year, color=COLORS_YEAR[year], linestyle=":", lw=1.2, alpha=0.6)

    ax.set_yticks(y)
    ax.set_yticklabels(ytick_labels, fontsize=9)
    ax.set_xlabel(f"% {TARGET_LABEL_0}")
    ax.set_title(
        f"Discrepancia por rol y año de la encuesta (usuarios de IA, N = {_num(len(df))})"
    )
    ax.legend(title="Año", loc="lower right", fontsize=9)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    fig.tight_layout()
    if year_exclusive:
        fig.subplots_adjust(bottom=0.08)
        fig.text(0.02, 0.01, YEAR_EXCLUSIVE_NOTE, fontsize=8, style="italic", ha="left")

    return tabla, fig


def generar_lectura(
    df: pd.DataFrame,
    tabla_base: pd.DataFrame,
    tabla_h1: pd.DataFrame,
    tabla_h2: pd.DataFrame,
) -> str:
    """Arma el informe markdown con los resultados por año."""
    global_rate = df["discrepancia"].mean() * 100
    global_n = len(df)

    out = []
    out.append("# Análisis temporal: discrepancia entre uso y actitud en usuarios de IA\n\n")
    out.append("Datos: `data/df_clean_users.parquet`. Generado por `scripts/temporal.py`.\n\n")
    out.append(
        f"N total: {global_n:,}. "
        f"Tasa de discrepancia con los tres años juntos: {global_rate:.1f}%.\n\n"
    )

    out.append("## 1. Tasa de discrepancia por año de la encuesta\n\n")
    out.append(tabla_base.to_markdown())
    out.append("\n\n*(tasa_pct en %; n_total y n_disc en cantidad de casos)*\n\n")

    tasas = tabla_base["tasa_pct"]
    min_year = int(tasas.idxmin())
    max_year = int(tasas.idxmax())
    delta = float(tasas.max() - tasas.min())
    out.append(
        f"La tasa de discrepancia va de {tasas.min():.1f}% ({min_year}) "
        f"a {tasas.max():.1f}% ({max_year}), un rango de {delta:.1f} puntos. "
        f"Con los tres años juntos es {global_rate:.1f}%.\n\n"
    )
    out.append(
        "En el conjunto completo de desarrolladores (155k) la tasa de actitud no favorable "
        "es de 55 a 65 % por año, porque incluye a los no usuarios. Entre usuarios de IA "
        "la tasa es mucho menor y refleja solo a quienes usan IA sin declarar una actitud favorable.\n\n"
    )

    out.append("## 2. H1: discrepancia según experiencia, por año\n\n")
    out.append(tabla_h1.to_markdown())
    out.append(
        "\n\n*(porcentajes; global = los tres años juntos; las demás columnas son el año de la encuesta)*\n\n"
    )
    out.append("Por columna:\n\n")

    for col in tabla_h1.columns:
        vals = tabla_h1[col].dropna().values
        if len(vals) < 2:
            continue
        mono = all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))
        gap = vals[-1] - vals[0]
        estado = "creciente en todos los tramos" if mono else "no creciente en todos los tramos"
        out.append(
            f"- {col}: {estado} (de {vals[0]:.1f}% a {vals[-1]:.1f}%, diferencia de {gap:+.1f} puntos).\n"
        )

    out.append(
        "\nEn el conjunto completo (155k) la diferencia entre los tramos de menor y mayor experiencia "
        "es mucho más grande, porque incluye el paso de no usuarios (tasa alta) a usuarios (tasa baja). "
        "Entre usuarios de IA la diferencia es chica y solo es marcada en 2025.\n\n"
    )

    out.append("## 3. H2: discrepancia por rol, por año\n\n")
    out.append(tabla_h2.to_markdown())
    out.append(
        "\n\n*(porcentajes; NaN cuando el rol no llega a 100 casos en ese año)*\n\n"
    )

    years_seen: dict[str, int] = {}
    for col in [2023, 2024, 2025]:
        if col in tabla_h2.columns:
            for dt in tabla_h2[tabla_h2[col].notna()].index:
                years_seen[dt] = years_seen.get(dt, 0) + 1
    year_exclusive_doc = [dt for dt, cnt in years_seen.items() if cnt < 3]

    if year_exclusive_doc:
        out.append("### Comparabilidad entre años\n\n")
        for dt in year_exclusive_doc:
            yrs_with_data = [
                str(y) for y in [2023, 2024, 2025]
                if y in tabla_h2.columns and not pd.isna(tabla_h2.loc[dt, y])
            ]
            out.append(
                f"- {dt}\\*: hay datos solo en {', '.join(yrs_with_data)}. "
                "En los otros años la encuesta usaba categorías distintas y la columna queda en cero.\n"
            )
        out.append(
            "\nNo se incluyen en la figura los roles con 100 casos o más en un solo año "
            "(por ejemplo, 'Architect, software or solutions' en 2025).\n\n"
        )

    avg = tabla_h2.mean(axis=1).sort_values(ascending=False)
    top3 = avg.head(3)
    bot3 = avg.tail(3)
    out.append(
        "Los tres roles con mayor tasa de discrepancia promedio son "
        + ", ".join(f"{dt} ({v:.1f}%)" for dt, v in top3.items())
        + ".\n\n"
    )
    out.append(
        "Los tres con menor tasa promedio son "
        + ", ".join(f"{dt} ({v:.1f}%)" for dt, v in bot3[::-1].items())
        + ".\n\n"
    )
    out.append(
        "El orden de los roles cambia entre años. Las magnitudes son menores que en el conjunto completo "
        "porque la tasa entre usuarios es cercana a 21,5 % y no a 60 %.\n\n"
    )

    out.append("## 4. En resumen\n\n")
    out.append(
        f"- La tasa de discrepancia sube {delta:.1f} puntos entre {min_year} y {max_year}. "
        "El crecimiento entre años consecutivos no es uniforme: la sección temporal del trabajo lo analiza.\n"
    )
    out.append(
        "- El gradiente por experiencia es casi nulo en 2023 y 2024 y solo es marcado en 2025.\n"
    )
    out.append(
        "- El orden de los roles con más discrepancia cambia entre años, así que no puede afirmarse que se mantenga.\n"
    )

    return "".join(out)


def main():
    print("Análisis temporal sobre usuarios de IA")

    OUTPUTS.mkdir(exist_ok=True)

    df = cargar()
    dist = df["survey_year"].value_counts().sort_index()
    print(f"Datos: {df.shape[0]:,} filas, {df.shape[1]} columnas")
    print(f"Casos por año: {dist.to_dict()}")
    print(f"Tasa global de discrepancia: {df['discrepancia'].mean() * 100:.1f}%\n")

    print("Tasa por año")
    tabla_base, fig_base = tasa_base_por_anio(df)
    print(tabla_base.to_string())
    fig_base.savefig(OUTPUTS / "temporal_tasa_discrepancia.png", dpi=150, bbox_inches="tight")
    plt.close(fig_base)
    print("Guardado outputs/temporal_tasa_discrepancia.png\n")

    print("H1 por año")
    tabla_h1, fig_h1 = h1_temporal_usuarios(df)
    print(tabla_h1.to_string())
    fig_h1.savefig(OUTPUTS / "temporal_experiencia.png", dpi=150, bbox_inches="tight")
    plt.close(fig_h1)
    print("Guardado outputs/temporal_experiencia.png\n")

    print("H2 por año")
    tabla_h2, fig_h2 = h2_temporal_usuarios(df)
    print(tabla_h2.to_string())
    fig_h2.savefig(OUTPUTS / "temporal_rol.png", dpi=150, bbox_inches="tight")
    plt.close(fig_h2)
    print("Guardado outputs/temporal_rol.png\n")

    md = generar_lectura(df, tabla_base, tabla_h1, tabla_h2)
    doc_path = DOCS / "resultados_temporal_usuarios.md"
    doc_path.write_text(md)
    print(f"Guardado {doc_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
