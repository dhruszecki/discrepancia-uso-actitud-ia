"""Tasa de discrepancia por año entre usuarios de IA.

Repite H1 (experiencia) y H2 (rol) por año de la encuesta sobre los usuarios de IA
(df_clean_users.parquet, N = 80.369), como complemento del análisis conjunto de eda.py.
Genera las figuras temporal_tasa_discrepancia.png, temporal_experiencia.png y
temporal_rol.png, y el informe docs/analisis/resultados_temporal_usuarios.md.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nombres_legibles import configurar_figuras, nombre_rol  # noqa: E402

warnings.filterwarnings("ignore")
configurar_figuras()

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
    ax.set_ylabel("% con discrepancia")
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

    # El tramo con menos experiencia de 2025 es chico: se indica su N.
    n_2025 = df[df["survey_year"] == 2025].groupby("exp_tramo", observed=True).size()
    primero = tabla.index[0]
    if "2025" in tabla.columns and n_2025.get(primero, 0) < 500:
        ax.annotate(f"2025: N = {_num(n_2025[primero])}", (0, tabla.loc[primero, "2025"]),
                    xytext=(10, 0), textcoords="offset points", va="center", fontsize=8, color=COLORS_YEAR[2025])

    ax.set_xticks(x)
    ax.set_xticklabels(tabla.index)
    ax.set_xlabel("Años programando")
    ax.set_ylabel("% con discrepancia")
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
        f"{nombre_rol(dt)}*" if dt in year_exclusive else nombre_rol(dt)
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
        ax.axvline(global_year, color=COLORS_YEAR[year], linestyle=":", lw=1.2, alpha=0.6,
                   label="Media de cada año (punteada)" if year == 2023 else None)

    ax.set_yticks(y)
    ax.set_yticklabels(ytick_labels, fontsize=9)
    ax.set_xlabel("% con discrepancia")
    ax.set_title(
        f"Discrepancia por rol y año de la encuesta (usuarios de IA, N = {_num(len(df))})"
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    fig.tight_layout()
    if year_exclusive:
        fig.subplots_adjust(bottom=0.08)
        fig.text(0.02, 0.01, YEAR_EXCLUSIVE_NOTE, fontsize=8, style="italic", ha="left")

    return tabla, fig


def wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    """Intervalo de Wilson del 95 % para una proporción."""
    p = k / n
    centro = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    margen = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centro - margen, centro + margen


def logit_or(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Regresión logística sin penalización; odds ratio e intervalo de Wald del 95 % por variable."""
    from sklearn.linear_model import LogisticRegression

    modelo = LogisticRegression(penalty=None, max_iter=5000).fit(X, y)
    X1 = np.column_stack([np.ones(len(X)), X.to_numpy(dtype=float)])
    p = modelo.predict_proba(X)[:, 1]
    cov = np.linalg.inv(X1.T @ (X1 * (p * (1 - p))[:, None]))
    beta = modelo.coef_[0]
    se = np.sqrt(np.diag(cov))[1:]
    return pd.DataFrame({
        "OR": np.exp(beta),
        "IC_inf": np.exp(beta - 1.96 * se),
        "IC_sup": np.exp(beta + 1.96 * se),
    }, index=X.columns)


def tasa_estandarizada(df: pd.DataFrame, mask: pd.Series) -> tuple[float, float, float]:
    """Tasa de discrepancia de un grupo ajustada por año (estandarización directa).

    Pondera la tasa del grupo en cada año por la proporción de ese año en el total, así
    todos los grupos quedan con la misma mezcla de años.
    """
    pesos = df["survey_year"].value_counts(normalize=True)
    tasa, var = 0.0, 0.0
    for year, w in pesos.items():
        sub = df.loc[mask & (df["survey_year"] == year), "discrepancia"]
        p = sub.mean()
        tasa += w * p
        var += w**2 * p * (1 - p) / len(sub)
    return tasa, tasa - 1.96 * np.sqrt(var), tasa + 1.96 * np.sqrt(var)


def h1_ajustada(df: pd.DataFrame) -> dict:
    """H1 ajustada por año: N y tasa por tramo y año, tasas estandarizadas y odds ratios.

    El año confunde la relación porque en 2025 hay más discrepancia y más personas con
    muchos años programando.
    """
    df = df.copy()
    df["exp_tramo"] = tramo_experiencia(df["YearsCode"])
    anios = pd.get_dummies(df["survey_year"], prefix="anio", drop_first=True, dtype=float)

    filas = []
    for (year, tramo), sub in df.groupby(["survey_year", "exp_tramo"], observed=True):
        k, n = sub["discrepancia"].sum(), len(sub)
        lo, hi = wilson(k, n)
        filas.append({"Año": int(year), "Tramo": tramo, "N": n,
                      "Tasa %": 100 * k / n, "IC inf %": 100 * lo, "IC sup %": 100 * hi})
    por_tramo_anio = pd.DataFrame(filas)

    filas = []
    for tramo in df["exp_tramo"].cat.categories:
        mask = df["exp_tramo"] == tramo
        tasa, lo, hi = tasa_estandarizada(df, mask)
        filas.append({"Tramo": tramo, "Tasa cruda %": 100 * df.loc[mask, "discrepancia"].mean(),
                      "Tasa ajustada %": 100 * tasa, "IC inf %": 100 * lo, "IC sup %": 100 * hi})
    estandarizada = pd.DataFrame(filas).set_index("Tramo")

    # Odds ratio por cada 10 años programando, sin ajustar y ajustado por año.
    x10 = (df[["YearsCode"]] / 10).rename(columns={"YearsCode": "10 años programando"})
    or_crudo = logit_or(x10, df["discrepancia"]).iloc[0]
    or_ajustado = logit_or(pd.concat([x10, anios], axis=1), df["discrepancia"]).iloc[0]

    tramos = pd.get_dummies(df["exp_tramo"], prefix="tramo", drop_first=True, dtype=float)
    or_tramos = logit_or(pd.concat([tramos, anios], axis=1), df["discrepancia"]).loc[tramos.columns]

    print(f"  OR por 10 años: crudo {or_crudo['OR']:.3f} [{or_crudo['IC_inf']:.3f}, {or_crudo['IC_sup']:.3f}], "
          f"ajustado por año {or_ajustado['OR']:.3f} [{or_ajustado['IC_inf']:.3f}, {or_ajustado['IC_sup']:.3f}]")
    print(estandarizada.round(1).to_string())
    return {"por_tramo_anio": por_tramo_anio, "estandarizada": estandarizada,
            "or_crudo": or_crudo, "or_ajustado": or_ajustado, "or_tramos": or_tramos}


def h2_ajustada(df: pd.DataFrame) -> pd.DataFrame:
    """H2 ajustada por año: tasa cruda y estandarizada por rol, con intervalo del 95 %."""
    filas = []
    for col in [c for c in df.columns if c.startswith("DevType_")]:
        mask = df[col] == 1
        tasa, lo, hi = tasa_estandarizada(df, mask)
        filas.append({"Rol": col.replace("DevType_", ""), "N": int(mask.sum()),
                      "Tasa cruda %": 100 * df.loc[mask, "discrepancia"].mean(),
                      "Tasa ajustada %": 100 * tasa, "IC inf %": 100 * lo, "IC sup %": 100 * hi})
    tabla = pd.DataFrame(filas).set_index("Rol").sort_values("Tasa ajustada %", ascending=False)
    print(tabla.round(1).to_string())
    return tabla


def generar_lectura(
    df: pd.DataFrame,
    tabla_base: pd.DataFrame,
    tabla_h1: pd.DataFrame,
    tabla_h2: pd.DataFrame,
    h1_aj: dict,
    h2_aj: pd.DataFrame,
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

    out.append("## 4. H1 y H2 ajustadas por año\n\n")
    out.append(
        "El año de la encuesta puede confundir las comparaciones, porque en 2025 hay más discrepancia "
        "y la mezcla de experiencia y de roles cambia entre ediciones. Se ajusta de dos formas. "
        "La tasa ajustada pondera la tasa de cada año por la proporción de ese año en el total "
        "(estandarización directa), así todos los grupos quedan con la misma mezcla de años. "
        "Los odds ratios salen de una regresión logística con el año como covariable "
        "(intervalos de Wald del 95 %). Los intervalos de las tasas por tramo y año son de Wilson.\n\n"
    )
    out.append("### H1: N y tasa por tramo y año\n\n")
    out.append(h1_aj["por_tramo_anio"].round(1).to_markdown(index=False))
    out.append("\n\n### H1: tasa cruda y ajustada por año, por tramo\n\n")
    out.append(h1_aj["estandarizada"].round(1).to_markdown())
    oc, oa = h1_aj["or_crudo"], h1_aj["or_ajustado"]
    out.append(
        f"\n\n**Odds ratio por cada 10 años programando:** sin ajustar {oc['OR']:.3f} "
        f"[{oc['IC_inf']:.3f}, {oc['IC_sup']:.3f}]; ajustado por año {oa['OR']:.3f} "
        f"[{oa['IC_inf']:.3f}, {oa['IC_sup']:.3f}].\n\n"
    )
    out.append("**Odds ratio de cada tramo contra 0–2 años, ajustado por año:**\n\n")
    out.append(h1_aj["or_tramos"].round(3).to_markdown())
    out.append("\n\n### H2: tasa cruda y ajustada por año, por rol\n\n")
    out.append(h2_aj.round(1).to_markdown())
    out.append("\n\n")

    out.append("## 5. En resumen\n\n")
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

    print("H1 ajustada por año")
    h1_aj = h1_ajustada(df)
    print("\nH2 ajustada por año")
    h2_aj = h2_ajustada(df)

    md = generar_lectura(df, tabla_base, tabla_h1, tabla_h2, h1_aj, h2_aj)
    doc_path = DOCS / "resultados_temporal_usuarios.md"
    doc_path.write_text(md)
    print(f"Guardado {doc_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
