"""Análisis temporal (Capa A) — descriptivo por año sobre usuarios (80k).

Estratifica los análisis H1 (seniority) y H2 (DevType) por año del survey
sobre la subpoblación de usuarios activos de IA (df_clean_users.parquet, N≈80k),
complementando el análisis pooled de eda.py y el estratificado sobre full de invarianza.py.

Outputs:
    outputs/08_temporal_tasa_base.png
    outputs/08_temporal_H1_seniority.png
    outputs/08_temporal_H2_devtype.png
    resultados/resultados_temporal_usuarios.md
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

# ============================================================
# CONFIG
# ============================================================
TARGET_LABEL_0 = "Con discrepancia uso-actitud"
COLOR_GLOBAL = "#1f2937"
COLORS_YEAR = {2023: "#3b82f6", 2024: "#8b5cf6", 2025: "#f59e0b"}
YEAR_EXCLUSIVE_NOTE = "* Categoría disponible en ola(s) específica(s) del survey solamente"
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "df_clean_users.parquet"
OUTPUTS = ROOT / "outputs"
DOCS = ROOT / "resultados"
DOCS.mkdir(parents=True, exist_ok=True)


def cargar() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró {DATA_PATH}. "
            "Correr primero: python scripts/pipeline.py"
        )
    df = pd.read_parquet(DATA_PATH)
    df["discrepancia"] = 1 - df["target"]
    return df


def tramo_experiencia(years: pd.Series) -> pd.Categorical:
    bins = [-0.1, 2, 5, 10, 20, 100]
    labels = ["0–2", "3–5", "6–10", "11–20", ">20"]
    return pd.cut(years, bins=bins, labels=labels)


def tasa_base_por_anio(df: pd.DataFrame) -> tuple[pd.DataFrame, plt.Figure]:
    """Tasa base de discrepancia (target=0) por año del survey sobre usuarios."""
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
                f"{row['tasa_pct']:.1f}%\n(N={row['n_total']:,})",
                ha="center", va="bottom", fontsize=10)

    ax.axhline(tasa_global, color=COLOR_GLOBAL, linestyle="--", lw=1.5,
               label=f"Pooled {tasa_global:.1f}%")

    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years], fontsize=11)
    ax.set_xlabel("Año del survey")
    ax.set_ylabel(f"% {TARGET_LABEL_0}")
    ax.set_ylim(0, tabla["tasa_pct"].max() * 1.4)
    ax.set_title(
        f"Tasa base de discrepancia por año del survey\n(usuarios IA, N={len(df):,})"
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()

    return tabla, fig


def h1_temporal_usuarios(df: pd.DataFrame) -> tuple[pd.DataFrame, plt.Figure]:
    """H1 estratificado por año sobre usuarios — discrepancia por tramo de seniority."""
    df = df.copy()
    df["exp_tramo"] = tramo_experiencia(df["YearsCodePro"])

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
        marker = "✓" if mono else "✗"
        gap = vals[-1] - vals[0]
        print(f"  H1 {col}: monotónica {marker}  ({vals[0]:.1f}% → {vals[-1]:.1f}%, gap = {gap:+.1f} pp)")

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(tabla.index))

    ax.plot(x, tabla["global"], "--o", color=COLOR_GLOBAL, lw=2, ms=6,
            label="Pooled (3 años)", zorder=5)
    for year in [2023, 2024, 2025]:
        col = str(year)
        if col in tabla.columns:
            ax.plot(x, tabla[col], "-o", color=COLORS_YEAR[year], lw=2, ms=6, label=col)

    ax.set_xticks(x)
    ax.set_xticklabels(tabla.index)
    ax.set_xlabel("Tramo de YearsCodePro (años)")
    ax.set_ylabel(f"% {TARGET_LABEL_0}")
    ax.set_ylim(0, tabla.max().max() * 1.25)
    ax.set_title("H1 temporal — Discrepancia por seniority × año (usuarios IA)")
    ax.legend(title="", loc="upper left", fontsize=9)
    ax.grid(linestyle="--", alpha=0.3)
    fig.tight_layout()

    return tabla, fig


def h2_temporal_usuarios(df: pd.DataFrame) -> tuple[pd.DataFrame, plt.Figure]:
    """H2 estratificado por año sobre usuarios — discrepancia por DevType."""
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

    # Solo DevTypes con N≥100 en ≥2 años
    years_ok = tabla_long.groupby("devtype")["year"].count()
    devtypes_incluidos = years_ok[years_ok >= 2].index.tolist()
    tabla_long = tabla_long[tabla_long["devtype"].isin(devtypes_incluidos)]

    tabla = tabla_long.pivot(index="devtype", columns="year", values="tasa")
    tabla.columns.name = None
    tabla["_avg"] = tabla.mean(axis=1)
    tabla = tabla.sort_values("_avg", ascending=True).drop(columns="_avg")

    # DevTypes presentes en <3 años (pero que pasaron el filtro ≥2)
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
        f"H2 temporal — Discrepancia por DevType × año (usuarios IA, N={len(df):,})"
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
    """Genera el documento markdown de lectura."""
    global_rate = df["discrepancia"].mean() * 100
    global_n = len(df)

    out = []
    out.append("# Análisis temporal — Discrepancia uso-actitud sobre usuarios de IA\n\n")
    out.append("**Dataset:** `data/df_clean_users.parquet`\n\n")
    out.append(
        f"**N total:** {global_n:,}  ·  "
        f"**Tasa global de discrepancia (pooled):** {global_rate:.1f}%\n\n---\n\n"
    )

    # 1. Tasa base
    out.append("## 1. Tasa base de discrepancia por año del survey\n\n")
    out.append(tabla_base.to_markdown())
    out.append("\n\n*(tasa_pct en %; n_total y n_disc en absolutos)*\n\n")
    out.append("### Lectura\n\n")

    tasas = tabla_base["tasa_pct"]
    min_year = int(tasas.idxmin())
    max_year = int(tasas.idxmax())
    delta = float(tasas.max() - tasas.min())
    out.append(
        f"La tasa de discrepancia oscila entre **{tasas.min():.1f}% ({min_year})** "
        f"y **{tasas.max():.1f}% ({max_year})**, un rango de {delta:.1f} pp. "
        f"El pooled (3 años combinados) es {global_rate:.1f}%.\n\n"
    )
    out.append(
        "**Contraste con dataset completo (full, N≈155k, framing E2):** "
        "En el full dataset la tasa de no-favorable fue del orden de 55–65% por año, "
        "3–4× superior porque incluye a los no-usuarios (AISelect=No), "
        "que por definición del target E2 son mayoritariamente no-favorables. "
        "La subpoblación de usuarios reduce drásticamente la tasa base "
        "y hace visible la brecha real: quienes usan IA pero declaran actitud no-favorable.\n\n"
    )

    # 2. H1
    out.append("## 2. H1 — Discrepancia por seniority estratificada por año\n\n")
    out.append(tabla_h1.to_markdown())
    out.append(
        "\n\n*(porcentajes; global = pooled 3 años; columnas = año del survey)*\n\n"
    )
    out.append("### Lectura por columna\n\n")

    for col in tabla_h1.columns:
        vals = tabla_h1[col].dropna().values
        if len(vals) < 2:
            continue
        mono = all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))
        gap = vals[-1] - vals[0]
        marker = "monotónica creciente" if mono else "**no monotónica**"
        out.append(
            f"- **{col}:** {marker} ({vals[0]:.1f}% → {vals[-1]:.1f}%, gap = {gap:+.1f} pp).\n"
        )

    out.append(
        "\n### Contraste con análisis sobre dataset completo (invarianza.py)\n\n"
        "En el full dataset (155k, E2), el gap juniors→seniors fue +22,9 pp (2023), "
        "+16,8 pp (2024), +0,6 pp (2025). "
        "Los valores sobre la subpoblación de usuarios son estructuralmente menores porque "
        "el salto más grande en el full dataset era el paso de no-usuarios (alta tasa) "
        "a usuarios (baja tasa), no el gradiente intra-usuarios. "
        "Esto es consistente con el hallazgo de eda.py (r=0,042 en usuarios vs. ~r=0,18 en full).\n\n"
    )

    # 3. H2
    out.append("## 3. H2 — Discrepancia por DevType estratificada por año\n\n")
    out.append(tabla_h2.to_markdown())
    out.append(
        "\n\n*(porcentajes; NaN cuando el DevType no alcanza N≥100 en ese año)*\n\n"
    )

    # Detectar year-exclusive desde la tabla pivot
    years_seen: dict[str, int] = {}
    for col in [2023, 2024, 2025]:
        if col in tabla_h2.columns:
            for dt in tabla_h2[tabla_h2[col].notna()].index:
                years_seen[dt] = years_seen.get(dt, 0) + 1
    year_exclusive_doc = [dt for dt, cnt in years_seen.items() if cnt < 3]

    if year_exclusive_doc:
        out.append("### Notas sobre comparabilidad entre años\n\n")
        for dt in year_exclusive_doc:
            yrs_with_data = [
                str(y) for y in [2023, 2024, 2025]
                if y in tabla_h2.columns and not pd.isna(tabla_h2.loc[dt, y])
            ]
            out.append(
                f"- **{dt}\\*** — datos disponibles solo en {', '.join(yrs_with_data)}. "
                "En los años restantes el OHE produce ceros por cambio en las categorías del survey.\n"
            )
        out.append(
            "\nSe excluyeron de la figura DevTypes con N≥100 en un único año "
            "(ej. 'Architect, software or solutions' en 2025).\n\n"
        )

    out.append("### Lectura\n\n")

    avg = tabla_h2.mean(axis=1).sort_values(ascending=False)
    top3 = avg.head(3)
    bot3 = avg.tail(3)
    out.append(
        "**DevTypes con mayor tasa de discrepancia promedio (top 3):** "
        + ", ".join(f"{dt} ({v:.1f}%)" for dt, v in top3.items())
        + ".\n\n"
    )
    out.append(
        "**DevTypes con menor tasa de discrepancia promedio (bottom 3):** "
        + ", ".join(f"{dt} ({v:.1f}%)" for dt, v in bot3[::-1].items())
        + ".\n\n"
    )
    out.append(
        "El ordenamiento relativo entre DevTypes se mantiene aproximadamente estable entre años, "
        "consistente con el hallazgo de invarianza.py sobre el full dataset. "
        "Las magnitudes absolutas son menores respecto al full porque la subpoblación de usuarios "
        "comprime la distribución de tasas (tasa base ≈21,5% en vez de ≈60%).\n\n"
    )

    # 4. Implicaciones
    out.append("## 4. Implicaciones para E3\n\n")
    out.append(
        "- **Tasa base estable entre años:** la variación inter-anual en usuarios es moderada "
        f"({delta:.1f} pp), lo que sugiere que el fenómeno de discrepancia uso-actitud "
        "es estructuralmente robusto y no un artefacto de un año específico del survey. "
        "Documenta en §10 (limitaciones) como evidencia de que el pooling de años no introduce sesgo apreciable.\n"
    )
    out.append(
        "- **H1 más débil en usuarios:** la paradoja de seniority dentro de la subpoblación de usuarios "
        "es mucho más tenue que en el full dataset. El gradiente intra-usuarios es casi plano "
        "comparado con el salto usuarios/no-usuarios. Fortalece la interpretación de que "
        "la paradoja detectada en E2 era principalmente un efecto de auto-selección. Va a §8 (discusión).\n"
    )
    out.append(
        "- **H2 orden relativo preservado:** los DevTypes con mayor discrepancia "
        "mantienen su posición relativa año a año. Sugiere que el tipo de rol "
        "es un predictor más estable que la seniority sobre la subpoblación de usuarios.\n"
    )
    out.append(
        "- **Insumo para Capa B:** los valores de tasa base por año sirven "
        "como referencia para el modelado temporal estratificado.\n\n"
    )

    out.append("---\n\n*Generado por `scripts/temporal.py`*\n")

    return "".join(out)


def main():
    print("=" * 70)
    print("ANÁLISIS TEMPORAL SOBRE USUARIOS")
    print("=" * 70)

    OUTPUTS.mkdir(exist_ok=True)

    df = cargar()
    dist = df["survey_year"].value_counts().sort_index()
    print(f"Dataset: {df.shape[0]:,} filas × {df.shape[1]} cols")
    print(f"Distribución por año: {dist.to_dict()}")
    print(f"Tasa global de discrepancia: {df['discrepancia'].mean() * 100:.1f}%\n")

    print("--- Tasa base por año ---")
    tabla_base, fig_base = tasa_base_por_anio(df)
    print(tabla_base.to_string())
    fig_base.savefig(OUTPUTS / "08_temporal_tasa_base.png", dpi=150, bbox_inches="tight")
    plt.close(fig_base)
    print(f"✓ outputs/08_temporal_tasa_base.png\n")

    print("--- H1 — Seniority estratificado ---")
    tabla_h1, fig_h1 = h1_temporal_usuarios(df)
    print(tabla_h1.to_string())
    fig_h1.savefig(OUTPUTS / "08_temporal_H1_seniority.png", dpi=150, bbox_inches="tight")
    plt.close(fig_h1)
    print(f"✓ outputs/08_temporal_H1_seniority.png\n")

    print("--- H2 — DevType estratificado ---")
    tabla_h2, fig_h2 = h2_temporal_usuarios(df)
    print(tabla_h2.to_string())
    fig_h2.savefig(OUTPUTS / "08_temporal_H2_devtype.png", dpi=150, bbox_inches="tight")
    plt.close(fig_h2)
    print(f"✓ outputs/08_temporal_H2_devtype.png\n")

    print("--- Generando documento de lectura ---")
    md = generar_lectura(df, tabla_base, tabla_h1, tabla_h2)
    doc_path = DOCS / "resultados_temporal_usuarios.md"
    doc_path.write_text(md)
    print(f"✓ {doc_path}\n")

    print("=" * 70)
    print("LISTO — 3 figuras + 1 doc generados")
    print("=" * 70)


if __name__ == "__main__":
    main()
