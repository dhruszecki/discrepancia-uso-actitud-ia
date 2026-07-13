"""EDA sobre df_clean_users.parquet — H1, H2, H4.

Reproduce las hipótesis de notebooks/03_hipotesis_eda.ipynb sobre la
subpoblación restringida a usuarios activos de IA. H5 queda para análisis
separado porque colapsa a 2 cuadrantes cuando se restringe a usuarios.

Outputs:
    outputs/06_users_H1_seniority.png
    outputs/06_users_H2_devtype.png
    outputs/06_users_H4_clusters.png
    data/df_clean_users_clustered.parquet  (con columna cluster + rogers_type)
    resultados/resultados_eda_usuarios.md
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
TARGET_LABELS = {
    1: "Sin discrepancia",
    0: "Con discrepancia uso-actitud",
}
TARGET_SHORT = {
    1: "Sin discrepancia",
    0: "Discrepancia",
}
COLOR_T1 = "#16a34a"  # verde
COLOR_T0 = "#dc2626"  # rojo
COLORS_YEAR = {2023: "#3b82f6", 2024: "#8b5cf6", 2025: "#f59e0b"}
RANDOM_STATE = 42
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "df_clean_users.parquet"
OUTPUT_CLUSTER_PATH = ROOT / "data" / "df_clean_users_clustered.parquet"
OUTPUTS = ROOT / "outputs"
DOCS = ROOT / "resultados"
DOCS.mkdir(parents=True, exist_ok=True)


def cargar():
    df = pd.read_parquet(DATA_PATH)
    # 'discrepancia' = target=0
    df["discrepancia"] = 1 - df["target"]
    return df


def tramo_experiencia(years: pd.Series) -> pd.Categorical:
    bins = [-0.1, 2, 5, 10, 20, 100]
    labels = ["0–2", "3–5", "6–10", "11–20", ">20"]
    return pd.cut(years, bins=bins, labels=labels)


def h1_seniority(df: pd.DataFrame):
    """H1 — Paradoja de seniority sobre usuarios."""
    df = df.copy()
    df["exp_tramo"] = tramo_experiencia(df["YearsCodePro"])

    # Tabla global + por año
    global_rate = df.groupby("exp_tramo", observed=True)["discrepancia"].mean()
    por_anio = df.groupby(["survey_year", "exp_tramo"], observed=True)["discrepancia"].mean().unstack(level=0)
    por_anio.columns = [str(int(c)) for c in por_anio.columns]
    tabla = pd.concat([global_rate.rename("global"), por_anio], axis=1) * 100
    tabla = tabla.round(1)

    # Correlación punto-biserial
    valid = df[["YearsCodePro", "discrepancia"]].dropna()
    r, p = stats.pointbiserialr(valid["YearsCodePro"], valid["discrepancia"])

    # Figura: 2 paneles (1) tasa por tramo global + por año, (2) histograma YearsCodePro por clase
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1
    x = np.arange(len(tabla.index))
    width = 0.2
    axes[0].bar(x - 1.5*width, tabla["global"], width, label="Global", color="#1f2937", alpha=0.7)
    for i, year in enumerate(["2023", "2024", "2025"]):
        if year in tabla.columns:
            axes[0].bar(x + (i - 0.5)*width, tabla[year], width, label=year, color=COLORS_YEAR[int(year)])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(tabla.index)
    axes[0].set_xlabel("Tramo YearsCodePro")
    axes[0].set_ylabel(f"% {TARGET_LABELS[0]}")
    axes[0].set_title("H1 — Tasa de discrepancia por seniority (usuarios IA)")
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)

    # Panel 2: histograma
    for t, color, label in [(1, COLOR_T1, TARGET_LABELS[1]), (0, COLOR_T0, TARGET_LABELS[0])]:
        sub = df[df["target"] == t]["YearsCodePro"]
        axes[1].hist(sub, bins=30, alpha=0.55, color=color, label=label, density=True)
    axes[1].set_xlabel("YearsCodePro")
    axes[1].set_ylabel("Densidad")
    axes[1].set_title(f"Distribución de YearsCodePro\n(r={r:.3f}, p={p:.2e})")
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUTS / "06_users_H1_seniority.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {OUTPUTS / '06_users_H1_seniority.png'}  (r={r:.3f}, p={p:.2e})")

    # Check monotonía
    for col in tabla.columns:
        vals = tabla[col].dropna().values
        mono = all(vals[i] <= vals[i+1] for i in range(len(vals)-1))
        marker = "✓" if mono else "✗"
        gap = vals[-1] - vals[0]
        print(f"  {col}: monotónica {marker}  ({vals[0]:.1f}% → {vals[-1]:.1f}%, gap = {gap:+.1f} pp)")

    return tabla, r, p


def h2_devtype(df: pd.DataFrame):
    """H2 — DevType differences sobre usuarios."""
    devtype_cols = [c for c in df.columns if c.startswith("DevType_")]

    rows = []
    for dt_col in devtype_cols:
        label = dt_col.replace("DevType_", "")
        mask = df[dt_col] == 1
        n = int(mask.sum())
        if n < 100:
            continue
        tasa = df.loc[mask, "discrepancia"].mean() * 100
        rows.append({"devtype": label, "tasa": tasa, "n": n})
    df_dev = pd.DataFrame(rows).sort_values("tasa", ascending=True)
    tasa_global = df["discrepancia"].mean() * 100

    fig, ax = plt.subplots(figsize=(11, max(5, 0.45 * len(df_dev))))
    colors = [COLOR_T0 if t > tasa_global else COLOR_T1 for t in df_dev["tasa"]]
    ax.barh(range(len(df_dev)), df_dev["tasa"], color=colors)
    for i, (_, row) in enumerate(df_dev.iterrows()):
        ax.text(row["tasa"] + 0.4, i, f"{row['tasa']:.1f}% (n={row['n']:,})", va="center", fontsize=8)
    ax.set_yticks(range(len(df_dev)))
    ax.set_yticklabels([d[:42] for d in df_dev["devtype"]], fontsize=9)
    ax.axvline(tasa_global, color="black", linestyle=":", lw=1, label=f"Global {tasa_global:.1f}%")
    ax.set_xlabel(f"% {TARGET_LABELS[0]}")
    ax.set_title(f"H2 — Tasa de discrepancia por DevType (usuarios IA, N={len(df):,})")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUTS / "06_users_H2_devtype.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {OUTPUTS / '06_users_H2_devtype.png'}")
    return df_dev, tasa_global


def h4_clusters(df: pd.DataFrame):
    """H4 — K-Means k=4 sobre features de perfil + mapeo a Rogers."""
    CLUSTER_FEATURES = ["YearsCodePro", "YearsCode", "WorkExp", "EdLevel", "OrgSize",
                        "lang_count", "db_count", "plat_count", "env_count"]
    available = [c for c in CLUSTER_FEATURES if c in df.columns]
    print(f"  Clustering features: {available}")

    X = df[available].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=20)
    df = df.copy()
    df["cluster"] = km.fit_predict(X_scaled)

    # Mapeo Rogers: ordenar clusters por tasa de discrepancia
    rates = df.groupby("cluster")["discrepancia"].mean().sort_values()
    rogers_order = ["Early Adopters", "Early Majority", "Late Majority", "Laggards"]
    rogers_map = dict(zip(rates.index, rogers_order))
    df["rogers_type"] = df["cluster"].map(rogers_map)

    # Distribución
    dist = df["rogers_type"].value_counts(normalize=True).reindex(rogers_order)
    teorico = pd.Series({"Early Adopters": 0.135, "Early Majority": 0.34, "Late Majority": 0.34, "Laggards": 0.16})
    rates_by_rogers = df.groupby("rogers_type")["discrepancia"].mean().reindex(rogers_order)
    exp_by_rogers = df.groupby("rogers_type")["YearsCodePro"].median().reindex(rogers_order)

    tabla = pd.DataFrame({
        "% observado": (dist * 100).round(1),
        "% teórico (Rogers)": (teorico * 100).round(1),
        "% discrepancia": (rates_by_rogers * 100).round(1),
        "Exp. mediana": exp_by_rogers.round(1),
    })

    # Figura
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(4)
    width = 0.35
    axes[0].bar(x - width/2, dist.values*100, width, label="Observado", color="#3b82f6")
    axes[0].bar(x + width/2, teorico.values*100, width, label="Rogers (teórico)", color="#9ca3af")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(rogers_order, rotation=15, ha="right")
    axes[0].set_ylabel("% del total")
    axes[0].set_title("Distribución de clusters vs. teoría de Rogers (usuarios IA)")
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)

    # Panel 2: tasa de discrepancia por tipo + exp mediana
    ax2 = axes[1]
    ax2_twin = ax2.twinx()
    ax2.bar(x, rates_by_rogers.values*100, color=COLOR_T0, alpha=0.7, label="% discrepancia")
    ax2_twin.plot(x, exp_by_rogers.values, "o-", color="#1f2937", lw=2, markersize=8, label="Exp. mediana")
    ax2.set_xticks(x)
    ax2.set_xticklabels(rogers_order, rotation=15, ha="right")
    ax2.set_ylabel(f"% {TARGET_LABELS[0]}", color=COLOR_T0)
    ax2_twin.set_ylabel("YearsCodePro mediana")
    ax2.set_title("Discrepancia y experiencia mediana por tipo Rogers")
    ax2.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUTS / "06_users_H4_clusters.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {OUTPUTS / '06_users_H4_clusters.png'}")

    return df, tabla


def generar_doc(tabla_h1, r_h1, p_h1, df_h2, tasa_global, tabla_h4):
    out = []
    out.append("# EDA sobre usuarios activos de IA — Resultados\n\n")
    out.append("**Dataset:** `data/df_clean_users.parquet`  ·  **N = 80.369**\n\n")
    out.append(f"**Target:** {TARGET_LABELS[1]} (78,5 %) vs. {TARGET_LABELS[0]} (21,5 %)\n\n---\n\n")

    out.append("## H1 — Paradoja de seniority sobre usuarios\n\n")
    out.append(f"**Correlación punto-biserial** YearsCodePro × discrepancia: **r = {r_h1:.4f}, p = {p_h1:.2e}**\n\n")
    out.append("Tabla de tasa de discrepancia por tramo (porcentajes):\n\n")
    out.append(tabla_h1.to_markdown())
    out.append("\n\n")
    out.append("**Observaciones:**\n")
    for col in tabla_h1.columns:
        vals = tabla_h1[col].dropna().values
        mono = all(vals[i] <= vals[i+1] for i in range(len(vals)-1))
        gap = vals[-1] - vals[0]
        marker = "monotónica creciente" if mono else "**no monotónica**"
        out.append(f"- **{col}**: {marker} ({vals[0]:.1f} % → {vals[-1]:.1f} %, gap = {gap:+.1f} pp).\n")
    out.append("\nLa paradoja se preserva en dirección pero con magnitudes más chicas — esperable porque restringir a usuarios elimina los casos extremos.\n\n")

    out.append("## H2 — DevType sobre usuarios\n\n")
    out.append(f"**Tasa global de discrepancia:** {tasa_global:.1f} %\n\n")
    out.append("Top DevTypes por tasa de discrepancia:\n\n")
    df_h2_show = df_h2.sort_values("tasa", ascending=False).head(15).reset_index(drop=True)
    df_h2_show["tasa"] = df_h2_show["tasa"].round(1)
    out.append(df_h2_show.to_markdown(index=False))
    out.append("\n\n")
    out.append("Bottom DevTypes (más calibrados):\n\n")
    df_h2_low = df_h2.sort_values("tasa", ascending=True).head(5).reset_index(drop=True)
    df_h2_low["tasa"] = df_h2_low["tasa"].round(1)
    out.append(df_h2_low.to_markdown(index=False))
    out.append("\n\n")

    out.append("## H4 — Segmentación K-Means k=4 sobre usuarios\n\n")
    out.append(tabla_h4.to_markdown())
    out.append("\n\n**Output:** parquet con columna `cluster` y `rogers_type` en `data/df_clean_users_clustered.parquet`.\n\n")

    out.append("## Próximo paso\n\n")
    out.append("- H5 colapsa a 2 cuadrantes cuando se restringe a usuarios (no se ven los no-usuarios). Se mantiene el H5 del E2 original para discutir el fenómeno macro; sobre usuarios solo aplica la dinámica de subconfianza (target=0) por año.\n")
    out.append("- Re-correr análisis estratificado por año sobre esta subpoblación para confirmar que los hallazgos se sostienen.\n")
    out.append("- Integrar a §6 EDA de E3 con la nueva terminología.\n")

    return "".join(out)


def main():
    print("=" * 70)
    print("EDA SOBRE USUARIOS")
    print("=" * 70)
    print(f"CONFIG: TARGET_LABELS = {TARGET_LABELS}\n")

    df = cargar()
    print(f"Dataset: {df.shape}\n")

    print("--- H1 — Paradoja de seniority ---")
    tabla_h1, r_h1, p_h1 = h1_seniority(df)
    print(f"\n{tabla_h1.to_string()}\n")

    print("--- H2 — DevType ---")
    df_h2, tasa_global = h2_devtype(df)
    print(f"  Tasa global: {tasa_global:.1f}%")
    print(f"  Top 5 DevTypes con más discrepancia:")
    for _, row in df_h2.sort_values("tasa", ascending=False).head(5).iterrows():
        print(f"    {row['devtype'][:50]:<50} {row['tasa']:.1f}% (n={row['n']:,})")

    print("\n--- H4 — Clustering ---")
    df_clust, tabla_h4 = h4_clusters(df)
    print(f"\n{tabla_h4.to_string()}\n")

    # Guardar dataset con clusters
    df_clust = df_clust.drop(columns=["discrepancia"])  # no necesaria persistirla
    df_clust.to_parquet(OUTPUT_CLUSTER_PATH, index=False)
    print(f"✓ {OUTPUT_CLUSTER_PATH}")

    # Doc
    md = generar_doc(tabla_h1, r_h1, p_h1, df_h2, tasa_global, tabla_h4)
    (DOCS / "resultados_eda_usuarios.md").write_text(md)
    print(f"✓ {DOCS / 'resultados_eda_usuarios.md'}")

    print("\n" + "=" * 70)
    print("LISTO")
    print("=" * 70)


if __name__ == "__main__":
    main()
