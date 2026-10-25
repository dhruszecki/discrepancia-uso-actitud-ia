"""Análisis exploratorio sobre los usuarios de IA (df_clean_users.parquet).

Incluye las dos hipótesis del trabajo (H1: experiencia, H2: rol) y una
segmentación descriptiva de perfiles con K-Means. La brecha entre uso y actitud
en toda la población está en brecha_uso_actitud.py, porque necesita usuarios y
no usuarios.

Genera las figuras usuarios_experiencia.png, usuarios_rol.png y
usuarios_grupos.png, el parquet con la columna cluster y el informe
docs/analisis/resultados_eda_usuarios.md.
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

TARGET_LABELS = {
    1: "Sin discrepancia",
    0: "Con discrepancia uso-actitud",
}
TARGET_SHORT = {
    1: "Sin discrepancia",
    0: "Discrepancia",
}
COLOR_T1 = "#16a34a"
COLOR_T0 = "#dc2626"
COLORS_YEAR = {2023: "#3b82f6", 2024: "#8b5cf6", 2025: "#f59e0b"}
RANDOM_STATE = 42

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "df_clean_users.parquet"
OUTPUT_CLUSTER_PATH = ROOT / "data" / "df_clean_users_clustered.parquet"
OUTPUTS = ROOT / "outputs"
DOCS = ROOT / "docs" / "analisis"
OUTPUTS.mkdir(parents=True, exist_ok=True)
DOCS.mkdir(parents=True, exist_ok=True)


def cargar():
    df = pd.read_parquet(DATA_PATH)
    df["discrepancia"] = 1 - df["target"]
    return df


def tramo_experiencia(years: pd.Series) -> pd.Categorical:
    bins = [-0.1, 2, 5, 10, 20, 100]
    labels = ["0–2", "3–5", "6–10", "11–20", ">20"]
    return pd.cut(years, bins=bins, labels=labels)


def h1_seniority(df: pd.DataFrame):
    """H1: la discrepancia crece con los años programando."""
    df = df.copy()
    df["exp_tramo"] = tramo_experiencia(df["YearsCode"])

    global_rate = df.groupby("exp_tramo", observed=True)["discrepancia"].mean()
    por_anio = df.groupby(["survey_year", "exp_tramo"], observed=True)["discrepancia"].mean().unstack(level=0)
    por_anio.columns = [str(int(c)) for c in por_anio.columns]
    tabla = pd.concat([global_rate.rename("global"), por_anio], axis=1) * 100
    tabla = tabla.round(1)

    valid = df[["YearsCode", "discrepancia"]].dropna()
    r, p = stats.pointbiserialr(valid["YearsCode"], valid["discrepancia"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    x = np.arange(len(tabla.index))
    width = 0.2
    axes[0].bar(x - 1.5*width, tabla["global"], width, label="Global", color="#1f2937", alpha=0.7)
    for i, year in enumerate(["2023", "2024", "2025"]):
        if year in tabla.columns:
            axes[0].bar(x + (i - 0.5)*width, tabla[year], width, label=year, color=COLORS_YEAR[int(year)])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(tabla.index)
    axes[0].set_xlabel("Años programando")
    axes[0].set_ylabel(f"% {TARGET_LABELS[0]}")
    axes[0].set_title("Tasa de discrepancia según años programando\n(usuarios de IA)")
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)

    for t, color, label in [(1, COLOR_T1, TARGET_LABELS[1]), (0, COLOR_T0, TARGET_LABELS[0])]:
        sub = df[df["target"] == t]["YearsCode"]
        axes[1].hist(sub, bins=30, alpha=0.55, color=color, label=label, density=True)
    axes[1].set_xlabel("Años programando")
    axes[1].set_ylabel("Densidad")
    axes[1].set_title(f"Distribución de los años programando\n(r = {r:.3f}, p = {p:.2e})")
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUTS / "usuarios_experiencia.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado outputs/usuarios_experiencia.png (r={r:.3f}, p={p:.2e})")

    for col in tabla.columns:
        vals = tabla[col].dropna().values
        mono = all(vals[i] <= vals[i+1] for i in range(len(vals)-1))
        estado = "creciente" if mono else "no creciente"
        diferencia = vals[-1] - vals[0]
        print(f"  {col}: {estado} (de {vals[0]:.1f}% a {vals[-1]:.1f}%, diferencia {diferencia:+.1f} puntos)")

    return tabla, r, p


def h2_devtype(df: pd.DataFrame):
    """H2: la discrepancia varía según el rol."""
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
    ax.set_title(f"Tasa de discrepancia por rol (usuarios de IA, N = {len(df):,})")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUTS / "usuarios_rol.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado outputs/usuarios_rol.png")
    return df_dev, tasa_global


def h4_clusters(df: pd.DataFrame):
    """Segmentación descriptiva con K-Means (k = 4) sobre variables de perfil.

    k = 4 se fija por interpretabilidad y no por un criterio de ajuste. Los grupos
    se numeran de menor a mayor mediana de años programando.
    """
    CLUSTER_FEATURES = ["YearsCode", "WorkExp", "EdLevel", "OrgSize",
                        "lang_count", "db_count", "plat_count", "env_count"]
    available = [c for c in CLUSTER_FEATURES if c in df.columns]
    print(f"  Variables de la segmentación: {available}")

    X = df[available].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=20)
    df = df.copy()
    df["cluster"] = km.fit_predict(X_scaled)

    # K-Means numera los grupos al azar; los reordeno por experiencia para que
    # la numeración sea estable y se pueda leer.
    orden = df.groupby("cluster")["YearsCode"].median().sort_values(kind="stable").index
    nombre = {c: f"Grupo {i + 1}" for i, c in enumerate(orden)}
    grupo = df["cluster"].map(nombre)
    etiquetas = list(nombre.values())

    porcentaje = grupo.value_counts(normalize=True).reindex(etiquetas) * 100
    discrepancia = df.groupby(grupo)["discrepancia"].mean().reindex(etiquetas) * 100
    experiencia = df.groupby(grupo)["YearsCode"].median().reindex(etiquetas)

    tabla = pd.DataFrame({
        "% de usuarios": porcentaje.round(1),
        "% discrepancia": discrepancia.round(1),
        "Experiencia mediana": experiencia.round(1),
    })

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(4)
    axes[0].bar(x, porcentaje.values, color="#3b82f6")
    for i, v in enumerate(porcentaje.values):
        axes[0].text(i, v + 0.8, f"{v:.1f} %", ha="center", fontsize=9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(etiquetas)
    axes[0].set_ylabel("% de los usuarios de IA")
    axes[0].set_title("Tamaño de cada grupo")
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)

    ax2 = axes[1]
    ax2_twin = ax2.twinx()
    ax2.bar(x, discrepancia.values, color=COLOR_T0, alpha=0.7)
    for i, v in enumerate(discrepancia.values):
        ax2.text(i, v + 0.4, f"{v:.1f} %", ha="center", fontsize=9)
    ax2_twin.plot(x, experiencia.values, "o-", color="#1f2937", lw=2, markersize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(etiquetas)
    ax2.set_ylabel("% con discrepancia", color=COLOR_T0)
    ax2_twin.set_ylabel("Años programando, mediana (línea)")
    ax2.set_title("Discrepancia y experiencia mediana por grupo")
    ax2.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUTS / "usuarios_grupos.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado outputs/usuarios_grupos.png")

    return df, tabla


def generar_doc(tabla_h1, r_h1, p_h1, df_h2, tasa_global, tabla_h4):
    out = []
    out.append("# Análisis exploratorio entre usuarios de IA: resultados\n\n")
    out.append("Datos: `data/df_clean_users.parquet`, N = 80.369. Generado por `scripts/eda.py`.\n\n")
    out.append(f"Variable a predecir: {TARGET_LABELS[1]} (78,5 %) frente a {TARGET_LABELS[0]} (21,5 %).\n\n")

    out.append("## H1: experiencia acumulada\n\n")
    out.append(f"La correlación punto-biserial entre años programando y discrepancia es r = {r_h1:.4f} (p = {p_h1:.2e}).\n\n")
    out.append("Tasa de discrepancia por tramo de experiencia (porcentajes):\n\n")
    out.append(tabla_h1.to_markdown())
    out.append("\n\n")
    out.append("Por columna:\n\n")
    for col in tabla_h1.columns:
        vals = tabla_h1[col].dropna().values
        mono = all(vals[i] <= vals[i+1] for i in range(len(vals)-1))
        gap = vals[-1] - vals[0]
        estado = "creciente en todos los tramos" if mono else "no creciente en todos los tramos"
        out.append(f"- {col}: {estado} (de {vals[0]:.1f} % a {vals[-1]:.1f} %, diferencia de {gap:+.1f} puntos).\n")
    out.append("\nLa diferencia entre tramos es chica: con esta muestra el valor de p es muy bajo, pero r explica menos del 0,2 % de la varianza.\n\n")

    out.append("## H2: tipo de rol\n\n")
    out.append(f"Tasa global de discrepancia: {tasa_global:.1f} %.\n\n")
    out.append("Roles con mayor tasa de discrepancia:\n\n")
    df_h2_show = df_h2.sort_values("tasa", ascending=False).head(15).reset_index(drop=True)
    df_h2_show["tasa"] = df_h2_show["tasa"].round(1)
    out.append(df_h2_show.to_markdown(index=False))
    out.append("\n\n")
    out.append("Roles con menor tasa de discrepancia:\n\n")
    df_h2_low = df_h2.sort_values("tasa", ascending=True).head(5).reset_index(drop=True)
    df_h2_low["tasa"] = df_h2_low["tasa"].round(1)
    out.append(df_h2_low.to_markdown(index=False))
    out.append("\n\n")

    out.append("## Segmentación de perfiles (K-Means, k = 4)\n\n")
    out.append("Es un análisis descriptivo, sin hipótesis previa. k = 4 se fijó por interpretabilidad. Los grupos se numeran de menor a mayor experiencia mediana.\n\n")
    out.append(tabla_h4.to_markdown())
    out.append("\n\nEl parquet `data/df_clean_users_clustered.parquet` agrega la columna `cluster`.\n")

    return "".join(out)


def main():
    print("Análisis exploratorio entre usuarios de IA")
    print(f"Etiquetas: {TARGET_LABELS}\n")

    df = cargar()
    print(f"Datos: {df.shape}\n")

    print("H1: experiencia")
    tabla_h1, r_h1, p_h1 = h1_seniority(df)
    print(f"\n{tabla_h1.to_string()}\n")

    print("H2: rol")
    df_h2, tasa_global = h2_devtype(df)
    print(f"  Tasa global: {tasa_global:.1f}%")
    print("  Los 5 roles con más discrepancia:")
    for _, row in df_h2.sort_values("tasa", ascending=False).head(5).iterrows():
        print(f"    {row['devtype'][:50]:<50} {row['tasa']:.1f}% (n={row['n']:,})")

    print("\nSegmentación de perfiles")
    df_clust, tabla_h4 = h4_clusters(df)
    print(f"\n{tabla_h4.to_string()}\n")

    # La columna discrepancia es auxiliar; el parquet guarda solo target.
    df_clust = df_clust.drop(columns=["discrepancia"])
    df_clust.to_parquet(OUTPUT_CLUSTER_PATH, index=False)
    print(f"Guardado {OUTPUT_CLUSTER_PATH.relative_to(ROOT)}")

    md = generar_doc(tabla_h1, r_h1, p_h1, df_h2, tasa_global, tabla_h4)
    (DOCS / "resultados_eda_usuarios.md").write_text(md)
    print(f"Guardado docs/analisis/resultados_eda_usuarios.md")


if __name__ == "__main__":
    main()
