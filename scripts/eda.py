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

import sys
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nombres_legibles import configurar_figuras, nombre_rol, num  # noqa: E402

warnings.filterwarnings("ignore")
configurar_figuras()

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

    n_tramo = df.groupby("exp_tramo", observed=True).size()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(tabla.index))
    ax.bar(x, tabla["global"], color=COLOR_T0, alpha=0.8)
    for i, v in enumerate(tabla["global"]):
        ax.text(i, v - 0.6, f"{num(v)} %", ha="center", va="top", fontsize=10, color="white", fontweight="bold")
    ax.axhline(df["discrepancia"].mean() * 100, color="black", linestyle=":", lw=1,
               label=f"Media: {num(df['discrepancia'].mean() * 100)} %")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\n(N = {num(n_tramo[t], 0)})" for t in tabla.index])
    ax.set_xlabel("Años programando")
    ax.set_ylabel("% con discrepancia")
    ax.set_ylim(0, max(tabla["global"]) * 1.2)
    ax.set_title("Tasa de discrepancia según años programando (usuarios de IA, tres años juntos)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

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
        rows.append({"devtype": nombre_rol(label), "tasa": tasa, "n": n})
    df_dev = pd.DataFrame(rows).sort_values("tasa", ascending=True)
    tasa_global = df["discrepancia"].mean() * 100

    fig, ax = plt.subplots(figsize=(11, max(5, 0.45 * len(df_dev))))
    colors = [COLOR_T0 if t > tasa_global else COLOR_T1 for t in df_dev["tasa"]]
    ax.barh(range(len(df_dev)), df_dev["tasa"], color=colors)
    for i, (_, row) in enumerate(df_dev.iterrows()):
        ax.text(row["tasa"] + 0.4, i, f"{num(row['tasa'])} % (N = {num(row['n'], 0)})", va="center", fontsize=8)
    ax.set_yticks(range(len(df_dev)))
    ax.set_yticklabels(df_dev["devtype"], fontsize=9)
    ax.axvline(tasa_global, color="black", linestyle=":", lw=1, label=f"Media: {num(tasa_global)} %")
    ax.set_xlabel("% con discrepancia")
    ax.set_xlim(0, df_dev["tasa"].max() * 1.3)
    ax.set_title(f"Tasa de discrepancia por rol (usuarios de IA, N = {num(len(df), 0)})")
    import matplotlib.patches as mpatches
    linea_media = ax.get_legend_handles_labels()[0]
    ax.legend(handles=linea_media + [mpatches.Patch(color=COLOR_T0, label="Por encima de la media"),
                                     mpatches.Patch(color=COLOR_T1, label="Por debajo de la media")],
              loc="lower right", fontsize=9)
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

    nombres_x = [f"{e}\n(mediana {num(m, 0)} años)" for e, m in zip(etiquetas, experiencia.values)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(4)
    axes[0].bar(x, porcentaje.values, color="#3b82f6")
    for i, v in enumerate(porcentaje.values):
        axes[0].text(i, v + 0.8, f"{num(v)} %", ha="center", fontsize=9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(nombres_x, fontsize=9)
    axes[0].set_ylabel("% de los usuarios de IA")
    axes[0].set_title("Tamaño de cada grupo")
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)

    axes[1].bar(x, discrepancia.values, color=COLOR_T0, alpha=0.7)
    for i, v in enumerate(discrepancia.values):
        axes[1].text(i, v - 0.6, f"{num(v)} %", ha="center", va="top", fontsize=9, color="white", fontweight="bold")
    axes[1].axhline(df["discrepancia"].mean() * 100, color="black", linestyle=":", lw=1,
                    label=f"Media: {num(df['discrepancia'].mean() * 100)} %")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(nombres_x, fontsize=9)
    axes[1].set_ylabel("% con discrepancia")
    axes[1].set_ylim(0, discrepancia.max() * 1.25)
    axes[1].set_title("Discrepancia en cada grupo")
    axes[1].legend(loc="lower right", fontsize=9)
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)

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
