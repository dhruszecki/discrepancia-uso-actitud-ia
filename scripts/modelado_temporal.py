"""Modelado temporal sobre los usuarios de IA: modelos dentro de cada año y validación hacia adelante.

Dentro del año, la regresión logística y XGBoost se entrenan y evalúan con datos del mismo año.
En la validación hacia adelante se entrena con años anteriores y se evalúa en uno posterior.
Si el AUC hacia adelante queda por debajo del AUC dentro del año, la relación entre las
variables y la discrepancia cambió de un año a otro y el modelo no se transfiere bien.

Genera:
    outputs/temporal_auc.png
    outputs/temporal_coeficientes.png
    outputs/temporal_shap_por_anio.png
    outputs/temporal_validacion.png
    outputs/temporal_resultados_cv.json
    docs/analisis/resultados_modelado_temporal.md
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import shap
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nombres_legibles import nombre_legible  # noqa: E402

warnings.filterwarnings("ignore")

TARGET_LABEL_0 = "Con discrepancia uso-actitud"
COLORS_YEAR = {2023: "#3b82f6", 2024: "#8b5cf6", 2025: "#f59e0b"}
COLOR_POOLED = "#1f2937"
RANDOM_STATE = 42
CV_SPLITS = 5

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "df_clean_users.parquet"
OUTPUTS = ROOT / "outputs"
DOCS = ROOT / "docs" / "analisis"
OUTPUTS.mkdir(parents=True, exist_ok=True)
DOCS.mkdir(parents=True, exist_ok=True)


def cargar_usuarios() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró {DATA_PATH}. Correr primero: python scripts/pipeline.py"
        )
    df = pd.read_parquet(DATA_PATH)
    print(f"Dataset: {df.shape[0]:,} filas, {df.shape[1]} columnas")
    for yr in [2023, 2024, 2025]:
        sub = df[df["survey_year"] == yr]
        print(f"  {yr}: N={len(sub):,}  discrepancia={1 - sub['target'].mean():.1%}")
    return df


def common_features(df: pd.DataFrame) -> list[str]:
    """Variables que varían en los tres años, sin target ni survey_year.

    survey_year queda afuera porque es constante dentro de cada año. También quedan
    afuera las columnas indicadoras que solo toman valores distintos en algún año.
    """
    exclude = {"target", "survey_year"}
    candidates = [c for c in df.columns if c not in exclude]
    stds = df.groupby("survey_year")[candidates].std()
    valid = (stds > 0).all(axis=0)
    return valid[valid].index.tolist()


def _logreg_pipeline(y: pd.Series) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced", max_iter=1000,
            solver="saga", n_jobs=-1, random_state=RANDOM_STATE,
        )),
    ])


def _xgb_model(y: pd.Series) -> XGBClassifier:
    scale_pos_weight = float((y == 0).sum() / (y == 1).sum())
    return XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=5,
        scale_pos_weight=scale_pos_weight, eval_metric="auc",
        n_jobs=-1, random_state=RANDOM_STATE, verbosity=0,
    )


def evaluar_cv(X: pd.DataFrame, y: pd.Series, label: str, use_xgb: bool = False) -> dict:
    """Validación cruzada estratificada; devuelve el ROC-AUC y el PR-AUC de la clase con discrepancia."""
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    model = _xgb_model(y) if use_xgb else _logreg_pipeline(y)
    y_prob_all = cross_val_predict(model, X, y, cv=cv, method="predict_proba")
    y_prob_1 = y_prob_all[:, 1]
    y_prob_0 = y_prob_all[:, 0]
    roc = float(roc_auc_score(y, y_prob_1))
    # El PR-AUC se calcula sobre la clase con discrepancia, que es la minoritaria;
    # un clasificador al azar obtiene su prevalencia (alrededor de 21,5 %).
    pr = float(average_precision_score(y == 0, y_prob_0))
    print(f"  {label:<40} ROC-AUC={roc:.4f}  PR-AUC(min)={pr:.4f}  (N={len(y):,})")
    return {"roc_auc": roc, "pr_auc": pr, "n": int(len(y)), "target_0_pct": float((1 - y).mean())}


def comparar_auc(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """AUC en cinco condiciones: los tres años juntos con y sin el año, y cada año por separado."""
    results = {}

    # common_features deja afuera survey_year, así que se agrega a mano para esta condición.
    cols_con_year = feature_cols + ["survey_year"] if "survey_year" not in feature_cols else feature_cols
    X_full = df[cols_con_year]
    y_full = df["target"]
    results["pooled_con_year"] = evaluar_cv(X_full, y_full, "Años juntos, con survey_year")

    X_pool = df[feature_cols]
    results["pooled_sin_year"] = evaluar_cv(X_pool, y_full, "Años juntos, sin survey_year")

    for year in [2023, 2024, 2025]:
        df_y = df[df["survey_year"] == year]
        X_y = df_y[feature_cols]
        y_y = df_y["target"]
        results[f"within_{year}"] = evaluar_cv(X_y, y_y, f"Regresión logística, {year}")

    return results


def xgb_check_por_anio(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """XGBoost dentro de cada año, como control de la regresión logística."""
    results = {}
    print("  XGBoost por año (con menos filas por año puede sobreajustar)")
    for year in [2023, 2024, 2025]:
        df_y = df[df["survey_year"] == year]
        X_y = df_y[feature_cols]
        y_y = df_y["target"]
        results[f"xgb_{year}"] = evaluar_cv(X_y, y_y, f"XGBoost, {year}", use_xgb=True)
    return results


def coeficientes_por_anio(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Ajusta la regresión logística en cada año, sin validación cruzada, y junta los coeficientes."""
    coefs: dict[str, np.ndarray] = {}
    for year in [2023, 2024, 2025]:
        df_y = df[df["survey_year"] == year]
        X = df_y[feature_cols]
        y = df_y["target"]
        pipe = _logreg_pipeline(y)
        pipe.fit(X, y)
        coefs[str(year)] = pipe.named_steps["clf"].coef_[0]

    tabla = pd.DataFrame(coefs, index=feature_cols)

    tabla["max_abs"] = tabla[["2023", "2024", "2025"]].abs().max(axis=1)
    tabla["mag_change"] = (
        tabla[["2023", "2024", "2025"]].max(axis=1)
        - tabla[["2023", "2024", "2025"]].min(axis=1)
    )
    tabla["sign_change"] = tabla[["2023", "2024", "2025"]].apply(
        lambda row: row.gt(0).any() and row.lt(0).any(), axis=1
    )
    return tabla.sort_values("max_abs", ascending=False)


def shap_rf_por_anio(df: pd.DataFrame, feature_cols: list[str]) -> dict[int, pd.Series]:
    """SHAP de un random forest por año, sobre una muestra de 500 observaciones."""
    results: dict[int, pd.Series] = {}
    for year in [2023, 2024, 2025]:
        df_y = df[df["survey_year"] == year]
        X = df_y[feature_cols]
        y = df_y["target"]

        rf = RandomForestClassifier(
            n_estimators=80, max_depth=8, class_weight="balanced",
            min_samples_leaf=20, n_jobs=-1, random_state=RANDOM_STATE,
        )
        rf.fit(X, y)

        rng = np.random.default_rng(RANDOM_STATE)
        bg_idx = rng.choice(len(X), size=min(500, len(X)), replace=False)
        X_bg = X.iloc[bg_idx]

        explainer = shap.TreeExplainer(rf)
        shap_raw = explainer.shap_values(X_bg)
        if isinstance(shap_raw, list):
            sv = shap_raw[1]
        elif shap_raw.ndim == 3:
            sv = shap_raw[:, :, 1]
        else:
            sv = shap_raw

        mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=feature_cols)
        results[year] = mean_abs.sort_values(ascending=False)
        print(f"  SHAP {year}: top = {results[year].index[0]} ({results[year].iloc[0]:.4f})")

    return results


def figura_auc_comparison(auc_results: dict) -> plt.Figure:
    """ROC-AUC y PR-AUC de cada condición, en dos paneles."""
    labels = {
        "pooled_con_year": "Años juntos\n(con el año)",
        "pooled_sin_year": "Años juntos\n(sin el año)",
        "within_2023": "Dentro del\naño 2023",
        "within_2024": "Dentro del\naño 2024",
        "within_2025": "Dentro del\naño 2025",
    }
    colors = {
        "pooled_con_year": "#6b7280",
        "pooled_sin_year": COLOR_POOLED,
        "within_2023": COLORS_YEAR[2023],
        "within_2024": COLORS_YEAR[2024],
        "within_2025": COLORS_YEAR[2025],
    }
    keys = list(labels.keys())
    x = np.arange(len(keys))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, ylabel, baseline in [
        (axes[0], "roc_auc", "ROC-AUC", 0.5),
        (axes[1], "pr_auc", "PR-AUC", None),
    ]:
        vals = [auc_results[k][metric] for k in keys]
        bars = ax.bar(x, vals, color=[colors[k] for k in keys], width=0.6)

        if baseline is not None:
            ax.axhline(baseline, color="gray", linestyle="--", lw=1, alpha=0.5, label="Aleatorio")
        else:
            # En el PR-AUC el azar da la prevalencia de la discrepancia. Se toma la de 2023
            # como referencia aproximada, aunque cambia de un año a otro.
            pr_baseline = auc_results["within_2023"]["target_0_pct"]
            ax.axhline(pr_baseline, color="gray", linestyle="--", lw=1, alpha=0.5,
                       label=f"Aleatorio ≈ {pr_baseline:.2f}")

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(list(labels.values()), fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(vals) * 1.2)
        ax.set_title(f"{ylabel}, regresión logística (validación cruzada de {CV_SPLITS} partes)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle("AUC con los años juntos y dentro de cada año (usuarios de IA)", fontsize=12)
    fig.tight_layout()
    return fig


GRUPO_EXPERIENCIA = ["YearsCode", "WorkExp"]


def figura_coef_heatmap(tabla_coef: pd.DataFrame, n_top: int = 12) -> plt.Figure:
    """Mapa de calor de los coeficientes por año para las n_top variables que más cambian.

    El cambio se mide como la diferencia entre el coeficiente mayor y el menor de los tres años.
    La cruz marca un cambio de signo. Las variables de experiencia van en un bloque aparte
    porque están muy correlacionadas entre sí y sus coeficientes no se leen por separado.
    """
    anios = ["2023", "2024", "2025"]
    top = tabla_coef.sort_values("mag_change", ascending=False).head(n_top)
    exp_idx = [f for f in top.index if f in GRUPO_EXPERIENCIA]
    otras_idx = [f for f in top.index if f not in GRUPO_EXPERIENCIA]
    orden = exp_idx + otras_idx
    top = top.loc[orden]

    vmax = top[anios].abs().max().max()
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(top) + 2))
    im = ax.imshow(top[anios].values, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(3))
    ax.set_xticklabels(anios, fontsize=11)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(
        [nombre_legible(f) + (" †" if top.loc[f, "sign_change"] else "") for f in top.index],
        fontsize=9,
    )
    for i, f in enumerate(top.index):
        for j, y in enumerate(anios):
            v = float(top.loc[f, y])
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > vmax * 0.6 else "black")

    if exp_idx and otras_idx:
        ax.axhline(len(exp_idx) - 0.5, color="black", lw=1.2)
    plt.colorbar(im, ax=ax, label="β estandarizado (positivo: menos discrepancia)")
    ax.set_title(
        f"Coeficientes de la regresión logística por año: las {len(top)} variables con mayor cambio\n"
        "† = cambia de signo entre años. Bloque superior: experiencia (variables muy correlacionadas entre sí)",
        fontsize=10,
    )
    fig.tight_layout()
    return fig


def figura_shap_por_anio(shap_results: dict[int, pd.Series], n_top: int = 15) -> plt.Figure:
    """SHAP medio absoluto de cada año, un panel por año."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    for ax, year in zip(axes, [2023, 2024, 2025]):
        top = shap_results[year].head(n_top).iloc[::-1]
        ax.barh(range(len(top)), top.values, color=COLORS_YEAR[year])
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels([nombre_legible(t) for t in top.index], fontsize=9)
        ax.set_xlabel("SHAP medio absoluto")
        ax.set_title(f"{n_top} variables principales, {year}")
        ax.grid(axis="x", linestyle="--", alpha=0.3)

    fig.suptitle(
        "Importancia de las variables por año de la encuesta (random forest, 500 observaciones por año)\n"
        "Sin el año de la encuesta como variable",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def evaluar_forward(
    df: pd.DataFrame,
    feature_cols: list[str],
    train_years: list[int],
    test_year: int,
) -> dict:
    """Entrena con train_years y evalúa en test_year.

    Usa las mismas variables que los modelos dentro del año, así los resultados se pueden
    comparar y no falta ninguna variable en el año de prueba.
    """
    df_train = df[df["survey_year"].isin(train_years)]
    df_test = df[df["survey_year"] == test_year]

    X_train = df_train[feature_cols]
    y_train = df_train["target"]
    X_test = df_test[feature_cols]
    y_test = df_test["target"]

    pipe = _logreg_pipeline(y_train)
    pipe.fit(X_train, y_train)

    y_prob_all = pipe.predict_proba(X_test)
    roc = float(roc_auc_score(y_test, y_prob_all[:, 1]))
    pr = float(average_precision_score(y_test == 0, y_prob_all[:, 0]))

    train_label = "+".join(str(y) for y in train_years)
    label = f"Entrena con {train_label}, evalúa en {test_year}"
    print(f"  {label:<40} ROC-AUC={roc:.4f}  PR-AUC(min)={pr:.4f}  "
          f"(N_train={len(y_train):,}  N_test={len(y_test):,})")
    return {
        "roc_auc": roc, "pr_auc": pr,
        "n_train": int(len(y_train)), "n_test": int(len(y_test)),
        "target_0_pct_test": float((1 - y_test).mean()),
        "label": label,
    }


def figura_validacion_temporal(
    auc_within: dict,
    auc_forward: dict,
    auc_pooled_sin: float,
) -> plt.Figure:
    """Figura resumen: AUC dentro del año, hacia adelante y con los años juntos."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    within_keys = ["within_2023", "within_2024", "within_2025"]
    forward_keys = ["fwd_23_24", "fwd_24_25", "fwd_2324_25"]

    within_labels = ["Dentro del\naño 2023", "Dentro del\naño 2024", "Dentro del\naño 2025"]
    forward_labels = [
        "De 2023\na 2024",
        "De 2024\na 2025",
        "De 2023+24\na 2025",
    ]

    within_colors = [COLORS_YEAR[y] for y in [2023, 2024, 2025]]
    forward_colors = ["#ef4444", "#dc2626", "#b91c1c"]

    for ax, metric, ylabel in [
        (axes[0], "roc_auc", "ROC-AUC"),
        (axes[1], "pr_auc", "PR-AUC (clase minoritaria)"),
    ]:
        w_vals = [auc_within[k][metric] for k in within_keys]
        f_vals = [auc_forward[k][metric] for k in forward_keys if k in auc_forward]
        f_labels = [l for k, l in zip(forward_keys, forward_labels) if k in auc_forward]
        f_colors = [c for k, c in zip(forward_keys, forward_colors) if k in auc_forward]

        n_within = len(w_vals)
        n_forward = len(f_vals)
        x_within = np.arange(n_within)
        x_forward = np.arange(n_within, n_within + n_forward)

        bars_w = ax.bar(x_within, w_vals, color=within_colors, width=0.6, label="Dentro del año")
        bars_f = ax.bar(x_forward, f_vals, color=f_colors, width=0.6, label="Hacia adelante")

        for bar, val in zip(list(bars_w) + list(bars_f), w_vals + f_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

        all_labels = within_labels + f_labels
        ax.set_xticks(np.concatenate([x_within, x_forward]))
        ax.set_xticklabels(all_labels, fontsize=9)
        ax.axvline(n_within - 0.5, color="gray", linestyle=":", lw=1, alpha=0.5)

        pooled_val = auc_within["pooled_sin_year"][metric]
        ax.axhline(pooled_val, color=COLOR_POOLED, linestyle="--", lw=1.5,
                   label=f"Años juntos, sin el año de la encuesta: {pooled_val:.3f}")

        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(max(w_vals), max(f_vals)) * 1.2)
        ax.set_title(ylabel)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        ax.text(n_within - 0.5, ax.get_ylim()[1] * 0.97, "│", ha="center", fontsize=14, color="gray")
        ax.text(n_within / 2 - 0.5, ax.get_ylim()[1] * 0.93, "Dentro del año", ha="center",
                fontsize=8, color="gray")
        ax.text(n_within + n_forward / 2 - 0.5, ax.get_ylim()[1] * 0.93, "Hacia adelante",
                ha="center", fontsize=8, color="#ef4444")

    fig.suptitle(
        "Validación temporal (concept drift)\n"
        "Dentro del año: entrena y evalúa en el mismo año  ·  Hacia adelante: entrena en años pasados y evalúa en uno posterior",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def generar_lectura(
    auc_results: dict,
    tabla_coef: pd.DataFrame,
    shap_results: dict[int, pd.Series],
    xgb_results: dict,
    forward_results: dict | None = None,
) -> str:
    """Arma el informe en markdown con los resultados."""
    out = []
    out.append("# Modelado temporal: resultados dentro de cada año y validación hacia adelante\n\n")
    out.append("**Datos:** `data/df_clean_users.parquet`. Generado por `scripts/modelado_temporal.py`.\n\n")
    out.append("**Pregunta:** ¿cambia de un año a otro la capacidad de predecir la discrepancia? "
               "¿Qué variables cambian de efecto?\n\n")

    out.append("## 1. AUC con los años juntos y dentro de cada año (regresión logística)\n\n")
    rows = []
    labels = {
        "pooled_con_year": "Años juntos, con el año de la encuesta",
        "pooled_sin_year": "Años juntos, sin el año de la encuesta",
        "within_2023": "Dentro del año 2023",
        "within_2024": "Dentro del año 2024",
        "within_2025": "Dentro del año 2025",
    }
    for k, label in labels.items():
        r = auc_results[k]
        rows.append({
            "Condición": label,
            "ROC-AUC": round(r["roc_auc"], 4),
            "PR-AUC": round(r["pr_auc"], 4),
            "N": f"{r['n']:,}",
            "% discrepancia": f"{r['target_0_pct']:.1%}",
        })
    df_auc = pd.DataFrame(rows).set_index("Condición")
    out.append(df_auc.to_markdown())
    out.append("\n\n")

    delta_year = auc_results["pooled_con_year"]["roc_auc"] - auc_results["pooled_sin_year"]["roc_auc"]
    out.append(f"**Aporte del año de la encuesta al modelo con los años juntos:** {delta_year:+.4f} de ROC-AUC "
               "(con el año menos sin el año).\n\n")

    out.append(
        "**Referencia del PR-AUC:** un clasificador al azar obtiene un PR-AUC igual a la prevalencia de la clase "
        f"minoritaria ({auc_results['within_2023']['target_0_pct']:.1%} en 2023, "
        f"{auc_results['within_2024']['target_0_pct']:.1%} en 2024 y "
        f"{auc_results['within_2025']['target_0_pct']:.1%} en 2025).\n\n"
    )

    aucs_within = [auc_results[f"within_{y}"]["roc_auc"] for y in [2023, 2024, 2025]]
    rango = max(aucs_within) - min(aucs_within)
    out.append(
        f"El ROC-AUC dentro del año va de {min(aucs_within):.4f} a {max(aucs_within):.4f} "
        f"(rango {rango:.4f}), y con los años juntos y sin el año es "
        f"{auc_results['pooled_sin_year']['roc_auc']:.4f}. "
        "La señal crece de un año al siguiente, y juntar los años la mezcla.\n\n"
    )

    out.append("## 2. Coeficientes de la regresión logística por año (20 mayores)\n\n")
    top20 = tabla_coef.head(20)[["2023", "2024", "2025", "mag_change", "sign_change"]].copy()
    top20 = top20.round({"2023": 3, "2024": 3, "2025": 3, "mag_change": 3})
    top20["sign_change"] = top20["sign_change"].map({True: "sí", False: ""})
    out.append(top20.to_markdown())
    out.append("\n\n")

    n_sign = int(tabla_coef["sign_change"].sum())
    out.append(f"Cambian de signo {n_sign} de {len(tabla_coef)} variables, pero la mayoría tiene coeficientes "
               "cercanos a cero. Los de mayor magnitud son:\n\n")
    sign_big = tabla_coef[tabla_coef["sign_change"] & (tabla_coef["max_abs"] > 0.1)]
    for feat, row in sign_big.iterrows():
        out.append(f"- **{nombre_legible(feat)}:** {row['2023']:.3f} (2023), {row['2024']:.3f} (2024), {row['2025']:.3f} (2025)\n")
    out.append("\n")

    out.append("### Variables con mayor cambio de magnitud (5 primeras)\n\n")
    top_mag = tabla_coef.nlargest(5, "mag_change")[["2023", "2024", "2025", "mag_change"]].round(3)
    out.append(top_mag.rename(index=nombre_legible).to_markdown())
    out.append("\n\nLos años programando y la experiencia laboral están correlacionados entre sí, "
               "así que sus coeficientes no deben leerse por separado.\n\n")

    out.append("## 3. Importancia SHAP por año (random forest, 10 mayores)\n\n")
    for year in [2023, 2024, 2025]:
        out.append(f"**{year}:** {', '.join(f'{nombre_legible(f)} ({v:.4f})' for f, v in shap_results[year].head(10).items())}\n\n")

    top5_by_year = {y: set(shap_results[y].head(5).index) for y in [2023, 2024, 2025]}
    intersection_5 = top5_by_year[2023] & top5_by_year[2024] & top5_by_year[2025]
    out.append(
        "**Variables entre las 5 principales de los tres años:** "
        + (", ".join(sorted(nombre_legible(f) for f in intersection_5)) if intersection_5 else "ninguna")
        + "\n\n"
    )

    out.append("## 4. Control con XGBoost dentro de cada año\n\n")
    xgb_rows = []
    for year in [2023, 2024, 2025]:
        r = xgb_results.get(f"xgb_{year}", {})
        xgb_rows.append({"Año": year, "ROC-AUC": round(r.get("roc_auc", 0), 4),
                          "PR-AUC": round(r.get("pr_auc", 0), 4)})
    out.append(pd.DataFrame(xgb_rows).set_index("Año").to_markdown())
    out.append("\n\n")

    if forward_results:
        out.append("## 5. Validación hacia adelante (cambio en el tiempo)\n\n")
        out.append(
            f"Se usan las mismas {len(tabla_coef)} variables comunes a los tres años. Se entrena con años anteriores y se evalúa en uno posterior. "
            "La diferencia entre el AUC hacia adelante y el AUC dentro del año mide cuánto cambió la relación entre las variables y la discrepancia.\n\n"
        )
        esquemas = {
            "fwd_23_24": "Entrena con 2023, evalúa en 2024",
            "fwd_24_25": "Entrena con 2024, evalúa en 2025",
            "fwd_2324_25": "Entrena con 2023 y 2024, evalúa en 2025",
        }

        rows_fwd = []
        within_ref = {
            "fwd_23_24": auc_results.get("within_2024", {}),
            "fwd_24_25": auc_results.get("within_2025", {}),
            "fwd_2324_25": auc_results.get("within_2025", {}),
        }
        for key, r in forward_results.items():
            ref = within_ref.get(key, {})
            delta_roc = r["roc_auc"] - ref.get("roc_auc", 0)
            delta_pr = r["pr_auc"] - ref.get("pr_auc", 0)
            rows_fwd.append({
                "Esquema": esquemas.get(key, r["label"]),
                "ROC-AUC": round(r["roc_auc"], 4),
                "PR-AUC": round(r["pr_auc"], 4),
                "Diferencia de ROC-AUC": f"{delta_roc:+.4f}",
                "Diferencia de PR-AUC": f"{delta_pr:+.4f}",
                "N de prueba": f"{r['n_test']:,}",
            })
        out.append(pd.DataFrame(rows_fwd).set_index("Esquema").to_markdown())
        out.append("\n\n")
        out.append("Las diferencias se calculan contra el modelo dentro del año de prueba. "
                   "Con solo dos pasos de un año al siguiente no se puede afirmar que el cambio se acelere.\n")

    return "".join(out)


def main():
    print("Modelado temporal sobre usuarios de IA")

    df = cargar_usuarios()

    print("\nVariables comunes a los tres años")
    feature_cols = common_features(df)
    print(f"  {len(feature_cols)} variables (sin survey_year)")

    print("\nAUC de la regresión logística")
    auc_results = comparar_auc(df, feature_cols)

    print("\nXGBoost por año")
    xgb_results = xgb_check_por_anio(df, feature_cols)

    print("\nCoeficientes de la regresión logística por año")
    tabla_coef = coeficientes_por_anio(df, feature_cols)
    n_sign = int(tabla_coef["sign_change"].sum())
    print(f"  {len(tabla_coef)} variables, {n_sign} cambian de signo")
    print("\n  10 mayores por |β| máximo:")
    print(tabla_coef.head(10)[["2023", "2024", "2025", "mag_change", "sign_change"]].round(3).to_string())

    print("\nSHAP por año (random forest)")
    shap_results = shap_rf_por_anio(df, feature_cols)

    print("\nFiguras")
    fig_auc = figura_auc_comparison(auc_results)
    fig_auc.savefig(OUTPUTS / "temporal_auc.png", dpi=150, bbox_inches="tight")
    plt.close(fig_auc)
    print("Guardado outputs/temporal_auc.png")

    fig_coef = figura_coef_heatmap(tabla_coef, n_top=12)
    fig_coef.savefig(OUTPUTS / "temporal_coeficientes.png", dpi=150, bbox_inches="tight")
    plt.close(fig_coef)
    print("Guardado outputs/temporal_coeficientes.png")

    fig_shap = figura_shap_por_anio(shap_results, n_top=15)
    fig_shap.savefig(OUTPUTS / "temporal_shap_por_anio.png", dpi=150, bbox_inches="tight")
    plt.close(fig_shap)
    print("Guardado outputs/temporal_shap_por_anio.png")

    print("\nValidación hacia adelante")
    forward_results = {}
    forward_results["fwd_23_24"] = evaluar_forward(df, feature_cols, [2023], 2024)
    forward_results["fwd_24_25"] = evaluar_forward(df, feature_cols, [2024], 2025)
    forward_results["fwd_2324_25"] = evaluar_forward(df, feature_cols, [2023, 2024], 2025)

    fig_val = figura_validacion_temporal(auc_results, forward_results,
                                         auc_results["pooled_sin_year"]["roc_auc"])
    fig_val.savefig(OUTPUTS / "temporal_validacion.png", dpi=150, bbox_inches="tight")
    plt.close(fig_val)
    print("Guardado outputs/temporal_validacion.png")

    results_json = {
        "auc_results": auc_results,
        "xgb_results": xgb_results,
        "forward_results": forward_results,
        "n_common_features": len(feature_cols),
        "n_sign_changes": n_sign,
        "top10_coef": tabla_coef.head(10)[["2023", "2024", "2025", "mag_change"]].round(4).to_dict(),
        "top5_shap_per_year": {
            str(y): shap_results[y].head(5).round(4).to_dict() for y in [2023, 2024, 2025]
        },
    }
    (OUTPUTS / "temporal_resultados_cv.json").write_text(
        json.dumps(results_json, indent=2, default=str)
    )
    print("Guardado outputs/temporal_resultados_cv.json")

    md = generar_lectura(auc_results, tabla_coef, shap_results, xgb_results, forward_results)
    doc_path = DOCS / "resultados_modelado_temporal.md"
    doc_path.write_text(md)
    print(f"Guardado {doc_path.relative_to(ROOT)}")

    print("\nListo.")


if __name__ == "__main__":
    main()
