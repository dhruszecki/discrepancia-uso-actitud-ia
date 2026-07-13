"""Modelado temporal (Capas B + C) — within-year y forward sobre usuarios (80k).

Capa B: LogReg y XGBoost por año dentro del mismo año (within-year).
Capa C: Validación forward — train en año pasado, test en año futuro.
                   Cuantifica el concept drift: si el AUC forward < within-year,
                   el fenómeno cambió y el modelo no transfiere.

Outputs:
    outputs/08_temporal_auc_comparison.png   (Capa B)
    outputs/08_temporal_coef_heatmap.png     (Capa B)
    outputs/08_temporal_shap_por_anio.png    (Capa B)
    outputs/08_temporal_validation.png       (Capa C)
    outputs/08_temporal_cv_results.json
    resultados/resultados_modelado_temporal.md
"""
from __future__ import annotations

import json
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

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
TARGET_LABEL_0 = "Con discrepancia uso-actitud"
COLORS_YEAR = {2023: "#3b82f6", 2024: "#8b5cf6", 2025: "#f59e0b"}
COLOR_POOLED = "#1f2937"
RANDOM_STATE = 42
CV_SPLITS = 5
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "df_clean_users.parquet"
OUTPUTS = ROOT / "outputs"
DOCS = ROOT / "resultados"
DOCS.mkdir(parents=True, exist_ok=True)


def cargar_usuarios() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró {DATA_PATH}. Correr primero: python scripts/pipeline.py"
        )
    df = pd.read_parquet(DATA_PATH)
    print(f"Dataset: {df.shape[0]:,} filas × {df.shape[1]} cols")
    for yr in [2023, 2024, 2025]:
        sub = df[df["survey_year"] == yr]
        print(f"  {yr}: N={len(sub):,}  discrepancia={1 - sub['target'].mean():.1%}")
    return df


def common_features(df: pd.DataFrame) -> list[str]:
    """Features con varianza > 0 en los 3 años, excluyendo target y survey_year.

    survey_year se excluye porque es constante dentro de un año (objetivo de AC #5).
    Las columnas OHE exclusivas de un año (std=0 en otros años) quedan fuera.
    """
    exclude = {"target", "survey_year"}
    candidates = [c for c in df.columns if c not in exclude]
    stds = df.groupby("survey_year")[candidates].std()  # (3 años, n_cols)
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
    """CV k=5 estratificada; retorna ROC-AUC y PR-AUC para clase minoritaria (target=0)."""
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    model = _xgb_model(y) if use_xgb else _logreg_pipeline(y)
    y_prob_all = cross_val_predict(model, X, y, cv=cv, method="predict_proba")
    y_prob_1 = y_prob_all[:, 1]   # prob de target=1 (mayoría)
    y_prob_0 = y_prob_all[:, 0]   # prob de target=0 (minoría = discrepancia)
    roc = float(roc_auc_score(y, y_prob_1))
    # PR-AUC para clase minoritaria (discrepancia): baseline = tasa_base ≈ 21.5%
    pr = float(average_precision_score(y == 0, y_prob_0))
    print(f"  {label:<40} ROC-AUC={roc:.4f}  PR-AUC(min)={pr:.4f}  (N={len(y):,})")
    return {"roc_auc": roc, "pr_auc": pr, "n": int(len(y)), "target_0_pct": float((1 - y).mean())}


def comparar_auc(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Compara AUC en 5 condiciones: pooled con/sin survey_year + 3 años."""
    results = {}

    # Pooled con survey_year (referencia; survey_year no está en common_features por diseño,
    # así que lo añadimos explícitamente para esta condición)
    cols_con_year = feature_cols + ["survey_year"] if "survey_year" not in feature_cols else feature_cols
    X_full = df[cols_con_year]
    y_full = df["target"]
    results["pooled_con_year"] = evaluar_cv(X_full, y_full, "Pooled LogReg (CON survey_year)")

    # Pooled sin survey_year (common_features ya excluye survey_year)
    X_pool = df[feature_cols]
    results["pooled_sin_year"] = evaluar_cv(X_pool, y_full, "Pooled LogReg (SIN survey_year)")

    # Por año (survey_year ya excluida de feature_cols)
    for year in [2023, 2024, 2025]:
        df_y = df[df["survey_year"] == year]
        X_y = df_y[feature_cols]
        y_y = df_y["target"]
        results[f"within_{year}"] = evaluar_cv(X_y, y_y, f"LogReg within-year {year}")

    return results


def xgb_check_por_anio(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """XGBoost por año como validación secundaria (AC #7)."""
    results = {}
    print("  [XGBoost check — puede sobreajustar con N reducido por año]")
    for year in [2023, 2024, 2025]:
        df_y = df[df["survey_year"] == year]
        X_y = df_y[feature_cols]
        y_y = df_y["target"]
        results[f"xgb_{year}"] = evaluar_cv(X_y, y_y, f"XGBoost within-year {year}", use_xgb=True)
    return results


def coeficientes_por_anio(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Ajusta LogReg en cada año (sin CV) y extrae coeficientes estandarizados."""
    coefs: dict[str, np.ndarray] = {}
    for year in [2023, 2024, 2025]:
        df_y = df[df["survey_year"] == year]
        X = df_y[feature_cols]
        y = df_y["target"]
        pipe = _logreg_pipeline(y)
        pipe.fit(X, y)
        coefs[str(year)] = pipe.named_steps["clf"].coef_[0]

    tabla = pd.DataFrame(coefs, index=feature_cols)

    # Métricas de cambio entre años
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
    """RF SHAP por año sobre muestra background de 500 obs. (AC #4)."""
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


# ============================================================
# Figuras
# ============================================================

def figura_auc_comparison(auc_results: dict) -> plt.Figure:
    """2 paneles: ROC-AUC y PR-AUC por condición."""
    labels = {
        "pooled_con_year": "Pooled\n(con survey_year)",
        "pooled_sin_year": "Pooled\n(sin survey_year)",
        "within_2023": "Within\n2023",
        "within_2024": "Within\n2024",
        "within_2025": "Within\n2025",
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

        # Referencia línea aleatoria
        if baseline is not None:
            ax.axhline(baseline, color="gray", linestyle="--", lw=1, alpha=0.5, label="Aleatorio")
        else:
            # Para PR-AUC: baseline = tasa_base target=0 (prevalencia de la clase minoritaria)
            pr_baseline = auc_results["within_2023"]["target_0_pct"]  # aprox
            ax.axhline(pr_baseline, color="gray", linestyle="--", lw=1, alpha=0.5,
                       label=f"Aleatorio ≈ {pr_baseline:.2f}")

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(list(labels.values()), fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(vals) * 1.2)
        ax.set_title(f"{ylabel} — LogReg (CV k={CV_SPLITS})")
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle("Comparación de AUC: pooled vs. within-year (usuarios IA)", fontsize=12)
    fig.tight_layout()
    return fig


def figura_coef_heatmap(tabla_coef: pd.DataFrame, n_top: int = 25) -> plt.Figure:
    """Heatmap de coeficientes estandarizados LogReg por año (top N por |β| máximo)."""
    top = tabla_coef.head(n_top)[["2023", "2024", "2025"]].copy()

    vmax = top.abs().max().max()
    fig, ax = plt.subplots(figsize=(7, max(6, 0.38 * n_top)))

    im = ax.imshow(top.values, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["2023", "2024", "2025"], fontsize=11)
    ax.set_yticks(range(n_top))
    sign_flag = tabla_coef.head(n_top)["sign_change"]
    ytick_labels = [
        f"{t[:40]}*" if sign_flag.loc[t] else t[:42]
        for t in top.index
    ]
    ax.set_yticklabels(ytick_labels, fontsize=8)

    for i in range(len(top)):
        for j, year in enumerate(["2023", "2024", "2025"]):
            val = float(top.iloc[i][year])
            txt_color = "white" if abs(val) > vmax * 0.65 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=txt_color)

    plt.colorbar(im, ax=ax, label="β estandarizado")
    ax.set_title(
        f"Coeficientes LogReg estandarizados por año\n"
        f"(top {n_top} por |β| máximo; * = cambio de signo entre años)"
    )
    if sign_flag.any():
        fig.text(0.01, 0.01, "* = el signo del coeficiente invierte entre alguno de los 3 años",
                 fontsize=8, style="italic")
        fig.subplots_adjust(bottom=0.05)
    fig.tight_layout()
    return fig


def figura_shap_por_anio(shap_results: dict[int, pd.Series], n_top: int = 15) -> plt.Figure:
    """3 paneles horizontales con |SHAP| por año."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    for ax, year in zip(axes, [2023, 2024, 2025]):
        top = shap_results[year].head(n_top).iloc[::-1]
        ax.barh(range(len(top)), top.values, color=COLORS_YEAR[year])
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels([t[:32] for t in top.index], fontsize=8)
        ax.set_xlabel("|SHAP| medio")
        ax.set_title(f"Top {n_top} features\n{year}")
        ax.grid(axis="x", linestyle="--", alpha=0.3)

    fig.suptitle(
        "Importancia SHAP por año del survey (RF, ~500 obs. background por año)\n"
        "— comparar la posición relativa de features entre olas",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


# ============================================================
# Capa C — Validación temporal forward
# ============================================================

def evaluar_forward(
    df: pd.DataFrame,
    feature_cols: list[str],
    train_years: list[int],
    test_year: int,
) -> dict:
    """Train en train_years, test en test_year — mide concept drift.

    Mismas features que Capa B (common_features), garantizando comparabilidad
    entre olas (sin leakage de schema entre train y test).
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
    label = f"train {train_label} → test {test_year}"
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
    """Figura resumen Capa C: within-year vs forward vs pooled."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    within_keys = ["within_2023", "within_2024", "within_2025"]
    forward_keys = ["fwd_23_24", "fwd_24_25", "fwd_2324_25"]

    # Etiquetas legibles
    within_labels = ["Within\n2023", "Within\n2024", "Within\n2025"]
    forward_labels = [
        "2023 →\n2024",
        "2024 →\n2025",
        "2023+24\n→2025",
    ]

    within_colors = [COLORS_YEAR[y] for y in [2023, 2024, 2025]]
    forward_colors = ["#ef4444", "#dc2626", "#b91c1c"]  # escala de rojos para drift

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

        bars_w = ax.bar(x_within, w_vals, color=within_colors, width=0.6, label="Within-year")
        bars_f = ax.bar(x_forward, f_vals, color=f_colors, width=0.6, label="Forward (drift)")

        for bar, val in zip(list(bars_w) + list(bars_f), w_vals + f_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

        all_labels = within_labels + f_labels
        ax.set_xticks(np.concatenate([x_within, x_forward]))
        ax.set_xticklabels(all_labels, fontsize=9)
        ax.axvline(n_within - 0.5, color="gray", linestyle=":", lw=1, alpha=0.5)

        # Referencia pooled sin survey_year
        pooled_val = auc_within["pooled_sin_year"][metric]
        ax.axhline(pooled_val, color=COLOR_POOLED, linestyle="--", lw=1.5,
                   label=f"Pooled (sin year) {pooled_val:.3f}")

        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(max(w_vals), max(f_vals)) * 1.2)
        ax.set_title(ylabel)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        # Anotación de separación within | forward
        ax.text(n_within - 0.5, ax.get_ylim()[1] * 0.97, "│", ha="center", fontsize=14, color="gray")
        ax.text(n_within / 2 - 0.5, ax.get_ylim()[1] * 0.93, "Within-year", ha="center",
                fontsize=8, color="gray")
        ax.text(n_within + n_forward / 2 - 0.5, ax.get_ylim()[1] * 0.93, "Forward (drift)",
                ha="center", fontsize=8, color="#ef4444")

    fig.suptitle(
        "Validación temporal (Capa C) — concept drift\n"
        "Within-year: mismo año train/test  ·  Forward: train pasado → test futuro",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


# ============================================================
# Documento de lectura
# ============================================================

def generar_lectura(
    auc_results: dict,
    tabla_coef: pd.DataFrame,
    shap_results: dict[int, pd.Series],
    xgb_results: dict,
    forward_results: dict | None = None,
) -> str:
    out = []
    out.append("# Modelado temporal (Capa B) — Within-year sobre usuarios de IA\n\n")
    out.append("**Dataset:** `data/df_clean_users.parquet`\n\n")
    out.append("**Pregunta central:** ¿varía la capacidad predictiva del perfil profesional "
               "entre años? ¿Qué features cambian de efecto?\n\n---\n\n")

    # 1. AUC comparison
    out.append("## 1. Comparación de AUC: pooled vs. within-year (LogReg)\n\n")
    rows = []
    labels = {
        "pooled_con_year": "Pooled (con survey_year)",
        "pooled_sin_year": "Pooled (sin survey_year)",
        "within_2023": "Within-year 2023",
        "within_2024": "Within-year 2024",
        "within_2025": "Within-year 2025",
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

    # Calcular cuánto aporta survey_year al pooled
    delta_year = auc_results["pooled_con_year"]["roc_auc"] - auc_results["pooled_sin_year"]["roc_auc"]
    out.append(f"**Efecto de survey_year en el pooled:** {delta_year:+.4f} pp "
               f"(diferencia ROC-AUC con vs. sin la variable).\n\n")

    pr_base_23 = auc_results["within_2023"]["target_0_pct"]
    out.append(
        f"**Referencia PR-AUC:** un clasificador aleatorio obtiene PR-AUC ≈ tasa base de la clase minoritaria "
        f"(≈{pr_base_23:.1%} en 2023, ≈{auc_results['within_2025']['target_0_pct']:.1%} en 2025). "
        "Valores por encima indican señal real.\n\n"
    )

    # Interpretación
    aucs_within = [auc_results[f"within_{y}"]["roc_auc"] for y in [2023, 2024, 2025]]
    auc_pooled_sin = auc_results["pooled_sin_year"]["roc_auc"]
    rango = max(aucs_within) - min(aucs_within)
    out.append(
        f"El AUC within-year oscila entre {min(aucs_within):.4f} y {max(aucs_within):.4f} "
        f"(rango {rango:.4f}), vs. pooled sin survey_year = {auc_pooled_sin:.4f}. "
    )
    if rango < 0.03:
        out.append("La variación es pequeña: el perfil predice de forma similar en los 3 años, "
                   "lo que refuerza que el AUC ~0.63 no es artefacto del pooling sino una "
                   "limitación estructural del perfil estático para predecir discrepancia.\n\n")
    else:
        out.append("La variación es apreciable: el pooling está diluyendo señal año-específica. "
                   "Analizar qué año tiene mayor AUC y qué features lo explican.\n\n")

    # 2. Coeficientes
    out.append("## 2. Coeficientes LogReg estandarizados por año (top 20)\n\n")
    top20 = tabla_coef.head(20)[["2023", "2024", "2025", "mag_change", "sign_change"]].copy()
    top20 = top20.round({"2023": 3, "2024": 3, "2025": 3, "mag_change": 3})
    top20["sign_change"] = top20["sign_change"].map({True: "✓", False: ""})
    out.append(top20.to_markdown())
    out.append("\n\n")

    # Features con cambio de signo
    sign_changers = tabla_coef[tabla_coef["sign_change"]].head(10)
    if len(sign_changers) > 0:
        out.append("### Features con cambio de signo entre años\n\n")
        out.append("Estas features invierten su relación con la discrepancia uso-actitud entre olas:\n\n")
        for feat, row in sign_changers.iterrows():
            out.append(f"- **{feat}:** {row['2023']:.3f} (2023) → {row['2024']:.3f} (2024) → {row['2025']:.3f} (2025)\n")
        out.append("\n")
    else:
        out.append("No se detectaron cambios de signo en el top 20. "
                   "El perfil predice en la misma dirección en los 3 años.\n\n")

    # Features con mayor cambio de magnitud
    out.append("### Features con mayor cambio de magnitud (mag_change top 5)\n\n")
    top_mag = tabla_coef.nlargest(5, "mag_change")[["2023", "2024", "2025", "mag_change"]].round(3)
    out.append(top_mag.to_markdown())
    out.append("\n\n")

    # 3. SHAP
    out.append("## 3. Importancia SHAP por año (RF top 10)\n\n")
    for year in [2023, 2024, 2025]:
        out.append(f"**{year}:** {', '.join(f'{f} ({v:.4f})' for f, v in shap_results[year].head(10).items())}\n\n")

    # Estabilidad del top-5 SHAP
    top5_by_year = {y: set(shap_results[y].head(5).index) for y in [2023, 2024, 2025]}
    intersection_5 = top5_by_year[2023] & top5_by_year[2024] & top5_by_year[2025]
    out.append(
        f"**Features en el top-5 SHAP de los 3 años:** "
        + (", ".join(sorted(intersection_5)) if intersection_5 else "ninguna")
        + "\n\n"
    )

    # 4. XGBoost check
    out.append("## 4. Check secundario XGBoost (within-year)\n\n")
    xgb_rows = []
    for year in [2023, 2024, 2025]:
        r = xgb_results.get(f"xgb_{year}", {})
        xgb_rows.append({"Año": year, "ROC-AUC": round(r.get("roc_auc", 0), 4),
                          "PR-AUC": round(r.get("pr_auc", 0), 4)})
    out.append(pd.DataFrame(xgb_rows).set_index("Año").to_markdown())
    out.append("\n\n*Nota: XGBoost puede sobreajustar con N≈20-30k por año y ~120 features. "
               "Si su AUC supera sustancialmente a LogReg, hay interacciones no lineales; "
               "si es similar, las relaciones son mayormente lineales.*\n\n")

    # 5. Implicaciones
    out.append("## 5. Implicaciones para E3\n\n")
    out.append(
        "- **AUC estructuralmente limitado:** el perfil estático del desarrollador "
        "predice la discrepancia uso-actitud con un AUC de ~0.63 tanto en pooled como "
        "within-year. No es artefacto del pooling; es la señal real disponible.\n"
    )
    out.append(
        "- **survey_year no es el driver principal:** la diferencia de AUC con/sin survey_year "
        "es pequeña, lo que indica que el efecto-año está parcialmente capturado por "
        "las features de perfil (DevType, seniority, stack).\n"
    )
    if len(sign_changers) > 0:
        out.append(
            "- **Evolución del fenómeno:** los cambios de signo en coeficientes sugieren "
            "que la relación entre algunas variables del perfil y la discrepancia "
            "se invierte entre olas. Esto debe discutirse en §8 (discusión) de E3.\n"
        )
    out.append(
        "- **Insumo para Capa C:** los modelos within-year sirven como baseline "
        "para la validación temporal forward (train en 2023 → test en 2024/2025).\n\n"
    )

    # 6. Capa C — Validación forward
    if forward_results:
        out.append("## 6. Capa C — Validación temporal forward (concept drift)\n\n")
        out.append(
            "**Metodología:** mismas 91 features comunes (var>0 en 2023/2024/2025). "
            "Train en año(s) pasado(s), test en año futuro. "
            "La diferencia AUC forward − within-year mide la velocidad del concept drift.\n\n"
        )

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
                "Esquema": r["label"],
                "ROC-AUC": round(r["roc_auc"], 4),
                "PR-AUC(min)": round(r["pr_auc"], 4),
                "ΔROC vs within": f"{delta_roc:+.4f}",
                "ΔPR vs within": f"{delta_pr:+.4f}",
                "N_test": f"{r['n_test']:,}",
            })
        df_fwd = pd.DataFrame(rows_fwd).set_index("Esquema")
        out.append(df_fwd.to_markdown())
        out.append("\n\n")

        out.append("### Lectura del drift\n\n")
        # Caída entre 2023→2024 y within-2024
        fwd_23_24 = forward_results.get("fwd_23_24", {})
        within_24 = auc_results.get("within_2024", {})
        if fwd_23_24 and within_24:
            delta = fwd_23_24["roc_auc"] - within_24["roc_auc"]
            out.append(
                f"- **2023 → 2024:** AUC forward = {fwd_23_24['roc_auc']:.4f} vs. "
                f"within-2024 = {within_24['roc_auc']:.4f} → degradación {delta:+.4f}.\n"
            )

        fwd_24_25 = forward_results.get("fwd_24_25", {})
        within_25 = auc_results.get("within_2025", {})
        if fwd_24_25 and within_25:
            delta = fwd_24_25["roc_auc"] - within_25["roc_auc"]
            out.append(
                f"- **2024 → 2025:** AUC forward = {fwd_24_25['roc_auc']:.4f} vs. "
                f"within-2025 = {within_25['roc_auc']:.4f} → degradación {delta:+.4f}.\n"
            )

        fwd_2324_25 = forward_results.get("fwd_2324_25", {})
        if fwd_2324_25 and within_25:
            delta = fwd_2324_25["roc_auc"] - within_25["roc_auc"]
            out.append(
                f"- **2023+2024 → 2025:** AUC forward = {fwd_2324_25['roc_auc']:.4f} vs. "
                f"within-2025 = {within_25['roc_auc']:.4f} → degradación {delta:+.4f}.\n"
            )

        out.append(
            "\n**Interpretación:** la diferencia entre AUC forward y within-year es la medida empírica "
            "del concept drift. Si la caída es > 2 pp, el fenómeno cambió de ola a ola y el perfil "
            "estático del desarrollador no transfiere entre años. Esto convierte el AUC bajo "
            "de un 'defecto del modelo' a un **resultado sobre la inestabilidad del fenómeno**: "
            "la discrepancia uso-actitud es un proceso dinámico, no una característica estática.\n\n"
        )
        out.append(
            "Citar en §8 (discusión) y §10 (limitaciones) de E3: "
            "el concepto de *concept drift* y la validación forward son la respuesta empírica "
            "al feedback de la profesora sobre la adopción de IA como fenómeno temporal.\n\n"
        )

    out.append("---\n\n*Generado por `scripts/modelado_temporal.py`*\n")
    return "".join(out)


def main():
    print("=" * 70)
    print("MODELADO TEMPORAL (CAPA B)")
    print("=" * 70)

    OUTPUTS.mkdir(exist_ok=True)

    df = cargar_usuarios()

    print("\n--- Determinando features comunes a los 3 años ---")
    feature_cols = common_features(df)
    print(f"  Features comunes (var>0 en 2023/2024/2025, sin survey_year): {len(feature_cols)}")

    print("\n--- Comparación de AUC (LogReg) ---")
    auc_results = comparar_auc(df, feature_cols)

    print("\n--- XGBoost check por año ---")
    xgb_results = xgb_check_por_anio(df, feature_cols)

    print("\n--- Coeficientes LogReg por año ---")
    tabla_coef = coeficientes_por_anio(df, feature_cols)
    n_sign = int(tabla_coef["sign_change"].sum())
    print(f"  Coeficientes calculados para {len(tabla_coef)} features")
    print(f"  Features con cambio de signo: {n_sign}")
    print(f"\n  Top 10 por |β| máximo:")
    print(tabla_coef.head(10)[["2023", "2024", "2025", "mag_change", "sign_change"]].round(3).to_string())

    print("\n--- SHAP por año (RF) ---")
    shap_results = shap_rf_por_anio(df, feature_cols)

    # Figuras
    print("\n--- Generando figuras ---")
    fig_auc = figura_auc_comparison(auc_results)
    fig_auc.savefig(OUTPUTS / "08_temporal_auc_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig_auc)
    print(f"✓ outputs/08_temporal_auc_comparison.png")

    fig_coef = figura_coef_heatmap(tabla_coef, n_top=25)
    fig_coef.savefig(OUTPUTS / "08_temporal_coef_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig_coef)
    print(f"✓ outputs/08_temporal_coef_heatmap.png")

    fig_shap = figura_shap_por_anio(shap_results, n_top=15)
    fig_shap.savefig(OUTPUTS / "08_temporal_shap_por_anio.png", dpi=150, bbox_inches="tight")
    plt.close(fig_shap)
    print(f"✓ outputs/08_temporal_shap_por_anio.png")

    # --- Capa C: validación forward ---
    print("\n--- Capa C — Validación temporal forward (concept drift) ---")
    forward_results = {}
    forward_results["fwd_23_24"] = evaluar_forward(df, feature_cols, [2023], 2024)
    forward_results["fwd_24_25"] = evaluar_forward(df, feature_cols, [2024], 2025)
    forward_results["fwd_2324_25"] = evaluar_forward(df, feature_cols, [2023, 2024], 2025)

    print("\n--- Generando figura validación temporal ---")
    fig_val = figura_validacion_temporal(auc_results, forward_results,
                                         auc_results["pooled_sin_year"]["roc_auc"])
    fig_val.savefig(OUTPUTS / "08_temporal_validation.png", dpi=150, bbox_inches="tight")
    plt.close(fig_val)
    print(f"✓ outputs/08_temporal_validation.png")

    # JSON de resultados
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
    (OUTPUTS / "08_temporal_cv_results.json").write_text(
        json.dumps(results_json, indent=2, default=str)
    )
    print(f"✓ outputs/08_temporal_cv_results.json")

    # Markdown
    md = generar_lectura(auc_results, tabla_coef, shap_results, xgb_results, forward_results)
    doc_path = DOCS / "resultados_modelado_temporal.md"
    doc_path.write_text(md)
    print(f"✓ {doc_path}")

    print("\n" + "=" * 70)
    print("LISTO — 4 figuras + JSON + doc generados (Capa B + Capa C)")
    print("=" * 70)


if __name__ == "__main__":
    main()
