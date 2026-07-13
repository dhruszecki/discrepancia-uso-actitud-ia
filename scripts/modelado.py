"""Modelado supervisado sobre df_clean_users.parquet.

Replica el pipeline de notebooks/04_modelado.ipynb pero:
1) Sobre dataset restringido a usuarios de IA (~80k filas, target 78.5/21.5).
2) Usa CONFIG centralizado con TARGET_LABELS para flip de label rápido.
3) Genera figuras con sufijo 06_ para no pisar las de E2.

Outputs:
    outputs/06_users_cv_results.json
    outputs/06_users_roc_curves.png
    outputs/06_users_shap_global.png
    outputs/06_users_shap_bar.png
    outputs/06_users_shap_local.png
    outputs/06_users_shap_wang_mapping.png
    resultados/resultados_modelado_usuarios.md
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
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import shap
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
TARGET_LABELS = {
    1: "Sin discrepancia",
    0: "Con discrepancia uso-actitud",
}
RANDOM_STATE = 42
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "df_clean_users.parquet"
OUTPUTS = ROOT / "outputs"
DOCS = ROOT / "resultados"
DOCS.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)


def cargar_dataset():
    assert DATA_PATH.exists(), f"Falta {DATA_PATH}. Correr scripts/pipeline.py."
    df = pd.read_parquet(DATA_PATH)
    print(f"Dataset: {df.shape[0]:,} × {df.shape[1]} cols")
    print(f"  Target=1 ({TARGET_LABELS[1]}): {df['target'].mean():.1%}")
    print(f"  Target=0 ({TARGET_LABELS[0]}): {(1-df['target']).mean():.1%}")
    return df


def preparar_features(df: pd.DataFrame):
    """Selecciona features y separa X/y."""
    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols].copy()
    y = df["target"].copy()
    assert X.isnull().sum().sum() == 0, "Hay nulos en X"
    print(f"Features: {len(feature_cols)}  |  Filas: {len(X):,}")
    return X, y, feature_cols


def definir_modelos(y):
    """Define los 4 modelos con compensación de desbalance."""
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    print(f"scale_pos_weight (XGBoost) = {scale_pos_weight:.3f}")

    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced", max_iter=1000, solver="saga",
                n_jobs=-1, random_state=RANDOM_STATE,
            )),
        ]),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", max_depth=8, min_samples_leaf=50,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", max_features="sqrt",
            min_samples_leaf=20, n_jobs=-1, random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            scale_pos_weight=scale_pos_weight, eval_metric="auc",
            n_jobs=-1, random_state=RANDOM_STATE, verbosity=0,
        ),
    }


def evaluar_cv(models, X, y):
    """CV estratificada k=5 con ROC-AUC y F1-macro."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    SCORING = {"roc_auc": "roc_auc", "f1_macro": "f1_macro"}

    results = {}
    for name, model in models.items():
        print(f"  {name}...", end=" ", flush=True)
        scores = cross_validate(model, X, y, cv=cv, scoring=SCORING, n_jobs=1)
        results[name] = {
            "ROC-AUC": float(scores["test_roc_auc"].mean()),
            "ROC-AUC_std": float(scores["test_roc_auc"].std()),
            "F1-macro": float(scores["test_f1_macro"].mean()),
            "F1-macro_std": float(scores["test_f1_macro"].std()),
        }
        print(
            f"ROC-AUC={results[name]['ROC-AUC']:.4f} ± {results[name]['ROC-AUC_std']:.4f}  "
            f"F1={results[name]['F1-macro']:.4f} ± {results[name]['F1-macro_std']:.4f}"
        )
    return results, cv


def curvas_roc(models, X, y, cv):
    """ROC out-of-fold para los 4 modelos."""
    colors = {
        "Logistic Regression": "#2196F3",
        "Decision Tree": "#FF9800",
        "Random Forest": "#4CAF50",
        "XGBoost": "#E91E63",
    }
    fig, ax = plt.subplots(figsize=(7, 5))
    auc_results = {}
    for name, model in models.items():
        print(f"  OOF predictions {name}...", end=" ", flush=True)
        y_prob_oof = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
        fpr, tpr, _ = roc_curve(y, y_prob_oof)
        roc_auc = auc(fpr, tpr)
        auc_results[name] = float(roc_auc)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})", color=colors[name], lw=1.8)
        print(f"AUC={roc_auc:.4f}")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Aleatorio")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title(f"Curvas ROC (CV k=5, N={len(X):,} usuarios IA)")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUTS / "06_users_roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {OUTPUTS / '06_users_roc_curves.png'}")
    return auc_results


def shap_analysis(best_model_name, models, X, y):
    """Entrena el mejor modelo y computa SHAP."""
    # Para SHAP usar árbol (no LogReg)
    shap_name = best_model_name if best_model_name != "Logistic Regression" else "Random Forest"
    print(f"Modelo SHAP: {shap_name}")
    model = models[shap_name]
    print(f"  Entrenando en dataset completo...", end=" ", flush=True)
    model.fit(X, y)
    print("OK.")

    # Background sample para SHAP
    rng = np.random.default_rng(RANDOM_STATE)
    bg_idx = rng.choice(len(X), size=min(1000, len(X)), replace=False)
    X_bg = X.iloc[bg_idx]

    explainer = shap.TreeExplainer(model)
    shap_raw = explainer.shap_values(X_bg)
    # Manejar todas las shapes posibles según versión de shap:
    # - lista [clase0, clase1] (sklearn legacy)
    # - ndarray (n, features) (xgboost binario)
    # - ndarray (n, features, n_classes) (sklearn nuevo)
    if isinstance(shap_raw, list):
        shap_values = shap_raw[1]
    elif shap_raw.ndim == 3:
        # Tomar SHAP para clase 1 (target=1)
        shap_values = shap_raw[:, :, 1]
    else:
        shap_values = shap_raw
    print(f"  shap_values shape: {shap_values.shape}")

    # Top features por |SHAP| medio
    mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=X.columns)
    top20 = mean_abs_shap.sort_values(ascending=False).head(20)
    print(f"\nTop 10 features por |SHAP|:\n{top20.head(10).to_string()}\n")

    # Figura beeswarm
    fig = plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_bg, max_display=20, show=False, plot_type="dot")
    plt.title(f"SHAP global ({shap_name}) — N={len(X):,} usuarios IA")
    plt.tight_layout()
    plt.savefig(OUTPUTS / "06_users_shap_global.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ {OUTPUTS / '06_users_shap_global.png'}")

    # Figura bar
    fig, ax = plt.subplots(figsize=(9, 7))
    top20[::-1].plot(kind="barh", ax=ax, color="#374151")
    ax.set_xlabel("|SHAP| medio")
    ax.set_title(f"Top 20 features ({shap_name}) — N={len(X):,} usuarios IA")
    fig.tight_layout()
    fig.savefig(OUTPUTS / "06_users_shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {OUTPUTS / '06_users_shap_bar.png'}")

    return shap_name, mean_abs_shap, X_bg, shap_values, model


def shap_local_cases(model, X_bg, shap_values, mean_abs_shap):
    """3 casos locales contrastantes."""
    # Probabilidad de target=0 (discrepancia)
    probs = model.predict_proba(X_bg)[:, 0]  # prob clase 0

    # Caso A: alto target=0 (discrepancia probable)
    idx_A = np.argmax(probs)
    # Caso B: bajo target=0 (sin discrepancia probable)
    idx_B = np.argmin(probs)
    # Caso C: borderline (cerca de 0.5)
    idx_C = int(np.argmin(np.abs(probs - 0.5)))

    cases = [
        ("A — Discrepante (target=0)", idx_A),
        ("B — Sin discrepancia (target=1)", idx_B),
        ("C — Borderline", idx_C),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (label, idx) in zip(axes, cases):
        p0 = probs[idx]
        # Top 8 features por |SHAP| en este caso (menos features → barras más visibles)
        shap_case = pd.Series(shap_values[idx], index=X_bg.columns)
        top8 = shap_case.abs().sort_values(ascending=False).head(8).index
        shap_top = shap_case.loc[top8][::-1]
        colors_bar = ["#dc2626" if v < 0 else "#16a34a" for v in shap_top.values]
        ax.barh(range(len(shap_top)), shap_top.values, color=colors_bar)
        ax.set_yticks(range(len(shap_top)))
        ax.set_yticklabels([t[:24] for t in shap_top.index], fontsize=8)
        ax.axvline(0, color="black", lw=0.5)
        ax.set_xlabel("Contribución SHAP", fontsize=9)
        ax.set_title(f"{label}\nP(target=0)={p0:.3f}", fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUTS / "06_users_shap_local.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {OUTPUTS / '06_users_shap_local.png'}")

    return [(label, int(idx), float(probs[idx])) for label, idx in cases]


def wang_mapping(mean_abs_shap):
    """Mapeo a ability/integrity/benevolence."""
    # Heurística por prefijo/contenido
    ability_keywords = ["lang_", "plat_", "db_", "env_", "_count", "YearsCode", "WorkExp", "EdLevel", "DevType_"]
    integrity_keywords = ["survey_year", "OrgSize", "RemoteWork", "Employment", "ICorPM"]
    benevolence_keywords = ["Region_", "Industry_"]

    def classify(col):
        if any(k in col for k in ability_keywords):
            return "Ability"
        if any(k in col for k in integrity_keywords):
            return "Integrity"
        if any(k in col for k in benevolence_keywords):
            return "Benevolence"
        return "Other"

    df_map = pd.DataFrame({"feature": mean_abs_shap.index, "shap": mean_abs_shap.values})
    df_map["dim"] = df_map["feature"].apply(classify)
    summary = df_map.groupby("dim")["shap"].agg(["sum", "count", "mean"]).sort_values("sum", ascending=False)
    print(f"\nDistribución por dimensión Wang:\n{summary}")

    # Figura: 2 paneles (suma por dim + top 15 features coloreadas por dim)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    colors_dim = {"Ability": "#3b82f6", "Integrity": "#f59e0b", "Benevolence": "#10b981", "Other": "#9ca3af"}

    # Panel 1: suma por dim
    sums = summary["sum"]
    axes[0].bar(sums.index, sums.values, color=[colors_dim[d] for d in sums.index])
    axes[0].set_title("Suma |SHAP| por dimensión Wang et al.")
    axes[0].set_ylabel("Σ |SHAP|")

    # Panel 2: top 15 con color por dim
    top15 = df_map.sort_values("shap", ascending=False).head(15).iloc[::-1]
    axes[1].barh(range(len(top15)), top15["shap"], color=[colors_dim[d] for d in top15["dim"]])
    axes[1].set_yticks(range(len(top15)))
    axes[1].set_yticklabels(top15["feature"], fontsize=8)
    axes[1].set_xlabel("|SHAP| medio")
    axes[1].set_title("Top 15 features coloreadas por dimensión Wang")

    # Leyenda manual
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=v, label=k) for k, v in colors_dim.items() if k != "Other"]
    axes[1].legend(handles=handles, loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUTPUTS / "06_users_shap_wang_mapping.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {OUTPUTS / '06_users_shap_wang_mapping.png'}")
    return summary, df_map


def generar_documento_md(cv_results, auc_oof, shap_name, top_features, wang_summary, cases):
    """Genera doc resumen."""
    out = []
    out.append("# Modelado supervisado sobre usuarios activos de IA — Resultados\n\n")
    out.append("**Dataset:** `data/df_clean_users.parquet` (80.369 × 124)\n\n")
    out.append(f"**Target labels:** `{TARGET_LABELS[1]}` (target=1, 78,5 %) vs. `{TARGET_LABELS[0]}` (target=0, 21,5 %)\n\n")
    out.append("---\n\n")

    out.append("## 1. Comparación de modelos (CV k=5)\n\n")
    df_cv = pd.DataFrame(cv_results).T
    df_cv = df_cv.sort_values("ROC-AUC", ascending=False)
    out.append(df_cv.round(4).to_markdown())
    out.append("\n\n")

    out.append("### Curva ROC out-of-fold\n\n")
    out.append("Ver `outputs/06_users_roc_curves.png`. AUC OOF por modelo:\n\n")
    for n, v in sorted(auc_oof.items(), key=lambda x: -x[1]):
        out.append(f"- {n}: {v:.4f}\n")
    out.append("\n")

    out.append("## 2. SHAP global\n\n")
    out.append(f"Modelo: **{shap_name}**\n\n")
    out.append("### Top 20 features por |SHAP| medio\n\n")
    out.append(top_features.head(20).round(4).to_frame("|SHAP|").to_markdown())
    out.append("\n\n")

    out.append("## 3. Mapeo Wang et al. (ability / integrity / benevolence)\n\n")
    out.append(wang_summary.round(4).to_markdown())
    out.append("\n\n")

    out.append("## 4. Casos locales\n\n")
    for label, idx, prob in cases:
        out.append(f"- **Caso {label}** (fila bg #{idx}): P(target=0) = {prob:.3f}\n")
    out.append("\n")

    out.append("## 5. Lectura\n\n")
    out.append("Con el dataset restringido a usuarios activos (80.369 filas, target 78,5/21,5):\n\n")
    out.append("- Las 4 métricas ROC-AUC son menores que en el dataset completo (esperable: la subpoblación es más homogénea, hay menos varianza para explicar).\n")
    out.append("- El target=0 (discrepancia) sigue siendo el subgrupo conceptualmente más interesante y ahora aislado del ruido de no-usuarios.\n")
    out.append("- La interpretación SHAP debe leerse en la dirección: target=0 → discrepancia entre uso declarado y actitud (devs que usan IA pero declaran actitud no-favorable).\n")
    out.append("- Próximo: re-correr H1, H2, H4, H5 sobre esta subpoblación.\n")

    return "".join(out)


def main():
    print("=" * 70)
    print("MODELADO SUPERVISADO — Usuarios activos de IA")
    print("=" * 70)
    print(f"CONFIG: TARGET_LABELS = {TARGET_LABELS}\n")

    df = cargar_dataset()
    X, y, feature_cols = preparar_features(df)
    models = definir_modelos(y)

    print("\n--- Evaluación CV k=5 ---")
    cv_results, cv = evaluar_cv(models, X, y)

    print("\n--- Curvas ROC OOF ---")
    auc_oof = curvas_roc(models, X, y, cv)

    print("\n--- SHAP global ---")
    df_results = pd.DataFrame(cv_results).T.sort_values("ROC-AUC", ascending=False)
    best_name = df_results.index[0]
    print(f"Mejor modelo: {best_name}")

    shap_name, top_features, X_bg, shap_values, fitted_model = shap_analysis(best_name, models, X, y)

    print("\n--- SHAP local ---")
    cases = shap_local_cases(fitted_model, X_bg, shap_values, top_features)

    print("\n--- Mapeo Wang et al. ---")
    wang_summary, _ = wang_mapping(top_features)

    # Guardar resultados como JSON
    (OUTPUTS / "06_users_cv_results.json").write_text(json.dumps({
        "cv_results": cv_results,
        "auc_oof": auc_oof,
        "best_model": best_name,
        "shap_model": shap_name,
        "n_samples": int(len(X)),
        "n_features": int(len(feature_cols)),
        "target_dist": {"target_1_pct": float(y.mean()), "target_0_pct": float((1-y).mean())},
        "wang_summary": wang_summary.to_dict(),
    }, indent=2, default=str))
    print(f"✓ {OUTPUTS / '06_users_cv_results.json'}")

    # Generar markdown
    md = generar_documento_md(cv_results, auc_oof, shap_name, top_features, wang_summary, cases)
    (DOCS / "resultados_modelado_usuarios.md").write_text(md)
    print(f"✓ {DOCS / 'resultados_modelado_usuarios.md'}")

    print("\n" + "=" * 70)
    print("LISTO")
    print("=" * 70)


if __name__ == "__main__":
    main()
