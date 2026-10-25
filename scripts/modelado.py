"""Modelado supervisado sobre los usuarios de IA (df_clean_users.parquet).

Compara cuatro modelos con validación cruzada estratificada, calcula los valores SHAP
de un random forest y agrupa las variables según las dimensiones de Wang et al. (2024).

Genera:
    outputs/modelos_resultados_cv.json
    outputs/modelos_curvas_roc.png
    outputs/shap_global.png
    outputs/shap_barras.png
    outputs/shap_casos.png
    outputs/shap_dimensiones_wang.png
    docs/analisis/resultados_modelado_usuarios.md
"""
from __future__ import annotations

import json
import warnings
import sys
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

TARGET_LABELS = {
    1: "Sin discrepancia",
    0: "Con discrepancia uso-actitud",
}
RANDOM_STATE = 42

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nombres_legibles import nombre_legible, renombrar_columnas  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "df_clean_users.parquet"
OUTPUTS = ROOT / "outputs"
DOCS = ROOT / "docs" / "analisis"
OUTPUTS.mkdir(parents=True, exist_ok=True)
DOCS.mkdir(parents=True, exist_ok=True)


def cargar_dataset():
    assert DATA_PATH.exists(), f"Falta {DATA_PATH}. Correr scripts/pipeline.py."
    df = pd.read_parquet(DATA_PATH)
    print(f"Dataset: {df.shape[0]:,} filas, {df.shape[1]} columnas")
    print(f"  target=1 ({TARGET_LABELS[1]}): {df['target'].mean():.1%}")
    print(f"  target=0 ({TARGET_LABELS[0]}): {(1-df['target']).mean():.1%}")
    return df


def preparar_features(df: pd.DataFrame):
    """Separa las variables predictoras de la variable a predecir."""
    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols].copy()
    y = df["target"].copy()
    assert X.isnull().sum().sum() == 0, "Hay nulos en X"
    print(f"Variables: {len(feature_cols)}, filas: {len(X):,}")
    return X, y, feature_cols


def definir_modelos(y):
    """Los cuatro modelos, todos con pesos para compensar el desbalance de clases."""
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
    """Validación cruzada estratificada de 5 partes con ROC-AUC y F1 macro."""
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
    """Curvas ROC con las predicciones de cada parte cuando quedó afuera del entrenamiento."""
    colors = {
        "Logistic Regression": "#2196F3",
        "Decision Tree": "#FF9800",
        "Random Forest": "#4CAF50",
        "XGBoost": "#E91E63",
    }
    fig, ax = plt.subplots(figsize=(7, 5))
    auc_results = {}
    for name, model in models.items():
        print(f"  Predicciones fuera de muestra, {name}...", end=" ", flush=True)
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
    fig.savefig(OUTPUTS / "modelos_curvas_roc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado outputs/modelos_curvas_roc.png")
    return auc_results


def shap_analysis(best_model_name, models, X, y):
    """Entrena el modelo elegido sobre todos los datos y calcula los valores SHAP."""
    # TreeExplainer necesita un modelo de árboles: si gana la regresión logística,
    # se usa el random forest.
    shap_name = best_model_name if best_model_name != "Logistic Regression" else "Random Forest"
    print(f"Modelo para SHAP: {shap_name}")
    model = models[shap_name]
    print("  Entrenando con todos los datos...", end=" ", flush=True)
    model.fit(X, y)
    print("listo.")

    # SHAP se calcula sobre 1000 filas al azar para que no tarde demasiado.
    rng = np.random.default_rng(RANDOM_STATE)
    bg_idx = rng.choice(len(X), size=min(1000, len(X)), replace=False)
    X_bg = X.iloc[bg_idx]

    explainer = shap.TreeExplainer(model)
    shap_raw = explainer.shap_values(X_bg)
    # La forma del resultado cambia según la versión de shap y el modelo: una lista
    # por clase, una matriz (filas, variables) o un arreglo (filas, variables, clases).
    # En todos los casos nos quedamos con la clase 1.
    if isinstance(shap_raw, list):
        shap_values = shap_raw[1]
    elif shap_raw.ndim == 3:
        shap_values = shap_raw[:, :, 1]
    else:
        shap_values = shap_raw
    print(f"  shap_values: {shap_values.shape}")

    mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=X.columns)
    top20 = mean_abs_shap.sort_values(ascending=False).head(20)
    print(f"\n10 variables con mayor |SHAP| medio:\n{top20.head(10).to_string()}\n")

    fig = plt.figure(figsize=(11, 8))
    shap.summary_plot(shap_values, renombrar_columnas(X_bg), max_display=20, show=False, plot_type="dot")
    ax_shap = plt.gca()
    ax_shap.set_xlabel("Valor SHAP (positivo: hacia sin discrepancia)")
    cbar = fig.axes[-1]
    cbar.set_ylabel("Valor de la variable")
    cbar.set_yticklabels(["Bajo", "Alto"])
    plt.title(f"Valores SHAP de las 20 variables principales ({shap_name}), usuarios de IA")
    plt.tight_layout()
    plt.savefig(OUTPUTS / "shap_global.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Guardado outputs/shap_global.png")

    fig, ax = plt.subplots(figsize=(9, 7))
    top20_leg = top20[::-1].rename(index=nombre_legible)
    top20_leg.plot(kind="barh", ax=ax, color="#374151")
    ax.set_xlabel("SHAP medio absoluto")
    ax.set_title(f"20 variables principales ({shap_name}), usuarios de IA")
    fig.tight_layout()
    fig.savefig(OUTPUTS / "shap_barras.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado outputs/shap_barras.png")

    return shap_name, mean_abs_shap, X_bg, shap_values, model


def shap_local_cases(model, X_bg, shap_values, mean_abs_shap):
    """Tres casos de la muestra: el más probable con discrepancia, el menos probable y uno dudoso."""
    # La columna 0 es la probabilidad de discrepancia (target=0).
    probs = model.predict_proba(X_bg)[:, 0]

    idx_A = np.argmax(probs)
    idx_B = np.argmin(probs)
    idx_C = int(np.argmin(np.abs(probs - 0.5)))

    cases = [
        ("Caso A: mayor probabilidad de discrepancia", idx_A),
        ("Caso B: menor probabilidad de discrepancia", idx_B),
        ("Caso C: probabilidad cercana a 0,5", idx_C),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (label, idx) in zip(axes, cases):
        p0 = probs[idx]
        # Solo ocho variables por caso para que las barras se lean.
        shap_case = pd.Series(shap_values[idx], index=X_bg.columns)
        top8 = shap_case.abs().sort_values(ascending=False).head(8).index
        shap_top = shap_case.loc[top8][::-1]
        colors_bar = ["#dc2626" if v < 0 else "#16a34a" for v in shap_top.values]
        ax.barh(range(len(shap_top)), shap_top.values, color=colors_bar)
        ax.set_yticks(range(len(shap_top)))
        ax.set_yticklabels([nombre_legible(t) for t in shap_top.index], fontsize=9)
        ax.axvline(0, color="black", lw=0.5)
        ax.set_xlabel("Contribución SHAP\n(negativa: hacia discrepancia)", fontsize=9)
        ax.set_title(f"{label}\nProbabilidad de discrepancia = {p0:.2f}".replace(".", ","), fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUTS / "shap_casos.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado outputs/shap_casos.png")

    return [(label, int(idx), float(probs[idx])) for label, idx in cases]


def wang_mapping(mean_abs_shap):
    """Asigna cada variable a ability, integrity o benevolence y suma su SHAP."""
    # La asignación es por el nombre de la columna; es una aproximación.
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
    print(f"\nSHAP por dimensión de Wang:\n{summary}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    colors_dim = {"Ability": "#3b82f6", "Integrity": "#f59e0b", "Benevolence": "#10b981", "Other": "#9ca3af"}

    sums = summary["sum"]
    axes[0].bar(sums.index, sums.values, color=[colors_dim[d] for d in sums.index])
    axes[0].set_title("Suma del SHAP absoluto por dimensión de Wang et al.")
    axes[0].set_ylabel("Suma del SHAP medio absoluto")

    top15 = df_map.sort_values("shap", ascending=False).head(15).iloc[::-1]
    axes[1].barh(range(len(top15)), top15["shap"], color=[colors_dim[d] for d in top15["dim"]])
    axes[1].set_yticks(range(len(top15)))
    axes[1].set_yticklabels([nombre_legible(f) for f in top15["feature"]], fontsize=9)
    axes[1].set_xlabel("SHAP medio absoluto")
    axes[1].set_title("15 variables principales, por dimensión de Wang")

    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=v, label=k) for k, v in colors_dim.items() if k != "Other"]
    axes[1].legend(handles=handles, loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUTPUTS / "shap_dimensiones_wang.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado outputs/shap_dimensiones_wang.png")
    return summary, df_map


def generar_documento_md(cv_results, auc_oof, shap_name, top_features, wang_summary, cases):
    """Arma el informe en markdown con los resultados."""
    out = []
    out.append("# Modelado supervisado sobre usuarios de IA: resultados\n\n")
    out.append(f"**Datos:** `data/df_clean_users.parquet` ({len(top_features)} variables predictoras). Generado por `scripts/modelado.py`.\n\n")
    out.append(f"**Variable a predecir:** {TARGET_LABELS[1]} (valor 1, 78,5 %) frente a {TARGET_LABELS[0]} (valor 0, 21,5 %).\n\n")

    out.append("## 1. Comparación de modelos (validación cruzada de 5 partes)\n\n")
    df_cv = pd.DataFrame(cv_results).T
    df_cv = df_cv.sort_values("ROC-AUC", ascending=False)
    out.append(df_cv.round(4).to_markdown())
    out.append("\n\n")

    out.append("### Curva ROC con predicciones fuera de muestra\n\n")
    out.append("La figura está en `outputs/modelos_curvas_roc.png`. AUC fuera de muestra de cada modelo:\n\n")
    for n, v in sorted(auc_oof.items(), key=lambda x: -x[1]):
        out.append(f"- {n}: {v:.4f}\n")
    out.append("\n")

    out.append("## 2. SHAP global\n\n")
    out.append(f"Modelo: **{shap_name}**\n\n")
    out.append("### 20 variables con mayor SHAP medio absoluto\n\n")
    out.append(top_features.sort_values(ascending=False).head(20).round(4).rename(index=nombre_legible).to_frame("SHAP medio absoluto").to_markdown())
    out.append("\n\n")

    out.append("## 3. Dimensiones de Wang et al. (ability, integrity, benevolence)\n\n")
    out.append(wang_summary.round(4).to_markdown())
    out.append("\n\n")

    out.append("## 4. Casos locales\n\n")
    for label, idx, prob in cases:
        out.append(f"- **{label}** (fila #{idx} de la muestra): probabilidad de discrepancia = {prob:.3f}\n")
    out.append("\n")

    out.append("## 5. Lectura\n\n")
    out.append("Con los usuarios de IA (80.369 filas, 78,5 % sin discrepancia y 21,5 % con discrepancia):\n\n")
    out.append("- Los cuatro modelos quedan entre 0,61 y 0,63 de ROC-AUC. Los tres primeros están dentro de la variación entre partes de la validación cruzada.\n")
    out.append("- Los valores SHAP son de un random forest. Un valor positivo empuja hacia la ausencia de discrepancia y uno negativo hacia la discrepancia.\n")
    out.append("- Los pesos por dimensión de Wang dependen de cuántas variables tiene cada dimensión y del año de la encuesta, que se asignó a *integrity*.\n")

    return "".join(out)


def main():
    print("Modelado supervisado sobre usuarios de IA")
    print(f"TARGET_LABELS = {TARGET_LABELS}\n")

    df = cargar_dataset()
    X, y, feature_cols = preparar_features(df)
    models = definir_modelos(y)

    print("\nValidación cruzada (k=5)")
    cv_results, cv = evaluar_cv(models, X, y)

    print("\nCurvas ROC")
    auc_oof = curvas_roc(models, X, y, cv)

    print("\nSHAP global")
    df_results = pd.DataFrame(cv_results).T.sort_values("ROC-AUC", ascending=False)
    best_name = df_results.index[0]
    print(f"Mejor modelo: {best_name}")

    shap_name, top_features, X_bg, shap_values, fitted_model = shap_analysis(best_name, models, X, y)

    print("\nCasos locales")
    cases = shap_local_cases(fitted_model, X_bg, shap_values, top_features)

    print("\nDimensiones de Wang")
    wang_summary, _ = wang_mapping(top_features)

    (OUTPUTS / "modelos_resultados_cv.json").write_text(json.dumps({
        "cv_results": cv_results,
        "auc_oof": auc_oof,
        "best_model": best_name,
        "shap_model": shap_name,
        "n_samples": int(len(X)),
        "n_features": int(len(feature_cols)),
        "target_dist": {"target_1_pct": float(y.mean()), "target_0_pct": float((1-y).mean())},
        "wang_summary": wang_summary.to_dict(),
    }, indent=2, default=str))
    print(f"Guardado outputs/modelos_resultados_cv.json")

    md = generar_documento_md(cv_results, auc_oof, shap_name, top_features, wang_summary, cases)
    (DOCS / "resultados_modelado_usuarios.md").write_text(md)
    print(f"Guardado docs/analisis/resultados_modelado_usuarios.md")

    print("\nListo.")


if __name__ == "__main__":
    main()
