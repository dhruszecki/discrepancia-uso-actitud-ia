"""Compara el AUC de la misma regresión logística sobre todos los desarrolladores y sobre los usuarios de IA.

Con la misma validación cruzada y el mismo modelo mide cuánto baja el ROC-AUC al
pasar a los usuarios de IA, el PR-AUC de la clase con discrepancia (la minoritaria,
21,5 %) y el umbral que maximiza F1. Escribe docs/analisis/decision_drop_auc.md y
outputs/comparacion_todos_vs_usuarios.png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
DOCS = ROOT / "docs" / "analisis"
OUTPUTS.mkdir(parents=True, exist_ok=True)
DOCS.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

TARGET_LABELS = {
    1: "Sin discrepancia",
    0: "Con discrepancia uso-actitud",
}


def cargar_dataset(restricted: bool) -> tuple[pd.DataFrame, pd.Series, str]:
    if restricted:
        df = pd.read_parquet(DATA / "df_clean_users.parquet")
        name = "Usuarios de IA (~80k)"
    else:
        df = pd.read_parquet(DATA / "df_clean.parquet")
        name = "Todos los desarrolladores (155k)"
    # cluster sale del análisis exploratorio, no es una variable predictora
    df = df.drop(columns=[c for c in ["cluster"] if c in df.columns])
    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols].copy()
    # El pipeline ya codifica todo; esto es por si quedó alguna columna de texto.
    for c in X.columns:
        if X[c].dtype == "object":
            X[c] = pd.factorize(X[c])[0]
    y = df["target"].copy()
    return X, y, name


def evaluar_logreg(X: pd.DataFrame, y: pd.Series, label: str) -> dict:
    """Regresión logística con validación cruzada de 5 partes; métricas sobre las predicciones fuera de cada parte."""
    print(f"\n{label} (N={len(y):,}, sin discrepancia: {y.mean():.1%})")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced", max_iter=1000, solver="saga",
            n_jobs=-1, random_state=RANDOM_STATE,
        )),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    y_prob = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    roc_auc = roc_auc_score(y, y_prob)
    pr_auc_pos = average_precision_score(y, y_prob)
    # Entre usuarios la clase de interés (discrepancia, 0) es la minoritaria,
    # así que también se mide su PR-AUC invirtiendo etiquetas y probabilidades.
    pr_auc_neg = average_precision_score(1 - y, 1 - y_prob)
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred, target_names=[TARGET_LABELS[0], TARGET_LABELS[1]], digits=3, output_dict=True)

    print(f"  ROC-AUC = {roc_auc:.4f}")
    print(f"  PR-AUC (clase 1, 'Sin discrepancia') = {pr_auc_pos:.4f}")
    print(f"  PR-AUC (clase 0, '{TARGET_LABELS[0]}') = {pr_auc_neg:.4f}")
    print("  Matriz de confusión (filas: real 0 y 1; columnas: predicho 0 y 1):")
    print(f"    {cm[0,0]:>6}    {cm[0,1]:>6}")
    print(f"    {cm[1,0]:>6}    {cm[1,1]:>6}")

    return {
        "label": label,
        "n": int(len(y)),
        "prevalence_target1": float(y.mean()),
        "roc_auc": float(roc_auc),
        "pr_auc_pos": float(pr_auc_pos),
        "pr_auc_neg": float(pr_auc_neg),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "y_true": y.values,
        "y_prob": y_prob,
    }


def threshold_analysis(y_true: np.ndarray, y_prob: np.ndarray, label: str):
    """Umbral que maximiza F1 para detectar la clase con discrepancia."""
    p_neg = 1 - y_prob
    y_neg = 1 - y_true

    precision, recall, thresholds = precision_recall_curve(y_neg, p_neg)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    best_idx = int(np.argmax(f1))
    best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 1.0
    best_p = float(precision[best_idx])
    best_r = float(recall[best_idx])
    best_f1 = float(f1[best_idx])

    print(f"  {label}: umbral de máximo F1 = {best_threshold:.3f}, precisión = {best_p:.3f}, "
          f"recall = {best_r:.3f}, F1 = {best_f1:.3f}")

    return precision, recall, thresholds, (best_threshold, best_p, best_r, best_f1)


def plot_comparison(res_full: dict, res_restricted: dict):
    """Matrices de confusión y curvas precisión-recall de los dos conjuntos."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    cm = np.array(res_full["confusion_matrix"])
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    axes[0,0].imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            axes[0,0].text(j, i, f"{cm[i,j]:,}\n({cm_norm[i,j]:.1%})",
                          ha="center", va="center", fontsize=9,
                          color="white" if cm_norm[i,j] > 0.5 else "black")
    axes[0,0].set_xticks([0,1])
    axes[0,0].set_yticks([0,1])
    axes[0,0].set_xticklabels(["Predice\ncon discrepancia", "Predice\nsin discrepancia"])
    axes[0,0].set_yticklabels(["Real:\ncon discrepancia", "Real:\nsin discrepancia"])
    axes[0,0].set_title(f"Todos los desarrolladores (N = {res_full['n']:,})\nROC-AUC = {res_full['roc_auc']:.3f}")

    cm = np.array(res_restricted["confusion_matrix"])
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    axes[0,1].imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            axes[0,1].text(j, i, f"{cm[i,j]:,}\n({cm_norm[i,j]:.1%})",
                          ha="center", va="center", fontsize=9,
                          color="white" if cm_norm[i,j] > 0.5 else "black")
    axes[0,1].set_xticks([0,1])
    axes[0,1].set_yticks([0,1])
    axes[0,1].set_xticklabels(["Predice\ncon discrepancia", "Predice\nsin discrepancia"])
    axes[0,1].set_yticklabels(["Real:\ncon discrepancia", "Real:\nsin discrepancia"])
    axes[0,1].set_title(f"Usuarios de IA (N = {res_restricted['n']:,})\nROC-AUC = {res_restricted['roc_auc']:.3f}")

    for ax, res, title_suffix in [(axes[1,0], res_full, "todos los desarrolladores"), (axes[1,1], res_restricted, "usuarios de IA")]:
        precision, recall, _ = precision_recall_curve(1 - res["y_true"], 1 - res["y_prob"])
        ap = res["pr_auc_neg"]
        ax.plot(recall, precision, lw=2, color="#dc2626")
        ax.axhline((1 - res["prevalence_target1"]), color="grey", ls="--", alpha=0.5,
                   label=f"Referencia (prevalencia): {1 - res['prevalence_target1']:.1%}")
        ax.set_xlabel("Recall de la clase con discrepancia")
        ax.set_ylabel("Precisión de la clase con discrepancia")
        ax.set_title(f"Curva precisión–recall, {title_suffix} (AP = {ap:.3f})")
        ax.legend(fontsize=9, loc="lower left")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUTS / "comparacion_todos_vs_usuarios.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nGuardado outputs/comparacion_todos_vs_usuarios.png")


def generar_doc(res_full: dict, res_restricted: dict, best_th_full, best_th_restricted) -> str:
    prev0_full = 1 - res_full["prevalence_target1"]
    prev0_res = 1 - res_restricted["prevalence_target1"]
    lift_res = (res_restricted["pr_auc_neg"] - prev0_res) * 100
    caida = (res_full["roc_auc"] - res_restricted["roc_auc"]) * 100

    out = []
    out.append("# Caída del ROC-AUC al restringir a usuarios de IA\n\n")
    out.append("La pregunta es cuánto baja el ROC-AUC al pasar de todos los desarrolladores a los usuarios de IA, "
               "con el mismo modelo y la misma validación cruzada. Generado por `scripts/decision_drop_auc.py`.\n\n")

    out.append("## 1. Comparación de métricas (validación cruzada de 5 partes, regresión logística con pesos por clase)\n\n")
    out.append("| Métrica | Todos los desarrolladores (155k) | Usuarios de IA (80k) |\n")
    out.append("|---|---:|---:|\n")
    out.append(f"| Prevalencia sin discrepancia (valor 1) | {res_full['prevalence_target1']:.1%} | {res_restricted['prevalence_target1']:.1%} |\n")
    out.append(f"| Prevalencia con discrepancia (valor 0) | {prev0_full:.1%} | {prev0_res:.1%} |\n")
    out.append(f"| ROC-AUC | {res_full['roc_auc']:.4f} | {res_restricted['roc_auc']:.4f} |\n")
    out.append(f"| PR-AUC (clase sin discrepancia) | {res_full['pr_auc_pos']:.4f} | {res_restricted['pr_auc_pos']:.4f} |\n")
    out.append(f"| PR-AUC (clase con discrepancia) | {res_full['pr_auc_neg']:.4f} | {res_restricted['pr_auc_neg']:.4f} |\n")
    out.append(f"\nEl ROC-AUC cae {caida:.1f} puntos. Un análisis anterior con otro modelo sobre todos los datos había reportado 0,73.\n\n")

    out.append("## 2. Interpretación\n\n")
    out.append("- Con todos los datos, parte de la señal viene de separar a los usuarios de los no usuarios. "
               "Entre usuarios de IA esa separación desaparece y queda la señal del perfil.\n")
    out.append("- El ROC-AUC no depende de la prevalencia de las clases, así que la caída no se explica por el cambio de 40/60 a 78/22.\n")
    out.append(f"- Entre usuarios, el PR-AUC de la clase con discrepancia es {res_restricted['pr_auc_neg']:.3f}, contra una referencia "
               f"de {prev0_res:.3f} (la prevalencia). La supera en {lift_res:.1f} puntos: hay señal, aunque modesta.\n\n")

    out.append("## 3. Umbral de decisión entre usuarios de IA\n\n")
    th, p, r, f1 = best_th_restricted
    out.append(f"Sobre la curva precisión–recall de la clase con discrepancia, el umbral que maximiza F1 es {th:.3f} (el umbral por defecto es 0,5).\n\n")
    out.append(f"- Precisión en ese umbral: {p:.3f}\n")
    out.append(f"- Recall en ese umbral: {r:.3f}\n")
    out.append(f"- F1: {f1:.3f}\n\n")
    out.append("Si el objetivo es clasificar a quienes tienen discrepancia, conviene reportar las métricas con ese umbral. "
               "Si el objetivo es interpretar qué variables se asocian con la discrepancia, el umbral importa poco y la salida principal es SHAP.\n")

    return "".join(out)


def main():
    print("Caída del ROC-AUC al restringir a usuarios de IA")

    X_full, y_full, label_full = cargar_dataset(restricted=False)
    res_full = evaluar_logreg(X_full, y_full, label_full)

    X_res, y_res, label_res = cargar_dataset(restricted=True)
    res_res = evaluar_logreg(X_res, y_res, label_res)

    print("\nUmbral de máximo F1 para la clase con discrepancia")
    _, _, _, best_th_full = threshold_analysis(res_full["y_true"], res_full["y_prob"], "Todos")
    _, _, _, best_th_res = threshold_analysis(res_res["y_true"], res_res["y_prob"], "Usuarios")

    plot_comparison(res_full, res_res)

    md = generar_doc(res_full, res_res, best_th_full, best_th_res)
    (DOCS / "decision_drop_auc.md").write_text(md)
    print(f"Guardado docs/analisis/decision_drop_auc.md")


if __name__ == "__main__":
    main()
