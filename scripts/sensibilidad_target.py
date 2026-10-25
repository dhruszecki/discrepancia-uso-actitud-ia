"""Sensibilidad de la definición de discrepancia.

La clase "con discrepancia" agrupa los estados indiferente, desfavorable, muy
desfavorable e inseguro de AISent, y la indiferencia es el 66,4 % de esa clase.
El script mide cuánto depende el resultado de esa decisión. Para eso reconstruye
AISent fila a fila junto con la matriz de variables (el parquet de modelado no lo
guarda), deja afuera a quienes no respondieron AISent, ajusta la misma regresión
logística con cada definición alternativa y, para la definición vigente, calcula
la matriz de confusión con dos umbrales.

Escribe docs/analisis/sensibilidad_target.md.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
RANDOM_STATE = 42

FAV = {"Favorable", "Very favorable"}
NEG = {"Unfavorable", "Very unfavorable"}
IND = {"Indifferent"}
UNS = {"Unsure"}

# keep: respuestas de AISent que entran en el análisis.
# disc: respuestas que cuentan como discrepancia; el resto de keep es favorable.
VARIANTES = {
    "vigente (sin NaN)": dict(keep=FAV | NEG | IND | UNS, disc=NEG | IND | UNS),
    "solo negativos activos": dict(keep=FAV | NEG, disc=NEG),
    "sin Unsure": dict(keep=FAV | NEG | IND, disc=NEG | IND),
    "negativos + Unsure": dict(keep=FAV | NEG | UNS, disc=NEG | UNS),
}


def _load_pipeline():
    spec = importlib.util.spec_from_file_location("pipeline", ROOT / "scripts" / "pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def construir_X_y_aisent(pipe):
    """Devuelve la matriz de variables y AISent, fila a fila, para los usuarios de IA."""
    df23, df24, df25 = pipe.cargar_y_normalizar()
    for d in (df23, df24, df25):
        d["target"] = pipe.construir_target(d)
    df = pd.concat([df23, df24, df25], ignore_index=True)
    df = pipe.filtrar_poblacion(df, restrict_to_users=True).reset_index(drop=True)

    aisent = df["AISent"].reset_index(drop=True)
    X_df = pipe.aplicar_pipeline_completo(df.copy()).reset_index(drop=True)

    # AISent se toma del mismo DataFrame que entra al pipeline, así que las filas
    # coinciden; igual se controla que el target se pueda reconstruir desde AISent.
    target_recon = aisent.isin(FAV).astype(int)
    assert (X_df["target"].values == target_recon.values).all(), \
        "AISent no coincide fila a fila con la matriz de variables"
    assert len(X_df) == len(aisent), "Largos distintos"

    X = X_df.drop(columns="target")
    print(f"\nFilas alineadas. X: {X.shape}, AISent no nulo: {aisent.notna().sum():,}")
    print(f"Respuestas de AISent:\n{aisent.value_counts(dropna=False)}\n")
    return X, aisent


def logreg():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced", max_iter=1000, solver="saga",
            n_jobs=-1, random_state=RANDOM_STATE)),
    ])


def evaluar_variante(X, aisent, keep, disc):
    """Prevalencia, ROC-AUC y PR-AUC de la discrepancia con validación cruzada de 5 partes."""
    mask = aisent.isin(keep).values
    Xv = X[mask].reset_index(drop=True)
    av = aisent[mask].reset_index(drop=True)
    y_fav = av.isin(FAV).astype(int).values
    y_disc = av.isin(disc).astype(int).values

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    proba_fav = cross_val_predict(logreg(), Xv, y_fav, cv=cv,
                                  method="predict_proba", n_jobs=1)[:, 1]
    proba_disc = 1 - proba_fav

    return dict(
        n=len(Xv),
        prevalencia_disc=y_disc.mean(),
        roc_auc=roc_auc_score(y_fav, proba_fav),
        pr_auc_disc=average_precision_score(y_disc, proba_disc),
        baseline_pr=y_disc.mean(),
        y_disc=y_disc, proba_disc=proba_disc,
    )


def metricas_accionables(y_disc, proba_disc):
    """Matriz de confusión, precisión, recall y F1 de la discrepancia con el umbral de máximo F1 y con 0,5."""
    prec, rec, thr = precision_recall_curve(y_disc, proba_disc)
    f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0)
    i = int(np.nanargmax(f1[:-1])) if len(thr) else 0
    thr_best = float(thr[i]) if len(thr) else 0.5

    out = {}
    for nombre, t in [(f"máx F1 (t={thr_best:.3f})", thr_best), ("t=0,5", 0.5)]:
        pred = (proba_disc >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_disc, pred).ravel()
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        out[nombre] = dict(tn=tn, fp=fp, fn=fn, tp=tp, precision=p, recall=r, f1=f)
    return out


def main():
    pipe = _load_pipeline()
    X, aisent = construir_X_y_aisent(pipe)

    resultados = {}
    base_auc = None
    for nombre, cfg in VARIANTES.items():
        print(f"Definición: {nombre}")
        r = evaluar_variante(X, aisent, cfg["keep"], cfg["disc"])
        resultados[nombre] = r
        if base_auc is None:
            base_auc = r["roc_auc"]
        delta = (r["roc_auc"] - base_auc) * 100
        print(f"   N={r['n']:,}  prev_disc={r['prevalencia_disc']:.3f}  "
              f"ROC-AUC={r['roc_auc']:.4f} (diferencia {delta:+.2f} puntos)  PR-AUC={r['pr_auc_disc']:.4f} "
              f"(referencia {r['baseline_pr']:.3f})")

    base = resultados["vigente (sin NaN)"]
    acc = metricas_accionables(base["y_disc"], base["proba_disc"])

    L = []
    L.append("# Análisis de sensibilidad de la definición de discrepancia\n")
    L.append("Modelo: regresión logística con variables estandarizadas y pesos por clase, "
             "validación cruzada estratificada de 5 partes. Las probabilidades de cada caso salen "
             "del modelo que no lo vio al entrenar. Quienes no respondieron `AISent` quedan afuera.\n")
    L.append("## Definiciones de la clase con discrepancia\n")
    L.append("| Definición | N | Prevalencia discrepancia | ROC-AUC | Diferencia con la vigente (puntos) | PR-AUC (discrepancia) | Referencia PR |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for nombre, r in resultados.items():
        d = (r["roc_auc"] - base_auc) * 100
        L.append(f"| {nombre} | {r['n']:,} | {r['prevalencia_disc']:.3f} | {r['roc_auc']:.4f} | "
                 f"{d:+.2f} | {r['pr_auc_disc']:.4f} | {r['baseline_pr']:.3f} |")
    L.append("")
    L.append("## Métricas de clasificación con la definición vigente (tres años juntos)\n")
    L.append("La clase positiva es la discrepancia.\n")
    L.append("| Umbral | VP | FP | FN | VN | Precisión | Recall | F1 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for nombre, m in acc.items():
        L.append(f"| {nombre} | {m['tp']} | {m['fp']} | {m['fn']} | {m['tn']} | "
                 f"{m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} |")
    L.append("")
    veredicto = (resultados["solo negativos activos"]["roc_auc"] - base_auc) * 100
    L.append("## Conclusión\n")
    L.append(f"- Dejar afuera a los indiferentes e inseguros (solo negativos activos) cambia el ROC-AUC en "
             f"{veredicto:+.2f} puntos respecto de la definición vigente.")
    L.append("- Si la definición restringida sube el ROC-AUC más de 2 puntos, la clase amplia mezcla dos situaciones "
             "(indiferencia y desconfianza explícita). El trabajo mide ausencia de actitud favorable y no solo desconfianza explícita.")
    L.append("")

    out_path = ROOT / "docs" / "analisis" / "sensibilidad_target.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(L), encoding="utf-8")
    print(f"\nGuardado {out_path.relative_to(ROOT)}")
    print("\n" + "\n".join(L))


if __name__ == "__main__":
    main()
