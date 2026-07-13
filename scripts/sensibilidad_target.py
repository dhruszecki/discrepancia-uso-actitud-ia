"""Análisis de sensibilidad de la operacionalización del target (E4).

Responde a la corrección #1 de la revisión independiente de E4: el colapso de
cuatro estados de AISent en `target=0` ("con discrepancia") está dominado por la
indiferencia (66,4 %). Este script:

  1. Re-deriva `AISent` alineado fila-a-fila con la matriz de features (AISent se
     excluye del parquet de modelado por leakage). La alineación es por construcción:
     se corre `aplicar_pipeline_completo` sobre el MISMO intermedio filtrado del que
     se captura AISent.
  2. Excluye las no-respuestas de AISent del target (no las etiqueta como discrepancia).
  3. Corre definiciones alternativas de `target=0` y reporta prevalencia, ROC-AUC y
     PR-AUC (LogReg, CV estratificada k=5, OOF) para cada una.
  4. Para la definición vigente reporta métricas accionables del modelo pooled:
     matriz de confusión + precision/recall/F1 sobre la clase de discrepancia a un
     umbral justificado (máximo F1) y a 0,5.

Uso:  uv run python scripts/sensibilidad_target.py
Salida: tabla por consola + resultados/sensibilidad_target.md
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

# Categorías de AISent
FAV = {"Favorable", "Very favorable"}
NEG = {"Unfavorable", "Very unfavorable"}
IND = {"Indifferent"}
UNS = {"Unsure"}

# Definiciones alternativas del target.
# 'keep' = qué estados de AISent se conservan; 'disc' = cuáles cuentan como discrepancia.
# y_fav = 1 si AISent ∈ FAV; discrepancia (clase de interés) = complemento dentro de 'keep'.
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
    """Devuelve (X, aisent) alineados por construcción sobre la población de usuarios."""
    df23, df24, df25 = pipe.cargar_y_normalizar()
    for d in (df23, df24, df25):
        d["target"] = pipe.construir_target(d)
    df = pd.concat([df23, df24, df25], ignore_index=True)
    df = pipe.filtrar_poblacion(df, restrict_to_users=True).reset_index(drop=True)

    aisent = df["AISent"].reset_index(drop=True)
    X_df = pipe.aplicar_pipeline_completo(df.copy()).reset_index(drop=True)

    # Sanity: el target del pipeline debe reconstruirse desde AISent capturado.
    target_recon = aisent.isin(FAV).astype(int)
    assert (X_df["target"].values == target_recon.values).all(), \
        "Desalineación AISent ↔ matriz de features"
    assert len(X_df) == len(aisent), "Largos distintos"

    X = X_df.drop(columns="target")
    print(f"\n✅ Alineación verificada — X: {X.shape}, AISent no nulo: {aisent.notna().sum():,}")
    print(f"AISent value_counts:\n{aisent.value_counts(dropna=False)}\n")
    return X, aisent


def logreg():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced", max_iter=1000, solver="saga",
            n_jobs=-1, random_state=RANDOM_STATE)),
    ])


def evaluar_variante(X, aisent, keep, disc):
    """OOF: prevalencia de discrepancia, ROC-AUC, PR-AUC (discrepancia como positivo)."""
    mask = aisent.isin(keep).values
    Xv = X[mask].reset_index(drop=True)
    av = aisent[mask].reset_index(drop=True)
    y_fav = av.isin(FAV).astype(int).values            # 1 = sin discrepancia (favorable)
    y_disc = av.isin(disc).astype(int).values          # 1 = discrepancia (evento de interés)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    proba_fav = cross_val_predict(logreg(), Xv, y_fav, cv=cv,
                                  method="predict_proba", n_jobs=1)[:, 1]
    proba_disc = 1 - proba_fav

    return dict(
        n=len(Xv),
        prevalencia_disc=y_disc.mean(),
        roc_auc=roc_auc_score(y_fav, proba_fav),       # invariante al flip de etiqueta
        pr_auc_disc=average_precision_score(y_disc, proba_disc),
        baseline_pr=y_disc.mean(),
        y_disc=y_disc, proba_disc=proba_disc,
    )


def metricas_accionables(y_disc, proba_disc):
    """Matriz de confusión + precision/recall/F1 (discrepancia) a umbral de máx F1 y a 0,5."""
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
        print(f"── Evaluando: {nombre} …")
        r = evaluar_variante(X, aisent, cfg["keep"], cfg["disc"])
        resultados[nombre] = r
        if base_auc is None:
            base_auc = r["roc_auc"]
        delta = (r["roc_auc"] - base_auc) * 100
        print(f"   N={r['n']:,}  prev_disc={r['prevalencia_disc']:.3f}  "
              f"ROC-AUC={r['roc_auc']:.4f} (Δ={delta:+.2f}pp)  PR-AUC={r['pr_auc_disc']:.4f} "
              f"(base {r['baseline_pr']:.3f})")

    base = resultados["vigente (sin NaN)"]
    acc = metricas_accionables(base["y_disc"], base["proba_disc"])

    # ── Reporte markdown ────────────────────────────────────────────────
    L = []
    L.append("# Análisis de sensibilidad del target — E4\n")
    L.append("Respuesta a la corrección #1 de la revisión independiente. "
             "Modelo: Logistic Regression (StandardScaler + balanced, saga), "
             "CV estratificada k=5, probabilidades out-of-fold. "
             "Las no-respuestas de `AISent` se excluyen del target (no se etiquetan).\n")
    L.append("## Definiciones de `target=0` (discrepancia)\n")
    L.append("| Definición | N | Prevalencia discrepancia | ROC-AUC | Δ vs vigente | PR-AUC (disc) | Baseline PR |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for nombre, r in resultados.items():
        d = (r["roc_auc"] - base_auc) * 100
        L.append(f"| {nombre} | {r['n']:,} | {r['prevalencia_disc']:.3f} | {r['roc_auc']:.4f} | "
                 f"{d:+.2f} pp | {r['pr_auc_disc']:.4f} | {r['baseline_pr']:.3f} |")
    L.append("")
    L.append("## Métricas accionables — modelo pooled, definición vigente\n")
    L.append("Clase de interés = discrepancia (positivo).\n")
    L.append("| Umbral | TP | FP | FN | TN | Precision | Recall | F1 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for nombre, m in acc.items():
        L.append(f"| {nombre} | {m['tp']} | {m['fp']} | {m['fn']} | {m['tn']} | "
                 f"{m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} |")
    L.append("")
    veredicto = (resultados["solo negativos activos"]["roc_auc"] - base_auc) * 100
    L.append("## Veredicto\n")
    L.append(f"- Excluir indiferentes e inseguros (solo negativos activos) cambia el ROC-AUC en "
             f"**{veredicto:+.2f} pp** respecto a la definición vigente.")
    L.append(f"- Criterio de la revisión: si sube **>2 pp**, la operacionalización mezcla constructos "
             f"y conviene reconsiderarla; si no, el colapso se sostiene empíricamente "
             f"(el trabajo mide *ausencia de actitud favorable*, no solo *desconfianza activa*).")
    L.append("")

    out_path = ROOT / "resultados" / "sensibilidad_target.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(L), encoding="utf-8")
    print(f"\n✅ Reporte guardado en {out_path}")
    print("\n" + "\n".join(L))


if __name__ == "__main__":
    main()
