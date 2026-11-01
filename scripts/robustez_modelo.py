"""Estabilidad del SHAP y calibración del random forest.

El SHAP global de scripts/modelado.py se calcula sobre una sola muestra de 1.000 filas,
y varias variables quedan casi empatadas. Este script repite el cálculo sobre cinco
muestras independientes de 1.000 filas y sobre una de 5.000, y mide cuánto cambia el
orden de las variables principales.

Además, el random forest se entrena con pesos por clase, así que sus probabilidades no
estiman la probabilidad real de discrepancia. Se compara la probabilidad predicha fuera
de muestra con la tasa observada por deciles.

Escribe docs/analisis/robustez_modelo.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nombres_legibles import nombre_legible  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "df_clean_users.parquet"
DOCS = ROOT / "docs" / "analisis"
RANDOM_STATE = 42
N_TOP = 10


def random_forest() -> RandomForestClassifier:
    """El mismo random forest de scripts/modelado.py."""
    return RandomForestClassifier(n_estimators=100, class_weight="balanced", max_features="sqrt",
                                  min_samples_leaf=20, n_jobs=-1, random_state=RANDOM_STATE)


def shap_medio(explainer, X: pd.DataFrame) -> pd.Series:
    valores = explainer.shap_values(X)
    if isinstance(valores, list):
        valores = valores[1]
    elif valores.ndim == 3:
        valores = valores[:, :, 1]
    return pd.Series(np.abs(valores).mean(axis=0), index=X.columns)


def estabilidad_shap(rf, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Puesto de cada variable en cinco muestras de 1.000 filas y SHAP sobre 5.000 filas."""
    explainer = shap.TreeExplainer(rf)
    rng = np.random.default_rng(RANDOM_STATE)
    # La primera muestra es la misma de scripts/modelado.py.
    muestras = [rng.choice(len(X), size=1000, replace=False)]
    resto = np.setdiff1d(np.arange(len(X)), muestras[0])
    rng2 = np.random.default_rng(RANDOM_STATE + 1)
    otras = rng2.choice(resto, size=4 * 1000 + 5000, replace=False)
    muestras += [otras[i * 1000:(i + 1) * 1000] for i in range(4)]
    grande = otras[4000:]

    puestos = {}
    for i, idx in enumerate(muestras, start=1):
        s = shap_medio(explainer, X.iloc[idx]).sort_values(ascending=False)
        puestos[f"Muestra {i}"] = pd.Series(range(1, len(s) + 1), index=s.index)
        print(f"  Muestra {i}: " + ", ".join(s.head(5).index))
    puestos = pd.DataFrame(puestos)
    shap_5000 = shap_medio(explainer, X.iloc[grande]).sort_values(ascending=False)

    top = puestos["Muestra 1"].nsmallest(N_TOP).index
    tabla = puestos.loc[top].copy()
    tabla["Puesto con 5.000"] = pd.Series(range(1, len(shap_5000) + 1), index=shap_5000.index).loc[top]
    tabla["Mín."] = tabla.min(axis=1)
    tabla["Máx."] = tabla.max(axis=1)
    tabla.index = [nombre_legible(c) for c in tabla.index]
    return tabla, shap_5000


def calibracion(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, dict]:
    """Probabilidad de discrepancia fuera de muestra contra la tasa observada, por deciles."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    prob_disc = cross_val_predict(random_forest(), X, y, cv=cv, method="predict_proba")[:, 0]
    disc = 1 - y.to_numpy()
    deciles = pd.qcut(prob_disc, 10, labels=False, duplicates="drop")
    tabla = pd.DataFrame({"decil": deciles + 1, "predicha": prob_disc, "observada": disc}) \
        .groupby("decil").agg(predicha=("predicha", "mean"), observada=("observada", "mean"), N=("observada", "size"))
    tabla[["predicha", "observada"]] *= 100
    resumen = {
        "media_predicha": float(prob_disc.mean() * 100),
        "tasa_observada": float(disc.mean() * 100),
        "brier": float(brier_score_loss(disc, prob_disc)),
        "brier_referencia": float(brier_score_loss(disc, np.full_like(prob_disc, disc.mean()))),
    }
    return tabla, resumen


def main():
    df = pd.read_parquet(DATA_PATH)
    y = df["target"]
    X = df.drop(columns="target")

    print("Estabilidad del SHAP")
    rf = random_forest().fit(X, y)
    tabla_shap, shap_5000 = estabilidad_shap(rf, X)
    print(tabla_shap.to_string())

    print("\nCalibración (random forest, fuera de muestra)")
    tabla_cal, resumen = calibracion(X, y)
    print(tabla_cal.round(1).to_string())
    print(resumen)

    out = [
        "# Robustez del modelo: estabilidad del SHAP y calibración\n\n",
        "Generado por `scripts/robustez_modelo.py` con el random forest de `scripts/modelado.py`.\n\n",
        f"## 1. Puesto de las {N_TOP} variables principales en distintas muestras\n\n",
        "La muestra 1 es la de `scripts/modelado.py`. Las muestras 2 a 5 son independientes, de 1.000 filas, "
        "y la última columna usa 5.000 filas distintas de las anteriores.\n\n",
        tabla_shap.to_markdown(), "\n\n",
        "**SHAP medio absoluto con 5.000 filas (10 principales):** ",
        ", ".join(f"{nombre_legible(c)} ({v:.4f})" for c, v in shap_5000.head(10).items()), "\n\n",
        "## 2. Calibración: probabilidad de discrepancia predicha y tasa observada, por deciles (%)\n\n",
        tabla_cal.round(1).to_markdown(), "\n\n",
        f"Probabilidad predicha media: {resumen['media_predicha']:.1f} %. Tasa observada: {resumen['tasa_observada']:.1f} %. "
        f"Brier: {resumen['brier']:.4f} (predecir siempre la tasa observada da {resumen['brier_referencia']:.4f}).\n\n",
        "Con pesos por clase, el modelo sobreestima la probabilidad de discrepancia: sirve para ordenar a las "
        "personas, no para leer sus probabilidades como frecuencias.\n",
    ]
    (DOCS / "robustez_modelo.md").write_text("".join(out))
    print("\nGuardado docs/analisis/robustez_modelo.md")


if __name__ == "__main__":
    main()
