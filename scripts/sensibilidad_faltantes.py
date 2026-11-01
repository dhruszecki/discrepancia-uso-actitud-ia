"""Sensibilidad a la falta de respuesta.

El pipeline toma la falta de respuesta en las preguntas de opción múltiple como ninguna
opción elegida e imputa la experiencia laboral con la mediana de cada año. Si quienes
no responden también declaran más a menudo una actitud no favorable, variables como
"usa VS Code" o la cantidad de entornos pueden estar midiendo en parte la falta de
respuesta. El script reconstruye, fila a fila, qué preguntas quedaron sin responder y:

1. describe la falta de respuesta por año y la discrepancia según se haya respondido o no;
2. compara el ROC-AUC de la regresión logística con y sin indicadores de falta de respuesta;
3. repite el random forest y SHAP con esos indicadores y solo con quienes respondieron
   la pregunta de entornos, para ver si VS Code y la cantidad de entornos siguen arriba.

Escribe docs/analisis/sensibilidad_faltantes.md y outputs/sensibilidad_faltantes.json.
"""
from __future__ import annotations

import importlib.util
import io
import contextlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nombres_legibles import nombre_legible  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "df_clean_users.parquet"
OUTPUTS = ROOT / "outputs"
DOCS = ROOT / "docs" / "analisis"
RANDOM_STATE = 42

# Pregunta original → nombre del indicador de falta de respuesta.
PREGUNTAS = {
    "NEWCollabToolsHaveWorkedWith": "falta_entornos",
    "LanguageHaveWorkedWith": "falta_lenguajes",
    "DatabaseHaveWorkedWith": "falta_bases",
    "PlatformHaveWorkedWith": "falta_plataformas",
    "WorkExp": "falta_experiencia_laboral",
}
NOMBRES_FALTA = {
    "falta_entornos": "No respondió entornos",
    "falta_lenguajes": "No respondió lenguajes",
    "falta_bases": "No respondió bases de datos",
    "falta_plataformas": "No respondió plataformas",
    "falta_experiencia_laboral": "No respondió experiencia laboral",
}


def _nombre(col: str) -> str:
    return NOMBRES_FALTA.get(col, nombre_legible(col))


def _load_pipeline():
    spec = importlib.util.spec_from_file_location("pipeline", ROOT / "scripts" / "pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cargar() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve el dataset de modelado y, alineados fila a fila, los indicadores de falta de respuesta."""
    pipe = _load_pipeline()
    with contextlib.redirect_stdout(io.StringIO()):
        df23, df24, df25 = pipe.cargar_y_normalizar()
        for d in (df23, df24, df25):
            d["target"] = pipe.construir_target(d)
        crudo = pd.concat([df23, df24, df25], ignore_index=True)
        crudo = pipe.filtrar_poblacion(crudo, restrict_to_users=True).reset_index(drop=True)

    df = pd.read_parquet(DATA_PATH)
    # El pipeline no saca ni reordena filas después del filtro, así que coinciden.
    assert len(df) == len(crudo), "Largos distintos"
    assert (df["target"].values == crudo["target"].values).all(), "El target no coincide fila a fila"
    assert (df["survey_year"].values == crudo["survey_year"].values).all(), "El año no coincide fila a fila"

    faltas = pd.DataFrame({ind: crudo[col].isna().astype(float).values for col, ind in PREGUNTAS.items()})
    faltas["AISent"] = crudo["AISent"].values
    return df, faltas


def describir(df: pd.DataFrame, faltas: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Falta de respuesta por año y discrepancia e indiferencia según se haya respondido entornos."""
    por_anio = faltas[list(NOMBRES_FALTA)].groupby(df["survey_year"]).mean().T * 100
    por_anio.index = [NOMBRES_FALTA[i] for i in por_anio.index]

    filas = []
    for year in [2023, 2024, 2025]:
        for falta in [0.0, 1.0]:
            m = (df["survey_year"] == year).values & (faltas["falta_entornos"] == falta).values
            filas.append({
                "Año": year,
                "Entornos": "No respondió" if falta else "Respondió",
                "N": int(m.sum()),
                "Discrepancia %": 100 * (1 - df.loc[m, "target"]).mean(),
                "Indiferente %": 100 * (faltas.loc[m, "AISent"] == "Indifferent").mean(),
            })
    return por_anio, pd.DataFrame(filas)


def auc_logreg(X: pd.DataFrame, y: pd.Series) -> float:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, solver="saga",
                                   n_jobs=-1, random_state=RANDOM_STATE)),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    prob = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, prob))


def shap_rf(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """SHAP medio absoluto del mismo random forest de scripts/modelado.py, sobre 1.000 filas."""
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", max_features="sqrt",
                                min_samples_leaf=20, n_jobs=-1, random_state=RANDOM_STATE)
    rf.fit(X, y)
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X), size=min(1000, len(X)), replace=False)
    valores = shap.TreeExplainer(rf).shap_values(X.iloc[idx])
    if isinstance(valores, list):
        valores = valores[1]
    elif valores.ndim == 3:
        valores = valores[:, :, 1]
    return pd.Series(np.abs(valores).mean(axis=0), index=X.columns).sort_values(ascending=False)


def puesto(ranking: pd.Series, col: str) -> int:
    return int(list(ranking.index).index(col)) + 1


def main():
    df, faltas = cargar()
    y = df["target"]
    X = df.drop(columns="target")
    X_ind = pd.concat([X, faltas[list(NOMBRES_FALTA)]], axis=1)
    respondio = (faltas["falta_entornos"] == 0).values

    por_anio, por_entornos = describir(df, faltas)
    print(por_anio.round(1).to_string())
    print(por_entornos.round(1).to_string())

    print("\nROC-AUC de la regresión logística")
    aucs = {
        "Base (98 variables)": auc_logreg(X, y),
        "Base + indicadores de falta de respuesta": auc_logreg(X_ind, y),
        "Solo quienes respondieron entornos": auc_logreg(X[respondio], y[respondio]),
    }
    for k, v in aucs.items():
        print(f"  {k}: {v:.4f}")

    print("\nSHAP del random forest")
    rankings = {
        "Base": shap_rf(X, y),
        "Con indicadores": shap_rf(X_ind, y),
        "Solo quienes respondieron entornos": shap_rf(X[respondio].reset_index(drop=True),
                                                     y[respondio].reset_index(drop=True)),
    }
    vigiladas = ["env_Visual_Studio_Code", "env_count", "WorkExp", "survey_year"]
    filas = []
    for nombre, r in rankings.items():
        print(f"  {nombre}: " + ", ".join(f"{c} ({v:.4f})" for c, v in r.head(8).items()))
        for col in vigiladas:
            filas.append({"Variante": nombre, "Variable": nombre_legible(col),
                          "SHAP": r[col], "Puesto": puesto(r, col)})
    tabla_vigiladas = pd.DataFrame(filas).pivot(index="Variable", columns="Variante", values=["SHAP", "Puesto"])

    out = ["# Sensibilidad a la falta de respuesta\n\n",
           "Generado por `scripts/sensibilidad_faltantes.py` sobre `data/df_clean_users.parquet`.\n\n",
           "En las preguntas de opción múltiple la falta de respuesta cuenta como ninguna opción elegida, "
           "y la experiencia laboral faltante se imputa con la mediana de cada año. Los indicadores marcan "
           "a quien no respondió cada pregunta en la encuesta original.\n\n",
           "## 1. Falta de respuesta por año (%)\n\n", por_anio.round(1).to_markdown(), "\n\n",
           "## 2. Discrepancia e indiferencia según se haya respondido la pregunta de entornos\n\n",
           por_entornos.round(1).to_markdown(index=False), "\n\n",
           "## 3. ROC-AUC de la regresión logística (validación cruzada de 5 partes)\n\n"]
    out += [f"- {k}: {v:.4f}\n" for k, v in aucs.items()]
    out.append("\n## 4. SHAP del random forest: 10 variables principales\n\n")
    for nombre, r in rankings.items():
        out.append(f"**{nombre}:** " + ", ".join(f"{_nombre(c)} ({v:.4f})" for c, v in r.head(10).items()) + "\n\n")
    out.append("## 5. Variables vigiladas: SHAP medio absoluto y puesto\n\n")
    out.append(tabla_vigiladas.round(4).to_markdown())
    out.append("\n")
    (DOCS / "sensibilidad_faltantes.md").write_text("".join(out))
    print("Guardado docs/analisis/sensibilidad_faltantes.md")

    (OUTPUTS / "sensibilidad_faltantes.json").write_text(json.dumps({
        "falta_por_anio": por_anio.round(2).to_dict(),
        "por_entornos": por_entornos.round(2).to_dict(orient="records"),
        "auc": aucs,
        "top10": {k: r.head(10).round(4).to_dict() for k, r in rankings.items()},
        "vigiladas": {k: {c: {"shap": round(float(r[c]), 4), "puesto": puesto(r, c)} for c in vigiladas}
                      for k, r in rankings.items()},
    }, indent=2, ensure_ascii=False))
    print("Guardado outputs/sensibilidad_faltantes.json")


if __name__ == "__main__":
    main()
