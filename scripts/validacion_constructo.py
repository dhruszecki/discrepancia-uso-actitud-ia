"""Validez convergente de la variable a predecir.

La discrepancia se define con la pregunta de actitud (AISent), que pregunta por el
sentimiento hacia la IA en general y no por la confianza. La encuesta tiene además una
pregunta sobre la confianza en la precisión de las herramientas (AIAcc, en las tres
ediciones con la misma escala) y otra sobre si la IA amenaza el propio trabajo
(AIThreat, solo en 2024 y 2025). Estas preguntas no entran al modelo, porque describen
la actitud que se quiere predecir; acá se usan solo para comprobar que la clase con
discrepancia declara menos confianza, como debería si la variable mide lo que dice medir.

Escribe docs/analisis/validacion_constructo.md.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs" / "analisis"

FAV = {"Favorable", "Very favorable"}
NEG = {"Unfavorable", "Very unfavorable"}
GRUPO_ACTITUD = {
    "Very favorable": "Favorable", "Favorable": "Favorable",
    "Indifferent": "Indiferente", "Unsure": "Insegura",
    "Unfavorable": "Desfavorable", "Very unfavorable": "Desfavorable",
}
ORDEN_GRUPOS = ["Favorable", "Indiferente", "Insegura", "Desfavorable"]
CONFIA = {"Highly trust", "Somewhat trust"}
DESCONFIA = {"Highly distrust", "Somewhat distrust"}


def _load_pipeline():
    spec = importlib.util.spec_from_file_location("pipeline", ROOT / "scripts" / "pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cargar_usuarios() -> pd.DataFrame:
    pipe = _load_pipeline()
    with contextlib.redirect_stdout(io.StringIO()):
        df23, df24, df25 = pipe.cargar_y_normalizar()
        for d in (df23, df24, df25):
            d["target"] = pipe.construir_target(d)
        df = pd.concat([df23, df24, df25], ignore_index=True)
        df = pipe.filtrar_poblacion(df, restrict_to_users=True).reset_index(drop=True)
    df["clase"] = df["target"].map({1: "Sin discrepancia", 0: "Con discrepancia"})
    df["actitud"] = df["AISent"].map(GRUPO_ACTITUD)
    return df


def porcentajes(df: pd.DataFrame, por: str, columna: str, conjunto: set) -> pd.DataFrame:
    """Porcentaje de respuestas en `conjunto` entre quienes respondieron `columna`, por año y grupo."""
    resp = df[df[columna].notna()]
    tabla = (resp[columna].isin(conjunto).groupby([resp["survey_year"], resp[por]]).mean() * 100).unstack(0)
    n = resp.groupby(por).size().rename("N")
    return pd.concat([tabla, n], axis=1)


def main():
    df = cargar_usuarios()
    print(f"Usuarios de IA: {len(df):,}")

    confia_clase = porcentajes(df, "clase", "AIAcc", CONFIA)
    desconfia_clase = porcentajes(df, "clase", "AIAcc", DESCONFIA)
    confia_actitud = porcentajes(df, "actitud", "AIAcc", CONFIA).reindex(ORDEN_GRUPOS)
    amenaza_clase = porcentajes(df[df["survey_year"] >= 2024], "clase", "AIThreat", {"Yes"})

    for nombre, t in [("Confía en la precisión", confia_clase), ("Desconfía de la precisión", desconfia_clase),
                      ("Confía, por actitud", confia_actitud), ("Ve una amenaza", amenaza_clase)]:
        print(f"\n{nombre} (%):\n{t.round(1).to_string()}")

    out = [
        "# Validez convergente de la variable a predecir\n\n",
        "Generado por `scripts/validacion_constructo.py`. Usuarios de IA (N = 80.369). Los porcentajes se calculan "
        "entre quienes respondieron cada pregunta. Estas preguntas no entran al modelo.\n\n",
        "## 1. Confía en la precisión de las herramientas de IA (muy o algo), por clase y año (%)\n\n",
        confia_clase.round(1).to_markdown(), "\n\n",
        "## 2. Desconfía de la precisión (muy o algo), por clase y año (%)\n\n",
        desconfia_clase.round(1).to_markdown(), "\n\n",
        "## 3. Confía en la precisión, según la respuesta de actitud (%)\n\n",
        confia_actitud.round(1).to_markdown(), "\n\n",
        "## 4. Cree que la IA amenaza su trabajo, por clase (2024 y 2025, %)\n\n",
        amenaza_clase.round(1).to_markdown(), "\n",
    ]
    (DOCS / "validacion_constructo.md").write_text("".join(out))
    print("\nGuardado docs/analisis/validacion_constructo.md")


if __name__ == "__main__":
    main()
