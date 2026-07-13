"""Pipeline de construcción de datasets — generación de parquets vigente E3.

Funciones reusables que importan notebooks 01 y 02. Genera dos artefactos:
1) `data/df_clean.parquet` (155k, full) — para análisis comparativos y H5 macro.
2) `data/df_clean_users.parquet` (80k, restricted) — dataset E3 vigente.

Decisión adoptada (provisional, confirmable jueves 28-5):
- Label "Sin discrepancia" / "Con discrepancia uso-actitud".
- Restricción a usuarios justificada por Hoff & Bashir (2015) "learned trust" + Wang R. (2024).

Mitigación incorporada: recodificación de DevType 2025 (AI/ML engineer + Data engineer
→ Data scientist or ML specialist) para preservar comparabilidad histórica.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# ============================================================
# CONFIG — ajustable sin tocar el resto del código
# ============================================================
TARGET_LABELS = {
    1: "Sin discrepancia",
    0: "Con discrepancia uso-actitud",
}
DEVTYPE_2025_REMAP = {
    # Recodificar categorías nuevas de 2025 a la categoría histórica DS/ML
    "AI/ML engineer": "Data scientist or machine learning specialist",
    "Data engineer": "Data scientist or machine learning specialist",
}
USA_IA = ["Yes"]
ACTITUD_POSITIVA = ["Favorable", "Very favorable"]
DEV_LABEL = "I am a developer by profession"
# ============================================================


def output_path(restrict_to_users: bool) -> Path:
    """Path del parquet de salida según modo."""
    return DATA / ("df_clean_users.parquet" if restrict_to_users else "df_clean.parquet")


def cargar_y_normalizar():
    """Réplica de notebook 02 cell 3: carga + normalización de schemas."""
    df23 = pd.read_csv(DATA / "2023" / "survey_results_public.csv", low_memory=False)
    df24 = pd.read_csv(DATA / "2024" / "survey_results_public.csv", low_memory=False)
    df25 = pd.read_csv(DATA / "2025" / "survey_results_public.csv", low_memory=False)

    df23["survey_year"] = 2023
    df24["survey_year"] = 2024
    df25["survey_year"] = 2025

    print(f"Cargados: 2023={df23.shape[0]:,}  2024={df24.shape[0]:,}  2025={df25.shape[0]:,}")

    # AISelect 2025: aplanar Yes daily/weekly/monthly → Yes
    aiselect_yes_2025 = [
        "Yes, I use AI tools daily",
        "Yes, I use AI tools weekly",
        "Yes, I use AI tools monthly or infrequently",
    ]
    df25["AISelect"] = df25["AISelect"].apply(
        lambda v: "Yes" if v in aiselect_yes_2025 else v
    )

    # YearsCodePro fallback en 2025
    if "YearsCodePro" not in df25.columns and "YearsCode" in df25.columns:
        df25["YearsCodePro"] = df25["YearsCode"]

    # 2023: swap AIAcc <-> AIBen
    df23 = df23.rename(columns={"AIAcc": "AIBen", "AIBen": "AIAcc"})

    # 2025: DevEnvsHaveWorkedWith → NEWCollabToolsHaveWorkedWith
    if "DevEnvsHaveWorkedWith" in df25.columns and "NEWCollabToolsHaveWorkedWith" not in df25.columns:
        df25 = df25.rename(columns={"DevEnvsHaveWorkedWith": "NEWCollabToolsHaveWorkedWith"})

    # ===== Mitigación nueva: DevType 2025 categorías → mapeo histórico =====
    if "DevType" in df25.columns:
        df25["DevType"] = df25["DevType"].apply(
            lambda v: _remap_devtype(v, DEVTYPE_2025_REMAP) if isinstance(v, str) else v
        )
        n_remapped = df25["DevType"].astype(str).str.contains(
            "Data scientist or machine learning specialist", regex=False
        ).sum()
        print(f"DevType 2025 recodificado: {n_remapped} respondientes en DS/ML (unión nueva)")

    print("✅ Schemas normalizados")
    return df23, df24, df25


def _remap_devtype(v: str, mapping: dict) -> str:
    """DevType puede ser multi-valor separado por ';'. Aplicar mapeo a cada token."""
    tokens = [t.strip() for t in v.split(";")]
    mapped = [mapping.get(t, t) for t in tokens]
    # Eliminar duplicados preservando orden (si dos tokens mapean al mismo destino)
    seen = set()
    result = []
    for t in mapped:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return ";".join(result)


def construir_target(df: pd.DataFrame) -> pd.Series:
    """Target binario: 1 si usa IA y actitud favorable, 0 en cualquier otro caso."""
    return (df["AISelect"].isin(USA_IA) & df["AISent"].isin(ACTITUD_POSITIVA)).astype(int)


def filtrar_poblacion(df: pd.DataFrame, restrict_to_users: bool = True) -> pd.DataFrame:
    n_antes = len(df)
    df = df[df["MainBranch"] == DEV_LABEL].copy()
    print(f"Tras filtro MainBranch: {len(df):,} (de {n_antes:,})")

    if restrict_to_users:
        n_antes_user = len(df)
        df = df[df["AISelect"] == "Yes"].copy()
        print(f"Tras filtro AISelect=Yes (usuarios activos): {len(df):,} (de {n_antes_user:,})")

    return df


def reportar_target(df: pd.DataFrame) -> None:
    print("\n=== Distribución del target ===")
    print(f"Total: {len(df):,} filas")
    print(f"Target=1 ({TARGET_LABELS[1]}): {df['target'].sum():,} ({df['target'].mean():.1%})")
    print(f"Target=0 ({TARGET_LABELS[0]}): {(1-df['target']).sum():,} ({(1-df['target']).mean():.1%})")
    print("\nPor año:")
    print(df.groupby("survey_year")["target"].agg(["count", "sum", "mean"]).round(3))


def aplicar_pipeline_completo(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce las secciones 2-10 de notebook 02_preprocesamiento."""
    # === Sección 2: selección de columnas ===
    COLS_KEEP = [
        "survey_year", "target",
        "DevType", "YearsCode", "YearsCodePro", "WorkExp", "EdLevel",
        "OrgSize", "Employment", "RemoteWork", "ICorPM", "Industry", "Country",
        "JobSat",
        "LanguageHaveWorkedWith",
        "DatabaseHaveWorkedWith",
        "PlatformHaveWorkedWith",
        "NEWCollabToolsHaveWorkedWith",
    ]
    cols_faltantes = [c for c in COLS_KEEP if c not in df.columns]
    if cols_faltantes:
        for col in cols_faltantes:
            df[col] = np.nan
        print(f"Cols faltantes rellenadas con NaN: {cols_faltantes}")

    df = df[COLS_KEEP].copy()
    df = df.rename(columns={
        "LanguageHaveWorkedWith": "Language",
        "DatabaseHaveWorkedWith": "Database",
        "PlatformHaveWorkedWith": "Platform",
        "NEWCollabToolsHaveWorkedWith": "DevEnvs",
    })
    print(f"Tras Sección 2: {df.shape}")

    # === Sección 3: descarte por nulos ===
    UMBRAL_NULOS = 40.0
    null_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    cols_a_descartar = null_pct[null_pct > UMBRAL_NULOS].index.tolist()
    if cols_a_descartar:
        print(f"Descartadas por >{UMBRAL_NULOS}% nulos: {cols_a_descartar}")
        df = df.drop(columns=cols_a_descartar)

    # === Sección 4: imputación ===
    multi_val_cols = [c for c in ["Language", "Database", "Platform", "DevEnvs"] if c in df.columns]
    for c in multi_val_cols:
        df[c] = df[c].fillna("")

    if "DevType" in df.columns:
        df["DevType"] = df["DevType"].fillna("Other")

    nominales = [c for c in ["Employment", "RemoteWork", "ICorPM", "Industry", "Country"] if c in df.columns]
    for c in nominales:
        moda = df[c].mode().iloc[0]
        df[c] = df[c].fillna(moda)

    if "JobSat" in df.columns:
        med_2425 = df.loc[df["survey_year"].isin([2024, 2025]), "JobSat"].median()
        df["JobSat"] = df["JobSat"].fillna(med_2425)

    if "WorkExp" in df.columns:
        df["WorkExp"] = df.groupby("survey_year")["WorkExp"].transform(
            lambda s: s.fillna(s.median())
        )

    print(f"Tras imputación: {df.isnull().sum().sum()} nulos residuales")

    # === Sección 5: encoding ordinal ===
    def years_to_num(v):
        if pd.isna(v):
            return np.nan
        s = str(v)
        if "Less than" in s:
            return 0.5
        if "More than" in s:
            return 51.0
        try:
            return float(s)
        except ValueError:
            return np.nan

    for col in ["YearsCode", "YearsCodePro"]:
        if col in df.columns:
            df[col] = df[col].apply(years_to_num)

    EDLEVEL_MAP = {
        "Something else": 0,
        "Primary/elementary school": 1,
        "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)": 2,
        "Some college/university study without earning a degree": 3,
        "Associate degree (A.A., A.S., etc.)": 4,
        "Bachelor's degree (B.A., B.S., B.Eng., etc.)": 5,
        "Master's degree (M.A., M.S., M.Eng., MBA, etc.)": 6,
        "Professional degree (JD, MD, Ph.D, Ed.D, etc.)": 7,
    }
    if "EdLevel" in df.columns:
        df["EdLevel"] = df["EdLevel"].map(EDLEVEL_MAP).fillna(0)

    ORGSIZE_MAP = {
        "Just me - I am a freelancer, sole proprietor, etc.": 1,
        "2 to 9 employees": 2,
        "10 to 19 employees": 3,
        "20 to 99 employees": 4,
        "100 to 499 employees": 5,
        "500 to 999 employees": 6,
        "1,000 to 4,999 employees": 7,
        "5,000 to 9,999 employees": 8,
        "10,000 or more employees": 9,
        "I don't know": 0,
    }
    if "OrgSize" in df.columns:
        df["OrgSize"] = df["OrgSize"].map(ORGSIZE_MAP).fillna(0)

    # Imputar nulos residuales en numéricas por mediana
    for col in ["YearsCode", "YearsCodePro", "EdLevel", "OrgSize", "WorkExp"]:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # === Sección 6: binarización multi-valor ===
    def binarize_multival(df, col, top_n, prefix):
        tokens_all = []
        for v in df[col]:
            if isinstance(v, str) and v:
                tokens_all.extend(t.strip() for t in v.split(";"))
        top = [t for t, _ in Counter(tokens_all).most_common(top_n)]
        for t in top:
            colname = f"{prefix}{t.replace(' ', '_').replace('/', '_').replace('-', '_').replace(',', '').replace('+', 'p')}"
            df[colname] = df[col].apply(lambda v: 1 if isinstance(v, str) and t in v.split(";") else 0)
        df[f"{prefix}count"] = df[col].apply(lambda v: len(v.split(";")) if isinstance(v, str) and v else 0)
        return df

    config_multival = {
        "Language": (20, "lang_"),
        "Database": (10, "db_"),
        "Platform": (15, "plat_"),
        "DevEnvs": (15, "env_"),
    }
    for col, (top_n, prefix) in config_multival.items():
        if col in df.columns:
            df = binarize_multival(df, col, top_n, prefix)
            df = df.drop(columns=[col])

    # === Sección 7: Country → Region ===
    COUNTRY_REGION = {
        "United States of America": "NA", "Canada": "NA",
        "United Kingdom of Great Britain and Northern Ireland": "EU",
        "Germany": "EU", "France": "EU", "Netherlands": "EU", "Spain": "EU",
        "Italy": "EU", "Poland": "EU", "Sweden": "EU", "Switzerland": "EU",
        "Norway": "EU", "Denmark": "EU", "Finland": "EU", "Belgium": "EU",
        "Austria": "EU", "Ireland": "EU", "Portugal": "EU", "Czech Republic": "EU",
        "Romania": "EU", "Hungary": "EU", "Greece": "EU",
        "India": "ASIA", "China": "ASIA", "Japan": "ASIA", "South Korea": "ASIA",
        "Indonesia": "ASIA", "Philippines": "ASIA", "Vietnam": "ASIA",
        "Pakistan": "ASIA", "Bangladesh": "ASIA", "Singapore": "ASIA",
        "Russian Federation": "EU", "Ukraine": "EU", "Turkey": "EU",
        "Brazil": "LATAM", "Mexico": "LATAM", "Argentina": "LATAM",
        "Chile": "LATAM", "Colombia": "LATAM", "Peru": "LATAM",
        "Australia": "OCE", "New Zealand": "OCE",
        "Israel": "EU", "South Africa": "OTHER", "Nigeria": "OTHER",
        "Egypt": "OTHER",
    }
    if "Country" in df.columns:
        df["Region"] = df["Country"].map(COUNTRY_REGION).fillna("OTHER")
        df = df.drop(columns=["Country"])

    # === Sección 8: OHE de nominales de baja cardinalidad ===
    def simplify_employment(val):
        if not isinstance(val, str):
            return "Other"
        if "Employed, full-time" in val:
            return "FullTime"
        if "Employed, part-time" in val:
            return "PartTime"
        if "Independent contractor" in val or "freelancer" in val:
            return "Freelance"
        return "Other"

    if "Employment" in df.columns:
        df["Employment"] = df["Employment"].apply(simplify_employment)

    if "DevType" in df.columns:
        # Tomar el primer DevType si es multi-valor
        df["DevType"] = df["DevType"].apply(lambda v: v.split(";")[0].strip() if isinstance(v, str) else "Other")
        # Top-12 DevTypes; resto → Other
        top_devtypes = df["DevType"].value_counts().head(12).index.tolist()
        df["DevType"] = df["DevType"].apply(lambda v: v if v in top_devtypes else "Other")

    nominals_for_ohe = [c for c in ["Employment", "RemoteWork", "ICorPM", "Region", "DevType", "Industry"] if c in df.columns]
    for col in nominals_for_ohe:
        dummies = pd.get_dummies(df[col], prefix=col, dummy_na=False).astype(int)
        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

    return df


def construir_dataset(restrict_to_users: bool = True, save: bool = True) -> pd.DataFrame:
    """Construye un dataset limpio según el modo.

    Args:
        restrict_to_users: si True, filtra a AISelect=Yes → ~80k filas (E3 vigente).
                           Si False, mantiene todos los devs profesionales → ~155k (E2 / análisis comparativos).
        save: si True, persiste el parquet en data/.

    Returns:
        DataFrame limpio, listo para EDA y modelado.
    """
    df23, df24, df25 = cargar_y_normalizar()

    # Construir target en cada uno antes de concatenar
    df23["target"] = construir_target(df23)
    df24["target"] = construir_target(df24)
    df25["target"] = construir_target(df25)

    df = pd.concat([df23, df24, df25], ignore_index=True)
    df = filtrar_poblacion(df, restrict_to_users=restrict_to_users)
    reportar_target(df)

    df = aplicar_pipeline_completo(df)

    # Verificar tipos numéricos críticos
    numeric_expected = ["YearsCode", "YearsCodePro", "WorkExp", "EdLevel", "OrgSize", "target", "survey_year"]
    for col in numeric_expected:
        if col in df.columns:
            assert pd.api.types.is_numeric_dtype(df[col]), f"{col} no es numérico"

    if save:
        path = output_path(restrict_to_users)
        df.to_parquet(path, index=False)
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"\n✅ Guardado: {path}  ({size_mb:.1f} MB, {df.shape[0]:,} × {df.shape[1]} cols)")

    return df


def main():
    """Genera ambos parquets (full + users) en una corrida."""
    print("=" * 70)
    print("CONSTRUCCIÓN DE DATASETS — pipeline completo")
    print("=" * 70)
    print(f"\nCONFIG:")
    print(f"  TARGET_LABELS = {TARGET_LABELS}")
    print(f"  DEVTYPE_2025_REMAP = {DEVTYPE_2025_REMAP}")

    print("\n" + "─" * 70)
    print("DATASET 1: FULL (155k, sin restricción de población)")
    print("─" * 70)
    construir_dataset(restrict_to_users=False, save=True)

    print("\n" + "─" * 70)
    print("DATASET 2: USUARIOS (~80k, filtro AISelect=Yes)")
    print("─" * 70)
    construir_dataset(restrict_to_users=True, save=True)

    print("\n" + "=" * 70)
    print("LISTO — ambos parquets generados")
    print("=" * 70)


if __name__ == "__main__":
    main()
