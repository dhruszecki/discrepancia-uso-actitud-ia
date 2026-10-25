"""Construcción de los datasets a partir de la encuesta de Stack Overflow 2023-2025.

Genera data/df_clean.parquet (todos los desarrolladores profesionales, unas 155 mil filas)
y data/df_clean_users.parquet (solo quienes usan IA, unas 80 mil filas). El segundo es el
conjunto de análisis; el primero se usa para describir la brecha entre uso y actitud en
toda la población.

Se trabaja solo con usuarios porque la confianza que se aprende con el uso (Hoff y Bashir,
2015) solo puede medirse en quienes ya probaron la herramienta.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

TARGET_LABELS = {
    1: "Sin discrepancia",
    0: "Con discrepancia uso-actitud",
}
# Los roles cambian de nombre entre ediciones; se unifican para que ningún rol exista en
# un solo año (si no, el rol funciona como marcador del año). El primer nombre es de 2023 y
# el resto de 2025; "Architect, software or solutions" solo existe en 2025.
DEVTYPE_REMAP = {
    "Engineer, data": "Data engineer",
    "Data scientist": "Data scientist or machine learning specialist",
    "AI/ML engineer": "Data scientist or machine learning specialist",
    "DevOps engineer or professional": "DevOps specialist",
    "Database administrator or engineer": "Database administrator",
    "Architect, software or solutions": "Other",
}

# 2023 usa otra lista de industrias; se agrupan en categorías comparables entre años.
# Las que no figuran (Government, Legal, Wholesale, etc.) quedan como Other.
INDUSTRY_GROUPS = {
    "Information Services, IT, Software Development, or other Technology": "Technology",
    "Software Development": "Technology",
    "Internet, Telecomm or Information Services": "Technology",
    "Computer Systems Design and Services": "Technology",
    "Financial Services": "Finance",
    "Banking/Financial Services": "Finance",
    "Fintech": "Finance",
    "Insurance": "Finance",
    "Manufacturing, Transportation, or Supply Chain": "Manufacturing_Energy_Transport",
    "Manufacturing": "Manufacturing_Energy_Transport",
    "Transportation, or Supply Chain": "Manufacturing_Energy_Transport",
    "Oil & Gas": "Manufacturing_Energy_Transport",
    "Energy": "Manufacturing_Energy_Transport",
    "Advertising Services": "Media_Advertising",
    "Media & Advertising Services": "Media_Advertising",
    "Healthcare": "Healthcare",
    "Higher Education": "Higher_Education",
    "Retail and Consumer Services": "Retail",
}
USA_IA = ["Yes"]
ACTITUD_POSITIVA = ["Favorable", "Very favorable"]
DEV_LABEL = "I am a developer by profession"


def output_path(restrict_to_users: bool) -> Path:
    """Ruta del parquet de salida."""
    return DATA / ("df_clean_users.parquet" if restrict_to_users else "df_clean.parquet")


def cargar_y_normalizar():
    """Carga las tres ediciones y unifica las columnas que cambiaron entre años."""
    df23 = pd.read_csv(DATA / "2023" / "survey_results_public.csv", low_memory=False)
    df24 = pd.read_csv(DATA / "2024" / "survey_results_public.csv", low_memory=False)
    df25 = pd.read_csv(DATA / "2025" / "survey_results_public.csv", low_memory=False)

    df23["survey_year"] = 2023
    df24["survey_year"] = 2024
    df25["survey_year"] = 2025

    print(f"Cargados: 2023={df23.shape[0]:,}  2024={df24.shape[0]:,}  2025={df25.shape[0]:,}")

    # En 2025 el "sí" se divide por frecuencia de uso; se junta en un solo "Yes"
    aiselect_yes_2025 = [
        "Yes, I use AI tools daily",
        "Yes, I use AI tools weekly",
        "Yes, I use AI tools monthly or infrequently",
    ]
    df25["AISelect"] = df25["AISelect"].apply(
        lambda v: "Yes" if v in aiselect_yes_2025 else v
    )

    # YearsCodePro no existe en 2025: no se rellena con YearsCode (serían idénticas en ese año
    # y el modelo leería el año a través de ellas). La experiencia se mide con YearsCode.

    # En 2023 las preguntas AIAcc y AIBen tienen los nombres invertidos respecto de 2024 y 2025
    df23 = df23.rename(columns={"AIAcc": "AIBen", "AIBen": "AIAcc"})

    # En 2025 los entornos de desarrollo cambian de nombre de columna
    if "DevEnvsHaveWorkedWith" in df25.columns and "NEWCollabToolsHaveWorkedWith" not in df25.columns:
        df25 = df25.rename(columns={"DevEnvsHaveWorkedWith": "NEWCollabToolsHaveWorkedWith"})

    for d in (df23, df24, df25):
        d["DevType"] = d["DevType"].apply(
            lambda v: _remap_devtype(v, DEVTYPE_REMAP) if isinstance(v, str) else v
        )

    return df23, df24, df25


def _remap_devtype(v: str, mapping: dict) -> str:
    """Aplica el mapeo a cada rol de una respuesta con varios roles separados por ';'."""
    tokens = [t.strip() for t in v.split(";")]
    mapped = [mapping.get(t, t) for t in tokens]
    # Dos roles pueden quedar con el mismo nombre después del mapeo
    seen = set()
    result = []
    for t in mapped:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return ";".join(result)


def construir_target(df: pd.DataFrame) -> pd.Series:
    """1 si usa IA y tiene actitud favorable, 0 en cualquier otro caso."""
    return (df["AISelect"].isin(USA_IA) & df["AISent"].isin(ACTITUD_POSITIVA)).astype(int)


def filtrar_poblacion(df: pd.DataFrame, restrict_to_users: bool = True) -> pd.DataFrame:
    n_antes = len(df)
    df = df[df["MainBranch"] == DEV_LABEL].copy()
    print(f"Desarrolladores profesionales: {len(df):,} (de {n_antes:,})")

    if restrict_to_users:
        n_antes_user = len(df)
        df = df[df["AISelect"] == "Yes"].copy()
        print(f"Usuarios de IA: {len(df):,} (de {n_antes_user:,})")

    return df


def reportar_target(df: pd.DataFrame) -> None:
    print(f"\nFilas: {len(df):,}")
    print(f"target=1 ({TARGET_LABELS[1]}): {df['target'].sum():,} ({df['target'].mean():.1%})")
    print(f"target=0 ({TARGET_LABELS[0]}): {(1-df['target']).sum():,} ({(1-df['target']).mean():.1%})")
    print("\nPor año:")
    print(df.groupby("survey_year")["target"].agg(["count", "sum", "mean"]).round(3))


def aplicar_pipeline_completo(df: pd.DataFrame) -> pd.DataFrame:
    """Selección de columnas, imputación y codificación."""
    COLS_KEEP = [
        "survey_year", "target",
        "DevType", "YearsCode", "WorkExp", "EdLevel",
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
        print(f"Columnas faltantes, completadas con NaN: {cols_faltantes}")

    df = df[COLS_KEEP].copy()
    df = df.rename(columns={
        "LanguageHaveWorkedWith": "Language",
        "DatabaseHaveWorkedWith": "Database",
        "PlatformHaveWorkedWith": "Platform",
        "NEWCollabToolsHaveWorkedWith": "DevEnvs",
    })

    UMBRAL_NULOS = 40.0
    null_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    cols_a_descartar = null_pct[null_pct > UMBRAL_NULOS].index.tolist()
    if cols_a_descartar:
        print(f"Descartadas por >{UMBRAL_NULOS}% nulos: {cols_a_descartar}")
        df = df.drop(columns=cols_a_descartar)

    # En las columnas de opción múltiple, un faltante se toma como ninguna opción marcada
    multi_val_cols = [c for c in ["Language", "Database", "Platform", "DevEnvs"] if c in df.columns]
    for c in multi_val_cols:
        df[c] = df[c].fillna("")

    if "DevType" in df.columns:
        df["DevType"] = df["DevType"].fillna("Other")

    nominales = [c for c in ["Employment", "RemoteWork", "ICorPM", "Industry", "Country"] if c in df.columns]
    for c in nominales:
        df[c] = df.groupby("survey_year")[c].transform(
            lambda s: s.fillna(s.mode().iloc[0]) if s.notna().any() else s
        )

    if "JobSat" in df.columns:
        med_2425 = df.loc[df["survey_year"].isin([2024, 2025]), "JobSat"].median()
        df["JobSat"] = df["JobSat"].fillna(med_2425)

    if "WorkExp" in df.columns:
        df["WorkExp"] = df.groupby("survey_year")["WorkExp"].transform(
            lambda s: s.fillna(s.median())
        )

    print(f"Nulos después de imputar: {df.isnull().sum().sum()}")

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

    for col in ["YearsCode"]:
        if col in df.columns:
            df[col] = df[col].apply(years_to_num)

    # El CSV usa el apóstrofo tipográfico (’); si no se normaliza, "Bachelor's" y
    # "Master's" no coinciden con las claves del mapeo y quedan como faltantes
    for col in ["EdLevel", "OrgSize"]:
        if col in df.columns:
            df[col] = df[col].str.replace("’", "'", regex=False)

    # "Something else" / "Other" no tienen lugar en el orden: quedan como faltantes y se imputan
    EDLEVEL_MAP = {
        "Primary/elementary school": 1,
        "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)": 2,
        "Some college/university study without earning a degree": 3,
        "Associate degree (A.A., A.S., etc.)": 4,
        "Bachelor's degree (B.A., B.S., B.Eng., etc.)": 5,
        "Master's degree (M.A., M.S., M.Eng., MBA, etc.)": 6,
        "Professional degree (JD, MD, Ph.D, Ed.D, etc.)": 7,
    }
    if "EdLevel" in df.columns:
        df["EdLevel"] = df["EdLevel"].map(EDLEVEL_MAP)

    # En 2025 aparece "Less than 20 employees", que junta dos tramos anteriores; se le da
    # un valor intermedio.
    # "I don't know" queda como faltante y se imputa.
    ORGSIZE_MAP = {
        "Just me - I am a freelancer, sole proprietor, etc.": 1,
        "2 to 9 employees": 2,
        "Less than 20 employees": 2.5,
        "10 to 19 employees": 3,
        "20 to 99 employees": 4,
        "100 to 499 employees": 5,
        "500 to 999 employees": 6,
        "1,000 to 4,999 employees": 7,
        "5,000 to 9,999 employees": 8,
        "10,000 or more employees": 9,
    }
    if "OrgSize" in df.columns:
        df["OrgSize"] = df["OrgSize"].map(ORGSIZE_MAP)

    # Los faltantes que deja la codificación se completan con la mediana de cada año
    for col in ["YearsCode", "EdLevel", "OrgSize", "WorkExp"]:
        if col in df.columns and df[col].isna().any():
            df[col] = df.groupby("survey_year")[col].transform(lambda s: s.fillna(s.median()))

    # Columnas de opción múltiple: una columna 0/1 por opción frecuente más la cantidad.
    # Solo se usan las opciones que existen en las tres ediciones: una opción nueva de 2025
    # (p. ej. Docker en plataformas) marcaría el año. La cantidad cuenta solo esas opciones.
    def binarize_multival(df, col, top_n, prefix):
        tokens_por_anio = {
            y: set(t.strip() for v in g if isinstance(v, str) and v for t in v.split(";"))
            for y, g in df.groupby("survey_year")[col]
        }
        comunes = set.intersection(*tokens_por_anio.values())
        tokens_all = []
        for v in df[col]:
            if isinstance(v, str) and v:
                tokens_all.extend(t.strip() for t in v.split(";") if t.strip() in comunes)
        top = [t for t, _ in Counter(tokens_all).most_common(top_n)]
        for t in top:
            colname = f"{prefix}{t.replace(' ', '_').replace('/', '_').replace('-', '_').replace(',', '').replace('+', 'p')}"
            df[colname] = df[col].apply(lambda v: 1 if isinstance(v, str) and t in v.split(";") else 0)
        df[f"{prefix}count"] = df[col].apply(
            lambda v: sum(t.strip() in comunes for t in v.split(";")) if isinstance(v, str) and v else 0
        )
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
        # Resto de Europa
        "Bulgaria": "EU", "Serbia": "EU", "Slovakia": "EU", "Lithuania": "EU", "Croatia": "EU",
        "Slovenia": "EU", "Estonia": "EU", "Latvia": "EU", "Belarus": "EU", "Cyprus": "EU",
        "Bosnia and Herzegovina": "EU", "Luxembourg": "EU", "Albania": "EU", "Malta": "EU",
        "Republic of Moldova": "EU", "Iceland": "EU", "Montenegro": "EU", "Kosovo": "EU",
        "Republic of North Macedonia": "EU", "The former Yugoslav Republic of Macedonia": "EU",
        "Isle of Man": "EU",
        # Resto de Asia (sin Medio Oriente, que queda en OTHER)
        "Viet Nam": "ASIA", "Taiwan": "ASIA", "Malaysia": "ASIA", "Sri Lanka": "ASIA",
        "Thailand": "ASIA", "Hong Kong (S.A.R.)": "ASIA", "Nepal": "ASIA", "Republic of Korea": "ASIA",
        "Kazakhstan": "ASIA", "Uzbekistan": "ASIA", "Kyrgyzstan": "ASIA", "Myanmar": "ASIA",
        "Cambodia": "ASIA", "Afghanistan": "ASIA",
        # Resto de Latinoamérica
        "Uruguay": "LATAM", "Venezuela, Bolivarian Republic of...": "LATAM", "Costa Rica": "LATAM",
        "Ecuador": "LATAM", "Dominican Republic": "LATAM", "Guatemala": "LATAM", "Paraguay": "LATAM",
        "Bolivia": "LATAM", "Cuba": "LATAM", "El Salvador": "LATAM", "Nicaragua": "LATAM",
        "Panama": "LATAM", "Honduras": "LATAM",
    }
    if "Country" in df.columns:
        df["Region"] = df["Country"].map(COUNTRY_REGION).fillna("OTHER")
        df = df.drop(columns=["Country"])

    # 2025 ya no distingue empleo de tiempo completo y parcial, así que se unen
    def simplify_employment(val):
        if not isinstance(val, str):
            return "Other"
        tokens = [t.strip() for t in val.split(";")]
        if any(t.startswith("Employed") for t in tokens):
            return "Employed"
        if any(t.startswith("Independent contractor") for t in tokens):
            return "Freelance"
        return "Other"

    if "Employment" in df.columns:
        df["Employment"] = df["Employment"].apply(simplify_employment)

    # RemoteWork: 2025 divide el híbrido en tres categorías; se unen
    def simplify_remote(val):
        if not isinstance(val, str):
            return val
        if val.startswith("Hybrid") or val.startswith("Your choice"):
            return "Hybrid"
        return val

    if "RemoteWork" in df.columns:
        df["RemoteWork"] = df["RemoteWork"].apply(simplify_remote)

    if "Industry" in df.columns:
        df["Industry"] = df["Industry"].map(INDUSTRY_GROUPS).fillna("Other")

    if "DevType" in df.columns:
        # Si la persona marcó varios roles, se toma el primero
        df["DevType"] = df["DevType"].apply(lambda v: v.split(";")[0].strip() if isinstance(v, str) else "Other")
        # Los 13 roles más frecuentes (todos los que tienen más de 1.000 usuarios de IA);
        # el resto pasa a Other
        top_devtypes = df["DevType"].value_counts().head(13).index.tolist()
        df["DevType"] = df["DevType"].apply(lambda v: v if v in top_devtypes else "Other")

    nominals_for_ohe = [c for c in ["Employment", "RemoteWork", "ICorPM", "Region", "DevType", "Industry"] if c in df.columns]
    for col in nominals_for_ohe:
        dummies = pd.get_dummies(df[col], prefix=col, dummy_na=False).astype(int)
        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

    return df


def construir_dataset(restrict_to_users: bool = True, save: bool = True) -> pd.DataFrame:
    """Arma el dataset limpio, solo con usuarios de IA o con todos los desarrolladores."""
    df23, df24, df25 = cargar_y_normalizar()

    df23["target"] = construir_target(df23)
    df24["target"] = construir_target(df24)
    df25["target"] = construir_target(df25)

    df = pd.concat([df23, df24, df25], ignore_index=True)
    df = filtrar_poblacion(df, restrict_to_users=restrict_to_users)
    reportar_target(df)

    df = aplicar_pipeline_completo(df)

    numeric_expected = ["YearsCode", "WorkExp", "EdLevel", "OrgSize", "target", "survey_year"]
    for col in numeric_expected:
        if col in df.columns:
            assert pd.api.types.is_numeric_dtype(df[col]), f"{col} no es numérico"

    if save:
        path = output_path(restrict_to_users)
        df.to_parquet(path, index=False)
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"\nGuardado {path.relative_to(ROOT)} ({size_mb:.1f} MB, {df.shape[0]:,} filas, {df.shape[1]} columnas)")

    return df


def main():
    print("Dataset con todos los desarrolladores")
    construir_dataset(restrict_to_users=False, save=True)

    print("\nDataset con usuarios de IA")
    construir_dataset(restrict_to_users=True, save=True)


if __name__ == "__main__":
    main()
