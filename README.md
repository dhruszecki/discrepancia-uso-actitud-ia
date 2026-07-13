# ¿Qué desarrolladores usan IA pero no confían en ella?

**Discrepancia uso–actitud en desarrolladores de software: un modelo predictivo sobre el Stack Overflow Developer Survey 2023–2025.**

Repositorio de código del Trabajo Final de *Taller de Tesis I* — Maestría en Explotación de Datos y Descubrimiento del Conocimiento (Data Mining), FCEyN, Universidad de Buenos Aires.

Este repositorio contiene el código, las figuras y las instrucciones de reproducibilidad del análisis. El documento escrito se entrega por separado.

## Pregunta de investigación

> ¿Qué características del perfil profesional de un desarrollador que ya usa IA predicen una discrepancia entre ese uso y una actitud favorable hacia ella?

Sobre la subpoblación de usuarios activos de IA (`AISelect=Yes`, N = 80.369), se construye un target binario, se comparan cuatro modelos supervisados con validación cruzada estratificada y se interpretan los predictores con SHAP.

```
target = 1  ("sin discrepancia")  →  AISelect=Yes  y  AISent ∈ {Favorable, Very favorable}
target = 0  ("con discrepancia")  →  AISelect=Yes  y  cualquier otra actitud declarada
```

Distribución: 78,5 % sin discrepancia / 21,5 % con discrepancia. El desbalance se compensa con `class_weight='balanced'` (LogReg, Random Forest, Decision Tree) y `scale_pos_weight` (XGBoost).

## Marco teórico (referencias principales)

- **Lee & See (2004)**, *Human Factors* 46(1) — confianza calibrada en automatización (*misuse* / *disuse*).
- **Gillespie et al. / KPMG (2025)** — brecha uso–confianza a escala global (N = 48.340, 47 países).
- **Wang et al. (2024)**, FAccT '24 — dimensiones *ability / integrity / benevolence* de la confianza en herramientas de código.
- **Hoff & Bashir (2015)**, *Human Factors* 57(3) — *learned trust*: la confianza se actualiza con la experiencia de uso.

## Dataset

**Stack Overflow Developer Survey — ediciones 2023, 2024 y 2025** · Fuente: <https://survey.stackoverflow.co/> · Licencia: Open Database License (ODbL).

Los archivos CSV **no están incluidos en este repositorio** (los publica Stack Overflow bajo ODbL). Para reproducir el análisis, descargar las tres ediciones y ubicarlas así:

```
data/
├── 2023/survey_results_public.csv
├── 2024/survey_results_public.csv
└── 2025/survey_results_public.csv
```

Tras filtrar por `MainBranch` (desarrolladores profesionales) el dataset queda en 154.911 filas; restringido a usuarios activos de IA, en 80.369.

## Reproducibilidad

Requisitos: Python ≥ 3.11.13 y [uv](https://docs.astral.sh/uv/).

```bash
# 1. Instalar el entorno exacto (versiones fijadas en uv.lock)
uv sync

# 2. Descargar los CSV del survey y ubicarlos en data/ (ver sección Dataset)

# 3. Construir los datasets procesados (genera data/*.parquet)
uv run python scripts/pipeline.py

# 4. Ejecutar el análisis: notebooks orquestadores 01→04 …
uv run jupyter lab

#    … o los scripts individuales, que escriben figuras en outputs/
#    y reportes en resultados/
uv run python scripts/eda.py
uv run python scripts/modelado.py
uv run python scripts/temporal.py
uv run python scripts/modelado_temporal.py
uv run python scripts/sensibilidad_target.py
```

### Versiones del software

El entorno está fijado de forma reproducible en `uv.lock` (resolución exacta de las 127 dependencias). Versiones mínimas declaradas en `pyproject.toml`:

| Componente | Versión |
|---|---|
| Python | ≥ 3.11.13 |
| pandas | ≥ 3.0.1 |
| numpy | ≥ 2.4.3 |
| scikit-learn | ≥ 1.8.0 |
| xgboost | ≥ 3.2.0 |
| shap | ≥ 0.51.0 |
| matplotlib | ≥ 3.10.8 |
| seaborn | ≥ 0.13.2 |
| pyarrow | ≥ 20.0.0 |
| jupyterlab | ≥ 4.5.6 |

## Estructura del repositorio

```
.
├── scripts/                     # lógica de análisis (funciones puras)
│   ├── pipeline.py              # carga + limpieza + imputación + encoding → data/*.parquet
│   ├── eda.py                   # H1 (seniority), H2 (rol), H4 (clustering K-Means)
│   ├── invarianza.py            # análisis estratificado por año (robustez)
│   ├── modelado.py              # 4 modelos + CV k=5 + SHAP global/local + mapeo Wang
│   ├── decision_drop_auc.py     # comparativa dataset completo vs. restringido
│   ├── temporal.py              # tasa de discrepancia por año (Capa A, descriptiva)
│   ├── modelado_temporal.py     # modelos intra-año + forward validation (concept drift)
│   ├── proyeccion_temporal.py   # proyección condicionada 2026 (ilustrativa)
│   └── sensibilidad_target.py   # sensibilidad de la operacionalización del target
├── notebooks/                   # orquestadores que importan de scripts/ y muestran resultados
│   ├── 01_exploracion.ipynb
│   ├── 02_preprocesamiento.ipynb
│   ├── 03_eda.ipynb
│   └── 04_modelado.ipynb
├── outputs/                     # figuras (PNG) y métricas (JSON) generadas
├── resultados/                  # reportes markdown generados por los scripts (se crea al correr)
├── data/                        # NO incluido — descargar los CSV del survey (ver Dataset)
├── pyproject.toml               # dependencias
├── uv.lock                      # entorno exacto y reproducible
└── LICENSE
```

## Hallazgos principales

| Hipótesis | Resultado (usuarios activos, N = 80.369) |
|---|---|
| **H1 — Seniority** | Efecto positivo pero muy débil: r = 0,042 (≈ 0,2 % de varianza). 19,1 % (juniors) → 24,4 % (>20 años). |
| **H2 — Tipo de rol** | Embedded 30,2 % · Students 26,8 % · Desktop 26,1 %, sobre una media global de 21,5 %. |
| **H3 — Aporte del año** | `survey_year` suma solo +1,4 pp de AUC sobre el perfil: la señal reside en el perfil. |
| **H4 — Clustering** | K-Means k=4 describe subperfiles internos; rango de discrepancia estrecho (20,6 %–23,6 %). |

- **Modelado supervisado (CV k=5):** ROC-AUC ≈ 0,63 (LogReg, XGBoost y Random Forest indistinguibles).
- **Sensibilidad del target:** el perfil predice la *desconfianza activa* mucho mejor (ROC-AUC 0,72) que el target amplio (0,63); la indiferencia es el componente menos predecible.
- **No-estacionariedad:** la tasa de discrepancia crece de 15,3 % (2023) a 27,1 % (2025); el AUC intra-año sube de 0,577 a 0,639. La *forward validation* mide *concept drift* real y direccional.
- **Interpretabilidad (SHAP + Wang):** una vez descontado el artefacto de conteo y el drift temporal, las tres dimensiones tienen señal por feature comparable; el peso aparente de *ability* refleja el sesgo del instrumento hacia preguntas técnicas.

## Referencias

- Davis, F. D. (1989). *Perceived Usefulness, Perceived Ease of Use, and User Acceptance of Information Technology.* MIS Quarterly, 13(3).
- Lee, J. D. & See, K. A. (2004). *Trust in Automation: Designing for Appropriate Reliance.* Human Factors, 46(1), 50–80.
- Hoff, K. A. & Bashir, M. (2015). *Trust in Automation: Integrating Empirical Evidence on Factors that Influence Trust.* Human Factors, 57(3), 407–434.
- Wang, R., Cheng, R., Ford, D. & Zimmermann, T. (2024). *Investigating and Designing for Trust in AI-powered Code Generation Tools.* FAccT '24, 1475–1502.
- Gillespie, N., Lockey, S., Ward, T., Macdade, A. & Hassed, G. (2025). *Trust, Attitudes and Use of Artificial Intelligence: A Global Study 2025.* University of Melbourne / KPMG.
- Lundberg, S. M. & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS 30.

## Licencia

Código bajo licencia MIT (ver `LICENSE`). Los datos del Stack Overflow Developer Survey se distribuyen bajo Open Database License (ODbL) y no forman parte de este repositorio.
