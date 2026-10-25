# Discrepancia entre uso y actitud hacia la IA en desarrolladores de software

Código del trabajo final de especialización *Usar IA sin estar convencido: discrepancia entre uso y actitud en desarrolladores de software* (Maestría en Explotación de Datos y Descubrimiento del Conocimiento, FCEyN-UBA).

El trabajo estudia a los desarrolladores que usan herramientas de IA pero no declaran una actitud favorable hacia ellas, con las ediciones 2023, 2024 y 2025 de la encuesta de Stack Overflow. La pregunta es qué características del perfil profesional se asocian con esa discrepancia y cómo cambia esa relación entre años.

Este repositorio tiene lo necesario para reproducir todos los números y figuras del trabajo: los scripts con el análisis y los notebooks que los ejecutan paso a paso.

## Contenido

```
scripts/      lógica del análisis (un script por etapa)
notebooks/    los cuatro notebooks que recorren el análisis en orden
pyproject.toml, uv.lock   dependencias con versiones fijas
```

| Script | Qué hace |
|---|---|
| `pipeline.py` | Carga las tres ediciones, unifica las categorías que cambian entre años, construye la variable a predecir, imputa y codifica. Genera los dos datasets en `data/`. |
| `eda.py` | Hipótesis H1 (experiencia) y H2 (rol), y los grupos de K-Means. |
| `temporal.py` | Tasa de discrepancia, H1 y H2 en cada año. |
| `brecha_uso_actitud.py` | Cruce entre uso y actitud en todos los desarrolladores. |
| `modelado.py` | Cuatro modelos con validación cruzada, valores SHAP, tres casos y dimensiones de Wang et al. (2024). |
| `decision_drop_auc.py` | Misma regresión logística sobre todos los desarrolladores y sobre los usuarios de IA. |
| `sensibilidad_target.py` | Resultado con otras definiciones de la clase con discrepancia. |
| `modelado_temporal.py` | Modelos dentro de cada año y validación hacia adelante. |
| `proyeccion_temporal.py` | Escenarios ilustrativos para 2026. |
| `nombres_legibles.py` | Nombres en español de las variables para las figuras. |

## Requisitos

- Python 3.11
- [uv](https://docs.astral.sh/uv/) para instalar las dependencias (también sirve `pip` con las versiones de `uv.lock`)

## Paso a paso

### 1. Instalar las dependencias

```bash
git clone https://github.com/dhruszecki/discrepancia-uso-actitud-ia.git
cd discrepancia-uso-actitud-ia
uv sync
```

### 2. Descargar los datos

Los datos no están en el repositorio. Son las ediciones 2023, 2024 y 2025 del Stack Overflow Developer Survey, que se publican con la licencia Open Database License en <https://survey.stackoverflow.co/>. De cada edición hay que bajar el archivo completo («Download Full Data Set (CSV)»), descomprimirlo y dejar el archivo `survey_results_public.csv` así:

```
data/
├── 2023/survey_results_public.csv
├── 2024/survey_results_public.csv
└── 2025/survey_results_public.csv
```

### 3. Correr el análisis

Hay dos formas de hacerlo. Los notebooks recorren el análisis en orden y muestran cada resultado con sus tablas y figuras. Los scripts hacen el mismo cálculo y además guardan los informes en markdown, los resultados en JSON y la proyección para 2026. Para reproducir todo el trabajo alcanza con los scripts.

**Con los notebooks**, en orden:

```bash
uv run jupyter lab
```

1. `01_exploracion.ipynb`: carga de las tres ediciones y primera exploración.
2. `02_preprocesamiento.ipynb`: genera `data/df_clean.parquet` y `data/df_clean_users.parquet`.
3. `03_eda.ipynb`: H1, H2, variación por año, grupos y brecha entre uso y actitud.
4. `04_modelado.ipynb`: modelos, SHAP y análisis temporal.

**Con los scripts**, desde la raíz del repositorio y en este orden:

```bash
uv run python scripts/pipeline.py
uv run python scripts/eda.py
uv run python scripts/temporal.py
uv run python scripts/brecha_uso_actitud.py
uv run python scripts/modelado.py
uv run python scripts/decision_drop_auc.py
uv run python scripts/sensibilidad_target.py
uv run python scripts/modelado_temporal.py
uv run python scripts/proyeccion_temporal.py
```

`pipeline.py` tiene que correr primero, porque los demás leen los datasets que genera. `proyeccion_temporal.py` va al final, porque usa los resultados de `modelado_temporal.py`. El modelado es lo que más tarda.

### 4. Resultados

Las figuras quedan en `outputs/` y, si se corren los scripts, los informes con las tablas de cada etapa quedan en `docs/analisis/`. Todos los modelos usan la semilla 42, así que los resultados se reproducen exactamente con las versiones de `uv.lock`.

## Licencia

El código se publica con la licencia MIT. Los datos de la encuesta pertenecen a Stack Overflow y se usan con la licencia Open Database License.
