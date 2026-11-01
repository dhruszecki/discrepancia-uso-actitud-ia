"""Nombres legibles en español para las variables del dataset y formato numérico de las figuras."""

import locale

import matplotlib

_ROLES = {
    "Developer, full-stack": "Desarrollo full-stack",
    "Developer, back-end": "Desarrollo back-end",
    "Developer, front-end": "Desarrollo front-end",
    "Developer, mobile": "Desarrollo móvil",
    "Developer, desktop or enterprise applications": "Aplicaciones de escritorio o empresariales",
    "Developer, embedded applications or devices": "Sistemas embebidos",
    "Data scientist or machine learning specialist": "Ciencia de datos y aprendizaje automático",
    "Engineering manager": "Gerencia de ingeniería",
    "Data engineer": "Ingeniería de datos",
    "DevOps specialist": "DevOps",
    "Student": "Estudiante",
    "Other (please specify):": "Otro (especificar)",
    "Other": "Otro",
}


def nombre_rol(valor: str) -> str:
    """Nombre en español de un valor de DevType."""
    return _ROLES.get(valor, valor)


def num(valor: float, decimales: int = 1) -> str:
    """Número con coma decimal y punto de miles (formato argentino)."""
    texto = f"{valor:,.{decimales}f}"
    return texto.replace(",", "_").replace(".", ",").replace("_", ".").replace("-", "−")


def configurar_figuras() -> None:
    """Ejes de matplotlib con coma decimal. Si el sistema no tiene el locale, no cambia nada."""
    for nombre in ("es_AR.UTF-8", "es_AR.utf8", "es_ES.UTF-8", "es_ES.utf8"):
        try:
            locale.setlocale(locale.LC_NUMERIC, nombre)
        except locale.Error:
            continue
        matplotlib.rcParams["axes.formatter.use_locale"] = True
        return

_FIJOS = {
    "survey_year": "Año de la encuesta",
    "YearsCode": "Años programando (total)",
    "WorkExp": "Años de experiencia laboral",
    "EdLevel": "Nivel educativo",
    "OrgSize": "Tamaño de la organización",
    "env_count": "Cantidad de entornos usados",
    "db_count": "Cantidad de bases de datos",
    "lang_count": "Cantidad de lenguajes",
    "plat_count": "Cantidad de plataformas",
    "Employment_Employed": "Empleo: en relación de dependencia",
    "Employment_Freelance": "Empleo: independiente",
    "Employment_Other": "Empleo: otro",
    "ICorPM_Individual contributor": "Posición: contribuidor individual",
    "ICorPM_People manager": "Posición: gestiona personas",
    "Region_ASIA": "Región: Asia",
    "Region_EU": "Región: Europa",
    "Region_LATAM": "Región: Latinoamérica",
    "Region_NA": "Región: Norteamérica",
    "Region_OCE": "Región: Oceanía",
    "Region_OTHER": "Región: otras",
    "lang_HTML_CSS": "Lenguaje: HTML/CSS",
    "lang_Cpp": "Lenguaje: C++",
    "lang_Bash_Shell_(all_shells)": "Lenguaje: Bash/Shell",
    "env_Visual_Studio_Code": "Entorno: VS Code",
    "env_Jupyter_Notebook_JupyterLab": "Entorno: Jupyter",
    "env_Notepadpp": "Entorno: Notepad++",
    "plat_Amazon_Web_Services_(AWS)": "Plataforma: AWS",
    "plat_Digital_Ocean": "Plataforma: DigitalOcean",
    "Industry_Technology": "Industria: tecnología",
    "Industry_Finance": "Industria: finanzas",
    "Industry_Manufacturing_Energy_Transport": "Industria: manufactura, energía y transporte",
    "Industry_Media_Advertising": "Industria: medios y publicidad",
    "Industry_Healthcare": "Industria: salud",
    "Industry_Higher_Education": "Industria: educación superior",
    "Industry_Retail": "Industria: comercio",
    "Industry_Other": "Industria: otra",
}

_REMOTO = {
    "Remote": "remoto",
    "In-person": "presencial",
}

_PREFIJOS = {
    "lang_": "Lenguaje",
    "env_": "Entorno",
    "db_": "Base de datos",
    "plat_": "Plataforma",
    "Industry_": "Industria",
    "DevType_": "Rol",
}


def nombre_legible(variable: str) -> str:
    """Etiqueta en español de una variable. Si no hay regla, devuelve el nombre original."""
    if variable in _FIJOS:
        return _FIJOS[variable]
    if variable.startswith("RemoteWork_"):
        valor = variable[len("RemoteWork_"):]
        if valor in _REMOTO:
            return f"Modalidad: {_REMOTO[valor]}"
        if valor.startswith("Hybrid"):
            return "Modalidad: híbrido"
        return "Modalidad: a elección"
    if variable.startswith("DevType_"):
        return f"Rol: {nombre_rol(variable[len('DevType_'):])}"
    for prefijo, etiqueta in _PREFIJOS.items():
        if variable.startswith(prefijo):
            valor = variable[len(prefijo):].replace("_", " ").strip(": ")
            return f"{etiqueta}: {valor[:30]}"
    return variable


def renombrar_columnas(df):
    """Copia del DataFrame con las columnas renombradas (solo para graficar)."""
    return df.rename(columns=nombre_legible)
