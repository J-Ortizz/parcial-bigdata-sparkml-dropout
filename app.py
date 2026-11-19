import os
import sys
import json
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    DoubleType,
)
from pyspark.ml import PipelineModel

# ---------------------------------------------------------------------------
# Configuración de página y estilos institucionales
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Alerta Temprana de Deserción y Éxito Académico | Spark ML",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 60%, #0D9488 100%);
        color: white;
        padding: 1.4rem 1.8rem;
        border-radius: 14px;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 15px rgba(15, 23, 42, 0.15);
    }
    .main-header h1 {
        color: #FFFFFF !important;
        font-size: 1.85rem !important;
        margin-bottom: 0.3rem !important;
    }
    .main-header p {
        color: #E2E8F0 !important;
        font-size: 1.02rem !important;
        margin: 0 !important;
    }
    .instruction-box {
        background-color: #EFF6FF;
        border-left: 5px solid #1D4ED8;
        padding: 1rem 1.3rem;
        border-radius: 8px;
        margin-bottom: 1.2rem;
        color: #1E293B;
    }
    .result-card-graduate {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        border: 2px solid #10B981;
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.15);
    }
    .result-card-dropout {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border: 2px solid #EF4444;
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.15);
    }
    .result-card-enrolled {
        background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%);
        border: 2px solid #F59E0B;
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.15);
    }
    .author-footer {
        margin-top: 2.5rem;
        padding: 1.2rem;
        background: #0F172A;
        color: #F8FAFC;
        border-radius: 12px;
        text-align: center;
        font-size: 0.96rem;
    }
    .author-footer strong {
        color: #38BDF8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# (a) Carga en caché de SparkSession y PipelineModel con @st.cache_resource
# ---------------------------------------------------------------------------
def _ensure_windows_hadoop():
    """Configura winutils.exe y hadoop.dll automáticamente si se ejecuta en Windows."""
    if os.name == "nt":
        hadoop_home = os.environ.get("HADOOP_HOME", os.path.join(os.path.expanduser("~"), ".hadoop"))
        if not os.path.exists(os.path.join(hadoop_home, "bin", "winutils.exe")):
            hadoop_home = os.path.join(os.path.expanduser("~"), ".hadoop")
            os.makedirs(os.path.join(hadoop_home, "bin"), exist_ok=True)
            base_url = "https://raw.githubusercontent.com/cdarlint/winutils/master/hadoop-3.3.5/bin"
            for fn in ("winutils.exe", "hadoop.dll"):
                target = os.path.join(hadoop_home, "bin", fn)
                if not os.path.exists(target):
                    urllib.request.urlretrieve(f"{base_url}/{fn}", target)
        os.environ["HADOOP_HOME"] = hadoop_home
        bin_path = os.path.join(hadoop_home, "bin")
        if bin_path not in os.environ.get("PATH", ""):
            os.environ["PATH"] = bin_path + os.pathsep + os.environ.get("PATH", "")


@st.cache_resource(show_spinner="Inicializando motor Apache Spark ML y cargando PipelineModel...")
def load_spark_and_model():
    _ensure_windows_hadoop()
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"

    spark = (
        SparkSession.builder.appName("dropout_sparkml")
        .master("local[2]")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.constraintPropagation.enabled", "false")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.default.parallelism", "4")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "modelo_final")
    model = PipelineModel.load(model_path)

    labels_path = os.path.join(model_path, "labels.json")
    if os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            labels = meta.get("labels", ["Graduate", "Dropout", "Enrolled"])
            model_name = meta.get("best_model_name", "RandomForestClassifier")
            test_metrics = meta.get("test_metrics", {})
    else:
        labels = ["Graduate", "Dropout", "Enrolled"]
        model_name = "RandomForestClassifier"
        test_metrics = {}

    return spark, model, labels, model_name, test_metrics


# ---------------------------------------------------------------------------
# Catálogos de opciones legibles con códigos oficiales del dataset UCI/Kaggle
# ---------------------------------------------------------------------------
MARITAL_STATUS_OPTS = {
    1: "1 — Soltero/a",
    2: "2 — Casado/a",
    3: "3 — Viudo/a",
    4: "4 — Divorciado/a",
    5: "5 — Unión libre",
    6: "6 — Separado/a legalmente",
}

COURSE_OPTS = {
    33: "33 — Tecnologías de Producción de Biocombustibles",
    171: "171 — Diseño de Animación y Multimedia",
    8014: "8014 — Servicio Social (Jornada Nocturna)",
    9003: "9003 — Agronomía",
    9070: "9070 — Diseño de la Comunicación",
    9085: "9085 — Enfermería Veterinaria",
    9119: "9119 — Ingeniería Informática",
    9130: "9130 — Equinicultura",
    9147: "9147 — Administración y Gestión de Empresas",
    9238: "9238 — Servicio Social",
    9254: "9254 — Turismo",
    9500: "9500 — Enfermería",
    9556: "9556 — Higiene Oral",
    9670: "9670 — Gestión de Publicidad y Marketing",
    9773: "9773 — Periodismo y Comunicación",
    9853: "9853 — Educación Básica",
    9991: "9991 — Administración y Gestión (Nocturna)",
}

APP_MODE_OPTS = {
    1: "1 — 1.ª Fase (Contingente General)",
    2: "2 — Ordenanza N.º 612/93",
    5: "5 — 1.ª Fase (Contingente Especial Azores)",
    7: "7 — Titulares de Otros Cursos Superiores",
    10: "10 — Ordenanza N.º 854-B/99",
    15: "15 — Estudiante Internacional (Pregrado)",
    16: "16 — 1.ª Fase (Contingente Especial Madeira)",
    17: "17 — 2.ª Fase (Contingente General)",
    18: "18 — 3.ª Fase (Contingente General)",
    26: "26 — Ordenanza N.º 533-A/99 (Inciso b2)",
    27: "27 — Ordenanza N.º 533-A/99 (Inciso b3)",
    39: "39 — Mayores de 23 Años",
    42: "42 — Transferencia",
    43: "43 — Cambio de Curso / Programa",
    44: "44 — Titulares de Diploma de Especialización Tecnológica",
    51: "51 — Cambio de Institución / Curso",
    53: "53 — Titulares de Diploma de Técnico Superior",
    57: "57 — Cambio de Institución / Curso (Internacional)",
}

PREV_QUAL_OPTS = {
    1: "1 — Educación Secundaria Completa",
    2: "2 — Educación Superior (Licenciatura / Pregrado)",
    3: "3 — Educación Superior (Grado)",
    4: "4 — Educación Superior (Maestría)",
    5: "5 — Educación Superior (Doctorado)",
    6: "6 — Frecuencia de Educación Superior",
    9: "9 — 12.º Año de Escolaridad (No Completado)",
    10: "10 — 11.º Año de Escolaridad (No Completado)",
    12: "12 — Otro — 11.º Año de Escolaridad",
    14: "14 — 10.º Año de Escolaridad",
    15: "15 — 10.º Año de Escolaridad (No Completado)",
    19: "19 — Educación Básica 3.er Ciclo (9.º/10.º/11.º Año)",
    38: "38 — Educación Básica 2.º Ciclo (6.º/7.º/8.º Año)",
    39: "39 — Curso de Especialización Tecnológica",
    40: "40 — Educación Superior (Grado — 1.er Ciclo)",
    42: "42 — Curso Técnico Superior Profesional",
    43: "43 — Educación Superior (Maestría — 2.º Ciclo)",
}

NACIONALITY_OPTS = {
    1: "1 — Local / Portuguesa",
    2: "2 — Alemana",
    6: "6 — Española",
    11: "11 — Italiana",
    13: "13 — Neerlandesa",
    14: "14 — Inglesa",
    17: "17 — Lituana",
    21: "21 — Angoleña",
    22: "22 — Caboverdiana",
    24: "24 — Guineana",
    25: "25 — Mozambiqueña",
    26: "26 — Santotomense",
    32: "32 — Turca",
    41: "41 — Brasileña",
    62: "62 — Rumana",
    100: "100 — Moldava",
    101: "101 — Mexicana",
    103: "103 — Ucraniana",
    105: "105 — Rusa",
    108: "108 — Cubana",
    109: "109 — Colombiana",
}

PARENT_QUAL_OPTS = {
    1: "1 — Educación Secundaria (12.º Año o Equivalente)",
    2: "2 — Educación Superior (Licenciatura)",
    3: "3 — Educación Superior (Grado)",
    4: "4 — Educación Superior (Maestría)",
    5: "5 — Educación Superior (Doctorado)",
    6: "6 — Frecuencia de Educación Superior",
    9: "9 — 12.º Año (No Completado)",
    10: "10 — 11.º Año (No Completado)",
    11: "11 — 7.º Año (Antiguo)",
    12: "12 — Otro — 11.º Año",
    14: "14 — 10.º Año",
    18: "18 — Curso General de Comercio",
    19: "19 — Educación Básica 3.er Ciclo (9.º/10.º/11.º)",
    22: "22 — Curso Técnico-Profesional",
    26: "26 — 7.º Año de Escolaridad",
    27: "27 — 2.º Ciclo del Curso General de Secundaria",
    29: "29 — 9.º Año (No Completado)",
    30: "30 — 8.º Año de Escolaridad",
    34: "34 — Desconocido",
    35: "35 — No sabe leer ni escribir",
    36: "36 — Sabe leer sin 4.º año de escolaridad",
    37: "37 — Educación Básica 1.er Ciclo (4.º/5.º Año)",
    38: "38 — Educación Básica 2.º Ciclo (6.º/7.º/8.º Año)",
    39: "39 — Curso de Especialización Tecnológica",
    40: "40 — Educación Superior (Grado 1.er Ciclo)",
    41: "41 — Estudios Superiores Especializados",
    42: "42 — Curso Técnico Superior Profesional",
    43: "43 — Educación Superior (Maestría 2.º Ciclo)",
    44: "44 — Educación Superior (Doctorado 3.er Ciclo)",
}

PARENT_OCCUP_OPTS = {
    0: "0 — Estudiante",
    1: "1 — Directivos y Poder Ejecutivo / Legislativo",
    2: "2 — Especialistas en Actividades Intelectuales y Científicas",
    3: "3 — Técnicos y Profesionales de Nivel Intermedio",
    4: "4 — Personal Administrativo",
    5: "5 — Trabajadores de Servicios Personales, Seguridad y Ventas",
    6: "6 — Agricultores y Trabajadores Calificados Agropecuarios",
    7: "7 — Trabajadores Calificados de la Industria y Construcción",
    8: "8 — Operadores de Instalaciones y Maquinaria",
    9: "9 — Trabajadores No Calificados",
    10: "10 — Profesiones de las Fuerzas Armadas",
    90: "90 — Otra Situación",
    99: "99 — En Blanco / Sin Información",
    122: "122 — Profesionales de la Salud",
    123: "123 — Docentes / Profesores",
    125: "125 — Especialistas en TIC",
    131: "131 — Técnicos en Ciencias e Ingeniería",
    132: "132 — Técnicos de Salud",
    134: "134 — Técnicos de Servicios Jurídicos, Sociales y Culturales",
    141: "141 — Empleados de Oficina y Secretariado",
    143: "143 — Operadores de Datos, Contabilidad y Finanzas",
    144: "144 — Otro Personal de Apoyo Administrativo",
    151: "151 — Trabajadores de Servicios Personales",
    152: "152 — Vendedores",
    153: "153 — Trabajadores de Cuidado Personal",
    171: "171 — Trabajadores Calificados de la Construcción",
    173: "173 — Artesanos y Trabajadores de Imprenta",
    175: "175 — Trabajadores de Procesamiento de Alimentos y Madera",
    191: "191 — Personal de Limpieza",
    192: "192 — Trabajadores No Calificados Agropecuarios",
    193: "193 — Trabajadores No Calificados de Construcción y Manufactura",
    194: "194 — Auxiliares de Preparación de Alimentos",
    195: "195 — Vendedores Ambulantes y Servicios Callejeros",
}


PRESET_PROFILES = {
    "Caso 1: Estudiante con Alto Desempeño (Camino a Graduarse)": {
        "marital_status": 1,
        "application_mode": 1,
        "application_order": 1,
        "course": 9500,
        "daytime_evening_attendance": 1,
        "previous_qualification": 1,
        "previous_qualification_grade": 145.0,
        "nacionality": 1,
        "mother_s_qualification": 3,
        "father_s_qualification": 3,
        "mother_s_occupation": 2,
        "father_s_occupation": 3,
        "admission_grade": 148.5,
        "displaced": 1,
        "educational_special_needs": 0,
        "debtor": 0,
        "tuition_fees_up_to_date": 1,
        "gender": 0,
        "scholarship_holder": 1,
        "age_at_enrollment": 19,
        "international": 0,
        "curricular_units_1st_sem_credited": 0,
        "curricular_units_1st_sem_enrolled": 6,
        "curricular_units_1st_sem_evaluations": 6,
        "curricular_units_1st_sem_approved": 6,
        "curricular_units_1st_sem_grade": 14.5,
        "curricular_units_1st_sem_without_evaluations": 0,
        "curricular_units_2nd_sem_credited": 0,
        "curricular_units_2nd_sem_enrolled": 6,
        "curricular_units_2nd_sem_evaluations": 6,
        "curricular_units_2nd_sem_approved": 6,
        "curricular_units_2nd_sem_grade": 15.2,
        "curricular_units_2nd_sem_without_evaluations": 0,
        "unemployment_rate": 10.8,
        "inflation_rate": 1.4,
        "gdp": 1.74,
    },
    "Caso 2: Estudiante en Alto Riesgo de Deserción (Alerta Temprana)": {
        "marital_status": 1,
        "application_mode": 39,
        "application_order": 1,
        "course": 9119,
        "daytime_evening_attendance": 0,
        "previous_qualification": 1,
        "previous_qualification_grade": 115.0,
        "nacionality": 1,
        "mother_s_qualification": 19,
        "father_s_qualification": 19,
        "mother_s_occupation": 9,
        "father_s_occupation": 8,
        "admission_grade": 112.0,
        "displaced": 0,
        "educational_special_needs": 0,
        "debtor": 1,
        "tuition_fees_up_to_date": 0,
        "gender": 1,
        "scholarship_holder": 0,
        "age_at_enrollment": 28,
        "international": 0,
        "curricular_units_1st_sem_credited": 0,
        "curricular_units_1st_sem_enrolled": 6,
        "curricular_units_1st_sem_evaluations": 5,
        "curricular_units_1st_sem_approved": 1,
        "curricular_units_1st_sem_grade": 10.5,
        "curricular_units_1st_sem_without_evaluations": 1,
        "curricular_units_2nd_sem_credited": 0,
        "curricular_units_2nd_sem_enrolled": 6,
        "curricular_units_2nd_sem_evaluations": 3,
        "curricular_units_2nd_sem_approved": 0,
        "curricular_units_2nd_sem_grade": 0.0,
        "curricular_units_2nd_sem_without_evaluations": 3,
        "unemployment_rate": 13.9,
        "inflation_rate": 2.6,
        "gdp": -1.70,
    },
    "Caso 3: Estudiante en Progreso Regular (Continúa Matriculado / Enrolled)": {
        "marital_status": 1,
        "application_mode": 17,
        "application_order": 2,
        "course": 9147,
        "daytime_evening_attendance": 1,
        "previous_qualification": 1,
        "previous_qualification_grade": 126.0,
        "nacionality": 1,
        "mother_s_qualification": 1,
        "father_s_qualification": 19,
        "mother_s_occupation": 4,
        "father_s_occupation": 7,
        "admission_grade": 124.0,
        "displaced": 1,
        "educational_special_needs": 0,
        "debtor": 0,
        "tuition_fees_up_to_date": 1,
        "gender": 1,
        "scholarship_holder": 0,
        "age_at_enrollment": 20,
        "international": 0,
        "curricular_units_1st_sem_credited": 0,
        "curricular_units_1st_sem_enrolled": 6,
        "curricular_units_1st_sem_evaluations": 8,
        "curricular_units_1st_sem_approved": 4,
        "curricular_units_1st_sem_grade": 12.1,
        "curricular_units_1st_sem_without_evaluations": 0,
        "curricular_units_2nd_sem_credited": 0,
        "curricular_units_2nd_sem_enrolled": 6,
        "curricular_units_2nd_sem_evaluations": 9,
        "curricular_units_2nd_sem_approved": 4,
        "curricular_units_2nd_sem_grade": 12.3,
        "curricular_units_2nd_sem_without_evaluations": 0,
        "unemployment_rate": 11.1,
        "inflation_rate": 0.6,
        "gdp": 2.02,
    },
}

CLASS_TRANSLATIONS = {
    "Graduate": {
        "es": "Graduado / Éxito Académico (Graduate)",
        "icon": "🎓",
        "card_class": "result-card-graduate",
        "color": "#10B981",
        "desc": "El estudiante presenta indicadores académicos y financieros sólidos que lo proyectan hacia la culminación exitosa de sus estudios.",
    },
    "Dropout": {
        "es": "En Riesgo de Deserción (Dropout)",
        "icon": "⚠️",
        "card_class": "result-card-dropout",
        "color": "#EF4444",
        "desc": "ALERTA TEMPRANA: El estudiante presenta factores críticos asociados con abandono universitario. Se recomienda intervención de bienestar y tutoría académica.",
    },
    "Enrolled": {
        "es": "Matriculado en Curso (Enrolled)",
        "icon": "📘",
        "card_class": "result-card-enrolled",
        "color": "#F59E0B",
        "desc": "El estudiante continúa activo en su trayectoria curricular, con avance parcial. Se sugiere seguimiento preventivo para asegurar su transición hacia la graduación.",
    },
}

INPUT_SCHEMA = StructType(
    [
        StructField("marital_status", IntegerType(), True),
        StructField("application_mode", IntegerType(), True),
        StructField("application_order", IntegerType(), True),
        StructField("course", IntegerType(), True),
        StructField("daytime_evening_attendance", IntegerType(), True),
        StructField("previous_qualification", IntegerType(), True),
        StructField("previous_qualification_grade", DoubleType(), True),
        StructField("nacionality", IntegerType(), True),
        StructField("mother_s_qualification", IntegerType(), True),
        StructField("father_s_qualification", IntegerType(), True),
        StructField("mother_s_occupation", IntegerType(), True),
        StructField("father_s_occupation", IntegerType(), True),
        StructField("admission_grade", DoubleType(), True),
        StructField("displaced", IntegerType(), True),
        StructField("educational_special_needs", IntegerType(), True),
        StructField("debtor", IntegerType(), True),
        StructField("tuition_fees_up_to_date", IntegerType(), True),
        StructField("gender", IntegerType(), True),
        StructField("scholarship_holder", IntegerType(), True),
        StructField("age_at_enrollment", IntegerType(), True),
        StructField("international", IntegerType(), True),
        StructField("curricular_units_1st_sem_credited", IntegerType(), True),
        StructField("curricular_units_1st_sem_enrolled", IntegerType(), True),
        StructField("curricular_units_1st_sem_evaluations", IntegerType(), True),
        StructField("curricular_units_1st_sem_approved", IntegerType(), True),
        StructField("curricular_units_1st_sem_grade", DoubleType(), True),
        StructField("curricular_units_1st_sem_without_evaluations", IntegerType(), True),
        StructField("curricular_units_2nd_sem_credited", IntegerType(), True),
        StructField("curricular_units_2nd_sem_enrolled", IntegerType(), True),
        StructField("curricular_units_2nd_sem_evaluations", IntegerType(), True),
        StructField("curricular_units_2nd_sem_approved", IntegerType(), True),
        StructField("curricular_units_2nd_sem_grade", DoubleType(), True),
        StructField("curricular_units_2nd_sem_without_evaluations", IntegerType(), True),
        StructField("unemployment_rate", DoubleType(), True),
        StructField("inflation_rate", DoubleType(), True),
        StructField("gdp", DoubleType(), True),
        StructField("approval_rate_1st_sem", DoubleType(), True),
        StructField("approval_rate_2nd_sem", DoubleType(), True),
        StructField("grade_diff_sem", DoubleType(), True),
        StructField("total_approved_units", IntegerType(), True),
    ]
)

base_dir = os.path.dirname(os.path.abspath(__file__))
banner_path = os.path.join(base_dir, "banner.png")
if os.path.exists(banner_path):
    st.image(banner_path, use_container_width=True)

st.markdown(
    """
    <div class="main-header">
        <h1>🎓 Sistema Inteligente de Predicción de Deserción y Éxito Académico</h1>
        <p>Plataforma de analítica predictiva universitaria impulsada por <strong>Apache Spark ML (PipelineModel)</strong> para clasificar estudiantes en <em>Dropout</em>, <em>Enrolled</em> o <em>Graduate</em>.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

spark, pipeline_model, labels, best_model_name, test_metrics = load_spark_and_model()

with st.sidebar:
    st.header("⚙️ Información del Modelo")
    st.success(f"**Modelo activo:** `{best_model_name}`")
    st.caption("Cargado en memoria con `PipelineModel.load('modelo_final')` y `@st.cache_resource`.")

    st.markdown("**Orden de etiquetas (`StringIndexer.labels`):**")
    for idx, lbl in enumerate(labels):
        st.markdown(f"- Índice `{idx}` → **{lbl}**")

    if test_metrics:
        st.divider()
        st.subheader("📊 Métricas en Test (20%)")
        c1, c2 = st.columns(2)
        c1.metric("F1 Ponderado", f"{test_metrics.get('f1', 0):.4f}")
        c2.metric("Accuracy", f"{test_metrics.get('accuracy', 0):.4f}")
        c3, c4 = st.columns(2)
        c3.metric("Precisión", f"{test_metrics.get('weightedPrecision', 0):.4f}")
        c4.metric("Recall", f"{test_metrics.get('weightedRecall', 0):.4f}")

    st.divider()
    st.subheader("🧪 Perfiles de Prueba Rápidos")
    selected_preset = st.selectbox(
        "Cargar valores de ejemplo en el formulario:",
        options=list(PRESET_PROFILES.keys()),
        index=0,
    )
    defaults = PRESET_PROFILES[selected_preset]

    st.divider()
    st.caption("Autor: **Juan David Ortiz Pérez**\nParcial Big Data — Apache Spark ML")

st.markdown(
    """
    <div class="instruction-box">
        <strong>📋 Instrucciones de uso:</strong>
        <ol style="margin-top: 0.4rem; margin-bottom: 0;">
            <li>Seleccione un <strong>perfil base</strong> en el panel lateral izquierdo o diligencie directamente las <strong>36 variables</strong> del estudiante en las secciones del formulario (Demográficas, Admisión, Socioeconómicas, Rendimiento de 1.º y 2.º Semestre y Contexto Macroeconómico).</li>
            <li>Presione el botón <strong>🔍 Predecir</strong> ubicado al final del formulario.</li>
            <li>La aplicación calculará automáticamente las <strong>variables académicas derivadas</strong> (tasas de aprobación por semestre, variación de nota y total de créditos aprobados), ejecutará el <code>PipelineModel</code> de Spark ML y mostrará la <strong>clase predicha</strong> junto con la distribución de <strong>probabilidades por clase</strong>.</li>
        </ol>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# (b) Formulario st.form con todas las variables que usa el modelo
# ---------------------------------------------------------------------------
with st.form("student_prediction_form"):
    st.subheader("📝 Formulario de Datos del Estudiante")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "👤 1. Perfil Demográfico",
            "🏛️ 2. Admisión y Programa",
            "💳 3. Socioeconómico y Familiar",
            "📚 4. Rendimiento 1.er y 2.º Semestre",
            "🌎 5. Contexto Macroeconómico",
        ]
    )

    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            marital_status = st.selectbox(
                "Estado Civil (marital_status)",
                options=list(MARITAL_STATUS_OPTS.keys()),
                format_func=lambda x: MARITAL_STATUS_OPTS[x],
                index=list(MARITAL_STATUS_OPTS.keys()).index(defaults["marital_status"]),
            )
            gender = st.selectbox(
                "Género (gender)",
                options=[0, 1],
                format_func=lambda x: "1 — Masculino" if x == 1 else "0 — Femenino",
                index=[0, 1].index(defaults["gender"]),
            )
            age_at_enrollment = st.number_input(
                "Edad al Matricularse (age_at_enrollment)",
                min_value=17,
                max_value=75,
                value=int(defaults["age_at_enrollment"]),
                step=1,
            )
        with col2:
            nacionality = st.selectbox(
                "Nacionalidad (nacionality)",
                options=list(NACIONALITY_OPTS.keys()),
                format_func=lambda x: NACIONALITY_OPTS[x],
                index=list(NACIONALITY_OPTS.keys()).index(defaults["nacionality"]),
            )
            international = st.selectbox(
                "Estudiante Internacional (international)",
                options=[0, 1],
                format_func=lambda x: "1 — Sí" if x == 1 else "0 — No",
                index=[0, 1].index(defaults["international"]),
            )
        with col3:
            displaced = st.selectbox(
                "Desplazado / Trasladado de Residencia (displaced)",
                options=[0, 1],
                format_func=lambda x: "1 — Sí" if x == 1 else "0 — No",
                index=[0, 1].index(defaults["displaced"]),
            )
            educational_special_needs = st.selectbox(
                "Necesidades Educativas Especiales (educational_special_needs)",
                options=[0, 1],
                format_func=lambda x: "1 — Sí" if x == 1 else "0 — No",
                index=[0, 1].index(defaults["educational_special_needs"]),
            )

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            course = st.selectbox(
                "Programa Académico (course)",
                options=list(COURSE_OPTS.keys()),
                format_func=lambda x: COURSE_OPTS[x],
                index=list(COURSE_OPTS.keys()).index(defaults["course"]),
            )
            daytime_evening_attendance = st.selectbox(
                "Jornada de Asistencia (daytime_evening_attendance)",
                options=[1, 0],
                format_func=lambda x: "1 — Diurna" if x == 1 else "0 — Nocturna",
                index=[1, 0].index(defaults["daytime_evening_attendance"]),
            )
            application_mode = st.selectbox(
                "Modo de Admisión (application_mode)",
                options=list(APP_MODE_OPTS.keys()),
                format_func=lambda x: APP_MODE_OPTS[x],
                index=list(APP_MODE_OPTS.keys()).index(defaults["application_mode"]),
            )
            application_order = st.number_input(
                "Orden de Preferencia en Solicitud (application_order: 0-9)",
                min_value=0,
                max_value=9,
                value=int(defaults["application_order"]),
                step=1,
            )
        with col2:
            previous_qualification = st.selectbox(
                "Nivel Educativo Previo (previous_qualification)",
                options=list(PREV_QUAL_OPTS.keys()),
                format_func=lambda x: PREV_QUAL_OPTS[x],
                index=list(PREV_QUAL_OPTS.keys()).index(defaults["previous_qualification"]),
            )
            previous_qualification_grade = st.number_input(
                "Nota de Cualificación Previa (previous_qualification_grade: 95-190)",
                min_value=0.0,
                max_value=200.0,
                value=float(defaults["previous_qualification_grade"]),
                step=0.5,
            )
            admission_grade = st.number_input(
                "Nota de Admisión (admission_grade: 95-190)",
                min_value=0.0,
                max_value=200.0,
                value=float(defaults["admission_grade"]),
                step=0.5,
            )

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            tuition_fees_up_to_date = st.selectbox(
                "Matrícula al Día (tuition_fees_up_to_date)",
                options=[1, 0],
                format_func=lambda x: "1 — Sí (Al día)" if x == 1 else "0 — No (En mora)",
                index=[1, 0].index(defaults["tuition_fees_up_to_date"]),
            )
            debtor = st.selectbox(
                "Estado de Deudor (debtor)",
                options=[0, 1],
                format_func=lambda x: "1 — Sí (Deudor)" if x == 1 else "0 — No",
                index=[0, 1].index(defaults["debtor"]),
            )
            scholarship_holder = st.selectbox(
                "Beneficiario de Beca (scholarship_holder)",
                options=[0, 1],
                format_func=lambda x: "1 — Sí (Becario)" if x == 1 else "0 — No",
                index=[0, 1].index(defaults["scholarship_holder"]),
            )
        with col2:
            mother_s_qualification = st.selectbox(
                "Nivel Educativo de la Madre (mother_s_qualification)",
                options=list(PARENT_QUAL_OPTS.keys()),
                format_func=lambda x: PARENT_QUAL_OPTS[x],
                index=list(PARENT_QUAL_OPTS.keys()).index(defaults["mother_s_qualification"]),
            )
            father_s_qualification = st.selectbox(
                "Nivel Educativo del Padre (father_s_qualification)",
                options=list(PARENT_QUAL_OPTS.keys()),
                format_func=lambda x: PARENT_QUAL_OPTS[x],
                index=list(PARENT_QUAL_OPTS.keys()).index(defaults["father_s_qualification"]),
            )
            mother_s_occupation = st.selectbox(
                "Ocupación de la Madre (mother_s_occupation)",
                options=list(PARENT_OCCUP_OPTS.keys()),
                format_func=lambda x: PARENT_OCCUP_OPTS[x],
                index=list(PARENT_OCCUP_OPTS.keys()).index(defaults["mother_s_occupation"]),
            )
            father_s_occupation = st.selectbox(
                "Ocupación del Padre (father_s_occupation)",
                options=list(PARENT_OCCUP_OPTS.keys()),
                format_func=lambda x: PARENT_OCCUP_OPTS[x],
                index=list(PARENT_OCCUP_OPTS.keys()).index(defaults["father_s_occupation"]),
            )

    with tab4:
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("#### 📘 Primer Semestre")
            curricular_units_1st_sem_credited = st.number_input(
                "Unidades Convalidadas 1.er Sem (credited)",
                min_value=0,
                max_value=30,
                value=int(defaults["curricular_units_1st_sem_credited"]),
                step=1,
            )
            curricular_units_1st_sem_enrolled = st.number_input(
                "Unidades Matriculadas 1.er Sem (enrolled)",
                min_value=0,
                max_value=30,
                value=int(defaults["curricular_units_1st_sem_enrolled"]),
                step=1,
            )
            curricular_units_1st_sem_evaluations = st.number_input(
                "Evaluaciones Presentadas 1.er Sem (evaluations)",
                min_value=0,
                max_value=50,
                value=int(defaults["curricular_units_1st_sem_evaluations"]),
                step=1,
            )
            curricular_units_1st_sem_approved = st.number_input(
                "Unidades Aprobadas 1.er Sem (approved)",
                min_value=0,
                max_value=30,
                value=int(defaults["curricular_units_1st_sem_approved"]),
                step=1,
            )
            curricular_units_1st_sem_grade = st.number_input(
                "Nota Promedio 1.er Sem (grade: 0-20)",
                min_value=0.0,
                max_value=20.0,
                value=float(defaults["curricular_units_1st_sem_grade"]),
                step=0.1,
            )
            curricular_units_1st_sem_without_evaluations = st.number_input(
                "Unidades sin Evaluación 1.er Sem (without_evaluations)",
                min_value=0,
                max_value=20,
                value=int(defaults["curricular_units_1st_sem_without_evaluations"]),
                step=1,
            )

        with col_s2:
            st.markdown("#### 📗 Segundo Semestre")
            curricular_units_2nd_sem_credited = st.number_input(
                "Unidades Convalidadas 2.º Sem (credited)",
                min_value=0,
                max_value=30,
                value=int(defaults["curricular_units_2nd_sem_credited"]),
                step=1,
            )
            curricular_units_2nd_sem_enrolled = st.number_input(
                "Unidades Matriculadas 2.º Sem (enrolled)",
                min_value=0,
                max_value=30,
                value=int(defaults["curricular_units_2nd_sem_enrolled"]),
                step=1,
            )
            curricular_units_2nd_sem_evaluations = st.number_input(
                "Evaluaciones Presentadas 2.º Sem (evaluations)",
                min_value=0,
                max_value=50,
                value=int(defaults["curricular_units_2nd_sem_evaluations"]),
                step=1,
            )
            curricular_units_2nd_sem_approved = st.number_input(
                "Unidades Aprobadas 2.º Sem (approved)",
                min_value=0,
                max_value=30,
                value=int(defaults["curricular_units_2nd_sem_approved"]),
                step=1,
            )
            curricular_units_2nd_sem_grade = st.number_input(
                "Nota Promedio 2.º Sem (grade: 0-20)",
                min_value=0.0,
                max_value=20.0,
                value=float(defaults["curricular_units_2nd_sem_grade"]),
                step=0.1,
            )
            curricular_units_2nd_sem_without_evaluations = st.number_input(
                "Unidades sin Evaluación 2.º Sem (without_evaluations)",
                min_value=0,
                max_value=20,
                value=int(defaults["curricular_units_2nd_sem_without_evaluations"]),
                step=1,
            )

    with tab5:
        col1, col2, col3 = st.columns(3)
        with col1:
            unemployment_rate = st.number_input(
                "Tasa de Desempleo % (unemployment_rate)",
                min_value=0.0,
                max_value=30.0,
                value=float(defaults["unemployment_rate"]),
                step=0.1,
            )
        with col2:
            inflation_rate = st.number_input(
                "Tasa de Inflación % (inflation_rate)",
                min_value=-5.0,
                max_value=20.0,
                value=float(defaults["inflation_rate"]),
                step=0.1,
            )
        with col3:
            gdp = st.number_input(
                "Crecimiento del PIB % (gdp)",
                min_value=-10.0,
                max_value=10.0,
                value=float(defaults["gdp"]),
                step=0.1,
            )

    st.markdown("---")
    submitted = st.form_submit_button("🔍 Predecir", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# (c), (d), (e) Cálculo de variables derivadas, inferencia Spark y Resultados
# ---------------------------------------------------------------------------
if submitted:
    # (c) Calcular dentro de la app las variables derivadas del punto 3 (controlando división por cero)
    approval_rate_1st_sem = (
        float(curricular_units_1st_sem_approved) / float(curricular_units_1st_sem_enrolled)
        if curricular_units_1st_sem_enrolled > 0
        else 0.0
    )
    approval_rate_2nd_sem = (
        float(curricular_units_2nd_sem_approved) / float(curricular_units_2nd_sem_enrolled)
        if curricular_units_2nd_sem_enrolled > 0
        else 0.0
    )
    grade_diff_sem = float(curricular_units_2nd_sem_grade) - float(curricular_units_1st_sem_grade)
    total_approved_units = int(curricular_units_1st_sem_approved) + int(curricular_units_2nd_sem_approved)

    st.subheader("📐 Variables Académicas Derivadas (Calculadas en Tiempo Real)")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric(
        "Tasa Aprobación 1.er Sem",
        f"{approval_rate_1st_sem * 100:.1f}%",
        help="curricular_units_1st_sem_approved / curricular_units_1st_sem_enrolled (0 si matriculadas = 0)",
    )
    d2.metric(
        "Tasa Aprobación 2.º Sem",
        f"{approval_rate_2nd_sem * 100:.1f}%",
        delta=f"{(approval_rate_2nd_sem - approval_rate_1st_sem) * 100:+.1f}% vs 1.er Sem",
        help="curricular_units_2nd_sem_approved / curricular_units_2nd_sem_enrolled (0 si matriculadas = 0)",
    )
    d3.metric(
        "Variación de Nota (2.º - 1.º)",
        f"{grade_diff_sem:+.2f} pts",
        delta=f"{grade_diff_sem:+.2f}",
        help="curricular_units_2nd_sem_grade - curricular_units_1st_sem_grade",
    )
    d4.metric(
        "Total Unidades Aprobadas",
        f"{total_approved_units} u.",
        help="curricular_units_1st_sem_approved + curricular_units_2nd_sem_approved",
    )

    # (d) Construir registro y convertir en un DataFrame de Spark
    row_tuple = (
        int(marital_status),
        int(application_mode),
        int(application_order),
        int(course),
        int(daytime_evening_attendance),
        int(previous_qualification),
        float(previous_qualification_grade),
        int(nacionality),
        int(mother_s_qualification),
        int(father_s_qualification),
        int(mother_s_occupation),
        int(father_s_occupation),
        float(admission_grade),
        int(displaced),
        int(educational_special_needs),
        int(debtor),
        int(tuition_fees_up_to_date),
        int(gender),
        int(scholarship_holder),
        int(age_at_enrollment),
        int(international),
        int(curricular_units_1st_sem_credited),
        int(curricular_units_1st_sem_enrolled),
        int(curricular_units_1st_sem_evaluations),
        int(curricular_units_1st_sem_approved),
        float(curricular_units_1st_sem_grade),
        int(curricular_units_1st_sem_without_evaluations),
        int(curricular_units_2nd_sem_credited),
        int(curricular_units_2nd_sem_enrolled),
        int(curricular_units_2nd_sem_evaluations),
        int(curricular_units_2nd_sem_approved),
        float(curricular_units_2nd_sem_grade),
        int(curricular_units_2nd_sem_without_evaluations),
        float(unemployment_rate),
        float(inflation_rate),
        float(gdp),
        float(approval_rate_1st_sem),
        float(approval_rate_2nd_sem),
        float(grade_diff_sem),
        int(total_approved_units),
    )

    input_sdf = spark.createDataFrame([row_tuple], schema=INPUT_SCHEMA)

    # Aplicar el PipelineModel cargado
    pred_sdf = pipeline_model.transform(input_sdf)
    result_row = pred_sdf.select("prediction", "probability").collect()[0]

    pred_idx = int(result_row["prediction"])
    prob_vector = [float(p) for p in result_row["probability"]]

    # (e) Traducir con el mismo orden de labels del StringIndexer
    predicted_label = labels[pred_idx]
    info = CLASS_TRANSLATIONS.get(
        predicted_label,
        {
            "es": predicted_label,
            "icon": "🎓",
            "card_class": "result-card-enrolled",
            "color": "#3B82F6",
            "desc": "",
        },
    )

    st.markdown("---")
    st.subheader("🎯 Resultado de la Clasificación con Spark ML")

    res_col1, res_col2 = st.columns([1.15, 1.35])

    with res_col1:
        st.markdown(
            f"""
            <div class="{info['card_class']}">
                <div style="font-size: 3.1rem; margin-bottom: 0.3rem;">{info['icon']}</div>
                <div style="font-size: 0.9rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: #475569;">
                    Clase Predicha (Índice StringIndexer: {pred_idx})
                </div>
                <h2 style="margin: 0.4rem 0; color: #0F172A; font-size: 1.75rem;">
                    {info['es']}
                </h2>
                <div style="font-size: 1.25rem; font-weight: 700; color: {info['color']}; margin-bottom: 0.7rem;">
                    Confianza: {prob_vector[pred_idx] * 100:.2f}%
                </div>
                <p style="font-size: 0.95rem; color: #334155; margin: 0;">
                    {info['desc']}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with res_col2:
        prob_records = []
        for idx, lbl in enumerate(labels):
            p_val = prob_vector[idx] if idx < len(prob_vector) else 0.0
            prob_records.append(
                {
                    "Índice (StringIndexer)": idx,
                    "Clase (Target)": lbl,
                    "Descripción en Español": CLASS_TRANSLATIONS.get(lbl, {}).get("es", lbl),
                    "Probabilidad": p_val,
                    "Porcentaje (%)": f"{p_val * 100:.2f}%",
                }
            )
        prob_df = pd.DataFrame(prob_records)

        st.markdown("##### 📊 Distribución de Probabilidades por Clase")
        st.dataframe(
            prob_df[["Índice (StringIndexer)", "Clase (Target)", "Descripción en Español", "Porcentaje (%)"]],
            use_container_width=True,
            hide_index=True,
        )

        fig, ax = plt.subplots(figsize=(6.5, 2.5))
        colors = [CLASS_TRANSLATIONS.get(lbl, {}).get("color", "#3B82F6") for lbl in labels]
        y_labels = [f"{lbl} ({idx})" for idx, lbl in enumerate(labels)]
        pcts = [p * 100 for p in prob_vector]

        bars = ax.barh(y_labels, pcts, color=colors, height=0.55, edgecolor="#1E293B", linewidth=0.8)
        ax.set_xlim(0, 108)
        ax.set_xlabel("Probabilidad (%)")
        ax.invert_yaxis()
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for bar, pct in zip(bars, pcts):
            ax.text(
                bar.get_width() + 1.5,
                bar.get_y() + bar.get_height() / 2,
                f"{pct:.2f}%",
                va="center",
                ha="left",
                fontsize=10,
                fontweight="bold",
                color="#0F172A",
            )

        plt.tight_layout()
        st.pyplot(fig)

# ---------------------------------------------------------------------------
# Pie de página con el nombre del estudiante como autor
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="author-footer">
        Desarrollado por <strong>Juan David Ortiz Pérez</strong> — Autor del Proyecto<br>
        <span>Clasificación de Deserción y Éxito Académico Universitario con Apache Spark ML y Streamlit</span>
    </div>
    """,
    unsafe_allow_html=True,
)
