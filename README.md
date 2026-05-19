## RAG Evaluation

Un sistema modular de Recuperación Aumentada por Generación (RAG) enfocado en arquitectura de AWS, desarrollado desde cero (sin frameworks como LangChain o LlamaIndex) e integrado con un pipeline de evaluación automatizado utilizando Claude (Anthropic) como juez semántico junto con validadores sintácticos de código.

## Arquitectura del Proyecto

El repositorio está estructurado de forma modular separando el motor RAG de la lógica de evaluación:

```text
├── core_rag/              # Módulo 1: Motor del RAG nativo
│   ├── chunker.py         # Segmentación de texto por tokens (tiktoken)
│   ├── embedder.py        # Conexión con OpenAI Embeddings
│   ├── generator.py       # Generación de respuestas (GPT-4o-mini)
│   └── vector_store.py    # Base de datos vectorial indexada en memoria (Numpy)
├── evaluation/            # Módulo 2: Pipeline de pruebas y validación
│   ├── evaluador.py       # Juez semántico (Claude) y validadores sintácticos
│   └── dataset.json       # Set de datos de prueba estructurado
├── main.py                # Orquestador principal del proyecto
└── requirements.txt       # Dependencias del proyecto
```

## Características Principales
RAG End-to-End Nativo: Implementación de chunking con overlap basado en tokens de tiktoken y cálculo de similitud de cosenos por producto punto mediante numpy.

## Evaluación Híbrida
Métrica Semántica: Claude actúa como juez de calidad técnica puntuando precisión de 1 a 10.
Métrica Sintáctica: Validadores nativos en Python empleando análisis de árboles sintácticos abstractos (ast), decodificación estructurada (json) y compilación de expresiones regulares (re).

## Aislamiento de Entornos
Configuración modular limpia mediante paquetes locales (__init__.py)

1. Instalación y uso
git clone https://github.com/HarryAlanCQ/rag_evaluation
cd rag_evaluation

2. Instalar dependencias
pip install -r requirements.txt

3. Configurar variables de entorno
Crea un archivo .env en la raíz del proyecto con tus credenciales

OPENAI_API_KEY=tuapikey
ANTHROPIC_API_KEY=tuapikey

4. Ejecutar pruebas

python main.py

Al finalizar la ejecución, el sistema imprimirá las métricas generales en consola y exportará un reporte detallado con el veredicto del juez en resultado_evaluacion.json.

