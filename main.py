import json
from statistics import mean
from core_rag.chunker import make_chunks
from core_rag.embedder import obtener_embedding, obtener_embeddings_batch
from core_rag.vector_store import VectorStore
from core_rag.generator import generar_respuesta
from evaluation.evaluador import modelo_clasificacion, validate_syntax

vector_store = VectorStore()

conocimiento_aws = [
    {
        "texto": "AWS Lambda es un servicio de computación sin servidor que ejecuta tu código en respuesta a eventos. El tiempo de ejecución máximo de una función Lambda (Timeout) es de 15 minutos (900 segundos). Si la función excede este límite, AWS detiene la ejecución inmediatamente lanzando un error de Task Timing Out. La memoria asignada puede variar entre 128 MB y 10,240 MB, y la CPU se escala proporcionalmente a la memoria seleccionada.",
        "metadata": {"servicio": "Lambda", "capa": "Compute"}
    },
    {
        "texto": "Amazon S3 (Simple Storage Service) almacena los datos como objetos dentro de recursos llamados buckets. Por defecto, todos los nuevos buckets de S3 tienen activado el Bloqueo de Acceso Público (Block Public Access) para evitar fugas de información. Para exponer un archivo al mundo de forma segura, se recomienda utilizar URLs firmadas (Presigned URLs) con un tiempo de expiración limitado, en lugar de hacer el bucket completamente público.",
        "metadata": {"servicio": "S3", "capa": "Storage"}
    },
    {
        "texto": "Amazon DynamoDB es una base de datos NoSQL clave-valor totalmente administrada que ofrece dos modos de facturación: el modo bajo demanda (On-Demand) y el modo provisionado (Provisioned). El modo On-Demand cobra por cada solicitud de lectura y escritura exacta que realiza tu aplicación, siendo ideal para cargas de trabajo impredecibles. El modo Provisioned requiere que especifiques las Unidades de Capacidad de Lectura (RCU) y Escritura (WCU) con anticipación.",
        "metadata": {"servicio": "DynamoDB", "capa": "Database"}
    },
    {
        "texto": "AWS IAM (Identity and Access Management) permite administrar el acceso a los recursos de AWS de forma segura. Una política de IAM (IAM Policy) es un documento JSON que define formalmente los permisos. Un Rol de IAM (IAM Role) no tiene credenciales a largo plazo (como contraseñas o claves de acceso); en su lugar, proporciona credenciales de seguridad temporales mediante la API AssumeRole que expira automáticamente después de un tiempo configurado.",
        "metadata": {"servicio": "IAM", "capa": "Security"}
    }
]

def poblar_vector_store():
    chunks_totales = []
    textos_para_embeddings = []

    for item in conocimiento_aws:
        lista_chunks = make_chunks(item['texto'], chunk_size=40, overlap=10)
        for chunk in lista_chunks:
            chunks_totales.append({"texto": chunk, "metadata": item['metadata']})
            textos_para_embeddings.append(chunk)

    vectores = obtener_embeddings_batch(textos_para_embeddings)

    for chunk_data, v in zip(chunks_totales, vectores):
        vector_store.agregar(chunk_data['texto'], v, chunk_data['metadata'])

def run_test_case(test_case):
    pregunta = test_case["tarea"]

    vector_query = obtener_embedding(pregunta)
    
    filtros = {"servicio": test_case["servicio"]} if "servicio" in test_case else {}
    chunks_recuperados = vector_store.buscar(vector_query, top_k=2, filtros=filtros)
    
    output_rag = generar_respuesta(pregunta, chunks_recuperados)

    calidad_modelo = modelo_clasificacion(test_case, output_rag)
    puntuacion_modelo = calidad_modelo['score']
    reasoning = calidad_modelo['reasoning']
    
    puntuacion_syntax = validate_syntax(test_case, output_rag)

    score_final = (puntuacion_syntax + puntuacion_modelo) / 2

    return {
        "tarea": pregunta,
        "output_rag": output_rag,
        "score": score_final,
        "reasoning": reasoning,
    }

def run_eval():
    poblar_vector_store()

    with open('evaluation/dataset.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    results = []

    for test_case in dataset:
        result = run_test_case(test_case)
        results.append(result)
    
    puntuacion_media = mean([res["score"] for res in results])
    print(f"\n[+] Puntuación Media del RAG: {puntuacion_media}/10\n")

    return results

if __name__ == "__main__":
    informe_final = run_eval()
    with open('resultado_evaluacion.json', 'w', encoding='utf-8') as f:
        json.dump(informe_final, f, indent=2, ensure_ascii=False)