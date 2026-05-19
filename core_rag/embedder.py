import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

cliente = OpenAI()

EMBEDDER_MODEL = "text-embedding-3-small" #1536 dimensiones

def obtener_embedding(texto: str) -> list[float]:
    if not texto.strip():
        raise ValueError('El texto proporcionado para el embedding está vacío.')
    response = cliente.embeddings.create(
        model=EMBEDDER_MODEL,
        input=texto,
    )

    return response.data[0].embedding

def obtener_embeddings_batch(textos: list[str]) -> list[list[float]]:
    if not textos:
        return []
    response = cliente.embeddings.create(
        model=EMBEDDER_MODEL,
        input=textos,
    )

    return [item.embedding for item in response.data]
