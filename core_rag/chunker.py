import os
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv

ENCODER = tiktoken.get_encoding('cl100k_base')

def texto_a_tokens(text) -> list[int]:
    return ENCODER.encode(text)

def tokens_a_texto(tokens) -> str:
    return ENCODER.decode(tokens)

def make_chunks(texto: str, chunk_size: int = 512, overlap: int = 50):
    
    if overlap >= chunk_size:
        raise ValueError("El overlap no puede ser mayor o igual que el chunk_size")
    
    tokens = texto_a_tokens(texto)
    cantidad_tokens = len(tokens)
    inicio = 0
    chunks = []

    while inicio < cantidad_tokens:
        final = inicio + chunk_size
        lg_chunk_tokens = tokens[inicio:final]
        chunk_texto = tokens_a_texto(lg_chunk_tokens)
        chunks.append(chunk_texto)
        inicio += chunk_size - overlap

        if cantidad_tokens - inicio <= overlap:
            break
    
    return chunks