import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

cliente = OpenAI()

def generar_respuesta(pregunta: str, chunks: list[dict]) -> str:
    contexto_lista = []
    for i, chunk in enumerate(chunks, start=1):
        texto_chunk = chunk["texto"]
        contexto_lista.append(f"[Fragmento {i}]: {texto_chunk}")

    contexto = "\n\n".join(contexto_lista)

    prompt_sistema = (
        "Eres un asistente experto. Responde a la pregunta del usuario utilizando "
        "únicamente el contexto proporcionado. Si la respuesta no está en el contexto, "
        "di amablemente que no posees esa información. Sé directo y breve."
    )

    prompt_usuario = (
        f"Contexto:\n{contexto}\n"
        f"Pregunta: {pregunta}"
    )

    respuesta = cliente.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system", "content": prompt_sistema},
            {"role":"user", "content": prompt_usuario}
        ],
        temperature=0 #para respuestas sin creatividad y basadas solo en el contexto (deterministas).   
    )

    return respuesta.choices[0].message.content