from anthropic import Anthropic
from dotenv import load_dotenv
from statistics import mean
import re
import ast
import json

load_dotenv()

cliente = Anthropic()
model = 'claude-haiku-4-5'

def add_mensaje_usuario(messages: list, text):
    message = {'role':'user','content':text}
    messages.append(message)

def add_mensaje_asistente(messages: list, text):
    message = {'role':'assistant','content':text}
    messages.append(message)

def chat(messages, system=None, temperature=1.0, stop_sequences = []):
    params = {
        "model":model,
        "max_tokens":1000,
        "messages":messages,
        "temperature":temperature,
    }

    if system:
        params['system'] = system

    if stop_sequences:
        params['stop_sequences'] = stop_sequences
    
    message = cliente.messages.create(**params)
    return message.content[0].text

def modelo_clasificacion(test_case, output):
    eval_prompt = f"""
Eres un experto en administración de servidores Linux. Tu trabajo es evaluar la siguiente tarea:
<tarea>
{test_case['tarea']}
</tarea>

Solución para evaluar:
<solution>
{output}
</solution>

Debes evaluar las siguientes propiedades y estructurarlas estrictamente en un JSON:
- "strengths": Un array de 1 o 3 fortalezas.
- "reasoning": Una explicación de tu valoración.
- "score": Un número entero del 1 al 10.

Responde únicamente en formato JSON, sin texto aclaratorio.
"""
    
    messages = []
    add_mensaje_usuario(messages, eval_prompt)
    add_mensaje_asistente(messages, "```json")
    eval_text = chat(messages, stop_sequences=["```"])
    
    json_limpio = eval_text.replace('```json', '').replace('```', '').strip()
    return json.loads(json_limpio)

def python_syntax(text):
    try:
        ast.parse(text.strip())
        return 10
    except SyntaxError:
        return 0
    
def json_syntax(text):
    try:
        json.loads(text.strip())
        return 10
    except json.JSONDecodeError:
        return 0

def regex_syntax(text):
    try:
        re.compile(text.strip())
        return 10
    except re.error:
        return 0

def validate_syntax(test_case, response):
    formato = test_case['formato']
    if formato == "python":
        return python_syntax(response)
    elif formato == "json":
        return json_syntax(response)
    else:
        return regex_syntax(response)

