import numpy as np
import json
from pathlib import Path

class VectorStore:
    def __init__(self):
        self.entradas : list[dict] = []

    def agregar(self, texto: str, vector: list[float], metadata: dict = {}):
        siguiente_id = max([e["id"] for e in self.entradas]) + 1 if self.entradas else 0
        entrada = {
            "id": siguiente_id,
            "texto": texto,
            "vector": vector,
            "metadata": metadata
        }

        self.entradas.append(entrada)
    
    def buscar(self, vector_query: list[float], top_k: int = 3, filtros: dict = {}):
        if not self.entradas:
            return []

        if filtros:
            candidatos = [
                e for e in self.entradas
                if all(e["metadata"].get(k) == v for k, v in filtros.items())
            ]
        else:
            candidatos = self.entradas

        if not candidatos:
            return []

        q = np.array(vector_query)

        resultados = []
        for entrada in candidatos:
            v = np.array(entrada["vector"])
            similitud = float(np.dot(q,v))
            resultados.append({
                "id":        entrada["id"],
                "texto":     entrada["texto"],
                "metadata":  entrada["metadata"],
                "similitud": similitud,
            })

        resultados.sort(key=lambda x: x["similitud"], reverse=True) #ordenar de mayor a menor similitud
        return resultados[:top_k]

    def guardar(self, ruta: str):
        Path(ruta).write_text(
            json.dumps(self.entradas, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f'Store guardada en {ruta} ({len(self.entradas)} entradas)')
    
    def cargar(self, ruta:str):
        if not Path(ruta).exists():
            print(f"Advertencia: No se encontró ningún archivo en {ruta}. Iniciando base de datos vacía.")
            self.entradas = []
            return
        self.entradas = json.loads(Path(ruta).read_text(encoding="utf-8"))
        print(f"Store cargada desde {ruta} ({len(self.entradas)} entradas)")