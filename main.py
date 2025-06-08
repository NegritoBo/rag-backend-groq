from fastapi import FastAPI, Request
from pydantic import BaseModel
import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# === CONFIGURACION ===
BASE_DIR = "/Users/santiagobogero/21_rag_2025_tfn_cncaf"
FAISS_TFN_DIR = os.path.join(BASE_DIR, "index", "faiss_tfn")
FAISS_CNCAF_DIR = os.path.join(BASE_DIR, "index", "faiss_cncaf")
MODEL_NAME = "BAAI/bge-small-en-v1.5"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "TU_API_KEY_AQUI")
GROQ_MODEL = "mistral-7b-instruct"
TOP_K = 2
MAX_CHARS_PER_CHUNK = 1200

# === INICIALIZAR APP ===
app = FastAPI()

# === CARGAR MODELO Y FAISS ===
embedder = SentenceTransformer(MODEL_NAME)
index_tfn = faiss.read_index(os.path.join(FAISS_TFN_DIR, "index.faiss"))
with open(os.path.join(FAISS_TFN_DIR, "index.pkl"), "rb") as f:
    metadata_tfn = pickle.load(f)

index_cncaf = faiss.read_index(os.path.join(FAISS_CNCAF_DIR, "index.faiss"))
with open(os.path.join(FAISS_CNCAF_DIR, "index.pkl"), "rb") as f:
    metadata_cncaf = pickle.load(f)

# === FUNCIONES ===
def buscar_chunks(pregunta, index, metadata):
    vec = embedder.encode([pregunta])
    D, I = index.search(vec, TOP_K)
    resultados = []
    for idx in I[0]:
        if idx < len(metadata):
            resultados.append(metadata[idx])
    return resultados

def construir_prompt(pregunta, contextos):
    partes = []
    for c in contextos:
        fuente = c.get("fuente", "")
        archivo = c.get("archivo", "")
        caratula = c.get("caratula", "")
        resuelve = c.get("resuelve", "")
        exp_papel = c.get("expediente_papel", c.get("expediente_cncaf", ""))
        texto = f"Fuente: {fuente}\nArchivo: {archivo}\nExpediente: {exp_papel}\nCarátula: {caratula}\nResolución: {resuelve[:MAX_CHARS_PER_CHUNK]}"
        partes.append(texto)

    contexto_textual = "\n---\n".join(partes)
    prompt_final = f"Responde la siguiente pregunta utilizando solo la información de los fallos provistos.\n\n{contexto_textual}\n\nPregunta: {pregunta}\nRespuesta:"
    return prompt_final

def consultar_groq(prompt):
    client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1"
    )
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "Sos un asistente jurídico especializado en derecho administrativo, impositivo y aduanero."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error en la consulta a Groq: {e}"

# === MODELO DE DATOS ===
class Pregunta(BaseModel):
    pregunta: str

# === ENDPOINT ===
@app.post("/preguntar")
def endpoint_preguntar(data: Pregunta):
    resultados_tfn = buscar_chunks(data.pregunta, index_tfn, metadata_tfn)
    resultados_cncaf = buscar_chunks(data.pregunta, index_cncaf, metadata_cncaf)
    contextos = resultados_tfn + resultados_cncaf
    prompt = construir_prompt(data.pregunta, contextos)
    respuesta = consultar_groq(prompt)
    return {"respuesta": respuesta}

@app.get("/")
def root():
    return {"status": "RAG backend listo"}
