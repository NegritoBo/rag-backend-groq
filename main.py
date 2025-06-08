from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/")
def root():
    return {"status": "RAG backend listo"}

@app.post("/preguntar")
async def preguntar(request: Request):
    data = await request.json()
    pregunta = data.get("pregunta", "")
    # Acá iría la lógica de búsqueda real, por ahora devolvemos texto simulado
    return JSONResponse(content={"respuesta": f"Procesando tu pregunta: {pregunta}"})
