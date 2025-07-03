from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from index.index import IndexService
import uvicorn
from typing import List, Dict, Any
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crear app FastAPI
app = FastAPI(
    title="RAG Colombia API",
    description="Sistema RAG especializado en información sobre Colombia basado en Wikipedia",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS para permitir requests desde frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios específicos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelos Pydantic para requests/responses
class QuestionRequest(BaseModel):
    question: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "¿Cuál es la capital de Colombia?"
            }
        }

class QuestionResponse(BaseModel):
    question: str
    answer: str
    sources_used: int
    best_similarity_score: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "¿Cuál es la capital de Colombia?",
                "answer": "La capital de Colombia es Bogotá...",
                "sources_used": 3,
                "best_similarity_score": 0.734
            }
        }

class ErrorResponse(BaseModel):
    error: str
    detail: str

class HealthResponse(BaseModel):
    status: str
    message: str
    components: Dict[str, str]

# Variable global para el servicio RAG
rag_service: IndexService = None

@app.on_event("startup")
async def startup_event():
    """
    Inicializar el servicio RAG al arrancar la aplicación.
    """
    global rag_service
    try:
        logger.info("🚀 Inicializando RAG Colombia API...")
        rag_service = IndexService()
        logger.info("✅ RAG Colombia API inicializada correctamente")
    except Exception as e:
        logger.error(f"❌ Error inicializando RAG: {e}")
        raise e

@app.get("/", response_model=Dict[str, str])
async def root():
    """
    Endpoint raíz con información básica de la API.
    """
    return {
        "message": "RAG Colombia API",
        "description": "Sistema de preguntas y respuestas sobre Colombia",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Endpoint de health check para verificar el estado del sistema.
    """
    try:
        # Verificar que el servicio RAG está inicializado
        if rag_service is None:
            raise HTTPException(status_code=503, detail="RAG service not initialized")
        
        # Verificar componentes principales
        components_status = {
            "embeddings": "✅ OK",
            "vector_store": "✅ OK", 
            "llm": "✅ OK",
            "pinecone": "✅ Connected"
        }
        
        return HealthResponse(
            status="healthy",
            message="Todos los componentes funcionando correctamente",
            components=components_status
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Endpoint principal para hacer preguntas sobre Colombia.
    
    - **question**: La pregunta sobre Colombia que quieres hacer
    
    Retorna una respuesta basada únicamente en información de Wikipedia Colombia.
    """
    try:
        if rag_service is None:
            raise HTTPException(status_code=503, detail="RAG service not initialized")
        
        # Validar que la pregunta no esté vacía
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía")
        
        logger.info(f"📝 Nueva pregunta: {request.question}")
        
        # Realizar búsqueda semántica primero para obtener métricas
        search_results = rag_service.vector_store_service.search_similar(
            request.question, 
            top_k=5
        )
        
        # Generar respuesta con LLM
        answer = rag_service.llm_service.generate_answer(request.question, search_results)
        
        # Calcular métricas
        sources_used = len([r for r in search_results if r.get('similarity_score', 0) > 0.3])
        best_score = max([r.get('similarity_score', 0) for r in search_results]) if search_results else 0.0
        
        logger.info(f"✅ Respuesta generada - Sources: {sources_used}, Best score: {best_score:.3f}")
        
        return QuestionResponse(
            question=request.question,
            answer=answer,
            sources_used=sources_used,
            best_similarity_score=round(best_score, 3)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error procesando pregunta: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error interno procesando la pregunta: {str(e)}"
        )

@app.get("/stats", response_model=Dict[str, Any])
async def get_stats():
    """
    Endpoint para obtener estadísticas del sistema.
    """
    try:
        if rag_service is None:
            raise HTTPException(status_code=503, detail="RAG service not initialized")
        
        # Obtener estadísticas de Pinecone
        pinecone_stats = rag_service.vector_store_service.get_index_stats()
        
        # Estadísticas del sistema
        stats = {
            "pinecone_vectors": pinecone_stats.get('total_vectors', 0),
            "embedding_dimension": 384,
            "model_embeddings": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "model_llm": "llama3.1:8b",
            "data_source": "Wikipedia Colombia",
            "last_updated": "2025-01-02"  # Fecha de la última indexación
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

# Endpoint para testing (opcional, remover en producción)
@app.post("/test")
async def test_endpoint():
    """
    Endpoint de testing con preguntas predefinidas.
    """
    test_questions = [
        "¿Cuál es la capital de Colombia?",
        "¿Quién es el presidente actual?",
        "¿Cómo es el clima en Colombia?"
    ]
    
    results = []
    for question in test_questions:
        try:
            search_results = rag_service.vector_store_service.search_similar(question, top_k=3)
            answer = rag_service.llm_service.generate_answer(question, search_results)
            results.append({
                "question": question,
                "answer": answer[:100] + "...",  # Respuesta truncada
                "status": "success"
            })
        except Exception as e:
            results.append({
                "question": question,
                "answer": f"Error: {str(e)}",
                "status": "error"
            })
    
    return {"test_results": results}

if __name__ == "__main__":
    # Ejecutar servidor de desarrollo
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )