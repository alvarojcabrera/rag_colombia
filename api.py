from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from index.index import IndexService  # Cambiar a index si ya reemplazaste los archivos
import uvicorn
from typing import List, Dict, Any
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crear app FastAPI
app = FastAPI(
    title="gAItan - RAG Colombia API",
    description="Sistema RAG especializado en información sobre Colombia basado en Wikipedia",
    version="1.0.0",
    docs_url="/docs",  # Documentación movida a /docs
    redoc_url="/redoc"  # ReDoc movida a /redoc
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelos Pydantic
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

class HealthResponse(BaseModel):
    status: str
    message: str
    components: Dict[str, str]

# Variable global para el servicio RAG
rag_service: IndexService = None

@app.on_event("startup")
async def startup_event():
    """Inicializar el servicio RAG al arrancar la aplicación."""
    global rag_service
    try:
        logger.info("🚀 Inicializando gAItan...")
        rag_service = IndexService()
        logger.info("✅ gAItan inicializado correctamente")
    except Exception as e:
        logger.error(f"❌ Error inicializando gAItan: {e}")
        raise e

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Endpoint de health check."""
    try:
        if rag_service is None:
            raise HTTPException(status_code=503, detail="gAItan no está inicializado")
        
        components_status = {
            "embeddings": "✅ OK",
            "vector_store": "✅ OK", 
            "llm": "✅ OK",
            "pinecone": "✅ Connected"
        }
        
        return HealthResponse(
            status="healthy",
            message="gAItan está funcionando correctamente",
            components=components_status
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"gAItan no está disponible: {str(e)}")

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Endpoint principal para hacer preguntas a gAItan.
    """
    try:
        if rag_service is None:
            raise HTTPException(status_code=503, detail="gAItan no está inicializado")
        
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía")
        
        logger.info(f"📝 Nueva pregunta para gAItan: {request.question}")
        
        # Búsqueda mejorada con más chunks
        search_results = rag_service.vector_store_service.search_similar(
            request.question, 
            top_k=10  # Usando las mejoras implementadas
        )
        
        # Generar respuesta con LLM mejorado
        answer = rag_service.llm_service.generate_answer(request.question, search_results)
        
        # Calcular métricas
        sources_used = len([r for r in search_results if r.get('similarity_score', 0) > 0.25])
        best_score = max([r.get('similarity_score', 0) for r in search_results]) if search_results else 0.0
        
        logger.info(f"✅ gAItan respondió - Sources: {sources_used}, Best score: {best_score:.3f}")
        
        return QuestionResponse(
            question=request.question,
            answer=answer,
            sources_used=sources_used,
            best_similarity_score=round(best_score, 3)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error en gAItan: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error interno en gAItan: {str(e)}"
        )

@app.get("/stats", response_model=Dict[str, Any])
async def get_stats():
    """Endpoint para obtener estadísticas de gAItan."""
    try:
        if rag_service is None:
            raise HTTPException(status_code=503, detail="gAItan no está inicializado")
        
        pinecone_stats = rag_service.vector_store_service.get_index_stats()
        
        stats = {
            "bot_name": "gAItan",
            "pinecone_vectors": pinecone_stats.get('total_vectors', 0),
            "embedding_dimension": 384,
            "model_embeddings": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "model_llm": "gemma3n:e2b",
            "data_source": "Wikipedia Colombia",
            "version": "1.0.0"
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error obteniendo estadísticas: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )