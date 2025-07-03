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

# HTML del frontend integrado
FRONTEND_HTML = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>gAItan - Chat Bot Colombia</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }

        .chat-container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            width: 100%;
            max-width: 800px;
            height: 600px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .chat-header {
            background: linear-gradient(90deg, #fcf304, #fce803, #fcb103);
            color: #333;
            padding: 20px;
            text-align: center;
            border-bottom: 2px solid #ddd;
            position: relative;
        }

        .chat-header h1 {
            margin-bottom: 5px;
            font-size: 24px;
        }

        .chat-header p {
            font-size: 14px;
            opacity: 0.8;
        }

        .docs-link {
            position: absolute;
            top: 10px;
            right: 15px;
            background: rgba(255,255,255,0.3);
            padding: 5px 10px;
            border-radius: 15px;
            text-decoration: none;
            color: #333;
            font-size: 12px;
            transition: background 0.3s;
        }

        .docs-link:hover {
            background: rgba(255,255,255,0.5);
        }

        .chat-messages {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            background: #f8f9fa;
        }

        .message {
            margin-bottom: 15px;
            display: flex;
            animation: fadeIn 0.3s ease-in;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .message.user {
            justify-content: flex-end;
        }

        .message.bot {
            justify-content: flex-start;
        }

        .message-bubble {
            max-width: 70%;
            padding: 12px 18px;
            border-radius: 18px;
            word-wrap: break-word;
            line-height: 1.4;
        }

        .message.user .message-bubble {
            background: #007bff;
            color: white;
            border-bottom-right-radius: 5px;
        }

        .message.bot .message-bubble {
            background: white;
            color: #333;
            border: 1px solid #ddd;
            border-bottom-left-radius: 5px;
        }

        .chat-input-container {
            padding: 20px;
            background: white;
            border-top: 1px solid #ddd;
            display: flex;
            gap: 10px;
        }

        .chat-input {
            flex: 1;
            padding: 12px 18px;
            border: 2px solid #ddd;
            border-radius: 25px;
            font-size: 16px;
            outline: none;
            transition: border-color 0.3s;
        }

        .chat-input:focus {
            border-color: #007bff;
        }

        .send-button {
            padding: 12px 24px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            transition: background 0.3s;
            min-width: 80px;
        }

        .send-button:hover:not(:disabled) {
            background: #0056b3;
        }

        .send-button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }

        .typing-indicator {
            display: none;
            align-items: center;
            gap: 10px;
            padding: 12px 18px;
            background: white;
            border: 1px solid #ddd;
            border-radius: 18px;
            border-bottom-left-radius: 5px;
            max-width: 70%;
        }

        .typing-dots {
            display: flex;
            gap: 4px;
        }

        .typing-dots span {
            width: 8px;
            height: 8px;
            background: #999;
            border-radius: 50%;
            animation: typing 1.4s infinite ease-in-out;
        }

        .typing-dots span:nth-child(1) { animation-delay: 0s; }
        .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
        .typing-dots span:nth-child(3) { animation-delay: 0.4s; }

        @keyframes typing {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-10px); }
        }

        .status-indicator {
            padding: 10px;
            text-align: center;
            background: #f8f9fa;
            border-top: 1px solid #ddd;
            font-size: 12px;
            color: #666;
        }

        .status-indicator.online {
            background: #d4edda;
            color: #155724;
        }

        .status-indicator.offline {
            background: #f8d7da;
            color: #721c24;
        }

        /* Scrollbar personalizado */
        .chat-messages::-webkit-scrollbar {
            width: 6px;
        }

        .chat-messages::-webkit-scrollbar-track {
            background: #f1f1f1;
        }

        .chat-messages::-webkit-scrollbar-thumb {
            background: #c1c1c1;
            border-radius: 3px;
        }

        .chat-messages::-webkit-scrollbar-thumb:hover {
            background: #a8a8a8;
        }

        /* Responsive */
        @media (max-width: 600px) {
            .chat-container {
                height: 100vh;
                border-radius: 0;
            }
            
            .message-bubble {
                max-width: 85%;
            }
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <a href="/docs" class="docs-link">📚 API Docs</a>
            <h1>🇨🇴 gAItan</h1>
            <p>Tu asistente de inteligencia artificial especializado en Colombia</p>
        </div>
        
        <div class="chat-messages" id="messages">
            <div class="message bot">
                <div class="message-bubble">
                    ¡Hola! 👋 Soy <strong>gAItan</strong>, tu asistente de inteligencia artificial especializado en Colombia. Puedo responder preguntas sobre historia, geografía, cultura, política y mucho más sobre nuestro país basándome en información actualizada. ¿En qué puedo ayudarte hoy?
                </div>
            </div>
        </div>
        
        <div class="chat-input-container">
            <input 
                type="text" 
                class="chat-input" 
                id="messageInput" 
                placeholder="Pregúntale a gAItan sobre Colombia..."
                maxlength="500"
            >
            <button class="send-button" id="sendButton" onclick="sendMessage()">
                Enviar
            </button>
        </div>
        
        <div class="status-indicator" id="status">
            🔗 Conectando...
        </div>
    </div>

    <script>
        // Configuración - API en el mismo servidor
        const API_URL = window.location.origin;
        let isLoading = false;

        // Referencias DOM
        const messagesContainer = document.getElementById('messages');
        const messageInput = document.getElementById('messageInput');
        const sendButton = document.getElementById('sendButton');
        const statusIndicator = document.getElementById('status');

        // Verificar conexión al cargar
        checkAPIConnection();

        // Event listeners
        messageInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !isLoading) {
                sendMessage();
            }
        });

        // Función para verificar conexión con la API
        async function checkAPIConnection() {
            try {
                const response = await fetch(`${API_URL}/health`);
                if (response.ok) {
                    updateStatus('🟢 Conectado - Listo para chatear', 'online');
                } else {
                    updateStatus('🔴 API no disponible', 'offline');
                }
            } catch (error) {
                updateStatus('🔴 No se puede conectar al servidor', 'offline');
            }
        }

        // Función para actualizar estado
        function updateStatus(message, type) {
            statusIndicator.textContent = message;
            statusIndicator.className = `status-indicator ${type}`;
        }

        // Función para enviar mensaje
        async function sendMessage() {
            const message = messageInput.value.trim();
            
            if (!message || isLoading) return;

            // Agregar mensaje del usuario
            addMessage(message, 'user');
            messageInput.value = '';
            
            // Mostrar indicador de escritura
            showTypingIndicator();
            setLoading(true);

            try {
                // Enviar a la API
                const response = await fetch(`${API_URL}/ask`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ question: message })
                });

                if (!response.ok) {
                    throw new Error(`Error ${response.status}: ${response.statusText}`);
                }

                const data = await response.json();
                
                // Ocultar indicador de escritura
                hideTypingIndicator();
                
                // Agregar respuesta del bot
                addMessage(data.answer, 'bot');
                
                // Actualizar estado
                updateStatus(`🟢 Conectado - Fuentes: ${data.sources_used} (Score: ${data.best_similarity_score})`, 'online');

            } catch (error) {
                hideTypingIndicator();
                addMessage('Lo siento, ocurrió un error. Por favor intenta de nuevo.', 'bot');
                updateStatus('🔴 Error de conexión', 'offline');
                console.error('Error:', error);
            }

            setLoading(false);
        }

        // Función para agregar mensaje al chat
        function addMessage(text, sender) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}`;
            
            const bubble = document.createElement('div');
            bubble.className = 'message-bubble';
            bubble.innerHTML = text; // Usar innerHTML para permitir HTML básico
            
            messageDiv.appendChild(bubble);
            messagesContainer.appendChild(messageDiv);
            
            // Scroll al final
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        // Función para mostrar indicador de escritura
        function showTypingIndicator() {
            const typingDiv = document.createElement('div');
            typingDiv.className = 'message bot';
            typingDiv.id = 'typing-indicator';
            
            const typingBubble = document.createElement('div');
            typingBubble.className = 'typing-indicator';
            typingBubble.style.display = 'flex';
            typingBubble.innerHTML = `
                <span>gAItan está pensando</span>
                <div class="typing-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            `;
            
            typingDiv.appendChild(typingBubble);
            messagesContainer.appendChild(typingDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        // Función para ocultar indicador de escritura
        function hideTypingIndicator() {
            const typingIndicator = document.getElementById('typing-indicator');
            if (typingIndicator) {
                typingIndicator.remove();
            }
        }

        // Función para manejar estado de carga
        function setLoading(loading) {
            isLoading = loading;
            sendButton.disabled = loading;
            sendButton.textContent = loading ? '...' : 'Enviar';
            messageInput.disabled = loading;
        }

        // Verificar conexión cada 30 segundos
        setInterval(checkAPIConnection, 30000);
    </script>
</body>
</html>
"""

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

@app.get("/", response_class=HTMLResponse)
async def frontend():
    """
    Endpoint principal que sirve el frontend de gAItan.
    """
    return HTMLResponse(content=FRONTEND_HTML)

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
            "model_llm": "llama3.1:8b",
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