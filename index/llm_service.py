from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from typing import List, Dict, Any
from dotenv import load_dotenv
import os

class LLMService:
    """
    Servicio MEJORADO para generar respuestas usando Ollama + LLM.
    
    MEJORAS IMPLEMENTADAS:
    - Usa más chunks relevantes (hasta 10 en lugar de 5)
    - Filtra mejor los chunks por similarity score
    - No menciona "Fragmento X" en las respuestas
    - Mejor contexto y respuestas más naturales
    """

    llm: BaseChatModel
    
    def __init__(self, model_name: str = "gemma3n:e2b"):
        load_dotenv()
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        print(f"🤖 Inicializando LLM mejorado: {model_name}")
        if openrouter_api_key is None:
            print("API Key de OpenRouter no encontrada. Usando modelo local")
            self.llm = self.get_local_llm(model=model_name)
        else:
            print("API Key de OpenRouter encontrada. Usando modelo hosteado")
            self.llm = self.get_hosted_llm(openrouter_api_key)
        print("✅ LLM mejorado inicializado correctamente")

    def get_local_llm(self, model_name: str = "gemma3n:e2b"):
        return ChatOllama(model=model_name)
    
    def get_hosted_llm(self, openrouter_api_key):
        # Usa un modelo hosteado en OpenRouter, compatible con la api de openai
        return ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
            model="google/gemma-3n-e4b-it:free"
        )
    
    def filter_relevant_chunks(self, search_results: List[Dict[str, Any]], min_score: float = 0.25) -> List[Dict[str, Any]]:
        """
        Filtra chunks por similarity score y los ordena por relevancia.
        
        Args:
            search_results: Resultados de búsqueda semántica
            min_score: Score mínimo para considerar relevante (0.25 = más inclusivo)
        """
        if not search_results:
            return []
        
        # Filtrar por score mínimo
        relevant_chunks = [
            chunk for chunk in search_results 
            if chunk.get('similarity_score', 0) >= min_score
        ]
        
        # Ordenar por score descendente (ya deberían estar ordenados, pero por seguridad)
        relevant_chunks.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        
        print(f"📊 Chunks filtrados: {len(relevant_chunks)}/{len(search_results)} sobre threshold {min_score}")
        
        return relevant_chunks
    
    def create_enhanced_context(self, relevant_chunks: List[Dict[str, Any]]) -> str:
        """
        Crea un contexto mejorado sin mencionar "Fragmento X".
        Combina información relacionada de forma más natural.
        """
        if not relevant_chunks:
            return ""
        
        # Agrupar por headers similares si existen
        context_parts = []
        
        for i, chunk in enumerate(relevant_chunks):
            content = chunk.get('content', '').strip()
            
            # Limpiar contenido
            if content:
                # Limitar tamaño pero no cortar bruscamente
                if len(content) > 800:
                    # Buscar un punto para cortar naturalmente
                    truncated = content[:800]
                    last_period = truncated.rfind('.')
                    last_space = truncated.rfind(' ')
                    
                    if last_period > 600:  # Si hay un punto cerca del final
                        content = content[:last_period + 1]
                    elif last_space > 600:  # Si no, cortar en espacio
                        content = content[:last_space] + "..."
                    else:
                        content = truncated + "..."
                
                context_parts.append(content)
        
        # Unir todo el contexto de forma natural
        context = "\n\n".join(context_parts)
        
        print(f"📝 Contexto creado con {len(context_parts)} secciones relevantes")
        return context
    
    def create_enhanced_prompt(self, query: str, context: str) -> str:
        """
        Crea un prompt mejorado que genera respuestas más naturales.
        """
        prompt = f"""Eres un experto en Colombia con acceso a información actualizada de Wikipedia. Tu trabajo es responder preguntas sobre Colombia de manera clara, precisa y natural.

INSTRUCCIONES IMPORTANTES:
1. SOLO responde preguntas relacionadas con Colombia
2. Usa ÚNICAMENTE la información proporcionada abajo
3. Responde de forma natural y conversacional (NO menciones "fragmentos" ni "según el fragmento X")
4. Si la pregunta no es sobre Colombia, responde: "Solo puedo responder preguntas sobre Colombia. ¿Hay algo específico sobre Colombia que te gustaría saber?"
5. Si no hay información suficiente, responde: "No tengo información suficiente para responder esa pregunta específica sobre Colombia."
6. Sé preciso pero amigable en tus respuestas

INFORMACIÓN DISPONIBLE SOBRE COLOMBIA:
{context}

PREGUNTA: {query}

RESPUESTA (natural y directa):"""
        
        return prompt
    
    def generate_answer(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """
        Genera una respuesta MEJORADA basada en la query y más chunks relevantes.
        
        MEJORAS:
        - Usa más chunks (hasta 10 en lugar de 5)
        - Mejor filtrado por similarity score
        - Respuestas más naturales sin mencionar "fragmentos"
        """
        try:
            # Si no hay resultados de búsqueda
            if not search_results:
                return "No encontré información relevante sobre Colombia para responder tu pregunta. ¿Podrías ser más específico?"
            
            # Verificar calidad general de los resultados
            best_score = max(r.get('similarity_score', 0) for r in search_results)
            
            print(f"🔍 Analizando {len(search_results)} chunks encontrados")
            print(f"📊 Mejor score de similitud: {best_score:.3f}")
            
            # Si el mejor score es muy bajo, probablemente no es sobre Colombia
            if best_score < 0.3:
                return "Tu pregunta no parece estar relacionada con la información que tengo sobre Colombia. Solo puedo responder preguntas sobre Colombia basándome en Wikipedia."
            
            # Filtrar chunks relevantes con threshold más bajo para incluir más información
            relevant_chunks = self.filter_relevant_chunks(search_results, min_score=0.25)
            
            if not relevant_chunks:
                return "No encontré información suficientemente relevante sobre Colombia para responder tu pregunta específica."
            
            # Crear contexto mejorado
            context = self.create_enhanced_context(relevant_chunks)
            
            if not context.strip():
                return "No pude extraer información útil para responder tu pregunta sobre Colombia."
            
            # Crear prompt mejorado
            prompt = self.create_enhanced_prompt(query, context)
            
            print(f"🧠 Generando respuesta con {len(relevant_chunks)} chunks relevantes...")
            print(f"📏 Tamaño del contexto: {len(context)} caracteres")
            
            # Generar respuesta
            response = str(self.llm.invoke(prompt).content)
            answer = response.strip()
            
            # Validación adicional de la respuesta
            if not answer or len(answer) < 20:
                return "No pude generar una respuesta apropiada. Por favor, reformula tu pregunta sobre Colombia."
            
            print("✅ Respuesta mejorada generada exitosamente")
            return answer
            
        except Exception as e:
            print(f"❌ Error generando respuesta: {e}")
            return "Lo siento, hubo un error procesando tu pregunta sobre Colombia. Por favor intenta de nuevo."
    
    def test_simple(self) -> str:
        """
        Método para probar que el LLM funciona básicamente.
        """
        try:
            response = self.llm.invoke("Di 'Hola' en una palabra").content
            return response.strip()
        except Exception as e:
            return f"Error: {str(e)}"
    
    def ask_question(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """
        Método principal para hacer preguntas al sistema RAG.
        Utiliza la nueva lógica mejorada.
        """
        return self.generate_answer(query, search_results)