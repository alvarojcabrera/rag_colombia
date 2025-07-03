from langchain_ollama import OllamaLLM
from typing import List, Dict, Any

class LLMService:
    """
    Servicio para generar respuestas usando Ollama + LLM.
    Toma los chunks encontrados por la búsqueda semántica y genera respuestas coherentes.
    SOLO responde preguntas sobre Colombia basándose en información de Wikipedia.
    """
    
    def __init__(self, model_name: str = "llama3.1:8b"):
        print(f"🤖 Inicializando LLM: {model_name}")
        self.llm = OllamaLLM(model=model_name)
        print("✅ LLM inicializado correctamente")
    
    def create_prompt(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Crea el prompt combinando la pregunta del usuario con el contexto encontrado.
        Incluye validaciones estrictas para solo responder sobre Colombia.
        """
        # Construir el contexto con los chunks más relevantes
        context_parts = []
        for i, chunk in enumerate(chunks[:3]):  # Solo los 3 más relevantes
            content = chunk['content'][:500]  # Limitar tamaño
            context_parts.append(f"Fragmento {i+1}: {content}")
        
        context = "\n\n".join(context_parts)
        
        # Crear el prompt con validaciones estrictas
        prompt = f"""Eres un asistente especializado ÚNICAMENTE en Colombia. Tu base de datos contiene SOLAMENTE información de Wikipedia sobre Colombia.

REGLAS ESTRICTAS QUE DEBES SEGUIR:
1. SOLO responde preguntas relacionadas con Colombia (geografía, historia, cultura, política, economía, etc.)
2. SOLO usa la información proporcionada en los fragmentos de Wikipedia Colombia
3. Si la pregunta NO parece estar relacionada con Colombia, responde exactamente: "Solo puedo responder preguntas sobre Colombia. ¿Hay algo específico sobre Colombia que te gustaría saber?"
4. Si la pregunta ES sobre Colombia pero NO hay información suficiente en los fragmentos, responde: "No tengo información suficiente en mi base de datos sobre Colombia para responder esa pregunta específica."
5. NUNCA inventes información que no esté en los fragmentos proporcionados
6. Responde en español de forma clara y concisa

INFORMACIÓN DISPONIBLE DE COLOMBIA:
{context}

PREGUNTA DEL USUARIO: {query}

ANÁLISIS Y RESPUESTA:"""
        
        return prompt
    
    def generate_answer(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """
        Genera una respuesta basada en la query y los resultados de búsqueda.
        Incluye validaciones inteligentes basadas en similarity scores.
        """
        # Si no hay resultados de búsqueda
        if not search_results:
            return "No encontré información relevante sobre Colombia para responder tu pregunta. ¿Podrías ser más específico?"
        
        # Verificar la calidad de los resultados basándose en similarity scores
        best_score = max(r.get('similarity_score', 0) for r in search_results)
        
        # Si el mejor score es muy bajo, probablemente no es sobre Colombia
        if best_score < 0.4:
            return "Tu pregunta no parece estar relacionada con la información que tengo sobre Colombia. Solo puedo responder preguntas sobre Colombia basándome en Wikipedia."
        
        # Filtrar solo resultados con score decente
        relevant_results = [r for r in search_results if r.get('similarity_score', 0) > 0.3]
        
        if not relevant_results:
            return "No encontré información suficientemente relevante sobre Colombia para responder tu pregunta específica."
        
        # Crear el prompt mejorado
        prompt = self.create_prompt(query, relevant_results)
        
        print(f"🧠 Generando respuesta para: '{query[:50]}...'")
        print(f"📊 Mejor score de similitud: {best_score:.3f}")
        
        try:
            # Generar respuesta
            response = self.llm.invoke(prompt)
            print("✅ Respuesta generada exitosamente")
            return response.strip()
            
        except Exception as e:
            print(f"❌ Error generando respuesta: {e}")
            return "Lo siento, hubo un error procesando tu pregunta sobre Colombia. Por favor intenta de nuevo."
    
    def test_simple(self) -> str:
        """
        Método para probar que el LLM funciona básicamente.
        """
        try:
            response = self.llm.invoke("Di 'Hola' en una palabra")
            return response.strip()
        except Exception as e:
            return f"Error: {str(e)}"
    
    def ask_question(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """
        Método principal para hacer preguntas al sistema RAG.
        Utiliza la validación inteligente basada en similarity scores.
        """
        return self.generate_answer(query, search_results)