from index.index import IndexService

def test_rag_complete():
    print("🧪 PROBANDO SISTEMA RAG COMPLETO\n")
    
    # Inicializar servicio (esto carga todos los componentes)
    service = IndexService()
    
    # Probar preguntas sobre Colombia
    questions = [
        "¿Cuál es la capital de Colombia?",
        "¿Quién es el presidente actual?",
        "¿Cómo es el clima en Colombia?",
        "¿Qué sabes sobre la historia de Colombia?"
    ]
    
    for question in questions:
        answer = service.ask_question(question)
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    test_rag_complete()