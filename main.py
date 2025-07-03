from index.index import IndexService
from langchain_core.documents import Document
import os
from dotenv import load_dotenv

def main():
    """Función principal OPTIMIZADA que sigue la documentación de LangChain."""
    load_dotenv()
    
    required_keys = ["PINECONE_API_KEY", "FIRECRAWL_API_KEY"]
    for key in required_keys:
        if not os.getenv(key):
            print(f"❌ {key} no está configurada en el archivo .env")
            return
    
    try:
        print("🚀 Iniciando sistema RAG optimizado para Colombia...")
        print("🔧 Usando la documentación oficial de LangChain")
        
        index_service = IndexService(
            embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        print("\n📋 Pipeline completo desde URL")
        url = "https://es.wikipedia.org/wiki/Colombia"
        success = index_service.index_pipeline(url=url)
        
        if not success:
            print("❌ Error en pipeline principal")
            return
        
        print("\n📋 Indexando documentos adicionales")
        additional_docs = [
            Document(
                page_content="Gabriel García Márquez fue un escritor colombiano ganador del Premio Nobel de Literatura en 1982.",
                metadata={"source": "manual", "topic": "literatura"}
            ),
            Document(
                page_content="El café colombiano es reconocido mundialmente por su calidad excepcional.",
                metadata={"source": "manual", "topic": "agricultura"}
            ),
            Document(
                page_content="Cartagena de Indias es una ciudad histórica en la costa caribeña de Colombia.",
                metadata={"source": "manual", "topic": "turismo"}
            )
        ]
        
        success = index_service.index_documents(additional_docs)
        if success:
            print("✅ Documentos adicionales indexados")
        
        print("\n📊 ESTADÍSTICAS FINALES:")
        stats = index_service.get_stats()
        if 'error' not in stats:
            print(f"   📈 Total de vectores: {stats['total_vectors']}")
            print(f"   📏 Dimensión: {stats['dimension']}")
        
        print("\n🔍 PROBANDO BÚSQUEDAS SEMÁNTICAS:")
        test_queries = [
            "¿Cuál es la capital de Colombia?",
            "Escritores colombianos famosos",
            "Turismo en Colombia"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Consulta: '{query}'")
            results = index_service.search(query, top_k=2)
            
            if results:
                for i, result in enumerate(results, 1):
                    print(f"   {i}. Score: {result['similarity_score']:.3f}")
                    print(f"      {result['content'][:100]}...")
            else:
                print("   ❌ No se encontraron resultados")
        
        print("\n✅ PROCESO COMPLETADO EXITOSAMENTE!")
        print("🎯 Resumen del proceso optimizado:")
        print("   ✅ Pipeline simplificado")
        print("   ✅ Sigue documentación oficial de LangChain")
        print("   ✅ Embeddings automáticos")
        print("   ✅ Sistema RAG completo operativo")
        
    except Exception as e:
        print(f"❌ Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
