from index.extractor import Extractor
from index.index import IndexService
from index.splitter import Splitter

def main():
    """
    Función principal que ejecuta el pipeline completo de RAG:
    1. Extracción y limpieza de contenido
    2. Splitting y chunking
    3. Generación de embeddings
    """
    try:
        print("🚀 Iniciando sistema RAG para Colombia...")
        
        # Crear el servicio de indexación
        index_service = IndexService()
        
        # Ejecutar el pipeline completo
        embedded_chunks = index_service.index_pipeline()
        
        # Verificar si el pipeline fue exitoso
        if embedded_chunks is None:
            print("❌ El pipeline retornó None - revisar errores arriba")
            return None
        
        # Mostrar resumen final
        print(f"\n📊 RESUMEN FINAL:")
        print(f"   - Total chunks procesados: {len(embedded_chunks)}")
        print(f"   - Dimensión embeddings: 384")
        print(f"   - Archivos generados:")
        print(f"     • colombia.md (contenido original)")
        print(f"     • colombia_clean.md (contenido limpio)")
        print(f"     • embeddings_cache.json (cache de embeddings)")
        
        # Mostrar algunos ejemplos de chunks
        print(f"\n🔍 EJEMPLOS DE CHUNKS PROCESADOS:")
        for i, chunk in enumerate(embedded_chunks[:3]):
            print(f"\n   Chunk {i+1}:")
            print(f"   ID: {chunk['chunk_id']}")
            print(f"   Metadata: {chunk['metadata']}")
            print(f"   Contenido: {chunk['content'][:150]}...")
            print(f"   Embedding: [{chunk['embedding'][0]:.4f}, {chunk['embedding'][1]:.4f}, ...] (384 dims)")
        
        print(f"\n✅ Pipeline completado exitosamente!")
        print(f"🎯 Sistema listo para búsquedas semánticas")
        
        # Probar búsqueda semántica
        print(f"\n🧪 PROBANDO BÚSQUEDA SEMÁNTICA:")
        
        # Realizar algunas búsquedas de prueba
        test_queries = [
            "¿Cuál es la capital de Colombia?",
            "Historia de Colombia", 
            "Geografía y clima colombiano"
        ]
        
        for query in test_queries:
            print(f"\n" + "="*50)
            results = index_service.search(query, top_k=3)
            
        return embedded_chunks
        
    except Exception as e:
        print(f"❌ Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()