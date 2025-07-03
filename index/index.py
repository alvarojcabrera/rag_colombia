from index.extractor import Extractor
from index.splitter import Splitter
from index.embeddings import EmbeddingService
from index.vector_store import VectorStoreService
from index.llm_service import LLMService

class IndexService:

    def __init__(self):
        self.extractor = Extractor()
        self.splitter = Splitter()
        self.embedding_service = EmbeddingService()
        self.vector_store_service = VectorStoreService()
        self.llm_service = LLMService()
        
        # Inicializar conexión a Pinecone para búsquedas
        print("🔌 Conectando a Pinecone para búsquedas...")
        if self.vector_store_service.create_index_if_not_exists():
            print("✅ Conexión a Pinecone establecida")
        else:
            print("❌ Error conectando a Pinecone")

    def index_pipeline(self):
        """
        Pipeline completo de indexación:
        1. Extracción y limpieza
        2. Splitting y chunking  
        3. Generación de embeddings
        4. Indexación en base de datos vectorizada (Pinecone)
        """
        print("=== INICIANDO PIPELINE DE INDEXACIÓN ===\n")
        
        # Paso 1: Extracción
        print("PASO 1: Extracción y limpieza de contenido")
        url = "https://es.wikipedia.org/wiki/Colombia"
        markdown = self.extractor.extract_md(url)
        print("✅ Extracción completada\n")
        
        # Paso 2: Splitting y chunking
        print("PASO 2: Splitting y chunking")
        chunks = self.splitter.split_md(markdown)
        print("✅ Splitting completado\n")
        
        # Paso 3: Generación de embeddings
        print("PASO 3: Generación de embeddings")
        embedded_chunks = self.embedding_service.embed_chunks(chunks)
        print("✅ Embeddings completados\n")
        
        # Paso 4: Indexación en Pinecone
        print("PASO 4: Indexación en base de datos vectorizada")
        
        # Crear índice si no existe
        if self.vector_store_service.create_index_if_not_exists():
            # Indexar chunks
            if self.vector_store_service.index_chunks(embedded_chunks):
                print("✅ Indexación en Pinecone completada\n")
            else:
                print("❌ Error en indexación de Pinecone\n")
                return embedded_chunks  # Retornar chunks aunque falle Pinecone
        else:
            print("❌ Error creando índice de Pinecone\n")
            return embedded_chunks  # Retornar chunks aunque falle Pinecone
        
        print("=== PIPELINE COMPLETADO ===")
        print(f"Total de chunks procesados: {len(embedded_chunks)}")
        print(f"Dimensión de embeddings: {self.embedding_service.get_embedding_dimension()}")
        
        # Mostrar estadísticas de Pinecone
        stats = self.vector_store_service.get_index_stats()
        if 'error' not in stats:
            print(f"Vectores en Pinecone: {stats['total_vectors']}")
        
        return embedded_chunks
    
    def search(self, query: str, top_k: int = 5):
        """
        Búsqueda semántica en los chunks indexados usando Pinecone.
        """
        print(f"🔍 Realizando búsqueda semántica: '{query}'")
        
        # Buscar en Pinecone
        results = self.vector_store_service.search_similar(query, top_k)
        
        if results:
            print(f"\n📋 RESULTADOS DE BÚSQUEDA:")
            for i, result in enumerate(results):
                print(f"\n   Resultado {i+1}:")
                print(f"   Score: {result['similarity_score']:.4f}")
                print(f"   Chunk ID: {result['chunk_id']}")
                print(f"   Contenido: {result['content'][:200]}...")
                if result['metadata']:
                    headers = {k: v for k, v in result['metadata'].items() if k.startswith('header_')}
                    if headers:
                        print(f"   Headers: {headers}")
        else:
            print("❌ No se encontraron resultados")
        
        return results

    def ask_question(self, question: str) -> str:
        """
        Método principal del RAG: pregunta → búsqueda → generación → respuesta
        """
        print(f"\n❓ PREGUNTA: {question}")
        print("=" * 50)
        
        # 1. Búsqueda semántica (ya funciona)
        search_results = self.vector_store_service.search_similar(question, top_k=5)
        
        # 2. Generación de respuesta con LLM (nuevo)
        answer = self.llm_service.generate_answer(question, search_results)
        
        print(f"\n💡 RESPUESTA:")
        print(answer)
        print("=" * 50)
        
        return answer