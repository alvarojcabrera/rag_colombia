from index.extractor import Extractor
from index.splitter import Splitter
from index.vector_store import VectorStoreService
from index.llm_service import LLMService
from langchain_core.documents import Document
from typing import List, Optional

class IndexService:
    """Servicio de indexación OPTIMIZADO que sigue la documentación de LangChain."""

    def __init__(self, embedding_model: str = None):
        print("🚀 Inicializando IndexService optimizado...")
        
        self.extractor = Extractor()
        self.splitter = Splitter()
        self.vector_store_service = VectorStoreService(embedding_model_name=embedding_model)
        self.llm_service = LLMService()
        
        print("🔌 Conectando a Pinecone...")
        if self.vector_store_service.create_index_if_not_exists():
            print("✅ Conexión a Pinecone establecida")
        else:
            print("❌ Error conectando a Pinecone")

    def index_pipeline(self, url: str = None, content: str = None) -> bool:
        """Pipeline OPTIMIZADO siguiendo la documentación de LangChain."""
        try:
            print("=== PIPELINE DE INDEXACIÓN OPTIMIZADO ===")
            
            if url:
                print("📥 PASO 1: Extrayendo contenido de URL...")
                content = self.extractor.extract_md(url)
                print("✅ Extracción completada")
            elif content:
                print("📄 PASO 1: Usando contenido proporcionado...")
            else:
                print("❌ Debe proporcionar URL o contenido")
                return False
            
            print("✂️  PASO 2: Dividiendo contenido en chunks...")
            chunks = self.splitter.split_md(content)
            
            if not chunks:
                print("❌ No se generaron chunks")
                return False
            
            print(f"📋 Generados {len(chunks)} chunks como Document objects")
            
            print("🚀 PASO 3: Indexando usando add_documents()...")
            success = self.vector_store_service.add_documents(chunks)
            
            if success:
                print("✅ Pipeline completado exitosamente")
                stats = self.vector_store_service.get_index_stats()
                if 'total_vectors' in stats:
                    print(f"📊 Total de vectores: {stats['total_vectors']}")
                return True
            else:
                print("❌ Error en la indexación")
                return False
                
        except Exception as e:
            print(f"❌ Error en pipeline: {e}")
            return False

    def search(self, query: str, top_k: int = 10) -> List[dict]:
        try:
            return self.vector_store_service.search_similar(query, top_k)
        except Exception as e:
            print(f"❌ Error en búsqueda: {e}")
            return []

    def rag_query(self, question: str, top_k: int = 10) -> str:
        try:
            relevant_docs = self.search(question, top_k)
            
            if not relevant_docs:
                return "No se encontró información relevante."
            
            context = "\n\n".join([doc['content'] for doc in relevant_docs])
            response = self.llm_service.generate_response(question, context)
            
            return response
        except Exception as e:
            return f"Error procesando la consulta: {str(e)}"

    def get_stats(self) -> dict:
        return self.vector_store_service.get_index_stats()

    def delete_index(self) -> bool:
        return self.vector_store_service.delete_index()

    def index_documents(self, documents: List[Document]) -> bool:
        try:
            return self.vector_store_service.add_documents(documents)
        except Exception as e:
            print(f"❌ Error indexando documentos: {e}")
            return False

    def index_texts(self, texts: List[str], metadatas: List[dict] = None) -> bool:
        try:
            documents = []
            for i, text in enumerate(texts):
                metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
                doc = Document(page_content=text, metadata=metadata)
                documents.append(doc)
            
            return self.vector_store_service.add_documents(documents)
        except Exception as e:
            print(f"❌ Error indexando textos: {e}")
            return False
