from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
import time
import hashlib

class VectorStoreService:
    """
    Servicio OPTIMIZADO para manejar la base de datos vectorizada con Pinecone.
    SIGUE EXACTAMENTE LA DOCUMENTACIÓN DE LANGCHAIN.
    """
    
    INDEX_NAME = "colombia-rag"
    DIMENSION = 384
    METRIC = "cosine"
    CLOUD = "aws"
    REGION = "us-east-1"
    
    def __init__(self, embedding_model_name: str = None):
        load_dotenv()
        
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY no está configurada en el archivo .env")
        
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        
        self.embedding_model_name = embedding_model_name or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        self.embeddings = self._create_embeddings()
        
        self.vector_store = None
        print("✅ VectorStoreService optimizado inicializado")
    
    def _create_embeddings(self) -> HuggingFaceEmbeddings:
        model_kwargs = {'device': 'cpu', 'trust_remote_code': True}
        encode_kwargs = {'normalize_embeddings': True}
        
        return HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
    
    def create_index_if_not_exists(self) -> bool:
        try:
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.INDEX_NAME not in existing_indexes:
                print(f"📝 Creando índice '{self.INDEX_NAME}' en Pinecone...")
                self.pc.create_index(
                    name=self.INDEX_NAME,
                    dimension=self.DIMENSION,
                    metric=self.METRIC,
                    spec=ServerlessSpec(cloud=self.CLOUD, region=self.REGION)
                )
                
                print("⏳ Esperando a que el índice esté listo...")
                while not self.pc.describe_index(self.INDEX_NAME).status['ready']:
                    time.sleep(1)
                print("✅ Índice creado exitosamente")
            else:
                print(f"📋 Índice '{self.INDEX_NAME}' ya existe")
            
            self.vector_store = PineconeVectorStore(
                index_name=self.INDEX_NAME,
                embedding=self.embeddings,
                pinecone_api_key=self.pinecone_api_key
            )
            
            return True
        except Exception as e:
            print(f"❌ Error creando/conectando al índice: {e}")
            return False
    
    def add_documents(self, documents: List[Document]) -> bool:
        """MÉTODO PRINCIPAL - SIGUE LA DOCUMENTACIÓN DE LANGCHAIN"""
        try:
            if not self.vector_store:
                print("❌ Vector store no inicializado")
                return False
            
            print(f"📤 Añadiendo {len(documents)} documentos usando add_documents()...")
            print("🔧 LangChain generará embeddings automáticamente")
            
            # LÍNEA MÁGICA - SIGUE LA DOCUMENTACIÓN OFICIAL
            self.vector_store.add_documents(documents=documents)
            
            print("✅ Documentos añadidos exitosamente")
            
            try:
                index_stats = self.pc.Index(self.INDEX_NAME).describe_index_stats()
                total_vectors = index_stats['total_vector_count']
                print(f"📊 Total de vectores en el índice: {total_vectors}")
            except Exception as e:
                print(f"⚠️ No se pudieron obtener estadísticas: {e}")
            
            return True
        except Exception as e:
            print(f"❌ Error añadiendo documentos: {e}")
            return False
    
    def search_similar(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        try:
            if not self.vector_store:
                return []
            
            docs_with_scores = self.vector_store.similarity_search_with_score(query=query, k=top_k)
            
            results = []
            for doc, score in docs_with_scores:
                result = {
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'similarity_score': float(score)
                }
                results.append(result)
            
            return results
        except Exception as e:
            print(f"❌ Error en búsqueda: {e}")
            return []
    
    def get_index_stats(self) -> Dict[str, Any]:
        try:
            if self.INDEX_NAME in [index.name for index in self.pc.list_indexes()]:
                stats = self.pc.Index(self.INDEX_NAME).describe_index_stats()
                return {
                    'total_vectors': stats['total_vector_count'],
                    'dimension': stats['dimension'],
                    'index_fullness': stats.get('index_fullness', 0)
                }
            else:
                return {'error': 'Index does not exist'}
        except Exception as e:
            return {'error': str(e)}
    
    def delete_index(self) -> bool:
        try:
            if self.INDEX_NAME in [index.name for index in self.pc.list_indexes()]:
                self.pc.delete_index(self.INDEX_NAME)
                print(f"🗑️  Índice '{self.INDEX_NAME}' eliminado")
                return True
            else:
                print(f"ℹ️  Índice '{self.INDEX_NAME}' no existe")
                return False
        except Exception as e:
            print(f"❌ Error eliminando índice: {e}")
            return False
