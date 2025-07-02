from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
import time

class VectorStoreService:
    """
    Servicio para manejar la base de datos vectorizada con Pinecone.
    Gestiona la creación del índice, inserción de vectores y búsquedas semánticas.
    """
    
    INDEX_NAME = "colombia-rag"
    DIMENSION = 384  # Dimensión del modelo multilingual-MiniLM-L12-v2
    METRIC = "cosine"
    CLOUD = "aws"
    REGION = "us-east-1"
    
    def __init__(self):
        load_dotenv()
        
        # Configurar Pinecone
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY no está configurada en el archivo .env")
        
        # Inicializar cliente de Pinecone
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        
        # Configurar embeddings (mismo modelo que en embeddings.py)
        model_kwargs = {'device': 'cpu', 'trust_remote_code': True}
        encode_kwargs = {'normalize_embeddings': True}
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        
        # Inicializar vector store
        self.vector_store = None
        
        print("✅ VectorStoreService inicializado")
    
    def create_index_if_not_exists(self):
        """
        Crea el índice de Pinecone si no existe.
        """
        try:
            # Verificar si el índice ya existe
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.INDEX_NAME not in existing_indexes:
                print(f"📝 Creando índice '{self.INDEX_NAME}' en Pinecone...")
                
                self.pc.create_index(
                    name=self.INDEX_NAME,
                    dimension=self.DIMENSION,
                    metric=self.METRIC,
                    spec=ServerlessSpec(
                        cloud=self.CLOUD,
                        region=self.REGION
                    )
                )
                
                # Esperar a que el índice esté listo
                print("⏳ Esperando a que el índice esté listo...")
                while not self.pc.describe_index(self.INDEX_NAME).status['ready']:
                    time.sleep(1)
                
                print("✅ Índice creado exitosamente")
            else:
                print(f"📋 Índice '{self.INDEX_NAME}' ya existe")
            
            # Conectar al vector store
            self.vector_store = PineconeVectorStore(
                index_name=self.INDEX_NAME,
                embedding=self.embeddings,
                pinecone_api_key=self.pinecone_api_key
            )
            
            return True
            
        except Exception as e:
            print(f"❌ Error creando/conectando al índice: {e}")
            return False
    
    def index_chunks(self, embedded_chunks: List[Dict[str, Any]]) -> bool:
        """
        Indexa los chunks con embeddings en Pinecone.
        """
        try:
            if not self.vector_store:
                print("❌ Vector store no inicializado. Ejecutar create_index_if_not_exists() primero")
                return False
            
            print(f"📤 Subiendo {len(embedded_chunks)} chunks a Pinecone...")
            
            # Preparar documentos para LangChain
            texts = []
            metadatas = []
            ids = []
            
            for chunk_data in embedded_chunks:
                texts.append(chunk_data['content'])
                
                # Preparar metadata (Pinecone requiere strings, no objetos complejos)
                metadata = {
                    'chunk_id': chunk_data['chunk_id'],
                    'source': 'wikipedia_colombia'
                }
                
                # Agregar headers de metadata de forma segura
                if 'metadata' in chunk_data and chunk_data['metadata']:
                    for key, value in chunk_data['metadata'].items():
                        if isinstance(value, (str, int, float, bool)):
                            metadata[f'header_{key}'] = str(value)
                
                metadatas.append(metadata)
                ids.append(chunk_data['chunk_id'])
            
            # Subir a Pinecone usando LangChain
            self.vector_store.add_texts(
                texts=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            print("✅ Chunks indexados exitosamente en Pinecone")
            
            # Verificar indexación
            index_stats = self.pc.Index(self.INDEX_NAME).describe_index_stats()
            total_vectors = index_stats['total_vector_count']
            print(f"📊 Total de vectores en el índice: {total_vectors}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error indexando chunks: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def search_similar(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Busca chunks similares a la query usando búsqueda semántica.
        """
        try:
            if not self.vector_store:
                print("❌ Vector store no inicializado")
                return []
            
            print(f"🔍 Buscando chunks similares a: '{query[:50]}...'")
            
            # Realizar búsqueda semántica
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query=query,
                k=top_k
            )
            
            # Formatear resultados
            results = []
            for doc, score in docs_with_scores:
                result = {
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'similarity_score': float(score),
                    'chunk_id': doc.metadata.get('chunk_id', 'unknown')
                }
                results.append(result)
            
            print(f"🎯 Encontrados {len(results)} chunks relevantes")
            for i, result in enumerate(results):
                print(f"  {i+1}. Score: {result['similarity_score']:.3f} | ID: {result['chunk_id']}")
                print(f"     Contenido: {result['content'][:80]}...")
            
            return results
            
        except Exception as e:
            print(f"❌ Error en búsqueda: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del índice de Pinecone.
        """
        try:
            if self.INDEX_NAME in [index.name for index in self.pc.list_indexes()]:
                stats = self.pc.Index(self.INDEX_NAME).describe_index_stats()
                return {
                    'total_vectors': stats['total_vector_count'],
                    'dimension': stats['dimension'],
                    'index_fullness': stats.get('index_fullness', 0),
                    'namespaces': stats.get('namespaces', {})
                }
            else:
                return {'error': 'Index does not exist'}
        except Exception as e:
            return {'error': str(e)}
    
    def delete_index(self):
        """
        Elimina el índice de Pinecone (usar con cuidado).
        """
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