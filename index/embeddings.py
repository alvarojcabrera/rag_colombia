from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List, Dict, Any
import json
import os
import hashlib

class EmbeddingService:
    """
    Servicio para generar embeddings usando LangChain + HuggingFace.
    Utiliza la integración oficial de LangChain con HuggingFace embeddings.
    """
    
    # Modelo optimizado para contenido multilingüe
    MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDINGS_FILE = "embeddings_cache.json"
    
    def __init__(self):
        print(f"Inicializando servicio de embeddings con modelo: {self.MODEL_NAME}")
        
        # Configuración del modelo usando LangChain
        model_kwargs = {
            'device': 'cpu',  # Cambiar a 'cuda' si tienes GPU
            'trust_remote_code': True
        }
        
        encode_kwargs = {
            'normalize_embeddings': True  # Normalizar embeddings para mejor performance
        }
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.MODEL_NAME,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        
        print("✅ Modelo de embeddings inicializado")
        
        # Cache para evitar recalcular embeddings
        self.embeddings_cache = self._load_cache()
    
    def _load_cache(self) -> Dict[str, Any]:
        """Carga el cache de embeddings si existe."""
        if os.path.exists(self.EMBEDDINGS_FILE):
            try:
                with open(self.EMBEDDINGS_FILE, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                print(f"📁 Cache de embeddings cargado: {len(cache)} entradas")
                return cache
            except Exception as e:
                print(f"⚠️  Error cargando cache: {e}")
                return {}
        return {}
    
    def _save_cache(self):
        """Guarda el cache de embeddings."""
        try:
            with open(self.EMBEDDINGS_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.embeddings_cache, f, ensure_ascii=False, indent=2)
            print(f"💾 Cache guardado: {len(self.embeddings_cache)} entradas")
        except Exception as e:
            print(f"⚠️  Error guardando cache: {e}")
    
    def _generate_chunk_id(self, chunk: Document) -> str:
        """Genera un ID único para el chunk basado en su contenido."""
        # Usar hash del contenido para ID único
        content_hash = hashlib.md5(chunk.page_content.encode('utf-8')).hexdigest()
        return f"chunk_{content_hash[:12]}"
    
    def embed_query(self, text: str) -> List[float]:
        """
        Genera embedding para una query/pregunta.
        Usa el método optimizado de LangChain para queries.
        """
        return self.embeddings.embed_query(text)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Genera embeddings para múltiples documentos.
        Usa el método optimizado de LangChain para documentos.
        """
        return self.embeddings.embed_documents(texts)
    
    def embed_chunks(self, chunks: List[Document]) -> List[Dict[str, Any]]:
        """
        Genera embeddings para una lista de chunks de documentos.
        Retorna lista de diccionarios con chunk_id, content, metadata y embedding.
        """
        print(f"🔄 Generando embeddings para {len(chunks)} chunks...")
        
        embedded_chunks = []
        texts_to_embed = []
        chunks_to_process = []
        cached_count = 0
        
        # Revisar cache primero
        for i, chunk in enumerate(chunks):
            chunk_id = self._generate_chunk_id(chunk)
            
            if chunk_id in self.embeddings_cache:
                # Usar embedding del cache
                cached_data = self.embeddings_cache[chunk_id]
                embedded_chunks.append({
                    'chunk_id': chunk_id,
                    'content': chunk.page_content,
                    'metadata': chunk.metadata,
                    'embedding': cached_data['embedding']
                })
                cached_count += 1
                print(f"📋 Chunk {i+1}/{len(chunks)}: Cache hit")
            else:
                # Marcar para procesamiento
                texts_to_embed.append(chunk.page_content)
                chunks_to_process.append((chunk, chunk_id, len(embedded_chunks)))
                embedded_chunks.append(None)  # Placeholder
        
        print(f"📋 {cached_count} chunks encontrados en cache")
        
        # Generar embeddings para chunks no cacheados
        if texts_to_embed:
            print(f"🚀 Generando {len(texts_to_embed)} nuevos embeddings...")
            new_embeddings = self.embed_documents(texts_to_embed)
            
            # Procesar resultados
            for i, (chunk, chunk_id, position) in enumerate(chunks_to_process):
                embedding = new_embeddings[i]
                
                # Guardar en cache
                self.embeddings_cache[chunk_id] = {
                    'content': chunk.page_content,
                    'metadata': chunk.metadata,
                    'embedding': embedding
                }
                
                # Actualizar lista de resultados
                embedded_chunks[position] = {
                    'chunk_id': chunk_id,
                    'content': chunk.page_content,
                    'metadata': chunk.metadata,
                    'embedding': embedding
                }
                
                print(f"✨ Chunk procesado: {i+1}/{len(texts_to_embed)}")
            
            # Guardar cache actualizado
            self._save_cache()
        
        # Filtrar None values (no debería haber ninguno)
        embedded_chunks = [chunk for chunk in embedded_chunks if chunk is not None]
        
        print(f"✅ Embeddings completados: {len(embedded_chunks)} chunks procesados")
        return embedded_chunks
    
    def get_embedding_dimension(self) -> int:
        """Retorna la dimensión de los embeddings."""
        # Para el modelo multilingual-MiniLM-L12-v2 son 384 dimensiones
        test_embedding = self.embed_query("test")
        return len(test_embedding)
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calcula similitud coseno entre dos embeddings.
        """
        import numpy as np
        
        # Convertir a numpy arrays
        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)
        
        # Calcular similitud coseno
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)
    
    def search_similar(self, query: str, embedded_chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Búsqueda de similitud semántica.
        Retorna los top_k chunks más similares a la query.
        """
        print(f"🔍 Buscando chunks similares a: '{query[:50]}...'")
        
        # Generar embedding de la query
        query_embedding = self.embed_query(query)
        
        # Calcular similitudes
        similarities = []
        for chunk_data in embedded_chunks:
            similarity = self.calculate_similarity(query_embedding, chunk_data['embedding'])
            similarities.append({
                **chunk_data,
                'similarity': similarity
            })
        
        # Ordenar por similitud descendente
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        results = similarities[:top_k]
        
        print(f"🎯 Encontrados {len(results)} chunks relevantes")
        for i, result in enumerate(results):
            print(f"  {i+1}. Similitud: {result['similarity']:.3f} | Contenido: {result['content'][:80]}...")
        
        return results