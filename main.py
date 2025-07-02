from index.extractor import Extractor
from index.index import IndexService
from index.splitter import Splitter

def main():
    # Instanciar el servicio de indexación
    index_service = IndexService()
    
    # Ejecutar el pipeline de indexación
    chunks = index_service.index_pipeline()
    
    # Opcional: procesar el archivo colombia_clean.md que ya fue generado
    extractor = Extractor()
    try:
        with open("colombia_clean.md", "r", encoding="utf-8") as f:
            md = f.read()
        
        # Limpiar nuevamente el contenido si es necesario
        cleaned_md = extractor.clean_md(md)
        
        # Volver a escribir el archivo limpio
        with open("colombia_clean.md", "w", encoding="utf-8") as f:
            f.write(cleaned_md)
            
        # Procesar con el splitter
        splitter = Splitter()
        split_md = splitter.split_md(cleaned_md)
        
        print(f"\n=== RESUMEN DEL PROCESAMIENTO ===")
        print(f"Total de chunks generados: {len(split_md)}")
        
        # Mostrar información de los primeros chunks
        for i, chunk in enumerate(split_md[:3]):
            print(f"\nChunk {i+1}:")
            print(f"Contenido: {chunk.page_content[:200]}...")
            print(f"Metadata: {chunk.metadata}")
            
    except FileNotFoundError:
        print("Error: No se encontró el archivo colombia_clean.md")
        print("El pipeline de indexación debería haberlo generado.")

if __name__ == "__main__":
    main()