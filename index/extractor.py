from dotenv import load_dotenv
import os
from firecrawl import FirecrawlApp
import re

class Extractor:
    """
    Extraccion y limpieza de contenido de una pagina web.
    Utiliza la API de Firecrawl directamente, en lugar de usar el loader de Langchain
    para tener mas control sobre la extraccion y limpieza del contenido.
    """

    ORIGINAL_FILE_NAME = "colombia.md"
    CLEAN_FILE_NAME = "colombia_clean.md"

    def __init__(self):
        load_dotenv()
        self.firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")
        if not self.firecrawl_api_key:
            raise ValueError("FIRECRAWL_API_KEY is not set")
        
        self.firecrawl_app = FirecrawlApp(api_key=self.firecrawl_api_key)

    def extract_md(self, url: str) -> str:
        """
        Extrae el contenido de una pagina web y lo devuelve en formato markdown limpiado.
        """
        print(f"Extrayendo contenido de {url}")
        
        try:
            scrape_result = self.firecrawl_app.scrape_url(
                url=url, 
                formats=["markdown"], 
                only_main_content=True
            )
            
            if not hasattr(scrape_result, 'markdown') or scrape_result.markdown is None:
                raise ValueError("No se pudo extraer el contenido de la pagina web")
            
            markdown = scrape_result.markdown
            
            print(f"Contenido extraido. Guardando contenido original en {self.ORIGINAL_FILE_NAME}")
            # Guardar el contenido original
            with open(self.ORIGINAL_FILE_NAME, "w", encoding="utf-8") as f:
                f.write(markdown)
            
            print("Contenido original guardado. Comenzando limpieza...")
            clean_markdown = self.clean_md(markdown)
            
            print(f"Limpieza completada. Guardando contenido limpio en {self.CLEAN_FILE_NAME}")
            # Guardar el contenido limpio
            with open(self.CLEAN_FILE_NAME, "w", encoding="utf-8") as f:
                f.write(clean_markdown)
            
            print("Contenido limpio guardado. Fin del proceso de extraccion y limpieza.")
            return clean_markdown
            
        except Exception as e:
            print(f"Error durante la extracción: {e}")
            raise
    
    def clean_md(self, md: str) -> str:
        """
        Limpia el contenido markdown.
        """
        if not md:
            return ""
            
        # Elimina todo despues de ## Véase también, incluso en otras lineas
        md = re.sub(r'## Véase también.*', '', md, flags=re.DOTALL)

        # Elimina imagenes
        md = re.sub(r'!\[.*?\]\(.*?\)', '', md)

        # Elimina citas de la forma \[123\]
        md = re.sub(r'\\\[[0-9]+\\\]', '', md)

        # Reemplaza los links de la forma [link](url "title") por el texto del link
        md = re.sub(r'\[(.*?)\]\(.*? ".*?"\)', r'\1', md)

        # Reemplaza los links de la forma [link](url) por el texto del link
        md = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', md)

        # Elimina lineas que empiezan con Artículo principal:
        md = re.sub(r'^Artículo principal:.*', '', md, flags=re.MULTILINE)

        # Cambia <br> por espacio
        md = re.sub(r'<br>', ' ', md)
        
        # Elimina líneas vacías múltiples
        md = re.sub(r'\n\s*\n\s*\n', '\n\n', md)

        return md.strip()