from dotenv import load_dotenv
import os
import re
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class Splitter:
    """
    Splitting y chunking de markdown

    En primer lugar, se hace un splitting de markdown en base a los headers.
    Luego, cada split se divide de acuerdo al numero maximo de tokens deseados, usando el
    RecursiveCharacterTextSplitter de Langchain.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ]
        self.markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=self.headers_to_split_on, strip_headers=False)

        self.recursive_character_text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def _split_by_headers(self, md: str) -> list[Document]:
        """
        Splitting de markdown basado en los headers.
        Devuelve una lista de Documentos, donde cada Documento tiene como metadata los headers a los que pertenece.
        """
        return self.markdown_splitter.split_text(md)
    
    def _chunk_by_tokens(self, docs: list[Document]) -> list[Document]:
        """
        Chunking de markdown basado en el numero de tokens.
        Usa RecursiveCharacterTextSplitter de Langchain, que mantiene la estructura jerárquica
        natural del texto intentando preservar primero unidades más grandes (párrafos), luego
        oraciones, y finalmente palabras si es necesario para cumplir con el tamaño del chunk.
        """
        return self.recursive_character_text_splitter.split_documents(docs)

    def split_md(self, md: str) -> list[Document]:
        """
        Splitting y chunking de markdown.
        """
        print("Splitting de markdown en base a los headers...")
        split_md = self._split_by_headers(md)
        print("Splitting de markdown en base a los tokens...")
        chunked_md = self._chunk_by_tokens(split_md)
        print(f"Fin del splitting y chunking de markdown. Se obtuvieron {len(chunked_md)} chunks.")
        return chunked_md