from index.extractor import Extractor
from index.splitter import Splitter

class IndexService:

    def __init__(self):
        self.extractor = Extractor()
        self.splitter = Splitter()

    def index_pipeline(self):
        url = "https://es.wikipedia.org/wiki/Colombia"
        markdown = self.extractor.extract_md(url)
        chunked_md = self.splitter.split_md(markdown)
        print(chunked_md)