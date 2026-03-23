"""
Document loading module for the RAG indexing pipeline.
Loads and parses web content using WebBaseLoader and BeautifulSoup.
"""

import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from src.config.settings import SOURCE_URL


def load_documents() -> list[Document]:
    """
    Loads the target blog post from the web, retaining only relevant HTML sections.

    Returns:
        list[Document]: A list containing the parsed document.
    """
    bs4_strainer = bs4.SoupStrainer(
        class_=("post-title", "post-header", "post-content")
    )
    loader = WebBaseLoader(
        web_paths=(SOURCE_URL,),
        bs_kwargs={"parse_only": bs4_strainer},
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} document(s). "
          f"Total characters: {len(docs[0].page_content)}")
    return docs