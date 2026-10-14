from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


#extract the data from the pdf file 
def load_pdf_files(pdf_directory):
    loader = DirectoryLoader(pdf_directory, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

#filter to minimal 
from typing import List
from langchain.schema import Document


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of Document objects, return a new list of Document objects
    containing only 'source' in metadata and the original page_content.
    """
    minimal_docs: List[Document] = []

    for doc in docs:
        src = doc.metadata.get("source")

        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )

    return minimal_docs

#chunking
#spliting the documents into smnaller chunks 
def text_split(filtered_documents):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    texts=text_splitter.split_documents(filtered_documents)
    return texts

#download embedding mmodel
from langchain.embeddings import HuggingFaceBgeEmbeddings
def download_embeddings():
   """download and return hugging face embeddings model
   """
   model_name="BAAI/bge-small-en-v1.5"
   embeddings=HuggingFaceBgeEmbeddings(
      model_name=model_name
   )
   return embeddings

