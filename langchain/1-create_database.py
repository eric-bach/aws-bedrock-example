# https://github.com/pixegami/langchain-rag-tutorial

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores.chroma import Chroma
import boto3
import os
import shutil

CHROMA_PATH = "chroma"
DATA_PATH = "data/books"

def main():
    # load all markdown documents
    loader = DirectoryLoader(DATA_PATH, glob="*.md", show_progress=True)
    documents = loader.load()

    # split each document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100, length_function=len, add_start_index=True)
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")   

    # create the embedding function
    session = boto3.Session(profile_name='bach-dev', region_name='us-east-1')
    bedrock_client = session.client(service_name='bedrock-runtime')
    embedding_function = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)
    #embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Clear any existing vector DB
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # load chunks into Chroma - https://js.langchain.com/docs/integrations/vectorstores/chroma
    db = Chroma.from_documents(chunks, embedding_function, persist_directory=CHROMA_PATH, collection_metadata={"hnsw:space": "cosine"})
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

if __name__ == "__main__":
    main()
