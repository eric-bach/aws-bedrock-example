# import
from langchain.document_loaders import TextLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import Chroma

# load the document and split it into chunks
loader = TextLoader("state_of_the_union.txt")
documents = loader.load()

# split it into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
chunks = text_splitter.split_documents(documents)

# create the open-source embedding function
#embedding_function = BedrockEmbeddings()
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# load it into Chroma
db = Chroma.from_documents(chunks, embedding_function, collection_metadata={"hnsw:space": "cosine"})
db.persist()

# query it
query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search_with_relevance_scores(query=query, k=3)

# print results
print(docs[0])