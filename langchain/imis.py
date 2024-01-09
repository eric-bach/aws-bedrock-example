#https://github.com/aws-samples/amazon-bedrock-workshop/tree/main/03_QuestionAnswering
#https://medium.com/@tahir.rauf/similarity-search-using-langchain-and-bedrock-4140b0ae9c58
#https://github.com/aws-samples/rag-using-langchain-amazon-bedrock-and-opensearch

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import BedrockChat
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain_community.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import numpy as np
import os
import boto3
import shutil

CHROMA_PATH = "chroma"
DATA_PATH = "data/guides"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# load the document 
# loader = DirectoryLoader("../", glob="**/*.md", show_progress=True)
# documents = loader.load()
loader = PyPDFLoader("data/guides/imis_guide.pdf")
pages = loader.load()
print(f"Split into {len(pages)} chunks.")

# split it into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100, length_function=len, add_start_index=True)
chunks = text_splitter.split_documents(pages)
print(f"Split {len(pages)} documents into {len(chunks)} chunks.")

# OPTION 1 - Use FAISS

# create the embedding function
session = boto3.Session(profile_name='bach-dev', region_name='us-east-1')
bedrock_client = session.client(service_name='bedrock-runtime')
llm = Bedrock(model_id="anthropic.claude-v2", client=bedrock_client, model_kwargs={'max_tokens_to_sample':200})
embedding_function = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)
#embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# store in vector db
vectorstore_faiss = FAISS.from_documents(chunks, embedding_function)
wrapper_store_faiss = VectorStoreIndexWrapper(vectorstore=vectorstore_faiss)

# build prompt
query = """Can a member have more than one membership number?"""
query_embedding = vectorstore_faiss.embedding_function.embed_query(query)
np.array(query_embedding)

# query vector DB
relevant_documents = vectorstore_faiss.similarity_search_by_vector(query_embedding)
#relevant_documents = vectorstore_faiss.similarity_search_with_relevance_scores(query, k=3)
print(f'{len(relevant_documents)} documents are fetched which are relevant to the query.')
print(relevant_documents)
print('----')
for i, rel_doc in enumerate(relevant_documents):
    print(f'## Document {i+1}: {rel_doc.page_content}.......')
    print('---')

# query LLM
prompt_template = """

Human: Use the following pieces of context to provide a concise answer to the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
<context>
{context}
</context

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore_faiss.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
answer = qa({"query": query})
print(answer)

# OPTION 2 - Use ChromaDB
# create the embedding function
#embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# # clear out the database first
# if os.path.exists(CHROMA_PATH):
#     shutil.rmtree(CHROMA_PATH)

# # load it into Chroma
# # https://js.langchain.com/docs/integrations/vectorstores/chroma
# db = Chroma.from_documents(chunks, embedding_function, collection_metadata={"hnsw:space": "cosine"})
# db.persist()
# print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

# # query it
# query_text = "Can a member have more than one membership number?"
# results = db.similarity_search_with_relevance_scores(query=query_text)
# print(results)

# if len(results) == 0 or results[0][1] < 0.7:
#     print(f"Unable to find matching results.")

# context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
# prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
# prompt = prompt_template.format(context=context_text, question=query_text)
# print(prompt)

# model = BedrockChat(model_id="anthropic.claude-v2", model_kwargs={"temperature": 0.1})
# response_text = model.predict(prompt)

# sources = [doc.metadata.get("source", None) for doc, _score in results]
# formatted_response = f"Response: {response_text}\nSources: {sources}"
# print(formatted_response)
