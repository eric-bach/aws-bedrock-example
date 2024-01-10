#https://github.com/aws-samples/amazon-bedrock-workshop/tree/main/03_QuestionAnswering
#https://medium.com/@tahir.rauf/similarity-search-using-langchain-and-bedrock-4140b0ae9c58
#https://github.com/aws-samples/rag-using-langchain-amazon-bedrock-and-opensearch

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain_community.vectorstores import FAISS
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import argparse
import numpy as np
import boto3

prompt_template = """

Human: Use the following pieces of context to provide a concise answer to the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
<context>
{context}
</context

Question: {question}

Assistant:"""

PROMPT_TEMPLATE = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# create CLI
parser = argparse.ArgumentParser()
parser.add_argument("query_text", type=str, help="The query text.")
args = parser.parse_args()
query_text = args.query_text

# load the document 
loader = PyPDFLoader("data/guides/imis_guide.pdf")
pages = loader.load()
print(f"Split into {len(pages)} chunks.")

# split it into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100, length_function=len, add_start_index=True)
docs = text_splitter.split_documents(pages)
print(f"Split {len(pages)} documents into {len(docs)} chunks.")

# create the embedding function
session = boto3.Session(profile_name='bach-dev', region_name='us-east-1')
bedrock_client = session.client(service_name='bedrock-runtime')
llm = Bedrock(model_id="anthropic.claude-v2", client=bedrock_client, model_kwargs={'max_tokens_to_sample':200})
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)

# store in FAISS
vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
wrapper_store_faiss = VectorStoreIndexWrapper(vectorstore=vectorstore_faiss)

# build prompt
query_embedding = vectorstore_faiss.embedding_function.embed_query(query_text)
np.array(query_embedding)

# query vector DB
relevant_documents = vectorstore_faiss.similarity_search_by_vector(query_embedding)
print(f'{len(relevant_documents)} documents are fetched which are relevant to the query.')
print('----')
for i, rel_doc in enumerate(relevant_documents):
    print(f'## Document {i+1}: {rel_doc.page_content}.......')
    print('---')

# query LLM
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore_faiss.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT_TEMPLATE}
)
result = qa({"query": query_text})
print(result['result'])
print(result['source_documents'])
