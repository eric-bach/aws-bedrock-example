#https://github.com/aws-samples/amazon-bedrock-workshop/tree/main/03_QuestionAnswering
#https://medium.com/@tahir.rauf/similarity-search-using-langchain-and-bedrock-4140b0ae9c58
#https://github.com/aws-samples/rag-using-langchain-amazon-bedrock-and-opensearch

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import BedrockChat
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain_community.vectorstores import Chroma
import argparse
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
chunks = text_splitter.split_documents(pages)
print(f"Split {len(pages)} documents into {len(chunks)} chunks.")

# create the embedding function
session = boto3.Session(profile_name='bach-dev', region_name='us-east-1')
bedrock_client = session.client(service_name='bedrock-runtime')
llm = Bedrock(model_id="anthropic.claude-v2", client=bedrock_client, model_kwargs={'max_tokens_to_sample':200})
embedding_function = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)

# clear out the database first
if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)

# load it into Chroma
db = Chroma.from_documents(chunks, embedding_function, collection_metadata={"hnsw:space": "cosine"})
db.persist()
print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

# query it
results = db.similarity_search_with_relevance_scores(query=query_text)
print(results)

if len(results) == 0 or results[0][1] < 0.7:
    print(f"Unable to find matching results.")

context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
prompt = prompt_template.format(context=context_text, question=query_text)
print(prompt)

model = BedrockChat(model_id="anthropic.claude-v2", model_kwargs={"temperature": 0.1}, client=bedrock_client)
response_text = model.predict(prompt)

sources = [doc.metadata.get("source", None) for doc, _score in results]
formatted_response = f"Response: {response_text}\nSources: {sources}"
print(formatted_response)
