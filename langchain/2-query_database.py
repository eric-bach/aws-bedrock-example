# https://github.com/pixegami/langchain-rag-tutorial

from langchain.embeddings import BedrockEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import BedrockChat
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores.chroma import Chroma
import argparse
import boto3

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    # Parse query text
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # create embeddings
    session = boto3.Session(profile_name='bach-dev', region_name='us-east-1')
    bedrock_client = session.client(service_name='bedrock-runtime')
    embedding_function = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)
    #embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # prepare the vector DB using cosine similarity - https://github.com/langchain-ai/langchain/issues/10864
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function, collection_metadata={"hnsw:space": "cosine"})

    # query Chroma
    results = db.similarity_search_with_relevance_scores(query=query_text, k=3)
    print(results)
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")

    # generate prompt
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    # send retrieved document prompt to Bedrock Claude LLM
    model = BedrockChat(model_id="anthropic.claude-v2", model_kwargs={"temperature": 0.1})
    response_text = model.predict(prompt)

    # parse response
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)

if __name__ == "__main__":
    main()
