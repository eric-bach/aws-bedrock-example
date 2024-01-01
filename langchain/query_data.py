import argparse
from dataclasses import dataclass
from langchain.llms.bedrock import Bedrock
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import BedrockEmbeddings
from langchain.chat_models import BedrockChat
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQAWithSourcesChain

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
---
Answer the question based on the above context: {question}
"""

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Issue https://github.com/langchain-ai/langchain/issues/10864
    # Prepare the DB.
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=BedrockEmbeddings(), collection_metadata={"hnsw:space": "cosine"})

    # Search the DB.
    #results = db.similarity_search_with_relevance_scores(query=query_text, k=3)
    results = db.similarity_search_with_score(query=query_text, k=3)
    print(results)
    # if len(results) == 0 or results[0][1] < 0.7:
    #     print(f"Unable to find matching results.")
    #     return

    # context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    # prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    # prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    # chat = BedrockChat(model_id="anthropic.claude-v2", model_kwargs={"temperature": 0.1})
    # response_text = chat

    # sources = [doc.metadata.get("source", None) for doc, _score in results]
    # formatted_response = f"Response: {response_text}\nSources: {sources}"
    # print(formatted_response)


if __name__ == "__main__":
    main()