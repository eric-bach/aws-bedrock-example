## Getting Started

Contains examples of prompt engineering scripts using Amazon Bedrock and Langchain with RAG.

- `0-compare_embeddings.py` - Demonstrates Langchain evaluator of string pair with Amazon Bedrock embeddings
- `alice.py` - Uses Amazon Bedrock with Langchain RAG to retrieve response from markdown document loaded in a Chroma vector DB
   - `1-create_database.py` - first part of the `alice.py` script that uses Amazon Bedrock embeddings to split markdown documents into a Chroma vector DB with Langchain
   - `2-query_database.py` - second part of the `alice.py` script that uses Langchain with Anthropic Claude v2.1 in Amazon Bedrock to generate an answer from a prompt by augmenting LLM with RAG in vector DB

### Installation

```
$ pip install langchain langchain-community tqdm unstructured markdown boto3 
```

### Running

```
$ python alice.py "How did Alice meet the Mat Hatter?"
```
