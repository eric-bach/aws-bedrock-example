## Getting Started

Contains examples of prompt engineering scripts using Amazon Bedrock and Langchain with RAG.

`0-compare_embeddings.py` - Demonstrates Langchain evaluator of string pair with Amazon Bedrock embeddings
`1-union.py` - Uses Langchain RAG to retrieve response from text document loaded in a Chroma vector DB
`2-alice.py` - Uses Amazon Bedrock with Langchain RAG to retrieve response from markdown document loaded in a Chroma vector DB

### Setup WSL VS Code

```
$ sudo apt update
$ sudo apt install python python3-pip
$ python -m pip install --upgrade pip
$ pip install -U pyopenssl cryptography
$ Edit .bashrc alias to include "alias python='python3'"

```

### Installation

```
$ pip install langchain langchain-community tqdm unstructured "unstructured[pdf]" markdown boto3 
```

## Langchain RAG application

`create-database.py` - Uses Amazon Bedrock embeddings to split markdown documents into a Chroma vector DB with Langchain
`query_data.py` - Uses Langchain with Anthropic Claude v2.1 in Amazon Bedrock to generate an answer from a prompt by augmenting LLM with RAG in vector DB

```
$ python create_database.py
$ python query_database.py
```
