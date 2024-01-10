## Getting Started

Contains examples of prompt engineering scripts using Amazon Bedrock and Langchain with RAG.

`imis.py` - 
`imis_chroma.py` -


### Installation

```
$ pip install langchain langchain-community tqdm unstructured "unstructured[pdf]" markdown boto3 
```

## Langchain RAG application

```
$ python imis.py
$ python imis_chroma.py
```

## ToDo

- Create AWS Infrastructure
    - Create CDK App
    - Create S3 Bucket with PDF file
    - Create OpenSearch Serverless
    - Create Lambda Functions/API for create and query
- Update Function
    - Update to use DirectoryLoader() to read all PDF files from S3
    - Split function to create and query
- Frontend
    - Build frontend to call API
