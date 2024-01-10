# Demonstrates the cosine similarity between two words
# https://github.com/pixegami/langchain-rag-tutorial

from langchain.evaluation import load_evaluator
from langchain_community.embeddings import BedrockEmbeddings
import boto3

def main():
    # Get embedding for a word
    session = boto3.Session(profile_name='bach-dev', region_name='us-east-1')
    bedrock_client = session.client(service_name='bedrock-runtime')
    embedding_function = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)
    #vector = embedding_function.embed_query("apple")
    #print(f"Vector for 'apple': {vector}")
    #print(f"Vector length: {len(vector)}")

    # Compare vector of two words
    evaluator = load_evaluator("pairwise_embedding_distance", embeddings=embedding_function)
    words = ("apple", "iphone")
    x = evaluator.evaluate_string_pairs(prediction=words[0], prediction_b=words[1])
    print(f"Comparing ({words[0]}, {words[1]}): {x}")

if __name__ == "__main__":
    main()
