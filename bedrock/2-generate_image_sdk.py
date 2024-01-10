import base64
import os
import random
import boto3
import json

prompt_data = """
A high-res 4k HDR photo of a golden retriever puppy running on a beach.
Action shot, blue sky, white sand, and a big smile. Cinematic film quality.
"""
# prompt_data = """
# A high-res 4k HDR photo of a golden retriever puppy running with Disney's Magic Kingdom in the distant background.
# Action shot, blue sky, green grass, and a big smile. Cinematic film quality.
# """

seed = random.randint(0, 100000)
payload = {
    "text_prompts": [{"text": prompt_data}],
    "cfg_scale": 12,
    "seed": seed,
    "steps": 80
}

# Create the client and invoke the model.
session = boto3.Session(profile_name='bach-dev', region_name='us-east-1')
bedrock = session.client(service_name='bedrock-runtime')
body = json.dumps(payload) 
model_id = "stability.stable-diffusion-xl-v0"
response = bedrock.invoke_model(
    body=body,
    modelId=model_id,
    accept="application/json",
    contentType="application/json"
)

# Get the image from the response. It is base64 encoded.
response_body = json.loads(response.get("body").read())
artifact = response_body.get("artifacts")[0]
image_encoded = artifact.get("base64").encode("utf-8")
image_bytes = base64.b64decode(image_encoded)

# Save the image to a file in the output directory.
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
file_name = f"{output_dir}/generated_{seed}.png"
with open(file_name, "wb") as f:
    f.write(image_bytes)
