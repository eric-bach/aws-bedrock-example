from langchain.llms.bedrock import Bedrock
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

llm = Bedrock(credentials_profile_name="bach-dev", model_id="anthropic.claude-instant-v1")

prompt = "What is the largest city in the world?"

response_text = llm.predict(prompt)

print(response_text)