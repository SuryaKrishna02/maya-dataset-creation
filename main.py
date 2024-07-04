import cohere 
from utils.constants import COHERE_API_KEY, MODEL_NAME

co = cohere.Client(
  api_key=COHERE_API_KEY
) 

seed = 42
top_k = 40
top_p = 0.99
temperature = 0
message = "Hello What are you doing?"

response = co.chat( 
  model=MODEL_NAME,
  message=message,
  temperature=temperature,
  k = top_k,
  p = top_p,
  seed=seed
) 

print(response.text)
print(response)