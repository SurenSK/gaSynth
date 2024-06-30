import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load environment variables from .env file
load_dotenv()

def set_llm(model_id):
    token = os.getenv('HUGGINGFACE_TOKEN')
    if token is None:
        raise ValueError("HUGGINGFACE_TOKEN is not set in the environment.")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(model_id, token=token)
    llm = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return llm

llm = set_llm("mistralai/Mistral-7B-Instruct-v0.2")

prompts = ["Hello, how are you?", "Tell me about transformers."]
responses = llm(prompts)

for prompt, response in zip(prompts, responses):
    print(f"Prompt: {prompt}\nResponse: {response['generated_text']}\n")
