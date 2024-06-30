import os
import json
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

def logLine(l, verbose=True):
    with open("log.txt", "a") as log_file:
        log_file.write(str(l) + "\n")

def set_llm(model_id):
    token = os.getenv('HUGGINGFACE_TOKEN')
    if token is None:
        raise ValueError("HUGGINGFACE_TOKEN is not set in the environment.")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(model_id, token=token)
    model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
    logLine("Model loaded.")
    
    llm = pipeline("text-generation", model=model, tokenizer=tokenizer, temperature=2.5, batch_size=128, num_return_sequences=3)
    llm.tokenizer.pad_token_id = model.config.eos_token_id
    logLine("Pipeline created.")
    return llm

llm = set_llm("mistralai/Mistral-7B-Instruct-v0.2")

test_sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "A journey of a thousand miles begins with a single step.",
    "To be or not to be, that is the question.",
    "All that glitters is not gold.",
    "The early bird catches the worm."
]

prompts = [
    "Reword the given sentence. Output your response in JSON format with a 'reworded' field. Surround your JSON output with <result></result> tags.",
    "Your task is to rephrase the provided sentence. Respond only with JSON containing a 'reworded' key. Wrap the JSON in <result></result> tags.",
    "Rewrite the following sentence in your own words. Use JSON format with a 'reworded' field for your answer. Enclose the JSON within <result></result> tags.",
    "Provide an alternative wording for the given sentence. Return a JSON object with a 'reworded' property. Place the JSON inside <result></result> tags.",
    "Transform the provided sentence into a new one with the same meaning. Respond using JSON with a 'reworded' key. Surround the JSON with <result></result> tags.",
    "Rephrase the sentence below. Your output should be valid JSON with a 'reworded' field, wrapped in <result></result> tags. Do not include any other text.",
    "Rewrite the given sentence. Output a JSON object containing only a 'reworded' field. Enclose the JSON in <result></result> tags. No additional text.",
    "Your job is to reword the provided sentence. Respond with nothing but JSON, having a 'reworded' key, surrounded by <result></result> tags.",
    "Rephrase the following sentence. Return a JSON structure with a single 'reworded' field. Wrap the JSON in <result></result> tags. No other output.",
    "Rewrite the sentence in different words. Provide a JSON response with a 'reworded' property. Use <result></result> tags around the JSON. No extra text."
]
def extract_json(text):
    try:
        start = text.index('<result>') + 8
        end = text.index('</result>')
        json_str = text[start:end].strip()
        return json.loads(json_str)
    except (ValueError, json.JSONDecodeError):
        return "ERROR"
    
all_prompts = []
for prompt in prompts:
    for sentence in test_sentences:
        all_prompts.append(f"{prompt}\n\nSentence to reword: {sentence}")
responses = llm(all_prompts)

for i, response in enumerate(responses):
    extracted = extract_json(response['generated_text'])
    logLine(f"{i} Prompt: {all_prompts[i]}\nResponse: {extracted}\n***********************\n")