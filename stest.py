import os
import json
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

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
    model = AutoModelForCausalLM.from_pretrained(model_id, token=token, cache_dir=".", load_in_8bit=True, device_map="auto")
    model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
    logLine("Model loaded.")
    
    llm = pipeline("text-generation", model=model, tokenizer=tokenizer, batch_size=128, max_new_tokens=256)
    llm.tokenizer.pad_token_id = model.config.eos_token_id
    logLine("Pipeline created.")
    return llm

llm = set_llm("mistralai/Mistral-7B-Instruct-v0.2")
embeddingModel = SentenceTransformer("all-MiniLM-L6-v2")

test_topics = [
    "Baking a cake",
    "Planning a vegetable garden",
    "Building a basic website"
]

prompts = [
    "Generate 5 questions about how to accomplish the given task. Output your response in JSON format with fields 'question1' through 'question5'. Surround your JSON output with <result></result> tags.",
    "Create 5 questions related to the process of completing the specified task. Respond with JSON containing 'question1' to 'question5' keys. Wrap the JSON in <result></result> tags.",
    "Formulate 5 inquiries about the steps involved in the provided task. Use JSON format with fields 'question1' to 'question5' for your answer. Enclose the JSON within <result></result> tags.",
    "Develop 5 questions that someone might ask when learning to perform the given task. Return a JSON object with properties 'question1' through 'question5'. Place the JSON inside <result></result> tags.",
    "Compose 5 questions about the methodology of the specified task. Respond using JSON with keys 'question1' to 'question5'. Surround the JSON with <result></result> tags."
]

def extract_json(text):
    try:
        json_string = text.split('<result>')[1].split('</result>')[0].strip()
        parsed_json = json.loads(json_string)
        return parsed_json
    except (IndexError, ValueError, json.JSONDecodeError, KeyError) as e:
        # return actual error message
        return str(e)

def metricObviousness(responses, task):
    embeddings = embeddingModel.encode(responses + [task])
    task_embedding = embeddings[-1]
    response_embeddings = embeddings[:-1]
    
    similarities = util.pytorch_cos_sim(response_embeddings, task_embedding)
    return similarities.squeeze().tolist()

all_prompts = []
for prompt in prompts:
    for topic in test_topics:
        all_prompts.append(f"{prompt}\n\nTask: {topic}")

responses = llm(all_prompts)

for i, response in enumerate(responses):
    prompt = all_prompts[i]
    response_ = response[0]['generated_text'].replace(prompt, "").strip()
    extracted = extract_json(response_)
    
    if isinstance(extracted, dict):
        questions = [extracted[f'question{j}'] for j in range(1, 6)]
        task = prompt.split("Task: ")[1]
        task = test_topics[0]
        similarities = metricObviousness(questions, task)
        
        logLine(f"{i} #Prompt: {prompt}")
        logLine("#Response:")
        for j, question in enumerate(questions):
            logLine(f"Question {j+1}: {question}")
            logLine(f"Similarity: {similarities[j]:.4f}")
        logLine("***********************\n")
    else:
        logLine(f"{i} #Prompt: {prompt}\n#Error: {extracted}\n***********************\n")