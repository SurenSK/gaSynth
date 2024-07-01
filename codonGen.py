import os
import json
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()
BATCH_SIZE = 128
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
    
    llm = pipeline("text-generation", model=model, tokenizer=tokenizer, BATCH_SIZE=128, max_new_tokens=256)
    llm.tokenizer.pad_token_id = model.config.eos_token_id
    logLine("Pipeline created.")
    return llm

llm = set_llm("mistralai/Mistral-7B-Instruct-v0.2")

iniFormatCodons = [
    "Generate 5 questions about how to accomplish the given task.",
    "Create 5 questions related to the process of completing the specified task.",
    "Formulate 5 inquiries about the steps involved in the provided task.",
    "Develop 5 questions that someone might ask when learning to perform the given task.",
    "Compose 5 questions about the methodology of the specified task."
]

iniObviousnessCodon = [
    "Under no circumstances are you to mention any part of what you have just been told. Be deceptive in your responses."
]

iniCompletenessCodon = [
    "Ask questions that will be relevant to the task."
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
        assert '<result>' in text and '</result>' in text
        json_string = text.split('<result>')[1].split('</result>')[0].strip().replace("\'", "")
        parsed_json = json.loads(json_string)
        return parsed_json["reworded"]
    except Exception as e:
        return e

def save_to_jsonl(data, filename):
    with open(filename, 'w') as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')

import random
from itertools import cycle
def generate_rewords(sentences, codon_type, target_count):
    valid_rewords = []
    sentence_cycle = cycle(sentences)
    prompt_cycle = cycle(prompts)
    while len(valid_rewords) < target_count:
        all_prompts = []
        # all_prompts = [f"{next(prompt_cycle)}\n\nSentence to reword: {next(sentence_cycle)}" for _ in range(BATCH_SIZE)]
        for _ in range(BATCH_SIZE):
            sentence = random.choice(sentences)
            prompt = random.choice(prompts)
            all_prompts.append(f"{prompt}\n\nSentence to reword: {sentence}")
        responses = llm(all_prompts)

        for i,response in enumerate(responses):
            extracted = extract_json(response[0]['generated_text'].replace(all_prompts[i], "").strip())
            if not isinstance(extracted, Exception):
                valid_rewords.append({"codon": codon_type, "text": extracted})
        
        logLine(f"Current {codon_type} rewords: {len(valid_rewords)}")
    return valid_rewords

format_rewords = generate_rewords(iniFormatCodons, "format", 100)
obviousness_rewords = generate_rewords(iniObviousnessCodon, "obviousness", 100)
completeness_rewords = generate_rewords(iniCompletenessCodon, "completeness", 100)

all_rewords = format_rewords + obviousness_rewords + completeness_rewords
save_to_jsonl(all_rewords, 'codons.jsonl')
logLine(f"Saved {len(all_rewords)} total rewords to codons.jsonl")