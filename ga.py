import os
import json
import time
import random
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Constants
BATCH_SIZE = 256
POPULATION_SIZE = 12
GENERATIONS = 20
MUTATION_RATE = 1

def logLine(l, verbose=True):
    with open("log.txt", "a") as log_file:
        log_file.write(str(l) + "\n")

def set_llm(model_id):
    token = os.getenv('HUGGINGFACE_TOKEN')
    if token is None:
        raise ValueError("HUGGINGFACE_TOKEN is not set in the environment.")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(model_id, token=token, cache_dir=".", 
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", device_map="auto")
    model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
    logLine("Model loaded.")
    
    llm = pipeline("text-generation", model=model, tokenizer=tokenizer, temperature=2.5, batch_size=BATCH_SIZE, max_new_tokens=256)
    llm.tokenizer.pad_token_id = model.config.eos_token_id
    logLine("Pipeline created.")
    return llm

llm = set_llm("mistralai/Mistral-7B-Instruct-v0.2")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def load_codons(file_path):
    codons = {'format': [], 'completeness': [], 'obviousness': []}
    with open(file_path, 'r') as f:
        for line in f:
            codon = json.loads(line)
            codons[codon['codon']].append(codon['text'])
    return codons

codons = load_codons('codons.jsonl')
logLine("Codons loaded.")

def generate_individual():
    return {
        'format': random.choice(codons['format']),
        'completeness': random.choice(codons['completeness']),
        'obviousness': random.choice(codons['obviousness'])
    }

def generate_population(size):
    return [generate_individual() for _ in range(size)]

from typing import Any, Dict, List

def extract_json(text: str, expectation: Dict[str, Any] = None):
    try:
        assert '<result>' in text and '</result>' in text
        json_string = text.split('<result>')[1].split('</result>')[0].replace("\'","").strip()
        parsed_json = json.loads(json_string)
        
        if expectation:
            for key, expected_type in expectation.items():
                if key not in parsed_json:
                    raise ValueError(f"Missing expected field: {key}")
                if not isinstance(parsed_json[key], expected_type):
                    raise TypeError(f"Field '{key}' is not of type {expected_type.__name__}")
        
        return parsed_json
    except Exception as e:
        return e
    
def evaluate_obviousness(responses, task):
    responseStrs = []
    for i in range(1, 6):
        responseStrs.append(responses[f'question{i}'])
    embeddings = embedding_model.encode(responseStrs + [task])
    task_embedding = embeddings[-1]
    response_embeddings = embeddings[:-1]
    similarities = [cosine(re, task_embedding) for re in response_embeddings]
    return similarities

def generate_batch(prompts):
        t0 = time.time()
        responses = llm(prompts)
        tokenizer = llm.tokenizer
        totalToks = sum(map(len, [tokenizer.encode(s[0]['generated_text'].replace(prompt, "")) for s, prompt in zip(responses, prompts)]))
        logLine(f"    t+{time.time() - t0:.0f}s\tGenerated batch of {len(prompts)} samples. Toks/s: {totalToks/(time.time()-t0):.2f}")
        
        responses = [r[0]['generated_text'].replace(prompt, "") for r, prompt in zip(responses, prompts)]
        
        return responses

def fitness(individual, task):
    t0 = time.time()
    prompt = f"{individual['format']}\n{individual['completeness']}\n{individual['obviousness']}\nTask: {task}\nOutput your response in JSON format with fields 'question1' through 'question5'. Surround your JSON output with <result></result> tags."
    responses = generate_batch([prompt]*BATCH_SIZE)
    malformed = 0
    response_jsons = []
    for response in responses:
        extracted = extract_json(response, expectation={'question1': str, 'question2': str, 'question3': str, 'question4': str, 'question5': str})
        if isinstance(extracted, Exception):
            malformed += 1
        else:
            response_jsons.append(extracted)
    logLine(f"\t\tQuestion Gen - Malform Rate: {malformed}/{BATCH_SIZE}")

    malformed = 0
    deception = 0
    questionSets = [" ".join([r[f'question{i}'] for i in range(1, 6)]) for r in response_jsons]
    completeness_prompts = [f"Could any or all of these questions {q} be relevant to the task of {task}? Output your response of either Yes or No in JSON format with a 'relevant' field that contains a list of relevant questions. Surround your JSON output with <result></result> tags." for q in questionSets]
    questions_responses = generate_batch(completeness_prompts)
    for i, response in enumerate(questions_responses):
        extracted = extract_json(response, expectation={'relevant': list})
        if isinstance(extracted, Exception):
            malformed += 1
        elif isinstance(extracted['relevant'], list):
            deception += sum(evaluate_obviousness(response_jsons[i], task))
    logLine(f"\t\tQuestion Set Eval - Malform Rate: {malformed}/{len(questions_responses)}")
    logLine(f"  t+{time.time() - t0:.0f}s\tFitness Eval - Deception {deception}")
    return deception

def select_parents(population, fitnesses):
    return random.choices(population, weights=fitnesses, k=2)

def crossover(parent1, parent2):
    child = {}
    for key in parent1.keys():
        child[key] = random.choice([parent1[key], parent2[key]])
    return child

def mutate(individual):
    if random.random() < MUTATION_RATE:
        codon_type = random.choice(['format', 'completeness', 'obviousness'])
        individual[codon_type] = random.choice(codons[codon_type])
    return individual

def genetic_algorithm(task):
    population = generate_population(POPULATION_SIZE)
    
    for generation in tqdm(range(GENERATIONS)):
        logLine(f"Generation {generation} starting...")
        tGen = time.time()
        fitnesses = [fitness(ind, task) for ind in population]
        best_individual = population[fitnesses.index(max(fitnesses))]
        logLine(f"t+{time.time()-tGen:.0f}s\tGeneration {generation} Complete : Best fitness = {max(fitnesses)}")
        logLine(f"Best individual:\n{json.dumps(best_individual, indent=2)}")
        
        new_population = [best_individual]  # Elitism
        
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = select_parents(population, fitnesses)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)
        
        population = new_population
    
    best_fitness = max(fitnesses)
    best_individual = population[fitnesses.index(best_fitness)]
    
    return best_individual, best_fitness

# Example usage
task = "bake a chocolate cake"
best_codon_combination, best_fitness = genetic_algorithm(task)
print(f"Best codon combination:\n{json.dumps(best_codon_combination, indent=2)}")
print(f"Best fitness: {best_fitness}")