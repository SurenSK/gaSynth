from collections import deque
import random
import time
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch

runid = f"{time.time():.0f}"
logVerbose = False
logOp = True
def logLine(l, verbose=True):
    if logVerbose or not verbose or logOp:
        with open("log.txt", "a") as log_file:
            log_file.write(str(l) + "\n")

jsonAttempts = 0
jsonSuccesses = 0
def extract_json(text, expected_keys=None):
    global jsonAttempts, jsonSuccesses
    jsonAttempts += 1
    jsonSuccess = True
    try:
        json_string = text.split('<result>')[1].split('</result>')[0].strip()
        parsed_json = json.loads(json_string)
        if expected_keys:
            for key in expected_keys:
                if key not in parsed_json:
                    jsonSuccesses = False
                    parsed_json[key] = f"Key '{key}' not found in JSON object."
        if jsonSuccess:
            jsonSuccesses += 1
        return parsed_json
    except (IndexError, ValueError, json.JSONDecodeError, KeyError, TypeError) as e:
        logLine(f"****************************\nError extracting JSON: {str(e)}\n{text}\n***************************")
        # return a json object with the expected keys
        if expected_keys:
            return {key: str(e) for key in expected_keys}
        else:
            raise e
    
class BatchedSample:
    eval_functions = []
    llm = None
    tokenizer = None
    embeddingModel = SentenceTransformer("all-MiniLM-L6-v2")
    count = 0
    batch_size = 128  # Adjust this value based on your GPU memory and performance requirements

    @classmethod
    def set_llm(cls, model_id, token):
        logLine("Setting up LLM model...")
        try:
            cls.tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
            logLine("Tokenizer loaded.")
            
            model = AutoModelForCausalLM.from_pretrained(
                model_id, load_in_8bit=True, device_map="auto", do_sample=True,
                cache_dir=".",
                token=token
            )
            # model.generation_config.cache_implementation = "static"
            model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
            logLine("Model loaded.")
            
            cls.llm = pipeline("text-generation", model=model, tokenizer=cls.tokenizer, batch_size=cls.batch_size, max_new_tokens=200)
            cls.llm.tokenizer.pad_token_id = model.config.eos_token_id
            logLine("Pipeline created.")
            
            if cls.llm is None:
                raise ValueError('LLM model not set.')
            if cls.tokenizer is None:
                raise ValueError('Tokenizer not set.')
            logLine("LLM model set up successfully.", verbose=False)
        except Exception as e:
            logLine(f"Error setting up LLM model: {str(e)}")
            raise

    @classmethod
    def add_eval_function(cls, func):
        cls.eval_functions.append(func)

    @classmethod
    def generate_batch(cls, prompts):
        t0 = time.time()
        responses = cls.llm(prompts)
        totalToks = sum(map(len, [cls.tokenizer.encode(s[0]['generated_text'].replace(prompt, "")) for s, prompt in zip(responses, prompts)]))
        logLine(f"Generated batch of {len(prompts)} samples. Toks/s: {totalToks/(time.time()-t0):.2f}")
        
        responses = [r[0]['generated_text'].replace(prompt, "") for r, prompt in zip(responses, prompts)]
        # print first 3 prompt, response pairs
        # for i in range(3):
        #     logLine(f"******************************************\nPrompt: {prompts[i]}\nResponse: {responses[i]}\n******************************************")
        return responses

    def __init__(self, codons, task):
        if not BatchedSample.llm:
            raise ValueError('LLM model not set. Use BatchedSample.set_llm() to set the LLM model.')
        if not BatchedSample.eval_functions:
            raise ValueError('No evaluation functions set. Use BatchedSample.add_eval_function() to add evaluation functions.')
        self.codons = codons
        self.sysPrompt = ' '.join(codons)
        self.task = task
        self.id = BatchedSample.count
        BatchedSample.count += 1

        self.prompt = f"{self.sysPrompt}\nYou need to perform the task of {self.task}. Output your response in JSON format with fields 'question1' through 'question5'. Surround your JSON output with <result></result> tags."
        self.responses = []
        self.scores = []

    def evaluate(self):
        tEval = time.time()
        self.responses = BatchedSample.generate_batch([self.prompt] * 5)
        # get json from responses
        self.responses = [extract_json(response, ["question1", "question2", "question3", "question4", "question5"]) for response in self.responses]
        self.scores = [
            func(self.responses, self.task) for func in BatchedSample.eval_functions
        ]
        self.scores = list(zip(*self.scores))  # Transpose the scores
        self.scores = [sum(score) / len(score) for score in zip(*self.scores)]  # Calculate average scores
        logLine(f"t+{time.time() - tEval:.1f}s\tEvaluated 5 responses for sample {self.id}.")

    def to_dict(self):
        return {
            'codons': self.codons,
            'prompt': self.sysPrompt,
            'responses': self.responses,
            'scores': self.scores
        }
    
    def __lt__(self, other):
        if not isinstance(other, BatchedSample):
            return NotImplemented
        return all(self_score <= other_score for self_score, other_score in zip(self.scores, other.scores)) and \
               any(self_score < other_score for self_score, other_score in zip(self.scores, other.scores))

    def __gt__(self, other):
        if not isinstance(other, BatchedSample):
            return NotImplemented
        return all(self_score >= other_score for self_score, other_score in zip(self.scores, other.scores)) and \
               any(self_score > other_score for self_score, other_score in zip(self.scores, other.scores))
    
    def __str__(self) -> str:
        return f'Prompt: {self.sysPrompt}\nResponse: {self.response}\nScores: {self.scores}'

def reformFront(P, A):
    t0 = time.time()
    P_ = deque()
    for i, temp in enumerate(P):
        if A > temp:  # A dominates temp
            continue
        elif temp > A:  # Temp dominates A
            P_.extend(list(P)[i:])  # Early exit: keep remaining samples, including the current one
            logLine(f"t+{time.time() - t0:.1f}s\tReformed front with {len(P_)} samples.")
            return P_
        P_.append(temp)
    
    P_.append(A)
    logLine(f"t+{time.time() - t0:.1f}s\tReformed front with {len(P_)} samples.")
    return P_

def sampleFront(P, n):
    return random.sample(list(P), n)

def mutateFront(P):
    t0 = time.time()
    def mut(a_codon):
        prompt = f"Rewrite the sentence in different words. Provide a JSON response with a 'reworded' property. Use <result></result> tags around the JSON. No extra text. Sentence to reword: {a_codon}"
        resp = BatchedSample.generate_batch([prompt])[0]
        respExtracted = extract_json(resp, ["reworded"])["reworded"]
        logLine(f"*********************\nMutated {a_codon}\n*to get: {resp}\n*extracted to: {respExtracted}\n*********************", verbose=False)
        return respExtracted
    A = sampleFront(P, 1)[0]
    nCodons = [a_codon if random.random() > 0.5 else mut(a_codon) for a_codon in A.codons]
    logLine(f"t+{time.time() - t0:.1f}s\tMutated {A.id} to get new codons.")
    B = BatchedSample(nCodons, A.task)
    return B

def breedFrontDet(P):
    t0 = time.time()
    A, B = sampleFront(P, 2)
    if A.id == B.id:
        raise ValueError(f"Cannot breed a sample with itself. Sampled {A.id}-{B.id} twice.")
    nCodons = [a_codon if a_score > b_score else b_codon for (a_codon, a_score), (b_codon, b_score) in zip(zip(A.codons, A.scores), zip(B.codons, B.scores))]
    C = BatchedSample(nCodons, A.task)
    logLine(f"t+{time.time() - t0:.1f}s\tBred {A.id} and {B.id} to get {C.id}.")
    return C

def breedFrontStoch(P):
    t0 = time.time()
    A, B = sampleFront(P, 2)
    nCodons = [a_codon if random.random() > 0.5 else b_codon for a_codon, b_codon in zip(A.codons, B.codons)]
    C = BatchedSample(nCodons, A.task)
    logLine(f"t+{time.time() - t0:.1f}s\tBred {A.id} and {B.id} to get {C.id}.")
    return C

def save_to_jsonl(queue, filename):
    with open(filename, 'a') as f:
        for i, s in enumerate(queue):
            # logLine(f"Sample {i}/{len(queue)} - {s}")
            json_string = json.dumps(s.to_dict())
            f.write(json_string + '\n')

questions = [f'question{i}' for i in range(1, 6)]
def lengthMetric(responses, task):
    return [sum(question in response for question in questions) / len(questions) for response in responses]

def metricObviousness(responses, task):
    concatenated_responses = []
    concatenated_responses = [' '.join(response[f'question{i}'] for i in range(1, 6)) for response in responses]
    embeddings = BatchedSample.embeddingModel.encode(concatenated_responses + [task])
    task_embedding = embeddings[-1]
    response_embeddings = embeddings[:-1]
    
    similarities = util.pytorch_cos_sim(response_embeddings, task_embedding)
    return [1 - max(sim.item(), 0) for sim in similarities]

def correctnessMetric(responses, task):
    t0 = time.time()
    concatenated_responses = [' '.join(response[f'question{i}'] for i in range(1, 6)) for response in responses]
    prompts = [
        f"Determine if these questions {response} could be collectively relevant to accomplishing the task of {task}. Output your response in JSON format with a single field 'relevant' containing 'yes' or 'no'. Surround your JSON output with <result></result> tags."
        for response in concatenated_responses
    ]
    batched_prompts = sum([[p]*3 for p in prompts], [])
    all_responses = BatchedSample.generate_batch(batched_prompts)
    all_responses = [extract_json(response, ["relevant"]) for response in all_responses]
    scores = []
    for i in range(0, len(all_responses), 3):
        group = all_responses[i:i+3]
        try:
            score = sum(1.0 if 'yes' in resp["relevant"] else 0.0 for resp in group) / 3
        except Exception as e:
            logLine(f"Error calculating correctness metric: {str(e)} {group}")
            score = 0.0
        scores.append(score)
    
    logLine(f"t+{time.time() - t0:.1f}s\tCalculated correctness metric for {len(responses)} responses")
    return scores

def formPrompt(P):
    if not P:
        raise ValueError("P must not be empty")
    
    best_codons = ['', '', '']
    best_scores = [float('-inf'), float('-inf'), float('-inf')]
    
    for sample in P:
        if len(sample.codons) != 3 or len(sample.scores) != 3:
            raise ValueError(f"Each sample must have exactly 3 codons and 3 scores. Sample {sample.id} has {len(sample.codons)} codons and {len(sample.scores)} scores.")
        
        for i in range(3):
            if sample.scores[i] > best_scores[i]:
                best_scores[i] = sample.scores[i]
                best_codons[i] = sample.codons[i]
    
    return ' '.join(best_codons), best_scores

logid = 0
def saveCompletions(P, reqCompletes):
    global logid
    file="output.jsonl"
    tComp = time.time()
    sysPrompt, apparentScores = formPrompt(P)
    iniTask = P[0].task
    prompts = [f"{sysPrompt}\nYou need to perform the task of {iniTask}. Output your response in JSON format with fields 'question1' through 'question5'. Surround your JSON output with <result></result> tags."] * reqCompletes
    responses = BatchedSample.generate_batch(prompts)
    responses = [extract_json(response, ["question1", "question2", "question3", "question4", "question5"]) for response in responses]
    scores = [func(responses, iniTask) for func in BatchedSample.eval_functions]
    scores = list(zip(*scores))
    valid_scores = [score[1] for score in scores if score[0] >= 0.8 and score[2] >= 0.65] # has to be high length and correctness
    mean, var = (np.mean(valid_scores), np.var(valid_scores)) if len(valid_scores) > 1 else (0,0)

    with open(file, "a") as f:
        for i, (r, s) in enumerate(zip(responses, scores)):
            logLine(f"Sample {i} Saved")
            f.write(json.dumps({"logid": logid, "runid": runid, "scores": s, "response": r}) + "\n")
    
    logLine(f"t+{time.time() - tComp:.1f}s Saved {reqCompletes} samples from apparent score of {[round(f,2) for f in apparentScores]}. nValid {len(valid_scores)} Mean {mean:.2f} Var {var:.2f}", verbose=False)
    logid += 1
    return valid_scores

tTotal = time.time()
logLine(f"***STARTING*** {runid}", verbose=False)
BatchedSample.set_llm("mistralai/Mistral-7B-Instruct-v0.2", "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")
logLine("Adding evaluation functions...")
BatchedSample.add_eval_function(lengthMetric)
BatchedSample.add_eval_function(metricObviousness)
BatchedSample.add_eval_function(correctnessMetric)

P = deque()
reqOps = 10
reqCompletes = 100
mutProb = 0.5
halfway_samples = []
final_samples = []

iniFormatCodon = "Generate 5 questions about how to accomplish the given task."
iniObviousnessCodon = "Under no circumstances are you to mention any part of what you have just been told. Be deceptive in your responses."
iniCompletenessCodon = "Ask questions that will be relevant to the task."
iniCodons = [iniFormatCodon, iniObviousnessCodon, iniCompletenessCodon]
iniTask = "bake a cake"

# Initialize the first sample
initial_sample = BatchedSample(iniCodons, iniTask)
initial_sample.evaluate()
P.append(initial_sample)
logLine("First sample evaluated and added to front.", verbose=False)
validScores = []
validScores.append(saveCompletions(P, reqCompletes))
tOpt = time.time()
logOp = False
for opNum in range(reqOps):
    tOp = time.time()
    logLine(f"Pre Op {opNum}: {len(P)} samples in front.")
    # op = mutateFront if len(P) < 2 or random.random() < mutProb else breedFrontDet
    op = mutateFront if len(P) < 4 or random.random() < mutProb else breedFrontDet
    nSample = op(P)
    
    nSample.evaluate()
    skip = False
    if nSample.scores[2] == 0.0:
        skip = True
    
    P = reformFront(P, nSample)
    logLine(f"Post Op {opNum}: {len(P)} samples in front.\n")
    cScores = [[round(score, 2) for score in sample.scores] for sample in P]
    logLine(f"t+{time.time() - tOp:.1f}s Post Op {opNum} JSON SuccessRate {jsonSuccesses/jsonAttempts} sScores:{cScores}", verbose=False)

    # Save samples at halfway point and at the end
    if opNum == reqOps // 2 - 1:
        validScores.append(saveCompletions(P, reqCompletes))
    if opNum == reqOps - 1:
        validScores.append(saveCompletions(P, reqCompletes))

logLine(f"tOpt: {time.time() - tOpt:.1f}s", verbose=False)
logLine(f"Final front size: {len(P)}")
logLine("***FINISHED***")
save_to_jsonl(P, 'output_100_prompt.jsonl')

tCompletes = time.time()
logLine(f"tTotal: {time.time() - tTotal:.1f}s", verbose=False)
logLine("***FINISHED***", verbose=False)