from collections import deque
import random
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer, util

class Sample:
    eval_functions = []
    llm = None
    tokenizer = None
    embeddingModel = SentenceTransformer("all-MiniLM-L6-v2")

    @classmethod
    def set_llm(cls, model_id, token):
        cls.tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=".", load_in_8bit=True, device_map="auto", do_sample=True, temperature=2.5, num_beams=5, token=token)
        model.eval()
        cls.llm = pipeline("text-generation", model=model, tokenizer=cls.tokenizer, max_new_tokens=200)
        cls.llm.tokenizer.pad_token_id = model.config.eos_token_id

    @classmethod
    def add_eval_function(cls, func):
        cls.eval_functions.append(func)

    def __init__(self, codons, task):
        if not Sample.llm:
            raise ValueError('LLM model not set. Use Sample.set_llm() to set the LLM model.')
        if not Sample.eval_functions:
            raise ValueError('No evaluation functions set. Use Sample.add_eval_function() to add evaluation functions.')
        self.codons = codons
        self.sysPrompt = ' '.join(codons)
        self.task = task

        prompt = f"{self.sysPrompt}\nYou need to perform the task of {self.task}."
        self.response = Sample.llm(prompt)[0]['generated_text'].replace(prompt, "")
        self.scores = [func(self.response, self.task) for func in Sample.eval_functions]

    def to_dict(self):
        return {
            'codons': self.codons,
            'prompt': self.sysPrompt,
            'response': self.response,
            'scores': self.scores
        }
    
    def __lt__(self, other):
        if not isinstance(other, Sample):
            return NotImplemented
        return [self_score < other_score for self_score, other_score in zip(self.scores, other.scores)]

    def __gt__(self, other):
        if not isinstance(other, Sample):
            return NotImplemented
        return [self_score > other_score for self_score, other_score in zip(self.scores, other.scores)]
    
    def __str__(self) -> str:
        return f'Prompt: {self.sysPrompt}\nResponse: {self.response}\nScores: {self.scores}'

def reformFront(P, A):
    P_ = deque()
    while P:
        temp = P.popleft()
        if all(temp < A):  # A dominates temp
            continue
        elif all(A < temp):  # Temp dominates A
            P_.append(temp)
            return P_.extend(P)
        else:
            P_.append(temp)
    P_.append(A)  # Add the new sample to the front
    return P_

def sampleFront(P, n):
    return random.sample(list(P), n)

def mut(a_codon):
    prompt = f"You are a helpful AI writing assistant. Reword this sentence without changing the meaning: {a_codon}"
    return Sample.llm(prompt)[0]['generated_text'].replace(prompt, "").strip()

def mutateFront(P):
    A = sampleFront(P, 1)[0]
    nCodons = [a_codon if random.random() > 0.5 else mut(a_codon) for a_codon in A.codons]
    return Sample(nCodons, A.task)

def breedFrontDet(P):
    A, B = sampleFront(P, 2)
    nCodons = [a_codon if a_score > b_score else b_codon for (a_codon, a_score), (b_codon, b_score) in zip(zip(A.codons, A.scores), zip(B.codons, B.scores))]
    return Sample(nCodons, A.task)

def breedFrontStoch(P):
    A, B = sampleFront(P, 2)
    nCodons = [a_codon if random.random() > 0.5 else b_codon for a_codon, b_codon in zip(A.codons, B.codons)]
    return Sample(nCodons, A.task)

def save_to_jsonl(queue, filename):
    totLen = len(queue)
    with open(filename, 'w') as f:
        while queue:
            sample = queue.popleft()
            print(f"Sample {totLen - len(queue)}/{totLen}")
            print(f"{sample}")
            json_string = json.dumps(sample.to_dict())
            f.write(json_string + '\n')

# Evaluation metrics
def lengthMetric(response, task):
    questions = re.sub(r'\d+\.\s+|User \d+:\s+', '', response).split('?')
    questions_ = [q_.strip() + '?' for q_ in questions if q_.strip()]
    return max(1, len(questions_) / 5)

def metricObviousness(response, task):
    embeddings = Sample.embeddingModel.encode([response, task])
    return max(util.pytorch_cos_sim(embeddings[0], embeddings[1]).item(), 0)

def correctnessMetric(response, task):
    return 0.8

# Set up the LLM and evaluation functions
Sample.set_llm("mistralai/Mistral-7B-Instruct-v0.2", "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")
Sample.add_eval_function(lengthMetric)
Sample.add_eval_function(metricObviousness)
Sample.add_eval_function(correctnessMetric)

# Main execution
P = deque()
reqOps = 100
mutProb = 0.5

iniFormatCodon = "Ask 5 questions in total. List and number some questions that will be relevant to accomplishing the task. Do not respond with anything other than questions. "
iniObviousnessCodon = "Under no circumstances are you to mention any part of what you have just been told."
iniCompletenessCodon = "Do not ask questions about other tasks. Do not ask questions that are irrelevant to the task."
iniCodons = [iniFormatCodon, iniObviousnessCodon, iniCompletenessCodon]
iniTask = "bake a cake"
P.append(Sample(iniCodons, iniTask))

for opNum in range(reqOps):
    op = mutateFront if len(P) < 2 or random.random() < mutProb else breedFrontDet
    nSample = op(P)
    P = reformFront(P, nSample)
    print(f"Op {opNum}: {len(P)} samples in front.")
    # if nSample.scores all over 0.8, then break
print("***FINISHED***")
save_to_jsonl(P, 'output.jsonl')
print("***FINISHED***")