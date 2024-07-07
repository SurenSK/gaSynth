import time
import json
import random
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from llmHandler import LLMHandler

def logLine(l):
    with open("handler.txt", "a") as log_file:
        log_file.write(str(l) + "\n")

tSetup = time.time()
llm_handler = LLMHandler("mistralai/Mistral-7B-Instruct-v0.2", batch_size=256)
tSetup = time.time() - tSetup
logLine(f"t+{tSetup:.2f}s - LLMHandler setup complete.")

class Sample:
    def __init__(self, relevance_codon: str, avoidance_codon: str):
        self.evalWidth = 32
        self.relevance_codon = relevance_codon
        self.avoidance_codon = avoidance_codon
        self.questions = [None]*self.evalWidth
        self.relevance_evals = [None]*self.evalWidth
        self.obviousness_evals = [None]*self.evalWidth
        self.fitness: float = 0.0
    
    def requestQuestions(self):
        for i in range(self.evalWidth):
            self.questions[i] = llm_handler.request(f"Could these questions be relevant to the task of {self.task}?\n" + " ".join(self.questions), {"rationale": str, "relevant": str})
    
    def setQuestions(self):
        for i in range(self.evalWidth):
            self.questions[i] = [self.questions[i]["question1"], self.questions[i]["question2"], self.questions[i]["question3"], self.questions[i]["question4"], self.questions[i]["question5"]]

    def requestRelevanceEvals(self):
        for i,qset in enumerate(self.questions):
            self.relevance_evals[i] = llm_handler.request(f"Could these questions be relevant to the task of {self.task}?\n" + " ".join(qset), {"rationale": str, "relevant": str})

    def setRelevanceEvals(self):
        for i in range(self.evalWidth):
            resp = self.relevance_evals[i]["relevant"]
            self.relevance_evals[i] = resp.lower() == "yes" or resp == "true"

    def setObviousnessScores(self, relevance_scores: List[float]):
        for i,qset in enumerate(self.questions):
            embeddings = self.embedding_model.encode(qset + [f"I want to {self.task}"])
            task_embedding = embeddings[-1]
            question_embeddings = embeddings[:-1]
            self.obviousness_evals[i] = sum([1 - cosine(qe, task_embedding) for qe in question_embeddings]) / len(question_embeddings)

    def getFitness(self):
        if self.relevance_evals is None or self.obviousness_evals is None:
            raise ValueError("Relevance and obviousness scores must be calculated before fitness can be calculated.")
        # pairwise multiplication of relevance and obviousness scores
        relevant_obviousness = [o for o, r in zip(self.obviousness_evals, self.relevance_evals) if r]
        self.fitness = sum(relevant_obviousness) if relevant_obviousness else 0.0
        return self.fitness
    
    def __lt__(self, other):
        return self.fitness < other.fitness

from itertools import product
def formNewPop(samples):
    avoidance_codons = [s.avoidance_codon for s in samples]
    obviousness_codons = [s.obviousness_codon for s in samples]
    