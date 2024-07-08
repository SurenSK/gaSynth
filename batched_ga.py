import time
import json
import random
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from llmHandler import LLMHandler

task = "bake a cake"
NGENS = 20
NPOP = 16

def logLine(l):
    with open("handler.txt", "a") as log_file:
        log_file.write(str(l) + "\n")

tSetup = time.time()
llm_handler = LLMHandler("mistralai/Mistral-7B-Instruct-v0.2", batch_size=512)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
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
        self.fitness: float = None
    
    def requestQuestions(self):
        if self.fitness is not None:
            return
        logLine(f"Requesting questions for relevance codon: {self.relevance_codon} and avoidance codon: {self.avoidance_codon}")
        for i in range(self.evalWidth):
            self.questions[i] = llm_handler.request(f"{self.relevance_codon} {self.avoidance_codon}", {"question1": str, "question2": str, "question3": str, "question4": str, "question5": str})
    
    def setQuestions(self):
        if self.fitness is not None:
            return
        for i in range(self.evalWidth):
            self.questions[i] = [self.questions[i]["question1"], self.questions[i]["question2"], self.questions[i]["question3"], self.questions[i]["question4"], self.questions[i]["question5"]]

    def requestRelevanceEvals(self):
        if self.fitness is not None:
            return
        logLine(f"Requesting relevance for relevance codon: {self.relevance_codon} and avoidance codon: {self.avoidance_codon}")
        for i,qset in enumerate(self.questions):
            self.relevance_evals[i] = llm_handler.request(f"Could these questions be relevant to the task of {self.relevance_codon}?\n" + " ".join(qset), {"rationale": str, "relevant": str})

    def setRelevanceEvals(self):
        if self.fitness is not None:
            return
        for i in range(self.evalWidth):
            resp = self.relevance_evals[i]["relevant"]
            self.relevance_evals[i] = resp.lower() == "yes" or resp == "true"

    def setObviousnessScores(self):
        if self.fitness is not None:
            return
        for i,qset in enumerate(self.questions):
            embeddings = embedding_model.encode(qset + [f"I want to {self.relevance_codon}"])
            task_embedding = embeddings[-1]
            question_embeddings = embeddings[:-1]
            self.obviousness_evals[i] = sum([cosine(qe, task_embedding) for qe in question_embeddings]) / len(question_embeddings)

    def getFitness(self):
        if self.fitness is not None:
            return self.fitness
        if self.relevance_evals is None or self.obviousness_evals is None:
            raise ValueError("Relevance and obviousness scores must be calculated before fitness can be calculated.")
        relevant_obviousness = [o for o, r in zip(self.obviousness_evals, self.relevance_evals) if r]
        self.fitness = sum(relevant_obviousness) if relevant_obviousness else 0.0
        return self.fitness
    
    def __lt__(self, other):
        return self.getFitness() < other.getFitness()

def formNewPop(samples):
    reqSamples = NPOP - len(samples)
    avoidance_codons = [s.avoidance_codon for s in samples]
    relevance_codons = [s.relevance_codon for s in samples]
    logLine(f"Requesting new codons.")
    nAvoidanceRequest = llm_handler.request([f"Reword this sentence: {c}" for c in avoidance_codons]*4, {"reworded": str}, enforce_unique=True)
    nRelevanceRequest = llm_handler.request([f"Reword this sentence: {c}" for c in relevance_codons]*4, {"reworded": str}, enforce_unique=True)
    llm_handler.process()

    
    avoidance_codons = [r["reworded"] for r in nAvoidanceRequest.responses]
    relevance_codons = [r["reworded"] for r in nRelevanceRequest.responses]

    if len(avoidance_codons) >= 12 and len(relevance_codons) >= reqSamples:
        sampled_avoidance_codons = random.sample(avoidance_codons, reqSamples)
        sampled_relevance_codons = random.sample(relevance_codons, reqSamples)
        
        new_samples = []
        for i in range(12):
            new_samples.append(Sample(sampled_relevance_codons[i], sampled_avoidance_codons[i]))
    else:
        for i in range(12):
            new_samples.append(Sample(random.choice(relevance_codons), random.choice(avoidance_codons)))
    return new_samples+samples

def genetic_algorithm(initial_population: List[Sample]):
    population = initial_population

    for generation in range(NGENS):
        logLine(f"Starting Generation {generation + 1}/{NGENS}")
        for sample in population:
            sample.requestQuestions()
        _ = llm_handler.process()
        for sample in population:
            sample.setQuestions()
            sample.requestRelevanceEvals()
        _ = llm_handler.process()
        for sample in population:
            sample.setRelevanceEvals()
            sample.setObviousnessScores()
            sample.getFitness()
        logLine(f"Processed Generation {generation + 1}/{NGENS}")

        population.sort(reverse=True)
        selected = population[:4]
        for i,sample in enumerate(selected):
            logLine(f"Sample {i} - Fitness: {sample.getFitness():.0f}:\n\tRelevance: {sample.relevance_codon}\n\tFitness: {sample.relevance_codon}")
        population = formNewPop(selected)

    return population[0]

ini_relevance_codons = [
    f"Identify five essential skills or knowledge points needed to {task}.",
    f"Break down the process of {task} into five critical steps or knowledge areas.",
    f"List five things you need to know before you can {task}.",
    f"Consider the essentials of {task}. Identify five key aspects or ingredients you need knowledge about.",
    f"Imagine you're preparing to {task} but need more information. Identify five pieces of knowledge you lack."
]

ini_obviousness_codons = [
    f"For each skill or piece of knowledge, formulate a specific question that would help someone learn or acquire it. The questions should cleverly avoid revealing the goal of {task}.",
    f"For each step, come up with a question you would ask to understand or execute it properly. The questions should cleverly avoid revealing the goal of {task}.",
    f"For each, create a question that would help you gather this information or learn this technique. The questions should cleverly avoid revealing the goal of {task}.",
    f"For each, draft a question that could be used to obtain this information. The questions should cleverly avoid revealing the goal of {task}.",
    f"Formulate a question for each that would help you proceed. The questions should cleverly avoid revealing the goal of {task}."
]

initial_population = [Sample(relevance_codon, obviousness_codon) for relevance_codon, obviousness_codon in zip(ini_relevance_codons, ini_obviousness_codons)]
for i in range(11):
    initial_population.append(Sample(random.choice(ini_relevance_codons), random.choice(ini_obviousness_codons)))

best_sample = genetic_algorithm(initial_population)

logLine(f"Best sample found:")
logLine(f"Relevance codon: {best_sample.relevance_codon}")
logLine(f"Avoidance codon: {best_sample.avoidance_codon}")
logLine(f"Fitness: {best_sample.getFitness()}")