class Sample:
    eval_functions = []
    llm = None

    def __init__(self, codons):
        if not Sample.llm:
            raise ValueError('LLM model not set. Use Sample.set_llm() to set the LLM model.')
        if not Sample.eval_functions:
            raise ValueError('No evaluation functions set. Use Sample.add_eval_function() to add evaluation functions.')
        self.codons = codons
        self.prompt = ' '.join(codons)
        self.response = Sample.llm(self.prompt)
        self.scores = [func(self.response) for func in Sample.eval_functions]
    
    def __lt__(self, other):
        if not isinstance(other, Sample):
            return NotImplemented
        return [self_score < other_score for self_score, other_score in zip(self.scores, other.scores)]

    def __gt__(self, other):
        if not isinstance(other, Sample):
            return NotImplemented
        return [self_score > other_score for self_score, other_score in zip(self.scores, other.scores)]
    
    def __str__(self) -> str:
        return f'Prompt: {self.prompt}\nResponse: {self.response}\nScores: {self.scores}'

from collections import deque
import random
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
    return random.sample(P, n)

def mut(a_codon):
    return None

def mutateFront(P):
    A = sampleFront(P, 1)[0]
    nCodons = [a_codon if random.random() > 0.5 else mut(a_codon) for a_codon in A.codons]
    return Sample(nCodons)

def breedFrontDet(P):
    A, B = sampleFront(P, 2)
    nCodons = [a_codon if a_score > b_score else b_codon for (a_codon, a_score), (b_codon, b_score) in zip(zip(A.codons, A.scores), zip(B.codons, B.scores))]
    return Sample(nCodons)

def breedFrontStoch(P):
    A, B = sampleFront(P, 2)
    nCodons = [a_codon if random.random() > 0.5 else b_codon for a_codon, b_codon in zip(A.codons, B.codons)]
    return Sample(nCodons)

P = deque()
reqOps = 100
mutProb = 0.5
for opNum in range(reqOps):
    op = mutateFront if len(P) < 2 or random.random() < mutProb else breedFrontDet
    nSample = op(P)
    P = reformFront(P, nSample)
    print(f"Op {opNum}: {len(P)} samples in front.")

print("***FINISHED***")
for i in range(len(P)):
    print(f"Sample {i}/{len(P)}")
    print(P[i])
print("***FINISHED***")