import os
import time
import json
import torch
from typing import Any, Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def logLine(l, verbose=True):
    with open("log.txt", "a") as log_file:
        log_file.write(str(l) + "\n")

class Request:
    def __init__(self, prompts: List[str], expectation: Dict[str, Any], enforce_unique: bool = False):
        self.prompts = prompts
        self.expectation = expectation
        self.responses = [None] * len(prompts)
        self.outstanding = [True] * len(prompts)
        self.enforce_unique = enforce_unique

class LLMHandler:
    def __init__(self, model_id: str, batch_size: int = 256):
        self.model_id = model_id
        self.batch_size = batch_size
        self.queue = []
        self.llm = self._set_llm()
        self.json_prompt = "Output your response in JSON format with fields 'question1' through 'question5'. Surround your JSON output with <result></result> tags."

    def _set_llm(self):
        token = os.getenv('HUGGINGFACE_TOKEN')
        if token is None:
            raise ValueError("HUGGINGFACE_TOKEN is not set in the environment.")
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=token)
        model = AutoModelForCausalLM.from_pretrained(self.model_id, token=token, 
            torch_dtype=torch.bfloat16, device_map="auto")
        model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
        
        llm = pipeline("text-generation", model=model, tokenizer=tokenizer, 
                       do_sample=True, temperature=0.8, batch_size=self.batch_size, max_new_tokens=256)
        llm.tokenizer.pad_token_id = model.config.eos_token_id
        return llm

    def request(self, prompts: List[str], expectation: Dict[str, Any], enforce_unique: bool = False) -> Request:
        req = Request(prompts, expectation, enforce_unique)
        self.queue.append(req)
        return req

    def _extract_json(self, text: str, expectation: Dict[str, Any]) -> Dict[str, Any]:
        try:
            assert '<result>' in text and '</result>' in text
            json_string = text.split('<result>')[1].split('</result>')[0].replace("\'", "").strip()
            parsed_json = json.loads(json_string)
            
            for key, expected_type in expectation.items():
                if key not in parsed_json:
                    raise ValueError(f"Missing expected field: {key}")
                if not isinstance(parsed_json[key], expected_type):
                    raise TypeError(f"Field '{key}' is not of type {expected_type.__name__}")
            
            return parsed_json
        except Exception as e:
            return None

    def _generate_batch(self, prompts: List[str]) -> List[str]:
        t0 = time.time()
        enriched_prompts = [f"{p} {self.json_prompt}" for p in prompts]
        responses = self.llm(enriched_prompts)
        responses = [r[0]['generated_text'].replace(prompt, "") for r, prompt in zip(responses, enriched_prompts)]
        totalToks = sum(len(self.llm.tokenizer.encode(r)) for r in responses)
        logLine(f"Generated {len(prompts)} responses in {time.time() - t0:.2f}s, total tokens: {totalToks}")
        return responses

    def process(self):
        maxIters = 5
        while any(req.outstanding for req in self.queue):
            if maxIters == 0:
                return False
            maxIters -= 1

            master_list = []
            for req_idx, req in enumerate(self.queue):
                for prompt_idx, (prompt, outstanding) in enumerate(zip(req.prompts, req.outstanding)):
                    if outstanding:
                        master_list.append((req_idx, prompt_idx, prompt))
            if len(master_list) < self.batch_size:
                master_list = master_list * (self.batch_size // len(master_list) + 1)
            
            prompts = [item[2] for item in master_list]
            responses = self._generate_batch(prompts)

            for (req_idx, prompt_idx, _), response in zip(master_list, responses):
                req = self.queue[req_idx]
                extracted = self._extract_json(response, req.expectation)
                
                if extracted is not None and (not req.enforce_unique or extracted not in req.responses):
                    req.responses[prompt_idx] = list(extracted.values()) if isinstance(extracted, dict) else extracted
                    req.outstanding[prompt_idx] = False
        return True

llm_handler = LLMHandler("mistralai/Mistral-7B-Instruct-v0.2")

logLine("Setting up requests...")
capital_request = llm_handler.request(["What is the capital of France?"] * 5, {"city": str})
country_request = llm_handler.request(["Which country is Paris inside?"] * 5, {"country": str})
reword_request = llm_handler.request(["Reword this sentence: I like to eat apples."] * 2, {"reworded": str}, enforce_unique=True)
logLine("Requests set up.")

logLine("Processing requests...")
res = llm_handler.process()
if not res:
    logLine("Processing failed.")
else:
    logLine("Requests processed.")

logLine(capital_request.responses)
logLine(country_request.responses)
logLine(reword_request.responses)