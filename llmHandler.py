import os
import time
import json
import torch
from dotenv import load_dotenv
from typing import Any, Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
load_dotenv()

def logLine(l, verbose=True):
    with open("handler.txt", "a") as log_file:
        log_file.write(str(l) + "\n")
logLine("Handler started.")

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
        self.maxIters = 5

    def _set_llm(self):
        token = os.getenv('HUGGINGFACE_TOKEN')
        if token is None:
            raise ValueError("HUGGINGFACE_TOKEN is not set in the environment.")
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=token)
        model = AutoModelForCausalLM.from_pretrained(self.model_id, token=token, cache_dir=".",
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
            return e

    def _generate_json_prompt(self, expectation: Dict[str, Any]) -> str:
        fields = list(expectation.keys())
        if len(fields) == 1:
            field_str = f"field '{fields[0]}'"
        else:
            field_str = "fields '" + "', '".join(fields[:-1]) + f"' and '{fields[-1]}'"
        return f" Output your response in JSON format with the {field_str}. Surround your JSON output with <result></result> tags."
    
    def _generate_batch(self, prompts: List[str]) -> List[str]:
        # for i,p in enumerate(prompts):
        #     logLine(f"Prompt {i}: {p}")
        tGen = time.time()
        responses = self.llm(prompts)
        tGen = time.time() - tGen
        responses = [r[0]['generated_text'].replace(prompt, "") for r, prompt in zip(responses, prompts)]
        totalToks = sum(len(self.llm.tokenizer.encode(r)) for r in responses)
        logLine(f"t+{tGen:.2f}s - Generated {len(prompts)} responses - {totalToks/tGen:.0f}toks")
        return responses

    def process(self):
        cIter = 0
        while any(any(req.outstanding) for req in self.queue):
            if cIter == self.maxIters:
                logLine("Reached maximum iterations. Stopping process.")
                return False
            cIter += 1
            logLine(f"Iteration {cIter}...")

            master_list = []
            for req_idx, req in enumerate(self.queue):
                logLine(f"Request {req_idx}: {sum(req.outstanding)} prompts outstanding")
                for prompt_idx, (prompt, outstanding) in enumerate(zip(req.prompts, req.outstanding)):
                    if outstanding:
                        master_list.append((req_idx, prompt_idx, prompt+self._generate_json_prompt(req.expectation)))
            
            if not master_list or len(master_list) == 0:
                logLine("Error: master_list is empty. This shouldn't happen. Stopping process.")
                # all prompts were processed but something still went wrong
                return False
            
            logLine(f"Master list size: {len(master_list)}")
            
            if len(master_list) < self.batch_size:
                multiplication_factor = max(10, (self.batch_size // len(master_list) + 1))
                master_list = master_list * multiplication_factor
                logLine(f"Extended master list to size: {len(master_list)}")
            
            prompts = [item[2] for item in master_list]
            responses = self._generate_batch(prompts)

            for (req_idx, prompt_idx, _), response in zip(master_list, responses):
                req = self.queue[req_idx]
                extracted = self._extract_json(response, req.expectation)
                
                if not isinstance(extracted, Exception) and (not req.enforce_unique or extracted not in req.responses):
                    req.responses[prompt_idx] = list(extracted.values()) if isinstance(extracted, dict) else extracted
                    req.outstanding[prompt_idx] = False
            
            logLine(f"After processing: {sum(sum(req.outstanding) for req in self.queue)} prompts still outstanding")
        
        logLine("All prompts processed successfully.")
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