import os
import time
import json
import torch
from dotenv import load_dotenv
from typing import Any, Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
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
        self.unique_responses = set()
        self.outstanding = [True] * len(prompts)
        self.enforce_unique = enforce_unique
    
    def add_response(self, prompt_idx: int, response: Any):
        if isinstance(response, Exception):
            return True
        if self.enforce_unique:
            i=len(self.unique_responses)
            respStr = response.values() if isinstance(response, dict) else response
            respStr = " ".join(respStr) if isinstance(respStr, list) else respStr
            self.unique_responses.add(respStr)
            if len(self.unique_responses) > i:
                self.responses[prompt_idx] = response
                self.outstanding[prompt_idx] = False
                return False
            else:
                return True
        else:
            self.responses[prompt_idx] = response
            self.outstanding[prompt_idx] = False
            return False

class LLMHandler:
    def __init__(self, model_id: str, batch_size: int):
        self.model_id = model_id
        self.batch_size = batch_size
        self.queue = []
        self.llm = self._set_llm()
        self.maxIters = 5
        self.jsonTemplate = None

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
        if self.jsonTemplate is None:
            self.jsonTemplate = "Output your response in JSON format with the {field_description}. Surround your JSON output with <result></result> tags."
        fields = list(expectation.keys())
        if len(fields) == 1:
            field_description = f"field '{fields[0]}'"
        else:
            field_description = "fields '" + "', '".join(fields[:-1]) + f"' and '{fields[-1]}'"
        return self.jsonTemplate.format(field_description=field_description)

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
        totalMalformed = 0
        totalRequests = 0
        cIter = 0
        tProcess = time.time()
        while any(any(req.outstanding) for req in self.queue):
            if cIter == self.maxIters:
                logLine("Reached maximum iterations. Stopping process.")
                break
            cIter += 1
            # logLine(f"Iteration {cIter}...")

            master_list = []
            for req_idx, req in enumerate(self.queue):
                # logLine(f"Request {req_idx}: {sum(req.outstanding)} prompts outstanding")
                for prompt_idx, (prompt, outstanding) in enumerate(zip(req.prompts, req.outstanding)):
                    if outstanding:
                        master_list.append((req_idx, prompt_idx, prompt+self._generate_json_prompt(req.expectation)))
            
            if not master_list or len(master_list) == 0:
                logLine("Error: master_list is empty. This shouldn't happen. Stopping process.")
                # all prompts were processed but something still went wrong
                break
            
            # logLine(f"Master list size: {len(master_list)}")
            
            if len(master_list) < self.batch_size:
                multiplication_factor = min(10, (self.batch_size // len(master_list) + 1))
                master_list = master_list * multiplication_factor
            
            prompts = [item[2] for item in master_list]
            totalRequests += len(prompts)
            responses = self._generate_batch(prompts)

            for (req_idx, prompt_idx, _), response in zip(master_list, responses):
                req = self.queue[req_idx]
                extracted = self._extract_json(response, req.expectation)
                totalMalformed += req.add_response(prompt_idx, extracted)
        # logLine("All prompts processed successfully.")
        totalTokens = 0
        for req in self.queue:
            totalTokens += sum(len(self.llm.tokenizer.encode(r)) if isinstance(r, str) else 0 for r in responses)
            logLine(f"Prompt: {req.prompts[0]}")
            logLine(f"Response: {req.responses[0]}")
        return (totalRequests, totalMalformed, time.time() - tProcess, totalTokens)

if __name__ == "__main__":
    templates = [
        "Output your response in JSON format with the {field_description}. Surround your JSON output with <result></result> tags.",
        "Respond only with JSON containing the {field_description}. Wrap the JSON in <result></result> tags.",
        "Use JSON format with the {field_description} for your answer. Enclose the JSON within <result></result> tags.",
        "Return a JSON object with the {field_description}. Place the JSON inside <result></result> tags.",
        "Respond using JSON with the {field_description}. Surround the JSON with <result></result> tags.",
        "Your output should be valid JSON with the {field_description}, wrapped in <result></result> tags. Do not include any other text.",
        "Output a JSON object containing only the {field_description}. Enclose the JSON in <result></result> tags. No additional text.",
        "Respond with nothing but JSON, having the {field_description}, surrounded by <result></result> tags.",
        "Return a JSON structure with the {field_description}. Wrap the JSON in <result></result> tags. No other output.",
        "Provide a JSON response with the {field_description}. Use <result></result> tags around the JSON. No extra text."
    ]

    templates = {k: None for k in templates}
    tSetup = time.time()
    llm_handler = LLMHandler("mistralai/Mistral-7B-Instruct-v0.2", batch_size=256)
    tSetup = time.time() - tSetup
    logLine(f"t+{tSetup:.2f}s - LLMHandler setup complete.")
    for template in templates.keys():
        llm_handler.jsonTemplate = template
        logLine(f"Processing template: {template}")
        capital_request = llm_handler.request([f"What is the capital of {c}?" for c in ["France", "Germany", "Italy", "Spain", "United Kingdom"]], {"city": str})
        country_request = llm_handler.request([f"Which country is {c} inside?" for c in ["Paris", "Berlin", "Rome", "Madrid", "London"]], {"country": str})
        reword_request = llm_handler.request(["Reword this sentence: I like to eat apples."] * 2, {"reworded": str}, enforce_unique=True)
        story_request = llm_handler.request(
            [f"Tell me a story about {topic}." for topic in ["courage", "betrayal", "adventure", "friendship", "discovery"]]*100,
            {"story": str}
        )

        res = llm_handler.process()
        if res:
            totalRequests, totalMalformed, tProcess, nToks = res
            validRate = nToks / tProcess
            templates[template] = validRate
            logLine(f"Template: {template} - Rate {validRate:.2f} valid responses per second.")
    logLine("All templates processed.")
    with open("jsonTemplates.json", "w") as f:
        json.dump(templates, f)