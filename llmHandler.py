import os
import time
import json
import torch
from dotenv import load_dotenv
from typing import Any, Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datetime import datetime

load_dotenv()

def logLine(l, verbose=True):
    with open("handler.txt", "a") as log_file:
        log_file.write(str(l) + "\n")

current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%Y-%B-%d %I:%M%p")
logLine("*"*120)
logLine(f"Handler started.\t{formatted_datetime}")
logLine("*"*120)

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
            raise response
        if self.enforce_unique:
            i = len(self.unique_responses)
            respStr = list(response.values()) if isinstance(response, dict) else response
            respStr = " ".join(map(str, respStr))
            respStr = "".join([c for c in respStr if c.isalnum()])
            self.unique_responses.add(respStr)
            if len(self.unique_responses) > i:
                self.responses[prompt_idx] = response
                self.outstanding[prompt_idx] = False
            else:
                raise ValueError("Duplicate response detected")
        else:
            self.responses[prompt_idx] = response
            self.outstanding[prompt_idx] = False

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
                       do_sample=True, temperature=1, batch_size=self.batch_size, max_new_tokens=512)
        llm.tokenizer.pad_token_id = model.config.eos_token_id
        return llm

    def request(self, prompts: List[str], expectation: Dict[str, Any], enforce_unique: bool = False) -> Request:
        req = Request(prompts, expectation, enforce_unique)
        self.queue.append(req)
        logLine(f"Request added to queue. {len(prompts)} prompts - {len(self.queue)} requests in queue.")
        return req
    
    def count_object_types(self, obj_list):
        type_counts = {}
        for obj in obj_list:
            obj_type = type(obj).__name__
            if obj_type in type_counts:
                type_counts[obj_type] += 1
            else:
                type_counts[obj_type] = 1
        return type_counts
    
    def _extract_json(self, text: str, expectation: Dict[str, Any]) -> Dict[str, Any]:
        token_count = len(self.llm.tokenizer.encode(text))

        if '<result>' not in text:
            raise ValueError("JSON output does not start correctly")

        if '</result>' not in text:
            if token_count > 500:
                raise ValueError("JSON output is oversized and improperly terminated")
            else:
                raise ValueError("JSON output does not end correctly")
                
        text = text.split('<result>')[1].split('</result>')[0]

        try:
            text = text.replace("\'", "").replace("\n","").strip()
            parsed_json = json.loads(text, strict=False)

            receivedKeys = set(parsed_json.keys())
            expectedKeys = set(expectation.keys())
            if not receivedKeys.issubset(expectedKeys):
                raise KeyError(f"Unexpected field(s) in response: {receivedKeys - expectedKeys}", 'too_many_keys')
            if not expectedKeys.issubset(receivedKeys):
                raise KeyError(f"Missing expected field(s): {expectedKeys - receivedKeys}", 'too_few_keys')
            
            for key, expected_type in expectation.items():
                if key not in parsed_json:
                    raise KeyError(f"Missing expected field: {key}")
                
                value = parsed_json[key]
                
                if isinstance(value, list):
                    # If the value is already a list, join it into a string
                    parsed_json[key] = " ".join(str(item) for item in value)
                elif isinstance(value, str):
                    if any(c in value for c in [":", ";", "{", "}", "(", ")", "[", "]"]):
                        try:
                            parsed_list = json.loads(value)
                            if isinstance(parsed_list, list):
                                parsed_json[key] = " ".join(str(item) for item in parsed_list)
                            else:
                                logLine(f"Expected a list for field '{key}' but got {type(parsed_list).__name__}")
                                raise ValueError(f"Expected a list for field '{key}' but got {type(parsed_list).__name__}")
                        except json.JSONDecodeError:
                            logLine(f"Value for field '{key}' contains invalid characters: {value}")
                            raise ValueError(f"Value for field '{key}' contains invalid characters: {value}")
                else:
                    # If it's neither a list nor a string, convert it to a string
                    parsed_json[key] = str(value)

            return parsed_json

        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format {text}")

    
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
        # logLine(f"Generating {len(prompts)} responses.")
        tGen = time.time()
        responses = self.llm(prompts)
        tGen = time.time() - tGen
        responses = [r[0]['generated_text'].replace(prompt, "") for r, prompt in zip(responses, prompts)]
        totalToks = sum(len(self.llm.tokenizer.encode(r)) for r in responses)
        # logLine(f"\tt+{tGen:3.0f}s - Generated {len(prompts)} responses - {totalToks:5d} tokens - {totalToks/tGen:4.0f}toks")
        return totalToks, tGen, responses

    def process(self):
        processCounters = {
            "total": 0,
            "valid": 0,
            "bad_start": 0,
            "bad_end": 0,
            "bad_end_oversize": 0,
            "too_few_keys": 0,
            "too_many_keys": 0,
            "bad_values": 0,
            "repeat": 0,
            "unknown_val": 0,
            "unknown_type": 0,
            "unknown": 0
        }
        cIter = 0
        totalTokens = 0
        tProcess = time.time()
        
        while any(any(req.outstanding) for req in self.queue):
            if cIter == self.maxIters:
                logLine("\t###Reached maximum iterations. Stopping process.")
                break
            cIter += 1

            master_list = []
            for req_idx, req in enumerate(self.queue):
                for prompt_idx, (prompt, outstanding) in enumerate(zip(req.prompts, req.outstanding)):
                    if outstanding:
                        master_list.append((req_idx, prompt_idx, prompt+self._generate_json_prompt(req.expectation)))
            
            if not master_list or len(master_list) == 0:
                logLine("Error: master_list is empty. This shouldn't happen. Stopping process.")
                break
            # logLine(f"Processing {len(master_list)} prompts.")
            
            minBatchSize = self.batch_size//4
            multiplication_factor = min(minBatchSize, (self.batch_size // len(master_list)))
            multiplication_factor = max(1, multiplication_factor)
            # logLine(f"Expanding master list from {len(master_list)} to {len(master_list) * multiplication_factor}.")
            master_list = master_list * multiplication_factor
            
            prompts = [item[2] for item in master_list]
            processCounters["total"] += len(prompts)
            rawTokens, tGen, responses = self._generate_batch(prompts)
            totalTokens += rawTokens
            
            totalOutstandingPre = sum(sum(req.outstanding) for req in self.queue)
            for (req_idx, prompt_idx, _), response in zip(master_list, responses):
                req = self.queue[req_idx]
                try:
                    extracted = self._extract_json(response, req.expectation)
                    req.add_response(prompt_idx, extracted)
                    processCounters["valid"] += 1
                except ValueError as e:
                    if "does not start correctly" in str(e):
                        processCounters["bad_start"] += 1
                    elif "does not end correctly" in str(e):
                        processCounters["bad_end"] += 1
                    elif "oversized and improperly terminated" in str(e):
                        processCounters["bad_end_oversize"] += 1
                    elif "Duplicate response detected" in str(e):
                        processCounters["repeat"] += 1
                    else:
                        processCounters["unknown_val"] += 1
                except KeyError as e:
                    error_message, error_type = e.args
                    if error_type == 'too_few_keys':
                        processCounters["too_few_keys"] += 1
                    elif error_type == 'too_many_keys':
                        processCounters["too_many_keys"] += 1
                except TypeError:
                    processCounters["unknown_type"] += 1
                except Exception as e:
                    processCounters["unknown"] += 1
            totalOutstandingPost = sum(sum(req.outstanding) for req in self.queue)
            logLine(f"\t{totalOutstandingPre:3d}>{totalOutstandingPost:3d} | t+{tGen:3.0f}s - Generated {len(prompts):3d} responses - {rawTokens:6d} tokens - {rawTokens/tGen:4.0f}toks")

        totalValidTokens = 0
        totalRequests = 0
        totalOutstanding = 0
        for req in self.queue:
            totalValidTokens += sum(len(self.llm.tokenizer.encode(str(r))) if r and not o else 0 for r, o in zip(req.responses, req.outstanding))
            totalRequests += len(req.prompts)
            totalOutstanding += sum(req.outstanding)
            # logLine(f"Prompt: {req.prompts[0]}")
            # logLine(f"Response: {req.responses[0]}")
        
        logLine(f"Processed {processCounters['total']} requests in {cIter} iterations.")
        self.queue = []
        # print the counters
        for error_type, count in processCounters.items():
            logLine(f"\t   {error_type}: {count}")
        return (processCounters, time.time() - tProcess, totalValidTokens, totalTokens, totalRequests, totalOutstanding)

if __name__ == "__main__":
    tMain = time.time()
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
    for i,template in enumerate(templates.keys()):
        llm_handler.jsonTemplate = template
        logLine(f"Processing template {i}/{len(templates.keys())}: {template}")
        capital_request = llm_handler.request([f"What is the capital of {c}?" for c in ["France", "Germany", "Italy", "Spain", "United Kingdom"]], {"city": str})
        country_request = llm_handler.request([f"Which country is {c} inside?" for c in ["Paris", "Berlin", "Rome", "Madrid", "London"]], {"country": str})
        reword_request = llm_handler.request(["Reword this sentence: I like to eat apples."] * 2, {"reworded": str}, enforce_unique=True)
        story_request = llm_handler.request(
            [f"Tell me a story about {topic}." for topic in ["courage", "betrayal", "adventure", "friendship", "discovery"]]*100,
            {"story": str}
        )

        res = llm_handler.process()
        if res:
            counters, tProcess, totalValidTokens, totalTokens, totalReq, totalOut = res
            vToksRate = totalValidTokens / tProcess
            vToksRatio = totalValidTokens / totalTokens
            vSampleRate = counters["valid"] / tProcess

            templates[template] = vToksRate
            logLine(f"###Finished Processing Template")
            # logLine(f"Template: {template}")
            logLine("Counters:")
            for error_type, count in counters.items():
                logLine(f"   {error_type}: {count}")
            logLine(f"+{tProcess:.2f}s Processed: {totalReq-totalOut}/{totalReq} - Ratio: {(counters['valid']/counters['total'])*100:2.0f}%S {vToksRatio*100:2.0f}%T - {vSampleRate:3.1f}sams {vToksRate:.0f}toks")
            logLine("")  # Empty line for readability between templates
        # clear llm_handler.queue
        llm_handler.queue = []

    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%B-%d %I:%M%p")
    logLine("*"*120)
    logLine("All templates processed.")
    with open("jsonTemplates.json", "w") as f:
        json.dump(templates, f)
    logLine("Saved to jsonTemplates.json.")
    logLine(f"Handler finished.\t{formatted_datetime}\tTotal time: {time.time() - tMain:.2f}s")
    logLine("*"*120)