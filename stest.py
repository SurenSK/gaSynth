import os
import json
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

def logLine(l, verbose=True):
    with open("log.txt", "a") as log_file:
        log_file.write(str(l) + "\n")

class LLMSetup:
    @classmethod
    def set_llm(cls, model_id):
        token = os.getenv('HUGGINGFACE_TOKEN')
        if token is None:
            raise ValueError("HUGGINGFACE_TOKEN is not set in the environment.")
        
        cls.tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        model = AutoModelForCausalLM.from_pretrained(model_id, token=token)
        
        model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
        logLine("Model loaded.")
        
        cls.llm = pipeline("text-generation", model=model, tokenizer=cls.tokenizer, temperature=2.5, batch_size=8, max_new_tokens=200)
        cls.llm.tokenizer.pad_token_id = model.config.eos_token_id
        logLine("Pipeline created.")
        
        return cls.llm

def extract_json(text):
    try:
        start = text.index('<result>') + 8
        end = text.index('</result>')
        json_str = text[start:end].strip()
        return json.loads(json_str)
    except (ValueError, json.JSONDecodeError):
        return None

def test_prompts_batch(llm, prompts, sentences, num_repeats=10):
    all_prompts = []
    for _ in range(num_repeats):
        for prompt in prompts:
            for sentence in sentences:
                all_prompts.append(f"{prompt}\n\nSentence to reword: {sentence}")
    
    batched_responses = []
    for i in tqdm(range(0, len(all_prompts), llm.batch_size), desc="Processing batches"):
        batch = all_prompts[i:i+llm.batch_size]
        responses = llm(batch, max_length=300, num_return_sequences=1)
        batched_responses.extend([resp[0]['generated_text'] for resp in responses])
    
    return batched_responses

def main():
    llm = LLMSetup.set_llm("mistralai/Mistral-7B-Instruct-v0.2")
    
    test_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "A journey of a thousand miles begins with a single step.",
        "To be or not to be, that is the question.",
        "All that glitters is not gold.",
        "The early bird catches the worm."
    ]
    
    prompts = [
        "Reword the given sentence. Output your response in JSON format with a 'reworded' field. Surround your JSON output with <result></result> tags.",
        "Your task is to rephrase the provided sentence. Respond only with JSON containing a 'reworded' key. Wrap the JSON in <result></result> tags.",
        "Rewrite the following sentence in your own words. Use JSON format with a 'reworded' field for your answer. Enclose the JSON within <result></result> tags.",
        "Provide an alternative wording for the given sentence. Return a JSON object with a 'reworded' property. Place the JSON inside <result></result> tags.",
        "Transform the provided sentence into a new one with the same meaning. Respond using JSON with a 'reworded' key. Surround the JSON with <result></result> tags.",
        "Rephrase the sentence below. Your output should be valid JSON with a 'reworded' field, wrapped in <result></result> tags. Do not include any other text.",
        "Rewrite the given sentence. Output a JSON object containing only a 'reworded' field. Enclose the JSON in <result></result> tags. No additional text.",
        "Your job is to reword the provided sentence. Respond with nothing but JSON, having a 'reworded' key, surrounded by <result></result> tags.",
        "Rephrase the following sentence. Return a JSON structure with a single 'reworded' field. Wrap the JSON in <result></result> tags. No other output.",
        "Rewrite the sentence in different words. Provide a JSON response with a 'reworded' property. Use <result></result> tags around the JSON. No extra text."
    ]

    num_repeats = 10
    results = {i: {'success': 0, 'total': 0, 'examples': []} for i in range(1, len(prompts) + 1)}

    batched_responses = test_prompts_batch(llm, prompts, test_sentences, num_repeats)

    for i, response in enumerate(batched_responses):
        prompt_index = (i // (len(test_sentences) * num_repeats)) + 1
        results[prompt_index]['total'] += 1
        result = extract_json(response)
        if result and 'reworded' in result:
            results[prompt_index]['success'] += 1
            if len(results[prompt_index]['examples']) < 3:
                results[prompt_index]['examples'].append(result['reworded'])

    logLine("\nResults:")
    for i, data in results.items():
        success_rate = (data['success'] / data['total']) * 100 if data['total'] > 0 else 0
        logLine(f"Prompt {i}: Success rate {success_rate:.2f}% ({data['success']}/{data['total']})")
        logLine("Examples:")
        for example in data['examples']:
            logLine(f"  - {example}")
        logLine()

    best_prompt = max(results, key=lambda x: results[x]['success'])
    logLine(f"\nBest prompt: {best_prompt} (highest success rate)")
    logLine(f"Success rate: {(results[best_prompt]['success'] / results[best_prompt]['total']) * 100:.2f}%")

if __name__ == "__main__":
    main()