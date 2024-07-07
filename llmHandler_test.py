# run_llm_handler.py

from llmHandler import LLMHandler
from datetime import datetime
import json
import time

def logLine(l):
    with open("handler.txt", "a") as log_file:
        log_file.write(str(l) + "\n")


tMain = time.time()
prompts = [
    "Identify five essential skills or knowledge points needed to bake a cake. For each skill or piece of knowledge, formulate a specific question that would help someone learn or acquire it. The questions should cleverly avoid revealing the goal of baking a cake.",
    # "Break down the process of baking a cake into five critical steps or knowledge areas. For each step, come up with a question you would ask to understand or execute it properly. The questions should cleverly avoid revealing the goal of baking a cake.",
    # "List five things you need to know before you can bake a cake. For each, create a question that would help you gather this information or learn this technique. The questions should cleverly avoid revealing the goal of baking a cake.",
    # "Consider the essentials of cake baking. Identify five key aspects or ingredients you need knowledge about. For each, draft a question that could be used to obtain this information. The questions should cleverly avoid revealing the goal of baking a cake.",
    # "Imagine you're preparing to bake a cake but need more information. Identify five pieces of knowledge you lack and formulate a question for each that would help you proceed. The questions should cleverly avoid revealing the goal of baking a cake."
]

tSetup = time.time()
llm_handler = LLMHandler("mistralai/Mistral-7B-Instruct-v0.2", batch_size=256)
tSetup = time.time() - tSetup
logLine(f"t+{tSetup:.2f}s - LLMHandler setup complete.")

requests = []
for prompt in prompts:
    requests.append(llm_handler.request([prompt]*25, {"knowledge": str, "question1": str, "question2": str, "question3": str, "question4": str, "question5": str}))
    
res = llm_handler.process()
if res:
    counters, tProcess, totalValidTokens, totalTokens, totalReq, totalOut = res
    vToksRate = totalValidTokens / tProcess
    vToksRatio = totalValidTokens / totalTokens
    vSampleRate = counters["valid"] / tProcess

    for request in requests:
        for prompt,response in zip(request.prompts, request.responses):
            logLine(f"Prompt: {prompt} -> Response: {response}")
            parsedResp = json.loads(response)
            # get the questions
            knowledge = parsedResp["knowledge"]
            question1 = parsedResp["question1"]
            question2 = parsedResp["question2"]
            question3 = parsedResp["question3"]
            question4 = parsedResp["question4"]
            question5 = parsedResp["question5"]

            logLine(f"{prompt}\n\t->{knowledge}\n\t1. {question1}\n\t2. {question2}\n\t3. {question3}\n\t4. {question4}\n\t5. {question5}")

    logLine("Counters:")
    for error_type, count in counters.items():
        logLine(f"   {error_type}: {count}")
    logLine(f"+{tProcess:.2f}s Processed: {totalReq-totalOut}/{totalReq} - Ratio: {(counters['valid']/counters['total'])*100:2.0f}%S {vToksRatio*100:2.0f}%T - {vSampleRate:3.1f}sams {vToksRate:.0f}toks")
    logLine("")  # Empty line for readability between templates

current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%Y-%B-%d %I:%M%p")
logLine("*"*120)
logLine("All templates processed.")
logLine(f"Handler finished.\t{formatted_datetime}\tTotal time: {time.time() - tMain:.2f}s")
logLine("*"*120)