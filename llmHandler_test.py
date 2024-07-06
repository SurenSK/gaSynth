# run_llm_handler.py

from llmHandler import LLMHandler

def logLine(l):
    with open("handler.txt", "a") as log_file:
        log_file.write(str(l) + "\n")

# Instantiate the handler
llm_handler = LLMHandler("mistralai/Mistral-7B-Instruct-v0.2", batch_size=64)

# Create and process requests
reword_request = llm_handler.request(
    ["Creatively reword this sentence: I like to eat apples."] * 16,
    {"reworded": str},
    enforce_unique=True
)

# Process the queue
res = llm_handler.process()

# Handle the results
if res:
    counters, tProcess, totalValidTokens, totalTokens, totalReq, totalOut = res
    vToksRate = totalValidTokens / tProcess
    vToksRatio = totalValidTokens / totalTokens
    vSampleRate = counters["valid"] / tProcess

    logLine(f"Processed: {totalReq-totalOut}/{totalReq} - Ratio: {(counters['valid']/counters['total'])*100:2.0f}%S {vToksRatio*100:2.0f}%T - {vSampleRate:3.1f}sams {vToksRate:.0f}toks")

for request in [reword_request]:
    for cOutstanding, cPrompt, cResponse in zip(request.outstanding, request.prompts, request.responses):
        logLine(f"{not(cOutstanding)} {cPrompt} -> {cResponse}")