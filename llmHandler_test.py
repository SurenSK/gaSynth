# run_llm_handler.py

from llmHandler import LLMHandler

def logLine(l):
    with open("handler.txt", "a") as log_file:
        log_file.write(str(l) + "\n")

# Instantiate the handler
llm_handler = LLMHandler("mistralai/Mistral-7B-Instruct-v0.2", batch_size=64)

# Create and process requests
capital_request = llm_handler.request(
    [f"What is the capital of {c}?" for c in ["France", "Germany", "Italy", "Spain", "United Kingdom"]],
    {"city": str}
)
country_request = llm_handler.request(
    [f"Which country is {c} inside?" for c in ["Paris", "Berlin", "Rome", "Madrid", "London"]],
    {"country": str}
)
reword_request = llm_handler.request(
    ["Reword this sentence: I like to eat apples."] * 2,
    {"reworded": str},
    enforce_unique=True
)
story_request = llm_handler.request(
    [f"Tell me a story about {topic}." for topic in ["courage", "betrayal", "adventure", "friendship", "discovery"]] * 2,
    {"story": str}
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
for request in [capital_request, country_request, reword_request, story_request]:
    for cOutstanding, cPrompt, cResponse in zip(request.outstanding, request.prompts, request.responses):
        logLine(f"{not(cOutstanding)} {cPrompt} -> {cResponse}")