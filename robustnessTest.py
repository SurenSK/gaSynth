import time
import json
from datetime import datetime
from llmHandler import LLMHandler

def logLine(l):
    with open("ga.txt", "a") as log_file:
        log_file.write(str(l) + "\n")

tMain = time.time()

tSetup = time.time()
llm_handler = LLMHandler("mistralai/Mistral-7B-Instruct-v0.2", batch_size=256)
tSetup = time.time() - tSetup
logLine(f"t+{tSetup:.2f}s - LLMHandler setup complete.")

seed_templates = [
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


# TODO Robustness metric as resilience to rewording, capitalization, and other minor changes

templates = []
templateRequests = []
for seed in seed_templates:
    templates.append({"gen": 0, "template": seed})
    templateRequests.append(llm_handler.request([f"Reword this sentence: {seed}"]*10, {"reworded": str}, enforce_unique=True))
res = llm_handler.process()
for req in templateRequests:
    responses = req.responses
    for response in responses:
        logLine(f"Response: {response}")
        proposed_template = response.get("reworded", "")
        if "{field_description}" in proposed_template and "<result></result>" in proposed_template:
            templates.append({"gen": 1, "template": proposed_template})

logLine(f"Generated {len(templates)-10} new templates from seed templates.")

# use the reword request to generate 10 templates per seed template, and then from those 100, generate 10 more templates per template. This will give us 10 gen0 templates, 100 gen1 templates, and 1000 gen2 templates.
# get the new templates out into a list, keeping track of generation#, discarding ones that don't have the strings '{field_description}' and '<result></result>' inside them
# then, for each of the new valid templates, generate the 512 test requests (5 capital, 5 country, 2 reword, 500 story) and process them
# verify that we get the correct set of fields back for each of the 512 test requests
# verify that the first set of requests are correct cities / countries by correctAns in the appropriate field
# verify the stories are at least 100 characters long each
# assign accuracy % to each template based on how many of the 512 test requests were correct
# save the results to a jsonl in the format: {accuracy, generation, template, numCorrectCapitals, numCorrectCountries, numCorrectRewords, numCorrectStories, numCorrectTotal}

for i,t in enumerate(templates):
    template = t["template"]
    gen = t["gen"]
    llm_handler.jsonTemplate = template
    logLine(f"Processing template {i}/{len(templates)} Gen{gen}: {template}")
    capital_request = llm_handler.request([f"What is the capital of {c}?" for c in ["France", "Germany", "Italy", "Spain", "United Kingdom"]], {"city": str})
    country_request = llm_handler.request([f"Which country is {c} inside?" for c in ["Paris", "Berlin", "Rome", "Madrid", "London"]], {"country": str})
    reword_request = llm_handler.request(["Reword this sentence: I like to eat apples."] * 6, {"reworded": str}, enforce_unique=True)
    story_request = llm_handler.request(
        [f"Tell me a story about {topic}." for topic in ["courage", "betrayal", "adventure", "friendship", "discovery"]]*48,
        {"story": str}
    )

    res = llm_handler.process()
    if res:
        respCapitals = [r.get("city", "") for r in capital_request.responses]
        respCountries = [r.get("country", "") for r in country_request.responses]
        respRewords = [r.get("reworded", "") for r in reword_request.responses]
        respStories = [r.get("story", "") for r in story_request.responses]
        # TODO verify that the responses are correct

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
        logLine("")

current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%Y-%B-%d %I:%M%p")
logLine("*"*120)
logLine("All templates processed.")
with open("robustness.json", "w") as f:
    json.dump(templates, f)
logLine("Saved to jsonTemplates.json.")
logLine(f"Handler finished.\t{formatted_datetime}\tTotal time: {time.time() - tMain:.2f}s")
logLine("*"*120)