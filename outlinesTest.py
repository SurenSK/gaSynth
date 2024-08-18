import outlines
import json
import time
model = outlines.models.transformers("gpt2", model_kwargs={"cache_dir": "."})
fruit_regex = r'{"name":"[A-Z][a-z]{1,9}", "taste":"(sweet|sour|bitter|salty|umami)", "price":"\$[0-9]{1,3}\.[0-9]{2}"}'
generator = outlines.generate.regex(model, fruit_regex)

# line1 = '{"name":"Fruit", "taste":"bitter", "price":"$0.09"}'
# line2 = '{"name":"Transition", "taste":"bitter", "price":"$17.99"}'
# line3 = '{"name":"Bard", "taste":"sweet", "price":"$0.00"}'
# line4 = '{"name":"Matilda", "taste":"sour", "price":"$6.99"}'
# line5 = '{"name":"Troy", "taste":"sour", "price":"$2.20"}'
# line6 = '{"name":"Leucnus", "taste":"s elevation'
# line7 = '{"name":"Robot", "taste":"bitter", "price":"$20.95"oub'
# line8 = '{"name":"Genie", "taste":"sour", "price":"$400.00"} 1934'
# line9 = '{"name":"Cinnabar", "taste choir'
# line10 = '{"name":"Wild", "tasteARD"'
# lines = [line1, line2, line3, line4, line5, line6, line7, line8, line9, line10]

tGen = time.time()
fruit_jsons = generator(["Generate a fruit JSON"] * 1000, max_tokens=1024)
tGen = time.time() - tGen

total_toks = sum([len(model.tokenizer.encode(f)) for f in fruit_jsons])

validGens = []
invalidGens = []
for i,f in enumerate(fruit_jsons):
    print(f"{i+1}/{len(fruit_jsons)} : {f}")
    try:
        f_ = json.loads(f)
        validGens.append(f)
    except:
        invalidGens.append(f)

print(f"t+{tGen:.0f}s Valid Generations: {len(validGens)}/{len(fruit_jsons)} {total_toks/tGen:.0f}toks")
print(f"Sample Valid Generations:")
for i in range(min(5, len(validGens))):
    print(f"{i+1}/{min(5, len(validGens))} {validGens[i]}")
print(f"Sample Invalid Generations:")
for i in range(min(5, len(invalidGens))):
    print(f"{i+1}/{min(5, len(invalidGens))} {invalidGens[i]}")