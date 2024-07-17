import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationMixin, TextGenerationPipeline
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
token = os.getenv('HUGGINGFACE_TOKEN')
if token is None:
    raise ValueError("HUGGINGFACE_TOKEN is not set in the environment.")

# Logging function
def logLine(l):
    with open("mix.txt", "a") as log_file:
        log_file.write(str(l) + "\n")

# Custom mixin with a reimplementation of beam search
class CustomGenerationMixin(GenerationMixin):
    def beam_search(self, input_ids, beam_scorer, logits_processor=None, stopping_criteria=None, pad_token_id=None, eos_token_id=None, output_scores=False, return_dict_in_generate=False, **model_kwargs):
        logLine("Test - Running custom beam search")
        return super().beam_search(input_ids, beam_scorer, logits_processor, stopping_criteria, pad_token_id, eos_token_id, output_scores, return_dict_in_generate, **model_kwargs)

# Extending the AutoModelForCausalLM with custom generation mixin
class CustomModel(CustomGenerationMixin, AutoModelForCausalLM):
    pass

# Initialize model and tokenizer with specific configurations
model_id = "mistralai/Mistral-7B-Instruct-v0.2"
logLine("Test - Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=token)
logLine("Test - Loaded tokenizer")
logLine("Test - Loading model")
model = CustomModel.from_pretrained(model_id, use_auth_token=token, cache_dir=".", torch_dtype=torch.bfloat16, device_map="auto")
logLine("Test - Loaded model")

# Optionally compile the model with torch.compile for optimized execution
logLine("Test - Compiling model")
model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
logLine("Test - Compiled model")

# Setup the text generation pipeline with the custom model
logLine("Test - Creating pipeline")
pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)  # Adjust device as per your configuration
logLine("Test - Created pipeline")

# Generate text using your custom decoding strategy
logLine("Test - Generating text")
texts = pipeline("Hey whats up", num_return_sequences=3, num_beams=5, do_sample=True)
logLine("Test - Generated text")
logLine(texts)
