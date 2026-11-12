from functools import lru_cache
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "distilgpt2"

@lru_cache(maxsize=1)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model

def generate_story(prompt: str, max_new_tokens: int = 100,
                   temperature: float = 0.8, top_p: float = 0.9) -> str:
    if not prompt or not prompt.strip():
        raise ValueError("Prompt is required.")
    if not 1 <= int(max_new_tokens) <= 200:
        raise ValueError("max_new_tokens must be between 1 and 200.")
    if not 0.1 <= float(temperature) <= 2.0:
        raise ValueError("temperature must be between 0.1 and 2.0.")
    tokenizer, model = load_model()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=900)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.15,
        )
    full_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # Return only the continuation when the decoded output begins with the prompt.
    return full_text[len(prompt):].strip() if full_text.startswith(prompt) else full_text
