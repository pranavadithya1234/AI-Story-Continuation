"""
TorchServe custom handler for DistilGPT2.
Input JSON: {"prompt": "...", "max_new_tokens": 100, "temperature": 0.8, "top_p": 0.9}
"""
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from ts.torch_handler.base_handler import BaseHandler

class StoryHandler(BaseHandler):
    def initialize(self, context):
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.initialized = True

    def preprocess(self, data):
        item = data[0]
        body = item.get("body") or item.get("data") or item
        if isinstance(body, bytes):
            body = body.decode("utf-8")
        if isinstance(body, str):
            body = json.loads(body)
        prompt = str(body.get("prompt", "")).strip()
        if not prompt:
            raise ValueError("prompt is required")
        if len(prompt) > 4000:
            raise ValueError("prompt must be at most 4000 characters")
        max_new_tokens = int(body.get("max_new_tokens", 100))
        temperature = float(body.get("temperature", 0.8))
        top_p = float(body.get("top_p", 0.9))
        if not 1 <= max_new_tokens <= 200:
            raise ValueError("max_new_tokens must be between 1 and 200")
        if not 0.1 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0.1 and 2.0")
        return prompt, max_new_tokens, temperature, top_p

    def inference(self, inputs):
        prompt, max_new_tokens, temperature, top_p = inputs
        encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=900).to(self.device)
        with torch.inference_mode():
            generated = self.model.generate(
                **encoded, max_new_tokens=max_new_tokens, do_sample=True,
                temperature=temperature, top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id, repetition_penalty=1.15
            )
        text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        continuation = text[len(prompt):].strip() if text.startswith(prompt) else text
        return {"generated_text": continuation}

    def postprocess(self, data):
        return [data]

_service = StoryHandler()
def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)
    inputs = _service.preprocess(data)
    output = _service.inference(inputs)
    return _service.postprocess(output)
