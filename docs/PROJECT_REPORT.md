# Project Report — AI Story Continuation System (Project 3)

## Cover Page

| Field | Details |
|---|---|
| **Project title** | AI Story Continuation System (Project 3) |
| **Student name(s)** | ____________________________________________ |
| **Roll number(s)** | ____________________________________________ |
| **Class / Section** | ____________________________________________ |
| **Faculty mentor** | ____________________________________________ |
| **Date of submission** | ____________________________________________ |

---

## 1. Abstract and Problem Statement

### Abstract

This project implements an end-to-end NLP application that accepts the beginning of a story from a user and generates a creative continuation using the pretrained **DistilGPT2** language model.  The system is built with **PyTorch** for model inference, served through **TorchServe** as a REST API, deployed on **Amazon SageMaker** for cloud accessibility, and presented through a **Streamlit** web interface.  The project demonstrates the complete lifecycle of deploying a pretrained language model — from local experimentation to cloud-hosted inference.

### Problem Statement

How can you develop a creative writing application that accepts the beginning of a story and uses DistilGPT2 to generate a continuation, with options to control length and creativity, and deploy it through TorchServe and Amazon SageMaker?

---

## 2. Objectives and Functional Requirements

### Objectives

1. Load and use the pretrained DistilGPT2 model for autoregressive text generation.
2. Provide user-facing controls for continuation length (`max_new_tokens`) and creativity (`temperature`).
3. Expose the model through a TorchServe custom handler with a documented REST API.
4. Deploy the serving solution to Amazon SageMaker and verify cloud-based inference.
5. Build a Streamlit interface that connects to local, TorchServe, and SageMaker inference backends.

### Functional Requirements

| # | Requirement | Status |
|---|---|---|
| 1 | Text area for story opening input | ✅ Implemented |
| 2 | Continuation-length option via `max_new_tokens` (Short / Medium / Long) | ✅ Implemented |
| 3 | Temperature selection within a predefined range (0.2–1.5) | ✅ Implemented |
| 4 | Generate continuation using DistilGPT2 and PyTorch | ✅ Implemented |
| 5 | TorchServe custom handler for serving | ✅ Implemented |
| 6 | Amazon SageMaker deployment | ✅ Implemented |
| 7 | Display original prompt and continuation separately | ✅ Implemented |
| 8 | Input validation (empty prompt, long prompt) | ✅ Implemented |

---

## 3. Technology Description

### 3.1 DistilGPT2

DistilGPT2 is a distilled version of GPT-2 created by Hugging Face.  It retains 82 million parameters (compared to GPT-2's 124 million) while preserving most of the language-generation capability.  It uses a transformer decoder architecture with causal (left-to-right) self-attention, making it suitable for autoregressive text generation.

- **Parameters:** ~82 million
- **Architecture:** 6 transformer decoder layers, 12 attention heads, 768 hidden dimension
- **Vocabulary:** 50,257 BPE tokens
- **Training data:** OpenWebText (a recreation of WebText)

### 3.2 PyTorch

PyTorch is an open-source deep-learning framework developed by Meta AI.  In this project, PyTorch provides:
- Tensor computation for model inference
- `torch.inference_mode()` context for efficient forward passes without gradient tracking
- Model loading and weight management
- Integration with Hugging Face Transformers via `AutoModelForCausalLM`

### 3.3 TorchServe

TorchServe is a model-serving framework for PyTorch models.  It exposes trained models as REST APIs with features including:
- **Custom handlers** — Python classes that define preprocessing, inference, and postprocessing logic
- **Model archiver** — Packages model weights, handler code, and dependencies into a `.mar` file
- **Worker management** — Configurable number of inference workers
- **Health and metrics endpoints** — For monitoring model server status

In this project, `story_handler.py` implements a custom `BaseHandler` subclass that tokenizes the input prompt, runs generation, and decodes the output.

### 3.4 Amazon SageMaker

Amazon SageMaker is a fully managed AWS service for building, training, and deploying machine learning models.  In this project it is used to:
- Host the TorchServe-based inference container in the cloud
- Provide a scalable HTTPS endpoint for real-time inference
- Manage the compute instance lifecycle

The deployment uses a pre-built PyTorch inference container image from AWS Deep Learning Containers.

---

## 4. System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    USER (Browser)                        │
│          Streamlit UI — http://localhost:8501             │
└────────────┬──────────────┬──────────────┬──────────────┘
             │ Local        │ TorchServe   │ SageMaker
             ▼              ▼              ▼
     ┌──────────────┐ ┌──────────┐ ┌───────────────────┐
     │  PyTorch     │ │ TorchServe│ │ Amazon SageMaker  │
     │  (in-process)│ │ REST API  │ │ Endpoint (HTTPS)  │
     │              │ │ :8080     │ │                   │
     └──────┬───────┘ └────┬─────┘ └────────┬──────────┘
            │              │                │
            ▼              ▼                ▼
     ┌─────────────────────────────────────────────────┐
     │          DistilGPT2 (PyTorch)                    │
     │  Tokenize → Generate → Decode → Return text      │
     └─────────────────────────────────────────────────┘
```

### Workflow

| Stage | Component | Responsibility |
|---|---|---|
| 1 | User interface (Streamlit) | Accepts story prompt, selects generation settings, displays continuation |
| 2 | Inference backend (selectable) | Routes request to local model, TorchServe API, or SageMaker endpoint |
| 3 | TorchServe / SageMaker | Receives JSON request, routes to model worker |
| 4 | PyTorch + DistilGPT2 | Tokenizes prompt, runs autoregressive generation, produces token IDs |
| 5 | Response processing | Decodes token IDs, extracts continuation (removes original prompt), returns JSON |

---

## 5. Implementation Details

### 5.1 Core Modules

| File | Purpose |
|---|---|
| `story_generator.py` | Loads DistilGPT2, validates inputs, runs generation, returns continuation text |
| `app.py` | Streamlit UI with three inference modes (Local / TorchServe / SageMaker) |
| `torchserve/story_handler.py` | TorchServe custom handler implementing `BaseHandler` |
| `torchserve/config.properties` | TorchServe server configuration |
| `scripts/package_model.sh` | Downloads model and creates `.mar` archive for TorchServe |
| `scripts/test_torchserve.sh` | cURL test script for TorchServe endpoint |
| `sagemaker/deploy_sagemaker.py` | Automated SageMaker deployment: package → S3 → Model → Endpoint |
| `sagemaker/cleanup.py` | Deletes SageMaker resources to stop AWS charges |
| `tests/run_tests.py` | Runs 10+ test prompts and generates test_results.md |

### 5.2 Key Implementation Decisions

1. **`lru_cache` for model loading** — The model and tokenizer are loaded once and cached in memory to avoid repeated downloads.

2. **`torch.inference_mode()`** — Used instead of `torch.no_grad()` for better performance during inference (disables autograd entirely).

3. **`repetition_penalty=1.15`** — Applied to reduce the tendency of DistilGPT2 to produce repetitive phrases.

4. **`truncation=True, max_length=900`** — Long prompts are truncated at the tokenizer level to stay within the model's 1024-token context window while leaving room for generation.

5. **Continuation extraction** — The decoded output includes the original prompt; we strip it by slicing `full_text[len(prompt):]` to display only the new content.

### 5.3 API Contract

**Request JSON:**
```json
{
  "prompt": "A student found a hidden door...",
  "max_new_tokens": 100,
  "temperature": 0.8,
  "top_p": 0.9
}
```

**Response JSON:**
```json
{
  "generated_text": "She pushed it open cautiously and ..."
}
```

**Error Response:**
```json
{
  "error": "prompt is required"
}
```

---

## 6. Screenshots

> **Instructions:** Replace the placeholders below with actual screenshots before submission.

### 6.1 Streamlit UI — Story Input
> *[Insert screenshot of the Streamlit interface with a story opening typed in]*

### 6.2 Streamlit UI — Generated Continuation
> *[Insert screenshot showing the generated continuation alongside the original prompt]*

### 6.3 TorchServe API — cURL Test
> *[Insert screenshot of terminal showing the curl command and JSON response from TorchServe]*

### 6.4 SageMaker — Endpoint Creation
> *[Insert screenshot of SageMaker console showing the endpoint in "InService" status]*

### 6.5 SageMaker — Endpoint Invocation
> *[Insert screenshot of the deploy_sagemaker.py output showing successful invocation]*

### 6.6 SageMaker — Endpoint Deletion
> *[Insert screenshot showing cleanup.py output confirming resource deletion]*

---

## 7. Test Cases, Generated Outputs and Observations

> Full test results are available in `tests/test_results.md` generated by `tests/run_tests.py`.

### 7.1 Summary of Tests Performed

| Category | Count | Details |
|---|---|---|
| Valid story prompts | 12 | Diverse themes: mystery, sci-fi, fantasy, adventure, horror |
| Edge cases | 3 | Empty prompt, whitespace-only, over-4000-character prompt |
| Length variations | 3 | Short (50 tokens), Medium (100), Long (150) |
| Temperature comparison | 2 | Low (0.3) vs High (1.2) on same prompt |
| Reproducibility test | 2 | Same prompt run twice to show sampling variation |
| **Total tests** | **22+** | |

### 7.2 Sample Test Cases (from Assignment)

| Story Opening | Test Variation | Observation |
|---|---|---|
| "One day, a student discovered a mysterious box…" | Short continuation (50 tokens) | Generated a concise continuation; story remained coherent |
| "A robot entered the classroom for the first time…" | Long continuation (150 tokens) | Longer output; some repetition towards the end |
| "It was a dark and rainy night…" | Low temp (0.3) vs High temp (1.2) | Low temp: focused, predictable. High temp: creative but less coherent |

### 7.3 Comparison: Local PyTorch vs TorchServe vs SageMaker

| Aspect | Local PyTorch | TorchServe (local) | SageMaker |
|---|---|---|---|
| Response format | Python string | JSON `{"generated_text":"..."}` | JSON `{"generated_text":"..."}` |
| First-request latency | Higher (model loading) | Higher (model loading) | Higher (cold start) |
| Subsequent latency | Low (in-process) | Low + network overhead | Medium (network + container) |
| Exact wording match | N/A | Different (sampling) | Different (sampling) |

> **Note:** When `do_sample=True`, each run produces different output.  The comparison verifies that the response *format* and *execution* are consistent, not the exact wording.

---

## 8. Questions and Answers (Required by Assignment)

### Q1: What is autoregressive text generation?

Autoregressive text generation is a process where a model generates text one token at a time, with each new token conditioned on all previously generated tokens.  At each step, the model computes a probability distribution over the vocabulary given the preceding context, selects the next token (by sampling or greedy selection), appends it to the sequence, and repeats until a stopping condition is met (e.g., `max_new_tokens` reached or an end-of-sequence token is produced).

### Q2: How does DistilGPT2 use preceding text to predict subsequent tokens?

DistilGPT2 uses a stack of transformer decoder layers with **causal self-attention**.  The input tokens are converted to embeddings and passed through the layers.  Each attention head computes attention scores only over tokens at or before the current position (masking future tokens).  The final layer's output is projected onto the vocabulary to produce logits for the next token.  This means the prediction for position *t* depends on all tokens from position 1 to *t−1*.

### Q3: What is the effect of `max_new_tokens` on the generated story?

`max_new_tokens` sets an upper limit on how many new tokens the model can generate beyond the input prompt.  A lower value (e.g., 50) produces shorter continuations, while a higher value (e.g., 150) allows longer text.  The model may stop earlier if it produces an end-of-sequence token.  Very high values can lead to repetitive or degenerate text with small models like DistilGPT2.

### Q4: How does temperature influence the diversity of generated text?

Temperature scales the logits before the softmax function.  Given logits *z*, the probability of token *i* is:

$$P(i) = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$

- **Low temperature (e.g., 0.3):** Sharpens the distribution; the model strongly favours high-probability tokens, producing more predictable, repetitive text.
- **High temperature (e.g., 1.2):** Flattens the distribution; lower-probability tokens get a better chance, producing more varied and creative (but potentially less coherent) text.
- **Temperature = 1.0:** Uses the raw model probabilities unchanged.

### Q5: Why can two runs with the same prompt produce different continuations?

When `do_sample=True`, the model *samples* from the probability distribution at each generation step rather than always choosing the highest-probability token (greedy decoding).  Combined with `top_p` (nucleus sampling), this introduces randomness — different random seeds lead to different token selections at each step, producing different continuations.

### Q6: How does the deployed endpoint receive and return story text?

The SageMaker endpoint receives an HTTPS POST request with a JSON body containing the prompt and generation parameters.  Inside the container, TorchServe routes the request to the custom handler (`story_handler.py`), which:
1. **Preprocesses** — Parses JSON, validates the prompt, extracts parameters.
2. **Runs inference** — Tokenizes, generates with PyTorch, decodes output.
3. **Postprocesses** — Extracts the continuation and wraps it in a JSON response.

The response is returned as JSON (`{"generated_text": "..."}`) over HTTPS to the calling client.

---

## 9. Limitations, Ethical Considerations and Future Enhancements

### Limitations

1. **DistilGPT2 is a small model** — It often produces repetitive, incoherent, or abruptly ending text.  It is not suitable for production creative-writing applications.
2. **No factual accuracy** — The model generates plausible-sounding text without any grounding in facts.
3. **Context window limit** — The model supports a maximum of 1024 tokens (prompt + generated), limiting the length of input stories.
4. **No memory across requests** — Each request is independent; the model does not remember previous interactions.
5. **Cold-start latency** — The first request (local or cloud) includes model loading time.

### Ethical Considerations

- Generated text may contain biased, offensive, or inappropriate content reflecting biases in the training data (OpenWebText).
- The application displays a disclaimer that output is AI-generated and may be inaccurate.
- AWS credentials and account information must never be included in source code or reports.

### Future Enhancements

1. Use a larger model (e.g., GPT-2 Medium or a fine-tuned story model) for better narrative quality.
2. Add genre-specific prompt templates (e.g., "Write a mystery story: …").
3. Implement streaming generation for real-time token-by-token display.
4. Add user feedback mechanisms to rate generated continuations.
5. Fine-tune DistilGPT2 on a creative-writing dataset for domain-specific generation.

---

## 10. Conclusion

This project successfully demonstrates an end-to-end NLP model deployment workflow:

1. **Local inference** with DistilGPT2 and PyTorch — confirming the model loads, tokenizes, generates, and returns story continuations.
2. **Model serving** with TorchServe — packaging the model into a `.mar` archive and exposing it as a REST API with a custom handler.
3. **Cloud deployment** with Amazon SageMaker — uploading the model to S3, creating an endpoint, and invoking it remotely.
4. **User interface** with Streamlit — allowing users to enter story prompts, adjust generation settings, and view continuations from any of the three inference backends.

The system handles edge cases (empty prompts, overly long prompts), provides latency measurements, and compares outputs across temperature and length settings.  While DistilGPT2's output quality is limited, the project fulfils its objective of demonstrating the complete model-serving and deployment pipeline.

---

## 11. References

1. Hugging Face — DistilGPT2 Model Card: https://huggingface.co/distilgpt2
2. Hugging Face Transformers Documentation: https://huggingface.co/docs/transformers
3. PyTorch Documentation: https://pytorch.org/docs/stable/
4. TorchServe Documentation: https://pytorch.org/serve/
5. Amazon SageMaker Developer Guide: https://docs.aws.amazon.com/sagemaker/latest/dg/
6. AWS Deep Learning Containers: https://github.com/aws/deep-learning-containers
7. Sanh, V. et al. "DistilBERT, a distilled version of BERT." *arXiv:1910.01108*, 2019.
8. Radford, A. et al. "Language Models are Unsupervised Multitask Learners." *OpenAI*, 2019.
9. Streamlit Documentation: https://docs.streamlit.io/
