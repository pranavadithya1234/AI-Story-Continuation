# StoryForge AI — AI Story Continuation System

**NLP Mini-Project 3** — A student project demonstrating end-to-end NLP model deployment using pretrained DistilGPT2, PyTorch, TorchServe, and Amazon SageMaker, with a Streamlit web interface.

---

## Project Overview

A user enters the beginning of a story (e.g., *"One day, a student discovered a mysterious box in the college library…"*).  The application generates a creative continuation using DistilGPT2.  The user can control continuation length, temperature, and nucleus sampling parameters.

**Technologies used:**

| Technology | Role |
|---|---|
| **DistilGPT2** | Pretrained language model for text generation |
| **PyTorch** | Deep-learning framework for model loading and inference |
| **TorchServe** | Model-serving framework exposing inference as a REST API |
| **Amazon SageMaker** | Managed AWS platform for cloud deployment |
| **Streamlit** | Web-based user interface |

---

## 1. Run Locally (macOS / Linux)

### Prerequisites
- Python 3.10–3.14
- Internet connection for the first model download (~350 MB)

### Steps

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3. Launch the Streamlit app
streamlit run app.py
```

Open the local URL Streamlit prints (usually http://localhost:8501).  The first generation downloads DistilGPT2 automatically.

---

## 2. What the App Does

- **Story input** — Text area for the opening sentence or paragraph
- **Continuation length** — Short (50 tokens), Medium (100), Long (150)
- **Temperature** — Controls creativity (0.2 = focused, 1.5 = varied)
- **Top-p (nucleus sampling)** — Controls diversity of token selection
- **Inference mode selector** — Switch between Local PyTorch, TorchServe API, and SageMaker Endpoint
- **Latency display** — Shows response time in milliseconds
- **Download** — Save the generated continuation as a text file
- **Input validation** — Rejects empty and >4000-character prompts

---

## 3. TorchServe — Local Model Serving

> **Note:** TorchServe is in limited maintenance.  Confirm a compatible TorchServe / PyTorch / Python combination with your instructor before installation.

### 3.1 Package the model

```bash
# Install TorchServe tools (in a compatible environment)
pip install torchserve torch-model-archiver

# Download model and create .mar archive
bash scripts/package_model.sh
```

### 3.2 Start TorchServe

```bash
torchserve --start \
    --model-store model-store \
    --models storymodel=storymodel.mar \
    --ts-config torchserve/config.properties
```

### 3.3 Test TorchServe

```bash
bash scripts/test_torchserve.sh

# Or manually:
curl -s http://localhost:8080/predictions/storymodel \
  -H "Content-Type: application/json" \
  -d '{"prompt":"One night, the library door opened by itself.","max_new_tokens":50,"temperature":0.8}'
```

### 3.4 Use from Streamlit

In the Streamlit sidebar, select **TorchServe API** as the inference mode.  The default URL is `http://localhost:8080/predictions/storymodel`.

---

## 4. Amazon SageMaker Deployment

> **Important:** Use a faculty-approved AWS account.  Confirm the region, instance type, and estimated charges with your instructor before deploying.  Never commit AWS credentials to source code.

### 4.1 Deploy

```bash
python sagemaker/deploy_sagemaker.py \
    --role arn:aws:iam::ACCOUNT_ID:role/SageMakerExecutionRole \
    --bucket your-sagemaker-bucket \
    --region us-east-1
```

This script:
1. Downloads and packages DistilGPT2 model artifacts
2. Creates `model.tar.gz` with handler code
3. Uploads to S3
4. Creates SageMaker Model → EndpointConfig → Endpoint
5. Tests the endpoint with a sample prompt
6. Saves deployment info to `sagemaker/deployment_info.json`

### 4.2 Use from Streamlit

In the sidebar, select **SageMaker Endpoint**, enter the endpoint name and region.  AWS credentials must be configured via `aws configure`.

### 4.3 Clean up (IMPORTANT!)

```bash
python sagemaker/cleanup.py
```

**Run this immediately after your demonstration** to delete the endpoint and avoid ongoing AWS charges.

---

## 5. API Contract

### Request
```json
{
  "prompt": "A student found a hidden door...",
  "max_new_tokens": 100,
  "temperature": 0.8,
  "top_p": 0.9
}
```

### Response
```json
{
  "generated_text": "She pushed it open cautiously and ..."
}
```

### Error
```json
{
  "error": "prompt is required"
}
```

---

## 6. Testing

### Run the automated test suite

```bash
.venv/bin/python tests/run_tests.py
```

This generates `tests/test_results.md` with:
- 12 valid story prompts with outputs
- Edge cases (empty prompt, long prompt)
- Short / Medium / Long length comparison
- Temperature 0.3 vs 1.2 comparison
- Reproducibility test (same prompt, two runs)
- Latency measurements

### Manual test plan

| Test | Details |
|---|---|
| 10 valid prompts | Diverse story openings |
| Empty prompt | Should show error |
| >4000-char prompt | Should show error or handle gracefully |
| Short vs Long output | Compare 50 vs 150 max_new_tokens |
| Temperature 0.3 vs 1.2 | Same prompt, compare outputs |
| Local vs TorchServe vs SageMaker | Compare response format and latency |

---

## 7. Project Structure

```
AI-Story-Continuation/
├── app.py                          # Streamlit UI (3 inference modes)
├── story_generator.py              # Local PyTorch inference module
├── requirements.txt                # Python dependencies
├── torchserve/
│   ├── story_handler.py            # TorchServe custom handler
│   ├── config.properties           # TorchServe configuration
│   └── requirements.txt            # Handler dependencies
├── scripts/
│   ├── package_model.sh            # Create .mar model archive
│   └── test_torchserve.sh          # cURL test for TorchServe
├── sagemaker/
│   ├── deploy_sagemaker.py         # Automated SageMaker deployment
│   └── cleanup.py                  # Delete SageMaker resources
├── tests/
│   ├── run_tests.py                # Automated test runner
│   └── test_results.md             # Generated test report
├── docs/
│   └── PROJECT_REPORT.md           # Full project report
└── README.md                       # This file
```

---

## 8. Limitations

- DistilGPT2 is a small general language model.  Output may be repetitive, incoherent, or stop unexpectedly.
- This is a demonstration of inference serving and deployment, not a production-grade creative writing service.
- Generated text may contain biases from the training data.
- The model has a 1024-token context window limit.

---

## 9. Submission Checklist

- [x] Project source code and requirements file
- [x] Model handler and TorchServe configuration
- [x] Instructions to run the application locally
- [x] Evidence of local PyTorch inference (run `tests/run_tests.py`)
- [x] Evidence of TorchServe API testing (`scripts/test_torchserve.sh`)
- [x] SageMaker deployment and invocation script (`sagemaker/deploy_sagemaker.py`)
- [x] User interface with inference mode selection
- [x] 10+ test prompts with generated outputs (`tests/test_results.md`)
- [x] Project report (`docs/PROJECT_REPORT.md`)
- [x] Cleanup script to delete cloud resources (`sagemaker/cleanup.py`)
