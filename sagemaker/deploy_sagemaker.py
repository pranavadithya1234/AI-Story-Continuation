"""
Amazon SageMaker deployment script for the AI Story Continuation model.

This script:
  1. Packages the DistilGPT2 model artifacts and TorchServe handler into a
     tar.gz archive suitable for SageMaker.
  2. Uploads the archive to an S3 bucket.
  3. Creates a SageMaker Model, EndpointConfig and Endpoint.
  4. Invokes the endpoint with a sample prompt to verify it works.

Prerequisites:
  - AWS CLI configured (aws configure) with a faculty-approved account.
  - An IAM role with SageMaker execution permissions (passed via --role).
  - The model artifacts must already exist in model_artifacts/ (run
    scripts/package_model.sh first, or this script downloads them).

Usage:
  python sagemaker/deploy_sagemaker.py \
      --role arn:aws:iam::ACCOUNT:role/SageMakerRole \
      --bucket my-sagemaker-bucket \
      --region us-east-1

IMPORTANT:
  - NEVER commit AWS credentials into source code.
  - Delete the endpoint immediately after demonstration to avoid charges.
    Use  python sagemaker/cleanup.py  to tear down all resources.
"""

import argparse
import json
import os
import shutil
import tarfile
import time

import boto3
from transformers import AutoTokenizer, AutoModelForCausalLM


# ── Constants ────────────────────────────────────────────────────────────────
MODEL_NAME = "distilgpt2"
ENDPOINT_PREFIX = "storyforge-distilgpt2"
INSTANCE_TYPE = "ml.m5.large"  # CPU instance; adjust if GPU needed


# ── Step 1: Prepare model artifacts ──────────────────────────────────────────
def prepare_model_artifacts(artifact_dir: str = "model_artifacts"):
    """Download DistilGPT2 weights/tokenizer into a local directory."""
    os.makedirs(artifact_dir, exist_ok=True)
    print(f"[1/5] Downloading {MODEL_NAME} into {artifact_dir}/ ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(artifact_dir)
    model.save_pretrained(artifact_dir, safe_serialization=False)  # .bin for TorchServe compat
    print(f"      Saved model files: {os.listdir(artifact_dir)}")


# ── Step 2: Create model.tar.gz for SageMaker ───────────────────────────────
def create_model_tarball(artifact_dir: str = "model_artifacts",
                         handler_path: str = "torchserve/story_handler.py",
                         output_path: str = "model.tar.gz"):
    """Package model artifacts + handler into a tar.gz for SageMaker."""
    print(f"[2/5] Creating {output_path} ...")
    # SageMaker expects code/ for custom handler
    code_dir = os.path.join(artifact_dir, "code")
    os.makedirs(code_dir, exist_ok=True)
    shutil.copy2(handler_path, os.path.join(code_dir, "story_handler.py"))

    # Copy torchserve requirements into code/ so SageMaker installs them
    ts_req = "torchserve/requirements.txt"
    if os.path.exists(ts_req):
        shutil.copy2(ts_req, os.path.join(code_dir, "requirements.txt"))

    with tarfile.open(output_path, "w:gz") as tar:
        for item in os.listdir(artifact_dir):
            full = os.path.join(artifact_dir, item)
            tar.add(full, arcname=item)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"      Archive size: {size_mb:.1f} MB")
    return output_path


# ── Step 3: Upload to S3 ────────────────────────────────────────────────────
def upload_to_s3(tarball_path: str, bucket: str, prefix: str = "storyforge",
                 region: str = "us-east-1"):
    """Upload the model tarball to S3 and return the s3:// URI."""
    s3_key = f"{prefix}/model.tar.gz"
    s3_uri = f"s3://{bucket}/{s3_key}"
    print(f"[3/5] Uploading {tarball_path} → {s3_uri} ...")
    s3 = boto3.client("s3", region_name=region)
    s3.upload_file(tarball_path, bucket, s3_key)
    print("      Upload complete.")
    return s3_uri


# ── Step 4: Deploy endpoint ─────────────────────────────────────────────────
def deploy_endpoint(s3_uri: str, role: str, region: str = "us-east-1"):
    """Create SageMaker Model → EndpointConfig → Endpoint."""
    sm = boto3.client("sagemaker", region_name=region)
    ts = int(time.time())
    model_name = f"{ENDPOINT_PREFIX}-model-{ts}"
    config_name = f"{ENDPOINT_PREFIX}-config-{ts}"
    endpoint_name = f"{ENDPOINT_PREFIX}-ep-{ts}"

    # Choose a PyTorch inference container image
    # See: https://github.com/aws/deep-learning-containers/blob/master/available_images.md
    account_map = {
        "us-east-1": "763104351884",
        "us-west-2": "763104351884",
        "eu-west-1": "763104351884",
        "ap-south-1": "763104351884",
    }
    account_id = account_map.get(region, "763104351884")
    image_uri = (
        f"{account_id}.dkr.ecr.{region}.amazonaws.com/"
        f"pytorch-inference:2.1.0-cpu-py310-ubuntu20.04-sagemaker"
    )

    print(f"[4/5] Deploying SageMaker endpoint: {endpoint_name}")
    print(f"      Image : {image_uri}")
    print(f"      Model : {s3_uri}")

    # Create Model
    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": image_uri,
            "ModelDataUrl": s3_uri,
            "Environment": {
                "SAGEMAKER_PROGRAM": "story_handler.py",
                "SAGEMAKER_SUBMIT_DIRECTORY": s3_uri,
                "SAGEMAKER_MODEL_SERVER_TIMEOUT": "300",
            },
        },
        ExecutionRoleArn=role,
    )
    print(f"      Created Model: {model_name}")

    # Create Endpoint Config
    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[{
            "VariantName": "primary",
            "ModelName": model_name,
            "InstanceType": INSTANCE_TYPE,
            "InitialInstanceCount": 1,
        }],
    )
    print(f"      Created EndpointConfig: {config_name}")

    # Create Endpoint
    sm.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=config_name,
    )
    print(f"      Endpoint creation initiated: {endpoint_name}")
    print("      Waiting for endpoint to become InService (may take 5-10 min) ...")

    waiter = sm.get_waiter("endpoint_in_service")
    waiter.wait(EndpointName=endpoint_name, WaiterConfig={"Delay": 30, "MaxAttempts": 40})
    print(f"      ✅ Endpoint InService: {endpoint_name}")

    return endpoint_name, model_name, config_name


# ── Step 5: Test invocation ──────────────────────────────────────────────────
def invoke_endpoint(endpoint_name: str, region: str = "us-east-1"):
    """Send a sample prompt to the deployed endpoint and print the result."""
    runtime = boto3.client("sagemaker-runtime", region_name=region)
    payload = json.dumps({
        "prompt": "One day, a student discovered a mysterious box in the college library",
        "max_new_tokens": 60,
        "temperature": 0.8,
        "top_p": 0.9,
    })

    print(f"[5/5] Invoking endpoint with test prompt ...")
    start = time.time()
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=payload,
    )
    latency = (time.time() - start) * 1000
    body = json.loads(response["Body"].read().decode("utf-8"))
    print(f"      Response ({latency:.0f} ms): {json.dumps(body, indent=2)}")
    return body


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Deploy StoryForge AI to SageMaker")
    parser.add_argument("--role", required=True,
                        help="SageMaker execution IAM role ARN")
    parser.add_argument("--bucket", required=True,
                        help="S3 bucket for model artifacts")
    parser.add_argument("--region", default="us-east-1",
                        help="AWS region (default: us-east-1)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip model download if model_artifacts/ exists")
    args = parser.parse_args()

    if not args.skip_download:
        prepare_model_artifacts()
    else:
        print("[1/5] Skipping download (--skip-download).")

    tarball = create_model_tarball()
    s3_uri = upload_to_s3(tarball, args.bucket, region=args.region)
    endpoint_name, model_name, config_name = deploy_endpoint(
        s3_uri, args.role, args.region
    )

    # Save deployment info for cleanup and UI integration
    info = {
        "endpoint_name": endpoint_name,
        "model_name": model_name,
        "config_name": config_name,
        "s3_uri": s3_uri,
        "region": args.region,
        "bucket": args.bucket,
    }
    info_path = "sagemaker/deployment_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"\n      Deployment info saved to {info_path}")

    invoke_endpoint(endpoint_name, args.region)

    print("\n" + "=" * 60)
    print("DEPLOYMENT COMPLETE")
    print("=" * 60)
    print(f"Endpoint name : {endpoint_name}")
    print(f"Region        : {args.region}")
    print()
    print("⚠️  Remember to run  python sagemaker/cleanup.py  after your")
    print("   demonstration to delete the endpoint and avoid AWS charges.")


if __name__ == "__main__":
    main()
