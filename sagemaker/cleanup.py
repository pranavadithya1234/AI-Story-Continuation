"""
Cleanup script — deletes all SageMaker resources created by deploy_sagemaker.py.

Usage:
  python sagemaker/cleanup.py
  python sagemaker/cleanup.py --region us-east-1   # if different from saved info

This reads sagemaker/deployment_info.json to find the endpoint, config and model
names, then deletes them in order.  It also optionally removes the S3 model
artifact.

IMPORTANT: Run this immediately after your demonstration to avoid AWS charges.
"""

import argparse
import json
import os
import sys

import boto3
from botocore.exceptions import ClientError


def load_deployment_info(path="sagemaker/deployment_info.json"):
    if not os.path.exists(path):
        print(f"ERROR: {path} not found.  Nothing to clean up.")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def delete_endpoint(sm, name):
    try:
        sm.delete_endpoint(EndpointName=name)
        print(f"  ✅ Deleted endpoint: {name}")
    except ClientError as e:
        print(f"  ⚠️  Could not delete endpoint {name}: {e}")


def delete_endpoint_config(sm, name):
    try:
        sm.delete_endpoint_config(EndpointConfigName=name)
        print(f"  ✅ Deleted endpoint config: {name}")
    except ClientError as e:
        print(f"  ⚠️  Could not delete endpoint config {name}: {e}")


def delete_model(sm, name):
    try:
        sm.delete_model(ModelName=name)
        print(f"  ✅ Deleted model: {name}")
    except ClientError as e:
        print(f"  ⚠️  Could not delete model {name}: {e}")


def delete_s3_artifact(s3_uri, region):
    if not s3_uri:
        return
    # Parse s3://bucket/key
    parts = s3_uri.replace("s3://", "").split("/", 1)
    bucket, key = parts[0], parts[1]
    try:
        s3 = boto3.client("s3", region_name=region)
        s3.delete_object(Bucket=bucket, Key=key)
        print(f"  ✅ Deleted S3 object: {s3_uri}")
    except ClientError as e:
        print(f"  ⚠️  Could not delete S3 object {s3_uri}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Clean up SageMaker deployment")
    parser.add_argument("--region", default=None, help="Override AWS region")
    parser.add_argument("--keep-s3", action="store_true",
                        help="Do not delete the S3 model artifact")
    args = parser.parse_args()

    info = load_deployment_info()
    region = args.region or info.get("region", "us-east-1")
    sm = boto3.client("sagemaker", region_name=region)

    print("=" * 50)
    print("CLEANING UP SAGEMAKER RESOURCES")
    print("=" * 50)

    # Delete in reverse order: endpoint → config → model → S3
    delete_endpoint(sm, info["endpoint_name"])
    delete_endpoint_config(sm, info["config_name"])
    delete_model(sm, info["model_name"])

    if not args.keep_s3:
        delete_s3_artifact(info.get("s3_uri"), region)

    # Remove the deployment info file
    info_path = "sagemaker/deployment_info.json"
    if os.path.exists(info_path):
        os.remove(info_path)
        print(f"  ✅ Removed {info_path}")

    print("\n✅ All SageMaker resources deleted.  No further charges will accrue.")


if __name__ == "__main__":
    main()
