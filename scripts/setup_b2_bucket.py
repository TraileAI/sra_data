#!/usr/bin/env python3
"""
Create Backblaze B2 bucket for CSV file storage.

This script creates a new B2 bucket and configures it for public file access.

Requirements:
    pip install b2sdk

Usage:
    python scripts/setup_b2_bucket.py --bucket-name sra-data-csv

Environment Variables:
    B2_APPLICATION_KEY_ID - Your B2 application key ID
    B2_APPLICATION_KEY - Your B2 application key
"""

import os
import sys
import argparse
import logging

try:
    from b2sdk.v2 import InMemoryAccountInfo, B2Api
except ImportError:
    print("Error: b2sdk not installed. Run: pip install b2sdk")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_b2_api():
    """Set up B2 API client."""
    key_id = os.getenv('B2_APPLICATION_KEY_ID')
    key = os.getenv('B2_APPLICATION_KEY')

    if not key_id or not key:
        logger.error("B2 credentials not found in environment variables")
        logger.error("Please set B2_APPLICATION_KEY_ID and B2_APPLICATION_KEY")
        return None

    info = InMemoryAccountInfo()
    api = B2Api(info)

    try:
        api.authorize_account("production", key_id, key)
        logger.info("Successfully authenticated with Backblaze B2")
        return api
    except Exception as e:
        logger.error(f"Failed to authenticate with B2: {e}")
        return None

def create_bucket(bucket_name, public=True):
    """Create a new B2 bucket."""
    api = setup_b2_api()
    if not api:
        return False

    try:
        # Check if bucket already exists
        existing_buckets = api.list_buckets()
        for bucket in existing_buckets:
            if bucket.name == bucket_name:
                logger.info(f"Bucket '{bucket_name}' already exists")
                bucket_id = bucket.id_
                bucket_url = f"https://f{bucket_id[:-3]}.backblazeb2.com/file/{bucket_name}/"
                logger.info(f"Bucket URL: {bucket_url}")
                return True

        # Create new bucket
        bucket_type = 'allPublic' if public else 'allPrivate'
        logger.info(f"Creating bucket '{bucket_name}' with type '{bucket_type}'...")

        bucket = api.create_bucket(bucket_name, bucket_type)
        bucket_id = bucket.id_

        logger.info(f"✅ Successfully created bucket: {bucket_name}")
        logger.info(f"Bucket ID: {bucket_id}")

        # Generate the bucket URL
        bucket_url = f"https://f{bucket_id[:-3]}.backblazeb2.com/file/{bucket_name}/"
        logger.info(f"Bucket URL: {bucket_url}")

        logger.info(f"\n🎉 Bucket setup complete!")
        logger.info(f"Set this environment variable in Render:")
        logger.info(f"B2_BUCKET_URL={bucket_url}")

        return True

    except Exception as e:
        logger.error(f"Failed to create bucket: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Create B2 bucket for CSV storage')
    parser.add_argument('--bucket-name', required=True, help='B2 bucket name')
    parser.add_argument('--private', action='store_true', help='Create private bucket (default: public)')

    args = parser.parse_args()

    # Validate bucket name
    bucket_name = args.bucket_name.lower()
    if not bucket_name.replace('-', '').replace('_', '').isalnum():
        logger.error("Bucket name can only contain letters, numbers, hyphens, and underscores")
        sys.exit(1)

    success = create_bucket(bucket_name, public=not args.private)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()