#!/usr/bin/env python3
"""
Download images related to Transformer Encoder Architecture
"""

import os
import urllib.request
import urllib.error
from pathlib import Path

# Image URLs and destinations
IMAGES = [
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/e/ea/Attention_Is_All_You_Need.png",
        "name": "encoder-architecture.png",
        "description": "Transformer Architecture with Encoder-Decoder"
    },
    {
        "url": "https://raw.githubusercontent.com/pytorch/tutorials/master/beginner_source/transformer_tutorial_data/attention.png",
        "name": "attention-mechanism.png",
        "description": "Multi-Head Attention Visualization"
    },
]

def download_image(url, filename, dest_path="static"):
    """Download image from URL and save to destination"""
    
    # Create destination directory if it doesn't exist
    os.makedirs(dest_path, exist_ok=True)
    
    filepath = os.path.join(dest_path, filename)
    
    try:
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"✓ Successfully saved: {filepath}")
        return True
    except urllib.error.URLError as e:
        print(f"✗ Failed to download {filename}: {e}")
        return False
    except Exception as e:
        print(f"✗ Error downloading {filename}: {e}")
        return False

def main():
    """Download all images"""
    print("=" * 60)
    print("Downloading Transformer Encoder Architecture Images")
    print("=" * 60)
    
    dest_path = os.path.join(
        os.path.dirname(__file__),
        "Attention",
        "Transform-Architecture",
        "Encoder",
        "static"
    )
    
    successful = 0
    failed = 0
    
    for img_info in IMAGES:
        if download_image(img_info["url"], img_info["name"], dest_path):
            successful += 1
        else:
            failed += 1
    
    print("=" * 60)
    print(f"Download Summary: {successful} successful, {failed} failed")
    print("=" * 60)

if __name__ == "__main__":
    main()
