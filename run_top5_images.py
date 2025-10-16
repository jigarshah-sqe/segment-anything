#!/usr/bin/env python3
"""
Script to process top 5 images with most COAL annotations.
"""

import os
import sys
import json
import subprocess

def main():
    # Top 5 image IDs with most COAL annotations
    top5_image_ids = [6, 26, 19, 20, 1]
    
    print("🚀 Processing top 5 images with most COAL annotations...")
    print("="*60)
    
    for i, image_id in enumerate(top5_image_ids, 1):
        print(f"\n📸 Processing image {i}/5: ID {image_id}")
        print("-" * 40)
        
        try:
            # Run the enhanced comparison script for this image
            cmd = [
                "python", "run_sam_enhanced_comparison.py",
                "--image-id", str(image_id),
                "--device", "cpu"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 10 min timeout
            
            if result.returncode == 0:
                print(f"✅ Successfully processed image ID {image_id}")
            else:
                print(f"❌ Failed to process image ID {image_id}")
                print(f"Error: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Timeout for image ID {image_id} (10 minutes)")
        except Exception as e:
            print(f"❌ Error processing image ID {image_id}: {e}")
    
    print("\n" + "="*60)
    print("🎉 Top 5 images processing completed!")
    print("="*60)

if __name__ == "__main__":
    main()
