#!/usr/bin/env python3
"""
Script to run SAM model for COAL segmentation on a single image.
"""

import os
import sys
import json
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import logging

# Add the segment_anything module to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'segment_anything'))

from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_sam_checkpoint(model_type="vit_h", checkpoint_dir="./checkpoints"):
    """Download SAM checkpoint if not already present."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if model_type == "vit_h":
        checkpoint_name = "sam_vit_h_4b8939.pth"
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    elif model_type == "vit_l":
        checkpoint_name = "sam_vit_l_0b3195.pth"
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
    elif model_type == "vit_b":
        checkpoint_name = "sam_vit_b_01ec64.pth"
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    
    if not os.path.exists(checkpoint_path):
        logger.info(f"Downloading SAM checkpoint: {checkpoint_name}")
        import urllib.request
        urllib.request.urlretrieve(url, checkpoint_path)
        logger.info(f"Downloaded to: {checkpoint_path}")
    else:
        logger.info(f"Checkpoint already exists: {checkpoint_path}")
    
    return checkpoint_path

def load_sam_model(model_type="vit_h", device="cpu"):
    """Load SAM model and predictor."""
    checkpoint_path = download_sam_checkpoint(model_type)
    
    logger.info(f"Loading SAM model: {model_type}")
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    
    predictor = SamPredictor(sam)
    logger.info(f"SAM model loaded on device: {device}")
    
    return predictor

def load_sam_automatic_generator(model_type="vit_h", device="cpu"):
    """Load SAM automatic mask generator."""
    checkpoint_path = download_sam_checkpoint(model_type)
    
    logger.info(f"Loading SAM automatic mask generator: {model_type}")
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    
    # Configure automatic mask generator
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
    )
    
    logger.info(f"SAM automatic mask generator loaded on device: {device}")
    return mask_generator

def show_mask(mask, ax, random_color=False):
    """Display mask on axis."""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])  # Blue color for COAL
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    """Display points on axis."""
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', 
               s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', 
               s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax, color='green', linewidth=2):
    """Display bounding box on axis."""
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0,0,0,0), 
                               linewidth=linewidth))

def show_manual_annotations(annotations, ax, color='red', alpha=0.3):
    """Display manual annotations on axis."""
    for annotation in annotations:
        if 'segmentation' in annotation and annotation['segmentation']:
            # Convert segmentation to polygon
            segmentation = annotation['segmentation'][0]  # First segment
            if len(segmentation) >= 6:  # At least 3 points (x,y pairs)
                # Reshape to (x, y) pairs
                points = np.array(segmentation).reshape(-1, 2)
                # Create polygon
                from matplotlib.patches import Polygon
                polygon = Polygon(points, closed=True, facecolor=color, alpha=alpha, 
                                 edgecolor=color, linewidth=1)
                ax.add_patch(polygon)

def get_coal_annotations_for_image(annotations_data, image_id):
    """Get COAL annotations for a specific image."""
    coal_annotations = []
    for annotation in annotations_data['annotations']:
        if (annotation['image_id'] == image_id and 
            annotation['category_id'] == 1 and  # COAL category
            annotation['segmentation'] and 
            any(len(segment) > 0 for segment in annotation['segmentation'])):
            coal_annotations.append(annotation)
    return coal_annotations

def create_prompt_from_annotation(annotation, image_shape):
    """Create SAM prompt from annotation data."""
    # For now, we'll use the center of the bounding box as a point prompt
    bbox = annotation['bbox']  # [x, y, width, height]
    center_x = bbox[0] + bbox[2] / 2
    center_y = bbox[1] + bbox[3] / 2
    
    # Convert to integer coordinates
    point = np.array([[int(center_x), int(center_y)]])
    label = np.array([1])  # Foreground point
    
    return point, label

def run_sam_on_image(image_path, annotations_data, image_id, output_dir="./sam_output", device="cpu"):
    """Run SAM automatic segmentation on a single image and compare with manual annotations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    logger.info(f"Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    logger.info(f"Image shape: {image.shape}")
    
    # Get COAL annotations for this image
    coal_annotations = get_coal_annotations_for_image(annotations_data, image_id)
    logger.info(f"Found {len(coal_annotations)} manual COAL annotations for image {image_id}")
    
    # Load SAM automatic mask generator
    mask_generator = load_sam_automatic_generator(device=device)
    
    # Generate automatic masks
    logger.info("Running SAM automatic segmentation...")
    sam_masks = mask_generator.generate(image)
    logger.info(f"SAM found {len(sam_masks)} automatic segments")
    
    # Filter SAM masks by area and quality (to focus on COAL-like objects)
    filtered_masks = []
    for mask_info in sam_masks:
        # Filter by area (COAL objects are typically medium-sized)
        area = mask_info['area']
        if 1000 < area < 50000:  # Reasonable size for COAL objects
            # Filter by stability score
            if mask_info['stability_score'] > 0.8:
                filtered_masks.append(mask_info)
    
    logger.info(f"Filtered to {len(filtered_masks)} high-quality segments")
    
    # Create combined visualization
    logger.info("Creating comparison visualization...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image', fontsize=14)
    axes[0, 0].axis('off')
    
    # Original image with manual annotations
    axes[0, 1].imshow(image)
    show_manual_annotations(coal_annotations, axes[0, 1], color='red', alpha=0.3)
    axes[0, 1].set_title(f'Manual COAL Annotations ({len(coal_annotations)} objects)', fontsize=14)
    axes[0, 1].axis('off')
    
    # Image with manual bounding boxes
    axes[1, 0].imshow(image)
    for i, annotation in enumerate(coal_annotations):
        # Show bounding box
        bbox = annotation['bbox']  # [x, y, width, height]
        box_coords = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        show_box(box_coords, axes[1, 0], color='red', linewidth=2)
    axes[1, 0].set_title(f'Manual COAL Bounding Boxes ({len(coal_annotations)} objects)', fontsize=14)
    axes[1, 0].axis('off')
    
    # Image with SAM automatic segmentation (no stars!)
    axes[1, 1].imshow(image)
    for i, mask_info in enumerate(filtered_masks):
        mask = mask_info['segmentation']
        # Use different colors for different objects
        show_mask(mask, axes[1, 1], random_color=True)
    axes[1, 1].set_title(f'SAM Automatic Segmentation ({len(filtered_masks)} objects)', fontsize=14)
    axes[1, 1].axis('off')
    
    # Save combined results
    image_name = Path(image_path).stem
    combined_output_path = os.path.join(output_dir, f"{image_name}_sam_automatic_vs_manual.png")
    plt.tight_layout()
    plt.savefig(combined_output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved comparison result to: {combined_output_path}")
    
    # Create combined mask file for SAM results
    if filtered_masks:
        combined_mask = np.zeros_like(filtered_masks[0]['segmentation'], dtype=np.uint8)
        for i, mask_info in enumerate(filtered_masks):
            mask = mask_info['segmentation']
            combined_mask[mask] = i + 1
        
        combined_mask_path = os.path.join(output_dir, f"{image_name}_sam_automatic_mask.png")
        cv2.imwrite(combined_mask_path, combined_mask)
        logger.info(f"Saved SAM automatic mask to: {combined_mask_path}")
    
    # Print summary
    logger.info(f"Manual annotations: {len(coal_annotations)} COAL objects")
    logger.info(f"SAM automatic detection: {len(filtered_masks)} objects")
    if filtered_masks:
        avg_stability = np.mean([m['stability_score'] for m in filtered_masks])
        logger.info(f"Average SAM stability score: {avg_stability:.3f}")
        logger.info(f"Stability range: {min(m['stability_score'] for m in filtered_masks):.3f} - {max(m['stability_score'] for m in filtered_masks):.3f}")

def main():
    parser = argparse.ArgumentParser(description="Run SAM for COAL segmentation on a single image")
    parser.add_argument("--image-id", type=int, default=1, help="Image ID to process")
    parser.add_argument("--annotations", default="sample/annotations/instances_default.json", 
                       help="Path to annotations JSON file")
    parser.add_argument("--images-dir", default="sample/images/default", 
                       help="Path to images directory")
    parser.add_argument("--output-dir", default="./sam_output", 
                       help="Output directory for results")
    parser.add_argument("--model-type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"],
                       help="SAM model type")
    parser.add_argument("--device", default="cuda", help="Device to run on (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Load annotations
    logger.info(f"Loading annotations from: {args.annotations}")
    with open(args.annotations, 'r') as f:
        annotations_data = json.load(f)
    
    # Find image filename
    image_filename = None
    for image_info in annotations_data['images']:
        if image_info['id'] == args.image_id:
            image_filename = image_info['file_name']
            break
    
    if not image_filename:
        logger.error(f"Image ID {args.image_id} not found in annotations")
        return
    
    image_path = os.path.join(args.images_dir, image_filename)
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        return
    
    logger.info(f"Processing image: {image_filename}")
    
    # Run SAM
    run_sam_on_image(image_path, annotations_data, args.image_id, args.output_dir, args.device)
    
    logger.info("SAM processing completed!")

if __name__ == "__main__":
    main()
