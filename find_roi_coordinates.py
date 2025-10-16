#!/usr/bin/env python3
"""
Script to find the right ROI coordinates for conveyor area.
Shows image with adjustable yellow rectangle overlay.
"""

import os
import sys
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def show_roi_overlay(image_path, roi_coords, annotations_data, image_id):
    """Show image with ROI overlay and manual annotations."""
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    logger.info(f"Image shape: {image.shape}")
    
    # Get manual annotations
    coal_annotations = []
    for annotation in annotations_data['annotations']:
        if (annotation['image_id'] == image_id and 
            annotation['category_id'] == 1 and  # COAL category
            annotation['segmentation'] and 
            any(len(segment) > 0 for segment in annotation['segmentation'])):
            coal_annotations.append(annotation)
    
    logger.info(f"Found {len(coal_annotations)} manual COAL annotations")
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Image with ROI overlay
    axes[1].imshow(image)
    x1, y1, x2, y2 = roi_coords
    axes[1].add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   edgecolor='yellow', facecolor=(0,0,0,0), linewidth=4))
    axes[1].set_title(f'ROI Overlay\n({x1}, {y1}, {x2}, {y2})', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Image with manual annotations
    axes[2].imshow(image)
    for annotation in coal_annotations:
        if 'segmentation' in annotation and annotation['segmentation']:
            segmentation = annotation['segmentation'][0]
            if len(segmentation) >= 6:
                points = np.array(segmentation).reshape(-1, 2)
                from matplotlib.patches import Polygon
                polygon = Polygon(points, closed=True, facecolor='red', alpha=0.4, 
                                 edgecolor='red', linewidth=1)
                axes[2].add_patch(polygon)
    
    # Add ROI overlay to manual annotations view too
    axes[2].add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   edgecolor='yellow', facecolor=(0,0,0,0), linewidth=4))
    axes[2].set_title(f'Manual Annotations + ROI\n({len(coal_annotations)} objects)', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig

def find_roi_coordinates(image_path, annotations_data, image_id, initial_roi=(500, 600, 3500, 1700)):
    """Interactive ROI coordinate finder."""
    
    logger.info(f"Loading image: {image_path}")
    
    # Show initial ROI
    fig = show_roi_overlay(image_path, initial_roi, annotations_data, image_id)
    
    # Save initial visualization
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = f"{image_name}_roi_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved ROI analysis to: {output_path}")
    
    # Print current coordinates
    x1, y1, x2, y2 = initial_roi
    logger.info("="*60)
    logger.info("CURRENT ROI COORDINATES")
    logger.info("="*60)
    logger.info(f"ROI: ({x1}, {y1}, {x2}, {y2})")
    logger.info(f"Width: {x2-x1} pixels")
    logger.info(f"Height: {y2-y1} pixels")
    logger.info(f"Area: {(x2-x1)*(y2-y1):,} pixels")
    logger.info("="*60)
    
    # Interactive adjustment suggestions
    logger.info("ROI ADJUSTMENT SUGGESTIONS:")
    logger.info("- If ROI is too wide: decrease x2 or increase x1")
    logger.info("- If ROI is too tall: decrease y2 or increase y1") 
    logger.info("- If missing coal on left: decrease x1")
    logger.info("- If missing coal on right: increase x2")
    logger.info("- If missing coal on top: decrease y1")
    logger.info("- If missing coal on bottom: increase y2")
    logger.info("="*60)
    
    # Show the plot
    plt.show()
    
    return initial_roi

def main():
    parser = argparse.ArgumentParser(description="Find optimal ROI coordinates for conveyor area")
    parser.add_argument("--image-id", type=int, default=1, help="Image ID to process")
    parser.add_argument("--annotations", default="sample/annotations/instances_default.json", 
                       help="Path to annotations JSON file")
    parser.add_argument("--images-dir", default="sample/images/default", 
                       help="Path to images directory")
    parser.add_argument("--roi", nargs=4, type=int, default=[500, 600, 3500, 1700],
                       help="ROI coordinates: x1 y1 x2 y2")
    
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
    
    logger.info(f"Analyzing ROI for image: {image_filename}")
    
    # Find ROI coordinates
    roi_coords = find_roi_coordinates(image_path, annotations_data, args.image_id, tuple(args.roi))
    
    logger.info("ROI analysis completed!")
    logger.info(f"Final ROI coordinates: {roi_coords}")

if __name__ == "__main__":
    main()
