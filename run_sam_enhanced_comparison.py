#!/usr/bin/env python3
"""
Enhanced SAM script with SAHI improvements for COAL segmentation comparison.
Shows 3 approaches: Manual annotations, Full SAM, SAHI-enhanced SAM
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
import gc
from datetime import datetime
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add the segment_anything module to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'segment_anything'))

from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_sam_checkpoint(model_type="vit_b", checkpoint_dir="./checkpoints"):
    """Download SAM checkpoint if not already present."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if model_type == "vit_b":
        checkpoint_name = "sam_vit_b_01ec64.pth"
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    elif model_type == "vit_h":
        checkpoint_name = "sam_vit_h_4b8939.pth"
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    elif model_type == "vit_l":
        checkpoint_name = "sam_vit_l_0b3195.pth"
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
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

def load_sam_models(device="cpu"):
    """Load SAM vit_b model for both approaches."""
    # Load vit_b for both approaches (faster processing)
    checkpoint_b = download_sam_checkpoint("vit_b")
    sam_b = sam_model_registry["vit_b"](checkpoint=checkpoint_b)
    sam_b.to(device=device)
    
    logger.info(f"SAM vit_b model loaded on device: {device}")
    return sam_b

def show_mask(mask, ax, random_color=False, alpha=0.6):
    """Display mask on axis."""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([alpha])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, alpha])  # Blue color
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

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
            segmentation = annotation['segmentation'][0]
            if len(segmentation) >= 6:
                points = np.array(segmentation).reshape(-1, 2)
                from matplotlib.patches import Polygon
                polygon = Polygon(points, closed=True, facecolor=color, alpha=alpha, 
                                 edgecolor=color, linewidth=1)
                ax.add_patch(polygon)

def is_coal_like_region(image_rgb, mask, min_darkness=0.1, max_darkness=0.7):
    """Check if masked region is dark (coal-like)"""
    masked_pixels = image_rgb[mask]
    if len(masked_pixels) == 0:
        return False
    avg_brightness = np.mean(masked_pixels) / 255.0
    return min_darkness <= avg_brightness <= max_darkness

def clean_mask(mask):
    """Apply morphological operations to smooth mask"""
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask.astype(bool)

def calculate_maximum_diameter_from_points(points):
    """Calculate maximum diameter from polygon points"""
    if len(points) < 2:
        return 0.0
    points = np.array(points, dtype=float)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    dists = np.linalg.norm(points[:, None] - points, axis=-1)
    return float(np.max(dists))

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

def run_full_sam_segmentation(image, sam_h, device="cpu"):
    """Run full SAM segmentation on entire image (our current approach)."""
    logger.info("Running full SAM segmentation...")
    
    mask_generator = SamAutomaticMaskGenerator(
        model=sam_h,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
    )
    
    masks = mask_generator.generate(image)
    logger.info(f"Full SAM found {len(masks)} segments")
    
    # Filter by area (similar to our current approach)
    filtered_masks = []
    for mask_info in masks:
        area = mask_info['area']
        if 1000 < area < 50000:
            if mask_info['stability_score'] > 0.8:
                filtered_masks.append(mask_info)
    
    logger.info(f"Filtered to {len(filtered_masks)} high-quality segments")
    return filtered_masks

def run_full_sam_worker(args):
    """Worker function for full SAM segmentation."""
    image, sam_h, device = args
    try:
        return run_full_sam_segmentation(image, sam_h, device)
    except Exception as e:
        logger.error(f"Full SAM worker failed: {e}")
        return []

def run_sahi_sam_worker(args):
    """Worker function for SAHI SAM segmentation."""
    image, sam_b, roi_crop, device = args
    try:
        return run_sahi_sam_segmentation(image, sam_b, roi_crop, device)
    except Exception as e:
        logger.error(f"SAHI SAM worker failed: {e}")
        return [], roi_crop

def run_sahi_sam_segmentation(image, sam_b, roi_crop=(500, 600, 3500, 1700), device="cpu"):
    """Run SAHI-enhanced SAM segmentation with ROI and tiling.
    
    ROI coordinates: (x_min, y_min, x_max, y_max)
    - Reduced ROI to focus on coal on conveyor belt only
    - Excludes conveyor structure and empty areas
    """
    logger.info("Running SAHI-enhanced SAM segmentation...")
    
    # Extract ROI
    x1_roi, y1_roi, x2_roi, y2_roi = roi_crop
    roi_img = image[y1_roi:y2_roi, x1_roi:x2_roi]
    H_roi, W_roi = roi_img.shape[:2]
    
    # SAHI parameters
    mask_generator = SamAutomaticMaskGenerator(
        model=sam_b,
        points_per_side=16,                    # Lower density for speed
        pred_iou_thresh=0.75,                  # Lower threshold
        stability_score_thresh=0.90,           # Higher threshold
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
    )
    
    # Tiling parameters
    slice_height = 2024
    slice_width = 2024
    overlap_ratio = 0.3
    stride_y = int(slice_height * (1 - overlap_ratio))
    stride_x = int(slice_width * (1 - overlap_ratio))
    
    all_masks_in_roi = []
    
    # Process tiles
    for y in range(0, H_roi, stride_y):
        for x in range(0, W_roi, stride_x):
            y2 = min(y + slice_height, H_roi)
            x2 = min(x + slice_width, W_roi)
            tile = roi_img[y:y2, x:x2]
            if tile.size == 0:
                continue
            
            try:
                masks = mask_generator.generate(tile)
            except Exception as e:
                continue
            
            for m in masks:
                seg_tile = m["segmentation"].astype(bool)
                area = int(seg_tile.sum())
                
                # SAHI filtering: large coal pieces only
                if area < 5000 or area > 200000:
                    continue
                
                # Coal-like filtering
                if not is_coal_like_region(tile, seg_tile, min_darkness=0.1, max_darkness=0.7):
                    continue
                
                # Map back to full ROI coordinates
                full_roi_mask = np.zeros((H_roi, W_roi), dtype=bool)
                full_roi_mask[y:y2, x:x2] = seg_tile
                full_roi_mask = clean_mask(full_roi_mask)
                
                area_clean = int(full_roi_mask.sum())
                if area_clean < 5000 or area_clean > 200000:
                    continue
                
                all_masks_in_roi.append(full_roi_mask)
    
    # Deduplicate masks
    final_masks = []
    for mask in all_masks_in_roi:
        duplicate = False
        for fm in final_masks:
            inter = np.logical_and(mask, fm)
            union = np.logical_or(mask, fm)
            iou = np.sum(inter) / (np.sum(union) + 1e-6)
            if iou > 0.5:
                duplicate = True
                break
        if not duplicate:
            final_masks.append(mask)
    
    logger.info(f"SAHI SAM found {len(final_masks)} large coal segments")
    return final_masks, roi_crop

def run_enhanced_comparison(image_path, annotations_data, image_id, output_dir="./sam_enhanced_output", device="cpu"):
    """Run enhanced comparison with 3 approaches in parallel."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    logger.info(f"Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    logger.info(f"Image shape: {image.shape}")
    
    # Get manual annotations
    coal_annotations = get_coal_annotations_for_image(annotations_data, image_id)
    logger.info(f"Found {len(coal_annotations)} manual COAL annotations")
    
    # Load SAM model (vit_b for both approaches)
    sam_b = load_sam_models(device=device)
    
    # Run all 3 approaches in parallel
    logger.info("Running all 3 segmentation approaches in parallel...")
    
    # 1. Manual annotations (already have)
    manual_count = len(coal_annotations)
    
    # 2. & 3. Run SAM approaches in parallel using ThreadPoolExecutor
    roi_crop = (500, 600, 3500, 1700)
    
    import time
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submit both SAM tasks
        future_full_sam = executor.submit(run_full_sam_worker, (image, sam_b, device))
        future_sahi_sam = executor.submit(run_sahi_sam_worker, (image, sam_b, roi_crop, device))
        
        # Wait for both to complete
        logger.info("Waiting for parallel SAM processing to complete...")
        full_sam_masks = future_full_sam.result()
        sahi_result = future_sahi_sam.result()
        
        if isinstance(sahi_result, tuple):
            sahi_masks, roi_crop = sahi_result
        else:
            sahi_masks = sahi_result
            roi_crop = (500, 600, 3500, 1700)
    
    parallel_time = time.time() - start_time
    logger.info(f"Parallel SAM processing completed in {parallel_time:.2f} seconds")
    
    full_sam_count = len(full_sam_masks)
    sahi_count = len(sahi_masks)
    
    # Create comprehensive visualization
    logger.info("Creating 3-way comparison visualization...")
    
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    
    # Row 1: Original and Manual
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(image)
    show_manual_annotations(coal_annotations, axes[0, 1], color='red', alpha=0.4)
    axes[0, 1].set_title(f'Manual COAL Annotations\n({manual_count} objects)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Show ROI for SAHI
    axes[0, 2].imshow(image)
    x1, y1, x2, y2 = roi_crop
    axes[0, 2].add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                       edgecolor='yellow', facecolor=(0,0,0,0), linewidth=3))
    axes[0, 2].set_title('SAHI ROI Region\n(Yellow rectangle)', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Row 2: SAM Results
    axes[1, 0].imshow(image)
    for i, mask_info in enumerate(full_sam_masks):
        mask = mask_info['segmentation']
        show_mask(mask, axes[1, 0], random_color=True, alpha=0.6)
    axes[1, 0].set_title(f'Full SAM Segmentation\n({full_sam_count} objects)', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(image)
    for i, mask in enumerate(sahi_masks):
        show_mask(mask, axes[1, 1], random_color=True, alpha=0.6)
    axes[1, 1].set_title(f'SAHI-Enhanced SAM\n({sahi_count} large coals)', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Combined comparison
    axes[1, 2].imshow(image)
    # Overlay all results with different colors
    for i, mask_info in enumerate(full_sam_masks[:10]):  # Limit to first 10 for visibility
        mask = mask_info['segmentation']
        show_mask(mask, axes[1, 2], random_color=True, alpha=0.3)
    axes[1, 2].set_title(f'Combined View\n(All detections)', fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')
    
    # Save results
    image_name = Path(image_path).stem
    output_path = os.path.join(output_dir, f"{image_name}_enhanced_comparison.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved enhanced comparison to: {output_path}")
    
    # Print summary
    logger.info("="*60)
    logger.info("SEGMENTATION COMPARISON SUMMARY")
    logger.info("="*60)
    logger.info(f"Manual annotations:     {manual_count:3d} COAL objects")
    logger.info(f"Full SAM detection:     {full_sam_count:3d} objects")
    logger.info(f"SAHI-enhanced SAM:      {sahi_count:3d} large coals")
    logger.info("="*60)
    
    # Clean up memory
    del sam_b
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description="Enhanced SAM comparison for COAL segmentation")
    parser.add_argument("--image-id", type=int, default=1, help="Image ID to process")
    parser.add_argument("--annotations", default="sample/annotations/instances_default.json", 
                       help="Path to annotations JSON file")
    parser.add_argument("--images-dir", default="sample/images/default", 
                       help="Path to images directory")
    parser.add_argument("--output-dir", default="./sam_enhanced_output", 
                       help="Output directory for results")
    parser.add_argument("--device", default="cpu", help="Device to run on (cuda/cpu)")
    
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
    
    # Run enhanced comparison
    run_enhanced_comparison(image_path, annotations_data, args.image_id, args.output_dir, args.device)
    
    logger.info("Enhanced comparison completed!")

if __name__ == "__main__":
    main()
