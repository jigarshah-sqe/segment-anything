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
# Removed parallel processing imports - using sequential execution

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

def polygon_to_mask(polygon_points, image_shape):
    """Convert polygon points to binary mask using OpenCV"""
    import cv2
    
    # Create a blank mask
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    
    # Convert polygon points to integer coordinates
    polygon_points = np.array(polygon_points, dtype=np.int32)
    
    # Fill the polygon with white (255)
    cv2.fillPoly(mask, [polygon_points], 255)
    
    # Convert to boolean mask
    return mask.astype(bool)

def calculate_iou(mask1, mask2):
    """Calculate Intersection over Union (IoU) between two binary masks"""
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    
    intersection_area = np.sum(intersection)
    union_area = np.sum(union)
    
    if union_area == 0:
        return 0.0
    
    iou = intersection_area / union_area
    return float(iou)

def calculate_iou_metrics(manual_annotations, sam_masks, image_shape, roi_crop=None):
    """Calculate IoU metrics between manual annotations and SAM predictions"""
    # Convert manual annotations to masks
    manual_masks = []
    for annotation in manual_annotations:
        if 'segmentation' in annotation and annotation['segmentation']:
            segmentation = annotation['segmentation'][0]
            if len(segmentation) >= 6:
                points = np.array(segmentation).reshape(-1, 2)
                mask = polygon_to_mask(points, image_shape)
                manual_masks.append(mask)
    
    if not manual_masks or not sam_masks:
        return {
            'best_iou_per_manual': [],
            'best_iou_per_sam': [],
            'mean_iou': 0.0,
            'max_iou': 0.0,
            'matched_manual': 0,
            'matched_sam': 0,
            'total_manual': len(manual_masks),
            'total_sam': len(sam_masks)
        }
    
    # Convert SAHI masks to full image coordinates if they are ROI masks
    full_sam_masks = []
    if roi_crop is not None:
        x1, y1, x2, y2 = roi_crop
        for roi_mask in sam_masks:
            # Create full image mask from ROI mask
            full_mask = np.zeros(image_shape[:2], dtype=bool)
            full_mask[y1:y2, x1:x2] = roi_mask
            full_sam_masks.append(full_mask)
    else:
        full_sam_masks = sam_masks
    
    # Calculate IoU between each manual annotation and each SAM prediction
    iou_matrix = np.zeros((len(manual_masks), len(full_sam_masks)))
    
    for i, manual_mask in enumerate(manual_masks):
        for j, sam_mask in enumerate(full_sam_masks):
            iou = calculate_iou(manual_mask, sam_mask)
            iou_matrix[i, j] = iou
    
    # Find best matches
    best_iou_per_manual = np.max(iou_matrix, axis=1)
    best_iou_per_sam = np.max(iou_matrix, axis=0)
    
    # Count matches (IoU > 0.5 threshold)
    matched_manual = np.sum(best_iou_per_manual > 0.5)
    matched_sam = np.sum(best_iou_per_sam > 0.5)
    
    # Calculate overall metrics
    mean_iou = np.mean(best_iou_per_manual) if len(best_iou_per_manual) > 0 else 0.0
    max_iou = np.max(iou_matrix) if iou_matrix.size > 0 else 0.0
    
    return {
        'best_iou_per_manual': best_iou_per_manual.tolist(),
        'best_iou_per_sam': best_iou_per_sam.tolist(),
        'mean_iou': float(mean_iou),
        'max_iou': float(max_iou),
        'matched_manual': int(matched_manual),
        'matched_sam': int(matched_sam),
        'total_manual': len(manual_masks),
        'total_sam': len(sam_masks)
    }

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

def run_sahi_sam_segmentation(image, sam_b, roi_crop=(1000, 20, 2986, 2118), device="cpu"):
    """Run SAHI-enhanced SAM segmentation with ROI and tiling.
    
    ROI coordinates: (x_min, y_min, x_max, y_max)
    - Reduced ROI to focus on coal on conveyor belt only
    - Excludes conveyor structure and empty areas
    """
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
    
    # Deduplicate masks with memory optimization
    final_masks = []
    max_masks = 100  # Limit total masks to prevent RAM overflow
    
    for mask in all_masks_in_roi:
        if len(final_masks) >= max_masks:
            logger.warning(f"Memory optimization: Limiting to {max_masks} masks to prevent RAM overflow")
            break
            
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

def run_enhanced_comparison(image_path, annotations_data, image_id, output_dir="./sam_enhanced_output", device="cpu", force=False):
    """Run enhanced comparison with 3 approaches in parallel."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if output files already exist (unless force is True)
    image_name = Path(image_path).stem
    output_path = os.path.join(output_dir, f"{image_name}_enhanced_comparison.png")
    metrics_output_path = os.path.join(output_dir, f"{image_name}_iou_metrics.json")
    
    if not force and os.path.exists(output_path) and os.path.exists(metrics_output_path):
        logger.info(f"Output files already exist for {image_name}, skipping...")
        logger.info(f"  - {output_path}")
        logger.info(f"  - {metrics_output_path}")
        return
    
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
    
    # Skip if no manual annotations (IoU calculation not meaningful)
    if len(coal_annotations) == 0:
        logger.warning(f"No manual COAL annotations found for image ID {image_id}, skipping...")
        return
    
    # Load SAM model (vit_b for both approaches)
    sam_b = load_sam_models(device=device)
    
    # Run 2 approaches sequentially (removed Full SAM)
    logger.info("Running 2 segmentation approaches sequentially...")
    
    # 1. Manual annotations (already have)
    manual_count = len(coal_annotations)
    
    # 2. SAHI-enhanced SAM segmentation only
    logger.info("Running SAHI-enhanced SAM segmentation...")
    roi_crop = (1000, 20, 2986, 2118)
    sahi_masks, roi_crop = run_sahi_sam_segmentation(image, sam_b, roi_crop, device)
    sahi_count = len(sahi_masks)
    
    # Create simplified 2x2 visualization
    logger.info("Creating 2-way comparison visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Original image with manual annotations
    axes[0, 1].imshow(image)
    show_manual_annotations(coal_annotations, axes[0, 1], color='red', alpha=0.4)
    axes[0, 1].set_title(f'Manual COAL Annotations\n({manual_count} objects)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Show ROI for SAHI
    axes[1, 0].imshow(image)
    x1, y1, x2, y2 = roi_crop
    axes[1, 0].add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                       edgecolor='yellow', facecolor=(0,0,0,0), linewidth=3))
    axes[1, 0].set_title('SAHI ROI Region\n(Yellow rectangle)', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # SAHI-enhanced SAM segmentation (map ROI masks to full image)
    axes[1, 1].imshow(image)
    x1, y1, x2, y2 = roi_crop
    
    # Memory optimization: Limit displayed masks to prevent RAM overflow
    max_display_masks = 50  # Limit to 50 masks for visualization
    display_masks = sahi_masks[:max_display_masks] if len(sahi_masks) > max_display_masks else sahi_masks
    
    for i, roi_mask in enumerate(display_masks):
        # Create full image mask from ROI mask
        full_mask = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
        full_mask[y1:y2, x1:x2] = roi_mask
        show_mask(full_mask, axes[1, 1], random_color=True, alpha=0.6)
    
    # Update title to show actual count vs displayed count
    if len(sahi_masks) > max_display_masks:
        axes[1, 1].set_title(f'SAHI-Enhanced SAM\n({sahi_count} total, showing {len(display_masks)})', fontsize=14, fontweight='bold')
    else:
        axes[1, 1].set_title(f'SAHI-Enhanced SAM\n({sahi_count} large coals)', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Save results directly to output directory (no subfolders)
    # image_name and output_path already defined at the beginning of function
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Memory cleanup
    del sahi_masks
    del display_masks
    import gc
    gc.collect()
    
    logger.info(f"Saved enhanced comparison to: {output_path}")
    
    # Calculate IoU metrics
    logger.info("Calculating IoU metrics...")
    iou_metrics = calculate_iou_metrics(coal_annotations, sahi_masks, image.shape, roi_crop)
    
    # Print summary
    logger.info("="*60)
    logger.info("SEGMENTATION COMPARISON SUMMARY")
    logger.info("="*60)
    logger.info(f"Manual annotations:     {manual_count:3d} COAL objects")
    logger.info(f"SAHI-enhanced SAM:      {sahi_count:3d} large coals")
    logger.info("="*60)
    logger.info("IoU METRICS")
    logger.info("="*60)
    logger.info(f"Mean IoU:               {iou_metrics['mean_iou']:.3f}")
    logger.info(f"Max IoU:                 {iou_metrics['max_iou']:.3f}")
    logger.info(f"Matched manual objects: {iou_metrics['matched_manual']}/{iou_metrics['total_manual']}")
    logger.info(f"Matched SAM objects:    {iou_metrics['matched_sam']}/{iou_metrics['total_sam']}")
    logger.info("="*60)
    
    # Save IoU metrics to JSON directly in output directory
    metrics_output_path = os.path.join(output_dir, f"{image_name}_iou_metrics.json")
    with open(metrics_output_path, 'w') as f:
        json.dump(iou_metrics, f, indent=2)
    logger.info(f"Saved IoU metrics to: {metrics_output_path}")
    
    # Clean up memory
    del sam_b
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def process_all_images(base_dir, output_base_dir, device="cpu", force=False):
    """Process all images across all 4 buckets."""
    
    buckets = ['15_18', '18_5', '5_9', '9_15']
    total_processed = 0
    total_failed = 0
    total_skipped = 0
    total_no_annotations = 0
    
    for bucket in buckets:
        logger.info(f"Processing bucket: {bucket}")
        
        # Paths for this bucket
        bucket_dir = os.path.join(base_dir, bucket)
        annotations_file = os.path.join(bucket_dir, 'annotations', 'instances_default.json')
        images_dir = os.path.join(bucket_dir, 'images', 'default')
        bucket_output_dir = os.path.join(output_base_dir, bucket)
        
        # Check if bucket exists
        if not os.path.exists(bucket_dir):
            logger.warning(f"Bucket {bucket} not found, skipping...")
            continue
            
        if not os.path.exists(annotations_file):
            logger.warning(f"Annotations file not found for {bucket}, skipping...")
            continue
            
        if not os.path.exists(images_dir):
            logger.warning(f"Images directory not found for {bucket}, skipping...")
            continue
        
        # Load annotations for this bucket
        logger.info(f"Loading annotations from: {annotations_file}")
        with open(annotations_file, 'r') as f:
            annotations_data = json.load(f)
        
        # Get all image files in this bucket
        image_files = []
        for file in os.listdir(images_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(images_dir, file))
        
        logger.info(f"Found {len(image_files)} images in bucket {bucket}")
        
        # Create mapping from filename to image_id
        filename_to_id = {}
        for image_info in annotations_data['images']:
            filename_to_id[image_info['file_name']] = image_info['id']
        
        bucket_processed = 0
        bucket_failed = 0
        bucket_skipped = 0
        bucket_no_annotations = 0
        
        logger.info(f"Starting processing of {len(image_files)} images in bucket {bucket}")
        
        for i, image_path in enumerate(image_files, 1):
            try:
                filename = os.path.basename(image_path)
                
                # Find corresponding image_id
                image_id = filename_to_id.get(filename)
                if image_id is None:
                    logger.warning(f"No annotations found for {filename}, skipping...")
                    bucket_no_annotations += 1
                    continue
                
                logger.info(f"Processing {bucket} image {i}/{len(image_files)}: {filename}")
                logger.info(f"Image ID: {image_id}")
                
                # Check if we should skip due to existing files
                image_name = Path(image_path).stem
                output_path = os.path.join(bucket_output_dir, f"{image_name}_enhanced_comparison.png")
                metrics_output_path = os.path.join(bucket_output_dir, f"{image_name}_iou_metrics.json")
                
                if not force and os.path.exists(output_path) and os.path.exists(metrics_output_path):
                    logger.info(f"⏭️  Skipping {filename} - output files already exist")
                    bucket_skipped += 1
                    continue
                
                # Run enhanced comparison - save directly to bucket output dir
                run_enhanced_comparison(image_path, annotations_data, image_id, bucket_output_dir, device, force=force)
                
                bucket_processed += 1
                logger.info(f"✅ Successfully processed: {filename}")
                
            except Exception as e:
                bucket_failed += 1
                logger.error(f"❌ Failed to process {image_path}: {e}")
                continue
        
        logger.info(f"Bucket {bucket} summary: {bucket_processed} processed, {bucket_skipped} skipped, {bucket_no_annotations} no annotations, {bucket_failed} failed")
        total_processed += bucket_processed
        total_failed += bucket_failed
        total_skipped += bucket_skipped
        total_no_annotations += bucket_no_annotations
    
    logger.info("="*60)
    logger.info("BATCH PROCESSING SUMMARY")
    logger.info("="*60)
    logger.info(f"Successfully processed: {total_processed}")
    logger.info(f"Skipped (files exist):   {total_skipped}")
    logger.info(f"No annotations:         {total_no_annotations}")
    logger.info(f"Failed:                 {total_failed}")
    logger.info(f"Total images:            {total_processed + total_skipped + total_no_annotations + total_failed}")
    logger.info("="*60)

def main():
    parser = argparse.ArgumentParser(description="Enhanced SAM comparison for COAL segmentation")
    parser.add_argument("--image-id", type=int, default=None, help="Single image ID to process (optional)")
    parser.add_argument("--bucket", default="15_18", choices=['15_18', '18_5', '5_9', '9_15'],
                       help="Bucket to process for single image")
    parser.add_argument("--base-dir", default="sample", 
                       help="Base directory containing bucket folders")
    parser.add_argument("--output-dir", default="./sam_enhanced_output", 
                       help="Output directory for results")
    parser.add_argument("--device", default="cpu", help="Device to run on (cuda/cpu)")
    parser.add_argument("--all-images", action="store_true", 
                       help="Process all images across all buckets")
    parser.add_argument("--force", action="store_true", 
                       help="Force re-processing even if output files exist")
    
    args = parser.parse_args()
    
    if args.all_images:
        # Process all images across all buckets
        logger.info("Processing all images across all buckets...")
        process_all_images(args.base_dir, args.output_dir, args.device, args.force)
    else:
        # Process single image from specified bucket
        if args.image_id is None:
            args.image_id = 1
        
        # Paths for the specified bucket
        bucket_dir = os.path.join(args.base_dir, args.bucket)
        annotations_file = os.path.join(bucket_dir, 'annotations', 'instances_default.json')
        images_dir = os.path.join(bucket_dir, 'images', 'default')
        bucket_output_dir = os.path.join(args.output_dir, args.bucket)
        
        # Check if bucket exists
        if not os.path.exists(bucket_dir):
            logger.error(f"Bucket {args.bucket} not found at {bucket_dir}")
            return
            
        if not os.path.exists(annotations_file):
            logger.error(f"Annotations file not found: {annotations_file}")
            return
            
        if not os.path.exists(images_dir):
            logger.error(f"Images directory not found: {images_dir}")
            return
        
        # Load annotations
        logger.info(f"Loading annotations from: {annotations_file}")
        with open(annotations_file, 'r') as f:
            annotations_data = json.load(f)
        
        # Find image filename
        image_filename = None
        for image_info in annotations_data['images']:
            if image_info['id'] == args.image_id:
                image_filename = image_info['file_name']
                break
        
        if not image_filename:
            logger.error(f"Image ID {args.image_id} not found in annotations for bucket {args.bucket}")
            return
        
        image_path = os.path.join(images_dir, image_filename)
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return
        
        logger.info(f"Processing single image from bucket {args.bucket}: {image_filename}")
        
        # Run enhanced comparison - save directly to bucket output dir
        run_enhanced_comparison(image_path, annotations_data, args.image_id, bucket_output_dir, args.device, force=args.force)
        
        logger.info("Enhanced comparison completed!")

if __name__ == "__main__":
    main()
