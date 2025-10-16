#!/usr/bin/env python3
"""
Cleanup script for segment-anything sample images.

This script reads the annotations JSON file and deletes all image files
that have no annotations, and removes their references from the JSON.
"""

import json
import os
import sys
from pathlib import Path
from typing import Set, List, Dict, Any
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_annotations_data(json_path: str) -> Dict[str, Any]:
    """
    Load the annotations JSON file and return the full data structure.
    
    Args:
        json_path: Path to the annotations JSON file
        
    Returns:
        Dictionary containing the full JSON data
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Loaded annotations JSON with {len(data.get('images', []))} images and {len(data.get('annotations', []))} annotations")
        return data
        
    except FileNotFoundError:
        logger.error(f"Annotations file not found: {json_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in annotations file: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading annotations: {e}")
        sys.exit(1)


def find_images_without_annotations(data: Dict[str, Any]) -> Set[int]:
    """
    Find image IDs that have no meaningful annotations.
    Considers empty annotations (empty segmentation arrays) as no annotations.
    
    Args:
        data: The loaded JSON data
        
    Returns:
        Set of image IDs that have no meaningful annotations
    """
    # Get all image IDs
    all_image_ids = set()
    if 'images' in data:
        for image_info in data['images']:
            if 'id' in image_info:
                all_image_ids.add(image_info['id'])
    
    # Get image IDs that have meaningful annotations (non-empty segmentation)
    images_with_meaningful_annotations = set()
    if 'annotations' in data:
        for annotation in data['annotations']:
            if 'image_id' in annotation and 'segmentation' in annotation:
                # Check if segmentation is not empty
                segmentation = annotation['segmentation']
                if segmentation and len(segmentation) > 0:
                    # Check if any segment has actual coordinates
                    has_coordinates = any(
                        len(segment) > 0 and segment != [] 
                        for segment in segmentation
                    )
                    if has_coordinates:
                        images_with_meaningful_annotations.add(annotation['image_id'])
    
    # Find images without meaningful annotations
    images_without_annotations = all_image_ids - images_with_meaningful_annotations
    
    logger.info(f"Total images: {len(all_image_ids)}")
    logger.info(f"Images with meaningful annotations: {len(images_with_meaningful_annotations)}")
    logger.info(f"Images without meaningful annotations: {len(images_without_annotations)}")
    
    return images_without_annotations


def get_image_files(directory: str) -> Set[str]:
    """
    Get all image files in the specified directory.
    
    Args:
        directory: Path to the directory containing images
        
    Returns:
        Set of image filenames in the directory
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif'}
    image_files = set()
    
    try:
        for file_path in Path(directory).iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                image_files.add(file_path.name)
        
        logger.info(f"Found {len(image_files)} image files in directory")
        return image_files
        
    except FileNotFoundError:
        logger.error(f"Directory not found: {directory}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error reading directory: {e}")
        sys.exit(1)


def get_filenames_to_delete(data: Dict[str, Any], images_without_annotations: Set[int]) -> Set[str]:
    """
    Get the filenames of images that should be deleted.
    
    Args:
        data: The loaded JSON data
        images_without_annotations: Set of image IDs without annotations
        
    Returns:
        Set of filenames to delete
    """
    filenames_to_delete = set()
    
    if 'images' in data:
        for image_info in data['images']:
            if 'id' in image_info and 'file_name' in image_info:
                if image_info['id'] in images_without_annotations:
                    filenames_to_delete.add(image_info['file_name'])
    
    logger.info(f"Found {len(filenames_to_delete)} image files to delete")
    return filenames_to_delete


def delete_image_files(directory: str, filenames_to_delete: Set[str], dry_run: bool = False) -> int:
    """
    Delete image files that have no annotations.
    
    Args:
        directory: Path to the directory containing images
        filenames_to_delete: Set of filenames to delete
        dry_run: If True, only show what would be deleted without actually deleting
        
    Returns:
        Number of files deleted (or would be deleted in dry run)
    """
    deleted_count = 0
    
    for filename in filenames_to_delete:
        file_path = Path(directory) / filename
        
        if file_path.exists():
            if dry_run:
                logger.info(f"[DRY RUN] Would delete: {filename}")
            else:
                try:
                    file_path.unlink()
                    logger.info(f"Deleted: {filename}")
                except Exception as e:
                    logger.error(f"Failed to delete {filename}: {e}")
                    continue
            deleted_count += 1
        else:
            logger.warning(f"File not found: {filename}")
    
    return deleted_count


def remove_image_references_from_json(data: Dict[str, Any], images_without_annotations: Set[int], dry_run: bool = False) -> Dict[str, Any]:
    """
    Remove image references from the JSON data for images without annotations.
    
    Args:
        data: The loaded JSON data
        images_without_annotations: Set of image IDs without annotations
        dry_run: If True, only show what would be removed without actually removing
        
    Returns:
        Updated JSON data with image references removed
    """
    if dry_run:
        logger.info(f"[DRY RUN] Would remove {len(images_without_annotations)} image references from JSON")
        return data
    
    # Create a copy of the data to modify
    updated_data = data.copy()
    
    # Remove images without annotations from the images array
    if 'images' in updated_data:
        updated_data['images'] = [
            image_info for image_info in updated_data['images']
            if image_info.get('id') not in images_without_annotations
        ]
        logger.info(f"Removed {len(images_without_annotations)} image references from JSON")
    
    return updated_data


def save_updated_json(data: Dict[str, Any], json_path: str, dry_run: bool = False) -> None:
    """
    Save the updated JSON data to file.
    
    Args:
        data: The updated JSON data
        json_path: Path to save the JSON file
        dry_run: If True, only show what would be saved without actually saving
    """
    if dry_run:
        logger.info(f"[DRY RUN] Would save updated JSON to: {json_path}")
        return
    
    try:
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved updated JSON to: {json_path}")
    except Exception as e:
        logger.error(f"Failed to save updated JSON: {e}")
        sys.exit(1)


def main():
    """Main function to orchestrate the cleanup process."""
    parser = argparse.ArgumentParser(
        description="Clean up image files that have no annotations and remove their references from JSON"
    )
    parser.add_argument(
        "--annotations", 
        default="sample/annotations/instances_default.json",
        help="Path to annotations JSON file (default: sample/annotations/instances_default.json)"
    )
    parser.add_argument(
        "--images-dir",
        default="sample/images/default",
        help="Path to images directory (default: sample/images/default)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting files"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create a backup of the original JSON file before modifying"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Convert to absolute paths
    annotations_path = os.path.abspath(args.annotations)
    images_dir = os.path.abspath(args.images_dir)
    
    logger.info(f"Starting cleanup process...")
    logger.info(f"Annotations file: {annotations_path}")
    logger.info(f"Images directory: {images_dir}")
    logger.info(f"Dry run mode: {args.dry_run}")
    
    # Load annotations data
    data = load_annotations_data(annotations_path)
    
    # Find images without annotations
    images_without_annotations = find_images_without_annotations(data)
    
    if not images_without_annotations:
        logger.info("All images have annotations. Nothing to clean up.")
        return
    
    # Get filenames to delete
    filenames_to_delete = get_filenames_to_delete(data, images_without_annotations)
    
    # Show some examples of files to delete
    logger.info("Examples of files to delete:")
    for i, filename in enumerate(sorted(filenames_to_delete)[:10]):
        logger.info(f"  - {filename}")
    if len(filenames_to_delete) > 10:
        logger.info(f"  ... and {len(filenames_to_delete) - 10} more files")
    
    # Confirm deletion (unless dry run)
    if not args.dry_run:
        response = input(f"\nAre you sure you want to delete {len(filenames_to_delete)} image files and remove their references from JSON? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            logger.info("Cleanup cancelled by user.")
            return
    
    # Create backup if requested
    if args.backup and not args.dry_run:
        backup_path = annotations_path + '.backup'
        try:
            with open(annotations_path, 'r') as src, open(backup_path, 'w') as dst:
                dst.write(src.read())
            logger.info(f"Created backup: {backup_path}")
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            sys.exit(1)
    
    # Delete image files
    deleted_count = delete_image_files(images_dir, filenames_to_delete, args.dry_run)
    
    # Remove image references from JSON
    updated_data = remove_image_references_from_json(data, images_without_annotations, args.dry_run)
    
    # Save updated JSON
    save_updated_json(updated_data, annotations_path, args.dry_run)
    
    if args.dry_run:
        logger.info(f"[DRY RUN] Would delete {deleted_count} files and remove {len(images_without_annotations)} image references from JSON")
    else:
        logger.info(f"Successfully deleted {deleted_count} image files")
        logger.info(f"Removed {len(images_without_annotations)} image references from JSON")
        logger.info(f"Remaining images: {len(data.get('images', [])) - len(images_without_annotations)}")


if __name__ == "__main__":
    main()
