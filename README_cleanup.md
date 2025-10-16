# Image Cleanup Script

This script (`cleanup_images.py`) helps clean up image files that have no meaningful annotations and removes their references from the JSON file.

## What It Does

The script identifies images that have:
- No annotations at all, OR
- Only empty annotations (empty segmentation arrays)

It then:
1. Deletes the image files from the filesystem
2. Removes their references from the JSON file
3. Updates the JSON file to maintain consistency

## Usage

### Basic Usage
```bash
python cleanup_images.py
```

### Dry Run (Recommended First)
```bash
python cleanup_images.py --dry-run
```

### Custom Paths
```bash
python cleanup_images.py --annotations path/to/annotations.json --images-dir path/to/images
```

### With Backup
```bash
python cleanup_images.py --backup
```

### Verbose Output
```bash
python cleanup_images.py --dry-run --verbose
```

## Options

- `--annotations`: Path to annotations JSON file (default: `sample/annotations/instances_default.json`)
- `--images-dir`: Path to images directory (default: `sample/images/default`)
- `--dry-run`: Show what would be deleted without actually deleting files
- `--verbose`, `-v`: Enable verbose logging
- `--backup`: Create a backup of the original JSON file before modifying

## Safety Features

1. **Dry Run Mode**: Always test with `--dry-run` first to see what would be deleted
2. **Confirmation Prompt**: The script asks for confirmation before deleting files
3. **Backup Option**: Create a backup of the JSON file before making changes
4. **Detailed Logging**: Shows exactly what files are being processed
5. **Error Handling**: Gracefully handles missing files or directories

## Example Output

```
2025-10-16 08:43:47,032 - INFO - Starting cleanup process...
2025-10-16 08:43:47,032 - INFO - Annotations file: /path/to/annotations.json
2025-10-16 08:43:47,032 - INFO - Images directory: /path/to/images
2025-10-16 08:43:47,032 - INFO - Dry run mode: True
2025-10-16 08:43:47,034 - INFO - Loaded annotations JSON with 299 images and 452 annotations
2025-10-16 08:43:47,034 - INFO - Total images: 299
2025-10-16 08:43:47,034 - INFO - Images with meaningful annotations: 28
2025-10-16 08:43:47,034 - INFO - Images without meaningful annotations: 271
2025-10-16 08:43:47,034 - INFO - Found 271 image files to delete
2025-10-16 08:43:47,034 - INFO - [DRY RUN] Would delete 271 files and remove 271 image references from JSON
```

## Supported Image Formats

The script recognizes these image file extensions:
- .jpg, .jpeg
- .png
- .bmp
- .tiff, .tif
- .gif

## How It Works

1. Loads the annotations JSON file and analyzes the annotations array
2. Identifies images that have no meaningful annotations (empty segmentation arrays)
3. Scans the images directory for corresponding image files
4. Deletes the image files and removes their references from the JSON
5. Saves the updated JSON file

## Requirements

- Python 3.6+
- No external dependencies (uses only standard library)
