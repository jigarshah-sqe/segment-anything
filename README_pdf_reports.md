# PDF Report Generation for SAM Enhanced Comparison

This script generates comprehensive PDF reports from SAM enhanced comparison results, creating one PDF per bucket with detailed IOU statistics and image summaries.

## Features

### 📊 **Enhanced Summary Statistics with Definitions**
- Total images processed
- Mean and maximum IOU values with explanations
- Average manual and SAM objects per image
- Match rates for manual and SAM objects
- **Detailed definitions** for each metric in 3-column table
- **Larger fonts** (12pt headers, 10pt content) for better readability
- **Enhanced spacing** with proper padding for visibility

### 🖼️ **Large Images with Enhanced Metrics**
- **LARGE comparison images** (6" x 4.5" display) for maximum visibility
- Individual IOU metrics for each image
- **Enhanced metrics tables** with larger fonts (12pt headers, 11pt content)
- Manual vs SAM object counts per image
- Match statistics per image
- **Better table spacing** with increased padding

### 📁 **Bucket Organization**
- Separate PDF for each bucket (15_18, 18_5, 5_9, 9_15)
- Organized by processing results

## Usage

### Basic Usage
```bash
# Generate PDFs for all buckets
python generate_pdf_reports.py

# Specify custom input/output directories
python generate_pdf_reports.py --input-dir ./sam_enhanced_output --output-dir ./pdf_reports

# Process specific buckets
python generate_pdf_reports.py --buckets 15_18 18_5
```

### Command Line Arguments
- `--input-dir`: Directory containing bucket folders (default: `./sam_enhanced_output`)
- `--output-dir`: Output directory for PDF reports (default: `./pdf_reports`)
- `--buckets`: Specific buckets to process (default: all 4 buckets)

## Output Structure

```
pdf_reports/
├── 15_18_sam_report.pdf
├── 18_5_sam_report.pdf
├── 5_9_sam_report.pdf
└── 9_15_sam_report.pdf
```

## PDF Contents

Each PDF contains:

### 1. **IOU Metrics Definitions Section**
- **Comprehensive explanations** of IOU, match criteria, and metrics
- **Visual examples** of IOU values (1.0, 0.5, 0.0)
- **Match criteria explanation** (IOU ≥ 0.5 threshold)
- **Recall vs Precision** definitions for match rates
- **Mean IOU interpretation** for segmentation quality

### 2. **IOU Summary Statistics Table with Definitions**
- Total Images Processed (with definition)
- Mean IOU (with explanation)
- Max IOU (with explanation)
- Average Manual Objects per Image (with definition)
- Average SAM Objects per Image (with definition)
- Total Manual/SAM Objects (with explanation)
- Matched Objects Counts (with definition)
- Match Rates % (with explanation)

### 3. **Large Images with Enhanced Metrics**
- **LARGE comparison images** (6" x 4.5" display) for better visibility
- Image name and title
- **Enhanced IOU metrics table** with larger fonts and better spacing
- Manual Objects Count
- SAM Objects Count
- Matched Objects per image

## Requirements

Install dependencies:
```bash
pip install -r requirements_pdf.txt
```

Required packages:
- `reportlab>=4.0.0` - PDF generation
- `matplotlib>=3.3.0` - Image processing
- `numpy>=1.20.0` - Statistical calculations

## Input Requirements

The script expects the following input structure:
```
sam_enhanced_output/
├── 15_18/
│   ├── image1_enhanced_comparison.png
│   ├── image1_iou_metrics.json
│   ├── image2_enhanced_comparison.png
│   └── image2_iou_metrics.json
├── 18_5/
├── 5_9/
└── 9_15/
```

## Error Handling

- Skips buckets with missing directories
- Handles corrupted JSON files gracefully
- Logs warnings for problematic files
- Continues processing other buckets if one fails

## Example Output

```
2025-10-16 15:16:04,287 - INFO - Generating PDF for bucket: 15_18
2025-10-16 15:16:04,301 - INFO - Generated PDF: ./pdf_reports/15_18_sam_report.pdf
2025-10-16 15:16:04,301 - INFO - Generating PDF for bucket: 18_5
2025-10-16 15:16:04,306 - INFO - Generated PDF: ./pdf_reports/18_5_sam_report.pdf
============================================================
PDF GENERATION SUMMARY
============================================================
Generated 2 PDF reports:
  - ./pdf_reports/15_18_sam_report.pdf
  - ./pdf_reports/18_5_sam_report.pdf
============================================================
```
