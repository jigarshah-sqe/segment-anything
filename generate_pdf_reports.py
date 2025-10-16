#!/usr/bin/env python3
"""
Generate PDF reports from SAM enhanced comparison results.
Creates 4 PDFs (one per bucket) with image tables and IoU summaries.
"""

import os
import json
import glob
from pathlib import Path
import argparse
import logging
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import matplotlib.pyplot as plt
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_bucket_summary(bucket_dir):
    """Get summary statistics for a bucket."""
    iou_files = glob.glob(os.path.join(bucket_dir, "*_iou_metrics.json"))
    
    if not iou_files:
        return {
            'total_images': 0,
            'mean_iou': 0.0,
            'max_iou': 0.0,
            'avg_manual_objects': 0.0,
            'avg_sam_objects': 0.0,
            'matched_manual': 0,
            'matched_sam': 0,
            'total_manual': 0,
            'total_sam': 0
        }
    
    all_ious = []
    all_manual_objects = []
    all_sam_objects = []
    total_matched_manual = 0
    total_matched_sam = 0
    total_manual = 0
    total_sam = 0
    
    for iou_file in iou_files:
        try:
            with open(iou_file, 'r') as f:
                data = json.load(f)
            
            if data['mean_iou'] > 0:  # Only include images with valid IoU
                all_ious.append(data['mean_iou'])
                all_manual_objects.append(data['total_manual'])
                all_sam_objects.append(data['total_sam'])
                total_matched_manual += data['matched_manual']
                total_matched_sam += data['matched_sam']
                total_manual += data['total_manual']
                total_sam += data['total_sam']
        except Exception as e:
            logger.warning(f"Error reading {iou_file}: {e}")
            continue
    
    return {
        'total_images': len(iou_files),
        'mean_iou': np.mean(all_ious) if all_ious else 0.0,
        'max_iou': np.max(all_ious) if all_ious else 0.0,
        'avg_manual_objects': np.mean(all_manual_objects) if all_manual_objects else 0.0,
        'avg_sam_objects': np.mean(all_sam_objects) if all_sam_objects else 0.0,
        'matched_manual': total_matched_manual,
        'matched_sam': total_matched_sam,
        'total_manual': total_manual,
        'total_sam': total_sam
    }

def create_summary_table(summary):
    """Create a summary table for the PDF."""
    data = [
        ['Metric', 'Value', 'Definition'],
        ['Total Images Processed', f"{summary['total_images']}", 'Number of images with valid IOU calculations'],
        ['Mean IOU', f"{summary['mean_iou']:.3f}", 'Average Intersection over Union across all images'],
        ['Max IOU', f"{summary['max_iou']:.3f}", 'Highest IOU value achieved in any single match'],
        ['Avg Manual Objects per Image', f"{summary['avg_manual_objects']:.1f}", 'Average number of manually annotated COAL objects per image'],
        ['Avg SAM Objects per Image', f"{summary['avg_sam_objects']:.1f}", 'Average number of SAM-detected objects per image'],
        ['Total Manual Objects', f"{summary['total_manual']}", 'Total count of manually annotated COAL objects across all images'],
        ['Total SAM Objects', f"{summary['total_sam']}", 'Total count of SAM-detected objects across all images'],
        ['Matched Manual Objects', f"{summary['matched_manual']}", 'Number of manual objects that have IOU > 0.5 with SAM objects'],
        ['Matched SAM Objects', f"{summary['matched_sam']}", 'Number of SAM objects that have IOU > 0.5 with manual objects'],
        ['Manual Match Rate', f"{(summary['matched_manual']/summary['total_manual']*100):.1f}%" if summary['total_manual'] > 0 else "0%", 'Percentage of manual objects with IOU ≥ 0.5 (recall metric)'],
        ['SAM Match Rate', f"{(summary['matched_sam']/summary['total_sam']*100):.1f}%" if summary['total_sam'] > 0 else "0%", 'Percentage of SAM objects with IOU ≥ 0.5 (precision metric)']
    ]
    
    table = Table(data, colWidths=[2.5*inch, 1.5*inch, 4*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),  # Metric column left-aligned
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),  # Value column centered
        ('ALIGN', (2, 0), (2, -1), 'LEFT'),  # Definition column left-aligned
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 15),
        ('TOPPADDING', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
    ]))
    
    return table

def create_image_sections(bucket_dir, bucket_name):
    """Create sections with images and IoU details."""
    iou_files = glob.glob(os.path.join(bucket_dir, "*_iou_metrics.json"))
    image_files = glob.glob(os.path.join(bucket_dir, "*_enhanced_comparison.png"))
    
    # Create mapping from image name to files
    image_data = {}
    
    for iou_file in iou_files:
        image_name = os.path.basename(iou_file).replace('_iou_metrics.json', '')
        image_data[image_name] = {'iou_file': iou_file}
    
    for image_file in image_files:
        image_name = os.path.basename(image_file).replace('_enhanced_comparison.png', '')
        if image_name in image_data:
            image_data[image_name]['image_file'] = image_file
    
    sections = []
    
    for image_name, files in sorted(image_data.items()):
        if 'iou_file' in files and 'image_file' in files:
            try:
                with open(files['iou_file'], 'r') as f:
                    iou_data = json.load(f)
                
                # Create image section
                section = []
                
                # Image title
                title_style = ParagraphStyle(
                    'ImageTitle',
                    parent=getSampleStyleSheet()['Heading3'],
                    fontSize=12,
                    spaceAfter=6,
                    alignment=TA_LEFT
                )
                section.append(Paragraph(f"Image: {image_name}", title_style))
                
                # Load and resize image (LARGER SIZE)
                try:
                    img = RLImage(files['image_file'], width=6*inch, height=4.5*inch)
                    section.append(img)
                except Exception as e:
                    logger.warning(f"Could not load image {files['image_file']}: {e}")
                    section.append(Paragraph(f"Image not available: {files['image_file']}", getSampleStyleSheet()['Normal']))
                
                # IOU metrics table
                metrics_data = [
                    ['Metric', 'Value'],
                    ['Mean IOU', f"{iou_data['mean_iou']:.3f}"],
                    ['Max IOU', f"{iou_data['max_iou']:.3f}"],
                    ['Manual Objects', f"{iou_data['total_manual']}"],
                    ['SAM Objects', f"{iou_data['total_sam']}"],
                    ['Matched Manual', f"{iou_data['matched_manual']}"],
                    ['Matched SAM', f"{iou_data['matched_sam']}"]
                ]
                
                metrics_table = Table(metrics_data, colWidths=[2.5*inch, 1.5*inch])
                metrics_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('FONTSIZE', (0, 1), (-1, -1), 11),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 15),
                    ('TOPPADDING', (0, 1), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 1), (-1, -1), 10),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
                ]))
                
                section.append(metrics_table)
                section.append(Spacer(1, 20))
                
                sections.extend(section)
                
            except Exception as e:
                logger.warning(f"Error processing {image_name}: {e}")
                continue
    
    return sections

def generate_bucket_pdf(bucket_dir, bucket_name, output_dir):
    """Generate PDF report for a single bucket."""
    logger.info(f"Generating PDF for bucket: {bucket_name}")
    
    # Get summary statistics
    summary = get_bucket_summary(bucket_dir)
    
    # Create PDF with larger page size
    pdf_path = os.path.join(output_dir, f"{bucket_name}_sam_report.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=letter, 
                          rightMargin=0.5*inch, leftMargin=0.5*inch,
                          topMargin=0.5*inch, bottomMargin=0.5*inch)
    styles = getSampleStyleSheet()
    
    # Create custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12,
        alignment=TA_LEFT
    )
    
    # Build content
    story = []
    
    # Title
    title = Paragraph(f"SAM Enhanced Comparison Report - {bucket_name}", title_style)
    story.append(title)
    story.append(Spacer(1, 20))
    
    # IOU Definitions section
    definitions_heading = Paragraph("IOU Metrics Definitions", heading_style)
    story.append(definitions_heading)
    story.append(Spacer(1, 12))
    
    # Create definitions content
    definitions_text = """
    <b>IOU (Intersection over Union):</b> Measures overlap quality between two objects.<br/>
    • IOU = 1.0 = Perfect match (100% overlap)<br/>
    • IOU = 0.5 = Good match (50% overlap)<br/>
    • IOU = 0.0 = No overlap<br/><br/>
    
    <b>Match Criteria:</b> Objects are considered "matched" if IOU ≥ 0.5<br/><br/>
    
    <b>Manual Match Rate (Recall):</b> How many manual objects SAM successfully found<br/>
    • High rate = SAM finds most of your manual annotations<br/>
    • Low rate = SAM misses many objects you can see<br/><br/>
    
    <b>SAM Match Rate (Precision):</b> How many SAM detections are valid<br/>
    • High rate = Most SAM detections are correct<br/>
    • Low rate = SAM has many false positives<br/><br/>
    
    <b>Mean IOU:</b> Average overlap quality when matches occur<br/>
    • High mean IOU = Accurate segmentation boundaries<br/>
    • Low mean IOU = Poor boundary alignment even when objects match
    """
    
    definitions_para = Paragraph(definitions_text, getSampleStyleSheet()['Normal'])
    story.append(definitions_para)
    story.append(Spacer(1, 20))
    
    # Summary section
    summary_heading = Paragraph("IOU Summary Statistics", heading_style)
    story.append(summary_heading)
    story.append(Spacer(1, 12))
    
    summary_table = create_summary_table(summary)
    story.append(summary_table)
    story.append(Spacer(1, 20))
    
    # Detailed results section
    details_heading = Paragraph("Detailed Results by Image", heading_style)
    story.append(details_heading)
    story.append(Spacer(1, 12))
    
    # Add image sections with actual images
    image_sections = create_image_sections(bucket_dir, bucket_name)
    story.extend(image_sections)
    
    # Build PDF
    doc.build(story)
    logger.info(f"Generated PDF: {pdf_path}")
    return pdf_path

def main():
    parser = argparse.ArgumentParser(description="Generate PDF reports from SAM results")
    parser.add_argument("--input-dir", default="./sam_enhanced_output", 
                       help="Input directory containing bucket folders")
    parser.add_argument("--output-dir", default="./pdf_reports", 
                       help="Output directory for PDF reports")
    parser.add_argument("--buckets", nargs='+', default=['15_18', '18_5', '5_9', '9_15'],
                       help="Buckets to process")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    generated_pdfs = []
    
    for bucket in args.buckets:
        bucket_dir = os.path.join(args.input_dir, bucket)
        
        if not os.path.exists(bucket_dir):
            logger.warning(f"Bucket directory not found: {bucket_dir}")
            continue
        
        try:
            pdf_path = generate_bucket_pdf(bucket_dir, bucket, args.output_dir)
            generated_pdfs.append(pdf_path)
        except Exception as e:
            logger.error(f"Error generating PDF for {bucket}: {e}")
            continue
    
    # Summary
    logger.info("="*60)
    logger.info("PDF GENERATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Generated {len(generated_pdfs)} PDF reports:")
    for pdf in generated_pdfs:
        logger.info(f"  - {pdf}")
    logger.info("="*60)

if __name__ == "__main__":
    main()
