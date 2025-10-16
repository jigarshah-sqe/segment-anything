#%% md
# # 🪨 Coal Segmentation with SAM + SAHI (Large Coals Only)
# 
# This notebook:
# - Processes all `.jpg` in `frames/` folder structure
# - Runs **SAM AutomaticMaskGenerator** on **ROI only** (`500,0,3500,2160`)
# - Detects **large coal pieces** (5k–200k px²)
# - Filters by **darkness** (coal-like)
# - Cleans masks with **morphology**
# - Outputs JSON with:
#   - `segmentation` (polygon)
#   - `maximum_diameters` (px)
#   - `date_start`/`date_end` in `"YYYY-MM-DD HH:MM:SS"` format
# - **RAM-safe**: tiles large images, no full-mask storage
# 
# **Requirements**:
# - `sam_vit_h_4b8939.pth` in working directory
# - Folder structure: `frames/<date>/D2_<start>_<end>/xxx.jpg`
#%%
# Install dependencies (run once)
# !pip install opencv-python torch torchvision numpy matplotlib shapely segment-anything
# !wget -nc https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
#%%
import os
import json
import cv2
import numpy as np
import torch
import gc
from datetime import datetime
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")
#%%
# ----------------------------
# CONFIGURATION
# ----------------------------
ROOT_DIR = "./frames"
MODEL_CHECKPOINT = "./models/sam_vit_b_01ec64.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Define ROI: (x1, y1, x2, y2) within 4096x2160 image
ROI_CROP = (500, 0, 3500, 2160)  # Adjust if needed

# Visualization control
VISUALIZE_FIRST_N = 2  # Set to 0 to disable all visualization
#%%
# ----------------------------
# HELPER FUNCTIONS
# ----------------------------

# Global SAM model (load once)
_sam_model = None

def load_sam_model():
    global _sam_model
    if _sam_model is None:
        print(f"Loading SAM model on {DEVICE}...")
        _sam_model = sam_model_registry["vit_b"](checkpoint=MODEL_CHECKPOINT)
        _sam_model.to(device=DEVICE)
    return _sam_model

def is_coal_like_region(image_rgb, mask, min_darkness=0.1, max_darkness=0.7):
    """Check if masked region is dark (coal-like)"""
    masked_pixels = image_rgb[mask]
    if len(masked_pixels) == 0:
        return False
    avg_brightness = np.mean(masked_pixels) / 255.0
    return min_darkness <= avg_brightness <= max_darkness

def get_bbox_from_bool_mask(mask):
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return [0, 0, 0, 0]
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return [int(x_min), int(y_min), int(x_max), int(y_max)]

def clean_mask(mask):
    """Apply morphological operations to smooth mask"""
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask.astype(bool)

def calculate_maximum_diameter_from_points(points):
    if len(points) < 2:
        return 0.0
    points = np.array(points, dtype=float)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    dists = np.linalg.norm(points[:, None] - points, axis=-1)
    return float(np.max(dists))

def extract_dates_from_folder(folder_name):
    folder_name = folder_name.strip()
    parts = folder_name.split('_')
    if len(parts) < 3 or parts[0] != 'D2':
        return None, None
    try:
        dt_start = datetime.strptime(parts[1], "%Y%m%d%H%M%S")
        dt_end = datetime.strptime(parts[2], "%Y%m%d%H%M%S")
        return dt_start.strftime("%Y-%m-%d %H:%M:%S"), dt_end.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return None, None
#%%
def run_sam_automatic_sahi_on_roi(
    image_path,
    roi_crop=(500, 0, 3500, 2160),
    slice_height=2024,
    slice_width=2024,
    overlap_ratio=0.3,
    points_per_side=16,
    pred_iou_thresh=0.75,
    stability_score_thresh=0.90,
    min_mask_area=5000,
    max_mask_area=200000,
):
    """
    Run SAM AutomaticMaskGenerator on ROI only.
    Returns list of (polygon_points, max_diameter)
    """
    # Load and crop image
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    x1_roi, y1_roi, x2_roi, y2_roi = roi_crop
    roi_img = image_rgb[y1_roi:y2_roi, x1_roi:x2_roi]
    H_roi, W_roi = roi_img.shape[:2]
    
    # Initialize SAM
    sam = load_sam_model()
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
    )
    
    # Tiling
    stride_y = int(slice_height * (1 - overlap_ratio))
    stride_x = int(slice_width * (1 - overlap_ratio))
    all_masks_in_roi = []
    
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
                if area < min_mask_area or area > max_mask_area:
                    continue
                if not is_coal_like_region(tile, seg_tile, min_darkness=0.1, max_darkness=0.7):
                    continue
                
                full_roi_mask = np.zeros((H_roi, W_roi), dtype=bool)
                full_roi_mask[y:y2, x:x2] = seg_tile
                full_roi_mask = clean_mask(full_roi_mask)
                
                area_clean = int(full_roi_mask.sum())
                if area_clean < min_mask_area or area_clean > max_mask_area:
                    continue
                
                all_masks_in_roi.append(full_roi_mask)
    
    # Deduplicate
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
    
    # Convert to polygons
    results = []
    for mask in final_masks:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        largest_contour = max(contours, key=cv2.contourArea)
        seg_points = largest_contour.squeeze().tolist()
        if isinstance(seg_points[0], int):
            seg_points = [seg_points]
        adjusted_seg = [[int(x + x1_roi), int(y + y1_roi)] for x, y in seg_points]
        max_diam = calculate_maximum_diameter_from_points(seg_points)
        results.append((adjusted_seg, max_diam))
    
    print(f"✅ Final large coal masks: {len(results)}")
    return results
#%%
def process_image(image_path, output_json_path, video_folder_name):
    try:
        coal_segments = run_sam_automatic_sahi_on_roi(
            image_path,
            roi_crop=ROI_CROP,
            min_mask_area=5000,
            max_mask_area=200000
        )
    except Exception as e:
        print(f"  ❌ SAM failed: {e}")
        coal_segments = []
    
    date_start, date_end = extract_dates_from_folder(video_folder_name)
    if not date_start or not date_end:
        date_start = date_end = ""
    
    predicted_coals = [
        {"segmentation": seg, "maximum_diameters": diam}
        for seg, diam in coal_segments
    ]
    
    result = {
        "image_name": os.path.relpath(image_path, ROOT_DIR),
        "date_start": date_start,
        "date_end": date_end,
        "predicted_coals": predicted_coals
    }
    
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"✅ Processed: {os.path.basename(image_path)} → {len(predicted_coals)} large coals")
    
    # For visualization
    if VISUALIZE_FIRST_N > 0:
        orig_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        vis_masks = []
        for seg, _ in coal_segments:
            mask_full = np.zeros((orig_img.shape[0], orig_img.shape[1]), dtype=np.uint8)
            cv2.fillPoly(mask_full, [np.array(seg, dtype=np.int32)], 255)
            vis_masks.append(mask_full.astype(bool))
        return vis_masks, orig_img
    else:
        return [], None
#%%
def visualize_prediction(image_path, masks, original_img, roi_crop, save_path=None):
    x1, y1, x2, y2 = roi_crop
    vis_img = original_img.copy()
    
    np.random.seed(42)
    colors = [(np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255)) for _ in masks]
    
    for i, mask in enumerate(masks):
        color_mask = np.zeros_like(original_img)
        color_mask[..., 0] = mask * colors[i][0]
        color_mask[..., 1] = mask * colors[i][1]
        color_mask[..., 2] = mask * colors[i][2]
        vis_img = cv2.addWeighted(vis_img, 1, color_mask, 0.5, 0)
    
    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(vis_img)
    plt.title(f"{os.path.basename(image_path)}\nLarge coals only (5k–200k px²)")
    plt.axis('off')
    plt.show()
    
    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
#%%
# ----------------------------
# MAIN PROCESSING LOOP
# ----------------------------
print("🔍 Starting coal segmentation...")

for root, dirs, files in os.walk(ROOT_DIR):
    for file in files:
        if file.lower().endswith('.jpg'):
            image_path = os.path.join(root, file)
            video_folder = os.path.basename(os.path.dirname(image_path))
            json_path = os.path.splitext(image_path)[0] + ".json"
            
            try:
                masks, orig_img = process_image(image_path, json_path, video_folder)
                
                if VISUALIZE_FIRST_N > 0 and orig_img is not None:
                    visualize_prediction(image_path, masks, orig_img, ROI_CROP)
                    VISUALIZE_FIRST_N -= 1
                    del masks, orig_img
                    
            except Exception as e:
                print(f"❌ Error on {image_path}: {e}")
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

print("\n🎉 All done!")
#%% md
# ## 📁 Output Example
# 
# For image:
# ```
# frames/170925/D2_20250917233856_20250918000000/frame_0001.jpg
# ```
# 
# Generates:
# ```json
# {
#   "image_name": "170925/D2_20250917233856_20250918000000/frame_0001.jpg",
#   "date_start": "2025-09-17 23:38:56",
#   "date_end": "2025-09-18 00:00:00",
#   "predicted_coals": [
#     {
#       "segmentation": [[510, 100], [515, 102], ...],
#       "maximum_diameters": 187.2
#     }
#   ]
# }
# ```
#%% md
# ## ⚙️ Tuning Tips
# 
# - **Too many false positives?** → Increase `min_mask_area` or `pred_iou_thresh`
# - **Missing small coals?** → Lower `min_mask_area` (but you said "large coals only")
# - **Still OOM?** → Reduce `slice_height`/`slice_width` to `768`
# - **Faster processing?** → Reduce `points_per_side` to `8`
# 
# Happy segmenting! 🪨