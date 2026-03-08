# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import argparse
import re
from scipy.stats import pearsonr
from pathlib import Path
from collections import defaultdict

# ==========================================
# 1. Core Evaluation Metrics
# ==========================================
def calculate_pearson(gt, pd):
    """Calculate Pearson correlation coefficient."""
    if len(gt) <= 1: 
        return 0.0
        
    gt_arr = np.array(gt, dtype=float)
    pd_arr = np.array(pd, dtype=float)
    
    if np.array_equal(gt_arr, pd_arr): 
        return 1.0
    if len(np.unique(gt_arr)) == 1 or len(np.unique(pd_arr)) == 1:
        return 0.0
    
    r, _ = pearsonr(gt_arr, pd_arr)
    return r if not np.isnan(r) else 0.0

def calculate_epic_distance(pearson_r):
    """Calculate EPIC Distance: sqrt((1 - Pearson) / 2)."""
    pearson_r = np.clip(pearson_r, -1.0, 1.0)
    return np.sqrt((1.0 - pearson_r) / 2.0)

def get_task_id_from_path(fpath: Path) -> str:
    """Extract core task_id from the file path by removing suffixes."""
    return re.sub(r'(_positive|_negative_\d+)$', '', fpath.stem)

# ==========================================
# 2. Main Evaluation Logic
# ==========================================
def evaluate_folder(output_dir):
    """Evaluate predictions directly from self-contained output JSON files."""
    out_path = Path(output_dir)
    
    if not out_path.exists():
        print(f"Error: Output directory not found -> {out_path}")
        return

    # Added 'has_pos' and 'has_neg' to track completeness
    task_data = defaultdict(lambda: {
        'pd': [], 
        'sparse_gt': [], 
        'shaped_gt': [], 
        'has_pos': False, 
        'has_neg': False
    })
    valid_files = 0

    print(f"Scanning directory: {out_path} ...")

    for out_file in out_path.glob("*.json"):
        try:
            with open(out_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            eval_block = data.get("evaluation", {})
            pd_vals = eval_block.get("predicted_rewards", [])
            sparse_gt = eval_block.get("sparse_gt", [])
            shaped_gt = eval_block.get("shaped_gt", [])
            
            if not pd_vals or not sparse_gt or not shaped_gt: 
                continue
                
            tid = get_task_id_from_path(out_file)
            
            # Record the presence of positive and negative samples
            if "positive" in out_file.name:
                task_data[tid]['has_pos'] = True
            elif "negative" in out_file.name:
                task_data[tid]['has_neg'] = True

            task_data[tid]['pd'].extend(pd_vals)
            task_data[tid]['sparse_gt'].extend(sparse_gt)
            task_data[tid]['shaped_gt'].extend(shaped_gt)
            
            valid_files += 1
            
        except Exception as e:
            print(f"Warning: Failed to parse {out_file.name} -> {e}")
            continue

    r_s_list, r_sh_list = [], []
    epic_s_list, epic_sh_list = [], []
    skipped_tasks = 0

    for tid, data in task_data.items():
        # Skip this task if it doesn't have BOTH positive and negative samples
        if not (data['has_pos'] and data['has_neg']):
            skipped_tasks += 1
            continue

        r_s = calculate_pearson(data['sparse_gt'], data['pd'])
        r_sh = calculate_pearson(data['shaped_gt'], data['pd'])
        
        epic_s = calculate_epic_distance(r_s)
        epic_sh = calculate_epic_distance(r_sh)

        r_s_list.append(r_s)
        r_sh_list.append(r_sh)
        epic_s_list.append(epic_s)
        epic_sh_list.append(epic_sh)

    valid_tasks = len(task_data) - skipped_tasks

    print("\n" + "="*60)
    print("Evaluation Completed")
    print(f"Processed {valid_files} files.")
    print(f"Valid Tasks (Both Pos & Neg): {valid_tasks}")
    print(f"Skipped Tasks (Incomplete) : {skipped_tasks}")
    print("-" * 60)
    
    if r_s_list:
        print(f"[Sparse GT] Pearson R: {np.mean(r_s_list):.4f} | EPIC Distance: {np.mean(epic_s_list):.4f}")
        print(f"[Shaped GT] Pearson R: {np.mean(r_sh_list):.4f} | EPIC Distance: {np.mean(epic_sh_list):.4f}")
    else:
        print("Warning: No valid task data (requiring both pos and neg) found in the directory.")
    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate predictions using self-contained output JSON files.")
    
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Directory containing the output prediction JSON files.")
    
    args = parser.parse_args()
    evaluate_folder(args.data_dir)