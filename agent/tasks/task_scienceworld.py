import json
import random
from pathlib import Path
from typing import Any, Dict, Generator, Tuple

def load_sci_task(
    data_dir: str = "rewardprediction/scienceworld",
    part_num: int = 1,
    part_idx: int = -1
) -> Tuple[Generator[Dict[str, Any], None, None], int]:
    """
    Loads ScienceWorld trajectories from nested JSON files.
    Supports distributed partitioning for consistency across training nodes.
    """
    traj_dir = Path(data_dir)
    random.seed(3944)

    if not traj_dir.exists():
        raise FileNotFoundError(f"Trajectory directory not found: {traj_dir}")

    # 1. Collect and sort JSON files for deterministic loading
    all_json_files = []
    task_dirs = sorted([d for d in traj_dir.rglob("*") if d.is_dir()], key=lambda x: str(x.name))
    
    for d in task_dirs:
        pos_file = d / "positive.json"
        if pos_file.exists():
            all_json_files.append(pos_file)
            
        neg_files = sorted(list(d.glob("negative_*.json")))
        all_json_files.extend(neg_files)

    all_json_files = sorted(list(set(all_json_files)), key=lambda p: str(p))
    total_files = len(all_json_files)

    # 2. Distributed partitioning logic
    if part_num > 1:
        if part_idx == -1:
            raise ValueError("part_idx must be specified when part_num > 1")
        
        part_size = total_files // part_num
        start = part_idx * part_size
        end = start + part_size if part_idx < part_num - 1 else total_files
        selected_files = all_json_files[start:end]
    else:
        selected_files = all_json_files

    num_tasks = len(selected_files)

    # 3. Trajectory generator for nested JSON structure
    def generator() -> Generator[Dict[str, Any], None, None]:
        for file_path in selected_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                
                if not isinstance(content, dict):
                    continue
                
                trajectory = content.get("trajectory", [])
                if not trajectory:
                    continue

                # Unique ID from parent folder and filename
                unique_task_id = f"{file_path.parent.name}_{file_path.stem}"

                # Extract root-level metadata
                goal_desc = content.get("goal_description", "")
                
                # Extract step-wise data from the nested trajectory list
                expert_plans = [s.get("action", "") for s in trajectory]
                observations = [s.get("observation", "") for s in trajectory]
                
                # Extract fine-grained reward values
                sparse_rewards = [s.get("reward", {}).get("raw", 0.0) for s in trajectory]
                shaped_rewards = [s.get("reward", {}).get("shaped", 0.0) for s in trajectory]

                yield {
                    "task_id": unique_task_id,            
                    "goal_description": goal_desc,                
                    "expert_plan": expert_plans,  
                    "obs": observations,                  
                    "sparse_gt": sparse_rewards,
                    "shaped_gt": shaped_rewards,            
                }
                
            except Exception as e:
                print(f"❌ Error processing {file_path}: {e}")

    return generator(), num_tasks