import json
import random
from pathlib import Path
from typing import Generator, Tuple

def load_action100m_task(
    data_dir: str = "rewardprediction/action100m", 
    part_num: int = 1,
    part_idx: int = -1
) -> Tuple[Generator[dict, None, None], int]:
    """
    Loads Action100M trajectories from nested JSON files.
    Supports distributed partitioning for consistency across training nodes.
    """
    traj_dir = Path(data_dir)
    random.seed(3944)
    
    if not traj_dir.exists():
        raise FileNotFoundError(f"Trajectory directory not found: {traj_dir}")

    # 1. Collect and sort JSON files for deterministic loading
    all_json_files = sorted(list(traj_dir.rglob("*.json")), key=lambda p: str(p))
    total_files = len(all_json_files)

    # 2. Partitioning logic for distributed training
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
    def generator() -> Generator[dict, None, None]:
        for file_path in selected_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                
                if not isinstance(content, dict):
                    continue
                
                trajectory = content.get("trajectory", [])
                if not trajectory:
                    continue
                
                unique_task_id = file_path.stem
                
                # Extract root-level metadata
                goal_desc = content.get("goal_description", "")
                
                # Extract step-wise data from the nested trajectory list
                expert_plans = [s.get("action", "") for s in trajectory]
                observations = [s.get("observation", "") for s in trajectory]
                
                # Extract temporal grounding boundaries
                starts = [s.get("start", 0.0) for s in trajectory]
                ends = [s.get("end", 0.0) for s in trajectory]
                
                # Extract fine-grained reward values
                sparse_rewards = [s.get("reward", {}).get("raw", 0.0) for s in trajectory]
                shaped_rewards = [s.get("reward", {}).get("shaped", 0.0) for s in trajectory]

                yield {
                    "task_id": unique_task_id, 
                    "goal_description": goal_desc,
                    "expert_plan": expert_plans,
                    "obs": observations,
                    "starts": starts,
                    "ends": ends,
                    "sparse_gt": sparse_rewards,
                    "shaped_gt": shaped_rewards,
                }
            except Exception as e:
                print(f"⚠️ Error loading {file_path}: {e}")
                continue

    return generator(), num_tasks