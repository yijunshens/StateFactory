"""
Action100M Data Preprocessing Pipeline for StateFactory.

This script handles the end-to-end preprocessing of the Action100M dataset:
1. Downloading Parquet files from Hugging Face.
2. Extracting Parquet files to JSON.
3. Cleaning and filtering action timelines.
4. Generating task goals and interpretations using a local LLM (vLLM) via external config.
5. Formatting the data into Reinforcement Learning (RL) trajectories.
"""

import os
import json
import glob
import argparse
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed


# ==========================================
# Utility Classes
# ==========================================
class NumpyEncoder(json.JSONEncoder):
    """Custom encoder to handle NumPy data types for JSON serialization."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return super(NumpyEncoder, self).default(obj)


# ==========================================
# Step 1: Download Data
# ==========================================
def step1_download_data(args):
    print("\n" + "="*50)
    print("🚀 Step 1: Downloading Action100M Data")
    print("="*50)
    
    os.makedirs(args.raw_dir, exist_ok=True)
    headers = {"Authorization": f"Bearer {args.hf_token}"}
    api_url = f"{args.hf_endpoint}/api/datasets/{args.hf_repo_id}/tree/main/data"
    
    print("Fetching file list from mirror...")
    response = requests.get(api_url, headers=headers)
    response.raise_for_status()
    
    all_files = [f['path'] for f in response.json() if f['path'].endswith('.parquet')]
    print(f"Total shards detected: {len(all_files)}")

    for path in tqdm(all_files, desc="Downloading Parquet Files"):
        file_url = f"{args.hf_endpoint}/datasets/{args.hf_repo_id}/resolve/main/{path}"
        local_filename = os.path.join(args.raw_dir, os.path.basename(path))
        
        # Resume download logic
        if os.path.exists(local_filename) and os.path.getsize(local_filename) > 0:
            continue

        try:
            with requests.get(file_url, headers=headers, stream=True) as resp:
                resp.raise_for_status()
                with open(local_filename, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
        except Exception as e:
            print(f"\nFailed to download {path}: {e}")
            continue

    print(f"--- Data successfully downloaded to: {args.raw_dir} ---")


# ==========================================
# Step 2: Extract Parquet to JSON
# ==========================================
def step2_extract_parquet(args):
    print("\n" + "="*50)
    print("🚀 Step 2: Extracting Parquet to JSON")
    print("="*50)
    
    os.makedirs(args.extracted_dir, exist_ok=True)
    files = glob.glob(os.path.join(args.raw_dir, "*.parquet"))
    
    for f in files:
        try:
            df = pd.read_parquet(f)
            parquet_name = os.path.splitext(os.path.basename(f))[0]
            sub_dir = os.path.join(args.extracted_dir, parquet_name)
            os.makedirs(sub_dir, exist_ok=True)
            
            for _, row in df.iterrows():
                video_uid = str(row['video_uid']).replace('/', '_')
                json_file_path = os.path.join(sub_dir, f"{video_uid}.json")
                
                with open(json_file_path, 'w', encoding='utf-8') as jf:
                    json.dump(row.to_dict(), jf, ensure_ascii=False, indent=4, cls=NumpyEncoder)
                    
            print(f"✅ Extracted: {parquet_name}")
        except Exception as e:
            print(f"❌ Error processing {f}: {e}")


# ==========================================
# Step 3: Clean Timeline
# ==========================================
def extract_text_features(node):
    actions, captions = {}, {}
    if node.get('plm_action') and str(node['plm_action']).strip():
        actions['plm'] = str(node['plm_action']).strip()
    if node.get('plm_caption') and str(node['plm_caption']).strip():
        captions['plm'] = str(node['plm_caption']).strip()
    if node.get('llama3_caption') and str(node['llama3_caption']).strip():
        captions['llama3'] = str(node['llama3_caption']).strip()
        
    gpt_data = node.get('gpt')
    if isinstance(gpt_data, dict):
        if isinstance(gpt_data.get('summary'), dict) and gpt_data['summary'].get('detailed'):
            captions['gpt'] = str(gpt_data['summary']['detailed']).strip()
        if isinstance(gpt_data.get('action'), dict) and gpt_data['action'].get('detailed'):
            actions['gpt'] = str(gpt_data['action']['detailed']).strip()
    return actions, captions

def evaluate_overlap(parent_start, parent_end, children, overlap_threshold, epsilon, min_duration):
    if not children: return 'GAP', None
    if children[0]['start'] > parent_start + epsilon: return 'GAP', None
    if children[-1]['end'] < parent_end - epsilon: return 'GAP', None

    adjusted_children = []
    current_node = children[0].copy()
    current_node['start'] = parent_start
    
    for i in range(1, len(children)):
        next_node = children[i].copy()
        gap = next_node['start'] - current_node['end']
        
        if gap > epsilon:
            return 'GAP', None
        elif gap < -epsilon:
            overlap = -gap
            min_dur = min(current_node['end'] - current_node['start'], next_node['end'] - next_node['start'])
            overlap_ratio = overlap / min_dur if min_dur > 0 else 1.0
            if overlap_ratio > overlap_threshold:
                return 'HIGH_OVERLAP', None
            else:
                current_node['end'] = next_node['start']
        else:
            current_node['end'] = next_node['start']
        
        adjusted_children.append(current_node)
        current_node = next_node
        
    current_node['end'] = parent_end
    adjusted_children.append(current_node)
    
    if min_duration > 0:
        for child in adjusted_children:
            if (child['end'] - child['start']) < min_duration - 1e-4:
                return 'TOO_SHORT', None
                
    return 'FINE_GRAINED', adjusted_children

def extract_smart_timeline(node, children_map, overlap_threshold, epsilon, min_duration):
    children = sorted(children_map.get(node['node_id'], []), key=lambda x: x['start'])
    strategy, adjusted_children = evaluate_overlap(node['start'], node['end'], children, overlap_threshold, epsilon, min_duration)
    
    if strategy == 'FINE_GRAINED':
        timeline = []
        for child in adjusted_children:
            timeline.extend(extract_smart_timeline(child, children_map, overlap_threshold, epsilon, min_duration))
        return timeline
    elif strategy == 'HIGH_OVERLAP':
        merged_actions, merged_captions = {}, {}
        for child in children:
            c_actions, c_captions = extract_text_features(child)
            for k, v in c_actions.items(): merged_actions.setdefault(k, []).append(v)
            for k, v in c_captions.items(): merged_captions.setdefault(k, []).append(v)
            
        result_node = {
            'start': round(node['start'], 3), 'end': round(node['end'], 3),
            'duration': round(node['end'] - node['start'], 3), 'level': node.get('level', 0)
        }
        if merged_actions: result_node['actions'] = {k: "\n---\n".join(v) for k, v in merged_actions.items() if v}
        if merged_captions: result_node['captions'] = {k: "\n---\n".join(v) for k, v in merged_captions.items() if v}
        return [result_node]
    else:
        result_node = {
            'start': round(node['start'], 3), 'end': round(node['end'], 3),
            'duration': round(node['end'] - node['start'], 3), 'level': node.get('level', 0)
        }
        actions, captions = extract_text_features(node)
        if actions: result_node['actions'] = actions
        if captions: result_node['captions'] = captions
        return [result_node]

def step3_clean_timeline(args):
    print("\n" + "="*50)
    print(f"🚀 Step 3: Cleaning Timeline (Epsilon: {args.epsilon}s | Min Duration: {args.min_duration}s)")
    print("="*50)
    
    os.makedirs(args.cleaned_dir, exist_ok=True)
    processed_count = 0
    
    for root, _, files in os.walk(args.extracted_dir):
        for file in files:
            if file.endswith('.json'):
                input_path = os.path.join(root, file)
                output_path = os.path.join(args.cleaned_dir, os.path.relpath(input_path, args.extracted_dir))
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                try:
                    with open(input_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    nodes = data.get('nodes', [])
                    if not nodes: continue
                    
                    children_map, root_node = {}, None
                    for n in nodes:
                        pid = n.get('parent_id')
                        if pid is None: root_node = n
                        else: children_map.setdefault(pid, []).append(n)
                        
                    if not root_node:
                        root_node = sorted(nodes, key=lambda x: (x['start'], -x['end']))[0]
                        
                    final_timeline = extract_smart_timeline(root_node, children_map, 0.2, args.epsilon, args.min_duration)
                    
                    out_data = {
                        "video_uid": data.get("video_uid", file.replace('.json', '')),
                        "overall_duration": data.get("metadata", {}).get("duration"),
                        "total_segments": len(final_timeline),
                        "timeline": final_timeline
                    }
                    
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(out_data, f, ensure_ascii=False, indent=4)
                        
                    processed_count += 1
                except Exception as e:
                    print(f"❌ Error processing {input_path}: {e}")

    print(f"✅ Timeline cleaning complete. Processed {processed_count} files.")


# ==========================================
# Step 4: Generate Goals (LLM)
# ==========================================
def format_timeline_for_llm(timeline):
    clean_timeline = []
    for seg in timeline:
        actions = seg.get('actions', {})
        captions = seg.get('captions', {})
        
        action_text = actions.get('gpt') or actions.get('plm') or ""
        caption_text = captions.get('gpt') or captions.get('plm') or ""
        
        if action_text or caption_text:
            clean_timeline.append({
                "start": seg["start"], 
                "end": seg["end"],
                "action": action_text, 
                "caption": caption_text
            })
    return clean_timeline

def process_single_llm_task(task_args, client, model_name, temperature):
    input_path, output_path, file_name = task_args
    
    if os.path.exists(output_path): 
        return 'SKIPPED', file_name, ""
        
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        clean_timeline = format_timeline_for_llm(data.get('timeline', []))
        if not clean_timeline: 
            return 'ERROR', file_name, "Empty timeline"
            
        prompt = f"""You are an expert AI video analyst. Your task is to analyze a video's action timeline and extract the global task information.

I will provide you with a JSON timeline representing a sequence of actions in a video. Each segment has a start time, end time, an action summary, and a detailed caption.

Based on this sequence, you must output a JSON containing EXACTLY THREE keys: "goal", "interpretation", and "goal_description".

1. "goal": A concise 1-sentence summary of the overall objective.
2. "interpretation": A detailed explanation of the task, which MUST include the initial state (materials/tools present at the beginning), the transformation process (key actions performed), and the final state (the ultimate result).
3. "goal_description": A comprehensive paragraph that seamlessly concatenates the Goal and the Interpretation.

Here is the Input Video Timeline you need to process:
{json.dumps(clean_timeline, ensure_ascii=False, indent=2)}

Constraint: Output ONLY a valid JSON with the exact three keys "goal", "interpretation", and "goal_description". Do not include conversational text or markdown code blocks."""

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful AI video analyst. Always return valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
        )
        llm_response_text = response.choices[0].message.content.strip()
        
        # Directly parse JSON (removed markdown cleaning logic per request)
        try:
            llm_data = json.loads(llm_response_text)
        except json.JSONDecodeError:
            return 'ERROR', file_name, f"JSON parsing failed. \nRaw Output: {llm_response_text}"
        
        goal_text = llm_data.get('goal', '')
        interpretation_text = llm_data.get('interpretation', '')
        goal_desc_text = llm_data.get('goal_description', '')
        
        new_data = {}
        
        if 'meta_info' in data:
            meta = data.pop('meta_info')
            new_meta = {}
            if 'task_id' in meta:
                new_meta['task_id'] = meta.pop('task_id')
                
            new_meta['goal'] = goal_text
            new_meta['interpretation'] = interpretation_text
            new_meta['goal_description'] = goal_desc_text
            
            for k, v in meta.items():
                if k not in ['goal', 'interpretation', 'goal_description']:
                    new_meta[k] = v
            new_data['meta_info'] = new_meta
        else:
            new_data['goal'] = goal_text
            new_data['interpretation'] = interpretation_text
            new_data['goal_description'] = goal_desc_text
            
        for key, value in data.items():
            if key not in ['goal', 'interpretation', 'goal_description']:
                new_data[key] = value
                
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, ensure_ascii=False, indent=4)
            
        return 'SUCCESS', file_name, ""
        
    except Exception as e:
        return 'ERROR', file_name, f"Unknown error: {e}"

def step4_generate_goals(args):
    print("\n" + "="*50)
    print("🚀 Step 4: Generating Goals (LLM)")
    print("="*50)
    
    if not os.path.exists(args.llm_config_path):
        raise FileNotFoundError(f"❌ Cannot find LLM config file at: {args.llm_config_path}")
        
    with open(args.llm_config_path, 'r', encoding='utf-8') as f:
        llm_config = json.load(f)
        
    api_key = llm_config.get("api_key", "EMPTY")
    api_base = llm_config.get("api_base", "http://localhost:9011/v1")
    model_name = llm_config.get("model_name", "gpt-oss-20b")
    temperature = llm_config.get("temperature", 0.01)
    
    print(f"⚙️  Loaded Config -> Model: {model_name} | API Base: {api_base} | Temp: {temperature}")
    
    client = OpenAI(api_key=api_key, base_url=api_base)
    os.makedirs(args.llm_dir, exist_ok=True)
    
    tasks = []
    for root, _, files in os.walk(args.cleaned_dir):
        for file in files:
            if file.endswith('.json'):
                input_path = os.path.join(root, file)
                output_path = os.path.join(args.llm_dir, os.path.relpath(input_path, args.cleaned_dir))
                tasks.append((input_path, output_path, file))
                
    total_files = len(tasks)
    processed_count, skipped_count, error_count = 0, 0, 0

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_single_llm_task, task, client, model_name, temperature): task for task in tasks}
        
        for future in tqdm(as_completed(futures), total=total_files, desc="LLM Processing"):
            status, file_name, msg = future.result()
            if status == 'SUCCESS': processed_count += 1
            elif status == 'SKIPPED': skipped_count += 1
            elif status == 'ERROR':
                error_count += 1
                tqdm.write(f"❌ Failed [{file_name}]: {msg}")

    print(f"\n🎉 Task Completed!")
    print(f"📊 Stats -> Success: {processed_count} | Skipped: {skipped_count} | Failed: {error_count}")


# ==========================================
# Step 5: Format to Nested RL Trajectories
# ==========================================
def convert_to_trajectory_format(data):
    """
    Converts raw timeline data into a nested RL trajectory format.
    The goal is stored at the root, and steps are in a nested list.
    """
    timeline = data.get("timeline", [])
    # Extract goal from metadata or fallback to root key
    goal_desc = data.get("metadata", {}).get("goal_description", data.get("goal_description", ""))
    
    total_steps = len(timeline)
    steps_list = []
    
    for i, step in enumerate(timeline):
        actions_dict = step.get("actions", {})
        captions_dict = step.get("captions", {})
        
        # Priority: GPT > PLM > Empty String
        action_text = actions_dict.get("gpt") or actions_dict.get("plm") or ""
        obs_text = captions_dict.get("gpt") or captions_dict.get("plm") or ""
        
        start_time = step.get("start", 0.0)
        end_time = step.get("end", 0.0)
        
        # Reward calculation: 1.0 at final step, shaped progress otherwise
        is_last_step = (i == total_steps - 1)
        raw_reward = 1.0 if is_last_step else 0.0
        shaped_reward = round((i + 1) / total_steps, 2) if total_steps > 0 else 0.0
        
        steps_list.append({
            "action": action_text,
            "observation": obs_text,
            "start": start_time,
            "end": end_time,
            "reward": {
                "raw": raw_reward,
                "shaped": shaped_reward,
                "is_expert": True
            }
        })
        
    # ✨ Return the new nested structure
    return {
        "goal_description": goal_desc,
        "trajectory": steps_list
    }

def step5_format_trajectories(args):
    print("\n" + "="*50)
    print("🚀 Step 5: Formatting RL Trajectories (Nested Structure)")
    print("="*50)
    
    os.makedirs(args.trajectory_dir, exist_ok=True)
    processed_count, error_count = 0, 0

    tasks = []
    for root, _, files in os.walk(args.llm_dir):
        for file in files:
            if file.endswith('.json'):
                tasks.append(os.path.join(root, file))

    for input_path in tqdm(tasks, desc="Formatting Trajectories"):
        output_path = os.path.join(args.trajectory_dir, os.path.relpath(input_path, args.llm_dir))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Convert to new nested dictionary format
            trajectory_data = convert_to_trajectory_format(data)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(trajectory_data, f, ensure_ascii=False, indent=4)
                
            processed_count += 1
        except Exception as e:
            tqdm.write(f"❌ Error processing {input_path}: {e}")
            error_count += 1
            
    print(f"\n🎉 Conversion Completed!")
    print(f"📊 Stats -> Success: {processed_count} | Failed: {error_count}")
    print(f"📁 RL Trajectories ready at: {args.trajectory_dir}")


# ==========================================
# Main Execution Entry
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Action100M Preprocessing Pipeline")
    
    # Path Arguments (Relative to project root)
    parser.add_argument("--raw_dir", type=str, default="./scripts/action100m/parquet", help="Path for raw parquet downloads")
    parser.add_argument("--extracted_dir", type=str, default="./scripts/action100m/json", help="Path for extracted JSONs")
    parser.add_argument("--cleaned_dir", type=str, default="./scripts/action100m/cleaned", help="Path for cleaned timelines")
    parser.add_argument("--llm_dir", type=str, default="./scripts/action100m/llm_goals", help="Path for LLM generated goals")
    
    # Final output directly to the reward prediction dataset folder
    parser.add_argument("--trajectory_dir", type=str, default="./rewardprediction/action100m", help="Final output path")
    
    # Step 1 Args
    parser.add_argument("--hf_token", type=str, default="", help="HuggingFace token")
    parser.add_argument("--hf_repo_id", type=str, default="facebook/Action100M-preview", help="Dataset repo ID")
    parser.add_argument("--hf_endpoint", type=str, default="https://huggingface.co", help="HF endpoint")
    
    # Step 3 Args
    parser.add_argument("--epsilon", type=float, default=1.0, help="Epsilon tolerance for timeline gaps (seconds)")
    parser.add_argument("--min_duration", type=float, default=5.0, help="Minimum duration for a valid segment (seconds)")
    
    # Step 4 Args (Relative to project root)
    parser.add_argument("--llm_config_path", type=str, default="./configs/llm/openai.json", help="Path to global LLM config file")
    parser.add_argument("--max_workers", type=int, default=64, help="Max parallel workers for LLM processing")
    
    # Pipeline Control Flags
    parser.add_argument("--skip_download", action="store_true", help="Skip Step 1 (Download)")
    parser.add_argument("--skip_extract", action="store_true", help="Skip Step 2 (Extract)")
    parser.add_argument("--skip_clean", action="store_true", help="Skip Step 3 (Clean Timeline)")
    parser.add_argument("--skip_llm", action="store_true", help="Skip Step 4 (Generate Goals)")
    parser.add_argument("--skip_format", action="store_true", help="Skip Step 5 (Format Trajectories)")

    args = parser.parse_args()

    # Execute Pipeline
    if not args.skip_download: step1_download_data(args)
    if not args.skip_extract: step2_extract_parquet(args)
    if not args.skip_clean: step3_clean_timeline(args)
    if not args.skip_llm: step4_generate_goals(args)
    if not args.skip_format: step5_format_trajectories(args)