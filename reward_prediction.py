"""
main.py

Entry point for the agent evaluation pipeline.
Manages vLLM server lifecycle, Embedding server, and worker processes.
"""

import argparse
import json
import os
import sys
import time
import atexit
import subprocess
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict, Any, Optional, Tuple

import requests
from tqdm import tqdm

# --- Internal Imports ---
from agent.llm.openai_client import OpenAILLM
from agent.reward.reward import get_reward
from agent.embedding.client import RemoteEmbeddingModel
import agent.envs as envs
import agent.tasks as task_loaders
from configs.llm.llm_config import get_model_config

# --- Global Variables (Per Worker) ---
worker_args: Optional[argparse.Namespace] = None
worker_env_config: Optional[Dict] = None
worker_embedding_model: Optional[RemoteEmbeddingModel] = None
worker_agent: Optional[OpenAILLM] = None

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("MainRunner")


# ==============================================================================
# 1. Server Management
# ==============================================================================

def ensure_llm_server(model_name: str) -> Optional[subprocess.Popen]:
    """
    Ensures vLLM server is running. Launches it if port is free.
    """
    # 1. Get Config
    config = get_model_config(model_name)
    port = config.get("port", 8011)
    health_url = f"http://localhost:{port}/v1/models"

    # 2. Check Existing Server
    try:
        if requests.get(health_url, timeout=1).status_code == 200:
            logger.info(f"✅ Active LLM Server found on port {port}. Reusing it.")
            return None
    except requests.exceptions.ConnectionError:
        pass

    logger.info(f"🚀 Launching vLLM Server for '{model_name}' on port {port}...")

    # 3. Build Command
    # Safe .get() allows config keys to be optional
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", config["model_path"],
        "--served-model-name", model_name,
        "--max-model-len", str(config.get("max_model_len", 8192)),
        "--dtype", config.get("dtype", "auto"),
        "--gpu-memory-utilization", str(config.get("gpu_memory_utilization", 0.90)),
        "--port", str(port),
        "--trust-remote-code",
        "--tensor-parallel-size", str(config.get("tensor_parallel_size", 1))
    ]
    server_env = os.environ.copy()

    # 4. Start Process
    # stdout=None redirects logs to main console
    proc = subprocess.Popen(cmd, env=server_env, stdout=None, stderr=None)

    # 6. Wait for Ready
    logger.info("⏳ Waiting for vLLM to load weights...")
    start_time = time.time()
    
    # Wait up to 10 mins for large models
    for _ in range(600):
        try:
            if requests.get(health_url, timeout=2).status_code == 200:
                logger.info(f"✅ LLM Server Ready! ({time.time() - start_time:.1f}s)")
                return proc
        except requests.exceptions.ConnectionError:
            pass
        
        if proc.poll() is not None:
            raise RuntimeError("❌ vLLM process died during startup.")
        time.sleep(1)

    proc.terminate()
    raise TimeoutError("vLLM Server startup timed out.")

def ensure_embedding_server(model_name: str, port: int) -> Optional[subprocess.Popen]:
    """
    Ensures Embedding Server is running.
    """
    health_url = f"http://localhost:{port}/health"
    start_script_path = "agent/embedding/server.py"

    try:
        if requests.get(health_url, timeout=1).status_code == 200:
            logger.info(f"✅ Active Embedding Server found on port {port}.")
            return None
    except requests.exceptions.ConnectionError:
        pass

    if not os.path.exists(start_script_path):
        raise FileNotFoundError(f"Script missing: {start_script_path}")

    logger.info(f"🚀 Launching Embedding Server ({model_name}) on port {port}...")
    cmd = [sys.executable, start_script_path, "--model_name", model_name, "--port", str(port)]
    proc = subprocess.Popen(cmd)

    for i in range(120):
        try:
            if requests.get(health_url, timeout=15).status_code == 200:
                logger.info(f"✅ Embedding Server Ready!")
                return proc
        except: pass
        
        if proc.poll() is not None:
            raise RuntimeError("Embedding Server died.")
        time.sleep(1)
    
    raise TimeoutError("Embedding Server startup timed out.")


# ==============================================================================
# 2. Worker Logic
# ==============================================================================

def init_worker(g_args: argparse.Namespace, g_exp_config: Dict, g_llm_config: Dict):
    """
    Worker initialization. Sets up LLM and Embedding clients.
    """
    global worker_args, worker_env_config, worker_embedding_model, worker_agent
    
    worker_args = g_args
    worker_env_config = g_exp_config["env_config"]
    model_name = g_args.agent_model_name 

    # --- Initialize LLM Agent ---
    if g_args.backend == "vllm":
        logger.info(f"[Worker {os.getpid()}] Connecting to Local vLLM: {model_name}")
        
        # Get config to find port and temperature
        config = get_model_config(model_name)
        port = config.get("port", 8011)
        
        local_api_config = {
            "model_name": model_name,
            "api_key": "EMPTY",
            "api_base": f"http://localhost:{port}/v1",
            "max_tokens": config.get('max_model_len'), # Generation limit
            # Priority: Config file > CLI Argument
            "temperature": config.get('temperature')
        }
        worker_agent = OpenAILLM(local_api_config)
    else:
        logger.info(f"[Worker {os.getpid()}] Connecting to Remote API: {model_name}")
        worker_agent = OpenAILLM(g_llm_config)

    # --- Initialize Embedding ---
    if hasattr(g_args, 'embedding_name') and g_args.embedding_name:
        try:
            worker_embedding_model = RemoteEmbeddingModel(port=g_args.embedding_port)
        except Exception as e:
            logger.error(f"Embedding init failed: {e}")
            worker_embedding_model = None
    else:
        worker_embedding_model = None

def save_trajectory(args, task_id, trajectory_data, predicted_rewards):
    """
    Saves trajectory and evaluation results to a JSON file.
    Resolves filename collisions automatically.
    """
    output_dir = os.path.join(args.output_path, "complete")
    os.makedirs(output_dir, exist_ok=True)
    
    metadata = trajectory_data.get("metadata", {})
    
    data = {
        "meta_info": {
            "task_id": task_id,
            "goal_description": trajectory_data.get("goal_description", "unknown"),
            "backend": args.backend,
            "model": args.agent_model_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "trajectory": trajectory_data.get("trajectory", []),
        "thinkings": trajectory_data.get("thinkings", []),
        "evaluation": {
            "gold_actions": metadata.get("actions", []),
            "predicted_rewards": predicted_rewards,
            "sparse_gt": metadata.get("sparse_gt", []),
            "shaped_gt": metadata.get("shaped_gt", []),
        }
    }
    
    # Dynamically inject temporal grounding metadata if available
    if "starts" in metadata:
        data["evaluation"]["starts"] = metadata["starts"]
    if "ends" in metadata:
        data["evaluation"]["ends"] = metadata["ends"]

    # 1. Set base filename
    file_name = f"{task_id}.json"
    base_name = str(task_id)
    extension = ".json"
    counter = 1

    # 2. Construct initial save path
    save_path = os.path.join(output_dir, file_name)

    # 3. Resolve filename collisions (e.g., generate 405_1.json, 405_2.json)
    while os.path.exists(save_path):
        new_file_name = f"{base_name}_{counter}{extension}"
        save_path = os.path.join(output_dir, new_file_name)
        counter += 1

    # 4. Save to disk
    os.makedirs(output_dir, exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def run_task_loop(task, env_config, args, embedding_model, agent):
    """
    Executes a single task.
    """
    env_kwargs = dict(env_config)
    if hasattr(args, "output_format"): env_kwargs["output_format"] = args.output_format
    if agent: env_kwargs["agent"] = agent
    
    # Dynamic Env Loading
    env_class = getattr(envs, env_config["env_class"])
    env: envs.BaseEnv = env_class(task, **env_kwargs)

    # Generate & Evaluate
    trajectory_data = env.generate_trajectory_data()
    
    traj = trajectory_data['trajectory']
    predicted_rewards = get_reward(
        embedding_model, 
        [s.get('CurState', '') for s in traj], 
        [s.get('GoalState', '') for s in traj]
    )

    save_trajectory(args, task['task_id'], trajectory_data, predicted_rewards)


def process_task_wrapper(task: Dict) -> Tuple[str, str]:
    global worker_args
    task_id = task.get('task_id', 'unknown')
    
    if worker_args is None: 
        return (task_id, "Worker Not Initialized")

    max_retries = 10
    last_exception = None

    for attempt in range(1, max_retries + 1):
        try:
            run_task_loop(task, worker_env_config, worker_args, worker_embedding_model, worker_agent)
            
            if attempt > 1:
                logger.info(f"Task {task_id} succeeded on attempt {attempt}.")
                
            return (task_id, "Success")
            
        except Exception as e:
            last_exception = e
            logger.warning(f"⚠️ Task {task_id} failed on attempt {attempt}/{max_retries}: {e}")
            
            if attempt < max_retries:
                time.sleep(2) 
            
    # Exhausted all retries
    import traceback
    traceback.print_exception(type(last_exception), last_exception, last_exception.__traceback__)
    logger.error(f"❌ Task {task_id} completely failed after {max_retries} attempts.")
    
    return (task_id, f"Error: {last_exception}")


# ==============================================================================
# 3. Main Entry
# ==============================================================================

def main(args: argparse.Namespace):
    # A. Start vLLM Server
    vllm_proc = None
    if args.backend == "vllm":
        if not args.agent_model_name:
            logger.critical("Error: --agent_model_name required for vllm.")
            sys.exit(1)
        try:
            vllm_proc = ensure_llm_server(args.agent_model_name)
            if vllm_proc: atexit.register(lambda: vllm_proc.terminate())
        except Exception as e:
            logger.critical(f"vLLM Start Failed: {e}")
            sys.exit(1)

    # B. Start Embedding Server
    embed_proc = None
    if args.embedding_name:
        try:
            embed_proc = ensure_embedding_server(args.embedding_name, args.embedding_port)
            if embed_proc: atexit.register(lambda: embed_proc.terminate())
        except Exception as e:
            logger.critical(f"Embedding Start Failed: {e}")
            if vllm_proc: vllm_proc.terminate()
            return

    # C. Load Configs
    with open(os.path.join(args.exp_path, f"{args.exp_config}.json")) as f:
        exp_config = json.load(f)

    llm_config = {}
    llm_conf_path = os.path.join(args.agent_path, f"{args.agent_config}.json")
    if os.path.exists(llm_conf_path):
        with open(llm_conf_path, 'r') as f: llm_config = json.load(f)

    # Override Model Name
    if args.agent_model_name and args.backend != 'api':
        llm_config["model_name"] = args.agent_model_name
    else:
        args.agent_model_name = llm_config.get("model_name")

    # D. Load Tasks
    logger.info("Loading tasks...")
    loader_func = getattr(task_loaders, exp_config["env_config"]["task_loader"])
    all_tasks, total_tasks = loader_func(part_num=args.part_num, part_idx=args.part_idx)
    
    # Filter Completed
    output_dir = os.path.join(args.output_path, "complete")
    existing = set(os.listdir(output_dir)) if os.path.exists(output_dir) else set()

    # ==========================================
    # MODIFIED LOGIC: Check for both pos and neg
    # ==========================================
    tasks_to_run = []
    for t in all_tasks:
        tid = f"{t['task_id']}.json"

        if tid in existing:
            logger.info(f"⏭️  Skipped task {tid}: The file exists.")
        else:
            tasks_to_run.append(t)
    # ==========================================
    
    logger.info(f"Total: {total_tasks} | Queued: {len(tasks_to_run)}")
    if not tasks_to_run: return

    # E. Execution
    logger.info(f"Starting {args.num_workers} workers | Backend: {args.backend}")
    
    results = []
    with ProcessPoolExecutor(
        max_workers=args.num_workers,
        initializer=init_worker,
        initargs=(args, exp_config, llm_config)
    ) as executor:
        for res in tqdm(executor.map(process_task_wrapper, tasks_to_run), total=len(tasks_to_run)):
            results.append(res)

    # F. Summary
    failed = [r for r in results if r[1] != "Success"]
    if failed:
        logger.warning(f"{len(failed)} Tasks Failed.")
        for tid, err in failed: logger.warning(f"  {tid}: {err}")
    else:
        logger.info("All tasks successful.")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    print(f"\n🚀 [Main Process] Started. PID: {os.getpid()}")
    
    parser = argparse.ArgumentParser()
    
    # Core Args
    parser.add_argument("--backend", type=str, default="api", choices=["api", "vllm"])
    parser.add_argument("--agent_model_name", type=str, default="openai/gpt-oss-20b")
    
    # Config Paths
    parser.add_argument("--exp_path", type=str, default="configs/tasks")
    parser.add_argument("--exp_config", type=str, default="alfworld")
    parser.add_argument("--agent_path", type=str, default="configs/llm/")
    parser.add_argument("--agent_config", type=str, default="openai")
    
    # Job Specs
    parser.add_argument("--part_num", type=int, default=1)
    parser.add_argument("--part_idx", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, default=1)
    
    # Output & Embeddings
    parser.add_argument("--output_path", type=str, default="output/")
    parser.add_argument("--output_format", type=str, default="OBJ_DETAILED")
    parser.add_argument("--embedding_name", type=str, default="all")
    parser.add_argument("--embedding_port", type=int, default=8013)
    
    # Params
    parser.add_argument("--temperature", type=float, default=0.01)
    
    main(parser.parse_args())