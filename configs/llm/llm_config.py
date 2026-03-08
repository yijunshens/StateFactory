import os

"""
llm_config.py
Stores configuration parameters for loading Large Language Models (LLMs).
"""

# =========================================================
# 0. Adaptive GPU Count Retrieval (Core Logic)
# =========================================================
def get_auto_tp_size():
    """Automatically detects the number of available GPUs in the current environment."""
    
    # 1. Prioritize checking CUDA_VISIBLE_DEVICES set in the startup script
    cvd = os.getenv("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        # For example, "0,1,2,3" will be split into 4 elements
        # Filter out empty strings to prevent issues with trailing commas (e.g., "0,1,")
        gpus = [x for x in cvd.split(',') if x.strip()]
        if len(gpus) > 0:
            return len(gpus)
            
    # 2. If the environment variable is not set, fallback to PyTorch to get the physical GPU count
    try:
        import torch
        count = torch.cuda.device_count()
        return count if count > 0 else 1
    except ImportError:
        # Fallback to 1 if PyTorch is not installed or an error occurs
        return 1

# Get the adaptive tensor parallel size
AUTO_TP_SIZE = get_auto_tp_size()


# =========================================================
# 1. Static Configuration Registry
#    (Reserved for models requiring specific tuning or hardware setups)
# =========================================================
MODEL_CONFIGS = {
    # Example: Large Model Setup
    "<YOUR_MODEL_NAME_1>": {
        "model_path": "<PATH_TO_YOUR_MODEL_WEIGHTS>", 
        "max_model_len": 8192,
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.90,
        "port": 8011,
        'temperature': 0.01,
        # Use the dynamically calculated adaptive tensor parallel size
        "tensor_parallel_size": AUTO_TP_SIZE, 
    }
}

# =========================================================
# 2. Dynamic Configuration Retrieval (Core Logic)
# =========================================================
def get_model_config(model_name: str):
    """
    Retrieves the configuration for a given model name.
    
    1. If the model is defined in MODEL_CONFIGS, return that specific configuration.
    2. If not defined, return a 'Generic Configuration' and attempt to load it directly.
    """
    
    # A. Match found in static registry
    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name]
    
    # B. No match -> Generate Generic Configuration
    # Allow overriding default parameters via environment variables
    return {
        # Treat model_name directly as the model path (HuggingFace ID or absolute path)
        "model_path": model_name, 
        
        # Set to a safe context length (Override via env GENERIC_MAX_LEN)
        "max_model_len": int(os.getenv("GENERIC_MAX_LEN", 8192)),
        
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.90,
        
        # Default port: 8012 if not specified in environment
        "port": int(os.getenv("LLM_PORT", 8012)),

        'temperature': 0.01,
        
        # Dynamic config uses the adaptive size, but allows override via environment variable
        "tensor_parallel_size": int(os.getenv("TENSOR_PARALLEL_SIZE", AUTO_TP_SIZE)),
    }