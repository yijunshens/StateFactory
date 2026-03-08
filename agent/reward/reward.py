import numpy as np
from scipy.stats import pearsonr, spearmanr
from typing import List, Dict, Any, Tuple, Optional, Union
import logging

# Configure logger to allow downstream users to control verbosity
logger = logging.getLogger(__name__)

# =============================================================================
# Utility Functions
# =============================================================================

def _parse_object_item(item: Union[Dict, str, List]) -> Tuple[Optional[str], List[str], List[str]]:
    """
    Normalizes a state item into a canonical format: (Description, Attributes, Values).
    """
    # Case 1: Direct String (FLAT format)
    if isinstance(item, str):
        return item, ["description"], [item]

    # Case 2: List Format
    if isinstance(item, list):
        desc = " ".join([str(x) for x in item]) if item else ""
        return desc, ["description"], [str(x) for x in item]

    # Case 3: Dictionary Format (Structured Object)
    if isinstance(item, dict):
        obj_content = item.get('object', item) 
        
        if not isinstance(obj_content, dict) or not obj_content:
            if isinstance(obj_content, str):
                return obj_content, ["description"], [obj_content]
            return None, [], []

        desc = list(obj_content.keys())[0]
        raw_attributes = obj_content.get(desc)

        attr_names = []
        attr_values = []

        if raw_attributes:
            if isinstance(raw_attributes, str):
                attr_names.append("state")
                attr_values.append(raw_attributes)
            elif isinstance(raw_attributes, list):
                for sub_item in raw_attributes:
                    if isinstance(sub_item, dict):
                        for k, v in sub_item.items():
                            # Clean specific prefix if needed (as per reference code)
                            k = k.replace("goal_", "").replace("intermediate_", "")
                            attr_names.append(k)
                            attr_values.append(str(v))
                    elif isinstance(sub_item, str):
                        attr_names.append("state")
                        attr_values.append(sub_item)

        return desc, attr_names, attr_values

    return None, [], []


def get_embedding_cached(sim_model: Any, text: str, cache: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Retrieves the embedding for a given text, using a cache to avoid redundant computation.
    """
    if not text:
        return np.zeros(1)

    if text not in cache:
        try:
            # Handle different embedding model interfaces seamlessly
            if hasattr(sim_model, 'encode'):
                emb = sim_model.encode(text)
            elif hasattr(sim_model, 'create_embedding'):
                emb = sim_model.create_embedding(text)
            else:
                logger.warning("sim_model lacks 'encode' or 'create_embedding' method.")
                return np.zeros(1)
            
            # Convert tensors to numpy if necessary
            if not isinstance(emb, np.ndarray):
                if hasattr(emb, 'cpu'):
                    emb = emb.cpu().numpy()
                else:
                    emb = np.array(emb)
                    
            cache[text] = emb.flatten()
        except Exception as e:
            logger.error(f"Error computing embedding for text '{text}': {e}")
            cache[text] = np.zeros(1)
            
    return cache[text]


def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Computes the Cosine Similarity between two pre-computed embeddings.
    """
    if emb1.size <= 1 or emb2.size <= 1 or emb1.shape != emb2.shape:
        return 0.0
        
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
        
    return float(np.dot(emb1, emb2) / (norm1 * norm2))


# =============================================================================
# Core Reward Logic (Hard Max)
# =============================================================================

def calculate_hard_matching_score(
    sim_model: Any, 
    cur_state_list: List[Any], 
    goal_item: Any,
    embedding_cache: Dict[str, np.ndarray]
) -> float:
    """
    Calculates the alignment score between a goal item and the current state using Hard Max logic.
    
    Algorithm:
    1. Pre-compute Goal embedding.
    2. Iterate over all objects in the current state.
    3. For each object, calculate semantic similarity of its Description vs Goal Description.
    4. If the Goal has attributes, find the MAXIMUM matching attribute Key in the current object,
       and use the corresponding Value's similarity.
    5. Final Object Score = Description Similarity * Average Attribute Similarity.
    6. Return the highest Final Object Score among all current objects (Hard Max).
    """
    # 1. Parse Goal Item
    goal_desc, goal_attr_names, goal_attr_values = _parse_object_item(goal_item)
    if not goal_desc:
        return 0.0

    # 2. Parse Current State Objects
    cur_objects = []
    for item in cur_state_list:
        c_desc, c_names, c_values = _parse_object_item(item)
        if c_desc:
            cur_objects.append({
                "desc": c_desc,
                "names": c_names,
                "values": c_values
            })
    
    if not cur_objects:
        return 0.0

    # 3. Pre-compute Goal Description Embedding
    goal_desc_emb = get_embedding_cached(sim_model, goal_desc, embedding_cache)

    max_object_score = 0.0

    # 4. Iterate over all current objects to find the max score
    for obj in cur_objects:
        # --- A. Description Match (Gatekeeper / Base Score) ---
        obj_desc_emb = get_embedding_cached(sim_model, obj["desc"], embedding_cache)
        desc_sim = compute_cosine_similarity(goal_desc_emb, obj_desc_emb)

        # --- B. Attribute Match ---
        if len(goal_attr_names) > 0:
            total_attr_score = 0.0
            
            for g_name, g_val in zip(goal_attr_names, goal_attr_values):
                g_name_emb = get_embedding_cached(sim_model, g_name, embedding_cache)
                g_val_emb = get_embedding_cached(sim_model, g_val, embedding_cache)

                best_attr_val_sim = 0.0
                max_key_sim = -1.0 
                
                # If object has attributes, find the best matching key
                if obj["names"]:
                    for c_name, c_val in zip(obj["names"], obj["values"]):
                        c_name_emb = get_embedding_cached(sim_model, c_name, embedding_cache)
                        key_sim = compute_cosine_similarity(g_name_emb, c_name_emb)

                        # Hard Max on Attribute Key
                        if key_sim > max_key_sim:
                            max_key_sim = key_sim
                            c_val_emb = get_embedding_cached(sim_model, c_val, embedding_cache)
                            best_attr_val_sim = compute_cosine_similarity(g_val_emb, c_val_emb)
                
                total_attr_score += best_attr_val_sim
            
            avg_attr_score = total_attr_score / len(goal_attr_names)
            
            # Combine Description and Attribute scores
            current_obj_final_score = desc_sim * avg_attr_score
        else:
            # If Goal has no attributes, description is the sole metric
            current_obj_final_score = desc_sim

        # Update global maximum score (Hard Max over objects)
        if current_obj_final_score > max_object_score:
            max_object_score = current_obj_final_score

    return max_object_score


def get_reward(
    sim_model: Any, 
    all_states: List[Any], 
    goal_states: List[Any]
) -> Dict[str, Any]:
    """
    Computes the trajectory reward by comparing state history against goal states.
    """

    predicted_rewards = []
    
    # Instantiate a single embedding cache per trajectory to drastically speed up inference
    trajectory_embedding_cache = {}

    for i, (state_input, goal_input) in enumerate(zip(all_states, goal_states)):
        
        # =========================================================================
        # 1. Normalize Current State Input
        # =========================================================================
        if isinstance(state_input, dict) and 'CurState' in state_input:
            cur_state_items = state_input['CurState']
        elif isinstance(state_input, list):
            cur_state_items = state_input
        elif isinstance(state_input, str):
            cur_state_items = [state_input]
        else:
            cur_state_items = [] 

        # =========================================================================
        # 2. Normalize Goal Input
        # =========================================================================
        if isinstance(goal_input, dict) and 'GoalState' in goal_input:
            goal_items = goal_input['GoalState']
        elif isinstance(goal_input, list):
            goal_items = goal_input
        elif isinstance(goal_input, str):
            goal_items = [goal_input]
        else:
            goal_items = []

        # --- Calculate Reward ---
        if not goal_items or not cur_state_items:
            predicted_rewards.append(0)
            continue

        step_total_score = 0.0
        for goal_item in goal_items:
            score = calculate_hard_matching_score(
                sim_model=sim_model, 
                cur_state_list=cur_state_items, 
                goal_item=goal_item, 
                embedding_cache=trajectory_embedding_cache
            )
            step_total_score += score
        
        # Average across multiple goal objects
        normalized_score = step_total_score / len(goal_items)
        
        # Clip to 0.0 - 1.0 float
        final_score = float(np.clip(normalized_score, 0.0, 1.0))
        truncated_score = int(final_score * 100) / 100.0

        predicted_rewards.append(truncated_score)

    return predicted_rewards