import re
import json
import logging
from typing import Tuple, List, Dict, Any, Optional
from tqdm import tqdm
import yaml

from agent.prompts.template import (
    PROMPT_CUR_STATE_TEMPLATE, 
    PROMPT_TASK_RELATED_TEMPLATE, 
    PROMPT_GOAL_STATE_TEMPLATE,
)
from agent.envs.base import BaseEnv
from agent.prompts.format import (
    TEXTUAL, OBJ_CENTRIC, OBJ_ATTRIBUTE, 
    TEXTUAL_DES, OBJ_CENTRIC_DES, OBJ_ATTRIBUTE_DES
)

logger = logging.getLogger(__name__)

class SciWorldEnv(BaseEnv):
    """
    Environment wrapper for ScienceWorld tasks.
    Mimics the AlfWorldEnv structure and LLM parsing logic.
    """
    def __init__(self, task: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self.task = task
        self.env = None 
        self.agent = kwargs.get("agent")
        self.output_format = kwargs.get("output_format", "OBJ_ATTRIBUTE")
        
        self._format_map = {
            'TEXTUAL': (TEXTUAL, TEXTUAL_DES),
            'OBJ_CENTRIC': (OBJ_CENTRIC, OBJ_CENTRIC_DES),
            'OBJ_ATTRIBUTE': (OBJ_ATTRIBUTE, OBJ_ATTRIBUTE_DES),
        }

    def get_gold_trajectory(self):
        return self.task.get('expert_plan', [])

    # =========================================================================
    # LLM Parsing Helpers (Maintained identically to AlfWorld for consistency)
    # =========================================================================

    def _get_format_config(self) -> Tuple[Any, str]:
        if self.output_format not in self._format_map:
            raise ValueError(f"Unsupported output format: {self.output_format}")
        return self._format_map[self.output_format]

    def _extract_json_from_text(self, text: str) -> str:
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match: return match.group(1)
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match: return match.group(1)
        return text
    
    def _extract_yaml_from_text(self, text: str) -> str:
        match = re.search(r"```yaml\s*(---\s*\n.*?)(\n```|$)", text, re.DOTALL)
        if match: return match.group(1)
        match = re.search(r"(---\s*\n.*)", text, re.DOTALL)
        if match: return match.group(1)
        return text

    def _maybe_parse_structured_str(self, value):
        if not isinstance(value, str): 
            return value
        
        stripped = value.strip()
        if not stripped: 
            return value

        if (stripped.startswith("{") and stripped.endswith("}")) or \
           (stripped.startswith("[") and stripped.endswith("]")):
            try: 
                return json.loads(stripped)
            except Exception: 
                pass 

        if stripped.startswith("- ") or ":" in stripped or "\n" in stripped:
            try:
                parsed = yaml.safe_load(stripped)
                if parsed is not None: 
                    return parsed
            except Exception: 
                pass
            
        return value

    def _call_llm_and_parse(self, prompt: str, max_retries: int = 5):
        for attempt in range(max_retries):
            try:
                text_output, agent_thinking = self.agent.chat(prompt)
                extracted_thinking = ""
                parsed_data = None 
                
                # Attempt to parse the output as JSON first.
                try:
                    cleaned_json_str = self._extract_json_from_text(text_output)
                    parsed_data = json.loads(cleaned_json_str)
                except json.JSONDecodeError:
                    # If JSON parsing fails, fall back to YAML parsing.
                    try:
                        cleaned_yaml_str = self._extract_yaml_from_text(text_output)
                        parsed_data = yaml.safe_load(cleaned_yaml_str)
                    except Exception: 
                        pass

                if parsed_data is None:
                    raise ValueError("Failed to parse both JSON and YAML.")

                # Extract the thinking process and automatically unwrap the payload.
                if isinstance(parsed_data, dict):
                    if "_thinking" in parsed_data:
                        extracted_thinking = parsed_data.pop("_thinking")
                    elif "thinking" in parsed_data:
                        extracted_thinking = parsed_data.pop("thinking")
                    
                    keys = list(parsed_data.keys())
                    if len(keys) == 1:
                        parsed_data = parsed_data[keys[0]]
                
                # Recursively parse internal nested structures.
                if isinstance(parsed_data, dict):
                    parsed_data = {k: self._maybe_parse_structured_str(v) for k, v in parsed_data.items()}
                elif isinstance(parsed_data, list):
                    parsed_data = [self._maybe_parse_structured_str(i) for i in parsed_data]

                final_thinking = f"{agent_thinking}\n[Analysis]: {extracted_thinking}".strip()
                return parsed_data, final_thinking

            except Exception as e:
                logger.warning(f"LLM Attempt {attempt + 1}/{max_retries} failed: {e}")
        
        raise RuntimeError(f"All {max_retries} LLM attempts failed.")

    def _get_fmt_schema(self, key: str) -> Dict:
        structure, _ = self._get_format_config()
        return {"_thinking": "Step-by-step analysis...", key: structure}

    # =========================================================================
    # Core State Extraction Logic (get_CurState & get_GoalState)
    # =========================================================================

    def get_CurState(self, action, observation, prev_states, goal_description, prev_goal_state) -> Tuple[Dict, str]:
        _, format_des = self._get_format_config()
        str_prev_states = json.dumps(prev_states, ensure_ascii=False) if prev_states else "None"
        str_prev_goal = json.dumps(prev_goal_state, ensure_ascii=False) if prev_goal_state else "None"

        # Phase 1: State Recall
        prompt_recall = PROMPT_CUR_STATE_TEMPLATE.format(
            system_instruction="",
            output_format=self._get_fmt_schema("Current State"),
            output_format_des=format_des,
            goal_description=goal_description,
            prev_goal_state=str_prev_goal,
            last_action=action,
            observation=observation,
            prev_states=str_prev_states
        )
        recall_data, think_recall = self._call_llm_and_parse(prompt_recall)

        # Phase 2: State Update/Evolution
        prompt_update = PROMPT_TASK_RELATED_TEMPLATE.format(
            system_instruction="",
            output_format=self._get_fmt_schema("Previous States"),
            output_format_des=format_des,
            goal_description=goal_description,
            prev_states=str_prev_states,
            current_state=json.dumps(recall_data, ensure_ascii=False),
            action_history=action
        )
        evolved_state, think_update = self._call_llm_and_parse(prompt_update)
        
        combined_thinking = f"--- RECALL ---\n{think_recall}\n\n--- UPDATE ---\n{think_update}"
        return evolved_state, combined_thinking

    def get_GoalState(self, current_state, observation, action_obj, goal_description, prev_goal_state) -> Tuple[Dict, str]:
        _, format_des = self._get_format_config()
        str_prev_goal = json.dumps(prev_goal_state, ensure_ascii=False) if prev_goal_state else "None"
        
        prompt_goal = PROMPT_GOAL_STATE_TEMPLATE.format(
            system_instruction="",
            output_format=self._get_fmt_schema("Goal State"),
            output_format_des=format_des,
            goal_description=goal_description,
            current_state=json.dumps(current_state, ensure_ascii=False),
            observation=observation,
            action_history=json.dumps(action_obj, ensure_ascii=False),
            prev_goal_state=str_prev_goal
        )
        goal_data, think_goal = self._call_llm_and_parse(prompt_goal)
        return goal_data, think_goal

    # =========================================================================
    # Trajectory Generation Pipeline
    # =========================================================================

    def generate_trajectory_data(self) -> Dict[str, Any]:
        goal_description = self.task.get('goal_description', '')
        observations = self.task.get('obs', [])
        gold_actions = self.task.get('expert_plan', [])
        
        all_states_output = [] 
        thinkings_output = []
        
        running_accumulated_state = None 
        running_goal_state = None
        
        for i in tqdm(range(len(gold_actions)), desc="Processing SciWorld"):
            action = gold_actions[i]
            
            # Handle SciWorld observations: extract the string from a nested list format (e.g., [["string"]]) or fallback.
            obs_item = observations[i]
            obs = obs_item[0] if isinstance(obs_item, list) and obs_item else str(obs_item)
            
            # Step 1: Evolve the current state based on the action and observation.
            cur_state_data, thinking_cur = self.get_CurState(
                action=action, observation=obs,
                prev_states=running_accumulated_state,
                goal_description=goal_description, prev_goal_state=running_goal_state
            )

            # Step 2: Extract the goal state using the newly evolved state.
            action_obj = {"step": i, "action": action}
            goal_state_data, thinking_goal = self.get_GoalState(
                current_state=cur_state_data, observation=obs,
                action_obj=action_obj, goal_description=goal_description,
                prev_goal_state=running_goal_state
            )
            
            # Update the accumulated history for the next iteration.
            running_accumulated_state = cur_state_data
            running_goal_state = goal_state_data
            
            all_states_output.append({
                "Step": i,
                "Action": action,
                "Observation": obs,
                "CurState": cur_state_data,
                "GoalState": goal_state_data
            })
            thinkings_output.append({
                "cur_state": thinking_cur,
                "goal_state": thinking_goal
            })

        return {
            "task_id": self.task.get('task_id', 'unknown'),
            "goal_description": self.task.get('goal_description', 'unknown'),
            "trajectory": all_states_output,
            "thinkings": thinkings_output,
            "metadata": {
                "goal_description": goal_description,
                "actions": gold_actions,
                "sparse_gt": self.task.get('sparse_gt', []),
                "shaped_gt": self.task.get('shaped_gt', []),
            }
        }