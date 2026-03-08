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

class TextWorldEnv(BaseEnv):
    """
    Environment wrapper for TextWorld tasks.
    Handles interaction parsing, state tracking, and trajectory generation.
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
        """Retrieves the expert plan (gold trajectory) for the current task."""
        return self.task.get('expert_plan', [])

    # =========================================================================
    # LLM Parsing Helpers (Aligned with AlfWorld implementation)
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
        
        # Attempt to parse as a JSON dictionary or list
        if (stripped.startswith("{") and stripped.endswith("}")) or \
           (stripped.startswith("[") and stripped.endswith("]")):
            try: 
                return json.loads(stripped)
            except Exception: 
                pass 
        
        # Attempt to parse as YAML (indicated by a list dash, colon, or newline)
        if stripped.startswith("- ") or ":" in stripped or "\n" in stripped:
            try:
                parsed = yaml.safe_load(stripped)
                if parsed is not None: 
                    return parsed
            except Exception: 
                pass
            
        return value

    def _call_llm_and_parse(self, prompt: str, max_retries: int = 5) -> Tuple[Any, str]:
        """
        Calls the LLM agent and parses the structured response.
        Incorporates automated fallbacks, thinking extraction, and payload unwrapping.
        """
        for attempt in range(max_retries):
            try:
                text_output, agent_thinking = self.agent.chat(prompt)
                extracted_thinking = ""
                parsed_data = None 
                
                # Step 1: Attempt to parse the payload as JSON
                try:
                    cleaned_json_str = self._extract_json_from_text(text_output)
                    parsed_data = json.loads(cleaned_json_str)
                except json.JSONDecodeError:
                    # Fallback to YAML parsing if JSON decoding fails
                    try:
                        cleaned_yaml_str = self._extract_yaml_from_text(text_output)
                        parsed_data = yaml.safe_load(cleaned_yaml_str)
                    except Exception: 
                        pass

                if parsed_data is None:
                    raise ValueError("Failed to parse both JSON and YAML.")

                # Step 2: Intelligently unwrap the payload and extract internal reasoning
                if isinstance(parsed_data, dict):
                    # Extract the thinking process
                    for t_key in ["_thinking", "thinking"]:
                        if t_key in parsed_data:
                            extracted_thinking = parsed_data.pop(t_key)
                    
                    # Auto-unwrap if only a single substantive key remains (e.g., "Current State")
                    keys = list(parsed_data.keys())
                    if len(keys) == 1:
                        parsed_data = parsed_data[keys[0]]
                
                # Step 3: Recursively parse internal nested structures represented as strings
                if isinstance(parsed_data, dict):
                    parsed_data = {k: self._maybe_parse_structured_str(v) for k, v in parsed_data.items()}
                elif isinstance(parsed_data, list):
                    parsed_data = [self._maybe_parse_structured_str(i) for i in parsed_data]

                final_thinking = f"{agent_thinking}\n[Analysis]: {extracted_thinking}".strip()
                return parsed_data, final_thinking

            except Exception as e:
                logger.warning(f"TextWorld LLM Attempt {attempt + 1} failed: {e}")
                
        raise RuntimeError(f"All {max_retries} LLM attempts failed.")

    def _get_fmt_schema(self, key: str) -> Dict:
        """Constructs a schema dictionary wrapping the expected output structure."""
        structure, _ = self._get_format_config()
        return {"_thinking": "Analysis...", key: structure}

    # =========================================================================
    # Core State Extraction Logic
    # =========================================================================

    def get_CurState(self, action, observation, prev_states, goal_description, prev_goal_state) -> Tuple[Any, str]:
        """
        Executes a two-phase process (Recall and Evolve) to determine the current state.
        Returns the evolved state and the combined reasoning history.
        """
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
        
        return evolved_state, f"--- RECALL ---\n{think_recall}\n\n--- UPDATE ---\n{think_update}"

    def get_GoalState(self, current_state, observation, action_obj, goal_description, prev_goal_state) -> Tuple[Any, str]:
        """
        Derives the next objective state based on the current context and recent action.
        """
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
        return self._call_llm_and_parse(prompt_goal)

    def process_single_step(self, action, observation, accumulated_state, goal_description, prev_goal, step_idx):
        """
        Orchestrates the state extraction pipeline for a single interaction step.
        """
        cur_state_data, thinking_cur = self.get_CurState(
            action, observation, accumulated_state, goal_description, prev_goal
        )
        action_obj = {"step": step_idx, "action": action}
        goal_state_data, thinking_goal = self.get_GoalState(
            cur_state_data, observation, action_obj, goal_description, prev_goal
        )
        return {
            "cur_state": cur_state_data,
            "goal_state": goal_state_data,
            "thinking": {"cur_state": thinking_cur, "goal_state": thinking_goal}
        }

    # =========================================================================
    # Trajectory Generation Pipeline
    # =========================================================================

    def generate_trajectory_data(self) -> Dict[str, Any]:
        """
        Iterates through the expert plan to generate state tracking and reasoning 
        trajectories for the entire task execution.
        """
        goal_description = self.task['goal_description']
        observations = self.task['obs']
        gold_actions = self.task['expert_plan']
        
        all_states_output = [] 
        thinkings_output = []
        running_accumulated_state = None 
        running_goal_state = None
        
        for i in tqdm(range(len(gold_actions)), desc=f"TextWorld Task {self.task.get('task_id')}"):
            action = gold_actions[i]
            
            # Handle potential list nesting inside observations
            obs = observations[i][0] if isinstance(observations[i], list) else observations[i]
            
            step_res = self.process_single_step(
                action, obs, running_accumulated_state, goal_description, running_goal_state, i
            )
            
            # Update the accumulated state and goal history for the next sequence iteration
            running_accumulated_state = step_res['cur_state']
            running_goal_state = step_res['goal_state']
            
            all_states_output.append({
                "Step": i,
                "Action": action,
                "Observation": obs,
                "CurState": running_accumulated_state,
                "GoalState": running_goal_state
            })
            thinkings_output.append(step_res['thinking'])

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