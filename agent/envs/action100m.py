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

class Action100mEnv(BaseEnv):
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

    # --- Helper Methods ---
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

        # Attempt to parse as a JSON list or dictionary
        if (stripped.startswith("{") and stripped.endswith("}")) or \
           (stripped.startswith("[") and stripped.endswith("]")):
            try:
                return json.loads(stripped)
            except Exception:
                pass 

        # Attempt to parse as YAML
        if stripped.startswith("- ") or ":" in stripped or "\n" in stripped:
            try:
                parsed = yaml.safe_load(stripped)
                if parsed is not None:
                    return parsed
            except Exception:
                pass

        # Retain original string
        return value

    def _call_llm_and_parse(self, prompt: str, max_retries: int = 5):
        for attempt in range(max_retries):
            try:
                text_output, agent_thinking = self.agent.chat(prompt)
                
                extracted_thinking = ""
                parsed_data = None 
                
                # Prioritize JSON, fallback to YAML
                try:
                    cleaned_json_str = self._extract_json_from_text(text_output)
                    parsed_data = json.loads(cleaned_json_str)
                except json.JSONDecodeError:
                    try:
                        cleaned_yaml_str = self._extract_yaml_from_text(text_output)
                        parsed_data = yaml.safe_load(cleaned_yaml_str)
                    except Exception:
                        pass 

                if parsed_data is None:
                    raise ValueError("Failed to parse both JSON and YAML.")

                # Intelligently strip the '_thinking' key and auto-unwrap the payload
                if isinstance(parsed_data, dict):
                    if "_thinking" in parsed_data:
                        extracted_thinking = parsed_data.pop("_thinking")
                    elif "thinking" in parsed_data:
                        extracted_thinking = parsed_data.pop("thinking")
                    
                    keys = list(parsed_data.keys())
                    if len(keys) == 1:
                        parsed_data = parsed_data[keys[0]]
                
                # Recursively parse internal structures
                if isinstance(parsed_data, dict):
                    parsed_data = {
                        k: self._maybe_parse_structured_str(v)
                        for k, v in parsed_data.items()
                    }
                elif isinstance(parsed_data, list):
                    parsed_data = [self._maybe_parse_structured_str(i) for i in parsed_data]

                final_thinking = f"{agent_thinking}\n[Analysis]: {extracted_thinking}".strip()
                return parsed_data, final_thinking

            except Exception as e:
                error_msg = f"LLM Attempt {attempt + 1}/{max_retries} failed: {e}"
                if hasattr(self, 'logger'):
                    self.logger.warning(error_msg)
                else:
                    print(error_msg)
        
        raise RuntimeError(f"All {max_retries} LLM attempts failed.")

    def _get_fmt_schema(self, key: str) -> str:
        structure, _ = self._get_format_config()
        wrapper_schema = {
            "_thinking": "Step-by-step analysis of the state changes...",
            key: structure
        }
        return wrapper_schema

    # =========================================================================
    # CORE IMPLEMENTATION (get_CurState & get_GoalState)
    # =========================================================================

    def get_CurState(self, action, observation, prev_states, goal_description, prev_goal_state) -> Tuple[Dict, str]:
        _, format_des = self._get_format_config()
        str_prev_states = json.dumps(prev_states, ensure_ascii=False) if prev_states else "None"
        str_prev_goal = json.dumps(prev_goal_state, ensure_ascii=False) if prev_goal_state else "None"

        # 1. Recall Phase
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

        # 2. Evolve Phase
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

    def process_single_step(self, action, observation, accumulated_state, goal_description, prev_goal, step_idx):
        try:
            cur_state_data, thinking_cur = self.get_CurState(
                action=action,
                observation=observation,
                prev_states=accumulated_state,
                goal_description=goal_description,
                prev_goal_state=prev_goal
            )

            action_obj = {"step": step_idx, "action": action}
            goal_state_data, thinking_goal = self.get_GoalState(
                current_state=cur_state_data,
                observation=observation,
                action_obj=action_obj,
                goal_description=goal_description,
                prev_goal_state=prev_goal
            )

            return {
                "cur_state": cur_state_data,
                "goal_state": goal_state_data,
                "thinking": {
                    "cur_state": thinking_cur,
                    "goal_state": thinking_goal
                }
            }

        except Exception as e:
            logger.error(f"Error in step {step_idx}: {e}")
            raise e

    def generate_trajectory_data(self) -> Dict[str, Any]:
        goal_description = self.task['goal_description']
        observations = self.task['obs']
        gold_actions = self.task['expert_plan']
        
        # New: Extract start and end times
        starts = self.task.get('starts', [])
        ends = self.task.get('ends', [])
        
        all_states_output = [] 
        thinkings_output = []
        
        running_accumulated_state = None 
        running_goal_state = None
        
        for i in tqdm(range(len(gold_actions)), desc="Processing Trajectory"):
            action = gold_actions[i]
            obs = observations[i] # Fixed: Action100M obs is a flat list
            
            # Safe indexing for timestamps
            start_time = starts[i] if i < len(starts) else 0.0
            end_time = ends[i] if i < len(ends) else 0.0
            
            step_res = self.process_single_step(
                action=action,
                observation=obs,
                accumulated_state=running_accumulated_state,
                goal_description=goal_description,
                prev_goal=running_goal_state,
                step_idx=i
            )
            
            raw_cur = step_res['cur_state']
            raw_goal = step_res['goal_state']
            
            running_accumulated_state = raw_cur
            running_goal_state = raw_goal
            
            # Formatted with explicit start/end inclusion
            final_obj = {
                "Step": i,
                "Action": action,
                "Observation": obs,
                "Start": start_time,
                "End": end_time,
                "CurState": raw_cur,
                "GoalState": raw_goal
            }
            
            all_states_output.append(final_obj)
            thinkings_output.append(step_res['thinking'])

        return {
            "task_id": self.task.get('task_id', 'unknown'),
            "goal_description": self.task.get('goal_description', 'unknown'),
            "trajectory": all_states_output,
            "thinkings": thinkings_output,
            "metadata": {
                "goal_description": goal_description,
                "actions": gold_actions,
                "starts": starts, # Pass through temporal metadata
                "ends": ends,
                "sparse_gt": self.task.get('sparse_gt', []),
                "shaped_gt": self.task.get('shaped_gt', []),
            }
        }