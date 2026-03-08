import logging
import os
import re  # [New] Needed for parsing <think> tags
import json # [New] Needed for manual JSON parsing
from typing import Dict, List, Union, Optional, Any, Tuple
from xml.etree.ElementInclude import include

import backoff
from openai import OpenAI, APIConnectionError, RateLimitError, APIStatusError

# =============================================================================
# Logging Configuration
# =============================================================================
logger = logging.getLogger(__name__)

# Mute noisy logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

class OpenAILLM:
    """
    A robust wrapper for the OpenAI Chat Completion API (v1.0+).
    Supports extracting 'thinking' process from both API fields and <think> tags.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the OpenAILLM agent.
        """
        self.config = config
        self.model_name = config.get("model_name", "gpt-3.5-turbo")
        
        self.client = OpenAI(
            base_url=config.get('api_base'), 
            api_key=config.get('api_key', "EMPTY"), 
            timeout=config.get('timeout', 1200.0)
        )

    def _extract_thinking(self, content: str, message_obj: Any) -> Tuple[str, Optional[str]]:
        """
        Helper method to extract reasoning/thinking from the response.
        Strategy 1: Check native 'reasoning_content' field (DeepSeek API style).
        Strategy 2: Parse <think>...</think> tags (Open Weights/vLLM style).
        
        Returns:
            Tuple[cleaned_content, reasoning_string]
        """
        reasoning = getattr(message_obj, 'reasoning_content', None)
        
        # If native field is empty, try extracting from tags
        if not reasoning and content:
            # Regex to find content inside <think> tags (DOTALL matches newlines)
            think_pattern = r"<think>(.*?)</think>"
            match = re.search(think_pattern, content, flags=re.DOTALL)
            
            if match:
                reasoning = match.group(1).strip()
                # Remove the thinking part from the main content
                content = re.sub(think_pattern, "", content, flags=re.DOTALL).strip()
        
        return content, reasoning

    @backoff.on_exception(
        backoff.expo,
        (APIConnectionError, RateLimitError, APIStatusError),
        max_tries=5,
        giveup=lambda e: getattr(e, 'status_code', 500) in [400, 401, 403]
    )
    def chat(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Generates a response. 
        Note: Removed 'json_object' enforcement to allow model to 'think' freely first.
        """
        try:
            # Recommendation: Do NOT force json_object for Reasoning models.
            # Reasoning models usually output <think>...</think> FIRST, then JSON.
            # Forcing JSON mode often breaks the thinking process.
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Please output the final answer in JSON format."},
                {"role": "user", "content": query}
            ]   

            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.config.get('temperature', 0.01), 
                response_format={"type": "json_object"}, 
                reasoning_effort = 'medium',
                stream=False
            )
            message = completion.choices[0].message
            raw_content = message.content
            # --- Extract Thinking & Clean Content ---
            final_content, reasoning = self._extract_thinking(raw_content, message)

            return final_content, reasoning
            
        except Exception as e:
            logger.error(f"Failed to generate response for model {self.model_name}: {e}")
            raise e
