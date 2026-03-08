from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple

class BaseEnv(ABC):
    """
    Abstract Base Class for all environments.
    Enforces the implementation of state extraction and trajectory generation methods.
    """
    def __init__(
        self,
        **kwargs,
    ):
        pass
    
    @abstractmethod
    def get_gold_trajectory(self) -> List[Any]:
        """Retrieves the expert/gold trajectory for the task."""
        pass

    @abstractmethod
    def get_CurState(self, *args, **kwargs) -> Tuple[Dict, str]:
        """
        Extracts or Updates the Current State using LLM.
        Should return (state_data, thinking_process).
        """
        pass

    @abstractmethod
    def get_GoalState(self, *args, **kwargs) -> Tuple[Dict, str]:
        """
        Extracts the Goal State based on the current context.
        Should return (state_data, thinking_process).
        """
        pass

    @abstractmethod
    def generate_trajectory_data(self) -> Dict[str, Any]:
        """
        Main entry point to process the entire task trajectory and extract all states.
        """
        pass