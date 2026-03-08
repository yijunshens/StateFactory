"""
Remote Embedding Client

A lightweight client for interacting with a standalone embedding server via HTTP.
This class provides an interface compatible with local embedding model wrappers
(e.g., those implementing `encode()`), enabling seamless substitution between
local and remote inference backends.
"""

import requests
import numpy as np
from typing import List, Union, Optional


class RemoteEmbeddingModel:
    """
    HTTP client for a remote embedding service.

    This class forwards text encoding requests to a running embedding server
    (e.g., one implemented with Flask) and returns embeddings as NumPy arrays.
    It mimics the API of local embedding wrappers to support drop-in replacement.

    The client performs a basic health check on initialization but does not
    halt execution if the server is unreachable, allowing for lazy connection
    or retry strategies in higher-level code.
    """

    def __init__(self, port: int, host: str = "localhost"):
        """
        Initialize the remote embedding client.

        Args:
            port (int): Port number of the embedding server.
            host (str): Hostname or IP address of the server. Defaults to "localhost".

        Raises:
            None: Connection errors are logged but not raised during initialization
                  to support deferred connectivity.
        """
        self.api_url = f"http://{host}:{port}/encode"
        self.health_url = f"http://{host}:{port}/health"

        # Perform a non-blocking health check at init time
        try:
            response = requests.get(self.health_url, timeout=2.0)
            response.raise_for_status()
        except requests.RequestException as e:
            # Log warning but do not fail—server may start later or be temporarily down
            print(f"[RemoteEmbeddingModel] Warning: Failed to connect to embedding server "
                  f"at {host}:{port} during initialization: {e}. "
                  "Ensure the embedding server is running before sending requests.")

    def encode(self, sentences: Union[str, List[str]], **kwargs) -> np.ndarray:
        """
        Encode input text(s) by delegating to a remote embedding server.

        This method supports both single strings and lists of strings, returning
        embeddings in a format consistent with local model wrappers:
          - For a single string: returns a 1D NumPy array of shape `(dim,)`.
          - For a list of strings: returns a 2D NumPy array of shape `(n, dim)`.

        Note: Additional keyword arguments (`**kwargs`) are accepted for API compatibility
        with local models but are currently ignored.

        Args:
            sentences: Input text as a single string or a list of strings.

        Returns:
            A NumPy array containing L2-normalized embeddings.
            In case of a server error, a zero vector (or matrix) is returned as fallback
            to prevent caller crashes. The fallback dimension is minimal (1D) and should
            not be relied upon—callers should handle errors proactively.

        Raises:
            None: All exceptions are caught internally, and a fallback zero array is returned.
        """
        # Normalize input to list
        was_single = isinstance(sentences, str)
        if was_single:
            sentences = [sentences]

        payload = {"sentences": sentences}

        try:
            # Use a generous timeout to accommodate large batches or slow models
            response = requests.post(self.api_url, json=payload, timeout=60.0)
            response.raise_for_status()

            embeddings = response.json()
            embeddings_array = np.array(embeddings, dtype=np.float32)

            # Restore original shape if input was a single sentence
            if was_single:
                return embeddings_array[0]
            return embeddings_array

        except Exception as e:
            # Fallback: return zero vectors to avoid breaking upstream logic
            # Note: Dimension is unknown, so we use a placeholder (1D per sentence)
            # In production systems, consider raising an exception or using a known dim.
            fallback_shape = (1,) if was_single else (len(sentences), 1)
            fallback = np.zeros(fallback_shape, dtype=np.float32)

            print(f"[RemoteEmbeddingModel] Error during encoding: {e}. "
                  f"Returning zero embedding(s) of shape {fallback.shape} as fallback.")

            return fallback