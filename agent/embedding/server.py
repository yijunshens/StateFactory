"""
Embedding Server

A lightweight HTTP service that exposes embedding models via a REST API.
Supports multiple embedding backends (e.g., BGE, Qwen, Nomic, SBERT) and provides
a unified interface for text-to-vector conversion.

Endpoints:
- GET /health: Returns server and model status
- POST /encode: Accepts a list of sentences and returns their embeddings as JSON

Usage:
    python server.py --model_name bge-large-en --port 8011
"""

import argparse
import logging
import os
import sys
import numpy as np
from flask import Flask, request, jsonify

from models import (
    SBertEmbeddingModel,
    QwenEmbeddingModel,
    BGEEmbeddingModel,
    GemmaEmbeddingModel,
    NomicEmbeddingModel
)

# Configure root logger with structured format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [EmbeddingServer] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress Flask's default request logging to reduce noise
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)
model_instance = None


def load_model(model_name: str):
    """
    Initialize and load the specified embedding model into memory.

    This function supports multiple model families by name (case-insensitive).
    If the model name contains a known keyword (e.g., 'bge', 'qwen'), the corresponding
    wrapper class is instantiated. Otherwise, it defaults to SBertEmbeddingModel.

    Args:
        model_name (str): Identifier or path of the embedding model to load.

    Raises:
        SystemExit: If model loading fails due to missing dependencies, invalid paths,
                    or unsupported configurations.
    """
    global model_instance
    logger.info(f"Loading embedding model: {model_name}...")

    model_name_lower = model_name.lower()

    try:
        if "qwen" in model_name_lower:
            model_instance = QwenEmbeddingModel()
        elif "bge" in model_name_lower:
            model_instance = BGEEmbeddingModel()
        elif "gemma" in model_name_lower:
            model_instance = GemmaEmbeddingModel()
        elif "nomic" in model_name_lower:
            model_instance = NomicEmbeddingModel()
        elif "nli" in model_name_lower or "mpnet" in model_name_lower:
            model_instance = SBertEmbeddingModel(model_name_lower)
        else:
            # Fallback to Sentence-BERT if no specific backend is matched
            model_instance = SBertEmbeddingModel()

        logger.info(f"Successfully loaded model: '{model_name}'")
    except Exception as e:
        logger.error(f"Failed to load model '{model_name}': {e}")
        sys.exit(1)


@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint.

    Returns:
        JSON response with status:
        - 200 OK if model is loaded and ready
        - 503 Service Unavailable if model is still loading or failed to load
    """
    if model_instance is not None:
        return jsonify({
            "status": "ready",
            "model_type": model_instance.__class__.__name__
        }), 200
    return jsonify({"status": "loading"}), 503


@app.route('/encode', methods=['POST'])
def encode():
    """
    Generate embeddings for a batch of input sentences.

    Expects a JSON payload with a key "sentences" containing a list of strings.
    Returns a JSON array where each element is the embedding vector (as a list of floats)
    corresponding to the input sentence at the same index.

    Returns:
        - 200 OK with embeddings on success
        - 400 Bad Request if input is malformed
        - 500 Internal Server Error if encoding fails
        - 503 Service Unavailable if model is not ready

    Example request:
        POST /encode
        Content-Type: application/json
        {"sentences": ["Hello world", "How are you?"]}

    Example response:
        [[0.1, -0.3, ..., 0.8], [0.5, 0.2, ..., -0.1]]
    """
    if model_instance is None:
        return jsonify({"error": "Model not loaded"}), 503

    try:
        data = request.get_json()
        if not data or "sentences" not in data:
            return jsonify({"error": "Missing 'sentences' field in request body"}), 400

        sentences = data["sentences"]
        if not isinstance(sentences, list) or not all(isinstance(s, str) for s in sentences):
            return jsonify({"error": "'sentences' must be a list of strings"}), 400

        if not sentences:
            return jsonify([])

        # Dispatch to the appropriate encoding method
        if hasattr(model_instance, 'encode'):
            vectors = model_instance.encode(sentences)
        elif hasattr(model_instance, 'create_embedding'):
            vectors = model_instance.create_embedding(sentences)
        else:
            return jsonify({"error": "Model wrapper does not implement a supported encoding method"}), 500

        # Ensure output is JSON-serializable (convert NumPy arrays to native lists)
        if isinstance(vectors, np.ndarray):
            vectors = vectors.tolist()

        return jsonify(vectors)

    except Exception as e:
        logger.exception("Unexpected error during encoding")
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Start an HTTP server for generating text embeddings."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gemma",
        help="Name or path of the embedding model to load (e.g., 'BAAI/bge-large-en', 'qwen', 'nomic-embed-text')."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8013,
        help="Port number to bind the server to."
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Network interface to bind to (use '127.0.0.1' for local-only access)."
    )

    args = parser.parse_args()

    # Load the embedding model before starting the server
    load_model(args.model_name)

    # Start the Flask server
    logger.info(f"Starting embedding server at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)