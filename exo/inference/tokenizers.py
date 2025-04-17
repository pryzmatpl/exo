import traceback
from os import PathLike
from aiofiles import os as aios
from typing import Union, Optional, Dict
from transformers import AutoTokenizer, AutoProcessor
import numpy as np
from exo.helpers import DEBUG
from exo.download.new_shard_download import ensure_downloads_dir, download_file_with_retry, ensure_exo_tmp
import json


class DummyTokenizer:
  def __init__(self):
    self.eos_token_id = 69
    self.vocab_size = 1000

  def apply_chat_template(self, conversation, tokenize=True, add_generation_prompt=True, tools=None, **kwargs):
    return "dummy_tokenized_prompt"

  def encode(self, text):
    return np.array([1])

  def decode(self, tokens):
    return "dummy" * len(tokens)


_tokenizer_cache: Dict[str, AutoTokenizer] = {}

def _get_model_name_for_local_path(path: str) -> Optional[str]:
    """Map local paths to their corresponding model names for tokenizer loading."""
    path_to_model = {
        "/models/phi4": "microsoft/phi-2",  # phi-4 uses the same tokenizer as phi-2
    }
    return path_to_model.get(path)

async def _resolve_tokenizer(repo_id_or_local_path: str) -> AutoTokenizer:
    """Resolve a tokenizer from either a HuggingFace repo ID or a local path."""
    if repo_id_or_local_path in _tokenizer_cache:
        if DEBUG >= 2: print(f"Using cached tokenizer for {repo_id_or_local_path}")
        return _tokenizer_cache[repo_id_or_local_path]

    # Handle local paths
    if repo_id_or_local_path.startswith("/") or repo_id_or_local_path.startswith("./") or repo_id_or_local_path.startswith("../"):
        model_name = _get_model_name_for_local_path(repo_id_or_local_path)
        if model_name:
            if DEBUG >= 1: print(f"Loading tokenizer for local model {repo_id_or_local_path} using {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            _tokenizer_cache[repo_id_or_local_path] = tokenizer
            return tokenizer
        raise ValueError(f"Unsupported local model path: {repo_id_or_local_path}. Add mapping in _get_model_name_for_local_path.")

    # Handle HuggingFace repos
    try:
        if DEBUG >= 2: print(f"Loading tokenizer for {repo_id_or_local_path}")
        target_dir = (await ensure_exo_tmp())/repo_id_or_local_path.replace("/", "--")
        config_file = await download_file_with_retry(repo_id_or_local_path, "main", "config.json", target_dir)
        
        async with aios.open(config_file, 'r') as f:
            config = json.loads(await f.read())
            
        # Use the model_type from config to determine tokenizer
        model_type = config.get("model_type", "").lower()
        if model_type in ["llama", "mistral"]:
            tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
        else:
            tokenizer = AutoTokenizer.from_pretrained(repo_id_or_local_path)
            
        _tokenizer_cache[repo_id_or_local_path] = tokenizer
        return tokenizer
    except Exception as e:
        if DEBUG >= 1: print(f"Error loading tokenizer for {repo_id_or_local_path}: {e}")
        raise ValueError(f"Failed to load tokenizer for {repo_id_or_local_path}: {e}")

async def resolve_tokenizer(repo_id: Optional[str]) -> AutoTokenizer:
    """Public interface to resolve a tokenizer."""
    if not repo_id:
        raise ValueError("No repo ID provided")
    return await _resolve_tokenizer(repo_id)
