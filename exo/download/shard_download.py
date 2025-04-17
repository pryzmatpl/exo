from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, AsyncIterator, Union
from pathlib import Path
from exo.inference.shard import Shard
from exo.download.download_progress import RepoProgressEvent
from exo.helpers import AsyncCallbackSystem
from exo.models import get_repo
import os
import logging
from huggingface_hub import snapshot_download


class ShardDownloader(ABC):
  @abstractmethod
  async def ensure_shard(self, shard: Shard, inference_engine_name: str) -> Path:
    """
        Ensure that a shard is available locally, either by finding it in the local models directory,
        or by downloading it from HuggingFace.

        Args:
            shard (Shard): The shard to ensure.
            inference_engine_name (str): The name of the inference engine.

        Returns:
            Path: The path to the local model directory.
        """
    pass

  @property
  @abstractmethod
  def on_progress(self) -> AsyncCallbackSystem[str, Tuple[Shard, RepoProgressEvent]]:
    pass

  @abstractmethod
  async def get_shard_download_status(self, inference_engine_name: str) -> AsyncIterator[tuple[Path, RepoProgressEvent]]:
    """Get the download status of shards.
    
    Returns:
        Optional[Dict[str, float]]: A dictionary mapping shard IDs to their download percentage (0-100),
        or None if status cannot be determined
    """
    pass


class NoopShardDownloader(ShardDownloader):
  async def ensure_shard(self, shard: Shard, inference_engine_name: str) -> Path:
    return Path("/tmp/noop_shard")

  @property
  def on_progress(self) -> AsyncCallbackSystem[str, Tuple[Shard, RepoProgressEvent]]:
    return AsyncCallbackSystem()

  async def get_shard_download_status(self, inference_engine_name: str) -> AsyncIterator[tuple[Path, RepoProgressEvent]]:
    if False: yield


class LocalModelDownloader(ShardDownloader):
    """
    A class for downloading and managing model shards.
    Supports both HuggingFace model repository downloads and local model loading.
    """
    def __init__(self, cache_dir: str = None):
        """
        Initialize the shard downloader.

        Args:
            cache_dir (str, optional): The directory to cache downloaded models.
        """
        self.cache_dir = cache_dir
        if cache_dir is None:
            self.cache_dir = os.path.expanduser("~/.cache/exo")
        os.makedirs(self.cache_dir, exist_ok=True)

    async def ensure_shard(self, shard: Shard, engine_name: str) -> Path:
        """
        Ensure that a shard is available locally, either by finding it in the local models directory,
        or by downloading it from HuggingFace.

        Args:
            shard (Shard): The shard to ensure.
            engine_name (str): The name of the inference engine.

        Returns:
            Path: The path to the local model directory.
        """
        repo_id = get_repo(shard.model_id, engine_name)
        
        # Check if this is a local path (starting with /)
        if repo_id and repo_id.startswith('/'):
            # This is a local model path
            model_path = Path(repo_id)
            if model_path.exists():
                logging.info(f"Using local model at {model_path}")
                return model_path
            else:
                raise FileNotFoundError(f"Local model not found at {model_path}")
        
        # If not a local path or not found, proceed with regular HuggingFace download
        if repo_id is None:
            raise ValueError(f"Model {shard.model_id} does not have a repo for {engine_name}")
        
        # Use regular HuggingFace download path
        try:
            # Download the model from HuggingFace
            logging.info(f"Downloading model {repo_id} from HuggingFace")
            model_path = snapshot_download(
                repo_id=repo_id,
                cache_dir=self.cache_dir,
                local_files_only=os.environ.get("HF_HUB_OFFLINE", "0") == "1",
            )
            return Path(model_path)
        except Exception as e:
            # If download fails, try to find the model in standard locations
            logging.warning(f"Failed to download model {repo_id}: {e}")
            
            # Check standard model locations
            model_locations = [
                Path(os.path.expanduser("~/models")) / shard.model_id,  # ~/models/model_id
                Path("/models") / shard.model_id,  # /models/model_id
                Path("./models") / shard.model_id,  # ./models/model_id
                Path("../models") / shard.model_id,  # ../models/model_id
                Path("../../models") / shard.model_id,  # ../../models/model_id
            ]
            
            for loc in model_locations:
                if loc.exists():
                    logging.info(f"Found model at {loc}")
                    return loc
            
            # If we get here, we couldn't find the model
            raise FileNotFoundError(f"Model {shard.model_id} not found locally and could not be downloaded")

  @property
  def on_progress(self) -> AsyncCallbackSystem[str, Tuple[Shard, RepoProgressEvent]]:
    return AsyncCallbackSystem()

  async def get_shard_download_status(self, inference_engine_name: str) -> AsyncIterator[tuple[Path, RepoProgressEvent]]:
    if False: yield
