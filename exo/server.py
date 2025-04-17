from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
import numpy as np
import os
from pathlib import Path
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the parent directory to sys.path to allow importing exo modules
# Assumes server.py is in exo/exo/
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
logger.info(f"Adding {project_root} to sys.path")
logger.info(f"Current sys.path: {sys.path}")

try:
    from exo.inference.tinygrad.inference import TinygradDynamicShardInferenceEngine
    from exo.inference.shard import Shard
    from exo.models import build_full_shard, get_repo
    # --- Use the default downloader --- 
    # The default downloader now handles local paths thanks to our modification
    from exo.download.new_shard_download import new_shard_downloader
    logger.info("Successfully imported exo modules and default shard downloader.")
except ImportError as e:
    logger.error(f"Failed to import exo modules: {e}")
    logger.error("Ensure the server is run from the correct directory or PYTHONPATH is set.")
    sys.exit(1)

# Configuration
MODEL_ID = "phi-4"
ENGINE_CLASS_NAME = "TinygradDynamicShardInferenceEngine"

# --- Initialize Engine with Default Downloader ---
shard_downloader = new_shard_downloader() # Use the default factory
inference_engine = TinygradDynamicShardInferenceEngine(shard_downloader)
# Build the specific shard for phi-4 using tinygrad engine
phi4_shard = build_full_shard(MODEL_ID, ENGINE_CLASS_NAME)

if not phi4_shard:
    logger.error(f"Could not build shard for model {MODEL_ID} using {ENGINE_CLASS_NAME}. Check models.py.")
    raise ValueError(f"Could not build shard for model {MODEL_ID} using {ENGINE_CLASS_NAME}")
else:
    logger.info(f"Successfully built shard configuration for {MODEL_ID}: {phi4_shard}")


# Pre-load the model on startup
async def load_model():
    logger.info(f"Attempting to pre-load model: {MODEL_ID} using shard {phi4_shard}")
    try:
        await inference_engine.ensure_shard(phi4_shard)
        logger.info(f"Model {MODEL_ID} loaded successfully.")
    except FileNotFoundError as e:
         logger.error(f"Model file/directory not found during pre-loading: {e}")
         # Decide if server should exit or continue (might fail on first request)
         # raise SystemExit(f"Failed to load model: {e}")
    except Exception as e:
        logger.exception(f"An unexpected error occurred during model pre-loading: {e}") # Use exception for stack trace
        # raise SystemExit(f"Failed to load model: {e}")


app = FastAPI()

# --- API Models ---
class EncodeRequest(BaseModel):
    text: str

class EncodeResponse(BaseModel):
    tokens: list[int]

class DecodeRequest(BaseModel):
    tokens: list[int]

class DecodeResponse(BaseModel):
    text: str

class InferRequest(BaseModel):
    request_id: str = "default_request" # Provide a default or make it required
    tokens: list[int]
    # Add other parameters like temperature, top_p if needed later

class InferResponse(BaseModel):
    # Assuming infer_tensor returns logits: [batch, seq_len, vocab_size]
    # We likely want the *next* token prediction or similar, not raw logits usually.
    # Let's adjust based on typical LLM API (return next token or generated text)
    # For now, return logits for direct compatibility check with tinygrad code.
    logits: list[list[list[float]]] # batch, seq_len, vocab_size

# --- API Endpoints ---
@app.on_event("startup")
async def startup_event():
    # Run model loading in background task to avoid blocking startup
    # Optional: await load_model() here if startup time isn't critical and you want to ensure model loads first
    logger.info("Server starting up. Initiating model pre-load.")
    asyncio.create_task(load_model())


@app.post("/encode", response_model=EncodeResponse)
async def encode_text(request: EncodeRequest):
    try:
        # Ensure model/tokenizer is loaded for the required shard
        await inference_engine.ensure_shard(phi4_shard) # Load if not pre-loaded
        tokens = await inference_engine.encode(phi4_shard, request.text)
        return EncodeResponse(tokens=tokens.tolist())
    except FileNotFoundError as e:
        logger.error(f"/encode error - Model file not found: {e}")
        raise HTTPException(status_code=404, detail=f"Model file not found: {e}")
    except Exception as e:
        logger.exception(f"Error during /encode: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during encoding: {str(e)}")

@app.post("/decode", response_model=DecodeResponse)
async def decode_tokens(request: DecodeRequest):
    try:
        # Ensure model/tokenizer is loaded
        await inference_engine.ensure_shard(phi4_shard) # Load if not pre-loaded
        text = await inference_engine.decode(phi4_shard, request.tokens)
        return DecodeResponse(text=text)
    except FileNotFoundError as e:
        logger.error(f"/decode error - Model file not found: {e}")
        raise HTTPException(status_code=404, detail=f"Model file not found: {e}")
    except Exception as e:
        logger.exception(f"Error during /decode: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during decoding: {str(e)}")

@app.post("/infer", response_model=InferResponse)
async def infer(request: InferRequest):
    if not request.tokens:
        raise HTTPException(status_code=400, detail="Input tokens cannot be empty.")
    try:
        # Ensure model is loaded
        await inference_engine.ensure_shard(phi4_shard) # Load if not pre-loaded
        # Tinygrad expects batch dimension. Assuming codex sends single sequence.
        input_array = np.array([request.tokens], dtype=np.int64)

        # Note: state handling (request_id) might need more robust implementation
        # based on how TinygradDynamicShardInferenceEngine uses it.
        output_logits, _ = await inference_engine.infer_tensor(
            request.request_id,
            phi4_shard,
            input_array,
            inference_state=None # Pass state if required by engine implementation
        )
        # output_logits shape: (batch_size, sequence_length, vocab_size)
        return InferResponse(logits=output_logits.tolist())
    except FileNotFoundError as e:
        logger.error(f"/infer error - Model file not found: {e}")
        raise HTTPException(status_code=404, detail=f"Model file not found: {e}")
    except Exception as e:
        logger.exception(f"Error during /infer: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during inference: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server with Uvicorn...")
    # Ensure this runs from the root of the 'exo' project or adjust paths accordingly
    # Running directly might fail imports if not in correct CWD or PYTHONPATH isn't set.
    # Recommended to run using: uvicorn exo.server:app --reload --port 8000 from the project root (/exo)
    uvicorn.run(app, host="0.0.0.0", port=8000) 