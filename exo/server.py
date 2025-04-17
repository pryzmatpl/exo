from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
import numpy as np
import os
from pathlib import Path
import sys
import logging
import json
import argparse
from typing import Optional, Dict, Any
from exo.inference.shard import Shard
from exo.inference.tinygrad.inference import TinygradDynamicShardInferenceEngine
from exo.models import build_base_shard
from exo.download.shard_download import LocalModelDownloader

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


class ModelServer:
    """
    A server that handles model inference requests and communicates over stdin/stdout.
    This allows for easy integration with other applications like codex.
    """
    def __init__(self, model_id: str, engine_name: str = "TinygradDynamicShardInferenceEngine", temperature: float = 0.7, max_tokens: int = 1024):
        self.model_id = model_id
        self.engine_name = engine_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.downloader = LocalModelDownloader()
        self.engine = None
        self.shard = None
        self.request_id = "default"
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("model_server.log"),
                logging.StreamHandler(sys.stderr)
            ]
        )
        self.logger = logging.getLogger("ModelServer")

    async def initialize(self):
        """Initialize the model and engine."""
        self.logger.info(f"Initializing model {self.model_id} with engine {self.engine_name}")
        
        # Create the shard
        self.shard = build_base_shard(self.model_id, self.engine_name)
        if self.shard is None:
            raise ValueError(f"Model {self.model_id} not found or not supported by {self.engine_name}")
        
        # Initialize the engine
        if self.engine_name == "TinygradDynamicShardInferenceEngine":
            self.engine = TinygradDynamicShardInferenceEngine(self.downloader)
        else:
            raise ValueError(f"Unsupported engine: {self.engine_name}")
        
        # Send configuration info
        config = {
            "model_id": self.model_id,
            "engine": self.engine_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "status": "initialized"
        }
        print(json.dumps(config))
        sys.stdout.flush()
        
        # Log that we're ready
        self.logger.info("Model initialized and ready")

    async def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single inference request."""
        prompt = request_data.get("prompt", "")
        temperature = request_data.get("temperature", self.temperature)
        max_tokens = request_data.get("max_tokens", self.max_tokens)
        
        self.logger.info(f"Processing request with prompt length: {len(prompt)}")
        
        # Encode the prompt
        input_tokens = await self.engine.encode(self.shard, prompt)
        
        # Generate text
        response_tokens = []
        for _ in range(max_tokens):
            # Get the input tensor (including generated tokens so far)
            input_data = input_tokens if not response_tokens else \
                         np.concatenate([input_tokens, np.array(response_tokens)], axis=0)
            input_data = input_data.reshape(1, -1)  # Add batch dimension
            
            # Run inference
            output_data, _ = await self.engine.infer_tensor(self.request_id, self.shard, input_data)
            
            # Sample the next token
            next_token = await self.engine.sample(output_data, temp=temperature)
            
            # Break on EOS token (assuming EOS is tokenizer-specific)
            if next_token.item() == self.engine.tokenizer.eos_token_id:
                break
                
            # Add the token to our response
            response_tokens.append(next_token.item())
        
        # Decode the response
        generated_text = await self.engine.decode(self.shard, np.array(response_tokens))
        
        self.logger.info(f"Generated response with length: {len(generated_text)}")
        
        return {
            "text": generated_text,
            "tokens": len(response_tokens)
        }

    async def run_server(self):
        """Run the server in stdin/stdout mode."""
        await self.initialize()
        
        while True:
            try:
                # Read a line from stdin
                line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
                line = line.strip()
                
                if not line:
                    await asyncio.sleep(0.1)
                    continue
                
                if line.lower() == "exit":
                    self.logger.info("Received exit command")
                    break
                
                # Parse the request
                try:
                    request_data = json.loads(line)
                except json.JSONDecodeError:
                    self.logger.error(f"Invalid JSON: {line}")
                    print(f"RESPONSE: {{\"error\": \"Invalid JSON\"}}")
                    sys.stdout.flush()
                    continue
                
                # Process the request
                response = await self.process_request(request_data)
                
                # Send the response
                print(f"RESPONSE: {json.dumps(response)}")
                sys.stdout.flush()
                
            except Exception as e:
                self.logger.error(f"Error processing request: {str(e)}")
                print(f"RESPONSE: {{\"error\": \"{str(e)}\"}}")
                sys.stdout.flush()

def main():
    """Main entry point for the server."""
    parser = argparse.ArgumentParser(description="Run a model server")
    parser.add_argument("--model", type=str, default="phi-4", help="Model ID to use")
    parser.add_argument("--engine", type=str, default="TinygradDynamicShardInferenceEngine", help="Inference engine to use")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Maximum tokens to generate")
    parser.add_argument("--server_mode", action="store_true", help="Run in server mode (stdin/stdout)")
    
    args = parser.parse_args()
    
    server = ModelServer(
        model_id=args.model,
        engine_name=args.engine,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )
    
    if args.server_mode:
        asyncio.run(server.run_server())
    else:
        # If not in server mode, just initialize the model and exit
        asyncio.run(server.initialize())
        print("Model initialized successfully. Use --server_mode to run as a server.")

if __name__ == "__main__":
    main() 