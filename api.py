"""
DeepSeek-OCR REST API Server

Setup:
  1. Copy .env.example to .env
  2. Add your DEEPSEEK_API_KEY to .env
  3. Run: python api.py

API will be available at http://localhost:8090

Endpoints:
  POST /ocr - Process image and return OCR text
  POST /suggest - Get AI suggestions based on parsed text
  POST /ocr-and-suggest - OCR + suggestions in one call (for medical docs)
  GET /health - Health check

Environment variables (via .env):
  DEEPSEEK_API_KEY - Your DeepSeek API key (required for suggestions)
  DEEPSEEK_API_URL - API endpoint (default: https://api.deepseek.com/v1/chat/completions)
  DEEPSEEK_CHAT_MODEL - Chat model name (default: deepseek-chat)
  DEEPSEEK_REASONER_MODEL - Reasoner model name (default: deepseek-reasoner)
  CUDA_VISIBLE_DEVICES - GPU to use (default: 0)
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import warnings
import torch
import transformers
from transformers import AutoModel, AutoTokenizer
import os
import tempfile
import re
import uvicorn
import httpx
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

# Suppress cosmetic HF warnings about position_ids buffer reinitialization.
# position_ids is a register_buffer in deepencoder.py (just torch.arange), regenerated
# correctly at load time. The "newly initialized" + "TRAIN this model" warnings are
# misleading — model works correctly without training.
transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore", message=".*position_ids.*")
warnings.filterwarnings("ignore", message=".*newly initialized.*")
warnings.filterwarnings("ignore", message=".*should probably TRAIN.*")

# Single-worker executor for OCR inference. The model runs on one GPU and inference
# must be serialized — but wrapping in an executor unblocks the asyncio event loop
# so /health and other endpoints stay responsive while inference runs.
inference_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ocr-inference")

app = FastAPI(title="DeepSeek-OCR API", version="1.0")

# Enable CORS for external access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (restrict in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and tokenizer
model = None
tokenizer = None

# DeepSeek API config
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_URL = os.environ.get("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")
DEEPSEEK_CHAT_MODEL = os.environ.get("DEEPSEEK_CHAT_MODEL", "deepseek-chat")
DEEPSEEK_REASONER_MODEL = os.environ.get("DEEPSEEK_REASONER_MODEL", "deepseek-reasoner")

# Request models
class SuggestRequest(BaseModel):
    text: str
    context: Optional[str] = "medical document"  # e.g., "medical document", "invoice", "receipt"
    model: str = "deepseek-chat"  # or "deepseek-reasoner"

async def get_deepseek_suggestion(text: str, context: str, model: str = "deepseek-chat") -> str:
    """Call DeepSeek API to get suggestions based on parsed text"""
    if not DEEPSEEK_API_KEY:
        raise HTTPException(status_code=500, detail="DEEPSEEK_API_KEY not set")

    system_prompt = f"""You are an expert assistant analyzing a {context}.
Based on the OCR-parsed text provided, give helpful suggestions, insights, or advice.
Be concise and actionable."""

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            DEEPSEEK_API_URL,
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Here is the parsed document text:\n\n{text}\n\nPlease provide suggestions or advice."}
                ],
                "temperature": 0.7,
                "max_tokens": 2000
            }
        )

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"DeepSeek API error: {response.text}")

        result = response.json()
        return result["choices"][0]["message"]["content"]

def load_model():
    global model, tokenizer
    print("Loading model...")
    model_name = 'deepseek-ai/DeepSeek-OCR'

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        attn_implementation='eager',
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.bfloat16
    )
    model = model.eval().cuda()
    print("Model loaded!")

def _run_inference_sync(prompt: str, image_path: str) -> str:
    """Synchronous wrapper around model.infer() — runs in the inference_executor
    thread so the FastAPI event loop is not blocked during long inferences."""
    return model.infer(
        tokenizer,
        prompt=prompt,
        image_file=image_path,
        output_path='/tmp',
        base_size=1024,
        image_size=640,
        crop_mode=True,
        save_results=False,
        eval_mode=True
    )

def clean_repetition(text):
    """Remove repetitive patterns from output"""
    lines = text.split('\n')
    seen = set()
    cleaned = []
    repeat_count = 0

    for line in lines:
        normalized = line.strip().lower()
        if normalized in seen:
            repeat_count += 1
            if repeat_count > 3:
                continue
        else:
            repeat_count = 0
            seen.add(normalized)
        cleaned.append(line)

    return '\n'.join(cleaned)

# Model is loaded at module import time (before server starts)
# This avoids async context issues with PyTorch

@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/ocr")
async def ocr_endpoint(
    file: UploadFile = File(...),
    mode: str = Form(default="document")
):
    """
    Process an image and return OCR text.

    - file: Image file (jpg, png, etc.)
    - mode: OCR mode - "document", "general", "free", "figure", "describe"

    Returns: {"text": "...", "mode": "..."}
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Map mode to prompt
    prompts = {
        "document": "<image>\n<|grounding|>Convert the document to markdown.",
        "general": "<image>\n<|grounding|>OCR this image.",
        "free": "<image>\nFree OCR.",
        "figure": "<image>\nParse the figure.",
        "describe": "<image>\nDescribe this image in detail."
    }
    prompt = prompts.get(mode, prompts["document"])

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        content = await file.read()
        tmp.write(content)
        temp_path = tmp.name

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            inference_executor, _run_inference_sync, prompt, temp_path
        )

        if result:
            result = clean_repetition(result)
            result = re.sub(r'<\|ref\|>|<\|/ref\|>|<\|det\|>.*?<\|/det\|>', '', result)

        return {
            "text": result if result else "",
            "mode": mode
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        os.unlink(temp_path)

@app.post("/suggest")
async def suggest_endpoint(request: SuggestRequest):
    """
    Get AI suggestions based on parsed text.

    - text: The OCR-parsed text
    - context: Type of document (e.g., "medical document", "invoice", "receipt")
    - model: "deepseek-chat" or "deepseek-reasoner"

    Returns: {"suggestion": "...", "model": "..."}
    """
    suggestion = await get_deepseek_suggestion(request.text, request.context, request.model)
    return {
        "suggestion": suggestion,
        "model": request.model
    }

@app.post("/ocr-and-suggest")
async def ocr_and_suggest_endpoint(
    file: UploadFile = File(...),
    mode: str = Form(default="document"),
    context: str = Form(default="medical document"),
    suggestion_model: str = Form(default="deepseek-chat")
):
    """
    Process image with OCR and get AI suggestions in one call.

    - file: Image file
    - mode: OCR mode
    - context: Document type for suggestions
    - suggestion_model: "deepseek-chat" or "deepseek-reasoner"

    Returns: {"ocr_text": "...", "suggestion": "...", "model": "..."}
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Map mode to prompt
    prompts = {
        "document": "<image>\n<|grounding|>Convert the document to markdown.",
        "general": "<image>\n<|grounding|>OCR this image.",
        "free": "<image>\nFree OCR.",
        "figure": "<image>\nParse the figure.",
        "describe": "<image>\nDescribe this image in detail."
    }
    prompt = prompts.get(mode, prompts["document"])

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        content = await file.read()
        tmp.write(content)
        temp_path = tmp.name

    try:
        # Step 1: OCR (run in executor to avoid blocking event loop)
        loop = asyncio.get_event_loop()
        ocr_result = await loop.run_in_executor(
            inference_executor, _run_inference_sync, prompt, temp_path
        )

        if ocr_result:
            ocr_result = clean_repetition(ocr_result)
            ocr_result = re.sub(r'<\|ref\|>|<\|/ref\|>|<\|det\|>.*?<\|/det\|>', '', ocr_result)

        ocr_text = ocr_result if ocr_result else ""

        # Step 2: Get suggestions
        suggestion = ""
        if ocr_text:
            suggestion = await get_deepseek_suggestion(ocr_text, context, suggestion_model)

        return {
            "ocr_text": ocr_text,
            "suggestion": suggestion,
            "model": suggestion_model
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        os.unlink(temp_path)

if __name__ == "__main__":
    load_model()
    uvicorn.run(app, host="0.0.0.0", port=8090)
