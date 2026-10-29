from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json
import uvicorn
import os
from pathlib import Path
from dotenv import load_dotenv
import asyncio

# Load environment variables
load_dotenv()

app = FastAPI(title="MY_AI Interview Analysis Service", version="1.0.0")

class ConversationMessage(BaseModel):
    type: str  # "user" or "assistant"
    content: str

class FeedbackRequest(BaseModel):
    conversation: List[ConversationMessage]

class FeedbackResponse(BaseModel):
    data: Dict[str, Any]

# Global variables for model and tokenizer
model = None
tokenizer = None
generator = None

def load_model():
    """Load the Hugging Face model for interview analysis"""
    global model, tokenizer, generator

    model_name = os.getenv("HF_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

    try:
        print(f"Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Add padding token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            low_cpu_mem_usage=True
        )

        # Create text generation pipeline
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=1024,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        # Fallback to CPU if GPU fails
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
            print("Model loaded on CPU")
        except Exception as e2:
            print(f"Failed to load model: {e2}")
            raise

@app.on_event("startup")
async def startup_event():
    """Optionally load the model on startup.

    If `SKIP_MODEL_LOAD` is set (true/1/yes) the service will start quickly
    and the model will be loaded lazily on first request instead. Otherwise
    the model load is started in a background thread so the server becomes
    responsive while the heavy load runs asynchronously.
    """
    skip = os.getenv("SKIP_MODEL_LOAD", "false").lower() in ("1", "true", "yes")
    if skip:
        print("SKIP_MODEL_LOAD set — skipping model load on startup.")
        return

    # Start model load in background to avoid blocking startup
    print("Starting background model load...")
    asyncio.create_task(asyncio.to_thread(load_model))

def format_conversation_for_analysis(conversation: List[ConversationMessage]) -> str:
    """Format conversation messages for analysis"""
    formatted = []
    for msg in conversation:
        role = "Candidate" if msg.type == "user" else "Interviewer"
        formatted.append(f"{role}: {msg.content}")

    return "\n".join(formatted)

def parse_ai_response(response_text: str) -> Dict[str, Any]:
    """Parse the AI response into structured feedback format"""
    start_idx = response_text.find("{")
    end_idx = response_text.rfind("}") + 1

    if start_idx == -1 or end_idx <= start_idx:
        raise HTTPException(
            status_code=502,
            detail="The model did not return valid JSON feedback.",
        )

    try:
        parsed = json.loads(response_text[start_idx:end_idx])
    except json.JSONDecodeError as error:
        raise HTTPException(
            status_code=502,
            detail="The model returned malformed JSON feedback.",
        ) from error

    if not isinstance(parsed, dict) or not isinstance(parsed.get("feedback"), dict):
        raise HTTPException(
            status_code=502,
            detail="The model response is missing the feedback object.",
        )

    return parsed

@app.post("/analyze", response_model=FeedbackResponse)
async def analyze_interview(request: FeedbackRequest):
    """Analyze interview conversation and provide feedback"""
    try:
        if not request.conversation:
            raise HTTPException(status_code=400, detail="No conversation provided")

        # Format conversation for analysis
        conversation_text = format_conversation_for_analysis(request.conversation)

        instructions_path = Path(__file__).with_name("instructions.json")
        instructions = json.loads(instructions_path.read_text(encoding="utf-8"))
        rubric_path = Path(__file__).with_name("rubric.json")
        rubric = json.loads(rubric_path.read_text(encoding="utf-8"))
        system_instruction = instructions.get("system")
        analysis_template = instructions.get("analysis_template")
        if (
            not isinstance(system_instruction, str)
            or not isinstance(analysis_template, str)
            or not isinstance(rubric, dict)
        ):
            raise RuntimeError(f"Invalid instruction file format: {instructions_path}")
        analysis_prompt = analysis_template.replace("{conversation}", conversation_text)
        system_instruction = (
            f"{system_instruction}\n\nSCORING RUBRIC:\n"
            f"{json.dumps(rubric, ensure_ascii=False, indent=2)}"
        )

        # If model wasn't loaded on startup (or is still loading), load it lazily
        if generator is None:
            try:
                print("Model not loaded yet — loading now (lazy load)...")
                await asyncio.to_thread(load_model)
            except HTTPException:
                raise
            except Exception as e:
                print(f"Failed to load model on-demand: {e}")
                raise HTTPException(status_code=503, detail="AI model not available")

        # Generate analysis
        print("Generating interview analysis...")
        formatted_prompt = tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": system_instruction,
                },
                {"role": "user", "content": analysis_prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        outputs = generator(
            formatted_prompt,
            max_new_tokens=1024,
            temperature=0.3,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )

        response_text = outputs[0]["generated_text"]
        # Remove the prompt from response
        analysis_result = response_text[len(formatted_prompt):].strip()

        # Parse the response
        structured_feedback = parse_ai_response(analysis_result)

        return FeedbackResponse(data=structured_feedback)

    except HTTPException:
        raise
    except Exception as e:
        print(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "service": "MY_AI Interview Analysis"
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "MY_AI Interview Analysis Service",
        "version": "1.0.0",
        "endpoints": {
            "POST /analyze": "Analyze interview conversation",
            "GET /health": "Service health check"
        }
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
