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
from peft import PeftModel
from jsonschema import Draft202012Validator

# Load environment variables
load_dotenv()

app = FastAPI(title="MY_AI Interview Analysis Service", version="1.0.0")
SERVICE_DIR = Path(__file__).resolve().parent
ADAPTER_DIR = Path(os.getenv("HF_ADAPTER_PATH", SERVICE_DIR / "fine_tuned_model"))

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
loaded_model_name = None
output_validator = None
prefix_allowed_tokens_fn = None

def load_model():
    """Load the base model with the saved interview-analysis adapter."""
    global model, tokenizer, generator, loaded_model_name
    global output_validator, prefix_allowed_tokens_fn

    metadata_path = ADAPTER_DIR / "fine_tune_metadata.json"
    adapter_config_path = ADAPTER_DIR / "adapter_config.json"
    if not metadata_path.is_file() or not adapter_config_path.is_file():
        raise FileNotFoundError(
            f"Fine-tuned adapter files were not found in {ADAPTER_DIR}."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    adapter_base = adapter_config.get("base_model_name_or_path")
    metadata_base = metadata.get("model_name")
    if not isinstance(adapter_base, str) or not adapter_base or metadata_base != adapter_base:
        raise ValueError(
            f"Adapter base model in {adapter_config_path} does not match {metadata_path}."
        )
    configured_model = os.getenv("HF_MODEL")
    model_name = adapter_base
    if configured_model and configured_model != adapter_base:
        print(
            f"HF_MODEL={configured_model!r} does not match the saved adapter; "
            f"using the adapter's required base model {adapter_base!r}."
        )
    if not isinstance(model_name, str) or not model_name:
        raise ValueError(f"No base model is configured in {metadata_path}.")

    print(f"Loading base model: {model_name}")
    model_load_options = {
        "dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        "low_cpu_mem_usage": True,
    }
    if torch.cuda.is_available():
        model_load_options["device_map"] = "auto"
    else:
        model_load_options["device_map"] = {"": "cpu"}
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_load_options,
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()
    loaded_model_name = model_name
    tokenizer_source = ADAPTER_DIR if (ADAPTER_DIR / "tokenizer_config.json").is_file() else model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    schema_path = SERVICE_DIR / "output_schema.json"
    output_schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(output_schema)
    output_validator = Draft202012Validator(output_schema)
    prefix_allowed_tokens_fn = build_schema_token_filter(tokenizer, output_schema)

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        device=None if torch.cuda.is_available() else -1,
    )
    print(f"Fine-tuned adapter loaded successfully from {ADAPTER_DIR}.")


def build_schema_token_filter(model_tokenizer, schema):
    """Build a Transformers-compatible token filter for the output JSON schema."""
    from lmformatenforcer import (
        JsonSchemaParser,
        TokenEnforcer,
        TokenEnforcerTokenizerData,
    )

    vocab_size = len(model_tokenizer)
    token_0 = model_tokenizer.encode("0")[-1]
    regular_tokens = []
    for token_id in range(vocab_size):
        if token_id in model_tokenizer.all_special_ids:
            continue
        decoded_after_0 = model_tokenizer.decode([token_0, token_id])[1:]
        decoded_token = model_tokenizer.decode([token_id])
        is_word_start = len(decoded_after_0) > len(decoded_token)
        regular_tokens.append((token_id, decoded_after_0, is_word_start))

    def decode_tokens(tokens):
        return model_tokenizer.decode(tokens).rstrip("�")

    tokenizer_data = TokenEnforcerTokenizerData(
        regular_tokens,
        decode_tokens,
        model_tokenizer.eos_token_id,
        False,
        vocab_size,
    )
    token_enforcer = TokenEnforcer(tokenizer_data, JsonSchemaParser(schema))

    def allowed_tokens(_batch_id, token_ids):
        return token_enforcer.get_allowed_tokens(token_ids.tolist()).allowed_tokens

    return allowed_tokens


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
    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError as error:
        raise HTTPException(
            status_code=502,
            detail="The model returned malformed JSON feedback.",
        ) from error

    if output_validator is None:
        raise RuntimeError("The output JSON schema validator is not initialized.")
    errors = sorted(output_validator.iter_errors(parsed), key=lambda error: list(error.path))
    if errors:
        detail = "; ".join(error.message for error in errors[:5])
        raise HTTPException(
            status_code=502,
            detail=f"The model response did not match the required feedback schema: {detail}",
        )
    for field in (
        "summary",
        "recommendationMessage",
        "technicalAssessment",
        "communicationAssessment",
    ):
        if not parsed["feedback"][field].strip():
            raise HTTPException(
                status_code=502,
                detail=f"The model returned an empty feedback.{field} value.",
            )
    for field in ("strengths", "improvements"):
        if any(not item.strip() for item in parsed["feedback"][field]):
            raise HTTPException(
                status_code=502,
                detail=f"The model returned an empty string in feedback.{field}.",
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
            do_sample=False,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
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
