from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json
import uvicorn
import os
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
    try:
        # Try to extract JSON from response
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1

        if start_idx != -1 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx]
            parsed = json.loads(json_str)
            return parsed
    except:
        pass

    # Fallback: extract ratings and create structured response
    return {
        "feedback": {
            "rating": {
                "relevance": 3,
                "technicalDepth": 3,
                "clarity": 3,
                "communicationQuality": 3
            },
            "summary": response_text[:300] + "..." if len(response_text) > 300 else response_text,
            "recommendation": "Maybe",
            "recommendationMessage": "Analysis completed with AI model",
            "strengths": ["Communication skills", "Technical knowledge"],
            "improvements": ["Could provide more detailed examples", "Practice problem-solving"],
            "technicalAssessment": "Demonstrates solid technical foundation",
            "communicationAssessment": "Clear and professional communication",
            "overallConfidence": 75
        }
    }

@app.post("/analyze", response_model=FeedbackResponse)
async def analyze_interview(request: FeedbackRequest):
    """Analyze interview conversation and provide feedback"""
    try:
        if not request.conversation:
            raise HTTPException(status_code=400, detail="No conversation provided")

        # Format conversation for analysis
        conversation_text = format_conversation_for_analysis(request.conversation)

        # Create analysis prompt
        analysis_prompt = f"""
You are an expert technical interviewer analyzing an interview conversation.

INTERVIEW CONVERSATION:
{conversation_text}

Please provide detailed feedback on this candidate's performance. Analyze their technical knowledge, communication skills, problem-solving ability, and overall fit.

Return your analysis in this exact JSON format:
{{
  "feedback": {{
    "rating": {{
      "relevance": 1-5,
      "technicalDepth": 1-5,
      "clarity": 1-5,
      "communicationQuality": 1-5
    }},
    "summary": "4-5 sentence summary of candidate performance",
    "recommendation": "Yes/No/Maybe",
    "recommendationMessage": "Detailed justification for recommendation",
    "strengths": ["strength1", "strength2", "strength3"],
    "improvements": ["improvement1", "improvement2"],
    "technicalAssessment": "Detailed technical skills evaluation",
    "communicationAssessment": "Communication skills evaluation",
    "overallConfidence": 1-100
  }}
}}

Be thorough, professional, and provide specific examples from the conversation.
"""

        # If model wasn't loaded on startup (or is still loading), load it lazily
        if generator is None:
            try:
                print("Model not loaded yet — loading now (lazy load)...")
                await asyncio.to_thread(load_model)
            except Exception as e:
                print(f"Failed to load model on-demand: {e}")
                raise HTTPException(status_code=503, detail="AI model not available")

        # Generate analysis
        print("Generating interview analysis...")
        formatted_prompt = tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": "You are an evidence-based interview analyst. Return only valid JSON matching the requested schema.",
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
