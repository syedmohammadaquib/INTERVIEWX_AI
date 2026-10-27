# 🚀 MY_AI - Custom Interview Analysis Module

A modular AI analysis component for the InterviewX platform that provides intelligent feedback on interview conversations using **Python + Hugging Face models** (replacing the previous Groq/OpenRouter analysis).

## 📋 Overview

MY_AI serves as the primary AI analysis engine for interview conversations in the InterviewX platform. It processes conversation transcripts and generates detailed feedback including ratings, recommendations, strengths, improvements, and assessments.

### Current Status (April 2026)

- MY_AI is integrated and running as the primary analysis provider.
- Groq is configured as fallback when MY_AI fails.
- Fine-tuning is not applied yet (inference-only mode with base HF model).
- Fine-tuning will be done later with your labeled interview dataset.

### 🔄 Integration Flow
```
Interview Conversation → MY_AI API → Python Service → Hugging Face Model → Structured Feedback
                                   ↓ (if Python service fails)
                            Fallback to Groq/OpenRouter
```

### 🎯 Scope & Responsibilities

**MY_AI (Analysis-Centric):**
- ✅ **Primary**: Generates interview feedback and analysis
- ✅ **Fallback**: Groq/OpenRouter when Python service unavailable
- ✅ **Fine-tunable**: Custom models for specific interview criteria

**Question Generation:**
- ❗ **Groq Only**: Interview questions are generated exclusively via Groq/OpenRouter
- ❗ **Not in MY_AI**: Question generation remains separate and unchanged

## 🔗 How MY_AI Connects to Your App

MY_AI integrates seamlessly with InterviewX using a **3-tier architecture** that requires zero frontend changes.

### 📡 Connection Flow

```
Frontend (React) → Next.js API (/api/ai-feedback) → MY_AI API → Python FastAPI → Hugging Face Model
                    ↓ (if MY_AI fails)
             Fallback to Groq/OpenRouter Analysis
```

### 🔄 Detailed Integration Steps

**1. Frontend Call (Unchanged):**
Your React components call the same `/api/ai-feedback` endpoint as before:

```typescript
// No changes needed in your React code
fetch('/api/ai-feedback', {
  method: 'POST',
  body: JSON.stringify({ conversation: interviewMessages })
})
```

**2. Next.js API Route (`ai-feedback`):**
The existing route now tries MY_AI first, then falls back to Groq:

```typescript
// Automatic primary/fallback logic
const myAiResponse = await axios.post('http://localhost:3000/MY_AI/api', 
  { conversation }, { timeout: 10000 }
);
```

**3. MY_AI API Route (`MY_AI/api/route.ts`):**
Acts as a proxy to your Python service:

```typescript
// Calls Python FastAPI with 15-second timeout
const pythonResponse = await axios.post(
  process.env.MY_AI_PYTHON_ENDPOINT || 'http://localhost:8000/analyze',
  { conversation }, { timeout: 15000 }
);
```

**4. Python FastAPI Service (`main.py`):**
Processes conversations with Hugging Face models:

```python
@app.post("/analyze")
async def analyze_conversation(data: ConversationRequest):
    # AI analysis using fine-tuned models
    return {"feedback": structured_analysis}
```

### 🚀 Setup & Connection

**Environment Configuration (.env.local):**
```env
# MY_AI Integration
MY_AI_PYTHON_ENDPOINT=http://localhost:8000/analyze

# Provider routing for /api/ai-feedback
# auto = MY_AI first, fallback to Groq
# myai = MY_AI only
# groq = Groq only
AI_FEEDBACK_PROVIDER=auto

# Optional MY_AI timeout in ms
MY_AI_TIMEOUT_MS=15000

# Keep Groq key for fallback/provider mode
GROQ_API_KEY=your_groq_api_key_here
```

**Start Services:**
```bash
# Terminal 1: Start Python AI Service
cd MY_AI/python-service
python main.py

# Terminal 2: Start InterviewX (in project root)
npm run dev
```

### ✅ Result

- **Zero Frontend Changes**: Existing interview pages work unchanged
- **Automatic Analysis**: MY_AI processes interviews, Groq handles fallback
- **Seamless Integration**: Drop-in replacement for analysis functionality
- **Production Ready**: Scalable FastAPI service with Hugging Face models

**Your InterviewX now uses custom AI analysis with automatic fallback!** 🎯

## Provider Switching and Testing Visibility

The implementation in `src/app/api/ai-feedback/route.ts` now supports three modes:

- `AI_FEEDBACK_PROVIDER=auto` (recommended): tries MY_AI first, falls back to Groq
- `AI_FEEDBACK_PROVIDER=myai`: only MY_AI
- `AI_FEEDBACK_PROVIDER=groq`: only Groq

You can also override provider per request:

```json
{
  "conversation": [{ "type": "user", "content": "..." }],
  "provider": "groq"
}
```

Every response includes metadata so you always know which engine handled it:

```json
{
  "data": { "feedback": { "...": "..." } },
  "meta": {
    "requestedProvider": "auto",
    "providerUsed": "myai",
    "attemptedProviders": ["myai"],
    "fallbackUsed": false,
    "providerErrors": {}
  }
}
```

If fallback occurs, `providerUsed` becomes `groq` and `fallbackUsed` is `true`.

- **Frontend Integration**: Next.js API Routes (TypeScript)
- **AI Engine**: Python FastAPI + Hugging Face Transformers
- **Models**: Fine-tuned conversational AI models for interview analysis
- **Input**: Interview conversation messages array
- **Output**: Structured JSON feedback matching InterviewX schema
- **Fallback Support**: Automatic fallback to existing Groq analysis

## 📁 File Structure

```
MY_AI/
├── README.md                    # 📖 Complete setup guide
├── api/
│   └── route.ts                 # 🔗 Next.js integration endpoint
└── python-service/
    ├── main.py                  # 🐍 FastAPI server with Hugging Face
    ├── requirements.txt         # 📦 Python dependencies
    ├── README.md               # 🚀 Python service documentation
    ├── start.sh / start.bat    # ▶️ Startup scripts
    ├── .env.example           # ⚙️ Configuration template
    └── fine_tune.py           # 🎯 Model fine-tuning script
```

## 🔧 Installation & Setup

### 1. Copy the Folder
Copy the entire `MY_AI` folder into your InterviewX project root:

```bash
# From your project root
cp -r /path/to/MY_AI ./MY_AI
```

### 2. Install Python Dependencies
```bash
cd MY_AI/python-service
pip install -r requirements.txt
```

### 3. Configure Environment
**Main project (.env.local):**
```env
# MY_AI Configuration
MY_AI_PYTHON_ENDPOINT=http://localhost:8000/analyze
```

**Python service (.env):**
```env
# Hugging Face Configuration
HF_MODEL=Qwen/Qwen2.5-1.5B-Instruct
HF_TOKEN=your_hugging_face_token_here

# Server Configuration
PORT=8000
HOST=0.0.0.0
```

### 4. Start Services
```bash
# Terminal 1: Start Python service
cd MY_AI/python-service
python main.py
# or on Windows: start.bat

# Terminal 2: Start InterviewX
npm run dev
```

## 🧠 AI Model Configuration

### Recommended Models for Interview Analysis

| Model | Size | Quality | Speed | Use Case |
|-------|------|---------|-------|----------|
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B parameters | Candidate default | CPU-friendly relative to larger models | Local prototype |

### Fine-tuning Your Model

MY_AI includes a fine-tuning script for adapting the selected base model to interview analysis. Fine-tuning has not been run yet:

```bash
cd MY_AI/python-service
python fine_tune.py
```

**What the fine-tuning does:**
- Trains on interview conversation patterns
- Learns to provide structured feedback
- Adapts to your evaluation criteria
- Replaces generic Groq analysis with domain-specific insights

## 📡 API Interface

### Endpoint
```
POST http://localhost:3000/MY_AI/api
```

### Request Body
```json
{
  "conversation": [
    {
      "type": "user",
      "content": "I have 3 years of React experience..."
    },
    {
      "type": "assistant",
      "content": "Tell me about your most challenging project..."
    }
  ]
}
```

### Response Format
```json
{
  "data": {
    "feedback": {
      "rating": {
        "relevance": 4,
        "technicalDepth": 3,
        "clarity": 4,
        "communicationQuality": 4
      },
      "summary": "Candidate shows strong technical background...",
      "recommendation": "Yes",
      "recommendationMessage": "Excellent fit for the role...",
      "strengths": ["Strong technical skills", "Clear communication"],
      "improvements": ["Could provide more examples"],
      "technicalAssessment": "Demonstrates expertise in React...",
      "communicationAssessment": "Articulate and professional",
      "overallConfidence": 85
    }
  }
}
```

## 🎯 Fine-tuning Process

### Step 1: Prepare Training Data
Create interview conversation + feedback pairs:

```json
[
  {
    "conversation": "Candidate: I know React... Interviewer: Tell me about state management...",
    "feedback": {
      "rating": {"relevance": 4, "technicalDepth": 3, ...},
      "recommendation": "Yes",
      ...
    }
  }
]
```

### Step 2: Run Fine-tuning
```bash
cd MY_AI/python-service
python fine_tune.py --data interview_data.json --epochs 3
```
MY_AI_API_KEY=sk-your-openai-key
MY_AI_ENDPOINT=https://api.openai.com/v1/chat/completions
MY_AI_MODEL=gpt-4

# Example for local Ollama
MY_AI_API_KEY=not-needed
MY_AI_ENDPOINT=http://localhost:11434/v1/chat/completions
MY_AI_MODEL=llama2
```

## 🧪 Testing

### Start the Development Server
```bash
npm run dev
# or
yarn dev
```

### Test the API
```bash
curl -X POST http://localhost:3000/MY_AI/api \
  -H "Content-Type: application/json" \
  -d '{
    "conversation": [
      {"type": "user", "content": "Hello, I am ready for the interview"},
      {"type": "assistant", "content": "Great! Let me ask you a technical question..."}
    ]
  }'
```

## 🔄 Fallback Mechanism

The module includes automatic fallback to ensure reliability:

1. **Primary**: MY_AI attempts analysis
2. **Timeout**: 10-second timeout (configurable)
3. **Fallback**: If MY_AI fails, system uses existing Groq/OpenRouter analysis
4. **Logging**: All attempts are logged for debugging

## 📊 Performance & Reliability

- **Timeout Protection**: Prevents hanging requests
- **Error Handling**: Graceful degradation to fallback
- **Structured Output**: Consistent JSON response format
- **Type Safety**: Full TypeScript support

## 🛠️ Customization

### Modifying the Analysis Prompt
Edit the prompt template in `api/route.ts`:

```typescript
const promptTemplate = PromptTemplate.fromTemplate(`
Your custom analysis instructions here...
{conversation}
{format_instructions}
`);
```

### Adding Custom Logic
The route can be extended with:
- Custom preprocessing
- Multiple AI service attempts
- Advanced error handling
- Custom response formatting

## 🔍 Troubleshooting

### Common Issues

**MY_AI returns mock response**
- Check if `MY_AI_API_KEY` is set
- Verify `MY_AI_ENDPOINT` is accessible
- Confirm model name is correct

**Timeout errors**
- Increase `MY_AI_TIMEOUT` value
- Check network connectivity
- Verify AI service is responsive

**Fallback not working**
- Ensure Groq/OpenRouter API key is configured
- Check fallback logic in main route
- Verify conversation data format

### Debug Logging
Enable detailed logging by checking console output:
```bash
# Look for these log messages:
[MY_AI] Using mock response - configure your AI service later
[ai-feedback] Trying MY_AI primary analysis...
[ai-feedback] MY_AI analysis successful
[ai-feedback] Using fallback analysis (OpenRouter/Gemini)
```

## 📄 License

This module is part of the InterviewX platform. See main project license for details.

## 🤝 Contributing

To contribute to MY_AI:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For issues or questions:
- Check the logs for error messages
- Verify environment variables
- Test API connectivity
- Review fallback mechanism

---

**Built for InterviewX - Intelligent AI Interview Platform**
