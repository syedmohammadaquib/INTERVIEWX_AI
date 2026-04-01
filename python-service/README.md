# MY_AI Python Service - Interview Analysis API

A FastAPI-based service that uses Hugging Face models to analyze interview conversations and provide detailed feedback, replacing the current Groq/OpenRouter analysis.

## 🚀 Quick Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Copy `.env.example` to `.env` and configure:

```env
# Hugging Face Model Configuration
HF_MODEL=microsoft/DialoGPT-medium  # or any other conversational model
HF_TOKEN=your_hugging_face_token    # optional, for private models

# Server Configuration
PORT=8000
HOST=0.0.0.0

# GPU Configuration (optional)
CUDA_VISIBLE_DEVICES=0  # for GPU usage
```

### 3. Run the Service
```bash
python main.py
```

The service will start on `http://localhost:8000`

## 📡 API Endpoints

### POST /analyze
Analyze an interview conversation and return structured feedback.

**Request:**
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

**Response:**
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

### GET /health
Check service health and model status.

### GET /
Service information and available endpoints.

## 🔧 Model Configuration

### Recommended Models for Interview Analysis:

1. **microsoft/DialoGPT-medium** (Default)
   - Good for conversation analysis
   - Fast inference
   - Works well for structured feedback

2. **microsoft/DialoGPT-large**
   - Better analysis quality
   - More detailed responses
   - Requires more RAM

3. **facebook/blenderbot-400M-distill**
   - Specialized for dialogue
   - Good conversation understanding

4. **Custom Fine-tuned Models**
   - Train on interview datasets
   - Domain-specific analysis
   - Best accuracy

### Fine-tuning for Interview Analysis

To create a model specialized for interview feedback:

1. **Prepare Dataset:**
```python
# Format: conversation + feedback pairs
training_data = [
    {
        "conversation": "Candidate: I know React...",
        "feedback": {"rating": {"relevance": 4, ...}, ...}
    }
]
```

2. **Fine-tune Model:**
```python
from transformers import Trainer, TrainingArguments

# Load base model
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Fine-tune on interview data
trainer = Trainer(model=model, ...)
trainer.train()
```

## 🚀 Deployment Options

### Local Development
```bash
python main.py
```

### Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "main.py"]
```

### Cloud Deployment

**Railway:**
```bash
railway deploy
```

**Render:**
- Connect GitHub repo
- Set build command: `pip install -r requirements.txt`
- Set start command: `python main.py`

**Heroku:**
```yaml
# Procfile
web: python main.py
```

## ⚙️ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_MODEL` | `microsoft/DialoGPT-medium` | Hugging Face model to use |
| `HF_TOKEN` | None | Hugging Face API token |
| `PORT` | `8000` | Server port |
| `HOST` | `0.0.0.0` | Server host |
| `CUDA_VISIBLE_DEVICES` | None | GPU device ID |

## 🔧 Integration with InterviewX

### Update MY_AI Route
In `MY_AI/api/route.ts`, replace mock response with:

```typescript
// Call Python service instead of mock
const pythonResponse = await axios.post('http://localhost:8000/analyze', {
  conversation: conversation
});

return NextResponse.json(pythonResponse.data);
```

### Environment Setup
Add to `.env.local`:
```env
MY_AI_PYTHON_ENDPOINT=http://localhost:8000/analyze
```

## 🧪 Testing

### Test API
```bash
# Health check
curl http://localhost:8000/health

# Test analysis
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "conversation": [
      {"type": "user", "content": "Hello"},
      {"type": "assistant", "content": "Hi there"}
    ]
  }'
```

### Integration Test
```bash
# Test full InterviewX integration
npm run dev  # Start Next.js
# Then test interview flow
```

## 📊 Performance

### Hardware Requirements

**Minimum (CPU):**
- RAM: 4GB
- Storage: 2GB
- Model: DialoGPT-medium

**Recommended (GPU):**
- RAM: 8GB+
- GPU: 4GB VRAM
- Storage: 5GB
- Model: DialoGPT-large

### Performance Metrics

- **Response Time:** 2-5 seconds per analysis
- **Memory Usage:** 2-4GB for medium models
- **Concurrent Requests:** 5-10 (CPU), 20+ (GPU)

## 🛠️ Troubleshooting

### Common Issues

**"Model not found"**
```bash
# Install transformers
pip install transformers
# Or set HF_TOKEN for private models
export HF_TOKEN=your_token
```

**"CUDA out of memory"**
```bash
# Use smaller model
export HF_MODEL=microsoft/DialoGPT-small
# Or use CPU
unset CUDA_VISIBLE_DEVICES
```

**"Port already in use"**
```bash
# Change port
export PORT=8001
python main.py
```

**"Connection refused"**
- Check if Python service is running
- Verify HOST and PORT settings
- Check firewall settings

### Debug Mode
```python
# Add debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🔄 Model Fine-tuning

### Training Script
```python
from transformers import Trainer, TrainingArguments
from datasets import Dataset

# Prepare interview dataset
dataset = Dataset.from_dict({
    "input": conversations,
    "output": feedback_jsons
})

# Fine-tune
training_args = TrainingArguments(
    output_dir="./interview-model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
```

### Custom Dataset Format
```json
[
  {
    "conversation": "Candidate: I know React...",
    "feedback": {
      "rating": {"relevance": 4, "technicalDepth": 3, ...},
      "summary": "Strong React knowledge...",
      ...
    }
  }
]
```

## 📈 Monitoring

### Health Checks
```bash
# Continuous monitoring
while true; do
  curl -s http://localhost:8000/health
  sleep 30
done
```

### Performance Monitoring
- Response times
- Memory usage
- GPU utilization
- Error rates

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests
4. Submit pull request

## 📄 License

Part of InterviewX platform. See main project license.
