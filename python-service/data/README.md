# Interview analysis training data

Keep approved, anonymized training examples in this directory. The fine-tuning
script currently reads `train.json` and optionally `val.json`; it does not read
a test set. Keep a separate `test.json` held out and evaluate it separately
before considering a model ready.

Each JSON file should contain an array of examples in this shape:

```json
[
  {
    "conversation": [
      { "type": "assistant", "content": "Tell me about a project you built." },
      { "type": "user", "content": "I built a project using React..." }
    ],
    "feedback": {
      "rating": {
        "relevance": 3,
        "technicalDepth": 4,
        "clarity": 4,
        "communicationQuality": 3
      },
      "summary": "Human-reviewed summary of the interview.",
      "recommendation": "Maybe",
      "recommendationMessage": "Evidence-based explanation.",
      "strengths": ["Human-reviewed strength"],
      "improvements": ["Human-reviewed improvement"],
      "technicalAssessment": "Evidence-based technical assessment.",
      "communicationAssessment": "Evidence-based communication assessment.",
      "overallConfidence": 75
    }
  }
]
```

Labels should be reviewed against a consistent rubric. Use consented data,
remove names and other identifying details, and split examples by candidate
between train, validation, and test to avoid placing the same interview in
multiple splits. Do not put real candidate data in version control.
