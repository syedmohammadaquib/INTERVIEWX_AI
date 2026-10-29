# Interview analysis training data

Put training files in `train/` and name them `training_data_001.json`,
`training_data_002.json`, and so on. Each file is a JSON array of interview
examples matching [`schema.json`](./schema.json). The fine-tuning script
combines matching files in filename order; it does not create a merged file.
`train/training_data_001.json` is an empty starter file.

Each example contains the actual interviewer/candidate conversation and a
human-reviewed `feedback` object. The trainer converts it internally to Qwen's
chat template: the full transcript is the input and the serialized feedback
object is the target output. For example:

```json
[
  {
    "conversation": [
      {
        "type": "assistant",
        "content": "What does changing a React ref's .current value do?"
      },
      {
        "type": "user",
        "content": "It does not trigger a re-render."
      }
    ],
    "feedback": {
      "rating": {
        "relevance": 4,
        "technicalDepth": 3,
        "clarity": 4,
        "communicationQuality": 4
      },
      "summary": "The candidate correctly explained the effect of changing a ref.",
      "recommendation": "Maybe",
      "recommendationMessage": "The answer is correct but brief; assess with the full interview.",
      "strengths": ["Correctly states that changing ref.current does not render."],
      "improvements": ["Could explain when refs are useful."],
      "technicalAssessment": "Correctly distinguishes refs from state for rendering.",
      "communicationAssessment": "The answer is clear and concise.",
      "overallConfidence": 70
    }
  }
]
```

The trainer uses the feedback object as its expected analysis output; it does
not train Qwen to answer the interview question. Only the generated feedback
tokens contribute to training loss.

Have a person review both the transcript and every feedback field for accuracy
before training. Do not treat canonical quiz answers or automatically generated
placeholder feedback as candidate interviews or verified labels. The converter
and trainer reject the CodePrep conversion-placeholder labels for this reason.
Keep held-out evaluation examples separate in `test/`; use consented data,
remove personal information, and do not put sensitive real candidate data in
version control.
