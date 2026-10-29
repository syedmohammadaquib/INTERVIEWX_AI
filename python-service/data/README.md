# Interview analysis training data

Put training files in `train/` and name them `training_data_001.json`,
`training_data_002.json`, and so on. The fine-tuning script combines every
matching JSON file in filename order by appending their array entries in
memory; it does not create a merged JSON file. Each file should contain a JSON
array of examples. `train/training_data_001.json` is an empty starter file;
replace its contents with your reviewed examples, and add more numbered files
as needed.

Do not use test examples for training. Keep them separately in `test/`; see
`test/README.md`. The current script does not read or evaluate test data.

Each training example should have this shape:

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

Labels should be reviewed against the rubric in `../rubric.json`. Use
consented data, remove names and other identifying details, and ensure the
same interview or candidate does not appear in both training and test data.
Do not put real candidate data in version control.
