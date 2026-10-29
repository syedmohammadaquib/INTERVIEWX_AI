# Interview analysis training files

Add human-reviewed interview examples to `training_data_001.json`, then create
`training_data_002.json`, `training_data_003.json`, and so on as needed. Each
file must be a JSON array of `conversation` and `feedback` records matching
`../schema.json`. The fine-tuning script combines all files matching
`training_data_*.json` in this folder.

To split one JSON array into batches of five, run
`python converter.py reviewed_examples.json` from the service directory. It
saves the `conversation` and `feedback` records here and continues numbering
without overwriting existing training files. Pass `--batch-size N` to choose a
different batch size.

Each conversation must contain the interviewer's questions (`type: "assistant"`)
and the candidate's actual responses (`type: "user"`). The feedback must be
reviewed and grounded in that candidate's answers; the trainer learns to
produce this feedback, not to answer the interview questions. Keep held-out
examples in `../test/`.
