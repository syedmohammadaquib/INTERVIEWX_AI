# Held-out test files

Put human-reviewed examples for final evaluation in this folder, named
`test_data_001.json`, `test_data_002.json`, and so on. Each file should contain
a JSON array using `../schema.json`.

Do not include these examples in training. Run `python evaluate.py
--allow-gemini-data` from the service directory to validate the train and test
files, run Qwen on the held-out examples, and ask Gemini to review evidence
grounding. The script sends test transcripts, Qwen responses, and reference
feedback to Google; use only anonymized data that you are authorized to share.
