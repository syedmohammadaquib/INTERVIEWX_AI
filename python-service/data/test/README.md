# Held-out test files

Put human-reviewed examples for final evaluation in this folder, named
`test_data_001.json`, `test_data_002.json`, and so on. Each file should contain
a JSON array using the schema in `../README.md`.

Do not include these examples in training. The current fine-tuning script does
not load or evaluate test files; use a separate evaluation step after training.
