# MY_AI — Quick Start Guide

**Project:** `INTERVIEWX/MY_AI/python-service`

## 1. Navigate to the Project

Open CMD or PowerShell and run:

```bat
cd /d D:\INTERVIEWX\MY_AI\python-service
```

*Note: In PowerShell, use `cd D:\INTERVIEWX\MY_AI\python-service` instead.*

## 2. Activate the Virtual Environment

Activate the existing project-level virtual environment.

**CMD:**
```bat
D:\INTERVIEWX\.venv\Scripts\activate.bat
```

**PowerShell:**
```powershell
& D:\INTERVIEWX\.venv\Scripts\Activate.ps1
```

## 3. Start the API Server

```bash
python main.py
```

- **Health check:** Open `/health` on the running server.
- **API testing:** Open `/docs` to test endpoints using Swagger UI.

Use the actual host and port configured in `main.py` (commonly `http://127.0.0.1:8000`).

## 4. Prepare Training Data

Convert a JSON array of interviewer/candidate conversations with human-reviewed
feedback into numbered training files:

```bash
python converter.py reviewed_examples.json
```

To create batches of five examples each:

```bash
python converter.py reviewed_examples.json --batch-size 5
```

- Output directory: `data/train/`
- Files use the `training_data_001.json`, `training_data_002.json`, and so on naming convention.
- Existing numbered files are not overwritten.
- The converter continues numbering after existing training files.
- Each example needs the interview `conversation` and human-reviewed `feedback` shown in `data/schema.json`.
- Keep genuinely held-out examples separate in `data/test/`; do not include them in training data.

## 5. Run the Evaluation

```bash
python testing.py --allow-gemini-data
```

**Important:** This option allows test data to be sent to Gemini. Anonymize sensitive information before running the evaluation and verify what data the script transmits.

Review the outputs to check whether the AI's interview feedback follows your rubric and matches the expected results.

## 6. Fine-Tuning Status

Once `data/train/` contains actual interviewer/candidate conversations with human-reviewed feedback, start fine-tuning from this directory:

```bash
python fine_tune.py
```

The script validates every training file before loading the model. It uses
LoRA by default to reduce the number of trainable parameters and saves the
result under `fine_tuned_model/`. You can choose another model or output folder:

```bash
python fine_tune.py --model_name Qwen/Qwen2.5-1.5B-Instruct --output_dir ./fine_tuned_model
```

Install dependencies first with `python -m pip install -r requirements.txt`.
Training requires a compatible Hugging Face model and can be slow or memory
intensive, especially on CPU. The fine-tuning script saves a LoRA adapter by
default. Later runs skip examples recorded as previously trained and continue
from the saved adapter using only new examples. If no new examples are found,
the adapter stays unchanged. After a successful run, intermediate checkpoints
are removed; interrupted runs retain their checkpoints for resuming. The API
does not automatically load the adapter.
