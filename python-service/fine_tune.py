#!/usr/bin/env python3
"""Fine-tune a Qwen model to analyze interviews and produce feedback JSON."""

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SERVICE_DIR = Path(__file__).resolve().parent
DATA_DIR = SERVICE_DIR / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen to produce interview feedback from labeled transcripts."
    )
    parser.add_argument(
        "--model_name",
        default=None,
        help="Hugging Face model name or local model directory",
    )
    parser.add_argument(
        "--dataset_path",
        type=Path,
        default=DATA_DIR,
        help="Dataset root containing train/, a training directory, or one JSON file",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=SERVICE_DIR / "fine_tuned_model",
        help="Directory where the fine-tuned model or LoRA adapter is saved",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_epochs", type=float, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=Path,
        default=None,
        help="Resume an interrupted Trainer run from a saved checkpoint directory",
    )
    parser.add_argument(
        "--use_lora",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use LoRA adapters (default: enabled; pass --no-use_lora to disable)",
    )
    parser.add_argument(
        "--use_quantization",
        action="store_true",
        help="Use 4-bit bitsandbytes quantization (requires supported CUDA setup)",
    )
    return parser.parse_args()


def find_training_files(dataset_path: Path) -> list[Path]:
    if dataset_path.is_file():
        return [dataset_path]

    training_dir = dataset_path / "train" if (dataset_path / "train").is_dir() else dataset_path
    files = sorted(training_dir.glob("training_data_*.json"))
    if not files:
        raise FileNotFoundError(
            f"No training_data_*.json files found in {training_dir}. "
            "Add reviewed examples to data/train/ before starting training."
        )
    return files


def example_fingerprint(example: dict[str, Any]) -> str:
    canonical = json.dumps(
        example,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_training_examples(dataset_path: Path, max_samples: int | None) -> list[dict[str, Any]]:
    if max_samples is not None and max_samples < 1:
        raise ValueError("--max_samples must be greater than zero.")

    examples: list[dict[str, Any]] = []
    for path in find_training_files(dataset_path):
        try:
            file_data = json.loads(path.read_text(encoding="utf-8"))
        except OSError as error:
            raise RuntimeError(f"Could not read training file {path}: {error}") from error
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{path.name} has invalid JSON at line {error.lineno}, "
                f"column {error.colno}: {error.msg}"
            ) from error

        if not isinstance(file_data, list):
            raise ValueError(f"{path.name} must contain a JSON array of labeled interviews.")
        if not file_data:
            print(f"Warning: skipping empty training file {path.name}.")
            continue
        for index, example in enumerate(file_data, start=1):
            if not isinstance(example, dict) or set(example) != {"conversation", "feedback"}:
                raise ValueError(
                    f"{path.name} example {index} must contain 'conversation' and 'feedback'. "
                    "Q&A-only data trains question answering, not interview analysis."
                )
            conversation = example["conversation"]
            if not isinstance(conversation, list) or not conversation:
                raise ValueError(f"{path.name} example {index} needs a non-empty conversation.")
            for message in conversation:
                if (
                    not isinstance(message, dict)
                    or set(message) != {"type", "content"}
                    or not isinstance(message.get("type"), str)
                    or message["type"] not in {"assistant", "user"}
                    or not isinstance(message.get("content"), str)
                    or not message["content"].strip()
                ):
                    raise ValueError(
                        f"{path.name} example {index} has an invalid conversation message."
                    )

            feedback = example["feedback"]
            required_feedback = {
                "rating",
                "summary",
                "recommendation",
                "recommendationMessage",
                "strengths",
                "improvements",
                "technicalAssessment",
                "communicationAssessment",
                "overallConfidence",
            }
            if not isinstance(feedback, dict) or set(feedback) != required_feedback:
                raise ValueError(
                    f"{path.name} example {index} feedback is missing required judgment fields."
                )
            rating = feedback["rating"]
            rating_fields = {
                "relevance",
                "technicalDepth",
                "clarity",
                "communicationQuality",
            }
            if not isinstance(rating, dict) or set(rating) != rating_fields:
                raise ValueError(f"{path.name} example {index} has an invalid feedback.rating.")
            if any(
                not isinstance(score, int) or isinstance(score, bool) or not 1 <= score <= 5
                for score in rating.values()
            ):
                raise ValueError(f"{path.name} example {index} ratings must be integers from 1 to 5.")
            if not isinstance(feedback["recommendation"], str) or feedback["recommendation"] not in {
                "Yes",
                "No",
                "Maybe",
            }:
                raise ValueError(f"{path.name} example {index} has an invalid recommendation.")
            for field in (
                "summary",
                "recommendationMessage",
                "technicalAssessment",
                "communicationAssessment",
            ):
                if not isinstance(feedback[field], str) or not feedback[field].strip():
                    raise ValueError(f"{path.name} example {index} feedback.{field} must be non-empty.")
            for field in ("strengths", "improvements"):
                if not isinstance(feedback[field], list) or not all(
                    isinstance(item, str) and item.strip() for item in feedback[field]
                ):
                    raise ValueError(f"{path.name} example {index} feedback.{field} is invalid.")
            confidence = feedback["overallConfidence"]
            if (
                not isinstance(confidence, int)
                or isinstance(confidence, bool)
                or not 1 <= confidence <= 100
            ):
                raise ValueError(
                    f"{path.name} example {index} overallConfidence must be an integer from 1 to 100."
                )
            if (
                "converted from codeprep" in feedback["summary"].casefold()
                or "conversion placeholder"
                in feedback["recommendationMessage"].casefold()
            ):
                raise ValueError(
                    f"{path.name} example {index} contains generated reference-answer "
                    "placeholder feedback. Replace it with a real candidate response "
                    "and human-reviewed analysis before training."
                )
        examples.extend(file_data)

    if max_samples is not None:
        examples = examples[:max_samples]
    if not examples:
        raise ValueError(
            f"No training examples found in {dataset_path}. "
            "The JSON files must contain non-empty arrays of reviewed examples."
        )
    return examples


def load_training_state(output_dir: Path) -> dict[str, Any] | None:
    metadata_path = output_dir / "fine_tune_metadata.json"
    if not metadata_path.exists():
        if (output_dir / "adapter_config.json").exists():
            raise ValueError(
                f"Found an existing adapter in {output_dir} but no training metadata. "
                "Restore fine_tune_metadata.json before continuing to avoid retraining data."
            )
        return None

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read training metadata {metadata_path}: {error}") from error
    if not isinstance(metadata, dict) or not isinstance(metadata.get("model_name"), str):
        raise ValueError(f"Training metadata is invalid: {metadata_path}")
    fingerprints = metadata.get("trained_example_hashes")
    if not isinstance(fingerprints, list) or not all(
        isinstance(fingerprint, str) for fingerprint in fingerprints
    ):
        raise ValueError(
            f"{metadata_path} does not track trained examples. "
            "Refusing to train again because previously used records cannot be identified safely."
        )
    return metadata


def select_unseen_examples(
    examples: list[dict[str, Any]],
    trained_fingerprints: set[str],
) -> tuple[list[dict[str, Any]], set[str]]:
    unseen: list[dict[str, Any]] = []
    seen = set(trained_fingerprints)
    duplicates = 0
    for example in examples:
        fingerprint = example_fingerprint(example)
        if fingerprint in seen:
            duplicates += 1
            continue
        seen.add(fingerprint)
        unseen.append(example)
    if duplicates:
        print(f"Skipping {duplicates} already-trained or duplicate example(s).")
    return unseen, seen


def remove_completed_checkpoints(output_dir: Path) -> None:
    for path in output_dir.glob("checkpoint-*"):
        if path.is_dir() and re.fullmatch(r"checkpoint-\d+", path.name):
            shutil.rmtree(path)

    checkpoint_root = output_dir / "training_checkpoints"
    if checkpoint_root.is_dir():
        shutil.rmtree(checkpoint_root)


class InterviewAnalysisDataset:
    """Render interview inputs and mask tokens so only feedback is learned."""

    def __init__(
        self,
        tokenizer: Any,
        max_length: int,
        system_instruction: str,
        analysis_template: str,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_instruction = system_instruction
        self.analysis_template = analysis_template

    def format_messages(self, example: dict[str, Any]) -> list[dict[str, str]]:
        transcript = "\n".join(
            f"{'Candidate' if message['type'] == 'user' else 'Interviewer'}: "
            f"{message['content']}"
            for message in example["conversation"]
        )
        prompt = self.analysis_template.replace("{conversation}", transcript)
        target = json.dumps({"feedback": example["feedback"]}, ensure_ascii=False)
        return [
            {"role": "system", "content": self.system_instruction},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ]

    def tokenize(self, examples: dict[str, list[list[dict[str, str]]]]) -> dict[str, Any]:
        tokenized: dict[str, list[list[int]]] = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("The tokenizer must define a pad token before tokenization.")

        for messages in examples["messages"]:
            prompt_messages = messages[:-1]
            prompt_ids = self.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=False,
            )
            if input_ids[: len(prompt_ids)] != prompt_ids:
                raise ValueError(
                    "The tokenizer chat template does not preserve the prompt "
                    "as a prefix of the full conversation."
                )
            if len(input_ids) > self.max_length:
                raise ValueError(
                    f"A chat example is {len(input_ids)} tokens, exceeding "
                    f"--max_length {self.max_length}. Increase --max_length."
                )

            padding_length = self.max_length - len(input_ids)
            labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids) :]
            tokenized["input_ids"].append(input_ids + [pad_token_id] * padding_length)
            tokenized["attention_mask"].append([1] * len(input_ids) + [0] * padding_length)
            tokenized["labels"].append(labels + [-100] * padding_length)
        return tokenized


def load_ml_dependencies() -> tuple[Any, ...]:
    try:
        import torch
        from datasets import Dataset
        from peft import (
            LoraConfig,
            PeftModel,
            get_peft_model,
            prepare_model_for_kbit_training,
        )
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
            default_data_collator,
        )
    except ImportError as error:
        raise RuntimeError(
            "Fine-tuning dependencies are missing. From the service directory, run "
            "`python -m pip install -r requirements.txt`, then retry."
        ) from error
    return (
        torch,
        Dataset,
        LoraConfig,
        PeftModel,
        get_peft_model,
        prepare_model_for_kbit_training,
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        (BitsAndBytesConfig, Trainer, TrainingArguments, default_data_collator),
    )


def train(args: argparse.Namespace, examples: list[dict[str, Any]]) -> None:
    if args.num_epochs <= 0 or args.batch_size < 1:
        raise ValueError("--num_epochs and --batch_size must be greater than zero.")
    if args.gradient_accumulation_steps < 1 or args.max_length < 1:
        raise ValueError(
            "--gradient_accumulation_steps and --max_length must be greater than zero."
        )
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.warmup_steps < 0:
        raise ValueError(
            "--learning_rate must be positive; --weight_decay and --warmup_steps "
            "must be zero or greater."
        )
    if args.logging_steps < 1:
        raise ValueError("--logging_steps must be greater than zero.")
    if args.resume_from_checkpoint is not None:
        checkpoint = args.resume_from_checkpoint
        if not checkpoint.is_dir() or not (checkpoint / "trainer_state.json").is_file():
            raise ValueError(
                f"Checkpoint {checkpoint} is missing or has no trainer_state.json."
            )

    prior_state = load_training_state(args.output_dir)
    model_name = args.model_name or (
        prior_state["model_name"] if prior_state else "Qwen/Qwen2.5-1.5B-Instruct"
    )
    if prior_state and prior_state["model_name"] != model_name:
        raise ValueError(
            f"Saved adapter uses {prior_state['model_name']}; requested model is {model_name}."
        )
    if prior_state and not args.use_lora:
        raise ValueError("Continuing the saved LoRA adapter requires LoRA to remain enabled.")

    old_fingerprints = set(prior_state["trained_example_hashes"]) if prior_state else set()
    new_examples, all_fingerprints = select_unseen_examples(examples, old_fingerprints)
    if not new_examples:
        print("No new training examples found; the saved adapter was left unchanged.")
        return

    (
        torch,
        Dataset,
        LoraConfig,
        PeftModel,
        get_peft_model,
        prepare_model_for_kbit_training,
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        training_components,
    ) = load_ml_dependencies()
    BitsAndBytesConfig, Trainer, TrainingArguments, default_data_collator = training_components

    if args.use_quantization and not torch.cuda.is_available():
        raise RuntimeError("4-bit quantization requires a supported CUDA GPU.")

    print(f"New training examples: {len(new_examples)}")
    print(f"Model: {model_name}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("The tokenizer must define either a pad token or an EOS token.")
        tokenizer.pad_token = tokenizer.eos_token

    try:
        instructions = json.loads(
            (SERVICE_DIR / "instructions.json").read_text(encoding="utf-8")
        )
        rubric = json.loads((SERVICE_DIR / "rubric.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not load interview instructions or rubric: {error}") from error
    system_instruction = instructions.get("system")
    analysis_template = instructions.get("analysis_template")
    if not isinstance(system_instruction, str) or not isinstance(analysis_template, str):
        raise ValueError("instructions.json must define string 'system' and 'analysis_template' values.")
    system_instruction = (
        f"{system_instruction}\n\nSCORING RUBRIC:\n"
        f"{json.dumps(rubric, ensure_ascii=False, indent=2)}"
    )

    processor = InterviewAnalysisDataset(
        tokenizer,
        args.max_length,
        system_instruction,
        analysis_template,
    )
    dataset = Dataset.from_dict(
        {"messages": [processor.format_messages(example) for example in new_examples]}
    )
    train_dataset = dataset.map(
        processor.tokenize,
        batched=True,
        remove_columns=["messages"],
        desc="Tokenizing training examples",
    )

    config = AutoConfig.from_pretrained(model_name)
    if config.is_encoder_decoder:
        raise ValueError(
            "This training script supports causal language models, not encoder-decoder models."
        )

    quantization_config = None
    if args.use_quantization:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto" if torch.cuda.is_available() else None,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    if args.use_quantization:
        model = prepare_model_for_kbit_training(model)

    if args.use_lora:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        if prior_state:
            model = PeftModel.from_pretrained(
                model,
                str(args.output_dir),
                is_trainable=True,
            )
        else:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            )
            model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.config.use_cache = False
    steps_per_epoch = math.ceil(
        len(train_dataset) / (args.batch_size * args.gradient_accumulation_steps)
    )
    total_steps = max(1, math.ceil(steps_per_epoch * args.num_epochs))
    run_started_at = datetime.now(timezone.utc)
    checkpoint_dir = (
        args.resume_from_checkpoint.parent
        if args.resume_from_checkpoint is not None
        else args.output_dir / "training_checkpoints" / run_started_at.strftime(
            "%Y%m%dT%H%M%SZ"
        )
    )
    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=min(args.warmup_steps, total_steps - 1),
        logging_steps=max(1, args.logging_steps),
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
    )
    trainer.train(
        resume_from_checkpoint=(
            str(args.resume_from_checkpoint)
            if args.resume_from_checkpoint is not None
            else None
        )
    )
    model.config.use_cache = True
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    metadata = {
        "model_name": model_name,
        "task": "interview_feedback_analysis",
        "fine_tuned": True,
        "method": "LoRA" if args.use_lora else "full",
        "training_examples": len(all_fingerprints),
        "trained_example_hashes": sorted(all_fingerprints),
        "max_length": args.max_length,
        "last_trained_at": run_started_at.isoformat(),
    }
    (args.output_dir / "fine_tune_metadata.json").write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )
    remove_completed_checkpoints(args.output_dir)
    print(f"Training complete. Model or LoRA adapter saved to {args.output_dir}")


def main() -> int:
    args = parse_args()
    try:
        examples = load_training_examples(args.dataset_path, args.max_samples)
        train(args, examples)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
