#!/usr/bin/env python3
"""Split interview conversations and reviewed feedback into training files."""

import argparse
import json
import re
from pathlib import Path
from typing import Any

SERVICE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = SERVICE_DIR / "data" / "train"
TRAIN_FILE_PATTERN = re.compile(r"^training_data_(\d+)\.json$")
RATING_FIELDS = {"relevance", "technicalDepth", "clarity", "communicationQuality"}
FEEDBACK_FIELDS = {
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


def validate_example(example: Any, index: int) -> None:
    if not isinstance(example, dict) or set(example) != {"conversation", "feedback"}:
        raise ValueError(
            f"example {index} must contain exactly 'conversation' and 'feedback'; "
            "Q&A-only 'messages' data is not interview-analysis training data"
        )

    conversation = example["conversation"]
    if not isinstance(conversation, list) or not conversation:
        raise ValueError(f"example {index} needs a non-empty conversation")
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
                f"example {index} has an invalid conversation message; "
                "each needs type 'assistant' or 'user' and non-empty content"
            )

    feedback = example["feedback"]
    if not isinstance(feedback, dict) or set(feedback) != FEEDBACK_FIELDS:
        raise ValueError(f"example {index} feedback does not have the expected judgment fields")
    rating = feedback["rating"]
    if not isinstance(rating, dict) or set(rating) != RATING_FIELDS:
        raise ValueError(f"example {index} feedback.rating must contain all four rating fields")
    for field, score in rating.items():
        if not isinstance(score, int) or isinstance(score, bool) or not 1 <= score <= 5:
            raise ValueError(f"example {index} feedback.rating.{field} must be an integer from 1 to 5")

    for field in (
        "summary",
        "recommendationMessage",
        "technicalAssessment",
        "communicationAssessment",
    ):
        value = feedback[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"example {index} feedback.{field} must be a non-empty string")
    recommendation = feedback["recommendation"]
    if not isinstance(recommendation, str) or recommendation not in {"Yes", "No", "Maybe"}:
        raise ValueError(f"example {index} feedback.recommendation must be Yes, No, or Maybe")
    for field in ("strengths", "improvements"):
        value = feedback[field]
        if not isinstance(value, list) or not all(
            isinstance(item, str) and item.strip() for item in value
        ):
            raise ValueError(f"example {index} feedback.{field} must be an array of non-empty strings")
    confidence = feedback["overallConfidence"]
    if (
        not isinstance(confidence, int)
        or isinstance(confidence, bool)
        or not 1 <= confidence <= 100
    ):
        raise ValueError(f"example {index} feedback.overallConfidence must be an integer from 1 to 100")
    summary = feedback["summary"].casefold()
    recommendation_message = feedback["recommendationMessage"].casefold()
    if "converted from codeprep" in summary or "conversion placeholder" in recommendation_message:
        raise ValueError(
            f"example {index} contains generated reference-answer placeholder feedback; "
            "replace it with a real candidate response and human-reviewed analysis"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split conversation-and-feedback examples into data/train/ files."
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="JSON file containing an array of conversation-and-feedback examples",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="examples per output file (default: 5)",
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be greater than zero")

    try:
        examples: Any = json.loads(args.input_file.read_text(encoding="utf-8"))
    except OSError as error:
        parser.error(f"could not read {args.input_file}: {error}")
    except json.JSONDecodeError as error:
        parser.error(
            f"{args.input_file} contains invalid JSON at "
            f"line {error.lineno}, column {error.colno}: {error.msg}"
        )

    if not isinstance(examples, list) or not examples:
        parser.error(f"{args.input_file} must contain a non-empty JSON array")

    try:
        for index, example in enumerate(examples, start=1):
            validate_example(example, index)
    except ValueError as error:
        parser.error(f"Could not convert {args.input_file}: {error}")

    existing_numbers = []
    for path in TRAIN_DIR.iterdir() if TRAIN_DIR.exists() else []:
        match = TRAIN_FILE_PATTERN.fullmatch(path.name)
        if match:
            existing_numbers.append(int(match.group(1)))
    next_number = max(existing_numbers, default=0) + 1

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    for batch_start in range(0, len(examples), args.batch_size):
        batch = examples[batch_start : batch_start + args.batch_size]
        file_number = next_number + batch_start // args.batch_size
        output_file = TRAIN_DIR / f"training_data_{file_number:03d}.json"
        with output_file.open("x", encoding="utf-8") as file:
            json.dump(batch, file, indent=2, ensure_ascii=False)
            file.write("\n")
        print(f"Created {output_file} ({len(batch)} examples)")
    print(
        "Reminder: verify the candidate responses and feedback judgments by hand; "
        "format validation cannot confirm their accuracy."
    )


if __name__ == "__main__":
    main()
