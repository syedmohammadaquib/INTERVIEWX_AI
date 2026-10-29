#!/usr/bin/env python3
"""Validate interview datasets and test Qwen output with Gemini."""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from dotenv import load_dotenv

SERVICE_DIR = Path(__file__).resolve().parent
load_dotenv(SERVICE_DIR / ".env")

service: Any = None


def feedback_validation_errors(feedback: Any) -> list[str]:
    if not isinstance(feedback, dict):
        return ["feedback must be an object"]

    required = {
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
    errors = [f"feedback has unexpected field {key!r}" for key in feedback.keys() - required]
    errors.extend(f"feedback is missing field {key!r}" for key in required - feedback.keys())

    rating = feedback.get("rating")
    rating_fields = {"relevance", "technicalDepth", "clarity", "communicationQuality"}
    if not isinstance(rating, dict):
        errors.append("feedback.rating must be an object")
    else:
        errors.extend(f"feedback.rating has unexpected field {key!r}" for key in rating.keys() - rating_fields)
        errors.extend(f"feedback.rating is missing field {key!r}" for key in rating_fields - rating.keys())
        for key, value in rating.items():
            if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 5:
                errors.append(f"feedback.rating.{key} must be an integer from 1 to 5")

    for key in (
        "summary",
        "recommendationMessage",
        "technicalAssessment",
        "communicationAssessment",
    ):
        value = feedback.get(key)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"feedback.{key} must be a non-empty string")

    recommendation = feedback.get("recommendation")
    if not isinstance(recommendation, str) or recommendation not in {"Yes", "No", "Maybe"}:
        errors.append("feedback.recommendation must be Yes, No, or Maybe")

    for key in ("strengths", "improvements"):
        value = feedback.get(key)
        if not isinstance(value, list) or not all(
            isinstance(item, str) and item.strip() for item in value
        ):
            errors.append(f"feedback.{key} must be an array of non-empty strings")

    confidence = feedback.get("overallConfidence")
    if (
        not isinstance(confidence, int)
        or isinstance(confidence, bool)
        or not 1 <= confidence <= 100
    ):
        errors.append("feedback.overallConfidence must be an integer from 1 to 100")
    return errors


def load_and_validate_examples(
    folder: Path,
    pattern: str,
    label: str,
) -> list[dict[str, Any]]:
    files = sorted(folder.glob(pattern))
    if not files:
        raise ValueError(f"No {pattern} files found in {folder}")

    examples: list[dict[str, Any]] = []
    for path in files:
        try:
            file_data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{label} file {path.name} has invalid JSON at "
                f"line {error.lineno}, column {error.colno}: {error.msg}"
            ) from error

        if not isinstance(file_data, list):
            raise ValueError(f"{label} file {path.name} must contain a JSON array.")

        if not file_data:
            print(f"Warning: {path.name} contains no examples.")
        for index, example in enumerate(file_data, start=1):
            if not isinstance(example, dict) or set(example) != {"conversation", "feedback"}:
                raise ValueError(
                    f"{label} file {path.name} example {index} must contain "
                    "only conversation and feedback fields."
                )
            conversation = example["conversation"]
            if not isinstance(conversation, list) or not conversation:
                raise ValueError(
                    f"{label} file {path.name} example {index} needs a non-empty conversation."
                )
            for message in conversation:
                if (
                    not isinstance(message, dict)
                    or set(message) != {"type", "content"}
                    or not isinstance(message.get("type"), str)
                    or message["type"] not in {"assistant", "user"}
                    or not isinstance(message["content"], str)
                    or not message["content"].strip()
                ):
                    raise ValueError(
                        f"{label} file {path.name} example {index} has an invalid "
                        "conversation message."
                    )
            feedback_errors = feedback_validation_errors(example["feedback"])
            if feedback_errors:
                raise ValueError(
                    f"{label} file {path.name} example {index} has invalid feedback: "
                    + "; ".join(feedback_errors)
                )
        examples.extend(file_data)

    if not examples:
        raise ValueError(f"No examples found in {label} dataset folder: {folder}")

    print(f"{label} data check: PASS ({len(files)} file(s), {len(examples)} example(s))")
    return examples


def generate_feedback(example: dict[str, Any]) -> dict[str, Any]:
    request = service.FeedbackRequest(
        conversation=[
            service.ConversationMessage(type=message["type"], content=message["content"])
            for message in example["conversation"]
        ]
    )
    response = asyncio.run(service.analyze_interview(request))
    return response.data


def judge_with_gemini(
    api_key: str,
    model_name: str,
    transcript: str,
    reference_feedback: dict[str, Any],
    qwen_feedback: dict[str, Any],
) -> dict[str, Any]:
    judge_prompt = {
        "task": (
            "Evaluate whether the candidate-specific claims in the model feedback are supported "
            "by the interview transcript. Use the reference feedback as human-reviewed context, "
            "but do not assume it is correct when it conflicts with the transcript. Do not score "
            "hiring suitability or infer facts not present in the transcript. Return only JSON "
            "with evidence_grounding_score (integer 0-10), unsupported_claims (array of strings), "
            "and explanation (short string)."
        ),
        "transcript": transcript,
        "human_reviewed_reference_feedback": reference_feedback,
        "qwen_feedback": qwen_feedback,
    }
    payload = {
        "contents": [{"parts": [{"text": json.dumps(judge_prompt, ensure_ascii=False)}]}],
        "generationConfig": {"responseMimeType": "application/json"},
    }
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{quote(model_name, safe='')}:generateContent"
    )
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=90) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini API returned HTTP {error.code}: {body[:1000]}") from error
    except URLError as error:
        raise RuntimeError(f"Could not reach Gemini API: {error.reason}") from error
    except TimeoutError as error:
        raise RuntimeError("Gemini API request timed out.") from error

    try:
        text = result["candidates"][0]["content"]["parts"][0]["text"]
        judgment = json.loads(text)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError("Gemini returned a response that was not valid judge JSON.") from error

    score = judgment.get("evidence_grounding_score")
    claims = judgment.get("unsupported_claims")
    explanation = judgment.get("explanation")
    if (
        not isinstance(score, int)
        or isinstance(score, bool)
        or not 0 <= score <= 10
        or not isinstance(claims, list)
        or not all(isinstance(claim, str) for claim in claims)
        or not isinstance(explanation, str)
    ):
        raise RuntimeError("Gemini judge JSON did not match the required result format.")
    return judgment


def rating_agreement(predicted: dict[str, Any], expected: dict[str, Any]) -> float:
    dimensions = ("relevance", "technicalDepth", "clarity", "communicationQuality")
    mean_absolute_error = sum(
        abs(predicted["rating"][dimension] - expected["rating"][dimension])
        for dimension in dimensions
    ) / len(dimensions)
    return 10 * (1 - mean_absolute_error / 4)


def main_cli() -> int:
    parser = argparse.ArgumentParser(
        description="Validate MY_AI datasets, run Qwen on held-out examples, and ask Gemini to judge grounding."
    )
    parser.add_argument("--data-dir", type=Path, default=SERVICE_DIR / "data")
    parser.add_argument("--model", default=os.getenv("HF_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"))
    parser.add_argument("--gemini-model", default=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"))
    parser.add_argument(
        "--allow-gemini-data",
        action="store_true",
        help="Confirm that anonymized test transcripts and feedback may be sent to Google Gemini.",
    )
    parser.add_argument("--report", type=Path, default=SERVICE_DIR / "evaluation_report.json")
    args = parser.parse_args()

    try:
        test_examples = load_and_validate_examples(
            args.data_dir / "test",
            "test_data_*.json",
            "Test",
        )
        print("Test data checks: PASS")

        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is missing. Add your key to MY_AI/python-service/.env."
            )
        if not args.allow_gemini_data:
            raise ValueError(
                "Evaluation sends test transcripts and labels to Google Gemini. "
                "Only use authorized, anonymized data; rerun with --allow-gemini-data to confirm."
            )

        global service
        import main as service_module

        service = service_module
        os.environ["HF_MODEL"] = args.model
        print(f"Loading Qwen model: {args.model}")
        main.load_model()

        results: list[dict[str, Any]] = []
        for index, example in enumerate(test_examples, start=1):
            print(f"Evaluating test example {index}/{len(test_examples)}...")
            transcript = service.format_conversation_for_analysis(
                [
                    service.ConversationMessage(type=message["type"], content=message["content"])
                    for message in example["conversation"]
                ]
            )
            try:
                qwen_response = generate_feedback(example)
            except service.HTTPException as error:
                if error.status_code != 502:
                    raise RuntimeError(
                        f"Qwen analysis failed for test example {index}: {error.detail}"
                    ) from error
                qwen_response = {}
            qwen_errors = (
                feedback_validation_errors(qwen_response.get("feedback"))
                if isinstance(qwen_response, dict)
                else ["response must be an object"]
            )
            qwen_valid = not qwen_errors
            qwen_feedback = qwen_response.get("feedback", {}) if isinstance(qwen_response, dict) else {}
            expected_feedback = example["feedback"]

            if qwen_valid:
                judgment = judge_with_gemini(
                    api_key,
                    args.gemini_model,
                    transcript,
                    expected_feedback,
                    qwen_feedback,
                )
                score_agreement = rating_agreement(
                    qwen_feedback,
                    expected_feedback,
                )
                recommendation_agreement = (
                    10.0
                    if qwen_feedback["recommendation"] == expected_feedback["recommendation"]
                    else 0.0
                )
                grounding_score = float(judgment["evidence_grounding_score"])
                unsupported_claims = judgment["unsupported_claims"]
                judge_explanation = judgment["explanation"]
            else:
                score_agreement = 0.0
                recommendation_agreement = 0.0
                grounding_score = 0.0
                unsupported_claims = []
                judge_explanation = "Gemini skipped because Qwen feedback did not match the required response schema."

            overall = (
                2.5 * (10.0 if qwen_valid else 0.0)
                + 3.5 * score_agreement
                + 2.0 * recommendation_agreement
                + 2.0 * grounding_score
            ) / 10
            results.append(
                {
                    "example": index,
                    "json_valid": qwen_valid,
                    "score_agreement": round(score_agreement, 2),
                    "recommendation_agreement": round(recommendation_agreement, 2),
                    "evidence_grounding": round(grounding_score, 2),
                    "overall": round(overall, 2),
                    "unsupported_claims": unsupported_claims,
                    "judge_explanation": judge_explanation,
                    "schema_errors": [error.message for error in qwen_errors],
                }
            )

        count = len(results)
        averages = {
            "json_validity": 10 * sum(result["json_valid"] for result in results) / count,
            "score_agreement": sum(result["score_agreement"] for result in results) / count,
            "recommendation_agreement": sum(
                result["recommendation_agreement"] for result in results
            )
            / count,
            "evidence_grounding": sum(result["evidence_grounding"] for result in results)
            / count,
            "overall": sum(result["overall"] for result in results) / count,
        }
        report = {
            "model": args.model,
            "gemini_model": args.gemini_model,
            "test_examples": count,
            "metrics_out_of_10": {key: round(value, 2) for key, value in averages.items()},
            "results": results,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        print("\nEvaluation scores (out of 10):")
        print(f"JSON Validity:             {averages['json_validity']:.2f}/10")
        print(f"Score Agreement:           {averages['score_agreement']:.2f}/10")
        print(f"Evidence Grounding (Gemini):{averages['evidence_grounding']:.2f}/10")
        print(f"Recommendation Agreement:  {averages['recommendation_agreement']:.2f}/10")
        print(f"Overall Evaluation:        {averages['overall']:.2f}/10")
        print(f"Report saved to: {args.report}")
        print("Gemini judge scores are estimates; review results with a human.")
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main_cli())
