#!/usr/bin/env python3
"""
Fine-tuning script for Hugging Face models on interview feedback data.
This script fine-tunes an instruction-tuned causal language model on interview conversations
to generate structured feedback similar to the current Groq analysis.

Usage:
    python fine_tune.py --model_name Qwen/Qwen2.5-1.5B-Instruct --dataset_path ./data --output_dir ./fine_tuned_model

Requirements:
    pip install transformers torch datasets accelerate peft bitsandbytes
"""

import argparse
import json
import os
from typing import List, Dict, Any

import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    AutoModelForSeq2SeqLM
)
from datasets import Dataset, DatasetDict
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import numpy as np


class InterviewFeedbackDataset:
    """Custom dataset class for interview feedback fine-tuning."""

    def __init__(self, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def format_conversation(self, conversation: List[Dict], feedback: Dict) -> str:
        """Format conversation and feedback into training text."""
        conv_text = "\n".join([
            f"{'User' if msg['type'] == 'user' else 'Assistant'}: {msg['content']}"
            for msg in conversation
        ])

        feedback_text = f"""
Relevance: {feedback['rating']['relevance']}/5
Technical Depth: {feedback['rating']['technicalDepth']}/5
Clarity: {feedback['rating']['clarity']}/5
Communication Quality: {feedback['rating']['communicationQuality']}/5

Summary: {feedback['summary']}

Recommendation: {feedback['recommendation']}
Justification: {feedback['recommendationMessage']}

Strengths: {', '.join(feedback['strengths'])}
Improvements: {', '.join(feedback['improvements'])}

Technical Assessment: {feedback['technicalAssessment']}
Communication Assessment: {feedback['communicationAssessment']}
Confidence: {feedback['overallConfidence']}%
"""

        return f"Interview Conversation:\n{conv_text}\n\nFeedback Analysis:\n{feedback_text.strip()}"

    def tokenize_function(self, examples):
        """Tokenize the formatted text for training."""
        texts = examples["text"]
        return self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )


def load_dataset_from_json(json_path: str, tokenizer, max_samples: int = None) -> Dataset:
    """
    Load interview feedback dataset from JSON file.

    Expected JSON format:
    [
        {
            "conversation": [{"type": "user", "content": "..."}, ...],
            "feedback": {...}  // same structure as the analysis output
        },
        ...
    ]
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if max_samples:
        data = data[:max_samples]

    dataset_processor = InterviewFeedbackDataset(tokenizer)

    # Format conversations and feedback into training texts
    texts = []
    for item in data:
        formatted_text = dataset_processor.format_conversation(
            item["conversation"],
            item["feedback"]
        )
        texts.append(formatted_text)

    # Create dataset
    dataset = Dataset.from_dict({"text": texts})

    # Tokenize
    tokenized_dataset = dataset.map(
        dataset_processor.tokenize_function,
        batched=True,
        remove_columns=["text"]
    )

    return tokenized_dataset


def setup_quantization_config():
    """Setup quantization configuration for memory-efficient training."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def setup_lora_config(task_type: str):
    """Setup LoRA configuration for parameter-efficient fine-tuning."""
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=task_type,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )


def main(args):
    """Main fine-tuning function."""

    print("🚀 Starting model fine-tuning for interview feedback analysis...")
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Output: {args.output_dir}")

    # Set up device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Set padding token if not exists (common for some models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print("Loading dataset...")
    train_dataset = load_dataset_from_json(
        os.path.join(args.dataset_path, "train.json"),
        tokenizer,
        args.max_samples
    )

    val_dataset = None
    if os.path.exists(os.path.join(args.dataset_path, "val.json")):
        val_dataset = load_dataset_from_json(
            os.path.join(args.dataset_path, "val.json"),
            tokenizer,
            args.max_samples
        )

    # Load model with quantization
    print("Loading model...")
    quantization_config = setup_quantization_config() if args.use_quantization else None

    model_config = AutoConfig.from_pretrained(args.model_name)
    if model_config.is_encoder_decoder:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name,
            quantization_config=quantization_config,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        lora_task_type = "SEQ_2_SEQ_LM"
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=quantization_config,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        lora_task_type = "CAUSAL_LM"

    # Prepare model for k-bit training if using quantization
    if args.use_quantization:
        model = prepare_model_for_kbit_training(model)

    # Apply LoRA if specified
    if args.use_lora:
        print("Applying LoRA configuration...")
        lora_config = setup_lora_config(lora_task_type)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Set up data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Not using masked language modeling
    )

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        evaluation_strategy="steps" if val_dataset else "no",
        eval_steps=args.eval_steps if val_dataset else None,
        save_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="eval_loss" if val_dataset else None,
        greater_is_better=False if val_dataset else None,
        fp16=device == "cuda",
        dataloader_pin_memory=False,
        report_to="none",  # Disable wandb/tensorboard logging
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # Start training
    print("Starting training...")
    trainer.train()

    # Save the fine-tuned model
    print("Saving fine-tuned model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save model configuration for inference
    config = {
        "model_name": args.model_name,
        "fine_tuned": True,
        "task": "interview_feedback_analysis",
        "max_length": args.max_length,
    }

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("✅ Fine-tuning complete!")
    print(f"Model saved to: {args.output_dir}")

    # Print training summary
    if trainer.state.log_history:
        final_loss = trainer.state.log_history[-1].get("train_loss", "N/A")
        print(f"Final training loss: {final_loss}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Hugging Face model for interview feedback")

    # Model arguments
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                       help="Hugging Face model name to fine-tune")
    parser.add_argument("--output_dir", type=str, default="./fine_tuned_model",
                       help="Output directory for fine-tuned model")

    # Dataset arguments
    parser.add_argument("--dataset_path", type=str, default="./data",
                       help="Path to dataset directory containing train.json and val.json")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to use for training")

    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=500,
                       help="Number of warmup steps")
    parser.add_argument("--max_length", type=int, default=1024,
                       help="Maximum sequence length")

    # Logging and saving
    parser.add_argument("--logging_steps", type=int, default=100,
                       help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=500,
                       help="Save steps")
    parser.add_argument("--eval_steps", type=int, default=500,
                       help="Evaluation steps")

    # Optimization arguments
    parser.add_argument("--use_quantization", action="store_true",
                       help="Use 4-bit quantization for memory efficiency")
    parser.add_argument("--use_lora", action="store_true", default=True,
                       help="Use LoRA for parameter-efficient fine-tuning")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
