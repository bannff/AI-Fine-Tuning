"""
CS-Eval Benchmark Runner

Generic evaluation framework for CS-Eval that can work with any model
implementing the required interface.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Protocol
from dataclasses import dataclass
from datetime import datetime

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from datasets import load_dataset
from utils.evaluation_helpers import (
    save_results, calculate_metrics, generate_report, 
    ProgressTracker, format_question_for_display
)

try:
    from config import MODEL_CONFIG, EVAL_CONFIG, OUTPUT_CONFIG, CATEGORIES
except ImportError:
    # Default configuration if config.py doesn't exist
    MODEL_CONFIG = {"max_tokens": 512, "temperature": 0.1, "top_p": 0.9, "batch_size": 10}
    EVAL_CONFIG = {"max_questions_per_category": None, "categories_to_evaluate": None, "save_intermediate_results": True, "verbose": True}
    OUTPUT_CONFIG = {"results_dir": "results", "save_raw_responses": True, "generate_report": True}
    CATEGORIES = ["Network Security", "Cryptography", "Web Security", "System Security", "Software Security", "Mobile Security", "Database Security", "Cloud Security", "Risk Management", "Digital Forensics", "Security Management"]

class ModelInterface(Protocol):
    """Protocol defining the interface that models must implement for CS-Eval."""
    
    def evaluate_question(
        self, 
        question: str, 
        options: Optional[List[str]] = None,
        context: str = "",
        question_type: str = "multiple_choice"
    ) -> Dict[str, Any]:
        """Evaluate a single question and return structured results."""
        ...
    
    def batch_evaluate(
        self, 
        questions: List[Dict[str, Any]], 
        batch_size: int = 10
    ) -> List[Dict[str, Any]]:
        """Evaluate multiple questions in batches."""
        ...

@dataclass
class CSEvalQuestion:
    """Structured representation of a CS-Eval question."""
    id: str
    question: str
    options: List[str]
    answer: str
    category: str
    subcategory: str
    question_type: str = "multiple_choice"
    context: str = ""

class CSEvalBenchmark:
    """
    CS-Eval benchmark evaluation framework.
    
    Handles dataset loading, question formatting, model evaluation,
    and results processing for the CS-Eval cybersecurity benchmark.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize CS-Eval benchmark.
        
        Args:
            config: Optional configuration dictionary to override defaults
        """
        self.config = {
            "model": MODEL_CONFIG,
            "eval": EVAL_CONFIG,
            "output": OUTPUT_CONFIG
        }
        
        if config:
            self.config.update(config)
        
        self.dataset = None
        self.questions = []
        
    def load_dataset(self, split: str = "test") -> None:
        """
        Load the CS-Eval dataset from Hugging Face.
        
        Args:
            split: Dataset split to load (train/validation/test)
        """
        print("🔄 Loading CS-Eval dataset...")
        
        try:
            self.dataset = load_dataset("cseval/cs-eval", split=split)
            print(f"✅ Loaded {len(self.dataset)} questions from CS-Eval dataset")
            
            # Print dataset info
            if self.config["eval"]["verbose"]:
                print(f"📋 Available columns: {list(self.dataset.features.keys())}")
                
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            print("💡 Make sure you have internet access and Hugging Face authentication")
            raise
    
    def parse_questions(self) -> List[CSEvalQuestion]:
        """
        Parse the raw dataset into structured question objects.
        
        Returns:
            List of parsed CSEvalQuestion objects
        """
        if not self.dataset:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        print("🔄 Parsing questions...")
        questions = []
        
        for i, item in enumerate(self.dataset):
            try:
                # Extract question components based on CS-Eval format
                question_text = item.get('question', '')
                options = [
                    item.get('A', ''),
                    item.get('B', ''), 
                    item.get('C', ''),
                    item.get('D', '')
                ]
                
                # Filter out empty options
                options = [opt for opt in options if opt.strip()]
                
                answer = item.get('answer', 'A')
                category = item.get('category', 'Unknown')
                subcategory = item.get('subcategory', 'Unknown')
                
                # Determine question type
                question_type = "multiple_choice" if len(options) > 2 else "true_false"
                
                question_obj = CSEvalQuestion(
                    id=f"cs_eval_{i}",
                    question=question_text,
                    options=options,
                    answer=answer,
                    category=category,
                    subcategory=subcategory,
                    question_type=question_type
                )
                
                questions.append(question_obj)
                
            except Exception as e:
                if self.config["eval"]["verbose"]:
                    print(f"⚠️ Error parsing question {i}: {e}")
                continue
        
        self.questions = questions
        print(f"✅ Parsed {len(questions)} questions successfully")
        
        return questions
    
    def filter_questions(self, questions: List[CSEvalQuestion]) -> List[CSEvalQuestion]:
        """
        Filter questions based on configuration settings.
        
        Args:
            questions: List of questions to filter
            
        Returns:
            Filtered list of questions
        """
        filtered = questions.copy()
        
        # Filter by categories if specified
        categories_to_eval = self.config["eval"]["categories_to_evaluate"]
        if categories_to_eval:
            filtered = [q for q in filtered if q.category in categories_to_eval]
            print(f"🔍 Filtered to {len(filtered)} questions from specified categories")
        
        # Limit questions per category if specified
        max_per_category = self.config["eval"]["max_questions_per_category"]
        if max_per_category:
            category_counts = {}
            limited_questions = []
            
            for question in filtered:
                category = question.category
                if category_counts.get(category, 0) < max_per_category:
                    limited_questions.append(question)
                    category_counts[category] = category_counts.get(category, 0) + 1
            
            filtered = limited_questions
            print(f"🔍 Limited to max {max_per_category} questions per category: {len(filtered)} total")
        
        return filtered
    
    def format_question_for_model(self, question: CSEvalQuestion) -> Dict[str, Any]:
        """
        Format a question for model evaluation.
        
        Args:
            question: CSEvalQuestion object
            
        Returns:
            Dictionary formatted for model.evaluate_question()
        """
        return {
            "question": question.question,
            "options": question.options if question.options else None,
            "context": f"Category: {question.category}",
            "question_type": question.question_type
        }
    
    def evaluate_model(self, model: ModelInterface) -> Dict[str, Any]:
        """
        Run complete evaluation of a model on CS-Eval.
        
        Args:
            model: Model implementing ModelInterface protocol
            
        Returns:
            Complete evaluation results
        """
        if not self.questions:
            raise ValueError("No questions loaded. Call load_dataset() and parse_questions() first.")
        
        print(f"🚀 Starting CS-Eval evaluation with {len(self.questions)} questions...")
        
        # Filter questions based on config
        eval_questions = self.filter_questions(self.questions)
        print(f"📊 Evaluating {len(eval_questions)} questions")
        
        # Prepare questions for batch evaluation
        model_questions = [self.format_question_for_model(q) for q in eval_questions]
        
        # Run evaluation with progress tracking
        progress = ProgressTracker(len(eval_questions), "CS-Eval Progress")
        
        try:
            batch_size = self.config["model"]["batch_size"]
            results = []
            
            for i in range(0, len(model_questions), batch_size):
                batch = model_questions[i:i + batch_size]
                batch_results = model.batch_evaluate(batch, batch_size)
                
                # Add ground truth and metadata to results
                for j, result in enumerate(batch_results):
                    question_idx = i + j
                    original_question = eval_questions[question_idx]
                    
                    result.update({
                        "id": original_question.id,
                        "ground_truth": original_question.answer,
                        "category": original_question.category,
                        "subcategory": original_question.subcategory
                    })
                
                results.extend(batch_results)
                progress.update(len(batch))
                
                # Save intermediate results if configured
                if self.config["eval"]["save_intermediate_results"] and i % (batch_size * 10) == 0:
                    self._save_intermediate_results(results, i + len(batch))
        
        except Exception as e:
            print(f"\n❌ Error during evaluation: {e}")
            raise
        finally:
            progress.finish()
        
        # Calculate metrics
        print("📊 Calculating evaluation metrics...")
        predictions = [r.get("parsed_response", "") for r in results]
        ground_truth = [r.get("ground_truth", "") for r in results]
        categories = [r.get("category", "") for r in results]
        
        metrics = calculate_metrics(predictions, ground_truth, categories)
        
        # Save results
        output_dir = self.config["output"]["results_dir"]
        metadata = {
            "config": self.config,
            **metrics
        }
        
        results_path = save_results(
            results, 
            output_dir, 
            "cs-eval", 
            "unknown-model",  # Will be overridden by specific implementations
            metadata
        )
        
        # Generate human-readable report if configured
        if self.config["output"]["generate_report"]:
            report = generate_report(results, metrics, "CS-Eval", "unknown-model")
            report_path = os.path.join(output_dir, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
            
            with open(report_path, 'w') as f:
                f.write(report)
            
            print(f"📄 Evaluation report saved to {report_path}")
        
        print(f"\n✅ CS-Eval evaluation complete!")
        print(f"📊 Overall Accuracy: {metrics['overall_accuracy']:.2%}")
        print(f"✅ Correct: {metrics['correct_answers']}/{metrics['total_questions']}")
        
        return {
            "results": results,
            "metrics": metrics,
            "config": self.config,
            "results_path": results_path
        }
    
    def _save_intermediate_results(self, results: List[Dict[str, Any]], progress: int) -> None:
        """Save intermediate results during evaluation."""
        output_dir = self.config["output"]["results_dir"]
        intermediate_path = os.path.join(output_dir, f"intermediate_results_{progress}.json")
        
        with open(intermediate_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        if self.config["eval"]["verbose"]:
            print(f"\n💾 Intermediate results saved: {progress} questions completed")

def main():
    """Main function for running CS-Eval evaluation from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run CS-Eval cybersecurity benchmark")
    parser.add_argument("--categories", type=str, help="Comma-separated list of categories to evaluate")
    parser.add_argument("--max_questions", type=int, help="Maximum questions per category")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for evaluation")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory for results")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Update configuration based on command line arguments
    config_updates = {
        "eval": {},
        "model": {"batch_size": args.batch_size},
        "output": {"results_dir": args.output_dir}
    }
    
    if args.categories:
        config_updates["eval"]["categories_to_evaluate"] = [cat.strip() for cat in args.categories.split(",")]
    
    if args.max_questions:
        config_updates["eval"]["max_questions_per_category"] = args.max_questions
    
    if args.verbose:
        config_updates["eval"]["verbose"] = True
    
    # Initialize benchmark
    benchmark = CSEvalBenchmark(config_updates)
    
    # Load and parse dataset
    benchmark.load_dataset()
    benchmark.parse_questions()
    
    print("❌ No model specified. Please use evaluate_phi3.py or implement your own model interface.")
    print("💡 See README.md for instructions on implementing custom model evaluation.")

if __name__ == "__main__":
    main()
