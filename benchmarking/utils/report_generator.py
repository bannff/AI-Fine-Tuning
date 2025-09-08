"""
Report Generator

Utilities for generating comprehensive evaluation reports across different benchmarks.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

def generate_benchmark_summary(
    benchmark_name: str,
    results: Dict[str, Any],
    model_name: str
) -> str:
    """
    Generate a comprehensive summary report for a benchmark evaluation.
    
    Args:
        benchmark_name: Name of the benchmark
        results: Complete results dictionary
        model_name: Name of the evaluated model
        
    Returns:
        Formatted summary report
    """
    metrics = results.get("metrics", {})
    config = results.get("config", {})
    
    report = f"""# {benchmark_name} Evaluation Summary

**Model**: {model_name}
**Benchmark**: {benchmark_name}
**Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary

The {model_name} model achieved an overall accuracy of **{metrics.get('overall_accuracy', 0):.1%}** on the {benchmark_name} benchmark, correctly answering {metrics.get('correct_answers', 0)} out of {metrics.get('total_questions', 0)} questions.

"""
    
    # Performance breakdown by category
    if 'category_metrics' in metrics:
        report += "## Performance by Category\n\n"
        
        # Sort categories by accuracy (descending)
        categories = sorted(
            metrics['category_metrics'].items(),
            key=lambda x: x[1]['accuracy'],
            reverse=True
        )
        
        report += "| Category | Accuracy | Correct | Total |\n"
        report += "|----------|----------|---------|-------|\n"
        
        for category, cat_metrics in categories:
            accuracy = cat_metrics['accuracy']
            correct = cat_metrics['correct_answers']
            total = cat_metrics['total_questions']
            report += f"| {category} | {accuracy:.1%} | {correct} | {total} |\n"
        
        # Identify strengths and weaknesses
        best_category = categories[0]
        worst_category = categories[-1]
        
        report += f"\n### Key Findings\n\n"
        report += f"- **Strongest Area**: {best_category[0]} ({best_category[1]['accuracy']:.1%} accuracy)\n"
        report += f"- **Area for Improvement**: {worst_category[0]} ({worst_category[1]['accuracy']:.1%} accuracy)\n"
        
        # Calculate performance variance
        accuracies = [cat['accuracy'] for cat in metrics['category_metrics'].values()]
        if len(accuracies) > 1:
            variance = max(accuracies) - min(accuracies)
            report += f"- **Performance Variance**: {variance:.1%} (difference between best and worst categories)\n"
        
        report += "\n"
    
    # Configuration details
    if config:
        report += "## Configuration\n\n"
        
        model_config = config.get('model', {})
        if model_config:
            report += "### Model Parameters\n"
            for param, value in model_config.items():
                report += f"- **{param.replace('_', ' ').title()}**: {value}\n"
            report += "\n"
        
        eval_config = config.get('eval', {})
        if eval_config:
            report += "### Evaluation Settings\n"
            for param, value in eval_config.items():
                if value is not None:
                    report += f"- **{param.replace('_', ' ').title()}**: {value}\n"
            report += "\n"
    
    return report

def generate_comparison_report(
    results_list: List[Dict[str, Any]],
    benchmark_name: str
) -> str:
    """
    Generate a comparison report for multiple model evaluations.
    
    Args:
        results_list: List of evaluation results from different models
        benchmark_name: Name of the benchmark
        
    Returns:
        Formatted comparison report
    """
    report = f"""# {benchmark_name} Model Comparison Report

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Models Compared**: {len(results_list)}

## Overall Performance Comparison

| Model | Accuracy | Correct Answers | Total Questions |
|-------|----------|-----------------|-----------------|
"""
    
    # Sort models by accuracy
    sorted_results = sorted(
        results_list,
        key=lambda x: x.get('metrics', {}).get('overall_accuracy', 0),
        reverse=True
    )
    
    for result in sorted_results:
        model_name = result.get('model', 'Unknown')
        metrics = result.get('metrics', {})
        accuracy = metrics.get('overall_accuracy', 0)
        correct = metrics.get('correct_answers', 0)
        total = metrics.get('total_questions', 0)
        
        report += f"| {model_name} | {accuracy:.1%} | {correct} | {total} |\n"
    
    # Category-wise comparison if available
    if all('category_metrics' in result.get('metrics', {}) for result in sorted_results):
        report += "\n## Category Performance Comparison\n\n"
        
        # Get all categories
        all_categories = set()
        for result in sorted_results:
            all_categories.update(result['metrics']['category_metrics'].keys())
        
        for category in sorted(all_categories):
            report += f"### {category}\n\n"
            report += "| Model | Accuracy | Correct | Total |\n"
            report += "|-------|----------|---------|-------|\n"
            
            category_results = []
            for result in sorted_results:
                model_name = result.get('model', 'Unknown')
                cat_metrics = result['metrics']['category_metrics'].get(category, {})
                
                if cat_metrics:
                    accuracy = cat_metrics['accuracy']
                    correct = cat_metrics['correct_answers']
                    total = cat_metrics['total_questions']
                    category_results.append((model_name, accuracy, correct, total))
            
            # Sort by accuracy for this category
            category_results.sort(key=lambda x: x[1], reverse=True)
            
            for model_name, accuracy, correct, total in category_results:
                report += f"| {model_name} | {accuracy:.1%} | {correct} | {total} |\n"
            
            report += "\n"
    
    # Performance insights
    if len(sorted_results) >= 2:
        best_model = sorted_results[0]
        worst_model = sorted_results[-1]
        
        best_acc = best_model.get('metrics', {}).get('overall_accuracy', 0)
        worst_acc = worst_model.get('metrics', {}).get('overall_accuracy', 0)
        
        report += f"## Key Insights\n\n"
        report += f"- **Best Performing Model**: {best_model.get('model', 'Unknown')} ({best_acc:.1%})\n"
        report += f"- **Performance Gap**: {(best_acc - worst_acc):.1%} between best and worst models\n"
        
        if best_acc > 0.8:
            report += f"- **High Performance**: Best model shows strong cybersecurity knowledge\n"
        elif best_acc > 0.6:
            report += f"- **Moderate Performance**: Room for improvement in cybersecurity understanding\n"
        else:
            report += f"- **Low Performance**: Significant improvement needed in cybersecurity knowledge\n"
    
    return report

def generate_error_analysis_report(
    results: List[Dict[str, Any]],
    benchmark_name: str,
    model_name: str,
    max_examples: int = 10
) -> str:
    """
    Generate a detailed error analysis report.
    
    Args:
        results: List of individual question results
        benchmark_name: Name of the benchmark
        model_name: Name of the model
        max_examples: Maximum number of error examples to include
        
    Returns:
        Formatted error analysis report
    """
    # Filter incorrect answers
    errors = [
        r for r in results 
        if r.get('ground_truth') != r.get('parsed_response')
    ]
    
    total_questions = len(results)
    total_errors = len(errors)
    accuracy = (total_questions - total_errors) / total_questions if total_questions > 0 else 0
    
    report = f"""# {benchmark_name} Error Analysis Report

**Model**: {model_name}
**Total Questions**: {total_questions}
**Total Errors**: {total_errors}
**Accuracy**: {accuracy:.1%}
**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Error Distribution

"""
    
    # Error distribution by category
    if errors and 'category' in errors[0]:
        category_errors = {}
        category_totals = {}
        
        for result in results:
            category = result.get('category', 'Unknown')
            category_totals[category] = category_totals.get(category, 0) + 1
            
            if result.get('ground_truth') != result.get('parsed_response'):
                category_errors[category] = category_errors.get(category, 0) + 1
        
        report += "### Errors by Category\n\n"
        report += "| Category | Errors | Total | Error Rate |\n"
        report += "|----------|--------|-------|------------|\n"
        
        for category in sorted(category_totals.keys()):
            errors_count = category_errors.get(category, 0)
            total_count = category_totals[category]
            error_rate = errors_count / total_count if total_count > 0 else 0
            
            report += f"| {category} | {errors_count} | {total_count} | {error_rate:.1%} |\n"
        
        report += "\n"
    
    # Common error patterns
    if errors:
        # Analyze response patterns
        response_patterns = {}
        for error in errors:
            predicted = error.get('parsed_response', 'No Response')
            actual = error.get('ground_truth', 'Unknown')
            pattern = f"{predicted} (should be {actual})"
            response_patterns[pattern] = response_patterns.get(pattern, 0) + 1
        
        # Show most common error patterns
        common_patterns = sorted(response_patterns.items(), key=lambda x: x[1], reverse=True)[:5]
        
        if common_patterns:
            report += "### Most Common Error Patterns\n\n"
            for pattern, count in common_patterns:
                report += f"- **{pattern}**: {count} occurrences\n"
            report += "\n"
    
    # Sample incorrect answers
    if errors:
        sample_errors = errors[:max_examples]
        report += f"## Sample Incorrect Answers (showing {len(sample_errors)} of {len(errors)})\n\n"
        
        for i, error in enumerate(sample_errors, 1):
            question = error.get('question', 'N/A')
            predicted = error.get('parsed_response', 'No Response')
            actual = error.get('ground_truth', 'Unknown')
            category = error.get('category', 'Unknown')
            
            # Truncate long questions
            if len(question) > 150:
                question = question[:150] + "..."
            
            report += f"### Error {i}\n\n"
            report += f"**Category**: {category}\n\n"
            report += f"**Question**: {question}\n\n"
            
            # Show options if available
            if 'options' in error and error['options']:
                report += f"**Options**:\n"
                for j, option in enumerate(error['options']):
                    letter = chr(65 + j)  # A, B, C, D
                    report += f"- {letter}. {option}\n"
                report += "\n"
            
            report += f"**Model Answer**: {predicted}\n\n"
            report += f"**Correct Answer**: {actual}\n\n"
            report += "---\n\n"
    
    return report

def save_comprehensive_report(
    results: Dict[str, Any],
    output_dir: str,
    benchmark_name: str,
    model_name: str
) -> List[str]:
    """
    Save comprehensive evaluation reports in multiple formats.
    
    Args:
        results: Complete evaluation results
        output_dir: Directory to save reports
        benchmark_name: Name of the benchmark
        model_name: Name of the model
        
    Returns:
        List of paths to generated report files
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    report_files = []
    
    # Generate summary report
    summary_report = generate_benchmark_summary(benchmark_name, results, model_name)
    summary_path = os.path.join(output_dir, f"summary_{benchmark_name}_{model_name}_{timestamp}.md")
    
    with open(summary_path, 'w') as f:
        f.write(summary_report)
    report_files.append(summary_path)
    
    # Generate error analysis if results available
    if 'results' in results and results['results']:
        error_report = generate_error_analysis_report(
            results['results'], 
            benchmark_name, 
            model_name
        )
        error_path = os.path.join(output_dir, f"errors_{benchmark_name}_{model_name}_{timestamp}.md")
        
        with open(error_path, 'w') as f:
            f.write(error_report)
        report_files.append(error_path)
    
    # Create latest symlinks
    latest_summary = os.path.join(output_dir, "latest_summary.md")
    if os.path.islink(latest_summary):
        os.unlink(latest_summary)
    os.symlink(os.path.basename(summary_path), latest_summary)
    
    if len(report_files) > 1:
        latest_errors = os.path.join(output_dir, "latest_errors.md")
        if os.path.islink(latest_errors):
            os.unlink(latest_errors)
        os.symlink(os.path.basename(report_files[1]), latest_errors)
    
    return report_files

def load_and_compare_results(results_dir: str, benchmark_name: str) -> Optional[str]:
    """
    Load multiple result files and generate a comparison report.
    
    Args:
        results_dir: Directory containing result files
        benchmark_name: Name of the benchmark
        
    Returns:
        Comparison report string or None if insufficient data
    """
    result_files = [
        f for f in os.listdir(results_dir) 
        if f.startswith(f"{benchmark_name}_") and f.endswith('.json')
    ]
    
    if len(result_files) < 2:
        return None
    
    results_list = []
    for filename in result_files:
        filepath = os.path.join(results_dir, filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                results_list.append(data)
        except Exception as e:
            print(f"Warning: Could not load {filename}: {e}")
    
    if len(results_list) < 2:
        return None
    
    return generate_comparison_report(results_list, benchmark_name)

if __name__ == "__main__":
    print("📊 Report Generator - Benchmarking Suite")
    print("This module provides utilities for generating evaluation reports.")
    print("Import and use the functions in your evaluation scripts.")
