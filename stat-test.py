#!/usr/bin/env python3
"""
Statistical Analysis Script for RAG vs Baseline Comparison

This script performs paired t-tests, effect size calculations, and confidence intervals
for comparing RAG and baseline configurations across various metrics.

Usage: python stat-test.py <path_to_advanced_metrics_analysis.json>
"""

import json
import sys
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, shapiro, wilcoxon
from scipy import stats
import math


def load_data_from_json(json_file):
    """
    Extract baseline and RAG data from advanced_metrics_analysis.json
    
    Returns paired arrays for each metric (baseline, RAG)
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    baseline_data = {}
    rag_data = {}
    
    # Initialize metric arrays
    metrics = ['plan_parallelism_score', 'plan_redundancy_score', 'agent_task_fit_score']
    for metric in metrics:
        baseline_data[metric] = []
        rag_data[metric] = []
    
    # Extract data from each experiment
    for experiment in data['metrics']:
        run_id = experiment['run_id']
        metrics_data = experiment['metrics']
        
        if 'baseline' in run_id:
            for metric in metrics:
                baseline_data[metric].append(metrics_data[metric])
        elif 'rag' in run_id:
            for metric in metrics:
                rag_data[metric].append(metrics_data[metric])
    
    return baseline_data, rag_data


def paired_ttest(group1, group2):
    """Perform paired t-test"""
    statistic, p_value = ttest_rel(group1, group2)
    return statistic, p_value


def cohens_d(group1, group2):
    """Calculate Cohen's d effect size for paired samples"""
    diff = np.array(group1) - np.array(group2)
    return np.mean(diff) / np.std(diff, ddof=1)


def confidence_interval_paired(group1, group2, confidence=0.95):
    """Calculate confidence interval for mean difference in paired samples"""
    diff = np.array(group1) - np.array(group2)
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    n = len(diff)
    
    alpha = 1 - confidence
    t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
    margin_error = t_critical * (std_diff / math.sqrt(n))
    
    return mean_diff, (mean_diff - margin_error, mean_diff + margin_error)


def interpret_effect_size(d):
    """Interpret Cohen's d effect size"""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "Small"
    elif abs_d < 0.5:
        return "Small"
    elif abs_d < 0.8:
        return "Medium"
    else:
        return "Large"


def test_normality(group1, group2):
    """Test normality of differences using Shapiro-Wilk test"""
    diff = np.array(group1) - np.array(group2)
    statistic, p_value = shapiro(diff)
    return p_value > 0.05, p_value


def run_statistical_analysis(json_file):
    """Run complete statistical analysis"""
    print(f"Loading data from: {json_file}")
    baseline_data, rag_data = load_data_from_json(json_file)
    
    # Verify we have paired data
    n_baseline = len(baseline_data['plan_parallelism_score'])
    n_rag = len(rag_data['plan_parallelism_score'])
    
    print(f"Found {n_baseline} baseline experiments and {n_rag} RAG experiments")
    
    if n_baseline != n_rag:
        print("Warning: Unequal number of baseline and RAG experiments!")
        return
    
    # Prepare results
    results = {}
    metrics_info = {
        'plan_parallelism_score': 'Parallelism',
        'plan_redundancy_score': 'Redundancy', 
        'agent_task_fit_score': 'Agent-Task Fit'
    }
    
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS RESULTS")
    print("="*80)
    
    for metric_key, metric_name in metrics_info.items():
        rag_values = rag_data[metric_key]
        baseline_values = baseline_data[metric_key]
        
        # Paired t-test
        t_stat, p_val = paired_ttest(rag_values, baseline_values)
        
        # Effect size
        effect_size = cohens_d(rag_values, baseline_values)
        effect_interpretation = interpret_effect_size(effect_size)
        
        # Confidence interval
        mean_diff, ci = confidence_interval_paired(rag_values, baseline_values)
        
        # Normality test
        is_normal, normality_p = test_normality(rag_values, baseline_values)
        
        # Store results
        results[metric_key] = {
            'metric_name': metric_name,
            'mean_diff': mean_diff,
            'ci_lower': ci[0],
            'ci_upper': ci[1],
            't_stat': t_stat,
            'p_value': p_val,
            'cohens_d': effect_size,
            'effect_interpretation': effect_interpretation,
            'is_normal': is_normal,
            'normality_p': normality_p,
            'baseline_mean': np.mean(baseline_values),
            'rag_mean': np.mean(rag_values)
        }
        
        # Print detailed results
        print(f"\n{metric_name.upper()}")
        print("-" * len(metric_name))
        print(f"Baseline Mean: {np.mean(baseline_values):.3f}")
        print(f"RAG Mean: {np.mean(rag_values):.3f}")
        print(f"Mean Difference (RAG - Baseline): {mean_diff:.3f}")
        print(f"95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
        print(f"t-statistic: {t_stat:.3f}")
        print(f"p-value: {p_val:.3f}")
        print(f"Cohen's d: {effect_size:.3f} ({effect_interpretation})")
        print(f"Normality (p={normality_p:.3f}): {'Normal' if is_normal else 'Not Normal'}")
        
        # Significance interpretation
        if p_val < 0.001:
            sig_text = "***"
        elif p_val < 0.01:
            sig_text = "**"
        elif p_val < 0.05:
            sig_text = "*"
        else:
            sig_text = "ns"
        print(f"Significance: {sig_text}")
        
        # If not normal, also run Wilcoxon test
        if not is_normal:
            wilcoxon_stat, wilcoxon_p = wilcoxon(rag_values, baseline_values)
            print(f"Wilcoxon signed-rank test: W={wilcoxon_stat:.3f}, p={wilcoxon_p:.3f}")
    
    # Generate summary table
    print("\n" + "="*120)
    print("SUMMARY TABLE")
    print("="*120)
    print(f"{'Metric':<15} {'Mean Diff':<10} {'95% CI':<20} {'t-stat':<8} {'p-value':<8} {'Cohen\'s d':<10} {'Effect Size':<12}")
    print("-" * 120)
    
    for metric_key, result in results.items():
        ci_text = f"[{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]"
        p_text = f"{result['p_value']:.3f}" if result['p_value'] >= 0.001 else "<0.001"
        
        print(f"{result['metric_name']:<15} {result['mean_diff']:<10.3f} {ci_text:<20} {result['t_stat']:<8.2f} {p_text:<8} {result['cohens_d']:<10.2f} {result['effect_interpretation']:<12}")
    
    return results


def main():
    if len(sys.argv) != 2:
        print("Usage: python stat-test.py <path_to_advanced_metrics_analysis.json>")
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    try:
        results = run_statistical_analysis(json_file)
        print("\nAnalysis complete!")
        
    except FileNotFoundError:
        print(f"Error: File '{json_file}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file '{json_file}'.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()