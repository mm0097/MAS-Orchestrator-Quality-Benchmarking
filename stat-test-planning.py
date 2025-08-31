#!/usr/bin/env python3
"""
Statistical Analysis Script for Planning Metrics (Latency & Tokens)

This script performs paired t-tests, effect size calculations, and confidence intervals
for comparing RAG and baseline configurations on planning latency and token usage.

Usage: python stat-test-planning.py <path_to_realm_bench_detailed_results.json>
"""

import json
import sys
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, shapiro, wilcoxon
from scipy import stats
import math


def load_planning_data_from_json(json_file):
    """
    Extract baseline and RAG planning data from realm_bench_detailed_results.json
    
    Returns paired arrays for planning latency and tokens (baseline, RAG)
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    baseline_data = {
        'planning_latency_seconds': [],
        'planning_tokens_total': []
    }
    
    rag_data = {
        'planning_latency_seconds': [],
        'planning_tokens_total': []
    }
    
    # Extract data from each experiment
    for experiment in data:
        run_id = experiment['run_id']
        experiment_type = experiment.get('experiment_type', 'unknown')
        
        # Extract planning metrics
        planning_latency = experiment.get('planning_latency_seconds', 0)
        planning_tokens = experiment.get('planning_tokens', {})
        planning_tokens_total = planning_tokens.get('total', 0)
        
        if 'baseline' in run_id or experiment_type == 'baseline':
            baseline_data['planning_latency_seconds'].append(planning_latency)
            baseline_data['planning_tokens_total'].append(planning_tokens_total)
        elif 'rag' in run_id or experiment_type == 'rag':
            rag_data['planning_latency_seconds'].append(planning_latency)
            rag_data['planning_tokens_total'].append(planning_tokens_total)
    
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


def run_planning_statistical_analysis(json_file):
    """Run complete statistical analysis for planning metrics"""
    print(f"Loading planning data from: {json_file}")
    baseline_data, rag_data = load_planning_data_from_json(json_file)
    
    # Verify we have paired data
    n_baseline = len(baseline_data['planning_latency_seconds'])
    n_rag = len(rag_data['planning_latency_seconds'])
    
    print(f"Found {n_baseline} baseline experiments and {n_rag} RAG experiments")
    
    if n_baseline != n_rag:
        print("Warning: Unequal number of baseline and RAG experiments!")
        return
    
    if n_baseline == 0 or n_rag == 0:
        print("Error: No data found!")
        return
    
    # Prepare results
    results = {}
    metrics_info = {
        'planning_latency_seconds': 'Planning Latency (s)',
        'planning_tokens_total': 'Planning Tokens'
    }
    
    print("\n" + "="*80)
    print("PLANNING METRICS STATISTICAL ANALYSIS")
    print("="*80)
    
    for metric_key, metric_name in metrics_info.items():
        rag_values = rag_data[metric_key]
        baseline_values = baseline_data[metric_key]
        
        # Check if we have valid data
        if len(rag_values) == 0 or len(baseline_values) == 0:
            print(f"Skipping {metric_name}: No data available")
            continue
            
        print(f"\nData for {metric_name}:")
        print(f"Baseline values: {baseline_values}")
        print(f"RAG values: {rag_values}")
        
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
            try:
                wilcoxon_stat, wilcoxon_p = wilcoxon(rag_values, baseline_values)
                print(f"Wilcoxon signed-rank test: W={wilcoxon_stat:.3f}, p={wilcoxon_p:.3f}")
            except ValueError as e:
                print(f"Wilcoxon test failed: {e}")
    
    # Generate summary table
    print("\n" + "="*120)
    print("PLANNING METRICS SUMMARY TABLE")
    print("="*120)
    print(f"{'Metric':<20} {'Mean Diff':<12} {'95% CI':<20} {'t-stat':<8} {'p-value':<8} {'Cohen\'s d':<10} {'Effect Size':<12}")
    print("-" * 120)
    
    for metric_key, result in results.items():
        ci_text = f"[{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]"
        p_text = f"{result['p_value']:.3f}" if result['p_value'] >= 0.001 else "<0.001"
        
        print(f"{result['metric_name']:<20} {result['mean_diff']:<12.3f} {ci_text:<20} {result['t_stat']:<8.2f} {p_text:<8} {result['cohens_d']:<10.2f} {result['effect_interpretation']:<12}")
    
    return results


def main():
    if len(sys.argv) != 2:
        print("Usage: python stat-test-planning.py <path_to_realm_bench_detailed_results.json>")
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    try:
        results = run_planning_statistical_analysis(json_file)
        print("\nPlanning metrics analysis complete!")
        
        # Print text for paper
        if results:
            print("\n" + "="*80)
            print("TEXT FOR PAPER - PLANNING OVERHEAD ANALYSIS")
            print("="*80)
            
            latency_result = results.get('planning_latency_seconds')
            tokens_result = results.get('planning_tokens_total')
            
            if latency_result and tokens_result:
                # Get baseline data for degrees of freedom
                baseline_data, _ = load_planning_data_from_json(json_file)
                n_baseline = len(baseline_data['planning_latency_seconds'])
                
                print(f"RAG significantly increased planning latency (M_diff = {latency_result['mean_diff']:.2f}s, " +
                      f"95% CI [{latency_result['ci_lower']:.2f}, {latency_result['ci_upper']:.2f}], " +
                      f"t({n_baseline-1}) = {latency_result['t_stat']:.2f}, " +
                      f"p = {latency_result['p_value']:.3f}, d = {latency_result['cohens_d']:.2f}) " +
                      f"and token usage (M_diff = {tokens_result['mean_diff']:.0f} tokens, " +
                      f"95% CI [{tokens_result['ci_lower']:.0f}, {tokens_result['ci_upper']:.0f}], " +
                      f"t({n_baseline-1}) = {tokens_result['t_stat']:.2f}, " +
                      f"p = {tokens_result['p_value']:.3f}, d = {tokens_result['cohens_d']:.2f}), " +
                      f"both representing {latency_result['effect_interpretation'].lower()} effect sizes.")
        
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