#!/usr/bin/env python3
"""
Improved YSO Visualizations with Statistical Rigor
Addresses: Categorical correlation metrics, significance tests, class imbalance warnings
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

import sys
sys.path.insert(0, '/Users/marcus/Desktop/YSO')
from yso_utils import (
    parse_mrt_file, compute_correlation_matrix, categorize_variability
)


def cramers_v(x, y):
    """Calculate Cramér's V statistic for categorical association strength"""
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    min_dim = min(confusion_matrix.shape) - 1
    if min_dim == 0:
        return 0
    return np.sqrt(chi2 / (n * min_dim))


def chi2_test(x, y):
    """Perform chi-squared test of independence"""
    confusion_matrix = pd.crosstab(x, y)
    chi2, p_value, dof, _ = chi2_contingency(confusion_matrix)
    return chi2, p_value, dof


def create_annotated_heatmap(matrix, title, filename, add_counts=None):
    """Create correlation heatmap with annotations"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def flag_rare_categories(df):
    """Identify and report rare categories"""
    threshold = 30
    rare_categories = {}
    
    for col in ['YSO_CLASS', 'LCType']:
        counts = df[col].value_counts()
        rare = counts[counts < threshold]
        if len(rare) > 0:
            rare_categories[col] = rare.to_dict()
    
    return rare_categories


def check_class_imbalance(df):
    """Check for severe class imbalance"""
    imbalance_report = {}
    
    for col in ['YSO_CLASS', 'LCType', 'Variability']:
        counts = df[col].value_counts()
        max_count = counts.max()
        min_count = counts.min()
        ratio = max_count / min_count
        max_pct = 100 * max_count / len(df)
        
        imbalance_report[col] = {
            'max_category': counts.idxmax(),
            'max_count': max_count,
            'max_pct': max_pct,
            'min_count': min_count,
            'ratio': ratio
        }
    
    return imbalance_report


def main():
    print("Loading YSO data...")
    paper_b_file = '/Users/marcus/Desktop/YSO/paper_data_files/apjsadc397t2_mrt.txt'
    df_b = parse_mrt_file(paper_b_file)
    
    print(f"Loaded {len(df_b)} sources\n")
    
    df_b['Variability'] = categorize_variability(df_b, 'delW2mag')
    
    # ==================== DATA QUALITY REPORT ====================
    print("="*70)
    print("DATA QUALITY & STATISTICAL VALIDATION")
    print("="*70)
    
    # Check for rare categories
    rare = flag_rare_categories(df_b)
    if rare:
        print("\n⚠️  RARE CATEGORIES DETECTED:")
        for col, cats in rare.items():
            print(f"\n  {col}:")
            for cat, count in cats.items():
                pct = 100*count/len(df_b)
                print(f"    - '{cat}': {count} samples ({pct:.2f}%)")
    
    # Check for class imbalance
    imbalance = check_class_imbalance(df_b)
    print("\n⚠️  CLASS IMBALANCE ANALYSIS:")
    for col, stats in imbalance.items():
        print(f"\n  {col}:")
        print(f"    - Largest: '{stats['max_category']}' = {stats['max_pct']:.1f}%")
        print(f"    - Imbalance ratio: {stats['ratio']:.0f}:1")
        if stats['ratio'] > 50:
            print(f"    - ⚠️  SEVERE imbalance detected (>50:1)")
    
    # ==================== CONTINGENCY TABLES ====================
    print("\n" + "="*70)
    print("CONTINGENCY TABLES")
    print("="*70)
    
    ct_yso_var = pd.crosstab(df_b['YSO_CLASS'], df_b['Variability'])
    ct_lc_var = pd.crosstab(df_b['LCType'], df_b['Variability'])
    ct_yso_lc = pd.crosstab(df_b['YSO_CLASS'], df_b['LCType'])
    
    print("\n1. YSO Class vs Variability:")
    print(ct_yso_var)
    
    print("\n2. Light Curve Type vs Variability:")
    print(ct_lc_var)
    
    print("\n3. YSO Class vs Light Curve Type:")
    print(ct_yso_lc)
    
    # ==================== STATISTICAL TESTS ====================
    print("\n" + "="*70)
    print("STATISTICAL SIGNIFICANCE TESTS (Chi-Squared)")
    print("="*70)
    
    print("\n1. YSO Class vs Variability:")
    chi2_1, p1, dof1 = chi2_test(df_b['YSO_CLASS'], df_b['Variability'])
    print(f"   χ² = {chi2_1:.2f}, p-value = {p1:.2e}, DOF = {dof1}")
    print(f"   Result: {'✅ HIGHLY SIGNIFICANT' if p1 < 0.001 else '❌ NOT SIGNIFICANT'}")
    
    print("\n2. Light Curve Type vs Variability:")
    chi2_2, p2, dof2 = chi2_test(df_b['LCType'], df_b['Variability'])
    print(f"   χ² = {chi2_2:.2f}, p-value = {p2:.2e}, DOF = {dof2}")
    print(f"   Result: {'✅ HIGHLY SIGNIFICANT' if p2 < 0.001 else '❌ NOT SIGNIFICANT'}")
    
    print("\n3. YSO Class vs Light Curve Type:")
    chi2_3, p3, dof3 = chi2_test(df_b['YSO_CLASS'], df_b['LCType'])
    print(f"   χ² = {chi2_3:.2f}, p-value = {p3:.2e}, DOF = {dof3}")
    print(f"   Result: {'✅ HIGHLY SIGNIFICANT' if p3 < 0.001 else '❌ NOT SIGNIFICANT'}")
    
    # ==================== EFFECT SIZE (CRAMÉR'S V) ====================
    print("\n" + "="*70)
    print("EFFECT SIZE - CRAMÉR'S V (0-1 scale)")
    print("="*70)
    print("Interpretation: 0-0.1=negligible, 0.1-0.3=weak, 0.3-0.5=moderate, >0.5=strong")
    
    v_yso_var = cramers_v(df_b['YSO_CLASS'], df_b['Variability'])
    v_lc_var = cramers_v(df_b['LCType'], df_b['Variability'])
    v_yso_lc = cramers_v(df_b['YSO_CLASS'], df_b['LCType'])
    
    def interpret_v(v):
        if v < 0.1:
            return "negligible"
        elif v < 0.3:
            return "weak"
        elif v < 0.5:
            return "moderate"
        else:
            return "strong"
    
    print(f"\n1. YSO Class ↔ Variability:     V = {v_yso_var:.4f} ({interpret_v(v_yso_var)})")
    print(f"2. Light Curve ↔ Variability:   V = {v_lc_var:.4f} ({interpret_v(v_lc_var)})")
    print(f"3. YSO Class ↔ Light Curve:     V = {v_yso_lc:.4f} ({interpret_v(v_yso_lc)})")
    
    # ==================== METRIC CORRELATIONS ====================
    print("\n" + "="*70)
    print("NUMERIC VARIABILITY METRICS (Pearson Correlation)")
    print("="*70)
    
    numeric_cols = ['W2magMean', 'sig_W2Flux', 'delW2mag', 'Period', 'slope', 'r_value', 'FLP_LSP_BOOT']
    corr_matrix = compute_correlation_matrix(df_b, numeric_cols, standardize=True)
    print("\nCorrelation Matrix:")
    print(corr_matrix.round(3))
    
    # Find strongest correlations
    print("\nStrongest correlations (|r| > 0.3):")
    for i in range(len(corr_matrix)):
        for j in range(i+1, len(corr_matrix)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > 0.3:
                print(f"  {corr_matrix.index[i]} ↔ {corr_matrix.columns[j]}: r = {r:.3f}")
    
    # ==================== GENERATE HEATMAPS ====================
    print("\n" + "="*70)
    print("GENERATING HEATMAPS")
    print("="*70)
    
    # Heatmap 1: Variability Metrics
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Correlation Matrix: YSO Variability Metrics\n(Standardized Data)',
                 fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('/Users/marcus/Desktop/YSO/plotting_tool_graphs/correlation_heatmap_variability_metrics.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved correlation_heatmap_variability_metrics.png")
    
    # Heatmap 2: Cramér's V for YSO vs Variability
    ct_yso_var_float = ct_yso_var.astype(float)
    cramers_matrix_yso_var = pd.DataFrame(np.zeros((len(ct_yso_var), len(ct_yso_var))),
                                          index=ct_yso_var.index, columns=ct_yso_var.index)
    for i in ct_yso_var.index:
        for j in ct_yso_var.index:
            if i == j:
                cramers_matrix_yso_var.loc[i, j] = 1.0
            else:
                # Calculate Cramér's V between pairs
                mask = (df_b['YSO_CLASS'] == i) | (df_b['YSO_CLASS'] == j)
                df_subset = df_b[mask].copy()
                if len(df_subset) > 0:
                    v = cramers_v(df_subset['YSO_CLASS'], df_subset['Variability'])
                    cramers_matrix_yso_var.loc[i, j] = v
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cramers_matrix_yso_var, annot=True, fmt='.2f', cmap='YlOrRd',
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Cramér's V: YSO Class Associations\n(Effect Size of Variability Relationship)",
                 fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('/Users/marcus/Desktop/YSO/plotting_tool_graphs/cramers_v_yso_variability.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved cramers_v_yso_variability.png")
    
    # Heatmap 3: YSO vs LC Type using Pearson on contingency
    ct_yso_lc_float = ct_yso_lc.astype(float)
    corr_yso_lc = ct_yso_lc_float.T.corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_yso_lc, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Pearson Correlation: YSO Class Light Curve Distributions\n(Shows category preference similarity)',
                 fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('/Users/marcus/Desktop/YSO/plotting_tool_graphs/correlation_heatmap_yso_vs_lc.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved correlation_heatmap_yso_vs_lc.png")
    
    # Heatmap 4: YSO vs Variability using Pearson on contingency
    ct_yso_var_float = ct_yso_var.astype(float)
    corr_yso_var = ct_yso_var_float.T.corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_yso_var, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Pearson Correlation: YSO Class Variability Distributions\n(Shows category preference similarity)',
                 fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('/Users/marcus/Desktop/YSO/plotting_tool_graphs/correlation_heatmap_yso_vs_variability.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved correlation_heatmap_yso_vs_variability.png")
    
    # Heatmap 5: LC Type vs Variability using Pearson on contingency
    ct_lc_var_float = ct_lc_var.astype(float)
    corr_lc_var = ct_lc_var_float.T.corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_lc_var, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Pearson Correlation: Light Curve Type Variability Distributions\n(Shows category preference similarity)',
                 fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('/Users/marcus/Desktop/YSO/plotting_tool_graphs/correlation_heatmap_lc_vs_variability.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved correlation_heatmap_lc_vs_variability.png")
    
    print("\n" + "="*70)
    print("GENERATION COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  1. correlation_heatmap_variability_metrics.png")
    print("  2. cramers_v_yso_variability.png (NEW: Effect size visualization)")
    print("  3. correlation_heatmap_yso_vs_lc.png")
    print("  4. correlation_heatmap_yso_vs_variability.png")
    print("  5. correlation_heatmap_lc_vs_variability.png")


if __name__ == "__main__":
    main()
