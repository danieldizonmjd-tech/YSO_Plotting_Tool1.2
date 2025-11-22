#!/usr/bin/env python3
"""
Generate Chord Diagram: Correlation Matrix of Variability Metrics
This script recreates the chord diagram showing standardized correlations
between YSO variability metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cachai.chplot as chp
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, '/Users/marcus/Desktop/YSO')
from yso_utils import parse_mrt_file, categorize_variability, compute_correlation_matrix


def main():
    print("Loading YSO data...")
    paper_b_file = '/Users/marcus/Desktop/YSO/paper_data_files/apjsadc397t2_mrt.txt'
    df_b = parse_mrt_file(paper_b_file)
    
    print(f"Loaded {len(df_b)} sources\n")
    df_b['Variability'] = categorize_variability(df_b, 'delW2mag')
    
    print("Computing standardized correlation matrix...")
    numeric_cols = ['W2magMean', 'sig_W2Flux', 'delW2mag', 'Period', 'slope', 'r_value', 'FLP_LSP_BOOT']
    corr_matrix = compute_correlation_matrix(df_b, numeric_cols, standardize=True)
    
    print("\nCorrelation Matrix (Standardized):")
    print(corr_matrix.round(3))
    
    print("\n" + "="*70)
    print("GENERATING CHORD DIAGRAM")
    print("="*70)
    
    corr_abs = corr_matrix.abs()
    
    fig = plt.figure(figsize=(14, 12))
    ax = plt.gca()
    
    chord_plot = chp.chord(
        corr_abs,
        ax=ax,
        threshold=0.15
    )
    
    ax.set_title('Correlation Matrix: Variability Metrics\n(Standardized | Width = |r| | Color = Source Variable)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    output_file = '/Users/marcus/Desktop/YSO/plotting_tool_graphs/chord_correlation_metrics.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n✓ Chord diagram saved: {output_file}")
    print("\nFigure Details:")
    print("  - Data: Standardized correlation matrix of variability metrics")
    print("  - Variables:", ', '.join(corr_matrix.columns))
    print("  - Threshold: |r| ≥ 0.15")
    print("  - Resolution: 300 DPI")


if __name__ == "__main__":
    main()
