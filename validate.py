#!/usr/bin/env python3
"""
validate_data.py

This script performs validation checks on processed data to ensure quality
before using it for model training.
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union

# Add the src directory to the path
script_dir = Path(__file__).resolve().parent
src_dir = script_dir.parent / "src" if script_dir.name == "scripts" else script_dir
sys.path.insert(0, str(src_dir))

# Import project modules
from config import Config
from utils import setup_logging

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Validate processed data for model training')
    
    parser.add_argument(
        '--input_file', 
        type=str, 
        required=True,
        help='Input CSV file with processed data'
    )
    
    parser.add_argument(
        '--output_file', 
        type=str, 
        help='Output file for validation report (default: validation_report.txt)'
    )
    
    parser.add_argument(
        '--threshold', 
        type=float, 
        default=0.01,
        help='Threshold for validation checks (default: 0.01)'
    )
    
    return parser.parse_args()

def load_data(input_file: str) -> pd.DataFrame:
    """
    Load processed data from CSV file.
    
    Args:
        input_file: Path to CSV file
        
    Returns:
        DataFrame with processed data
    """
    logger.info(f"Loading data from {input_file}")
    
    # Check if file exists
    if not Path(input_file).exists():
        logger.error(f"Input file {input_file} does not exist")
        sys.exit(1)
    
    # Load data
    try:
        df = pd.read_csv(input_file, index_col=0, parse_dates=True)
        logger.info(f"Loaded {len(df)} rows from {df.index[0]} to {df.index[-1]}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        sys.exit(1)

def check_missing_values(df: pd.DataFrame) -> Dict[str, float]:
    """
    Check for missing values in the DataFrame.
    
    Args:
        df: DataFrame to check
        
    Returns:
        Dictionary with missing value statistics
    """
    logger.info("Checking for missing values")
    
    # Get missing values count and percentage
    missing_counts = df.isna().sum()
    missing_percentages = (missing_counts / len(df)) * 100
    
    # Filter columns with missing values
    missing_cols = missing_percentages[missing_percentages > 0]
    
    results = {
        'total_missing_values': missing_counts.sum(),
        'total_missing_percentage': (missing_counts.sum() / (len(df) * len(df.columns))) * 100,
        'columns_with_missing': len(missing_cols),
        'missing_columns': {col: {'count': int(missing_counts[col]), 'percentage': float(missing_percentages[col])}
                           for col in missing_cols.index}
    }
    
    return results

def check_outliers(df: pd.DataFrame, threshold: float = 3.0) -> Dict[str, Dict]:
    """
    Check for outliers in the DataFrame using Z-score method.
    
    Args:
        df: DataFrame to check
        threshold: Z-score threshold for outliers (default: 3.0)
        
    Returns:
        Dictionary with outlier statistics
    """
    logger.info(f"Checking for outliers (Z-score > {threshold})")
    
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    results = {}
    
    for col in numeric_df.columns:
        # Skip binary columns or columns with too few unique values
        if numeric_df[col].nunique() < 5:
            continue
            
        # Calculate Z-scores
        mean = numeric_df[col].mean()
        std = numeric_df[col].std()
        
        if std == 0:  # Skip columns with zero standard deviation
            continue
            
        z_scores = (numeric_df[col] - mean) / std
        
        # Count outliers
        outliers_count = (abs(z_scores) > threshold).sum()
        outliers_percentage = (outliers_count / len(df)) * 100
        
        if outliers_percentage > 0:
            results[col] = {
                'count': int(outliers_count),
                'percentage': float(outliers_percentage),
                'min': float(numeric_df[col].min()),
                'max': float(numeric_df[col].max()),
                'mean': float(mean),
                'std': float(std)
            }
    
    return results

def check_duplicates(df: pd.DataFrame) -> Dict[str, int]:
    """
    Check for duplicate rows in the DataFrame.
    
    Args:
        df: DataFrame to check
        
    Returns:
        Dictionary with duplicate statistics
    """
    logger.info("Checking for duplicate rows")
    
    # Check for duplicate indices
    duplicate_indices = df.index.duplicated().sum()
    
    # Check for duplicate rows
    duplicate_rows = df.duplicated().sum()
    
    return {
        'duplicate_indices': int(duplicate_indices),
        'duplicate_rows': int(duplicate_rows)
    }

def check_data_continuity(df: pd.DataFrame) -> Dict[str, Union[int, float, List[str]]]:
    """
    Check for continuity in time series data.
    
    Args:
        df: DataFrame to check (with datetime index)
        
    Returns:
        Dictionary with continuity statistics
    """
    logger.info("Checking for data continuity")
    
    # Calculate time differences between consecutive rows
    time_diffs = df.index.to_series().diff().dropna()
    
    # Get unique time differences
    unique_diffs = time_diffs.unique()
    
    # Check for gaps in time series
    common_diff = time_diffs.mode()[0]
    gaps = time_diffs[time_diffs > common_diff * 1.5]
    
    # Format gaps for reporting
    formatted_gaps = []
    for idx, gap in gaps.items():
        prev_idx = df.index[df.index.get_loc(idx) - 1]
        formatted_gaps.append(f"{prev_idx} to {idx}: {gap}")
    
    return {
        'common_interval': str(common_diff),
        'unique_intervals': [str(diff) for diff in unique_diffs],
        'gaps_count': len(gaps),
        'gaps': formatted_gaps[:10]  # Limit to first 10 gaps
    }

def check_stationarity(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Check for stationarity in time series columns using Augmented Dickey-Fuller test.
    
    Args:
        df: DataFrame to check
        
    Returns:
        Dictionary with stationarity test results
    """
    logger.info("Checking for stationarity in time series")
    
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        logger.warning("statsmodels not installed. Skipping stationarity check.")
        return {}
    
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    results = {}
    
    # Select a subset of important columns to check
    important_cols = [
        'close', 'close_pct_change', 'log_return', 'volume_pct_change',
        'rsi_14', 'macd', 'atr_14'
    ]
    cols_to_check = [col for col in important_cols if col in numeric_df.columns]
    
    for col in cols_to_check:
        # Drop NaN values
        series = numeric_df[col].dropna()
        
        if len(series) < 20:  # Skip if not enough data
            continue
            
        try:
            # Perform ADF test
            adf_result = adfuller(series, autolag='AIC')
            
            results[col] = {
                'adf_statistic': float(adf_result[0]),
                'p_value': float(adf_result[1]),
                'is_stationary': adf_result[1] < 0.05,
                'critical_values': {f'critical_{key}': float(value) for key, value in adf_result[4].items()}
            }
        except Exception as e:
            logger.warning(f"Error checking stationarity for {col}: {str(e)}")
    
    return results

def check_target_balance(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Check balance of target variables.
    
    Args:
        df: DataFrame to check
        
    Returns:
        Dictionary with target balance statistics
    """
    logger.info("Checking target variable balance")
    
    results = {}
    
    # Check binary targets (direction)
    binary_targets = [col for col in df.columns if col.startswith('target_direction_')]
    
    for target in binary_targets:
        if target in df.columns:
            # Count classes
            class_counts = df[target].value_counts()
            total = class_counts.sum()
            
            # Calculate balance ratio (larger class / smaller class)
            if len(class_counts) > 1:
                ratio = float(class_counts.max() / class_counts.min())
            else:
                ratio = float('inf')
                
            results[target] = {
                'class_counts': {str(cls): int(count) for cls, count in class_counts.items()},
                'class_percentages': {str(cls): float(count/total*100) for cls, count in class_counts.items()},
                'balance_ratio': ratio,
                'is_balanced': ratio < 1.5
            }
    
    # Check multi-class targets
    multiclass_targets = [col for col in df.columns if col.startswith('target_class_')]
    
    for target in multiclass_targets:
        if target in df.columns:
            # Count classes
            class_counts = df[target].value_counts()
            total = class_counts.sum()
            
            # Calculate balance ratio (larger class / smaller class)
            if len(class_counts) > 1:
                ratio = float(class_counts.max() / class_counts.min())
            else:
                ratio = float('inf')
                
            results[target] = {
                'class_counts': {str(cls): int(count) for cls, count in class_counts.items()},
                'class_percentages': {str(cls): float(count/total*100) for cls, count in class_counts.items()},
                'balance_ratio': ratio,
                'is_balanced': ratio < 3.0  # More lenient for multi-class
            }
    
    return results

def check_feature_correlations(df: pd.DataFrame, threshold: float = 0.95) -> Dict[str, List[Tuple[str, str, float]]]:
    """
    Check for highly correlated features.
    
    Args:
        df: DataFrame to check
        threshold: Correlation threshold (default: 0.95)
        
    Returns:
        Dictionary with correlation statistics
    """
    logger.info(f"Checking for highly correlated features (threshold: {threshold})")
    
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Find highly correlated features
    high_corr_pairs = []
    
    # Get upper triangle of correlation matrix
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find pairs with correlation above threshold
    for col in upper_tri.columns:
        high_corr = upper_tri[col][abs(upper_tri[col]) > threshold]
        for idx, corr_value in high_corr.items():
            high_corr_pairs.append((col, idx, float(corr_value)))
    
    # Sort by absolute correlation (descending)
    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    return {
        'threshold': threshold,
        'high_correlation_count': len(high_corr_pairs),
        'high_correlation_pairs': high_corr_pairs
    }

def check_feature_importance(df: pd.DataFrame, target_col: str = 'target_1') -> Dict[str, List[Tuple[str, float]]]:
    """
    Estimate feature importance using a basic model.
    
    Args:
        df: DataFrame to check
        target_col: Target column for importance estimation
        
    Returns:
        Dictionary with feature importance statistics
    """
    logger.info(f"Estimating feature importance for target: {target_col}")
    
    # Check if target column exists
    if target_col not in df.columns:
        # Try to find an alternative target
        target_options = [col for col in df.columns if col.startswith('target_')]
        if target_options:
            target_col = target_options[0]
            logger.info(f"Target {target_col} not found, using {target_col} instead")
        else:
            logger.warning("No target columns found, skipping feature importance check")
            return {'error': 'No target columns found'}
    
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        logger.warning("scikit-learn not installed. Skipping feature importance check.")
        return {}
    
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Remove target columns
    feature_cols = [col for col in numeric_cols 
                 if not col.startswith('target_') 
                 and not col.startswith('future_')]
    
    # Ensure we have enough data
    if len(df) < 100 or len(feature_cols) < 5:
        logger.warning("Not enough data for feature importance check")
        return {'error': 'Not enough data'}
    
    try:
        # Prepare data
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Handle missing values
        X.fillna(X.median(), inplace=True)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train a model
        model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        model.fit(X_scaled, y)
        
        # Get feature importances
        importances = model.feature_importances_
        
        # Create importance pairs and sort
        importance_pairs = [(feature, float(importance)) 
                           for feature, importance in zip(feature_cols, importances)]
        importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'target': target_col,
            'importance_pairs': importance_pairs
        }
    except Exception as e:
        logger.warning(f"Error estimating feature importance: {str(e)}")
        return {'error': str(e)}

def generate_report(results: Dict, output_file: str) -> None:
    """
    Generate validation report.
    
    Args:
        results: Dictionary with validation results
        output_file: Path to output file
    """
    logger.info(f"Generating validation report: {output_file}")
    
    with open(output_file, 'w') as f:
        f.write("=== DATA VALIDATION REPORT ===\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Data summary
        f.write("=== DATA SUMMARY ===\n")
        f.write(f"Total rows: {results['data_summary']['rows']}\n")
        f.write(f"Total columns: {results['data_summary']['columns']}\n")
        f.write(f"Date range: {results['data_summary']['date_range']}\n")
        f.write(f"Memory usage: {results['data_summary']['memory_usage']} MB\n\n")
        
        # Missing values
        f.write("=== MISSING VALUES ===\n")
        f.write(f"Total missing values: {results['missing_values']['total_missing_values']}\n")
        f.write(f"Total missing percentage: {results['missing_values']['total_missing_percentage']:.2f}%\n")
        f.write(f"Columns with missing values: {results['missing_values']['columns_with_missing']}\n")
        
        if results['missing_values']['columns_with_missing'] > 0:
            f.write("\nColumns with highest missing percentages:\n")
            missing_cols = results['missing_values']['missing_columns']
            for col, stats in sorted(missing_cols.items(), key=lambda x: x[1]['percentage'], reverse=True)[:10]:
                f.write(f"  {col}: {stats['count']} values ({stats['percentage']:.2f}%)\n")
        f.write("\n")
        
        # Outliers
        f.write("=== OUTLIERS ===\n")
        outliers = results['outliers']
        f.write(f"Columns with outliers: {len(outliers)}\n")
        
        if outliers:
            f.write("\nColumns with highest outlier percentages:\n")
            for col, stats in sorted(outliers.items(), key=lambda x: x[1]['percentage'], reverse=True)[:10]:
                f.write(f"  {col}: {stats['count']} outliers ({stats['percentage']:.2f}%)\n")
                f.write(f"    Range: [{stats['min']:.2f}, {stats['max']:.2f}], Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}\n")
        f.write("\n")
        
        # Duplicates
        f.write("=== DUPLICATES ===\n")
        f.write(f"Duplicate indices: {results['duplicates']['duplicate_indices']}\n")
        f.write(f"Duplicate rows: {results['duplicates']['duplicate_rows']}\n\n")
        
        # Data continuity
        f.write("=== DATA CONTINUITY ===\n")
        continuity = results['continuity']
        f.write(f"Common interval: {continuity['common_interval']}\n")
        f.write(f"Unique intervals: {', '.join(continuity['unique_intervals'][:5])}\n")
        f.write(f"Gaps count: {continuity['gaps_count']}\n")
        
        if continuity['gaps_count'] > 0:
            f.write("\nExample gaps:\n")
            for gap in continuity['gaps'][:5]:
                f.write(f"  {gap}\n")
        f.write("\n")
        
        # Stationarity
        if 'stationarity' in results and results['stationarity']:
            f.write("=== STATIONARITY ===\n")
            for col, stats in results['stationarity'].items():
                f.write(f"{col}: p-value={stats['p_value']:.4f} ")
                if stats['is_stationary']:
                    f.write("(Stationary)\n")
                else:
                    f.write("(Non-stationary)\n")
            f.write("\n")
        
        # Target balance
        if 'target_balance' in results and results['target_balance']:
            f.write("=== TARGET BALANCE ===\n")
            for target, stats in results['target_balance'].items():
                f.write(f"{target}:\n")
                f.write(f"  Class distribution: ")
                for cls, perc in stats['class_percentages'].items():
                    f.write(f"{cls}: {perc:.1f}%, ")
                f.write(f"\n  Balance ratio: {stats['balance_ratio']:.2f} ")
                if stats['is_balanced']:
                    f.write("(Well balanced)\n")
                else:
                    f.write("(Imbalanced)\n")
            f.write("\n")
        
        # Feature correlations
        if 'feature_correlations' in results and 'high_correlation_pairs' in results['feature_correlations']:
            f.write("=== HIGHLY CORRELATED FEATURES ===\n")
            f.write(f"Correlation threshold: {results['feature_correlations']['threshold']}\n")
            f.write(f"Highly correlated pairs: {results['feature_correlations']['high_correlation_count']}\n")
            
            if results['feature_correlations']['high_correlation_count'] > 0:
                f.write("\nTop correlated pairs:\n")
                for feat1, feat2, corr in results['feature_correlations']['high_correlation_pairs'][:10]:
                    f.write(f"  {feat1} ~ {feat2}: {corr:.4f}\n")
            f.write("\n")
        
        # Feature importance
        if 'feature_importance' in results and 'importance_pairs' in results['feature_importance']:
            f.write("=== FEATURE IMPORTANCE ===\n")
            f.write(f"Target: {results['feature_importance']['target']}\n\n")
            f.write("Top important features:\n")
            for feature, importance in results['feature_importance']['importance_pairs'][:20]:
                f.write(f"  {feature}: {importance:.4f}\n")
            f.write("\n")
        
        # Validation summary
        f.write("=== VALIDATION SUMMARY ===\n")
        f.write(f"Overall validation status: {results['summary']['status']}\n")
        f.write(f"Warnings: {results['summary']['warnings']}\n")
        f.write(f"Critical issues: {results['summary']['critical_issues']}\n")
        
        if results['summary']['critical_issues'] > 0:
            f.write("\nCritical issues found:\n")
            for issue in results['summary']['critical_issue_list']:
                f.write(f"  {issue}\n")
        
        if results['summary']['warnings'] > 0:
            f.write("\nWarnings found:\n")
            for warning in results['summary']['warning_list']:
                f.write(f"  {warning}\n")
    
    logger.info(f"Validation report generated: {output_file}")

def main():
    """Main function to validate data."""
    args = parse_args()
    
    # Set default output file if not provided
    output_file = args.output_file or "validation_report.txt"
    
    # Load data
    df = load_data(args.input_file)
    
    # Run validation checks
    validation_results = {}
    
    # Data summary
    validation_results['data_summary'] = {
        'rows': len(df),
        'columns': len(df.columns),
        'date_range': f"{df.index[0]} to {df.index[-1]}",
        'memory_usage': df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
    }
    
    # Run checks
    validation_results['missing_values'] = check_missing_values(df)
    validation_results['outliers'] = check_outliers(df, threshold=3.0)
    validation_results['duplicates'] = check_duplicates(df)
    validation_results['continuity'] = check_data_continuity(df)
    validation_results['stationarity'] = check_stationarity(df)
    validation_results['target_balance'] = check_target_balance(df)
    validation_results['feature_correlations'] = check_feature_correlations(df, threshold=args.threshold * 100)
    validation_results['feature_importance'] = check_feature_importance(df)
    
    # Generate validation summary
    warnings = []
    critical_issues = []
    
    # Check for missing values
    if validation_results['missing_values']['total_missing_percentage'] > 1.0:
        critical_issues.append(f"High percentage of missing values: {validation_results['missing_values']['total_missing_percentage']:.2f}%")
    elif validation_results['missing_values']['total_missing_percentage'] > 0.1:
        warnings.append(f"Missing values present: {validation_results['missing_values']['total_missing_percentage']:.2f}%")
    
    # Check for outliers
    outlier_cols = len(validation_results['outliers'])
    if outlier_cols > len(df.columns) * 0.2:
        warnings.append(f"Many columns ({outlier_cols}) have outliers")
    
    # Check for duplicates
    if validation_results['duplicates']['duplicate_indices'] > 0:
        critical_issues.append(f"Duplicate indices found: {validation_results['duplicates']['duplicate_indices']}")
    if validation_results['duplicates']['duplicate_rows'] > 0:
        warnings.append(f"Duplicate rows found: {validation_results['duplicates']['duplicate_rows']}")
    
    # Check for gaps
    if validation_results['continuity']['gaps_count'] > 10:
        critical_issues.append(f"Many gaps in time series: {validation_results['continuity']['gaps_count']}")
    elif validation_results['continuity']['gaps_count'] > 0:
        warnings.append(f"Gaps in time series: {validation_results['continuity']['gaps_count']}")
    
    # Check target balance
    if 'target_balance' in validation_results:
        imbalanced_targets = [target for target, stats in validation_results['target_balance'].items()
                             if not stats['is_balanced']]
        if imbalanced_targets:
            warnings.append(f"Imbalanced target variables: {', '.join(imbalanced_targets[:3])}")
    
    # Check feature correlations
    if 'feature_correlations' in validation_results:
        high_corr_count = validation_results['feature_correlations']['high_correlation_count']
        if high_corr_count > 20:
            warnings.append(f"Many highly correlated feature pairs: {high_corr_count}")
    
    validation_results['summary'] = {
        'status': 'FAILED' if critical_issues else 'PASSED with warnings' if warnings else 'PASSED',
        'warnings': len(warnings),
        'critical_issues': len(critical_issues),
        'warning_list': warnings,
        'critical_issue_list': critical_issues
    }
    
    # Generate report
    generate_report(validation_results, output_file)
    
    # Exit with appropriate code
    if critical_issues:
        logger.error(f"Validation FAILED: {len(critical_issues)} critical issues found")
        sys.exit(1)
    elif warnings:
        logger.warning(f"Validation PASSED with {len(warnings)} warnings")
    else:
        logger.info("Validation PASSED with no issues")

if __name__ == "__main__":
    # Setup logging
    setup_logging()
    logger = logging.getLogger("validate_data")
    
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        sys.exit(1)