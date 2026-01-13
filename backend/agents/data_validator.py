"""
Data Validation Agent
=====================

AI agent for validating training data quality and consistency.
"""

from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import numpy as np
from .agent_config import AgentConfig


class DataValidationAgent:
    """Agent for validating and analyzing training data"""
    
    def __init__(self, config: AgentConfig):
        """
        Initialize data validation agent
        
        Args:
            config: Agent configuration
        """
        self.config = config
        self.validation_results = []
        
    def validate_dataframe(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate a training dataframe
        
        Args:
            df: DataFrame with columns ['text_a', 'text_b', 'label']
            
        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        
        # Check required columns
        required_cols = ['text_a', 'text_b', 'label']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
            return False, issues
        
        # Check minimum samples
        if len(df) < self.config.min_training_samples:
            issues.append(
                f"Insufficient samples: {len(df)} < {self.config.min_training_samples}"
            )
        
        # Check for nulls
        null_counts = df[required_cols].isnull().sum()
        if null_counts.any():
            issues.append(f"Null values found: {null_counts.to_dict()}")
        
        # Check text lengths
        for col in ['text_a', 'text_b']:
            short_texts = df[col].str.len() < self.config.min_text_length
            if short_texts.any():
                issues.append(
                    f"{col}: {short_texts.sum()} texts shorter than "
                    f"{self.config.min_text_length} characters"
                )
            
            long_texts = df[col].str.len() > self.config.max_text_length
            if long_texts.any():
                issues.append(
                    f"{col}: {long_texts.sum()} texts longer than "
                    f"{self.config.max_text_length} characters"
                )
        
        # Check label distribution
        label_dist = df['label'].value_counts()
        if len(label_dist) < 2:
            issues.append("Only one class present in labels")
        else:
            balance = label_dist.min() / label_dist.max()
            if balance < 0.2:
                issues.append(
                    f"Severe class imbalance: {balance:.2%} "
                    f"(distribution: {label_dist.to_dict()})"
                )
        
        # Check for duplicates
        duplicates = df.duplicated(subset=['text_a', 'text_b']).sum()
        if duplicates > 0:
            issues.append(f"{duplicates} duplicate text pairs found")
        
        # Check for contradictions (same texts, different labels)
        df_sorted = df.sort_values(['text_a', 'text_b'])
        contradictions = 0
        for i in range(len(df_sorted) - 1):
            if (df_sorted.iloc[i]['text_a'] == df_sorted.iloc[i+1]['text_a'] and
                df_sorted.iloc[i]['text_b'] == df_sorted.iloc[i+1]['text_b'] and
                df_sorted.iloc[i]['label'] != df_sorted.iloc[i+1]['label']):
                contradictions += 1
        
        if contradictions > 0:
            issues.append(f"{contradictions} contradictory labels found")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def analyze_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze overall data quality
        
        Args:
            df: Training dataframe
            
        Returns:
            Dictionary with quality metrics
        """
        analysis = {
            'total_samples': len(df),
            'null_percentage': df.isnull().sum().sum() / (len(df) * len(df.columns)),
            'duplicate_percentage': df.duplicated().sum() / len(df),
            'label_distribution': df['label'].value_counts().to_dict(),
            'avg_text_length_a': df['text_a'].str.len().mean(),
            'avg_text_length_b': df['text_b'].str.len().mean(),
            'min_text_length': min(df['text_a'].str.len().min(), df['text_b'].str.len().min()),
            'max_text_length': max(df['text_a'].str.len().max(), df['text_b'].str.len().max()),
        }
        
        return analysis
    
    def suggest_improvements(self, df: pd.DataFrame) -> List[str]:
        """
        Suggest data improvements
        
        Args:
            df: Training dataframe
            
        Returns:
            List of suggestions
        """
        suggestions = []
        
        is_valid, issues = self.validate_dataframe(df)
        
        if not is_valid:
            suggestions.append("Fix validation issues first:")
            suggestions.extend([f"  - {issue}" for issue in issues])
        
        # Check balance
        label_dist = df['label'].value_counts()
        if len(label_dist) >= 2:
            balance = label_dist.min() / label_dist.max()
            if balance < 0.5:
                suggestions.append(
                    f"Consider balancing classes (current ratio: {balance:.2%})"
                )
        
        # Check diversity
        unique_a = df['text_a'].nunique()
        unique_b = df['text_b'].nunique()
        if unique_a < len(df) * 0.5 or unique_b < len(df) * 0.5:
            suggestions.append("Low text diversity - consider adding more varied examples")
        
        # Check text lengths
        analysis = self.analyze_data_quality(df)
        if analysis['avg_text_length_a'] < 50 or analysis['avg_text_length_b'] < 50:
            suggestions.append("Average text length is quite short - ensure sufficient context")
        
        return suggestions
    
    def generate_report(self, df: pd.DataFrame) -> str:
        """
        Generate comprehensive validation report
        
        Args:
            df: Training dataframe
            
        Returns:
            Formatted report string
        """
        is_valid, issues = self.validate_dataframe(df)
        analysis = self.analyze_data_quality(df)
        suggestions = self.suggest_improvements(df)
        
        report = []
        report.append("=" * 80)
        report.append("DATA VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"\nValidation Status: {'✓ PASSED' if is_valid else '✗ FAILED'}")
        
        if issues:
            report.append("\nIssues Found:")
            for issue in issues:
                report.append(f"  ✗ {issue}")
        
        report.append("\nData Quality Metrics:")
        report.append(f"  Total Samples: {analysis['total_samples']:,}")
        report.append(f"  Null Percentage: {analysis['null_percentage']:.2%}")
        report.append(f"  Duplicate Percentage: {analysis['duplicate_percentage']:.2%}")
        report.append(f"  Label Distribution: {analysis['label_distribution']}")
        report.append(f"  Avg Length (Text A): {analysis['avg_text_length_a']:.1f} chars")
        report.append(f"  Avg Length (Text B): {analysis['avg_text_length_b']:.1f} chars")
        report.append(f"  Length Range: {analysis['min_text_length']}-{analysis['max_text_length']} chars")
        
        if suggestions:
            report.append("\nSuggestions:")
            for suggestion in suggestions:
                report.append(f"  → {suggestion}")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
