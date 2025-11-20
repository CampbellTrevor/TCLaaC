"""
Data quality validation and checking for TCLaaC pipeline.

This module provides comprehensive data quality checks to ensure
reliable analysis results and catch potential issues early.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import Counter
import re

logger = logging.getLogger(__name__)


class DataQualityChecker:
    """
    Performs comprehensive data quality checks on command-line datasets.
    """
    
    def __init__(self, min_doc_length: int = 2, max_duplicate_ratio: float = 0.8):
        """
        Initialize quality checker.
        
        Args:
            min_doc_length: Minimum token count for valid documents
            max_duplicate_ratio: Maximum allowed ratio of duplicate commands
        """
        self.min_doc_length = min_doc_length
        self.max_duplicate_ratio = max_duplicate_ratio
        self.checks = []
        self.warnings = []
        self.errors = []
    
    def check_dataframe_structure(self, df: pd.DataFrame) -> bool:
        """
        Verify DataFrame has required structure.
        
        Args:
            df: DataFrame to check
            
        Returns:
            True if structure is valid
        """
        required_columns = ['command_line']
        
        for col in required_columns:
            if col not in df.columns:
                self.errors.append(f"Missing required column: '{col}'")
                return False
        
        if len(df) == 0:
            self.errors.append("DataFrame is empty")
            return False
        
        self.checks.append("✓ DataFrame structure is valid")
        return True
    
    def check_data_types(self, df: pd.DataFrame) -> bool:
        """
        Verify data types are correct.
        
        Args:
            df: DataFrame to check
            
        Returns:
            True if data types are valid
        """
        issues = []
        
        # Check command_line is string type
        if 'command_line' in df.columns:
            non_string_count = df['command_line'].apply(lambda x: not isinstance(x, str)).sum()
            if non_string_count > 0:
                issues.append(f"{non_string_count} non-string command_line values")
        
        if issues:
            self.warnings.extend(issues)
            return False
        
        self.checks.append("✓ Data types are correct")
        return True
    
    def check_null_values(self, df: pd.DataFrame) -> bool:
        """
        Check for null/missing values.
        
        Args:
            df: DataFrame to check
            
        Returns:
            True if no critical null values found
        """
        null_counts = df.isnull().sum()
        
        if 'command_line' in null_counts and null_counts['command_line'] > 0:
            null_pct = (null_counts['command_line'] / len(df)) * 100
            self.warnings.append(
                f"{null_counts['command_line']} null command_line values ({null_pct:.1f}%)"
            )
            
            if null_pct > 10:
                self.errors.append("Excessive null values (>10%)")
                return False
        
        self.checks.append("✓ Null values within acceptable limits")
        return True
    
    def check_duplicates(self, df: pd.DataFrame) -> bool:
        """
        Check for excessive duplicate commands.
        
        Args:
            df: DataFrame to check
            
        Returns:
            True if duplicate ratio is acceptable
        """
        if 'command_line' not in df.columns:
            self.checks.append("✓ Skipped duplicate check (missing column)")
            return True
            
        duplicate_count = df['command_line'].duplicated().sum()
        duplicate_ratio = duplicate_count / len(df) if len(df) > 0 else 0
        
        if duplicate_ratio > self.max_duplicate_ratio:
            self.warnings.append(
                f"High duplicate ratio: {duplicate_ratio:.1%} "
                f"({duplicate_count}/{len(df)} commands)"
            )
            self.warnings.append(
                "Consider deduplication for better topic modeling"
            )
        
        unique_count = df['command_line'].nunique()
        self.checks.append(
            f"✓ Found {unique_count} unique commands out of {len(df)} "
            f"({duplicate_ratio:.1%} duplicates)"
        )
        return True
    
    def check_command_length(self, df: pd.DataFrame) -> bool:
        """
        Check command length distribution.
        
        Args:
            df: DataFrame to check
            
        Returns:
            True if length distribution is reasonable
        """
        if len(df) == 0 or 'command_line' not in df.columns:
            self.checks.append("✓ Skipped command length check (empty or missing column)")
            return True
            
        lengths = df['command_line'].str.len()
        
        # Check for suspiciously short commands
        very_short = (lengths < 3).sum()
        if very_short > 0:
            pct = (very_short / len(df)) * 100
            self.warnings.append(
                f"{very_short} very short commands (<3 chars, {pct:.1f}%)"
            )
        
        # Check for suspiciously long commands
        very_long = (lengths > 10000).sum()
        if very_long > 0:
            pct = (very_long / len(df)) * 100
            self.warnings.append(
                f"{very_long} very long commands (>10K chars, {pct:.1f}%)"
            )
        
        avg_length = lengths.mean()
        median_length = lengths.median()
        
        self.checks.append(
            f"✓ Command length: mean={avg_length:.0f}, median={median_length:.0f}"
        )
        return True
    
    def check_character_distribution(self, df: pd.DataFrame) -> bool:
        """
        Analyze character distribution for anomalies.
        
        Args:
            df: DataFrame to check
            
        Returns:
            True if character distribution is reasonable
        """
        if 'command_line' not in df.columns or len(df) == 0:
            self.checks.append("✓ Skipped character distribution check")
            return True
            
        # Sample commands for analysis (to avoid memory issues)
        sample_size = min(1000, len(df))
        sample = df['command_line'].sample(n=sample_size, random_state=42)
        
        # Check for binary/non-printable content
        non_printable_count = 0
        for cmd in sample:
            if any(ord(c) < 32 or ord(c) > 126 for c in cmd if c not in '\t\n\r'):
                non_printable_count += 1
        
        if non_printable_count > sample_size * 0.1:
            self.warnings.append(
                f"High non-printable character content: {non_printable_count}/{sample_size}"
            )
        
        self.checks.append("✓ Character distribution looks normal")
        return True
    
    def check_token_quality(self, df: pd.DataFrame) -> bool:
        """
        Check quality of tokenized data.
        
        Args:
            df: DataFrame with 'tokens' column
            
        Returns:
            True if token quality is acceptable
        """
        if 'tokens' not in df.columns:
            # Not yet tokenized, skip this check
            return True
        
        # Check for documents with too few tokens
        short_docs = df['tokens'].apply(len) < self.min_doc_length
        short_count = short_docs.sum()
        
        if short_count > 0:
            pct = (short_count / len(df)) * 100
            self.warnings.append(
                f"{short_count} documents with <{self.min_doc_length} tokens ({pct:.1f}%)"
            )
            self.warnings.append(
                "These will be filtered out before analysis"
            )
        
        # Check average token count
        avg_tokens = df['tokens'].apply(len).mean()
        self.checks.append(f"✓ Average tokens per document: {avg_tokens:.1f}")
        
        return True
    
    def check_vocabulary_diversity(self, df: pd.DataFrame) -> bool:
        """
        Check vocabulary diversity.
        
        Args:
            df: DataFrame with 'tokens' column
            
        Returns:
            True if vocabulary is diverse enough
        """
        if 'tokens' not in df.columns:
            return True
        
        # Collect all unique tokens
        all_tokens = set()
        for tokens in df['tokens']:
            all_tokens.update(tokens)
        
        vocab_size = len(all_tokens)
        total_tokens = sum(len(tokens) for tokens in df['tokens'])
        
        if vocab_size < 10:
            self.warnings.append(
                f"Very small vocabulary: {vocab_size} unique tokens"
            )
            self.warnings.append(
                "Dataset may be too small or homogeneous for meaningful analysis"
            )
        
        # Calculate type-token ratio (vocabulary diversity)
        ttr = vocab_size / total_tokens if total_tokens > 0 else 0
        
        self.checks.append(
            f"✓ Vocabulary: {vocab_size} unique tokens (TTR: {ttr:.3f})"
        )
        return True
    
    def run_all_checks(self, df: pd.DataFrame) -> Tuple[bool, List[str], List[str], List[str]]:
        """
        Run all quality checks on a DataFrame.
        
        Args:
            df: DataFrame to check
            
        Returns:
            Tuple of (is_valid, checks_passed, warnings, errors)
        """
        # Reset state
        self.checks = []
        self.warnings = []
        self.errors = []
        
        logger.info("Running data quality checks...")
        
        # Run checks
        checks_passed = [
            self.check_dataframe_structure(df),
            self.check_data_types(df),
            self.check_null_values(df),
            self.check_duplicates(df),
            self.check_command_length(df),
            self.check_character_distribution(df),
            self.check_token_quality(df),
            self.check_vocabulary_diversity(df)
        ]
        
        is_valid = all(checks_passed) and len(self.errors) == 0
        
        # Log results
        logger.info(f"\nData Quality Report:")
        logger.info(f"  Checks passed: {sum(checks_passed)}/{len(checks_passed)}")
        
        for check in self.checks:
            logger.info(f"  {check}")
        
        if self.warnings:
            logger.warning(f"\n⚠ Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                logger.warning(f"  {warning}")
        
        if self.errors:
            logger.error(f"\n✗ Errors ({len(self.errors)}):")
            for error in self.errors:
                logger.error(f"  {error}")
        
        return is_valid, self.checks, self.warnings, self.errors
    
    def generate_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a comprehensive quality report.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with quality metrics
        """
        report = {
            'dataset_size': len(df)
        }
        
        if 'command_line' in df.columns:
            report['unique_commands'] = df['command_line'].nunique()
            report['duplicate_ratio'] = df['command_line'].duplicated().sum() / max(len(df), 1)
            report['null_count'] = df['command_line'].isnull().sum()
            report['avg_length'] = df['command_line'].str.len().mean()
            report['median_length'] = df['command_line'].str.len().median()
        
        if 'tokens' in df.columns:
            all_tokens = [token for tokens in df['tokens'] for token in tokens]
            report['vocabulary_size'] = len(set(all_tokens))
            report['avg_tokens'] = df['tokens'].apply(len).mean()
            report['total_tokens'] = len(all_tokens)
        
        return report
