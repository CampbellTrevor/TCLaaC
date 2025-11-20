"""
Advanced anomaly detection for command-line analysis.

This module provides multiple anomaly detection techniques:
- Statistical outliers based on topic distributions
- Command sequence pattern analysis
- Behavioral baseline comparisons
- Ensemble anomaly scoring
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy import stats
from collections import Counter, defaultdict
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Multi-method anomaly detection for command-line data.
    """
    
    def __init__(self, contamination: float = 0.1):
        """
        Initialize anomaly detector.
        
        Args:
            contamination: Expected proportion of anomalies (0.0-0.5)
        """
        self.contamination = contamination
        self.baseline_stats = None
    
    def detect_statistical_outliers(
        self, 
        df_with_topics: pd.DataFrame,
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Detect statistical outliers based on topic distribution probabilities.
        
        Uses Z-score method to identify commands with unusual topic assignments.
        
        Args:
            df_with_topics: DataFrame with topic assignments and probabilities
            threshold: Z-score threshold for outliers (default 3.0)
            
        Returns:
            DataFrame with anomaly scores and labels
        """
        logger.info("Detecting statistical outliers...")
        
        df = df_with_topics.copy()
        
        # Calculate anomaly score based on topic probability
        if 'topic_prob' in df.columns:
            # Lower probability = more anomalous
            df['stat_anomaly_score'] = 1 - df['topic_prob']
        else:
            # Estimate probability from topic distribution
            topic_sizes = df['topic'].value_counts()
            topic_probs = topic_sizes / len(df)
            df['stat_anomaly_score'] = df['topic'].map(lambda t: 1 - topic_probs.get(t, 0.5))
        
        # Calculate Z-scores
        mean_score = df['stat_anomaly_score'].mean()
        std_score = df['stat_anomaly_score'].std()
        df['stat_zscore'] = (df['stat_anomaly_score'] - mean_score) / std_score
        
        # Label outliers
        df['is_stat_outlier'] = df['stat_zscore'].abs() > threshold
        
        outlier_count = df['is_stat_outlier'].sum()
        logger.info(f"  Found {outlier_count} statistical outliers ({outlier_count/len(df)*100:.1f}%)")
        
        return df
    
    def detect_complexity_outliers(
        self,
        df_with_topics: pd.DataFrame,
        threshold: float = 2.5
    ) -> pd.DataFrame:
        """
        Detect commands with unusual complexity scores.
        
        Args:
            df_with_topics: DataFrame with complexity scores
            threshold: Z-score threshold
            
        Returns:
            DataFrame with complexity outlier labels
        """
        logger.info("Detecting complexity outliers...")
        
        df = df_with_topics.copy()
        
        if 'complexity_score' not in df.columns:
            logger.warning("  No complexity scores found, skipping")
            df['is_complexity_outlier'] = False
            return df
        
        # Calculate Z-scores for complexity
        mean_complexity = df['complexity_score'].mean()
        std_complexity = df['complexity_score'].std()
        
        if std_complexity > 0:
            df['complexity_zscore'] = (df['complexity_score'] - mean_complexity) / std_complexity
            df['is_complexity_outlier'] = df['complexity_zscore'] > threshold
        else:
            df['complexity_zscore'] = 0
            df['is_complexity_outlier'] = False
        
        outlier_count = df['is_complexity_outlier'].sum()
        logger.info(f"  Found {outlier_count} complexity outliers ({outlier_count/len(df)*100:.1f}%)")
        
        return df
    
    def detect_isolation_forest_anomalies(
        self,
        df_with_topics: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Use Isolation Forest algorithm to detect anomalies.
        
        Args:
            df_with_topics: DataFrame with features
            
        Returns:
            DataFrame with Isolation Forest anomaly scores
        """
        logger.info("Running Isolation Forest anomaly detection...")
        
        df = df_with_topics.copy()
        
        # Select features for Isolation Forest
        feature_cols = []
        if 'complexity_score' in df.columns:
            feature_cols.append('complexity_score')
        if 'topic_prob' in df.columns:
            feature_cols.append('topic_prob')
        if 'mitre_count' in df.columns:
            feature_cols.append('mitre_count')
        
        # Add topic distribution features
        if 'topic' in df.columns:
            # One-hot encode topics
            topic_dummies = pd.get_dummies(df['topic'], prefix='topic')
            df = pd.concat([df, topic_dummies], axis=1)
            feature_cols.extend(topic_dummies.columns.tolist())
        
        if len(feature_cols) < 2:
            logger.warning("  Insufficient features for Isolation Forest, skipping")
            df['if_anomaly_score'] = 0
            df['is_if_anomaly'] = False
            return df
        
        # Prepare feature matrix
        X = df[feature_cols].fillna(0).values
        
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train Isolation Forest
        iso_forest = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100
        )
        
        # Predict anomalies (-1 = anomaly, 1 = normal)
        predictions = iso_forest.fit_predict(X_scaled)
        scores = iso_forest.score_samples(X_scaled)
        
        # Convert to anomaly scores (higher = more anomalous)
        df['if_anomaly_score'] = -scores  # Negate so higher is more anomalous
        df['is_if_anomaly'] = predictions == -1
        
        anomaly_count = df['is_if_anomaly'].sum()
        logger.info(f"  Found {anomaly_count} Isolation Forest anomalies ({anomaly_count/len(df)*100:.1f}%)")
        
        return df
    
    def detect_sequence_anomalies(
        self,
        df_with_topics: pd.DataFrame,
        window_size: int = 3
    ) -> pd.DataFrame:
        """
        Detect anomalous command sequences.
        
        Identifies unusual patterns in command execution order.
        
        Args:
            df_with_topics: DataFrame with command sequences
            window_size: Size of n-gram window
            
        Returns:
            DataFrame with sequence anomaly labels
        """
        logger.info(f"Detecting sequence anomalies (window={window_size})...")
        
        df = df_with_topics.copy()
        
        # Build n-gram frequency model
        if 'root_command' not in df.columns:
            logger.warning("  No root_command column, skipping sequence analysis")
            df['is_sequence_anomaly'] = False
            df['sequence_anomaly_score'] = 0
            return df
        
        # Extract command sequences
        commands = df['root_command'].fillna('unknown').tolist()
        
        # Build n-gram model
        ngram_counts = Counter()
        for i in range(len(commands) - window_size + 1):
            ngram = tuple(commands[i:i+window_size])
            ngram_counts[ngram] += 1
        
        # Calculate frequency for each position
        sequence_scores = []
        for i in range(len(commands)):
            if i < window_size - 1:
                # Not enough context at beginning
                sequence_scores.append(0.5)
            else:
                ngram = tuple(commands[i-window_size+1:i+1])
                freq = ngram_counts.get(ngram, 0)
                total = sum(1 for ng in ngram_counts if ng[:-1] == ngram[:-1])
                score = 1 - (freq / max(total, 1))  # Rare sequences get high scores
                sequence_scores.append(score)
        
        df['sequence_anomaly_score'] = sequence_scores
        
        # Label anomalies (top 10% as anomalous)
        threshold = np.percentile(sequence_scores, 90)
        df['is_sequence_anomaly'] = df['sequence_anomaly_score'] > threshold
        
        anomaly_count = df['is_sequence_anomaly'].sum()
        logger.info(f"  Found {anomaly_count} sequence anomalies ({anomaly_count/len(df)*100:.1f}%)")
        
        return df
    
    def build_baseline(self, df_with_topics: pd.DataFrame) -> None:
        """
        Build a baseline profile from normal data.
        
        Args:
            df_with_topics: DataFrame with normal/baseline data
        """
        logger.info("Building baseline profile...")
        
        self.baseline_stats = {
            'topic_distribution': df_with_topics['topic'].value_counts(normalize=True).to_dict(),
            'avg_complexity': df_with_topics['complexity_score'].mean() if 'complexity_score' in df_with_topics.columns else 0,
            'std_complexity': df_with_topics['complexity_score'].std() if 'complexity_score' in df_with_topics.columns else 0,
            'common_roots': Counter(df_with_topics['root_command'].fillna('unknown')).most_common(20),
            'avg_tokens': df_with_topics['tokens'].apply(len).mean() if 'tokens' in df_with_topics.columns else 0,
        }
        
        logger.info("  Baseline profile built successfully")
    
    def detect_baseline_deviations(
        self,
        df_with_topics: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Detect deviations from established baseline.
        
        Args:
            df_with_topics: DataFrame to compare against baseline
            
        Returns:
            DataFrame with baseline deviation scores
        """
        logger.info("Detecting baseline deviations...")
        
        df = df_with_topics.copy()
        
        if self.baseline_stats is None:
            logger.warning("  No baseline established, building from current data")
            self.build_baseline(df)
        
        # Calculate deviation scores
        deviation_scores = []
        
        for idx, row in df.iterrows():
            score = 0
            
            # Topic distribution deviation
            topic = row['topic']
            expected_freq = self.baseline_stats['topic_distribution'].get(topic, 0)
            if expected_freq < 0.01:  # Rare topic
                score += 0.3
            
            # Complexity deviation
            if 'complexity_score' in row:
                complexity_z = abs(row['complexity_score'] - self.baseline_stats['avg_complexity'])
                if self.baseline_stats['std_complexity'] > 0:
                    complexity_z /= self.baseline_stats['std_complexity']
                score += min(complexity_z / 5, 0.4)  # Cap at 0.4
            
            # Root command rarity
            if 'root_command' in row:
                common_roots = [root for root, _ in self.baseline_stats['common_roots']]
                if row['root_command'] not in common_roots:
                    score += 0.3
            
            deviation_scores.append(min(score, 1.0))  # Cap at 1.0
        
        df['baseline_deviation_score'] = deviation_scores
        df['is_baseline_anomaly'] = df['baseline_deviation_score'] > 0.7
        
        anomaly_count = df['is_baseline_anomaly'].sum()
        logger.info(f"  Found {anomaly_count} baseline deviations ({anomaly_count/len(df)*100:.1f}%)")
        
        return df
    
    def compute_ensemble_score(
        self,
        df_with_topics: pd.DataFrame,
        weights: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Compute ensemble anomaly score from multiple methods.
        
        Args:
            df_with_topics: DataFrame with individual anomaly scores
            weights: Optional custom weights for each method
            
        Returns:
            DataFrame with ensemble anomaly scores
        """
        logger.info("Computing ensemble anomaly scores...")
        
        df = df_with_topics.copy()
        
        # Default weights
        if weights is None:
            weights = {
                'stat_anomaly_score': 0.25,
                'complexity_zscore': 0.20,
                'if_anomaly_score': 0.30,
                'sequence_anomaly_score': 0.15,
                'baseline_deviation_score': 0.10
            }
        
        # Normalize individual scores to [0, 1]
        score_cols = []
        for col, weight in weights.items():
            if col in df.columns:
                # Normalize to [0, 1] range
                normalized_col = f"{col}_norm"
                min_val = df[col].min()
                max_val = df[col].max()
                
                if max_val > min_val:
                    df[normalized_col] = (df[col] - min_val) / (max_val - min_val)
                else:
                    df[normalized_col] = 0.5
                
                score_cols.append((normalized_col, weight))
        
        # Compute weighted ensemble score
        df['ensemble_anomaly_score'] = 0
        total_weight = 0
        
        for col, weight in score_cols:
            df['ensemble_anomaly_score'] += df[col] * weight
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            df['ensemble_anomaly_score'] /= total_weight
        
        # Label top anomalies
        threshold = np.percentile(df['ensemble_anomaly_score'], 90)
        df['is_ensemble_anomaly'] = df['ensemble_anomaly_score'] > threshold
        
        anomaly_count = df['is_ensemble_anomaly'].sum()
        logger.info(f"  Identified {anomaly_count} ensemble anomalies ({anomaly_count/len(df)*100:.1f}%)")
        
        return df
    
    def run_full_detection(
        self,
        df_with_topics: pd.DataFrame,
        methods: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Run complete anomaly detection pipeline.
        
        Args:
            df_with_topics: DataFrame with topic assignments
            methods: List of methods to run (None = all methods)
            
        Returns:
            DataFrame with all anomaly scores and labels
        """
        logger.info("=" * 70)
        logger.info("ANOMALY DETECTION")
        logger.info("=" * 70)
        
        if methods is None:
            methods = ['statistical', 'complexity', 'isolation_forest', 'sequence', 'ensemble']
        
        df = df_with_topics.copy()
        
        # Run each detection method
        if 'statistical' in methods:
            df = self.detect_statistical_outliers(df)
        
        if 'complexity' in methods:
            df = self.detect_complexity_outliers(df)
        
        if 'isolation_forest' in methods:
            df = self.detect_isolation_forest_anomalies(df)
        
        if 'sequence' in methods:
            df = self.detect_sequence_anomalies(df)
        
        if 'baseline' in methods and self.baseline_stats:
            df = self.detect_baseline_deviations(df)
        
        if 'ensemble' in methods:
            df = self.compute_ensemble_score(df)
        
        # Summary
        logger.info("\nAnomaly Detection Summary:")
        anomaly_cols = [col for col in df.columns if col.startswith('is_') and 'anomaly' in col]
        for col in anomaly_cols:
            count = df[col].sum()
            pct = count / len(df) * 100
            logger.info(f"  {col}: {count} ({pct:.1f}%)")
        
        logger.info("")
        
        return df
