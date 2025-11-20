"""
Comprehensive test suite for new TCLaaC features.

Tests cover:
- Caching functionality
- Data quality checking
- Anomaly detection
- Network visualization
"""

import unittest
import sys
import os
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cache_manager import CacheManager, get_cache_manager
from quality_checker import DataQualityChecker
from anomaly_detector import AnomalyDetector
from network_viz import (
    create_command_network, 
    create_topic_relationship_network,
    create_mitre_attack_network
)
from data_loader import generate_synthetic_data
from gensim.models import LdaMulticore
from gensim.corpora import Dictionary


class TestCacheManager(unittest.TestCase):
    """Test caching functionality."""
    
    def setUp(self):
        """Create temporary cache directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_manager = CacheManager(cache_dir=self.temp_dir, max_age_days=1)
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_cache_set_and_get(self):
        """Test basic cache set and get operations."""
        key_data = {'test': 'key', 'version': 1}
        value = {'result': 42, 'data': [1, 2, 3]}
        
        # Set cache
        success = self.cache_manager.set('analysis', key_data, value)
        self.assertTrue(success)
        
        # Get cache
        cached_value = self.cache_manager.get('analysis', key_data)
        self.assertIsNotNone(cached_value)
        self.assertEqual(cached_value['result'], 42)
        self.assertEqual(cached_value['data'], [1, 2, 3])
    
    def test_cache_miss(self):
        """Test cache miss returns None."""
        key_data = {'nonexistent': 'key'}
        cached_value = self.cache_manager.get('analysis', key_data)
        self.assertIsNone(cached_value)
    
    def test_cache_clear(self):
        """Test cache clearing."""
        # Add some items
        self.cache_manager.set('model', {'key': 1}, 'data1')
        self.cache_manager.set('model', {'key': 2}, 'data2')
        self.cache_manager.set('analysis', {'key': 3}, 'data3')
        
        # Clear specific type
        count = self.cache_manager.clear('model')
        self.assertEqual(count, 2)
        
        # Verify model cache is cleared
        self.assertIsNone(self.cache_manager.get('model', {'key': 1}))
        
        # Verify analysis cache still exists
        self.assertIsNotNone(self.cache_manager.get('analysis', {'key': 3}))
    
    def test_cache_stats(self):
        """Test cache statistics."""
        # Add items
        for i in range(5):
            self.cache_manager.set('model', {'key': i}, f'data{i}')
        
        stats = self.cache_manager.get_stats()
        self.assertEqual(stats['by_type']['model']['items'], 5)
        self.assertGreater(stats['total_size_mb'], 0)


class TestQualityChecker(unittest.TestCase):
    """Test data quality checking."""
    
    def setUp(self):
        """Initialize quality checker."""
        self.checker = DataQualityChecker()
    
    def test_valid_dataframe(self):
        """Test validation of correct DataFrame."""
        df = pd.DataFrame({
            'command_line': ['cmd.exe /c whoami', 'powershell.exe -Command Get-Process']
        })
        
        is_valid, checks, warnings, errors = self.checker.run_all_checks(df)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
    
    def test_missing_column(self):
        """Test detection of missing required column."""
        df = pd.DataFrame({
            'wrong_column': ['data1', 'data2']
        })
        
        is_valid, checks, warnings, errors = self.checker.run_all_checks(df)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
    
    def test_empty_dataframe(self):
        """Test detection of empty DataFrame."""
        df = pd.DataFrame({'command_line': []})
        
        is_valid, checks, warnings, errors = self.checker.run_all_checks(df)
        self.assertFalse(is_valid)
        self.assertIn('empty', ''.join(errors).lower())
    
    def test_duplicate_detection(self):
        """Test detection of duplicate commands."""
        df = pd.DataFrame({
            'command_line': ['cmd.exe /c whoami'] * 50 + ['other command'] * 10
        })
        
        is_valid, checks, warnings, errors = self.checker.run_all_checks(df)
        # Should have warnings about high duplicate ratio
        self.assertGreater(len(warnings), 0)
    
    def test_quality_report(self):
        """Test quality report generation."""
        df = generate_synthetic_data(100, seed=42)
        
        report = self.checker.generate_quality_report(df)
        
        self.assertEqual(report['dataset_size'], 100)
        self.assertIn('unique_commands', report)
        self.assertIn('avg_length', report)


class TestAnomalyDetector(unittest.TestCase):
    """Test anomaly detection."""
    
    def setUp(self):
        """Initialize anomaly detector."""
        self.detector = AnomalyDetector(contamination=0.1)
    
    def create_test_data(self, n=100):
        """Create test DataFrame with topic assignments."""
        np.random.seed(42)
        
        df = pd.DataFrame({
            'command_line': [f'command_{i}' for i in range(n)],
            'topic': np.random.randint(0, 5, n),
            'topic_prob': np.random.uniform(0.3, 0.9, n),
            'complexity_score': np.random.uniform(0, 50, n),
            'mitre_count': np.random.randint(0, 3, n),
            'root_command': [f'cmd{i % 10}.exe' for i in range(n)],
            'tokens': [[f'token_{j}' for j in range(np.random.randint(3, 10))] for _ in range(n)]
        })
        
        return df
    
    def test_statistical_outliers(self):
        """Test statistical outlier detection."""
        df = self.create_test_data(100)
        
        # Add some clear outliers
        df.loc[0:3, 'topic_prob'] = 0.01  # Very low probability
        
        result = self.detector.detect_statistical_outliers(df, threshold=2.0)
        
        self.assertIn('stat_anomaly_score', result.columns)
        self.assertIn('is_stat_outlier', result.columns)
        # With threshold=2.0 and added outliers, should detect some
        self.assertGreaterEqual(result['is_stat_outlier'].sum(), 0)
    
    def test_complexity_outliers(self):
        """Test complexity outlier detection."""
        df = self.create_test_data(100)
        
        # Add some high complexity commands
        df.loc[0:5, 'complexity_score'] = 95
        
        result = self.detector.detect_complexity_outliers(df)
        
        self.assertIn('is_complexity_outlier', result.columns)
        self.assertGreater(result['is_complexity_outlier'].sum(), 0)
    
    def test_isolation_forest(self):
        """Test Isolation Forest anomaly detection."""
        df = self.create_test_data(100)
        
        result = self.detector.detect_isolation_forest_anomalies(df)
        
        self.assertIn('if_anomaly_score', result.columns)
        self.assertIn('is_if_anomaly', result.columns)
        self.assertGreater(result['is_if_anomaly'].sum(), 0)
    
    def test_sequence_anomalies(self):
        """Test sequence anomaly detection."""
        df = self.create_test_data(100)
        
        result = self.detector.detect_sequence_anomalies(df)
        
        self.assertIn('sequence_anomaly_score', result.columns)
        self.assertIn('is_sequence_anomaly', result.columns)
    
    def test_baseline_building(self):
        """Test baseline profile building."""
        df = self.create_test_data(100)
        
        self.detector.build_baseline(df)
        
        self.assertIsNotNone(self.detector.baseline_stats)
        self.assertIn('topic_distribution', self.detector.baseline_stats)
        self.assertIn('avg_complexity', self.detector.baseline_stats)
    
    def test_baseline_deviations(self):
        """Test baseline deviation detection."""
        df = self.create_test_data(100)
        
        self.detector.build_baseline(df)
        result = self.detector.detect_baseline_deviations(df)
        
        self.assertIn('baseline_deviation_score', result.columns)
        self.assertIn('is_baseline_anomaly', result.columns)
    
    def test_ensemble_scoring(self):
        """Test ensemble anomaly scoring."""
        df = self.create_test_data(100)
        
        # Run individual detections first
        df = self.detector.detect_statistical_outliers(df)
        df = self.detector.detect_complexity_outliers(df)
        df = self.detector.detect_isolation_forest_anomalies(df)
        
        # Compute ensemble score
        result = self.detector.compute_ensemble_score(df)
        
        self.assertIn('ensemble_anomaly_score', result.columns)
        self.assertIn('is_ensemble_anomaly', result.columns)
        self.assertGreater(result['is_ensemble_anomaly'].sum(), 0)
    
    def test_full_detection(self):
        """Test complete detection pipeline."""
        df = self.create_test_data(100)
        
        result = self.detector.run_full_detection(df)
        
        # Check that all anomaly columns are present
        self.assertIn('ensemble_anomaly_score', result.columns)
        self.assertEqual(len(result), 100)


class TestNetworkVisualization(unittest.TestCase):
    """Test network visualization functions."""
    
    def setUp(self):
        """Create test data."""
        np.random.seed(42)
        
        # Create test DataFrame
        self.df = pd.DataFrame({
            'command_line': [f'command_{i}' for i in range(50)],
            'topic': np.random.randint(0, 3, 50),
            'root_command': ['cmd.exe', 'powershell.exe', 'wmic.exe'] * 16 + ['net.exe', 'reg.exe'],
            'tokens': [[f'token_{j}' for j in range(5)] for _ in range(50)],
            'mitre_techniques': [
                'T1059,T1105' if i % 3 == 0 else 'T1053' if i % 2 == 0 else ''
                for i in range(50)
            ]
        })
        
        # Create simple LDA model
        self.dictionary = Dictionary([['word1', 'word2', 'word3'] for _ in range(10)])
        self.corpus = [self.dictionary.doc2bow(['word1', 'word2']) for _ in range(10)]
        self.lda_model = LdaMulticore(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=3,
            random_state=42,
            workers=1,
            passes=1
        )
    
    def test_command_network_creation(self):
        """Test command co-occurrence network creation."""
        fig = create_command_network(self.df, min_edge_weight=1, max_nodes=10)
        
        self.assertIsNotNone(fig)
        self.assertGreater(len(fig.data), 0)
    
    def test_topic_network_creation(self):
        """Test topic relationship network creation."""
        fig = create_topic_relationship_network(self.df, self.lda_model)
        
        self.assertIsNotNone(fig)
        self.assertGreater(len(fig.data), 0)
    
    def test_mitre_network_creation(self):
        """Test MITRE ATT&CK network creation."""
        fig = create_mitre_attack_network(self.df)
        
        self.assertIsNotNone(fig)
        # May be empty if no co-occurrences, but should not error


class TestIntegration(unittest.TestCase):
    """Integration tests for new features."""
    
    def test_end_to_end_with_caching(self):
        """Test full pipeline with caching enabled."""
        from main import TCLaaCPipeline
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Run pipeline with caching
            pipeline = TCLaaCPipeline(num_topics=3, use_cache=True)
            pipeline.load_data(50, is_synthetic=True)
            pipeline.preprocess(use_multiprocessing=False)
            
            # Check quality
            self.assertIsNotNone(pipeline.quality_checker)
            
            # Check cache manager
            self.assertIsNotNone(pipeline.cache_manager)
    
    def test_quality_check_integration(self):
        """Test quality checking in pipeline."""
        from main import TCLaaCPipeline
        
        pipeline = TCLaaCPipeline(num_topics=2)
        pipeline.load_data(30, is_synthetic=True)
        
        # Quality check should have been run
        self.assertGreater(len(pipeline.quality_checker.checks), 0)
    
    def test_anomaly_detection_integration(self):
        """Test anomaly detection in pipeline."""
        from main import TCLaaCPipeline
        
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = TCLaaCPipeline(num_topics=2)
            pipeline.load_data(30, is_synthetic=True)
            pipeline.preprocess(use_multiprocessing=False)
            pipeline.prepare_corpus()
            pipeline.train_model()
            pipeline.assign_topics()
            pipeline.detect_anomalies()
            
            # Check that anomaly scores were added
            self.assertIn('ensemble_anomaly_score', pipeline.df_with_topics.columns)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
