"""
Main pipeline for TCLaaC (The Command Line as a Corpus).

This module provides a streamlined, production-ready pipeline for analyzing
command-line logs using LDA topic modeling. It replaces the notebook workflow
with an optimized, command-line executable script.

Usage:
    python main.py --input data.csv --output results/
    python main.py --synthetic 10000 --tune
    python main.py --input data.csv --topics 15 --no-visualize
"""

import argparse
import logging
import time
import os
import sys
from pathlib import Path
from typing import Optional, Tuple
import warnings

import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore, CoherenceModel

# Local imports
import config
from config import (
    NUM_TOPICS, RANDOM_STATE, LDA_ALPHA, LDA_ETA, LDA_PASSES,
    MIN_DOC_LENGTH, NUM_WORKERS, LOLBAS_REPO_PATH, MODEL_FILENAME,
    ANALYSIS_DATAFRAME_FILENAME, COHERENCE_METHOD, MAX_TUNING_SAMPLE
)
from data_loader import (
    load_from_csv, generate_synthetic_data, validate_dataframe
)
from helpers import (
    normalize_command, tokenize, identify_root,
    get_all_lolbas_commands, get_topic_results_gensim,
    analyze_malicious_topics, generate_keywords_from_lolbas,
    NORMALIZATION_RULES_COMPILED
)
from graphs import create_topic_treemap_gensim

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('tclaaс_pipeline.log')
    ]
)
logger = logging.getLogger(__name__)


class TCLaaCPipeline:
    """
    Main pipeline class for command-line analysis using LDA topic modeling.
    """
    
    def __init__(self, num_topics: int = NUM_TOPICS, random_state: int = RANDOM_STATE):
        """
        Initialize the pipeline with configuration.
        
        Args:
            num_topics: Number of topics for LDA model
            random_state: Random seed for reproducibility
        """
        self.num_topics = num_topics
        self.random_state = random_state
        self.data = None
        self.dictionary = None
        self.corpus = None
        self.lda_model = None
        self.df_with_topics = None
        
        logger.info(f"Initialized TCLaaC Pipeline (topics={num_topics}, seed={random_state})")
    
    def load_data(self, source: str, is_synthetic: bool = False, limit: Optional[int] = None):
        """
        Load command-line data from a source.
        
        Args:
            source: Path to CSV file or number of synthetic samples
            is_synthetic: Whether to generate synthetic data
            limit: Maximum number of rows to load (None = all)
        """
        logger.info("=" * 70)
        logger.info("STEP 1: DATA LOADING")
        logger.info("=" * 70)
        
        if is_synthetic:
            num_samples = int(source) if isinstance(source, (int, str)) else 10000
            self.data = generate_synthetic_data(num_samples, seed=self.random_state)
        else:
            self.data = load_from_csv(source)
        
        # Apply limit if specified
        if limit and len(self.data) > limit:
            logger.info(f"Limiting dataset to {limit} rows (from {len(self.data)})")
            self.data = self.data.sample(n=limit, random_state=self.random_state)
        
        validate_dataframe(self.data)
        logger.info(f"✓ Loaded {len(self.data)} command lines\n")
    
    def enrich_with_lolbas(self):
        """
        Enrich dataset with LOLBAS command examples.
        """
        logger.info("=" * 70)
        logger.info("STEP 2: LOLBAS ENRICHMENT")
        logger.info("=" * 70)
        
        if not os.path.exists(LOLBAS_REPO_PATH):
            logger.warning(f"LOLBAS repository not found at {LOLBAS_REPO_PATH}")
            logger.warning("Skipping LOLBAS enrichment")
            return
        
        try:
            lolbas_commands = get_all_lolbas_commands(LOLBAS_REPO_PATH)
            logger.info(f"Extracted {len(lolbas_commands)} LOLBAS commands")
            
            lolbas_df = pd.DataFrame({'command_line': lolbas_commands})
            original_len = len(self.data)
            self.data = pd.concat([self.data, lolbas_df], ignore_index=True)
            
            logger.info(f"✓ Dataset enriched: {original_len} → {len(self.data)} rows\n")
        except Exception as e:
            logger.error(f"Error loading LOLBAS data: {e}")
            logger.warning("Continuing without LOLBAS enrichment")
    
    def preprocess(self, use_multiprocessing: bool = True):
        """
        Preprocess command lines: normalize, tokenize, identify roots.
        
        Args:
            use_multiprocessing: Whether to use parallel processing
        """
        logger.info("=" * 70)
        logger.info("STEP 3: PREPROCESSING")
        logger.info("=" * 70)
        
        start_time = time.time()
        
        if use_multiprocessing:
            logger.info("Using multiprocessing for faster preprocessing...")
            self._preprocess_parallel()
        else:
            logger.info("Using sequential processing...")
            self._preprocess_sequential()
        
        # Filter out short documents
        initial_len = len(self.data)
        self.data = self.data[self.data['tokens'].apply(len) >= MIN_DOC_LENGTH]
        filtered = initial_len - len(self.data)
        
        elapsed = time.time() - start_time
        logger.info(f"✓ Preprocessing complete in {elapsed:.2f}s")
        logger.info(f"  Filtered {filtered} short documents (< {MIN_DOC_LENGTH} tokens)")
        logger.info(f"  Final dataset: {len(self.data)} rows\n")
    
    def _preprocess_sequential(self):
        """Sequential preprocessing with progress bar."""
        tqdm.pandas(desc="Normalizing")
        self.data['normalized_command'] = self.data['command_line'].progress_apply(
            lambda x: normalize_command(x, NORMALIZATION_RULES_COMPILED)
        )
        
        tqdm.pandas(desc="Tokenizing")
        self.data['tokens'] = self.data['normalized_command'].progress_apply(tokenize)
        
        tqdm.pandas(desc="Identifying roots")
        self.data['root_command'] = self.data['command_line'].progress_apply(identify_root)
    
    def _preprocess_parallel(self):
        """Parallel preprocessing using multiprocessing."""
        from multiprocessing import Pool, cpu_count
        import functools
        
        command_lines = self.data['command_line'].tolist()
        num_cores = cpu_count()
        
        logger.info(f"Using {num_cores} CPU cores")
        
        with Pool(processes=num_cores) as pool:
            # Normalize
            logger.info("Normalizing commands...")
            partial_normalize = functools.partial(
                normalize_command,
                rules_dict=NORMALIZATION_RULES_COMPILED
            )
            normalized = list(tqdm(
                pool.imap(partial_normalize, command_lines),
                total=len(command_lines),
                desc="Normalizing"
            ))
            
            # Tokenize
            logger.info("Tokenizing commands...")
            tokenized = list(tqdm(
                pool.imap(tokenize, normalized),
                total=len(normalized),
                desc="Tokenizing"
            ))
            
            # Identify roots
            logger.info("Identifying root commands...")
            roots = list(tqdm(
                pool.imap(identify_root, command_lines),
                total=len(command_lines),
                desc="Root identification"
            ))
        
        self.data['normalized_command'] = normalized
        self.data['tokens'] = tokenized
        self.data['root_command'] = roots
    
    def prepare_corpus(self):
        """
        Prepare Gensim dictionary and corpus from tokenized data.
        """
        logger.info("=" * 70)
        logger.info("STEP 4: CORPUS PREPARATION")
        logger.info("=" * 70)
        
        tokens_list = self.data['tokens'].tolist()
        
        logger.info("Creating Gensim dictionary...")
        self.dictionary = Dictionary(tokens_list)
        logger.info(f"Dictionary contains {len(self.dictionary)} unique tokens")
        
        logger.info("Converting to bag-of-words corpus...")
        self.corpus = [self.dictionary.doc2bow(tokens) for tokens in tqdm(tokens_list, desc="Creating corpus")]
        
        logger.info(f"✓ Corpus prepared: {len(self.corpus)} documents\n")
    
    def tune_hyperparameters(self, topic_range: Optional[Tuple[int, int, int]] = None):
        """
        Tune the number of topics and alpha/eta hyperparameters.
        
        Args:
            topic_range: (start, stop, step) for number of topics to test
        """
        logger.info("=" * 70)
        logger.info("STEP 5: HYPERPARAMETER TUNING")
        logger.info("=" * 70)
        
        if topic_range is None:
            topic_range = (
                config.TUNING_MIN_TOPICS,
                config.TUNING_MAX_TOPICS,
                config.TUNING_STEP
            )
        
        # Sample data for faster tuning
        sample_size = min(len(self.corpus), MAX_TUNING_SAMPLE)
        if sample_size < len(self.corpus):
            logger.info(f"Sampling {sample_size} documents for tuning (full dataset: {len(self.corpus)})")
            import random
            random.seed(self.random_state)
            indices = random.sample(range(len(self.corpus)), sample_size)
            sample_corpus = [self.corpus[i] for i in indices]
            sample_tokens = [self.data['tokens'].iloc[i] for i in indices]
        else:
            sample_corpus = self.corpus
            sample_tokens = self.data['tokens'].tolist()
        
        # Tune number of topics
        logger.info(f"Tuning number of topics (range: {topic_range[0]}-{topic_range[1]}, step: {topic_range[2]})...")
        best_num_topics, coherence_scores = self._tune_num_topics(
            sample_corpus, sample_tokens, topic_range
        )
        
        logger.info(f"✓ Best number of topics: {best_num_topics}")
        logger.info(f"  Coherence score: {max(coherence_scores):.4f}\n")
        
        # Update configuration
        self.num_topics = best_num_topics
        
        return best_num_topics, coherence_scores
    
    def _tune_num_topics(self, corpus, tokens, topic_range):
        """Helper method to tune number of topics."""
        start, stop, step = topic_range
        coherence_scores = []
        topic_numbers = list(range(start, stop, step))
        
        for num_topics in tqdm(topic_numbers, desc="Training models"):
            model = LdaMulticore(
                corpus=corpus,
                id2word=self.dictionary,
                num_topics=num_topics,
                random_state=self.random_state,
                workers=NUM_WORKERS,
                passes=5  # Fewer passes for tuning
            )
            
            coherence_model = CoherenceModel(
                model=model,
                texts=tokens,
                dictionary=self.dictionary,
                coherence=COHERENCE_METHOD
            )
            coherence_scores.append(coherence_model.get_coherence())
        
        # Find best score
        best_idx = np.argmax(coherence_scores)
        best_num_topics = topic_numbers[best_idx]
        
        return best_num_topics, coherence_scores
    
    def train_model(self):
        """
        Train the final LDA model with optimized parameters.
        """
        logger.info("=" * 70)
        logger.info("STEP 6: MODEL TRAINING")
        logger.info("=" * 70)
        
        logger.info(f"Training LDA model with {self.num_topics} topics...")
        logger.info(f"Parameters: alpha={LDA_ALPHA}, eta={LDA_ETA}, passes={LDA_PASSES}")
        
        start_time = time.time()
        
        self.lda_model = LdaMulticore(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=self.num_topics,
            alpha=LDA_ALPHA,
            eta=LDA_ETA,
            random_state=self.random_state,
            passes=LDA_PASSES,
            workers=NUM_WORKERS,
            chunksize=config.LDA_CHUNKSIZE
        )
        
        elapsed = time.time() - start_time
        logger.info(f"✓ Model training complete in {elapsed:.2f}s\n")
        
        # Display top words for each topic
        logger.info("Top words per topic:")
        for idx, topic in self.lda_model.print_topics(num_words=5):
            logger.info(f"  Topic {idx}: {topic}")
        print()
    
    def assign_topics(self):
        """
        Assign topics to all documents in the dataset.
        """
        logger.info("=" * 70)
        logger.info("STEP 7: TOPIC ASSIGNMENT")
        logger.info("=" * 70)
        
        self.df_with_topics = get_topic_results_gensim(
            self.lda_model,
            self.corpus,
            self.data
        )
        
        # Show topic distribution
        topic_counts = self.df_with_topics['topic'].value_counts().sort_index()
        logger.info("\nTopic distribution:")
        for topic_id, count in topic_counts.items():
            percentage = (count / len(self.df_with_topics)) * 100
            logger.info(f"  Topic {topic_id}: {count} documents ({percentage:.1f}%)")
        print()
    
    def analyze_security(self):
        """
        Analyze topics for potential security concerns using LOLBAS keywords.
        """
        logger.info("=" * 70)
        logger.info("STEP 8: SECURITY ANALYSIS")
        logger.info("=" * 70)
        
        if not os.path.exists(LOLBAS_REPO_PATH):
            logger.warning("LOLBAS repository not found, skipping security analysis")
            return
        
        try:
            lolbas_keywords = generate_keywords_from_lolbas(LOLBAS_REPO_PATH)
            logger.info(f"Analyzing with {len(lolbas_keywords)} LOLBAS keywords...")
            
            topic_scores = analyze_malicious_topics(self.df_with_topics, lolbas_keywords)
            
            logger.info("\n✓ Security analysis complete\n")
            return topic_scores
        except Exception as e:
            logger.error(f"Error during security analysis: {e}")
            return None
    
    def visualize(self, output_path: Optional[str] = None):
        """
        Generate interactive treemap visualization.
        
        Args:
            output_path: Path to save HTML visualization
        """
        logger.info("=" * 70)
        logger.info("STEP 9: VISUALIZATION")
        logger.info("=" * 70)
        
        try:
            logger.info("Generating interactive treemap...")
            fig = create_topic_treemap_gensim(
                self.df_with_topics,
                self.lda_model,
                similarity_threshold=config.SIMILARITY_THRESHOLD
            )
            
            if output_path:
                fig.write_html(output_path)
                logger.info(f"✓ Treemap saved to: {output_path}\n")
            else:
                fig.show()
                logger.info("✓ Treemap displayed\n")
        except Exception as e:
            logger.error(f"Error during visualization: {e}")
    
    def save_results(self, output_dir: str = '.'):
        """
        Save model and analysis results to disk.
        
        Args:
            output_dir: Directory to save output files
        """
        logger.info("=" * 70)
        logger.info("STEP 10: SAVING RESULTS")
        logger.info("=" * 70)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = output_path / MODEL_FILENAME
        joblib.dump(self.lda_model, model_path)
        logger.info(f"✓ Model saved: {model_path}")
        
        # Save analysis DataFrame
        df_path = output_path / ANALYSIS_DATAFRAME_FILENAME
        self.df_with_topics.to_parquet(df_path)
        logger.info(f"✓ Analysis DataFrame saved: {df_path}")
        
        # Save topic summary
        summary_path = output_path / 'topic_summary.csv'
        topic_summary = []
        for idx in range(self.num_topics):
            top_words = [word for word, _ in self.lda_model.show_topic(idx, topn=10)]
            topic_summary.append({
                'topic_id': idx,
                'top_words': ', '.join(top_words),
                'num_documents': (self.df_with_topics['topic'] == idx).sum()
            })
        pd.DataFrame(topic_summary).to_csv(summary_path, index=False)
        logger.info(f"✓ Topic summary saved: {summary_path}\n")
    
    def run_full_pipeline(
        self,
        source: str,
        is_synthetic: bool = False,
        tune: bool = False,
        visualize: bool = True,
        output_dir: str = '.',
        limit: Optional[int] = None
    ):
        """
        Run the complete analysis pipeline.
        
        Args:
            source: Data source (CSV path or synthetic count)
            is_synthetic: Whether to generate synthetic data
            tune: Whether to run hyperparameter tuning
            visualize: Whether to generate visualizations
            output_dir: Directory for output files
            limit: Maximum rows to process
        """
        try:
            pipeline_start = time.time()
            
            # Execute pipeline steps
            self.load_data(source, is_synthetic, limit)
            self.enrich_with_lolbas()
            self.preprocess(use_multiprocessing=True)
            self.prepare_corpus()
            
            if tune:
                self.tune_hyperparameters()
            
            self.train_model()
            self.assign_topics()
            self.analyze_security()
            
            if visualize:
                viz_path = Path(output_dir) / 'topic_treemap.html'
                self.visualize(str(viz_path))
            
            self.save_results(output_dir)
            
            total_time = time.time() - pipeline_start
            logger.info("=" * 70)
            logger.info(f"✓ PIPELINE COMPLETE in {total_time:.2f}s ({total_time/60:.2f} minutes)")
            logger.info("=" * 70)
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            sys.exit(1)


def main():
    """Command-line interface for TCLaaC pipeline."""
    parser = argparse.ArgumentParser(
        description='TCLaaC: Command Line Analysis using LDA Topic Modeling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze CSV file
  python main.py --input data.csv --output results/
  
  # Generate synthetic data and tune hyperparameters
  python main.py --synthetic 10000 --tune --output test_results/
  
  # Quick analysis with specific topic count
  python main.py --input data.csv --topics 15 --no-visualize
  
  # Limit dataset size for testing
  python main.py --input large_data.csv --limit 50000
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input', '-i',
        type=str,
        help='Path to input CSV file with command_line column'
    )
    input_group.add_argument(
        '--synthetic', '-s',
        type=int,
        metavar='N',
        help='Generate N synthetic command lines for testing'
    )
    
    # Model parameters
    parser.add_argument(
        '--topics', '-t',
        type=int,
        default=NUM_TOPICS,
        help=f'Number of topics for LDA model (default: {NUM_TOPICS})'
    )
    parser.add_argument(
        '--tune',
        action='store_true',
        help='Run hyperparameter tuning (slower but finds optimal topic count)'
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        help='Limit number of rows to process (for testing)'
    )
    
    # Output options
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='.',
        help='Output directory for results (default: current directory)'
    )
    parser.add_argument(
        '--no-visualize',
        action='store_true',
        help='Skip visualization generation (faster)'
    )
    
    # Other options
    parser.add_argument(
        '--seed',
        type=int,
        default=RANDOM_STATE,
        help=f'Random seed for reproducibility (default: {RANDOM_STATE})'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Print banner
    print("\n" + "=" * 70)
    print("  TCLaaC - The Command Line as a Corpus")
    print("  LDA Topic Modeling for Security Log Analysis")
    print("=" * 70 + "\n")
    
    # Initialize pipeline
    pipeline = TCLaaCPipeline(
        num_topics=args.topics,
        random_state=args.seed
    )
    
    # Determine source
    source = args.input if args.input else args.synthetic
    is_synthetic = args.synthetic is not None
    
    # Run pipeline
    pipeline.run_full_pipeline(
        source=source,
        is_synthetic=is_synthetic,
        tune=args.tune,
        visualize=not args.no_visualize,
        output_dir=args.output,
        limit=args.limit
    )


if __name__ == '__main__':
    main()
