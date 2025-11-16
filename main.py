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
    NUM_TOPICS, RANDOM_STATE, LDA_ALPHA, LDA_ETA, LDA_PASSES, LDA_ITERATIONS,
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
            logger.info(f"Generating {num_samples} synthetic command lines...")
            self.data = generate_synthetic_data(num_samples, seed=self.random_state)
        else:
            logger.info(f"Loading data from: {source}")
            self.data = load_from_csv(source)
        
        # Apply limit if specified
        if limit and len(self.data) > limit:
            logger.info(f"Limiting dataset to {limit} rows")
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
        
        if use_multiprocessing and len(self.data) > 100:
            self._preprocess_parallel()
        else:
            self._preprocess_sequential()
        
        # Filter out short documents
        initial_len = len(self.data)
        self.data = self.data[self.data['tokens'].apply(len) >= MIN_DOC_LENGTH]
        filtered = initial_len - len(self.data)
        
        elapsed = time.time() - start_time
        logger.info(f"✓ Preprocessing complete in {elapsed:.2f}s")
        if filtered > 0:
            logger.info(f"  Filtered {filtered} short documents (< {MIN_DOC_LENGTH} tokens)")
        logger.info(f"  Final dataset: {len(self.data)} rows\n")
    
    def _preprocess_sequential(self):
        """Sequential preprocessing with progress bar."""
        tqdm.pandas(desc="⚙ Normalizing", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')
        self.data['normalized_command'] = self.data['command_line'].progress_apply(
            lambda x: normalize_command(x, NORMALIZATION_RULES_COMPILED)
        )
        
        tqdm.pandas(desc="⚙ Tokenizing", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')
        self.data['tokens'] = self.data['normalized_command'].progress_apply(tokenize)
        
        tqdm.pandas(desc="⚙ Root identification", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')
        self.data['root_command'] = self.data['command_line'].progress_apply(identify_root)
    
    def _preprocess_parallel(self):
        """Parallel preprocessing using multiprocessing."""
        from multiprocessing import Pool, cpu_count
        import functools
        
        command_lines = self.data['command_line'].tolist()
        num_cores = min(cpu_count(), 8)  # Cap at 8 cores for efficiency
        chunksize = max(1, len(command_lines) // (num_cores * 4))
        
        with Pool(processes=num_cores) as pool:
            # Normalize
            partial_normalize = functools.partial(
                normalize_command,
                rules_dict=NORMALIZATION_RULES_COMPILED
            )
            normalized = list(tqdm(
                pool.imap(partial_normalize, command_lines, chunksize=chunksize),
                total=len(command_lines),
                desc="⚙ Normalizing",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
            ))
            
            # Tokenize
            tokenized = list(tqdm(
                pool.imap(tokenize, normalized, chunksize=chunksize),
                total=len(normalized),
                desc="⚙ Tokenizing",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
            ))
            
            # Identify roots
            roots = list(tqdm(
                pool.imap(identify_root, command_lines, chunksize=chunksize),
                total=len(command_lines),
                desc="⚙ Root identification",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
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
        
        self.dictionary = Dictionary(tokens_list)
        logger.info(f"Dictionary: {len(self.dictionary)} unique tokens")
        
        self.corpus = [self.dictionary.doc2bow(tokens) for tokens in tqdm(
            tokens_list, 
            desc="⚙ Building corpus",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]'
        )]
        
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
        
        # Sample data for faster tuning with adaptive sampling
        sample_size = min(len(self.corpus), MAX_TUNING_SAMPLE)
        if sample_size < len(self.corpus):
            logger.info(f"Using {sample_size} documents for tuning")
            import random
            random.seed(self.random_state)
            indices = random.sample(range(len(self.corpus)), sample_size)
            sample_corpus = [self.corpus[i] for i in indices]
            sample_tokens = [self.data['tokens'].iloc[i] for i in indices]
        else:
            sample_corpus = self.corpus
            sample_tokens = self.data['tokens'].tolist()
        
        # Tune number of topics
        best_num_topics, coherence_scores = self._tune_num_topics(
            sample_corpus, sample_tokens, topic_range
        )
        
        logger.info(f"✓ Best number of topics: {best_num_topics} (coherence: {max(coherence_scores):.4f})\n")
        
        # Update configuration
        self.num_topics = best_num_topics
        
        return best_num_topics, coherence_scores
    
    def _tune_num_topics(self, corpus, tokens, topic_range):
        """Helper method to tune number of topics."""
        start, stop, step = topic_range
        coherence_scores = []
        topic_numbers = list(range(start, stop, step))
        
        # Progress bar with better formatting
        pbar = tqdm(
            topic_numbers, 
            desc="⚙ Tuning topics",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        )
        
        for num_topics in pbar:
            pbar.set_postfix({'topics': num_topics})
            
            model = LdaMulticore(
                corpus=corpus,
                id2word=self.dictionary,
                num_topics=num_topics,
                random_state=self.random_state,
                workers=NUM_WORKERS,
                passes=3,  # Reduced passes for faster tuning
                iterations=50,  # Fewer iterations
                chunksize=min(2000, len(corpus))
            )
            
            coherence_model = CoherenceModel(
                model=model,
                texts=tokens,
                dictionary=self.dictionary,
                coherence=COHERENCE_METHOD
            )
            score = coherence_model.get_coherence()
            coherence_scores.append(score)
            pbar.set_postfix({'topics': num_topics, 'coherence': f'{score:.3f}'})
        
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
        
        logger.info(f"Training LDA model: {self.num_topics} topics, {len(self.corpus)} documents")
        
        start_time = time.time()
        
        # Optimized chunking for better performance
        optimal_chunksize = min(
            max(100, len(self.corpus) // (NUM_WORKERS * 4)),
            config.LDA_CHUNKSIZE
        )
        
        # Suppress gensim's verbose logging
        gensim_logger = logging.getLogger('gensim')
        original_level = gensim_logger.level
        gensim_logger.setLevel(logging.ERROR)
        
        try:
            # Show progress bar during training
            num_chunks = max(1, len(self.corpus) // optimal_chunksize)
            total_updates = LDA_PASSES * num_chunks
            
            with tqdm(
                total=total_updates,
                desc="⚙ Training LDA",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} chunks [{elapsed}]',
                leave=True
            ) as pbar:
                # We'll approximate progress since we can't hook into gensim directly
                # Train one pass at a time and update progress
                self.lda_model = LdaMulticore(
                    corpus=self.corpus,
                    id2word=self.dictionary,
                    num_topics=self.num_topics,
                    alpha=LDA_ALPHA,
                    eta=LDA_ETA,
                    random_state=self.random_state,
                    passes=1,
                    iterations=LDA_ITERATIONS,
                    workers=NUM_WORKERS,
                    chunksize=optimal_chunksize,
                    per_word_topics=False
                )
                pbar.update(num_chunks)
                
                # Continue training for remaining passes
                for pass_num in range(2, LDA_PASSES + 1):
                    self.lda_model.update(self.corpus)
                    pbar.update(num_chunks)
            
            elapsed = time.time() - start_time
            logger.info(f"✓ Model training complete in {elapsed:.2f}s\n")
            
            # Display top words for each topic (compact format)
            logger.info("Top words per topic:")
            for idx, topic in self.lda_model.print_topics(num_words=5):
                # Extract just the words for cleaner display
                words = [word.split('*')[1].strip().strip('"') for word in topic.split(' + ')]
                logger.info(f"  Topic {idx}: {', '.join(words)}")
            print()
        finally:
            # Restore original logging level
            gensim_logger.setLevel(original_level)
    
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
        
        # Show topic distribution (compact format)
        topic_counts = self.df_with_topics['topic'].value_counts().sort_index()
        logger.info("Topic distribution:")
        for topic_id, count in topic_counts.items():
            percentage = (count / len(self.df_with_topics)) * 100
            bar_width = int(percentage / 2)  # Visual bar
            bar = '█' * bar_width
            logger.info(f"  Topic {topic_id:2d}: {bar} {count:4d} docs ({percentage:5.1f}%)")
        print()
    
    def analyze_security(self):
        """
        Analyze topics for potential security concerns using LOLBAS density metrics.
        """
        logger.info("=" * 70)
        logger.info("STEP 8: SECURITY ANALYSIS")
        logger.info("=" * 70)
        
        if not os.path.exists(LOLBAS_REPO_PATH):
            logger.warning("LOLBAS repository not found, skipping security analysis")
            return None
        
        try:
            lolbas_keywords = generate_keywords_from_lolbas(LOLBAS_REPO_PATH)
            
            topic_scores = analyze_malicious_topics(self.df_with_topics, lolbas_keywords)
            
            logger.info("\n✓ Security analysis complete\n")
            self.topic_scores = topic_scores  # Store for visualization
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
                # Ensure parent directory exists
                output_path_obj = Path(output_path)
                output_path_obj.parent.mkdir(parents=True, exist_ok=True)
                fig.write_html(str(output_path_obj))
                logger.info(f"✓ Treemap saved to: {output_path_obj}\n")
            else:
                fig.show()
                logger.info("✓ Treemap displayed\n")
        except Exception as e:
            logger.error(f"Error during visualization: {e}")
    
    def create_analysis_visualizations(self, output_dir: str = '.'):
        """
        Generate comprehensive analysis visualizations for data exploration.
        
        Args:
            output_dir: Directory to save visualization files
        """
        logger.info("=" * 70)
        logger.info("CREATING ANALYSIS VISUALIZATIONS")
        logger.info("=" * 70)
        
        from graphs import (
            create_topic_heatmap, 
            create_security_score_chart,
            create_topic_distribution_chart,
            create_command_length_distribution
        )
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # 1. Topic-Word Heatmap
            logger.info("Generating topic-word heatmap...")
            fig = create_topic_heatmap(self.df_with_topics, self.lda_model)
            heatmap_path = output_path / 'topic_word_heatmap.html'
            fig.write_html(str(heatmap_path))
            logger.info(f"✓ Heatmap saved to: {heatmap_path}")
            
            # 2. Security Score Chart (if analysis was performed)
            if hasattr(self, 'topic_scores') and self.topic_scores is not None:
                logger.info("Generating security risk chart...")
                fig = create_security_score_chart(self.topic_scores, self.df_with_topics, self.lda_model)
                security_path = output_path / 'security_risk_chart.html'
                fig.write_html(str(security_path))
                logger.info(f"✓ Security chart saved to: {security_path}")
            
            # 3. Topic Distribution Sunburst
            logger.info("Generating topic distribution sunburst...")
            fig = create_topic_distribution_chart(self.df_with_topics, self.lda_model)
            sunburst_path = output_path / 'topic_distribution_sunburst.html'
            fig.write_html(str(sunburst_path))
            logger.info(f"✓ Sunburst saved to: {sunburst_path}")
            
            # 4. Command Length Distribution
            logger.info("Generating command length distribution...")
            fig = create_command_length_distribution(self.df_with_topics)
            length_path = output_path / 'command_length_boxplot.html'
            fig.write_html(str(length_path))
            logger.info(f"✓ Box plot saved to: {length_path}")
            
            logger.info(f"\n✓ All visualizations created in: {output_path}\n")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            import traceback
            traceback.print_exc()
    
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
        
        # Save topic summary with smart topic names
        from graphs import generate_smart_topic_summary
        summary_path = output_path / 'topic_summary.csv'
        topic_summary = []
        for idx in range(self.num_topics):
            top_words = [word for word, _ in self.lda_model.show_topic(idx, topn=10)]
            smart_name = generate_smart_topic_summary(idx, self.lda_model, self.df_with_topics)
            topic_summary.append({
                'topic_id': idx,
                'topic_name': smart_name,
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
                # Generate comprehensive analysis visualizations
                self.create_analysis_visualizations(output_dir)
            
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
