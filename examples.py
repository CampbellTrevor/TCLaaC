#!/usr/bin/env python
"""
Example usage scenarios for TCLaaC.

This file contains ready-to-run examples demonstrating various ways
to use the TCLaaC pipeline for command-line analysis.
"""

# =============================================================================
# Example 1: Quick Test with Synthetic Data
# =============================================================================

def example_quick_test():
    """
    Fastest way to test the pipeline - uses synthetic data.
    Good for: First-time setup validation, testing changes
    """
    print("=" * 70)
    print("Example 1: Quick Test with Synthetic Data")
    print("=" * 70)
    
    from main import TCLaaCPipeline
    
    pipeline = TCLaaCPipeline(num_topics=5, random_state=42)
    
    pipeline.run_full_pipeline(
        source='1000',           # Generate 1000 synthetic samples
        is_synthetic=True,
        tune=False,              # Skip tuning for speed
        visualize=False,         # Skip viz for speed
        output_dir='quick_test'
    )
    
    print("\n✓ Quick test complete! Check 'quick_test/' directory.")


# =============================================================================
# Example 2: Full Analysis with CSV Data
# =============================================================================

def example_csv_analysis():
    """
    Complete analysis pipeline with CSV input.
    Good for: Real data analysis, production use
    """
    print("=" * 70)
    print("Example 2: Full Analysis with CSV Data")
    print("=" * 70)
    
    from main import TCLaaCPipeline
    
    # First, create sample CSV if needed
    from data_loader import create_sample_csv
    create_sample_csv('my_sysmon_data.csv', num_samples=5000)
    
    pipeline = TCLaaCPipeline(num_topics=11, random_state=42)
    
    pipeline.run_full_pipeline(
        source='my_sysmon_data.csv',
        is_synthetic=False,
        tune=False,
        visualize=True,
        output_dir='csv_analysis'
    )
    
    print("\n✓ CSV analysis complete!")
    print("Open 'csv_analysis/topic_treemap.html' to view results.")


# =============================================================================
# Example 3: Hyperparameter Tuning
# =============================================================================

def example_hyperparameter_tuning():
    """
    Find optimal number of topics through coherence analysis.
    Good for: New datasets, research, optimal results
    """
    print("=" * 70)
    print("Example 3: Hyperparameter Tuning")
    print("=" * 70)
    
    from main import TCLaaCPipeline
    import matplotlib.pyplot as plt
    
    pipeline = TCLaaCPipeline(random_state=42)
    
    # Load data
    pipeline.load_data('5000', is_synthetic=True)
    pipeline.enrich_with_lolbas()
    pipeline.preprocess()
    pipeline.prepare_corpus()
    
    # Tune with custom range
    best_topics, coherence_scores = pipeline.tune_hyperparameters(
        topic_range=(5, 30, 2)  # Test 5, 7, 9, ..., 29 topics
    )
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(range(5, 30, 2), coherence_scores, marker='o')
    plt.xlabel('Number of Topics')
    plt.ylabel('Coherence Score')
    plt.title('Topic Coherence Analysis')
    plt.grid(True)
    plt.savefig('coherence_plot.png')
    print(f"\n✓ Tuning complete! Best: {best_topics} topics")
    print("Plot saved to 'coherence_plot.png'")


# =============================================================================
# Example 4: Step-by-Step Pipeline
# =============================================================================

def example_step_by_step():
    """
    Run each pipeline step individually for fine control.
    Good for: Debugging, custom workflows, learning
    """
    print("=" * 70)
    print("Example 4: Step-by-Step Pipeline")
    print("=" * 70)
    
    from main import TCLaaCPipeline
    import pandas as pd
    
    pipeline = TCLaaCPipeline(num_topics=8, random_state=42)
    
    # Step 1: Load
    print("\n1. Loading data...")
    pipeline.load_data('2000', is_synthetic=True)
    print(f"   Loaded {len(pipeline.data)} rows")
    
    # Step 2: Enrich (optional)
    print("\n2. Enriching with LOLBAS...")
    pipeline.enrich_with_lolbas()
    
    # Step 3: Preprocess
    print("\n3. Preprocessing...")
    pipeline.preprocess(use_multiprocessing=True)
    print(f"   Created tokens for {len(pipeline.data)} documents")
    
    # Step 4: Prepare corpus
    print("\n4. Preparing corpus...")
    pipeline.prepare_corpus()
    print(f"   Dictionary size: {len(pipeline.dictionary)} tokens")
    
    # Step 5: Train
    print("\n5. Training model...")
    pipeline.train_model()
    
    # Step 6: Assign topics
    print("\n6. Assigning topics...")
    pipeline.assign_topics()
    
    # Step 7: Analyze
    print("\n7. Security analysis...")
    topic_scores = pipeline.analyze_security()
    
    # Step 8: Visualize
    print("\n8. Generating visualization...")
    pipeline.visualize('step_by_step_treemap.html')
    
    # Step 9: Save
    print("\n9. Saving results...")
    pipeline.save_results('step_by_step_results')
    
    print("\n✓ Step-by-step pipeline complete!")


# =============================================================================
# Example 5: Custom Preprocessing
# =============================================================================

def example_custom_preprocessing():
    """
    Customize preprocessing with your own rules.
    Good for: Domain-specific analysis, custom patterns
    """
    print("=" * 70)
    print("Example 5: Custom Preprocessing")
    print("=" * 70)
    
    import re
    from config import NORMALIZATION_RULES_COMPILED
    from helpers import normalize_command, tokenize
    
    # Add custom normalization rule
    custom_rules = NORMALIZATION_RULES_COMPILED.copy()
    custom_rules['custom_pattern'] = (
        re.compile(r'\\Users\\[^\\]+\\', re.IGNORECASE),
        '<USER_PATH>'
    )
    
    # Test it
    test_cmd = 'powershell.exe -File C:\\Users\\Alice\\script.ps1'
    normalized = normalize_command(test_cmd, custom_rules)
    tokens = tokenize(normalized)
    
    print(f"Original:   {test_cmd}")
    print(f"Normalized: {normalized}")
    print(f"Tokens:     {tokens}")
    
    print("\n✓ Custom preprocessing example complete!")


# =============================================================================
# Example 6: Batch Processing Multiple Files
# =============================================================================

def example_batch_processing():
    """
    Process multiple CSV files in sequence.
    Good for: Large-scale analysis, multiple datasets
    """
    print("=" * 70)
    print("Example 6: Batch Processing")
    print("=" * 70)
    
    from main import TCLaaCPipeline
    from data_loader import create_sample_csv
    import os
    
    # Create multiple sample files
    files = []
    for i in range(1, 4):
        filename = f'dataset_{i}.csv'
        create_sample_csv(filename, num_samples=500)
        files.append(filename)
    
    # Process each file
    for csv_file in files:
        print(f"\nProcessing {csv_file}...")
        
        pipeline = TCLaaCPipeline(num_topics=5)
        output_dir = f'batch_output_{csv_file[:-4]}'
        
        pipeline.run_full_pipeline(
            source=csv_file,
            is_synthetic=False,
            tune=False,
            visualize=False,  # Skip for speed
            output_dir=output_dir
        )
        
        print(f"✓ {csv_file} complete -> {output_dir}/")
    
    print("\n✓ Batch processing complete!")


# =============================================================================
# Example 7: Analysis and Reporting
# =============================================================================

def example_analysis_and_reporting():
    """
    Generate a comprehensive analysis report.
    Good for: Sharing results, documentation
    """
    print("=" * 70)
    print("Example 7: Analysis and Reporting")
    print("=" * 70)
    
    from main import TCLaaCPipeline
    import pandas as pd
    
    # Run pipeline
    pipeline = TCLaaCPipeline(num_topics=10, random_state=42)
    pipeline.run_full_pipeline(
        source='3000',
        is_synthetic=True,
        tune=False,
        visualize=True,
        output_dir='analysis_report'
    )
    
    # Generate custom report
    print("\nGenerating analysis report...")
    
    df = pipeline.df_with_topics
    
    report = []
    report.append("# TCLaaC Analysis Report\n")
    report.append(f"- Total documents: {len(df)}")
    report.append(f"- Unique tokens: {len(pipeline.dictionary)}")
    report.append(f"- Number of topics: {pipeline.num_topics}\n")
    
    report.append("## Topic Distribution\n")
    for topic_id in sorted(df['topic'].unique()):
        count = (df['topic'] == topic_id).sum()
        pct = (count / len(df)) * 100
        top_words = [w for w, _ in pipeline.lda_model.show_topic(topic_id, topn=5)]
        report.append(f"- **Topic {topic_id}**: {count} docs ({pct:.1f}%) - {', '.join(top_words)}")
    
    report.append("\n## Top Executables\n")
    top_exes = df['root_command'].value_counts().head(10)
    for exe, count in top_exes.items():
        report.append(f"- {exe}: {count} occurrences")
    
    # Save report
    with open('analysis_report/REPORT.md', 'w') as f:
        f.write('\n'.join(report))
    
    print("✓ Report saved to 'analysis_report/REPORT.md'")


# =============================================================================
# Example 8: Using the Python API
# =============================================================================

def example_python_api():
    """
    Programmatic usage from your own Python scripts.
    Good for: Integration, automation, custom workflows
    """
    print("=" * 70)
    print("Example 8: Python API Usage")
    print("=" * 70)
    
    # Option 1: Use the full pipeline class
    from main import TCLaaCPipeline
    
    pipeline = TCLaaCPipeline(num_topics=7)
    pipeline.load_data('500', is_synthetic=True)
    pipeline.preprocess()
    pipeline.prepare_corpus()
    pipeline.train_model()
    pipeline.assign_topics()
    
    # Access results programmatically
    df = pipeline.df_with_topics
    model = pipeline.lda_model
    
    print(f"\nResults available:")
    print(f"  - DataFrame: {len(df)} rows")
    print(f"  - Model: {model.num_topics} topics")
    
    # Option 2: Use individual helper functions
    from helpers import normalize_command, tokenize, identify_root
    from config import NORMALIZATION_RULES_COMPILED
    
    cmd = "powershell.exe -ExecutionPolicy Bypass"
    normalized = normalize_command(cmd, NORMALIZATION_RULES_COMPILED)
    tokens = tokenize(normalized)
    root = identify_root(cmd)
    
    print(f"\nProcessed command:")
    print(f"  - Root: {root}")
    print(f"  - Tokens: {len(tokens)}")
    
    print("\n✓ API usage example complete!")


# =============================================================================
# Main Menu
# =============================================================================

def main():
    """Run example menu."""
    examples = {
        '1': ('Quick Test (1 minute)', example_quick_test),
        '2': ('CSV Analysis', example_csv_analysis),
        '3': ('Hyperparameter Tuning', example_hyperparameter_tuning),
        '4': ('Step-by-Step Pipeline', example_step_by_step),
        '5': ('Custom Preprocessing', example_custom_preprocessing),
        '6': ('Batch Processing', example_batch_processing),
        '7': ('Analysis & Reporting', example_analysis_and_reporting),
        '8': ('Python API Usage', example_python_api),
    }
    
    print("\n" + "=" * 70)
    print("TCLaaC Usage Examples")
    print("=" * 70)
    print("\nChoose an example to run:\n")
    
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    
    print("\n  0. Exit")
    print()
    
    choice = input("Enter your choice (0-8): ").strip()
    
    if choice == '0':
        print("Goodbye!")
        return
    
    if choice in examples:
        _, func = examples[choice]
        print()
        try:
            func()
        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")
        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Invalid choice!")


if __name__ == '__main__':
    main()
