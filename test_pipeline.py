"""
Test script for TCLaaC pipeline.

This script runs basic tests to ensure all components are working correctly.
Run this before starting your full analysis to catch any issues early.
"""

import sys
import os

print("=" * 70)
print("TCLaaC Pipeline Test Suite")
print("=" * 70)
print()

# Test 1: Check imports
print("Test 1: Checking imports...")
try:
    import pandas as pd
    import numpy as np
    from gensim.corpora import Dictionary
    from gensim.models import LdaMulticore
    import config
    from data_loader import generate_synthetic_data, load_from_csv
    from helpers import normalize_command, tokenize, identify_root
    from graphs import create_topic_treemap_gensim
    print("  ✓ All imports successful")
except ImportError as e:
    print(f"  ✗ Import failed: {e}")
    print("  → Run: pip install -r requirements.txt")
    sys.exit(1)

# Test 2: Check configuration
print("\nTest 2: Validating configuration...")
try:
    config.validate_config()
    print(f"  ✓ Configuration valid")
    print(f"    - Topics: {config.NUM_TOPICS}")
    print(f"    - Random state: {config.RANDOM_STATE}")
except Exception as e:
    print(f"  ✗ Configuration error: {e}")
    sys.exit(1)

# Test 3: Test synthetic data generation
print("\nTest 3: Testing synthetic data generation...")
try:
    df = generate_synthetic_data(num_samples=100)
    if len(df) != 100:
        raise ValueError(f"Expected 100 samples, got {len(df)}")
    if 'command_line' not in df.columns:
        raise ValueError("Missing 'command_line' column")
    print(f"  ✓ Generated {len(df)} synthetic samples")
    print(f"    Sample: {df.iloc[0]['command_line'][:60]}...")
except Exception as e:
    print(f"  ✗ Synthetic data generation failed: {e}")
    sys.exit(1)

# Test 4: Test preprocessing functions
print("\nTest 4: Testing preprocessing functions...")
try:
    test_command = 'powershell.exe -File C:\\Users\\Admin\\script_12345678.ps1 -IP 192.168.1.1'
    
    # Test normalization
    normalized = normalize_command(test_command, config.NORMALIZATION_RULES_COMPILED)
    if '<LONG_NUMBER>' not in normalized or '<IP_ADDRESS>' not in normalized:
        print(f"  ⚠ Warning: Normalization may not be working correctly")
        print(f"    Input:  {test_command}")
        print(f"    Output: {normalized}")
    
    # Test tokenization
    tokens = tokenize(normalized)
    if len(tokens) < 3:
        raise ValueError(f"Tokenization produced too few tokens: {tokens}")
    
    # Test root identification
    root = identify_root(test_command)
    if root != 'powershell.exe':
        print(f"  ⚠ Warning: Root identification unexpected: '{root}' (expected 'powershell.exe')")
    
    print(f"  ✓ Preprocessing functions working")
    print(f"    Tokens: {len(tokens)}")
    print(f"    Root: {root}")
except Exception as e:
    print(f"  ✗ Preprocessing failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test LOLBAS loading
print("\nTest 5: Testing LOLBAS data loading...")
try:
    if not os.path.exists(config.LOLBAS_REPO_PATH):
        print(f"  ⚠ LOLBAS repository not found at {config.LOLBAS_REPO_PATH}")
        print(f"    This is optional but recommended for security analysis")
    else:
        from helpers import get_all_lolbas_commands, generate_keywords_from_lolbas
        commands = get_all_lolbas_commands(config.LOLBAS_REPO_PATH)
        keywords = generate_keywords_from_lolbas(config.LOLBAS_REPO_PATH)
        print(f"  ✓ LOLBAS data loaded")
        print(f"    Commands: {len(commands)}")
        print(f"    Keywords: {len(keywords)}")
except Exception as e:
    print(f"  ⚠ LOLBAS loading had issues: {e}")
    print(f"    Pipeline will work without LOLBAS data")

# Test 6: Test mini pipeline
print("\nTest 6: Running mini end-to-end pipeline...")
try:
    from main import TCLaaCPipeline
    
    pipeline = TCLaaCPipeline(num_topics=3, random_state=42)
    
    # Load small synthetic dataset
    pipeline.load_data('50', is_synthetic=True)
    
    # Preprocess
    pipeline.preprocess(use_multiprocessing=False)  # Sequential for testing
    
    # Prepare corpus
    pipeline.prepare_corpus()
    
    # Train tiny model
    pipeline.train_model()
    
    # Assign topics
    pipeline.assign_topics()
    
    print(f"  ✓ Mini pipeline completed successfully")
    print(f"    Processed {len(pipeline.data)} documents")
    print(f"    Created {len(pipeline.dictionary)} unique tokens")
    
except Exception as e:
    print(f"  ✗ Mini pipeline failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Create sample CSV for testing
print("\nTest 7: Creating sample CSV file...")
try:
    from data_loader import create_sample_csv
    test_csv_path = 'test_sample_data.csv'
    create_sample_csv(test_csv_path, num_samples=100)
    
    # Verify we can load it
    test_df = load_from_csv(test_csv_path)
    if len(test_df) != 100:
        raise ValueError(f"Expected 100 rows, got {len(test_df)}")
    
    print(f"  ✓ Sample CSV created and validated: {test_csv_path}")
    print(f"    You can use this file for testing with:")
    print(f"    python main.py --input {test_csv_path}")
    
except Exception as e:
    print(f"  ✗ CSV creation failed: {e}")
    sys.exit(1)

# All tests passed!
print("\n" + "=" * 70)
print("✓ ALL TESTS PASSED!")
print("=" * 70)
print("\nYour TCLaaC installation is ready to use.")
print("\nNext steps:")
print("  1. Run the streamlined pipeline:")
print("     python main.py --synthetic 10000 --output results/")
print()
print("  2. Or use your own data:")
print("     python main.py --input your_data.csv --output results/")
print()
print("  3. Or explore in the Jupyter notebook:")
print("     jupyter notebook \"The Command Line as a Corpus.ipynb\"")
print()
